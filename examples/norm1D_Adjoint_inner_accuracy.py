#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Accuracy-controlled direct-affine regression diagnostic.

E* differs from the existing "E direct affine" ablation only in the
stopping rule for the inner regression.  Instead of exactly 100 updates,
the same optimizer/network are trained until the relative velocity MSE

    E |u_theta - u_oracle|^2 / E |u_oracle|^2

falls below a prescribed tolerance, or max_inner updates are reached.

The velocity error is evaluated on a fixed, independent RAM-distributed
validation batch at each outer iteration.
"""

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from matplotlib import pyplot as plt

from norm1D_Adjoint_instability import (
    INITIAL_MEAN,
    INITIAL_VARIANCE,
    MU,
    TARGET_VARIANCE,
    analytic_multiplier,
    analytic_sequence,
)
from norm1D_Adjoint_instability_learned import (
    build_model,
    empirical_moments,
    exact_affine_labels,
    sample_gaussian,
    train_direct_affine,
)


def build_velocity_validation_batch(
    model,
    mean,
    variance,
    nsamples,
    key,
):
    """Build one fixed independent validation batch for the affine field."""
    endpoint_key, bridge_key = jax.random.split(key)

    endpoints = sample_gaussian(
        endpoint_key,
        mean,
        variance,
        nsamples,
    )

    dummy_gradients = jnp.zeros_like(endpoints)

    batch = model.build_ram_batch(
        bridge_key,
        endpoints,
        dummy_gradients,
        batch_size=None,
    )

    labels = exact_affine_labels(
        batch["input"],
        mean,
        variance,
    )

    return batch["input"], labels


def velocity_error(
    model,
    inputs,
    labels,
):
    """Return absolute MSE and target-normalized relative MSE."""
    x = inputs[:, :1]
    t = inputs[:, 1:]

    model.nnmodel.eval()

    prediction = model.nnmodel(
        x,
        t,
    )

    mse = jnp.mean(
        (prediction - labels) ** 2
    )

    target_energy = jnp.mean(
        labels**2
    )

    relative_mse = mse / jnp.maximum(
        target_energy,
        1.0e-12,
    )

    return (
        float(mse),
        float(relative_mse),
    )


def run_accuracy_controlled(
    seed,
    outer_iterations,
    train_samples,
    eval_samples,
    velocity_eval_samples,
    batch_size,
    steps,
    tolerance,
    max_inner,
    check_every,
):
    model = build_model(
        seed=seed,
        batch_size=batch_size,
        train_samples=train_samples,
        inner_steps=check_every,
        steps=steps,
    )

    optimizer = model._build_optimizer()

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    key = jax.random.PRNGKey(
        30000 + int(seed)
    )

    current_mean = INITIAL_MEAN
    current_variance = INITIAL_VARIANCE

    sequence = [
        {
            "k": 0,
            "mean": INITIAL_MEAN,
            "variance": INITIAL_VARIANCE,
            "error": INITIAL_MEAN - MU,
            "ratio": np.nan,
            "inner_steps": 0,
            "velocity_mse": np.nan,
            "velocity_rel_mse": np.nan,
            "converged": False,
        }
    ]

    previous_error = INITIAL_MEAN - MU

    for outer in range(outer_iterations):
        # Gaussianized current outer state, exactly as in row E.
        (
            key,
            train_endpoint_key,
            validation_key,
        ) = jax.random.split(
            key,
            3,
        )

        training_endpoints = sample_gaussian(
            train_endpoint_key,
            current_mean,
            current_variance,
            train_samples,
        )

        validation_input, validation_labels = (
            build_velocity_validation_batch(
                model=model,
                mean=current_mean,
                variance=current_variance,
                nsamples=velocity_eval_samples,
                key=validation_key,
            )
        )

        velocity_mse, velocity_rel_mse = velocity_error(
            model,
            validation_input,
            validation_labels,
        )

        print("")
        print(
            f"E* seed={seed} outer={outer} "
            f"initial velocity relMSE={velocity_rel_mse:.6e}"
        )

        total_inner = 0
        last_loss = np.nan

        while (
            velocity_rel_mse > tolerance
            and total_inner < max_inner
        ):
            chunk = min(
                check_every,
                max_inner - total_inner,
            )

            last_loss, key = train_direct_affine(
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                endpoints=training_endpoints,
                outer_mean=current_mean,
                outer_variance=current_variance,
                inner_steps=chunk,
                batch_size=batch_size,
                key=key,
            )

            total_inner += chunk

            (
                velocity_mse,
                velocity_rel_mse,
            ) = velocity_error(
                model,
                validation_input,
                validation_labels,
            )

            print(
                f"  inner={total_inner:5d} "
                f"loss={last_loss:.6e} "
                f"velocity relMSE={velocity_rel_mse:.6e}"
            )

        converged = (
            velocity_rel_mse <= tolerance
        )

        key, rollout_key = jax.random.split(key)

        endpoints = model.generate_endpoints(
            eval_samples,
            rollout_key,
        )

        (
            next_mean,
            next_variance,
        ) = empirical_moments(
            endpoints
        )

        next_error = next_mean - MU
        ratio = next_error / previous_error

        sequence.append(
            {
                "k": outer + 1,
                "mean": next_mean,
                "variance": next_variance,
                "error": next_error,
                "ratio": ratio,
                "inner_steps": total_inner,
                "velocity_mse": velocity_mse,
                "velocity_rel_mse": velocity_rel_mse,
                "converged": converged,
            }
        )

        print(
            f"E* seed={seed} k={outer + 1} "
            f"mean={next_mean:.9f} "
            f"error={next_error:.9f} "
            f"ratio={ratio:.9f} "
            f"variance={next_variance:.9f} "
            f"inner={total_inner} "
            f"relMSE={velocity_rel_mse:.6e} "
            f"converged={converged}"
        )

        current_mean = next_mean
        current_variance = next_variance
        previous_error = next_error

    return sequence


def write_csv(
    results,
    output_path,
):
    filename = output_path / "direct_accuracy_control.csv"

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "k",
                "mean",
                "error",
                "ratio",
                "variance",
                "inner_steps",
                "velocity_mse",
                "velocity_rel_mse",
                "converged",
            ],
        )

        writer.writeheader()

        for seed, sequence in results.items():
            for row in sequence:
                writer.writerow(
                    {
                        "seed": seed,
                        **row,
                    }
                )

    return filename


def load_standard_e(
    filename,
):
    """Load the existing 100-inner-step E row."""
    filename = Path(
        filename
    ).expanduser()

    if not filename.is_file():
        return None

    values = {}

    with open(
        filename,
        "r",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            if row["method"] != "E direct affine":
                continue

            k = int(row["k"])

            values.setdefault(
                k,
                [],
            ).append(
                float(row["error"])
            )

    if not values:
        return None

    ks = np.asarray(
        sorted(values),
        dtype=int,
    )

    mean = np.asarray(
        [
            np.mean(values[k])
            for k in ks
        ]
    )

    std = np.asarray(
        [
            np.std(values[k])
            for k in ks
        ]
    )

    return ks, mean, std


def plot_results(
    results,
    standard_e,
    outer_iterations,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(9, 5.5),
    )

    analytic = analytic_sequence(
        outer_iterations,
        eta=1.0,
    )

    analytic_error = np.asarray(
        [
            row["mean"] - MU
            for row in analytic
        ]
    )

    ax.plot(
        np.arange(len(analytic_error)),
        analytic_error,
        marker="o",
        linestyle="--",
        label="analytic",
    )

    if standard_e is not None:
        ks, mean, std = standard_e

        line = ax.plot(
            ks,
            mean,
            marker="o",
            label="E: direct affine, 100 inner",
        )[0]

        ax.fill_between(
            ks,
            mean - std,
            mean + std,
            alpha=0.12,
            color=line.get_color(),
        )

    sequences = list(
        results.values()
    )

    n = min(
        len(sequence)
        for sequence in sequences
    )

    errors = np.asarray(
        [
            [
                row["error"]
                for row in sequence[:n]
            ]
            for sequence in sequences
        ]
    )

    mean = np.mean(
        errors,
        axis=0,
    )

    std = np.std(
        errors,
        axis=0,
    )

    ks = np.arange(n)

    line = ax.plot(
        ks,
        mean,
        marker="o",
        label="E*: accuracy-controlled direct affine",
    )[0]

    ax.fill_between(
        ks,
        mean - std,
        mean + std,
        alpha=0.12,
        color=line.get_color(),
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Outer iteration"
    )

    ax.set_ylabel(
        r"Mean error $m_k-\mu$"
    )

    ax.set_title(
        "Direct-affine inner-solve accuracy control"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    filename = (
        output_path
        / "direct_accuracy_mean_error.png"
    )

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return filename


def plot_velocity_error(
    results,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(8, 5),
    )

    for seed, sequence in results.items():
        rows = sequence[1:]

        ax.semilogy(
            [
                row["k"]
                for row in rows
            ],
            [
                row["velocity_rel_mse"]
                for row in rows
            ],
            marker="o",
            label=f"seed {seed}",
        )

    ax.set_xlabel(
        "Outer iteration"
    )

    ax.set_ylabel(
        "Velocity relative MSE"
    )

    ax.set_title(
        "Accuracy of direct affine velocity fit"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    filename = (
        output_path
        / "direct_accuracy_velocity_error.png"
    )

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return filename


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outer",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--eval-samples",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--velocity-eval-samples",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-3,
    )

    parser.add_argument(
        "--max-inner",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--check-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
    )

    parser.add_argument(
        "--learned-csv",
        default=(
            "~/ROU-project/adjoint-runs/"
            "instability/learned/"
            "learned_ablation_eta1.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "~/ROU-project/adjoint-runs/"
            "instability/inner-accuracy"
        ),
    )

    args = parser.parse_args()

    output_path = Path(
        args.output
    ).expanduser()

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("============================================================")
    print("Accuracy-controlled direct-affine diagnostic")
    print("============================================================")
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())
    print(f"analytic A:       {analytic_multiplier():.10f}")
    print(f"outer:            {args.outer}")
    print(f"tolerance:        {args.tol:.3e}")
    print(f"max inner:        {args.max_inner}")
    print(f"check every:      {args.check_every}")
    print(f"train samples:    {args.train_samples}")
    print(f"velocity val:     {args.velocity_eval_samples}")
    print(f"rollout samples:  {args.eval_samples}")
    print(f"seeds:            {args.seeds}")

    results = {}

    for seed in args.seeds:
        results[seed] = run_accuracy_controlled(
            seed=seed,
            outer_iterations=args.outer,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            velocity_eval_samples=args.velocity_eval_samples,
            batch_size=args.batch_size,
            steps=args.steps,
            tolerance=args.tol,
            max_inner=args.max_inner,
            check_every=args.check_every,
        )

    csv_path = write_csv(
        results,
        output_path,
    )

    standard_e = load_standard_e(
        args.learned_csv
    )

    mean_plot = plot_results(
        results,
        standard_e,
        args.outer,
        output_path,
    )

    velocity_plot = plot_velocity_error(
        results,
        output_path,
    )

    print("")
    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {mean_plot}")
    print(f"  {velocity_plot}")


if __name__ == "__main__":
    main()
