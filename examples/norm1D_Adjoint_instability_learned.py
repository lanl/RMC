#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Learned 1D Gaussian Adjoint-Sampling instability ablations.

This extends norm1D_Adjoint_instability.py with learned approximations.

Rows:

E  direct regression onto the exact affine velocity,
   Gaussianized current population, no replay, no target clipping.

F  RAM regression, Gaussianized current population,
   no replay, no target clipping.

G  RAM regression, actual learned rollout population,
   no replay, no target clipping.

H  RAM regression, actual learned rollout population,
   replay, no target clipping.

I  RAM regression, actual learned rollout population,
   no replay, terminal-adjoint clipping at 150.

J  RAM regression, actual learned rollout population,
   replay, terminal-adjoint clipping at 150.

All learned runs use the same network architecture and optimizer as the
working repository-style 2D Adjoint experiment, while using the constant
unit diffusion and uniform time grid required by the Brownian diagnostic.

The prescribed initial outer state is

    p_0 = N(0.2, 0.1).

For rows G--J, after this initial state the actual learned rollout supplies
the endpoint population for the next RAM update.

This script first studies eta=1 only. Outer damping is added only after
the undamped learned ablations have been inspected.
"""

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from matplotlib import pyplot as plt

from rmc.flax.nn_config_dict import NNConfigDict
from rmc.flax.trainer import train_step
from rmc.modules.adjoint import AdjointSampler, _ReplayBuffer

from norm1D_Adjoint_instability import (
    INITIAL_MEAN,
    INITIAL_VARIANCE,
    MU,
    TARGET_VARIANCE,
    analytic_multiplier,
    oracle_coefficients,
)


METHODS = {
    "E direct affine": {
        "regression": "direct",
        "gaussianize": True,
        "replay": False,
        "clip": None,
    },
    "F RAM Gaussian": {
        "regression": "ram",
        "gaussianize": True,
        "replay": False,
        "clip": None,
    },
    "G RAM rollout": {
        "regression": "ram",
        "gaussianize": False,
        "replay": False,
        "clip": None,
    },
    "H RAM rollout + replay": {
        "regression": "ram",
        "gaussianize": False,
        "replay": True,
        "clip": None,
    },
    "I RAM rollout + clip": {
        "regression": "ram",
        "gaussianize": False,
        "replay": False,
        "clip": 150.0,
    },
    "J RAM rollout + replay + clip": {
        "regression": "ram",
        "gaussianize": False,
        "replay": True,
        "clip": 150.0,
    },
}


class GaussianTarget:
    """One-dimensional Gaussian target with the RMC density interface."""

    def __init__(
        self,
        mean,
        variance,
    ):
        self.mean = jnp.asarray(mean)
        self.variance = float(variance)

    def log_target(
        self,
        x,
    ):
        return -0.5 * jnp.sum(
            (x - self.mean) ** 2,
            axis=-1,
        ) / self.variance

    def der_log_target_proposal(
        self,
        x,
        tempering=1.0,
    ):
        del tempering

        return -(x - self.mean) / self.variance


def build_config(
    seed,
    batch_size,
    train_samples,
    inner_steps,
):
    """Match the working repository-style network/optimizer configuration."""
    config: NNConfigDict = {
        "seed": int(seed),
        "batch_size": int(batch_size),
        "dim": 1,
        "layer_widths": [64, 64, 64],
        "activation_func": nnx.silu,
        "nn_type": "time_embed",
        "opt_type": "ADAM",
        "base_lr": 1.0e-4,
        "opt_grad_max_norm": 10.0,
        "max_samples": int(train_samples),
        "nsamples": int(train_samples),
        "max_subiter": 1,
        "eval_every": 1,
        "has_aux": False,
        "adjoint_repo_features": True,
        "adjoint_outer_samples": int(train_samples),
        "adjoint_batch_size": int(batch_size),
        "adjoint_inner_steps": int(inner_steps),

        # This diagnostic is specifically the standard-Brownian problem.
        "adjoint_diffusion_schedule": "constant",
        "adjoint_sigma": 1.0,
        "adjoint_time_discretization": "uniform",
    }

    return config


def build_model(
    seed,
    batch_size,
    train_samples,
    inner_steps,
    steps,
):
    target = GaussianTarget(
        MU,
        TARGET_VARIANCE,
    )

    return AdjointSampler(
        config=build_config(
            seed,
            batch_size,
            train_samples,
            inner_steps,
        ),
        densitycl=target,
        h=1.0 / steps,
        T=steps,
        sigma_schedule=lambda t: 1.0 + 0.0 * t,
        integrated_variance=lambda t: t,
    )


def sample_gaussian(
    key,
    mean,
    variance,
    nsamples,
):
    noise = jax.random.normal(
        key,
        (nsamples, 1),
        dtype=jnp.float32,
    )

    return (
        jnp.asarray(mean, dtype=jnp.float32)
        + jnp.sqrt(
            jnp.asarray(
                variance,
                dtype=jnp.float32,
            )
        )
        * noise
    )


def empirical_moments(
    endpoints,
):
    values = np.asarray(
        endpoints
    ).reshape(-1)

    return (
        float(np.mean(values)),
        float(np.var(values, ddof=0)),
    )


def exact_affine_labels(
    train_input,
    mean,
    variance,
):
    x = train_input[:, :1]
    t = train_input[:, 1]

    alpha, beta = oracle_coefficients(
        t,
        mean,
        variance,
    )

    return (
        alpha[:, None] * x
        + beta[:, None]
    )


def train_direct_affine(
    model,
    optimizer,
    metrics,
    endpoints,
    outer_mean,
    outer_variance,
    inner_steps,
    batch_size,
    key,
):
    """Fit the NN directly to the exact Gaussian affine velocity."""
    dummy_gradients = jnp.zeros_like(
        endpoints
    )

    losses = []

    for _ in range(inner_steps):
        key, batch_key = jax.random.split(
            key
        )

        train_ds = model.build_ram_batch(
            batch_key,
            endpoints,
            dummy_gradients,
            batch_size=batch_size,
        )

        labels = exact_affine_labels(
            train_ds["input"],
            outer_mean,
            outer_variance,
        )

        model.nnmodel.train()

        loss = train_step(
            model.nnmodel,
            model.compute_ram_loss,
            optimizer,
            metrics,
            train_ds["input"],
            labels,
            False,
        )

        metrics.reset()
        losses.append(loss)

    return (
        float(jnp.mean(jnp.asarray(losses))),
        key,
    )


def train_ram_current(
    model,
    optimizer,
    metrics,
    endpoints,
    inner_steps,
    batch_size,
    clip_norm,
    key,
):
    """RAM updates using only the current outer endpoint population."""
    terminal_gradients = model.eval_terminal_gradient(
        endpoints
    )

    (
        terminal_gradients,
        fraction_clipped,
    ) = model._clip_terminal_gradients(
        terminal_gradients,
        clip_norm,
    )

    losses = []

    for _ in range(inner_steps):
        key, batch_key = jax.random.split(
            key
        )

        train_ds = model.build_ram_batch(
            batch_key,
            endpoints,
            terminal_gradients,
            batch_size=batch_size,
        )

        model.nnmodel.train()

        loss = train_step(
            model.nnmodel,
            model.compute_ram_loss,
            optimizer,
            metrics,
            train_ds["input"],
            train_ds["label"],
            False,
        )

        metrics.reset()
        losses.append(loss)

    return (
        float(jnp.mean(jnp.asarray(losses))),
        float(fraction_clipped),
        key,
    )


def train_ram_replay(
    model,
    optimizer,
    metrics,
    replay,
    endpoints,
    inner_steps,
    batch_size,
    clip_norm,
    key,
):
    """RAM updates using the existing finite FIFO replay mechanism."""
    terminal_gradients = model.eval_terminal_gradient(
        endpoints
    )

    (
        terminal_gradients,
        fraction_clipped,
    ) = model._clip_terminal_gradients(
        terminal_gradients,
        clip_norm,
    )

    replay.add(
        endpoints,
        terminal_gradients,
    )

    losses = []

    for _ in range(inner_steps):
        (
            key,
            replay_key,
            bridge_key,
        ) = jax.random.split(
            key,
            3,
        )

        (
            replay_endpoints,
            replay_gradients,
        ) = replay.sample(
            replay_key,
            batch_size,
        )

        train_ds = model.build_ram_batch(
            bridge_key,
            replay_endpoints,
            replay_gradients,
            batch_size=None,
        )

        model.nnmodel.train()

        loss = train_step(
            model.nnmodel,
            model.compute_ram_loss,
            optimizer,
            metrics,
            train_ds["input"],
            train_ds["label"],
            False,
        )

        metrics.reset()
        losses.append(loss)

    return (
        float(jnp.mean(jnp.asarray(losses))),
        float(fraction_clipped),
        key,
    )


def run_method(
    method_name,
    method_options,
    seed,
    outer_iterations,
    inner_steps,
    train_samples,
    eval_samples,
    batch_size,
    steps,
):
    """Run one undamped learned ablation."""
    model = build_model(
        seed=seed,
        batch_size=batch_size,
        train_samples=train_samples,
        inner_steps=inner_steps,
        steps=steps,
    )

    optimizer = model._build_optimizer()

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    replay = None

    if method_options["replay"]:
        replay = _ReplayBuffer(
            dim=1,
            capacity=1000,
        )

    key = jax.random.PRNGKey(
        10000 + int(seed)
    )

    # The prescribed p_0 is supplied directly rather than generated
    # by the initially random velocity network.
    key, p0_key = jax.random.split(
        key
    )

    current_training_endpoints = sample_gaussian(
        p0_key,
        INITIAL_MEAN,
        INITIAL_VARIANCE,
        train_samples,
    )

    current_mean = INITIAL_MEAN
    current_variance = INITIAL_VARIANCE

    sequence = [
        {
            "k": 0,
            "mean": INITIAL_MEAN,
            "variance": INITIAL_VARIANCE,
            "loss": np.nan,
            "clip_fraction": 0.0,
            "buffer_size": 0,
        }
    ]

    for outer in range(outer_iterations):
        # E/F Gaussianize the current distribution using its accurately
        # estimated first two moments.  G--J retain actual rollout samples.
        if (
            outer > 0
            and method_options["gaussianize"]
        ):
            key, gaussian_key = jax.random.split(
                key
            )

            training_endpoints = sample_gaussian(
                gaussian_key,
                current_mean,
                current_variance,
                train_samples,
            )
        else:
            training_endpoints = (
                current_training_endpoints
            )

        if method_options["regression"] == "direct":
            loss, key = train_direct_affine(
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                endpoints=training_endpoints,
                outer_mean=current_mean,
                outer_variance=current_variance,
                inner_steps=inner_steps,
                batch_size=batch_size,
                key=key,
            )

            clip_fraction = 0.0
            buffer_size = 0

        elif method_options["replay"]:
            (
                loss,
                clip_fraction,
                key,
            ) = train_ram_replay(
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                replay=replay,
                endpoints=training_endpoints,
                inner_steps=inner_steps,
                batch_size=batch_size,
                clip_norm=method_options["clip"],
                key=key,
            )

            buffer_size = len(replay)

        else:
            (
                loss,
                clip_fraction,
                key,
            ) = train_ram_current(
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                endpoints=training_endpoints,
                inner_steps=inner_steps,
                batch_size=batch_size,
                clip_norm=method_options["clip"],
                key=key,
            )

            buffer_size = 0

        # Use a larger independent endpoint population for diagnostics.
        key, eval_key = jax.random.split(
            key
        )

        eval_endpoints = model.generate_endpoints(
            eval_samples,
            eval_key,
        )

        (
            current_mean,
            current_variance,
        ) = empirical_moments(
            eval_endpoints
        )

        # G--J feed actual learned rollout endpoints into the next
        # outer iteration.  E/F instead Gaussianize the diagnostic moments.
        if not method_options["gaussianize"]:
            key, rollout_key = jax.random.split(
                key
            )

            current_training_endpoints = (
                model.generate_endpoints(
                    train_samples,
                    rollout_key,
                )
            )

        sequence.append(
            {
                "k": outer + 1,
                "mean": current_mean,
                "variance": current_variance,
                "loss": loss,
                "clip_fraction": clip_fraction,
                "buffer_size": buffer_size,
            }
        )

        error = current_mean - MU

        print(
            f"{method_name:31s} "
            f"seed={seed:2d} "
            f"k={outer + 1:2d} "
            f"mean={current_mean: .7f} "
            f"error={error: .7f} "
            f"var={current_variance: .7f} "
            f"loss={loss: .6e} "
            f"clip={clip_fraction:.3f} "
            f"buffer={buffer_size}"
        )

    return sequence


def add_ratios(
    sequence,
):
    errors = np.asarray(
        [
            entry["mean"] - MU
            for entry in sequence
        ],
        dtype=float,
    )

    ratios = np.full(
        len(sequence),
        np.nan,
    )

    ratios[:-1] = (
        errors[1:]
        / errors[:-1]
    )

    for entry, ratio in zip(
        sequence,
        ratios,
    ):
        entry["ratio"] = float(ratio)


def write_csv(
    all_results,
    output_path,
):
    filename = (
        output_path
        / "learned_ablation_eta1.csv"
    )

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "method",
            "seed",
            "k",
            "mean",
            "error",
            "ratio",
            "variance",
            "loss",
            "clip_fraction",
            "buffer_size",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for method, seed_results in all_results.items():
            for seed, sequence in seed_results.items():
                for entry in sequence:
                    writer.writerow(
                        {
                            "method": method,
                            "seed": seed,
                            "k": entry["k"],
                            "mean": entry["mean"],
                            "error": entry["mean"] - MU,
                            "ratio": entry["ratio"],
                            "variance": entry["variance"],
                            "loss": entry["loss"],
                            "clip_fraction": entry[
                                "clip_fraction"
                            ],
                            "buffer_size": entry[
                                "buffer_size"
                            ],
                        }
                    )

    return filename


def aggregate_method(
    seed_results,
    field,
):
    seeds = sorted(
        seed_results
    )

    values = np.asarray(
        [
            [
                entry[field]
                for entry in seed_results[seed]
            ]
            for seed in seeds
        ],
        dtype=float,
    )

    return (
        np.mean(values, axis=0),
        np.std(values, axis=0),
    )


def load_oracle_curves(
    oracle_csv,
):
    """Load selected eta=1 oracle curves from the first-stage experiment."""
    wanted = {
        "analytic",
        "EM moments",
        "oracle particles, feedback",
    }

    curves = {
        name: []
        for name in wanted
    }

    if oracle_csv is None:
        return {}

    path = Path(
        oracle_csv
    ).expanduser()

    if not path.is_file():
        print(
            f"Oracle CSV not found; combined plot will omit it: {path}"
        )
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row in reader:
            if float(row["eta"]) != 1.0:
                continue

            method = row["method"]

            if method not in wanted:
                continue

            curves[method].append(
                (
                    int(row["k"]),
                    float(row["error"]),
                )
            )

    result = {}

    for method, values in curves.items():
        values = sorted(
            values
        )

        if values:
            result[method] = (
                np.asarray(
                    [item[0] for item in values]
                ),
                np.asarray(
                    [item[1] for item in values]
                ),
            )

    return result


def plot_mean_error(
    all_results,
    oracle_curves,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(10, 6),
    )

    for name, (
        iterations,
        errors,
    ) in oracle_curves.items():
        ax.plot(
            iterations,
            errors,
            marker="o",
            linestyle="--",
            linewidth=1.5,
            label=name,
        )

    for method, seed_results in all_results.items():
        mean, std = aggregate_method(
            seed_results,
            "mean",
        )

        error = mean - MU

        all_seed_errors = np.asarray(
            [
                [
                    entry["mean"] - MU
                    for entry in sequence
                ]
                for sequence in seed_results.values()
            ]
        )

        error_std = np.std(
            all_seed_errors,
            axis=0,
        )

        iterations = np.arange(
            len(error)
        )

        line = ax.plot(
            iterations,
            error,
            marker="o",
            label=method,
        )[0]

        ax.fill_between(
            iterations,
            error - error_std,
            error + error_std,
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
        "Undamped 1D Gaussian Adjoint instability"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()

    filename = (
        output_path
        / "mean_error_ablation_eta1.png"
    )

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return filename


def plot_variance(
    all_results,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(9, 5.5),
    )

    for method, seed_results in all_results.items():
        mean, std = aggregate_method(
            seed_results,
            "variance",
        )

        iterations = np.arange(
            len(mean)
        )

        line = ax.plot(
            iterations,
            mean,
            marker="o",
            label=method,
        )[0]

        ax.fill_between(
            iterations,
            mean - std,
            mean + std,
            alpha=0.12,
            color=line.get_color(),
        )

    ax.axhline(
        TARGET_VARIANCE,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Outer iteration"
    )

    ax.set_ylabel(
        r"Terminal variance $s_k^2$"
    )

    ax.set_title(
        "Undamped learned Adjoint variance"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()

    filename = (
        output_path
        / "variance_ablation_eta1.png"
    )

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return filename


def print_summary(
    all_results,
):
    print("")
    print("============================================================")
    print("LEARNED ABLATION SUMMARY")
    print("============================================================")
    print(
        f"continuous analytic multiplier: "
        f"{analytic_multiplier():.10f}"
    )

    for method, seed_results in all_results.items():
        print("")
        print(method)
        print(
            " k     mean error mean      error std"
            "      variance mean      ratio median"
        )

        nsteps = len(
            next(
                iter(
                    seed_results.values()
                )
            )
        )

        for k in range(nsteps):
            errors = np.asarray(
                [
                    sequence[k]["mean"] - MU
                    for sequence in seed_results.values()
                ]
            )

            variances = np.asarray(
                [
                    sequence[k]["variance"]
                    for sequence in seed_results.values()
                ]
            )

            ratios = np.asarray(
                [
                    sequence[k]["ratio"]
                    for sequence in seed_results.values()
                ]
            )

            finite_ratios = ratios[
                np.isfinite(ratios)
            ]

            ratio_text = (
                f"{np.median(finite_ratios): .7f}"
                if len(finite_ratios) > 0
                else "       --"
            )

            print(
                f"{k:2d}  "
                f"{np.mean(errors): .9f}  "
                f"{np.std(errors): .9f}  "
                f"{np.mean(variances): .9f}  "
                f"{ratio_text}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outer",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--inner",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--eval-samples",
        type=int,
        default=2000,
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
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )

    parser.add_argument(
        "--oracle-csv",
        default=(
            "~/ROU-project/adjoint-runs/"
            "instability/oracle/"
            "oracle_ablation.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "~/ROU-project/adjoint-runs/"
            "instability/learned"
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
    print("Learned 1D Gaussian Adjoint instability ablations")
    print("============================================================")
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    print("")
    print(f"target mean:       {MU}")
    print(f"target variance:   {TARGET_VARIANCE}")
    print(f"initial mean:      {INITIAL_MEAN}")
    print(f"initial variance:  {INITIAL_VARIANCE}")
    print(f"outer iterations:  {args.outer}")
    print(f"inner updates:     {args.inner}")
    print(f"training samples:  {args.train_samples}")
    print(f"diagnostic samples:{args.eval_samples}")
    print(f"batch size:        {args.batch_size}")
    print(f"EM steps:          {args.steps}")
    print(f"seeds:             {args.seeds}")
    print(f"analytic A:        {analytic_multiplier():.10f}")
    print("")

    all_results = {}

    for method_name, method_options in METHODS.items():
        all_results[method_name] = {}

        print("")
        print("============================================================")
        print(method_name)
        print("============================================================")

        for seed in args.seeds:
            sequence = run_method(
                method_name=method_name,
                method_options=method_options,
                seed=seed,
                outer_iterations=args.outer,
                inner_steps=args.inner,
                train_samples=args.train_samples,
                eval_samples=args.eval_samples,
                batch_size=args.batch_size,
                steps=args.steps,
            )

            add_ratios(
                sequence
            )

            all_results[
                method_name
            ][seed] = sequence

    print_summary(
        all_results
    )

    csv_path = write_csv(
        all_results,
        output_path,
    )

    oracle_curves = load_oracle_curves(
        args.oracle_csv
    )

    mean_plot = plot_mean_error(
        all_results,
        oracle_curves,
        output_path,
    )

    variance_plot = plot_variance(
        all_results,
        output_path,
    )

    print("")
    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {mean_plot}")
    print(f"  {variance_plot}")


if __name__ == "__main__":
    main()
