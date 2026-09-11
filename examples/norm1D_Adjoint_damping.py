#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Outer parameter-relaxation diagnostic for the 1D Gaussian experiment.

Methods:
    E  direct regression onto the exact affine velocity
    F  RAM regression on Gaussianized current population
    G  RAM regression on actual learned rollout population
    H  RAM regression on actual learned rollout population + replay

The first learned update is left undamped because p_0=N(0.2,0.1) is
externally prescribed and is not represented by the randomly initialized
network.  From the second outer update onward,

    theta_{k+1}
      = (1-eta) theta_k + eta theta_hat_{k+1},

with eta=0.5.

The optimizer, architecture, RAM loss, rollout discretization, and training
budget are otherwise unchanged.
"""

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from matplotlib import pyplot as plt

from rmc.modules.adjoint import _ReplayBuffer

from norm1D_Adjoint_instability import (
    INITIAL_MEAN,
    INITIAL_VARIANCE,
    MU,
    TARGET_VARIANCE,
    em_proposal_moments,
)

from norm1D_Adjoint_instability_learned import (
    METHODS as ALL_METHODS,
    build_model,
    empirical_moments,
    sample_gaussian,
    train_direct_affine,
    train_ram_current,
    train_ram_replay,
)


METHOD_NAMES = [
    "E direct affine",
    "F RAM Gaussian",
    "G RAM rollout",
    "H RAM rollout + replay",
]

METHODS = {
    name: ALL_METHODS[name]
    for name in METHOD_NAMES
}


def snapshot_params(model):
    """Take a functional snapshot of the network parameters."""
    state = nnx.state(
        model.nnmodel,
        nnx.Param,
    )

    return jax.tree.map(
        lambda x: jnp.array(x),
        state,
    )


def relax_params(
    model,
    previous_params,
    eta,
):
    """Relax the completed inner proposal back toward the previous iterate."""
    if eta >= 1.0:
        return

    proposal_params = nnx.state(
        model.nnmodel,
        nnx.Param,
    )

    relaxed_params = jax.tree.map(
        lambda old, new: (
            (1.0 - eta) * old
            + eta * new
        ),
        previous_params,
        proposal_params,
    )

    nnx.update(
        model.nnmodel,
        relaxed_params,
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
    eta,
):
    """Run one learned damping experiment."""
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

    key, p0_key = jax.random.split(key)

    current_training_endpoints = sample_gaussian(
        p0_key,
        INITIAL_MEAN,
        INITIAL_VARIANCE,
        train_samples,
    )

    current_mean = INITIAL_MEAN
    current_variance = INITIAL_VARIANCE
    previous_error = INITIAL_MEAN - MU

    sequence = [
        {
            "k": 0,
            "mean": INITIAL_MEAN,
            "variance": INITIAL_VARIANCE,
            "error": previous_error,
            "ratio": np.nan,
            "loss": np.nan,
            "buffer_size": 0,
            "applied_eta": np.nan,
        }
    ]

    for outer in range(outer_iterations):
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

        # Once an actual learned sampler exists, theta_k is the state
        # against which the next completed inner solve is relaxed.
        previous_params = snapshot_params(
            model
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

            buffer_size = 0

        elif method_options["replay"]:
            (
                loss,
                _,
                key,
            ) = train_ram_replay(
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                replay=replay,
                endpoints=training_endpoints,
                inner_steps=inner_steps,
                batch_size=batch_size,
                clip_norm=None,
                key=key,
            )

            buffer_size = len(replay)

        else:
            (
                loss,
                _,
                key,
            ) = train_ram_current(
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                endpoints=training_endpoints,
                inner_steps=inner_steps,
                batch_size=batch_size,
                clip_norm=None,
                key=key,
            )

            buffer_size = 0

        # p_0 was imposed externally, so theta_0 is unrelated to p_0.
        # Establish theta_1 without relaxation, then damp subsequent
        # learned outer updates.
        applied_eta = (
            1.0
            if outer == 0
            else eta
        )

        relax_params(
            model,
            previous_params,
            applied_eta,
        )

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

        error = current_mean - MU
        ratio = error / previous_error

        sequence.append(
            {
                "k": outer + 1,
                "mean": current_mean,
                "variance": current_variance,
                "error": error,
                "ratio": ratio,
                "loss": loss,
                "buffer_size": buffer_size,
                "applied_eta": applied_eta,
            }
        )

        print(
            f"{method_name:31s} "
            f"seed={seed:2d} "
            f"k={outer + 1:2d} "
            f"eta={applied_eta:.2f} "
            f"mean={current_mean: .7f} "
            f"error={error: .7f} "
            f"ratio={ratio: .7f} "
            f"var={current_variance: .7f} "
            f"loss={loss: .6e} "
            f"buffer={buffer_size}"
        )

        previous_error = error

    return sequence


def discrete_oracle_sequence(
    outer_iterations,
    steps,
    eta,
    damp_after_first,
):
    """Exact EM-moment counterpart using the same damping schedule."""
    times = np.linspace(
        0.0,
        1.0,
        steps + 1,
    )

    mean = INITIAL_MEAN
    variance = INITIAL_VARIANCE

    result = [
        {
            "k": 0,
            "mean": mean,
            "variance": variance,
        }
    ]

    for outer in range(outer_iterations):
        proposal_mean, proposal_variance = (
            em_proposal_moments(
                mean,
                variance,
                times,
            )
        )

        if damp_after_first:
            applied_eta = (
                1.0
                if outer == 0
                else eta
            )
        else:
            applied_eta = 1.0

        mean = (
            (1.0 - applied_eta) * mean
            + applied_eta * proposal_mean
        )

        variance = (
            (1.0 - applied_eta) * variance
            + applied_eta * proposal_variance
        )

        result.append(
            {
                "k": outer + 1,
                "mean": float(mean),
                "variance": float(variance),
            }
        )

    return result


def write_csv(
    results,
    output_path,
):
    filename = (
        output_path
        / "learned_ablation_eta05.csv"
    )

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "seed",
                "k",
                "mean",
                "error",
                "ratio",
                "variance",
                "loss",
                "buffer_size",
                "applied_eta",
            ],
        )

        writer.writeheader()

        for method, seed_results in results.items():
            for seed, sequence in seed_results.items():
                for row in sequence:
                    writer.writerow(
                        {
                            "method": method,
                            "seed": seed,
                            **row,
                        }
                    )

    return filename


def load_undamped(
    filename,
):
    filename = Path(
        filename
    ).expanduser()

    results = {
        name: {}
        for name in METHOD_NAMES
    }

    with open(
        filename,
        "r",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row in reader:
            method = row["method"]

            if method not in results:
                continue

            seed = int(
                row["seed"]
            )

            results[method].setdefault(
                seed,
                []
            )

            results[method][seed].append(
                {
                    "k": int(row["k"]),
                    "mean": float(row["mean"]),
                    "variance": float(
                        row["variance"]
                    ),
                }
            )

    return results


def aggregate(
    seed_results,
    field,
):
    values = np.asarray(
        [
            [
                row[field]
                for row in sequence
            ]
            for sequence in seed_results.values()
        ],
        dtype=float,
    )

    return (
        np.mean(values, axis=0),
        np.std(values, axis=0),
    )


def plot_mean_error(
    undamped,
    damped,
    oracle_undamped,
    oracle_damped,
    output_path,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        sharex=True,
    )

    axes = axes.reshape(-1)

    oracle_undamped_error = np.asarray(
        [
            row["mean"] - MU
            for row in oracle_undamped
        ]
    )

    oracle_damped_error = np.asarray(
        [
            row["mean"] - MU
            for row in oracle_damped
        ]
    )

    iterations = np.arange(
        len(oracle_undamped)
    )

    for ax, method in zip(
        axes,
        METHOD_NAMES,
    ):
        mean_u, std_u = aggregate(
            undamped[method],
            "mean",
        )

        mean_d, std_d = aggregate(
            damped[method],
            "mean",
        )

        error_u = mean_u - MU
        error_d = mean_d - MU

        line_u = ax.plot(
            np.arange(len(error_u)),
            error_u,
            marker="o",
            label="learned eta=1",
        )[0]

        ax.fill_between(
            np.arange(len(error_u)),
            error_u - std_u,
            error_u + std_u,
            alpha=0.12,
            color=line_u.get_color(),
        )

        line_d = ax.plot(
            np.arange(len(error_d)),
            error_d,
            marker="o",
            label="learned eta=0.5",
        )[0]

        ax.fill_between(
            np.arange(len(error_d)),
            error_d - std_d,
            error_d + std_d,
            alpha=0.12,
            color=line_d.get_color(),
        )

        ax.plot(
            iterations,
            oracle_undamped_error,
            linestyle="--",
            linewidth=1.2,
            label="EM oracle eta=1",
        )

        ax.plot(
            iterations,
            oracle_damped_error,
            linestyle="--",
            linewidth=1.2,
            label="EM oracle damped",
        )

        ax.axhline(
            0.0,
            linewidth=0.8,
        )

        ax.set_title(
            method
        )

        ax.grid(
            True,
            alpha=0.25,
        )

    axes[2].set_xlabel(
        "Outer iteration"
    )
    axes[3].set_xlabel(
        "Outer iteration"
    )

    axes[0].set_ylabel(
        r"Mean error $m_k-\mu$"
    )
    axes[2].set_ylabel(
        r"Mean error $m_k-\mu$"
    )

    axes[0].legend(
        fontsize=8,
    )

    fig.suptitle(
        "1D Gaussian Adjoint outer damping",
        y=0.995,
    )

    fig.tight_layout()

    filename = (
        output_path
        / "damping_mean_error.png"
    )

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return filename


def plot_variance(
    undamped,
    damped,
    oracle_undamped,
    oracle_damped,
    output_path,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        sharex=True,
    )

    axes = axes.reshape(-1)

    oracle_u = np.asarray(
        [
            row["variance"]
            for row in oracle_undamped
        ]
    )

    oracle_d = np.asarray(
        [
            row["variance"]
            for row in oracle_damped
        ]
    )

    iterations = np.arange(
        len(oracle_u)
    )

    for ax, method in zip(
        axes,
        METHOD_NAMES,
    ):
        mean_u, std_u = aggregate(
            undamped[method],
            "variance",
        )

        mean_d, std_d = aggregate(
            damped[method],
            "variance",
        )

        line_u = ax.plot(
            np.arange(len(mean_u)),
            mean_u,
            marker="o",
            label="learned eta=1",
        )[0]

        ax.fill_between(
            np.arange(len(mean_u)),
            mean_u - std_u,
            mean_u + std_u,
            alpha=0.12,
            color=line_u.get_color(),
        )

        line_d = ax.plot(
            np.arange(len(mean_d)),
            mean_d,
            marker="o",
            label="learned eta=0.5",
        )[0]

        ax.fill_between(
            np.arange(len(mean_d)),
            mean_d - std_d,
            mean_d + std_d,
            alpha=0.12,
            color=line_d.get_color(),
        )

        ax.plot(
            iterations,
            oracle_u,
            linestyle="--",
            linewidth=1.2,
            label="EM oracle eta=1",
        )

        ax.plot(
            iterations,
            oracle_d,
            linestyle="--",
            linewidth=1.2,
            label="EM oracle damped",
        )

        ax.axhline(
            TARGET_VARIANCE,
            linewidth=0.8,
        )

        ax.set_title(
            method
        )

        ax.grid(
            True,
            alpha=0.25,
        )

    axes[2].set_xlabel(
        "Outer iteration"
    )
    axes[3].set_xlabel(
        "Outer iteration"
    )

    axes[0].set_ylabel(
        r"Terminal variance $s_k^2$"
    )
    axes[2].set_ylabel(
        r"Terminal variance $s_k^2$"
    )

    axes[0].legend(
        fontsize=8,
    )

    fig.suptitle(
        "1D Gaussian Adjoint variance under outer damping",
        y=0.995,
    )

    fig.tight_layout()

    filename = (
        output_path
        / "damping_variance.png"
    )

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return filename


def print_summary(
    undamped,
    damped,
):
    print("")
    print("============================================================")
    print("DAMPING SUMMARY")
    print("============================================================")

    for method in METHOD_NAMES:
        print("")
        print(method)
        print(
            " k      undamped error"
            "       damped error"
            "       damped variance"
        )

        mean_u, _ = aggregate(
            undamped[method],
            "mean",
        )

        mean_d, _ = aggregate(
            damped[method],
            "mean",
        )

        var_d, _ = aggregate(
            damped[method],
            "variance",
        )

        for k in range(
            len(mean_d)
        ):
            print(
                f"{k:2d}  "
                f"{mean_u[k] - MU: .9f}  "
                f"{mean_d[k] - MU: .9f}  "
                f"{var_d[k]: .9f}"
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
        "--eta",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )

    parser.add_argument(
        "--undamped-csv",
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
            "instability/damping"
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
    print("1D Gaussian learned outer damping")
    print("============================================================")
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())
    print(f"eta after first update: {args.eta}")
    print(f"outer:                  {args.outer}")
    print(f"inner:                  {args.inner}")
    print(f"training samples:       {args.train_samples}")
    print(f"evaluation samples:     {args.eval_samples}")
    print(f"seeds:                  {args.seeds}")

    damped = {}

    for method_name, options in METHODS.items():
        print("")
        print("============================================================")
        print(method_name)
        print("============================================================")

        damped[method_name] = {}

        for seed in args.seeds:
            damped[method_name][seed] = run_method(
                method_name=method_name,
                method_options=options,
                seed=seed,
                outer_iterations=args.outer,
                inner_steps=args.inner,
                train_samples=args.train_samples,
                eval_samples=args.eval_samples,
                batch_size=args.batch_size,
                steps=args.steps,
                eta=args.eta,
            )

    filename = write_csv(
        damped,
        output_path,
    )

    undamped = load_undamped(
        args.undamped_csv
    )

    oracle_undamped = discrete_oracle_sequence(
        outer_iterations=args.outer,
        steps=args.steps,
        eta=1.0,
        damp_after_first=False,
    )

    oracle_damped = discrete_oracle_sequence(
        outer_iterations=args.outer,
        steps=args.steps,
        eta=args.eta,
        damp_after_first=True,
    )

    print_summary(
        undamped,
        damped,
    )

    mean_plot = plot_mean_error(
        undamped,
        damped,
        oracle_undamped,
        oracle_damped,
        output_path,
    )

    variance_plot = plot_variance(
        undamped,
        damped,
        oracle_undamped,
        oracle_damped,
        output_path,
    )

    print("")
    print("Saved:")
    print(f"  {filename}")
    print(f"  {mean_plot}")
    print(f"  {variance_plot}")


if __name__ == "__main__":
    main()
