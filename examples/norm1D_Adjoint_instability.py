#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
1D Gaussian Adjoint-Sampling instability diagnostic.

Target:
    N(mu, target_variance), mu=5, target_variance=0.1.

Initial outer state:
    p_0 = N(0.2, 0.1).

This first-stage diagnostic compares:

    A. exact analytic matched-variance recurrence;
    B. exact moment propagation through the RMC Euler--Maruyama grid;
    C. affine-oracle particles, with particle error observed but not
       fed back into the next outer iteration;
    D. affine-oracle particles, with empirical moments fed back.

The same calculations are performed for eta=1 and eta=0.5.

No neural-network regression is used in this stage.
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
from rmc.modules.adjoint import AdjointSampler
from rmc.utils.density import BaseLogDensity


MU = 5.0
TARGET_VARIANCE = 0.1
INITIAL_MEAN = 0.2
INITIAL_VARIANCE = 0.1

ETAS = (1.0, 0.5)


class GaussianTarget(BaseLogDensity):
    """One-dimensional Gaussian target."""

    def __init__(self, mean, variance):
        self.mean = jnp.asarray(mean)
        self.variance = float(variance)

    def log_target(self, x):
        return -0.5 * jnp.sum(
            (x - self.mean) ** 2
        ) / self.variance


class AffineOracleSampler(AdjointSampler):
    """Adjoint sampler whose drift is the exact Gaussian affine oracle."""

    def __init__(
        self,
        config,
        densitycl,
        h,
        T,
        outer_mean,
        outer_variance,
    ):
        super().__init__(
            config=config,
            densitycl=densitycl,
            h=h,
            T=T,
            sigma_schedule=lambda t: 1.0 + 0.0 * t,
            integrated_variance=lambda t: t,
        )

        self.outer_mean = float(outer_mean)
        self.outer_variance = float(outer_variance)

    def set_outer_state(
        self,
        mean,
        variance,
    ):
        self.outer_mean = float(mean)
        self.outer_variance = float(variance)

    def eval_drift(
        self,
        x,
        t,
    ):
        """Exact affine adjoint-matching velocity for current Gaussian state."""
        s2 = self.outer_variance
        m = self.outer_mean

        denominator = TARGET_VARIANCE * (
            1.0 - t * (1.0 - s2)
        )

        alpha = (
            -s2
            * (1.0 - TARGET_VARIANCE)
            / denominator
        )

        beta = (
            -(1.0 - TARGET_VARIANCE)
            * m
            * (1.0 - t)
            / denominator
            + MU / TARGET_VARIANCE
        )

        return alpha * x + beta


def build_config():
    """Minimal RMC configuration for the oracle rollout."""
    config: NNConfigDict = {
        "seed": 0,
        "batch_size": 64,
        "dim": 1,
        "layer_widths": [8, 8],
        "activation_func": nnx.silu,
        "nn_type": "time",
        "opt_type": "ADAM",
        "base_lr": 1.0e-3,
        "opt_grad_max_norm": 1.0e20,
        "has_aux": False,
        "adjoint_repo_features": False,
        "adjoint_time_discretization": "uniform",
    }

    return config


def analytic_multiplier():
    """Exact matched-variance outer multiplier."""
    return 1.0 + np.log(TARGET_VARIANCE) / (
        1.0 - TARGET_VARIANCE
    )


def oracle_coefficients(
    t,
    outer_mean,
    outer_variance,
):
    """Exact affine Gaussian adjoint velocity coefficients."""
    denominator = TARGET_VARIANCE * (
        1.0 - t * (1.0 - outer_variance)
    )

    alpha = (
        -outer_variance
        * (1.0 - TARGET_VARIANCE)
        / denominator
    )

    beta = (
        -(1.0 - TARGET_VARIANCE)
        * outer_mean
        * (1.0 - t)
        / denominator
        + MU / TARGET_VARIANCE
    )

    return alpha, beta


def relax_state(
    old_mean,
    old_variance,
    proposal_mean,
    proposal_variance,
    eta,
):
    """Outer fixed-point relaxation in Gaussian moment coordinates."""
    mean = (
        (1.0 - eta) * old_mean
        + eta * proposal_mean
    )

    variance = (
        (1.0 - eta) * old_variance
        + eta * proposal_variance
    )

    return float(mean), float(variance)


def analytic_sequence(
    outer_iterations,
    eta,
):
    """Exact matched-variance mean recurrence."""
    A = analytic_multiplier()
    relaxed_A = 1.0 - eta + eta * A

    result = [
        {
            "mean": INITIAL_MEAN,
            "variance": INITIAL_VARIANCE,
        }
    ]

    error = INITIAL_MEAN - MU

    for _ in range(outer_iterations):
        error = relaxed_A * error

        result.append(
            {
                "mean": MU + error,
                "variance": TARGET_VARIANCE,
            }
        )

    return result


def em_proposal_moments(
    outer_mean,
    outer_variance,
    times,
):
    """Exact mean/variance of the discretized affine EM rollout.

    The generative process starts from X_0=0. For one EM step,

        X_{n+1}
          = (1 + alpha_n dt_n) X_n
            + beta_n dt_n
            + sqrt(dt_n) xi_n.

    Therefore its first two moments propagate exactly.
    """
    mean = 0.0
    variance = 0.0

    for t, t_next in zip(
        times[:-1],
        times[1:],
    ):
        t = float(t)
        dt = float(t_next - t)

        alpha, beta = oracle_coefficients(
            t,
            outer_mean,
            outer_variance,
        )

        factor = 1.0 + alpha * dt

        mean = factor * mean + beta * dt
        variance = factor**2 * variance + dt

    return float(mean), float(variance)


def em_moment_sequence(
    outer_iterations,
    eta,
    times,
):
    """Outer iteration with exact discrete EM moments."""
    result = [
        {
            "mean": INITIAL_MEAN,
            "variance": INITIAL_VARIANCE,
        }
    ]

    mean = INITIAL_MEAN
    variance = INITIAL_VARIANCE

    for _ in range(outer_iterations):
        proposal_mean, proposal_variance = em_proposal_moments(
            mean,
            variance,
            times,
        )

        mean, variance = relax_state(
            mean,
            variance,
            proposal_mean,
            proposal_variance,
            eta,
        )

        result.append(
            {
                "mean": mean,
                "variance": variance,
            }
        )

    return result


def empirical_moments(samples):
    """Population-convention empirical moments."""
    samples = np.asarray(samples).reshape(-1)

    return (
        float(np.mean(samples)),
        float(np.var(samples, ddof=0)),
    )


def oracle_particle_observation_sequence(
    model,
    deterministic_sequence,
    outer_iterations,
    eta,
    particles,
    key,
):
    """Particle error without feeding that error into the next update.

    At iteration k, the affine field is constructed from the deterministic
    discrete-moment state. The particle rollout is used only to observe
    the numerical Monte Carlo approximation to the proposed state.
    """
    result = [
        {
            "mean": INITIAL_MEAN,
            "variance": INITIAL_VARIANCE,
        }
    ]

    for k in range(outer_iterations):
        current = deterministic_sequence[k]

        model.set_outer_state(
            current["mean"],
            current["variance"],
        )

        key, sample_key = jax.random.split(key)

        endpoints = model.generate_endpoints(
            particles,
            sample_key,
        )

        proposal_mean, proposal_variance = empirical_moments(
            endpoints
        )

        observed_mean, observed_variance = relax_state(
            current["mean"],
            current["variance"],
            proposal_mean,
            proposal_variance,
            eta,
        )

        result.append(
            {
                "mean": observed_mean,
                "variance": observed_variance,
            }
        )

    return result


def oracle_particle_feedback_sequence(
    model,
    outer_iterations,
    eta,
    particles,
    key,
):
    """Particle moments are fed back into the next affine oracle update."""
    result = [
        {
            "mean": INITIAL_MEAN,
            "variance": INITIAL_VARIANCE,
        }
    ]

    mean = INITIAL_MEAN
    variance = INITIAL_VARIANCE

    for _ in range(outer_iterations):
        model.set_outer_state(
            mean,
            variance,
        )

        key, sample_key = jax.random.split(key)

        endpoints = model.generate_endpoints(
            particles,
            sample_key,
        )

        proposal_mean, proposal_variance = empirical_moments(
            endpoints
        )

        mean, variance = relax_state(
            mean,
            variance,
            proposal_mean,
            proposal_variance,
            eta,
        )

        result.append(
            {
                "mean": mean,
                "variance": variance,
            }
        )

    return result


def transition_ratios(sequence):
    """Compute e_{k+1}/e_k."""
    errors = np.asarray(
        [
            entry["mean"] - MU
            for entry in sequence
        ]
    )

    ratios = np.full(
        errors.shape,
        np.nan,
        dtype=float,
    )

    ratios[:-1] = errors[1:] / errors[:-1]

    return ratios


def write_results(
    results,
    output_path,
):
    """Save all iteration diagnostics to one CSV file."""
    csv_path = output_path / "oracle_ablation.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "eta",
                "method",
                "k",
                "mean",
                "error",
                "ratio",
                "variance",
            ],
        )

        writer.writeheader()

        for eta, eta_results in results.items():
            for method, sequence in eta_results.items():
                ratios = transition_ratios(sequence)

                for k, entry in enumerate(sequence):
                    writer.writerow(
                        {
                            "eta": eta,
                            "method": method,
                            "k": k,
                            "mean": entry["mean"],
                            "error": entry["mean"] - MU,
                            "ratio": ratios[k],
                            "variance": entry["variance"],
                        }
                    )

    return csv_path


def plot_mean_error(
    results,
    eta,
    output_path,
):
    """Plot signed mean error for all oracle-stage ablations."""
    fig, ax = plt.subplots(
        figsize=(8, 5),
    )

    for method, sequence in results[eta].items():
        error = np.asarray(
            [
                entry["mean"] - MU
                for entry in sequence
            ]
        )

        ax.plot(
            np.arange(len(error)),
            error,
            marker="o",
            label=method,
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(r"Mean error $m_k-\mu$")
    ax.set_title(
        rf"1D Gaussian Adjoint iteration, $\eta={eta:g}$"
    )
    ax.legend()
    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    filename = (
        "oracle_mean_error_eta1.png"
        if eta == 1.0
        else "oracle_mean_error_eta05.png"
    )

    fig.savefig(
        output_path / filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_variance(
    results,
    output_path,
):
    """Plot terminal variance over outer iterations."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        sharey=True,
    )

    for ax, eta in zip(
        axes,
        ETAS,
    ):
        for method, sequence in results[eta].items():
            variance = np.asarray(
                [
                    entry["variance"]
                    for entry in sequence
                ]
            )

            ax.plot(
                np.arange(len(variance)),
                variance,
                marker="o",
                label=method,
            )

        ax.axhline(
            TARGET_VARIANCE,
            linewidth=1.0,
        )

        ax.set_xlabel("Outer iteration")
        ax.set_title(rf"$\eta={eta:g}$")
        ax.grid(
            True,
            alpha=0.25,
        )

    axes[0].set_ylabel(r"Terminal variance $s_k^2$")
    axes[1].legend()

    fig.tight_layout()

    fig.savefig(
        output_path / "oracle_variance.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def print_table(
    results,
):
    """Print the complete compact diagnostic table."""
    for eta in ETAS:
        predicted = (
            1.0
            - eta
            + eta * analytic_multiplier()
        )

        print("")
        print("============================================================")
        print(f"eta = {eta:g}")
        print(f"continuous predicted multiplier = {predicted:.10f}")
        print("============================================================")

        for method, sequence in results[eta].items():
            ratios = transition_ratios(sequence)

            print("")
            print(method)
            print(
                " k        mean            error"
                "          ratio       variance"
            )

            for k, entry in enumerate(sequence):
                ratio = ratios[k]

                ratio_text = (
                    f"{ratio: .8f}"
                    if np.isfinite(ratio)
                    else "        --"
                )

                print(
                    f"{k:2d}  "
                    f"{entry['mean']: .9f}  "
                    f"{entry['mean'] - MU: .9f}  "
                    f"{ratio_text}  "
                    f"{entry['variance']: .9f}"
                )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outer",
        type=int,
        default=8,
        help="Number of outer iterations.",
    )

    parser.add_argument(
        "--particles",
        type=int,
        default=50000,
        help="Particles in the affine-oracle Monte Carlo runs.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Euler--Maruyama steps on [0,1].",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for oracle particle experiments.",
    )

    parser.add_argument(
        "--output",
        default=(
            "~/ROU-project/adjoint-runs/"
            "instability/oracle"
        ),
        help="Output directory.",
    )

    args = parser.parse_args()

    output_path = Path(
        args.output
    ).expanduser()

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())

    print("")
    print("Target:")
    print(f"  mu                = {MU}")
    print(f"  variance          = {TARGET_VARIANCE}")
    print(f"  initial mean      = {INITIAL_MEAN}")
    print(f"  initial variance  = {INITIAL_VARIANCE}")
    print(f"  outer iterations  = {args.outer}")
    print(f"  EM steps          = {args.steps}")
    print(f"  particles         = {args.particles}")

    A = analytic_multiplier()

    print("")
    print(f"Analytic A = {A:.10f}")
    print(
        "Analytic eta=0.5 multiplier = "
        f"{0.5 + 0.5 * A:.10f}"
    )

    target = GaussianTarget(
        MU,
        TARGET_VARIANCE,
    )

    config = build_config()

    model = AffineOracleSampler(
        config=config,
        densitycl=target,
        h=1.0 / args.steps,
        T=args.steps,
        outer_mean=INITIAL_MEAN,
        outer_variance=INITIAL_VARIANCE,
    )

    times = np.asarray(
        model._build_time_grid(),
        dtype=np.float64,
    )

    first_em_mean, first_em_variance = em_proposal_moments(
        INITIAL_MEAN,
        INITIAL_VARIANCE,
        times,
    )

    first_em_ratio = (
        (first_em_mean - MU)
        / (INITIAL_MEAN - MU)
    )

    print("")
    print("First undamped discrete EM proposal:")
    print(f"  mean       = {first_em_mean:.10f}")
    print(f"  variance   = {first_em_variance:.10f}")
    print(f"  multiplier = {first_em_ratio:.10f}")
    print(
        f"  difference from continuous A = "
        f"{first_em_ratio - A:+.6e}"
    )

    results = {}

    root_key = jax.random.PRNGKey(
        args.seed
    )

    for eta_index, eta in enumerate(ETAS):
        deterministic = em_moment_sequence(
            args.outer,
            eta,
            times,
        )

        observation_key = jax.random.fold_in(
            root_key,
            2 * eta_index,
        )

        feedback_key = jax.random.fold_in(
            root_key,
            2 * eta_index + 1,
        )

        particle_observation = (
            oracle_particle_observation_sequence(
                model=model,
                deterministic_sequence=deterministic,
                outer_iterations=args.outer,
                eta=eta,
                particles=args.particles,
                key=observation_key,
            )
        )

        particle_feedback = (
            oracle_particle_feedback_sequence(
                model=model,
                outer_iterations=args.outer,
                eta=eta,
                particles=args.particles,
                key=feedback_key,
            )
        )

        results[eta] = {
            "analytic": analytic_sequence(
                args.outer,
                eta,
            ),
            "EM moments": deterministic,
            "oracle particles, no feedback": particle_observation,
            "oracle particles, feedback": particle_feedback,
        }

    print_table(
        results
    )

    csv_path = write_results(
        results,
        output_path,
    )

    for eta in ETAS:
        plot_mean_error(
            results,
            eta,
            output_path,
        )

    plot_variance(
        results,
        output_path,
    )

    print("")
    print("Saved:")
    print(f"  {csv_path}")
    print(
        f"  {output_path / 'oracle_mean_error_eta1.png'}"
    )
    print(
        f"  {output_path / 'oracle_mean_error_eta05.png'}"
    )
    print(
        f"  {output_path / 'oracle_variance.png'}"
    )


if __name__ == "__main__":
    main()
