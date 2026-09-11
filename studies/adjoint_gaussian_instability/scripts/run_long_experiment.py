#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Four-regime 1D Gaussian Adjoint-Sampling report experiment.

Shared problem:
    target      N(1.0, 0.02)
    initial p0  N(1.05, 0.02)
    sigma(t)    1
    t           in [0,1]

Regimes:
    oracle
        Exact affine Gaussian adjoint drift.

    learned_affine
        Existing time-embedded neural network trained directly against
        the exact affine Gaussian drift.

    full_adjoint
        Reciprocal adjoint matching with rollout feedback, finite replay,
        and terminal-gradient clipping.

    full_adjoint_damped
        Same as full_adjoint, with eta=0.5 outer parameter relaxation
        after the first learned update.

Outputs:
    trace.csv
    summary.csv
    terminal_samples.npz
    four mean-trace figures
    four terminal-histogram figures
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
from rmc.utils.density import BaseLogDensity


MU = 1.0
TARGET_VARIANCE = 0.02
INITIAL_MEAN = 1.05
INITIAL_VARIANCE = 0.02

REPLAY_CAPACITY = 1000
TARGET_CLIP = 150.0
DAMPING_ETA = 0.5


class GaussianTarget(BaseLogDensity):
    """One-dimensional Gaussian target."""

    def __init__(self, mean, variance):
        self.mean = jnp.asarray(
            [mean],
            dtype=jnp.float32,
        )
        self.variance = float(variance)

    def log_target(self, x):
        return -0.5 * jnp.sum(
            (x - self.mean) ** 2,
            axis=-1,
        ) / self.variance


def analytic_multiplier():
    """Matched-variance continuous outer mean multiplier."""
    return (
        1.0
        + np.log(TARGET_VARIANCE)
        / (1.0 - TARGET_VARIANCE)
    )


def analytic_mean_trace(nouter):
    """Continuous matched-variance prediction."""
    A = analytic_multiplier()

    error = INITIAL_MEAN - MU

    means = [
        INITIAL_MEAN
    ]

    for _ in range(nouter):
        error = A * error
        means.append(
            MU + error
        )

    return np.asarray(
        means,
        dtype=float,
    )


def affine_coefficients(
    t,
    mean,
    variance,
):
    """Exact affine Gaussian adjoint velocity coefficients."""
    denominator = (
        TARGET_VARIANCE
        * (
            1.0
            - t * (1.0 - variance)
        )
    )

    alpha = (
        -variance
        * (1.0 - TARGET_VARIANCE)
        / denominator
    )

    beta = (
        -(1.0 - TARGET_VARIANCE)
        * mean
        * (1.0 - t)
        / denominator
        + MU / TARGET_VARIANCE
    )

    return alpha, beta


def sample_gaussian(
    key,
    mean,
    variance,
    nsamples,
):
    return (
        jnp.asarray(
            mean,
            dtype=jnp.float32,
        )
        + jnp.sqrt(
            jnp.asarray(
                variance,
                dtype=jnp.float32,
            )
        )
        * jax.random.normal(
            key,
            (nsamples, 1),
            dtype=jnp.float32,
        )
    )


def empirical_moments(x):
    values = np.asarray(
        x
    ).reshape(-1)

    return (
        float(np.mean(values)),
        float(np.var(values, ddof=0)),
    )


def build_config(
    seed,
    train_samples,
    batch_size,
    inner_steps,
):
    """Use the practical Adjoint network and optimizer configuration."""
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
        "adjoint_replay_capacity": REPLAY_CAPACITY,
        "adjoint_target_clip": TARGET_CLIP,

        # The experiment explicitly supplies p_0.
        "adjoint_init_base_samples": 0,

        # Shared numerical setup.
        "adjoint_time_discretization": "ql",
    }

    return config


def build_model(
    seed,
    train_samples,
    batch_size,
    inner_steps,
    steps,
):
    return AdjointSampler(
        config=build_config(
            seed,
            train_samples,
            batch_size,
            inner_steps,
        ),
        densitycl=GaussianTarget(
            MU,
            TARGET_VARIANCE,
        ),
        h=1.0 / steps,
        T=steps,
        sigma_schedule=lambda t: 1.0 + 0.0 * t,
        integrated_variance=lambda t: t,
    )


class AffineOracleSampler(AdjointSampler):
    """Sampler using the exact Gaussian affine velocity."""

    def __init__(
        self,
        config,
        steps,
        outer_mean,
        outer_variance,
    ):
        super().__init__(
            config=config,
            densitycl=GaussianTarget(
                MU,
                TARGET_VARIANCE,
            ),
            h=1.0 / steps,
            T=steps,
            sigma_schedule=lambda t: 1.0 + 0.0 * t,
            integrated_variance=lambda t: t,
        )

        self.outer_mean = float(
            outer_mean
        )
        self.outer_variance = float(
            outer_variance
        )

    def set_outer_state(
        self,
        mean,
        variance,
    ):
        self.outer_mean = float(
            mean
        )
        self.outer_variance = float(
            variance
        )

    def eval_drift(
        self,
        x,
        t,
    ):
        alpha, beta = affine_coefficients(
            t,
            self.outer_mean,
            self.outer_variance,
        )

        return alpha * x + beta


def em_proposal_moments(
    mean,
    variance,
    times,
):
    """Exact first two moments of the discrete affine EM rollout."""
    rollout_mean = 0.0
    rollout_variance = 0.0

    for t, t_next in zip(
        times[:-1],
        times[1:],
    ):
        t = float(t)
        dt = float(t_next - t)

        alpha, beta = affine_coefficients(
            t,
            mean,
            variance,
        )

        factor = (
            1.0
            + alpha * dt
        )

        rollout_mean = (
            factor * rollout_mean
            + beta * dt
        )

        rollout_variance = (
            factor**2
            * rollout_variance
            + dt
        )

    return (
        float(rollout_mean),
        float(rollout_variance),
    )


def exact_affine_labels(
    train_input,
    mean,
    variance,
):
    x = train_input[:, :1]
    t = train_input[:, 1]

    alpha, beta = affine_coefficients(
        t,
        mean,
        variance,
    )

    return (
        alpha[:, None] * x
        + beta[:, None]
    )


def train_direct(
    model,
    optimizer,
    metrics,
    endpoints,
    mean,
    variance,
    inner_steps,
    batch_size,
    key,
):
    """Train directly against the exact affine velocity."""
    dummy_gradients = jnp.zeros_like(
        endpoints
    )

    losses = []

    for _ in range(inner_steps):
        key, batch_key = jax.random.split(
            key
        )

        batch = model.build_ram_batch(
            batch_key,
            endpoints,
            dummy_gradients,
            batch_size=batch_size,
        )

        labels = exact_affine_labels(
            batch["input"],
            mean,
            variance,
        )

        model.nnmodel.train()

        loss = train_step(
            model.nnmodel,
            model.compute_ram_loss,
            optimizer,
            metrics,
            batch["input"],
            labels,
            False,
        )

        metrics.reset()
        losses.append(loss)

    return (
        float(
            jnp.mean(
                jnp.asarray(losses)
            )
        ),
        key,
    )


def train_full_adjoint(
    model,
    optimizer,
    metrics,
    replay,
    endpoints,
    inner_steps,
    batch_size,
    key,
):
    """One full reciprocal-adjoint-matching outer training block."""
    gradients = model.eval_terminal_gradient(
        endpoints
    )

    gradients, clip_fraction = (
        model._clip_terminal_gradients(
            gradients,
            TARGET_CLIP,
        )
    )

    replay.add(
        endpoints,
        gradients,
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

        batch = model.build_ram_batch(
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
            batch["input"],
            batch["label"],
            False,
        )

        metrics.reset()
        losses.append(loss)

    return (
        float(
            jnp.mean(
                jnp.asarray(losses)
            )
        ),
        float(clip_fraction),
        key,
    )


def snapshot_params(model):
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
    proposal_params = nnx.state(
        model.nnmodel,
        nnx.Param,
    )

    relaxed = jax.tree.map(
        lambda old, new: (
            (1.0 - eta) * old
            + eta * new
        ),
        previous_params,
        proposal_params,
    )

    nnx.update(
        model.nnmodel,
        relaxed,
    )


def run_oracle(
    outer_iterations,
    samples,
    steps,
):
    """Exact affine drift with deterministic EM moments driving the outer map."""
    config = build_config(
        seed=0,
        train_samples=128,
        batch_size=64,
        inner_steps=100,
    )

    model = AffineOracleSampler(
        config=config,
        steps=steps,
        outer_mean=INITIAL_MEAN,
        outer_variance=INITIAL_VARIANCE,
    )

    times = np.asarray(
        model._build_time_grid(),
        dtype=np.float64,
    )

    mean = INITIAL_MEAN
    variance = INITIAL_VARIANCE

    trace = [
        {
            "k": 0,
            "mean": mean,
            "variance": variance,
        }
    ]

    key = jax.random.PRNGKey(
        1000
    )

    terminal = None

    for outer in range(outer_iterations):
        model.set_outer_state(
            mean,
            variance,
        )

        key, sample_key = jax.random.split(
            key
        )

        terminal = model.generate_endpoints(
            samples,
            sample_key,
        )

        proposal_mean, proposal_variance = (
            em_proposal_moments(
                mean,
                variance,
                times,
            )
        )

        mean = proposal_mean
        variance = proposal_variance

        trace.append(
            {
                "k": outer + 1,
                "mean": mean,
                "variance": variance,
            }
        )

        sample_mean, sample_variance = (
            empirical_moments(
                terminal
            )
        )

        print(
            f"oracle k={outer + 1} "
            f"mean={mean:.8f} "
            f"var={variance:.8f} "
            f"sample_mean={sample_mean:.8f} "
            f"sample_var={sample_variance:.8f}"
        )

    return (
        trace,
        np.asarray(
            terminal
        ).reshape(-1),
    )


def run_learned(
    regime,
    seed,
    outer_iterations,
    inner_steps,
    train_samples,
    eval_samples,
    batch_size,
    steps,
    damping_eta=None,
):
    """Run learned-affine or full Adjoint regimes."""
    model = build_model(
        seed=seed,
        train_samples=train_samples,
        batch_size=batch_size,
        inner_steps=inner_steps,
        steps=steps,
    )

    optimizer = model._build_optimizer()

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    replay = None

    if regime == "full":
        replay = _ReplayBuffer(
            dim=1,
            capacity=REPLAY_CAPACITY,
        )

    # Full and damped-full use the same random stream.
    offset = (
        20000
        if regime == "direct"
        else 30000
    )

    key = jax.random.PRNGKey(
        offset + int(seed)
    )

    key, p0_key = jax.random.split(
        key
    )

    training_endpoints = sample_gaussian(
        p0_key,
        INITIAL_MEAN,
        INITIAL_VARIANCE,
        train_samples,
    )

    mean = INITIAL_MEAN
    variance = INITIAL_VARIANCE

    trace = [
        {
            "k": 0,
            "mean": mean,
            "variance": variance,
        }
    ]

    terminal = None

    for outer in range(outer_iterations):
        previous_params = None

        if (
            regime == "full"
            and damping_eta is not None
            and outer > 0
        ):
            previous_params = snapshot_params(
                model
            )

        if regime == "direct":
            loss, key = train_direct(
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                endpoints=training_endpoints,
                mean=mean,
                variance=variance,
                inner_steps=inner_steps,
                batch_size=batch_size,
                key=key,
            )

            clip_fraction = 0.0
            buffer_size = 0

        elif regime == "full":
            (
                loss,
                clip_fraction,
                key,
            ) = train_full_adjoint(
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                replay=replay,
                endpoints=training_endpoints,
                inner_steps=inner_steps,
                batch_size=batch_size,
                key=key,
            )

            buffer_size = len(replay)

            if previous_params is not None:
                relax_params(
                    model,
                    previous_params,
                    damping_eta,
                )

        else:
            raise ValueError(
                f"Unknown regime: {regime}"
            )

        # Independent population for report diagnostics.
        key, eval_key = jax.random.split(
            key
        )

        terminal = model.generate_endpoints(
            eval_samples,
            eval_key,
        )

        mean, variance = empirical_moments(
            terminal
        )

        trace.append(
            {
                "k": outer + 1,
                "mean": mean,
                "variance": variance,
            }
        )

        # Independent endpoint population for the next outer update.
        if outer + 1 < outer_iterations:
            key, train_key = jax.random.split(
                key
            )

            training_endpoints = (
                model.generate_endpoints(
                    train_samples,
                    train_key,
                )
            )

        eta_text = (
            "1.00"
            if damping_eta is None or outer == 0
            else f"{damping_eta:.2f}"
        )

        print(
            f"{regime:7s} "
            f"seed={seed} "
            f"k={outer + 1} "
            f"eta={eta_text} "
            f"mean={mean:.8f} "
            f"var={variance:.8f} "
            f"loss={loss:.6e} "
            f"clip={clip_fraction:.3f} "
            f"buffer={buffer_size}"
        )

    return (
        trace,
        np.asarray(
            terminal
        ).reshape(-1),
    )


def gaussian_density(x):
    return (
        np.exp(
            -0.5
            * (x - MU) ** 2
            / TARGET_VARIANCE
        )
        / np.sqrt(
            2.0
            * np.pi
            * TARGET_VARIANCE
        )
    )


def plot_trace(
    regime_name,
    traces,
    outer_iterations,
    output_path,
):
    """One clean mean trace per report regime."""
    fig, ax = plt.subplots(
        figsize=(6.2, 3.8),
    )

    analytic = analytic_mean_trace(
        outer_iterations
    )

    ax.plot(
        np.arange(
            len(analytic)
        ),
        analytic,
        linestyle="--",
        marker="o",
        label="continuous prediction",
    )

    if len(traces) == 1:
        trace = next(
            iter(
                traces.values()
            )
        )

        values = np.asarray(
            [
                row["mean"]
                for row in trace
            ]
        )

        ax.plot(
            np.arange(
                len(values)
            ),
            values,
            marker="o",
            label="numerical result",
        )

    else:
        all_values = np.asarray(
            [
                [
                    row["mean"]
                    for row in trace
                ]
                for trace in traces.values()
            ],
            dtype=float,
        )

        for values in all_values:
            ax.plot(
                np.arange(
                    len(values)
                ),
                values,
                marker="o",
                alpha=0.30,
            )

        mean_values = np.mean(
            all_values,
            axis=0,
        )

        ax.plot(
            np.arange(
                len(mean_values)
            ),
            mean_values,
            marker="o",
            linewidth=2.0,
            label="3-seed mean",
        )

    ax.axhline(
        MU,
        linewidth=1.0,
        label="target mean",
    )

    ax.set_xlabel(
        "Outer iteration"
    )

    ax.set_ylabel(
        "Terminal mean"
    )

    ax.set_title(
        regime_name
    )

    ax.grid(
        True,
        alpha=0.20,
    )

    ax.legend(
        fontsize=8,
    )

    fig.tight_layout()

    filename = (
        output_path
        / f"{regime_name}_mean.png"
    )

    fig.savefig(
        filename,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    return filename


def plot_histogram(
    regime_name,
    samples,
    output_path,
):
    """Terminal histogram with exact target-density overlay."""
    samples = np.asarray(
        samples
    ).reshape(-1)

    target_std = np.sqrt(
        TARGET_VARIANCE
    )

    low_sample = np.quantile(
        samples,
        0.002,
    )

    high_sample = np.quantile(
        samples,
        0.998,
    )

    lo = min(
        low_sample,
        MU - 4.0 * target_std,
    )

    hi = max(
        high_sample,
        MU + 4.0 * target_std,
    )

    x = np.linspace(
        lo,
        hi,
        1200,
    )

    fig, ax = plt.subplots(
        figsize=(6.2, 3.8),
    )

    ax.hist(
        samples,
        bins=80,
        density=True,
        alpha=0.55,
        label="terminal samples",
    )

    ax.plot(
        x,
        gaussian_density(x),
        linewidth=2.0,
        label=r"target $N(1,0.02)$",
    )

    ax.set_xlabel(
        "x"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        regime_name
    )

    ax.grid(
        True,
        alpha=0.15,
    )

    ax.legend(
        fontsize=8,
    )

    fig.tight_layout()

    filename = (
        output_path
        / f"{regime_name}_hist.png"
    )

    fig.savefig(
        filename,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    return filename


def write_trace_csv(
    all_traces,
    output_path,
):
    filename = (
        output_path
        / "trace.csv"
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
                "regime",
                "seed",
                "k",
                "mean",
                "variance",
                "mean_error",
            ],
        )

        writer.writeheader()

        for regime, traces in all_traces.items():
            for seed, trace in traces.items():
                for row in trace:
                    writer.writerow(
                        {
                            "regime": regime,
                            "seed": seed,
                            "k": row["k"],
                            "mean": row["mean"],
                            "variance": row[
                                "variance"
                            ],
                            "mean_error": (
                                row["mean"]
                                - MU
                            ),
                        }
                    )

    return filename


def write_summary_csv(
    terminal_samples,
    output_path,
):
    filename = (
        output_path
        / "summary.csv"
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
                "regime",
                "sample_count",
                "terminal_mean",
                "terminal_variance",
                "target_mean",
                "target_variance",
            ],
        )

        writer.writeheader()

        for regime, samples in terminal_samples.items():
            samples = np.asarray(
                samples
            ).reshape(-1)

            writer.writerow(
                {
                    "regime": regime,
                    "sample_count": len(samples),
                    "terminal_mean": float(
                        np.mean(samples)
                    ),
                    "terminal_variance": float(
                        np.var(
                            samples,
                            ddof=0,
                        )
                    ),
                    "target_mean": MU,
                    "target_variance": TARGET_VARIANCE,
                }
            )

    return filename


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outer",
        type=int,
        default=4,
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
        default=10000,
    )

    parser.add_argument(
        "--oracle-samples",
        type=int,
        default=50000,
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
        "--eta",
        type=float,
        default=DAMPING_ETA,
    )

    parser.add_argument(
        "--output",
        default=(
            "~/ROU-project/adjoint-runs/"
            "gaussian002-report"
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
    print("1D Gaussian Adjoint report experiment")
    print("============================================================")
    print(f"target:           N({MU}, {TARGET_VARIANCE})")
    print(
        f"initial:          "
        f"N({INITIAL_MEAN}, {INITIAL_VARIANCE})"
    )
    print("sigma(t):         1")
    print(f"analytic A:       {analytic_multiplier():.10f}")
    print(
        f"analytic eta=.5:  "
        f"{0.5 + 0.5 * analytic_multiplier():.10f}"
    )
    print(f"outer:            {args.outer}")
    print(f"inner:            {args.inner}")
    print(f"train samples:    {args.train_samples}")
    print(f"eval samples:     {args.eval_samples}")
    print(f"oracle samples:   {args.oracle_samples}")
    print(f"steps:            {args.steps}")
    print("time grid:        ql")
    print(f"seeds:            {args.seeds}")
    print(f"damping eta:      {args.eta}")

    all_traces = {}
    terminal_by_regime = {}
    npz_data = {}

    print("")
    print("================ AFFINE ORACLE ================")

    oracle_trace, oracle_terminal = run_oracle(
        outer_iterations=args.outer,
        samples=args.oracle_samples,
        steps=args.steps,
    )

    all_traces["affine_oracle"] = {
        0: oracle_trace
    }

    terminal_by_regime[
        "affine_oracle"
    ] = oracle_terminal

    npz_data[
        "affine_oracle"
    ] = oracle_terminal

    print("")
    print("================ LEARNED AFFINE ================")

    all_traces[
        "learned_affine"
    ] = {}

    learned_affine_samples = []

    for seed in args.seeds:
        trace, terminal = run_learned(
            regime="direct",
            seed=seed,
            outer_iterations=args.outer,
            inner_steps=args.inner,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            batch_size=args.batch_size,
            steps=args.steps,
        )

        all_traces[
            "learned_affine"
        ][seed] = trace

        learned_affine_samples.append(
            terminal
        )

        npz_data[
            f"learned_affine_seed{seed}"
        ] = terminal

    terminal_by_regime[
        "learned_affine"
    ] = np.concatenate(
        learned_affine_samples
    )

    print("")
    print("================ FULL ADJOINT ================")

    all_traces[
        "full_adjoint"
    ] = {}

    full_samples = []

    for seed in args.seeds:
        trace, terminal = run_learned(
            regime="full",
            seed=seed,
            outer_iterations=args.outer,
            inner_steps=args.inner,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            batch_size=args.batch_size,
            steps=args.steps,
            damping_eta=None,
        )

        all_traces[
            "full_adjoint"
        ][seed] = trace

        full_samples.append(
            terminal
        )

        npz_data[
            f"full_adjoint_seed{seed}"
        ] = terminal

    terminal_by_regime[
        "full_adjoint"
    ] = np.concatenate(
        full_samples
    )

    print("")
    print("============= FULL ADJOINT + DAMPING =============")

    all_traces[
        "full_adjoint_damped"
    ] = {}

    damped_samples = []

    for seed in args.seeds:
        trace, terminal = run_learned(
            regime="full",
            seed=seed,
            outer_iterations=args.outer,
            inner_steps=args.inner,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            batch_size=args.batch_size,
            steps=args.steps,
            damping_eta=args.eta,
        )

        all_traces[
            "full_adjoint_damped"
        ][seed] = trace

        damped_samples.append(
            terminal
        )

        npz_data[
            f"full_adjoint_damped_seed{seed}"
        ] = terminal

    terminal_by_regime[
        "full_adjoint_damped"
    ] = np.concatenate(
        damped_samples
    )

    trace_csv = write_trace_csv(
        all_traces,
        output_path,
    )

    summary_csv = write_summary_csv(
        terminal_by_regime,
        output_path,
    )

    npz_path = (
        output_path
        / "terminal_samples.npz"
    )

    np.savez(
        npz_path,
        **npz_data,
    )

    for regime, traces in all_traces.items():
        plot_trace(
            regime,
            traces,
            args.outer,
            output_path,
        )

        plot_histogram(
            regime,
            terminal_by_regime[
                regime
            ],
            output_path,
        )

    print("")
    print("================ SUMMARY ================")

    for regime, samples in terminal_by_regime.items():
        mean, variance = empirical_moments(
            samples
        )

        print(
            f"{regime:24s} "
            f"mean={mean: .8f} "
            f"variance={variance: .8f}"
        )

    print("")
    print("Saved:")
    print(f"  {trace_csv}")
    print(f"  {summary_csv}")
    print(f"  {npz_path}")

    for regime in all_traces:
        print(
            f"  {output_path / (regime + '_mean.png')}"
        )
        print(
            f"  {output_path / (regime + '_hist.png')}"
        )


if __name__ == "__main__":
    main()
