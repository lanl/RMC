#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
1D Adjoint Sampling failure-mode suite.

Four problems, all with component variance 0.05:

1. single_gaussian
   target:  N(1, 0.05)
   initial: N(1.01, 0.05)

2. displaced_modes
   target:  0.5 N(-1, 0.05) + 0.5 N(1, 0.05)
   initial: 0.5 N(-1.01, 0.05) + 0.5 N(1.01, 0.05)

3. wrong_weights
   target:  (10/11) N(-1, 0.05) + (1/11) N(1, 0.05)
   initial: 0.5 N(-1, 0.05) + 0.5 N(1, 0.05)

4. missing_mode
   target:  0.5 N(-1, 0.05) + 0.5 N(1, 0.05)
   initial: N(1, 0.05)

For every problem:

    replay       off/on
    velocity damping off/on

Velocity damping uses the direct function-space RAM target

    target_damped
        = eta * target_RAM
          + (1 - eta) * u_previous,

with one frozen lagged network.  The first learned update is undamped
because the initial endpoint law is externally prescribed.

Default eta = 0.5.

All variants within one problem/seed use the same initial endpoint
population and PRNG stream wherever their algorithms permit it.
"""

import argparse
import csv
import functools
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from rmc.utils.density import BaseLogDensity

import run_experiment as base
import run_damping_v2_experiment as damping


COMPONENT_VARIANCE = 0.05
REPLAY_CAPACITY = base.REPLAY_CAPACITY

STOP_MEAN_ERROR = 100.0
STOP_VARIANCE = 1000.0
STOP_SAMPLE_ABS = 1000.0


PROBLEMS = {
    "single_gaussian": {
        "target_means": [1.0],
        "target_weights": [1.0],
        "initial_means": [1.01],
        "initial_weights": [1.0],
        "key_offset": 50000,
        "target_ref_seed": 101,
    },
    "displaced_modes": {
        "target_means": [-1.0, 1.0],
        "target_weights": [0.5, 0.5],
        "initial_means": [-1.01, 1.01],
        "initial_weights": [0.5, 0.5],
        "key_offset": 60000,
        "target_ref_seed": 102,
    },
    "wrong_weights": {
        "target_means": [-1.0, 1.0],
        "target_weights": [10.0 / 11.0, 1.0 / 11.0],
        "initial_means": [-1.0, 1.0],
        "initial_weights": [0.5, 0.5],
        "key_offset": 70000,
        "target_ref_seed": 103,
    },
    "missing_mode": {
        "target_means": [-1.0, 1.0],
        "target_weights": [0.5, 0.5],
        "initial_means": [1.0],
        "initial_weights": [1.0],
        "key_offset": 80000,
        "target_ref_seed": 104,
    },
}


TRACE_FIELDS = [
    "problem",
    "regime",
    "seed",
    "k",
    "replay",
    "u_damping",
    "eta",
    "applied_eta",
    "mean",
    "variance",
    "second_moment",
    "w2",
    "p_left",
    "left_mean",
    "right_mean",
    "left_variance",
    "right_variance",
    "half_separation",
    "center",
    "training_left_count",
    "training_n",
    "replay_left_count",
    "replay_n",
    "inner_steps",
    "loss",
    "clip_fraction",
    "stop_reason",
]


SUMMARY_FIELDS = [
    "problem",
    "regime",
    "seed",
    "completed_k",
    "n",
    "mean",
    "variance",
    "second_moment",
    "w2",
    "p_left",
    "left_mean",
    "right_mean",
    "left_variance",
    "right_variance",
    "half_separation",
    "center",
    "stop_reason",
]


SETTINGS_FIELDS = [
    "problem",
    "target_means",
    "target_weights",
    "initial_means",
    "initial_weights",
    "component_variance",
]


INIT_KEY_OFFSET = 1_000_000

INITIALIZATION_FIELDS = [
    "problem",
    "seed",
    "initial_samples",
    "initial_inner_steps",
    "eval_samples",
    "mean",
    "exact_mean",
    "variance",
    "exact_variance",
    "p_left",
    "reference_p_left",
    "w2_to_initial",
    "loss",
    "clip_fraction",
]


class GaussianMixture1DTarget(BaseLogDensity):
    """One-dimensional Gaussian mixture with common component variance."""

    def __init__(
        self,
        means,
        weights,
        variance,
    ):
        self.means = jnp.asarray(
            means,
            dtype=jnp.float32,
        )

        self.weights = jnp.asarray(
            weights,
            dtype=jnp.float32,
        )

        self.variance = float(
            variance
        )

        self.log_weights = jnp.log(
            self.weights
        )

        self.log_norm = (
            -0.5
            * jnp.log(
                2.0
                * jnp.pi
                * self.variance
            )
        )

    def log_target(self, x):
        x = jnp.asarray(x)

        z = x[..., 0][..., None]

        log_components = (
            self.log_weights
            + self.log_norm
            - 0.5
            * (
                z - self.means
            ) ** 2
            / self.variance
        )

        return jax.scipy.special.logsumexp(
            log_components,
            axis=-1,
        )


def append_csv(
    path,
    fields,
    row,
):
    exists = path.exists()

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def exact_component_counts(
    weights,
    n,
):
    weights = np.asarray(
        weights,
        dtype=float,
    )

    expected = weights * n

    counts = np.floor(
        expected
    ).astype(int)

    remainder = (
        n - int(counts.sum())
    )

    if remainder:
        order = np.argsort(
            -(expected - counts)
        )

        for i in order[:remainder]:
            counts[i] += 1

    assert int(counts.sum()) == n

    return counts


def sample_mixture_jax(
    key,
    means,
    weights,
    variance,
    n,
):
    """
    Stratified mixture sampling.

    Component counts are deterministic, so nominal 50/50 initial
    mixtures are exactly balanced rather than only balanced in expectation.
    """
    means = np.asarray(
        means,
        dtype=float,
    )

    counts = exact_component_counts(
        weights,
        n,
    )

    keys = jax.random.split(
        key,
        len(means),
    )

    chunks = []

    std = float(
        np.sqrt(variance)
    )

    for (
        mean,
        count,
        subkey,
    ) in zip(
        means,
        counts,
        keys,
    ):
        if count == 0:
            continue

        noise = jax.random.normal(
            subkey,
            (int(count), 1),
            dtype=jnp.float32,
        )

        chunks.append(
            float(mean)
            + std * noise
        )

    return jnp.concatenate(
        chunks,
        axis=0,
    )


def sample_mixture_numpy(
    rng,
    means,
    weights,
    variance,
    n,
):
    means = np.asarray(
        means,
        dtype=float,
    )

    counts = exact_component_counts(
        weights,
        n,
    )

    chunks = []

    std = float(
        np.sqrt(variance)
    )

    for mean, count in zip(
        means,
        counts,
    ):
        if count == 0:
            continue

        chunks.append(
            rng.normal(
                loc=mean,
                scale=std,
                size=int(count),
            )
        )

    values = np.concatenate(
        chunks
    )

    rng.shuffle(values)

    return values


@functools.lru_cache(
    maxsize=None,
)
def target_quantiles(
    problem_name,
    n,
    reference_n,
):
    spec = PROBLEMS[
        problem_name
    ]

    rng = np.random.default_rng(
        spec["target_ref_seed"]
    )

    reference = sample_mixture_numpy(
        rng=rng,
        means=spec[
            "target_means"
        ],
        weights=spec[
            "target_weights"
        ],
        variance=COMPONENT_VARIANCE,
        n=reference_n,
    )

    q = (
        np.arange(n)
        + 0.5
    ) / n

    return np.quantile(
        reference,
        q,
    )


def target_mean(
    spec,
):
    return float(
        np.dot(
            np.asarray(
                spec[
                    "target_weights"
                ],
                dtype=float,
            ),
            np.asarray(
                spec[
                    "target_means"
                ],
                dtype=float,
            ),
        )
    )


def diagnostics(
    terminal,
    target_q,
):
    x = np.asarray(
        terminal
    ).reshape(-1)

    mean = float(
        np.mean(x)
    )

    variance = float(
        np.var(x)
    )

    second_moment = float(
        np.mean(x**2)
    )

    sorted_x = np.sort(x)

    w2 = float(
        np.sqrt(
            np.mean(
                (
                    sorted_x
                    - target_q
                ) ** 2
            )
        )
    )

    left = x[x < 0.0]
    right = x[x >= 0.0]

    p_left = float(
        left.size / x.size
    )

    left_mean = (
        float(np.mean(left))
        if left.size
        else np.nan
    )

    right_mean = (
        float(np.mean(right))
        if right.size
        else np.nan
    )

    left_variance = (
        float(np.var(left))
        if left.size >= 2
        else np.nan
    )

    right_variance = (
        float(np.var(right))
        if right.size >= 2
        else np.nan
    )

    if (
        np.isfinite(left_mean)
        and np.isfinite(
            right_mean
        )
    ):
        half_separation = (
            0.5
            * (
                right_mean
                - left_mean
            )
        )

        center = (
            0.5
            * (
                right_mean
                + left_mean
            )
        )

    else:
        half_separation = np.nan
        center = np.nan

    return {
        "mean": mean,
        "variance": variance,
        "second_moment": (
            second_moment
        ),
        "w2": w2,
        "p_left": p_left,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "left_variance": (
            left_variance
        ),
        "right_variance": (
            right_variance
        ),
        "half_separation": (
            half_separation
        ),
        "center": center,
    }


def stopping_reason(
    stats,
    terminal,
    target_mean_value,
):
    for key in (
        "mean",
        "variance",
        "second_moment",
        "w2",
    ):
        if not np.isfinite(
            stats[key]
        ):
            return (
                f"nonfinite_{key}"
            )

    values = np.asarray(
        terminal
    ).reshape(-1)

    if not np.all(
        np.isfinite(values)
    ):
        return (
            "nonfinite_samples"
        )

    if (
        abs(
            stats["mean"]
            - target_mean_value
        )
        > STOP_MEAN_ERROR
    ):
        return (
            f"mean_error>"
            f"{STOP_MEAN_ERROR:g}"
        )

    if (
        stats["variance"]
        > STOP_VARIANCE
    ):
        return (
            f"variance>"
            f"{STOP_VARIANCE:g}"
        )

    if (
        values.size
        and np.max(
            np.abs(values)
        )
        > STOP_SAMPLE_ABS
    ):
        return (
            f"sample_abs>"
            f"{STOP_SAMPLE_ABS:g}"
        )

    return ""


def count_left(
    endpoints,
):
    if endpoints is None:
        return 0

    x = np.asarray(
        endpoints
    ).reshape(-1)

    return int(
        np.sum(x < 0.0)
    )


def make_model(
    target,
    seed,
    train_samples,
    batch_size,
    inner_steps,
    steps,
):
    # Reuse the already validated 1D Gaussian experimental
    # configuration, then replace only the target density.
    model = base.build_model(
        seed=seed,
        train_samples=train_samples,
        batch_size=batch_size,
        inner_steps=inner_steps,
        steps=steps,
    )

    model.Dcl = target

    return model


def regime_name(
    use_replay,
    use_u_damping,
):
    replay = (
        "replay"
        if use_replay
        else "no_replay"
    )

    damping_name = (
        "u_damped"
        if use_u_damping
        else "undamped"
    )

    return (
        f"{replay}_"
        f"{damping_name}"
    )



def sample_initial_reference(
    spec,
    n,
    seed,
):
    rng = np.random.default_rng(
        int(seed)
    )

    means = np.asarray(
        spec["initial_means"],
        dtype=float,
    )

    weights = np.asarray(
        spec["initial_weights"],
        dtype=float,
    )

    counts = exact_component_counts(
        weights,
        n,
    )

    chunks = []

    for mean, count in zip(
        means,
        counts,
    ):
        if count == 0:
            continue

        chunks.append(
            rng.normal(
                loc=mean,
                scale=np.sqrt(
                    COMPONENT_VARIANCE
                ),
                size=int(count),
            )
        )

    values = np.concatenate(
        chunks
    )

    rng.shuffle(values)

    return values


def initialization_statistics(
    rollout,
    spec,
    reference_samples=200000,
    reference_seed=123456,
):
    x = np.asarray(
        rollout,
        dtype=float,
    ).reshape(-1)

    reference = sample_initial_reference(
        spec=spec,
        n=reference_samples,
        seed=reference_seed,
    )

    x_sorted = np.sort(
        x
    )

    q = (
        np.arange(
            x_sorted.size,
            dtype=float,
        )
        + 0.5
    ) / x_sorted.size

    reference_q = np.quantile(
        reference,
        q,
    )

    w2 = float(
        np.sqrt(
            np.mean(
                (
                    x_sorted
                    - reference_q
                ) ** 2
            )
        )
    )

    weights = np.asarray(
        spec["initial_weights"],
        dtype=float,
    )

    means = np.asarray(
        spec["initial_means"],
        dtype=float,
    )

    exact_mean = float(
        np.sum(
            weights * means
        )
    )

    exact_variance = float(
        COMPONENT_VARIANCE
        + np.sum(
            weights
            * (
                means
                - exact_mean
            ) ** 2
        )
    )

    return {
        "mean": float(
            np.mean(x)
        ),
        "exact_mean": (
            exact_mean
        ),
        "variance": float(
            np.var(x)
        ),
        "exact_variance": (
            exact_variance
        ),
        "p_left": float(
            np.mean(
                x < 0.0
            )
        ),
        "reference_p_left": float(
            np.mean(
                reference < 0.0
            )
        ),
        "w2_to_initial": (
            w2
        ),
    }


def initialize_control(
    problem_name,
    spec,
    seed,
    initial_samples,
    initial_inner_steps,
    eval_samples,
    batch_size,
    steps,
    initialization_trace_path,
    initialization_terminal_dir,
):
    """
    Learn one initial control u_0 whose RAM endpoint population and
    RAM target are both the prescribed analytic initial law p^(0).

    This preprocessing replay/optimizer state is discarded afterward.
    Only the trained network parameters are retained.
    """

    initial_density = (
        GaussianMixture1DTarget(
            means=spec[
                "initial_means"
            ],
            weights=spec[
                "initial_weights"
            ],
            variance=(
                COMPONENT_VARIANCE
            ),
        )
    )

    model = make_model(
        target=initial_density,
        seed=seed,
        train_samples=initial_samples,
        batch_size=batch_size,
        inner_steps=(
            initial_inner_steps
        ),
        steps=steps,
    )

    optimizer = (
        model._build_optimizer()
    )

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average(
            "loss"
        ),
    )

    # This replay belongs only to preprocessing.
    # It is deliberately discarded before the actual experiment.
    replay = base._ReplayBuffer(
        dim=1,
        capacity=max(
            initial_samples,
            batch_size,
        ),
    )

    # Separate PRNG namespace from the actual outer experiment.
    key = jax.random.PRNGKey(
        int(
            spec["key_offset"]
        )
        + int(seed)
        + INIT_KEY_OFFSET
    )

    (
        key,
        p0_key,
        _,
    ) = jax.random.split(
        key,
        3,
    )

    endpoints = (
        sample_mixture_jax(
            key=p0_key,
            means=spec[
                "initial_means"
            ],
            weights=spec[
                "initial_weights"
            ],
            variance=(
                COMPONENT_VARIANCE
            ),
            n=initial_samples,
        )
    )

    (
        loss,
        clip_fraction,
        key,
    ) = base.train_full_adjoint(
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        replay=replay,
        endpoints=endpoints,
        inner_steps=(
            initial_inner_steps
        ),
        batch_size=batch_size,
        key=key,
    )

    # Diagnostic rollout only.  It does NOT become the endpoint
    # population for the first target-informed outer update.
    key, rollout_key = (
        jax.random.split(key)
    )

    rollout = (
        model.generate_endpoints(
            eval_samples,
            rollout_key,
        )
    )

    initialization_terminal_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        initialization_terminal_dir
        / (
            f"{problem_name}__"
            f"seed{seed}.npy"
        ),
        np.asarray(
            rollout
        ).reshape(-1),
    )

    stats = initialization_statistics(
        rollout,
        spec,
        reference_seed=(
            int(
                spec[
                    "target_ref_seed"
                ]
            )
            + 500000
            + int(seed)
        ),
    )

    append_csv(
        initialization_trace_path,
        INITIALIZATION_FIELDS,
        {
            "problem": problem_name,
            "seed": seed,
            "initial_samples": (
                initial_samples
            ),
            "initial_inner_steps": (
                initial_inner_steps
            ),
            "eval_samples": (
                eval_samples
            ),
            **stats,
            "loss": loss,
            "clip_fraction": (
                clip_fraction
            ),
        },
    )

    print(
        f"INIT "
        f"{problem_name:18s} "
        f"seed={seed} "
        f"W2(p_u0,p0)="
        f"{stats['w2_to_initial']:.6f} "
        f"mean="
        f"{stats['mean']:.6f}/"
        f"{stats['exact_mean']:.6f} "
        f"var="
        f"{stats['variance']:.6f}/"
        f"{stats['exact_variance']:.6f}",
        flush=True,
    )

    # Return only an independent copy of u_0.
    # Optimizer and preprocessing replay state are intentionally lost.
    return nnx.clone(
        model.nnmodel
    )


def run_one(
    problem_name,
    spec,
    initialized_nn,
    use_replay,
    use_u_damping,
    eta,
    seed,
    outer_iterations,
    inner_steps,
    train_samples,
    eval_samples,
    batch_size,
    steps,
    target_reference_samples,
    trace_path,
    summary_path,
    terminal_dir,
):
    target = GaussianMixture1DTarget(
        means=spec[
            "target_means"
        ],
        weights=spec[
            "target_weights"
        ],
        variance=COMPONENT_VARIANCE,
    )

    model = make_model(
        target=target,
        seed=seed,
        train_samples=train_samples,
        batch_size=batch_size,
        inner_steps=inner_steps,
        steps=steps,
    )

    # All regimes for this problem/seed start from the same
    # independently cloned initialized control u_0.
    model.nnmodel = nnx.clone(
        initialized_nn
    )

    # Fresh optimizer state for the actual experiment.
    optimizer = (
        model._build_optimizer()
    )

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average(
            "loss"
        ),
    )

    persistent_replay = None

    if use_replay:
        persistent_replay = (
            base._ReplayBuffer(
                dim=1,
                capacity=REPLAY_CAPACITY,
            )
        )

    key = jax.random.PRNGKey(
        int(
            spec["key_offset"]
        )
        + int(seed)
    )

    (
        key,
        p0_train_key,
        p0_eval_key,
    ) = jax.random.split(
        key,
        3,
    )

    training_endpoints = (
        sample_mixture_jax(
            key=p0_train_key,
            means=spec[
                "initial_means"
            ],
            weights=spec[
                "initial_weights"
            ],
            variance=COMPONENT_VARIANCE,
            n=train_samples,
        )
    )

    initial_terminal = (
        sample_mixture_jax(
            key=p0_eval_key,
            means=spec[
                "initial_means"
            ],
            weights=spec[
                "initial_weights"
            ],
            variance=COMPONENT_VARIANCE,
            n=eval_samples,
        )
    )

    target_q = target_quantiles(
        problem_name,
        eval_samples,
        target_reference_samples,
    )

    stats = diagnostics(
        initial_terminal,
        target_q,
    )

    target_mean_value = (
        target_mean(spec)
    )

    regime = regime_name(
        use_replay=use_replay,
        use_u_damping=(
            use_u_damping
        ),
    )

    append_csv(
        trace_path,
        TRACE_FIELDS,
        {
            "problem": problem_name,
            "regime": regime,
            "seed": seed,
            "k": 0,
            "replay": (
                "on"
                if use_replay
                else "off"
            ),
            "u_damping": (
                "on"
                if use_u_damping
                else "off"
            ),
            "eta": (
                eta
                if use_u_damping
                else ""
            ),
            "applied_eta": "",
            **stats,
            "training_left_count": (
                count_left(
                    training_endpoints
                )
            ),
            "training_n": (
                train_samples
            ),
            "replay_left_count": "",
            "replay_n": "",
            "inner_steps": 0,
            "loss": "",
            "clip_fraction": "",
            "stop_reason": "",
        },
    )

    terminal = None
    completed_k = 0
    final_reason = ""

    for outer in range(
        outer_iterations
    ):
        current_inner = (
            inner_steps
        )

        training_left_count = (
            count_left(
                training_endpoints
            )
        )

        previous_nn = None

        if use_u_damping:
            previous_nn = (
                damping.clone_nnmodel(
                    model
                )
            )

        if use_replay:
            replay = (
                persistent_replay
            )
        else:
            replay = (
                base._ReplayBuffer(
                    dim=1,
                    capacity=max(
                        train_samples,
                        batch_size,
                    ),
                )
            )

        if use_u_damping:
            applied_eta = (
                eta
            )

            (
                loss,
                clip_fraction,
                key,
            ) = (
                damping.train_full_adjoint_u_damped(
                    model=model,
                    optimizer=optimizer,
                    metrics=metrics,
                    replay=replay,
                    endpoints=training_endpoints,
                    inner_steps=current_inner,
                    batch_size=batch_size,
                    key=key,
                    previous_nn=previous_nn,
                    eta=applied_eta,
                )
            )

        else:
            applied_eta = 1.0

            (
                loss,
                clip_fraction,
                key,
            ) = (
                base.train_full_adjoint(
                    model=model,
                    optimizer=optimizer,
                    metrics=metrics,
                    replay=replay,
                    endpoints=training_endpoints,
                    inner_steps=current_inner,
                    batch_size=batch_size,
                    key=key,
                )
            )

        replay_left_count = (
            count_left(
                replay.endpoints
            )
        )

        replay_n = len(
            replay
        )

        if not np.isfinite(
            loss
        ):
            final_reason = (
                "nonfinite_loss"
            )

            break

        key, eval_key = (
            jax.random.split(
                key
            )
        )

        terminal = (
            model.generate_endpoints(
                eval_samples,
                eval_key,
            )
        )

        stats = diagnostics(
            terminal,
            target_q,
        )

        final_reason = (
            stopping_reason(
                stats,
                terminal,
                target_mean_value,
            )
        )

        completed_k = (
            outer + 1
        )

        append_csv(
            trace_path,
            TRACE_FIELDS,
            {
                "problem": problem_name,
                "regime": regime,
                "seed": seed,
                "k": completed_k,
                "replay": (
                    "on"
                    if use_replay
                    else "off"
                ),
                "u_damping": (
                    "on"
                    if use_u_damping
                    else "off"
                ),
                "eta": (
                    eta
                    if use_u_damping
                    else ""
                ),
                "applied_eta": (
                    applied_eta
                ),
                **stats,
                "training_left_count": (
                    training_left_count
                ),
                "training_n": (
                    train_samples
                ),
                "replay_left_count": (
                    replay_left_count
                ),
                "replay_n": (
                    replay_n
                ),
                "inner_steps": (
                    current_inner
                ),
                "loss": loss,
                "clip_fraction": (
                    clip_fraction
                ),
                "stop_reason": (
                    final_reason
                ),
            },
        )

        np.save(
            terminal_dir
            / (
                f"{problem_name}__"
                f"{regime}__"
                f"seed{seed}.npy"
            ),
            np.asarray(
                terminal
            ).reshape(-1),
        )

        if (
            problem_name
            == "single_gaussian"
        ):
            primary = (
                f"mean="
                f"{stats['mean']: .6f}"
            )

        elif (
            problem_name
            == "displaced_modes"
        ):
            primary = (
                f"sep="
                f"{stats['half_separation']: .6f}"
            )

        else:
            primary = (
                f"pL="
                f"{stats['p_left']: .6f}"
            )

        print(
            f"{problem_name:18s} "
            f"{regime:20s} "
            f"seed={seed} "
            f"k={completed_k:3d} "
            f"eta={applied_eta:.3f} "
            f"{primary} "
            f"W2={stats['w2']:.6f} "
            f"loss={loss:.3e} "
            f"clip={clip_fraction:.3f}"
            + (
                f" STOP={final_reason}"
                if final_reason
                else ""
            ),
            flush=True,
        )

        if final_reason:
            break

        if (
            completed_k
            < outer_iterations
        ):
            key, train_key = (
                jax.random.split(
                    key
                )
            )

            training_endpoints = (
                model.generate_endpoints(
                    train_samples,
                    train_key,
                )
            )

    if terminal is not None:
        append_csv(
            summary_path,
            SUMMARY_FIELDS,
            {
                "problem": problem_name,
                "regime": regime,
                "seed": seed,
                "completed_k": (
                    completed_k
                ),
                "n": int(
                    np.asarray(
                        terminal
                    ).size
                ),
                **stats,
                "stop_reason": (
                    final_reason
                ),
            },
        )


def write_problem_settings(
    path,
):
    for (
        problem_name,
        spec,
    ) in PROBLEMS.items():
        append_csv(
            path,
            SETTINGS_FIELDS,
            {
                "problem": (
                    problem_name
                ),
                "target_means": (
                    repr(
                        spec[
                            "target_means"
                        ]
                    )
                ),
                "target_weights": (
                    repr(
                        spec[
                            "target_weights"
                        ]
                    )
                ),
                "initial_means": (
                    repr(
                        spec[
                            "initial_means"
                        ]
                    )
                ),
                "initial_weights": (
                    repr(
                        spec[
                            "initial_weights"
                        ]
                    )
                ),
                "component_variance": (
                    COMPONENT_VARIANCE
                ),
            },
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outer",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--inner",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--initial-inner",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--initial-samples",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--eval-samples",
        type=int,
        default=10000,
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
        "--problems",
        nargs="+",
        choices=list(
            PROBLEMS.keys()
        ),
        default=list(
            PROBLEMS.keys()
        ),
    )

    parser.add_argument(
        "--target-reference-samples",
        type=int,
        default=200000,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "studies/"
            "adjoint_gaussian_instability/"
            "results/"
            "multimodal_initialized_v2_50"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    out = args.output

    if (
        out.exists()
        and any(
            out.iterdir()
        )
    ):
        if args.overwrite:
            shutil.rmtree(
                out
            )
        else:
            raise SystemExit(
                f"{out} already exists "
                "and is not empty. "
                "Use --overwrite."
            )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    terminal_dir = (
        out
        / "terminal_samples"
    )
    terminal_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    initialization_terminal_dir = (
        out
        / "initialization_samples"
    )
    initialization_terminal_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trace_path = (
        out
        / "trace.csv"
    )

    summary_path = (
        out
        / "summary.csv"
    )

    initialization_trace_path = (
        out
        / "initialization.csv"
    )

    settings_path = (
        out
        / "problem_settings.csv"
    )

    write_problem_settings(
        settings_path
    )

    single_A = (
        1.0
        + np.log(
            COMPONENT_VARIANCE
        )
        / (
            1.0
            - COMPONENT_VARIANCE
        )
    )

    single_A_eta = (
        1.0
        - args.eta
        + args.eta
        * single_A
    )

    print("=" * 78)
    print(
        "1D Adjoint failure-mode suite "
        "(initialized-control version)"
    )
    print("=" * 78)
    print(
        f"component variance: "
        f"{COMPONENT_VARIANCE}"
    )
    print(
        f"outer updates:      "
        f"{args.outer}"
    )
    print(
        f"initial samples:    "
        f"{args.initial_samples}"
    )
    print(
        f"initial RAM steps:  "
        f"{args.initial_inner}"
    )
    print(
        f"outer RAM steps:    "
        f"{args.inner}"
    )
    print(
        f"outer train samples:"
        f" {args.train_samples}"
    )
    print(
        f"eval samples:       "
        f"{args.eval_samples}"
    )
    print(
        f"batch size:         "
        f"{args.batch_size}"
    )
    print(
        f"EM steps:           "
        f"{args.steps}"
    )
    print(
        f"velocity eta:       "
        f"{args.eta}"
    )
    print(
        f"seeds:              "
        f"{args.seeds}"
    )
    print(
        f"single-Gaussian A:  "
        f"{single_A:.10f}"
    )
    print(
        f"single A_eta:       "
        f"{single_A_eta:.10f}"
    )
    print("=" * 78)

    regimes = (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    )

    for problem_name in (
        args.problems
    ):
        spec = PROBLEMS[
            problem_name
        ]

        print(
            f"\n######## "
            f"{problem_name.upper()} "
            f"########",
            flush=True,
        )

        # Initialization is stochastic, so do it once per seed.
        # Every regime for this problem/seed receives an independent
        # clone of exactly this same u_0.
        for seed in args.seeds:
            print(
                f"\n----- INITIALIZING "
                f"seed={seed} -----",
                flush=True,
            )

            initialized_nn = (
                initialize_control(
                    problem_name=(
                        problem_name
                    ),
                    spec=spec,
                    seed=seed,
                    initial_samples=(
                        args.initial_samples
                    ),
                    initial_inner_steps=(
                        args.initial_inner
                    ),
                    eval_samples=(
                        args.eval_samples
                    ),
                    batch_size=(
                        args.batch_size
                    ),
                    steps=args.steps,
                    initialization_trace_path=(
                        initialization_trace_path
                    ),
                    initialization_terminal_dir=(
                        initialization_terminal_dir
                    ),
                )
            )

            for (
                use_u_damping,
                use_replay,
            ) in regimes:
                regime = regime_name(
                    use_replay,
                    use_u_damping,
                )

                print(
                    f"\n===== "
                    f"{regime.upper()} "
                    f"seed={seed} "
                    f"=====",
                    flush=True,
                )

                run_one(
                    problem_name=(
                        problem_name
                    ),
                    spec=spec,
                    initialized_nn=(
                        initialized_nn
                    ),
                    use_replay=(
                        use_replay
                    ),
                    use_u_damping=(
                        use_u_damping
                    ),
                    eta=args.eta,
                    seed=seed,
                    outer_iterations=(
                        args.outer
                    ),
                    inner_steps=(
                        args.inner
                    ),
                    train_samples=(
                        args.train_samples
                    ),
                    eval_samples=(
                        args.eval_samples
                    ),
                    batch_size=(
                        args.batch_size
                    ),
                    steps=args.steps,
                    target_reference_samples=(
                        args.target_reference_samples
                    ),
                    trace_path=(
                        trace_path
                    ),
                    summary_path=(
                        summary_path
                    ),
                    terminal_dir=(
                        terminal_dir
                    ),
                )

    print()
    print(
        f"initialization: "
        f"{initialization_trace_path}"
    )
    print(
        f"trace:          "
        f"{trace_path}"
    )
    print(
        f"summary:        "
        f"{summary_path}"
    )
    print(
        f"settings:       "
        f"{settings_path}"
    )
    print(
        f"init samples:   "
        f"{initialization_terminal_dir}"
    )
    print(
        f"terminal:       "
        f"{terminal_dir}"
    )
    print("DONE")


if __name__ == "__main__":
    main()
