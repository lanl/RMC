#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Long 1D Gaussian Adjoint instability experiment.

Problem:
    target      N(1.0, 0.02)
    initial     N(1.05, 0.02)
    sigma(t)    1

Regimes:
    affine oracle
    learned affine
    full Adjoint, no replay
    full Adjoint, replay
    full Adjoint, no replay, eta=0.5
    full Adjoint, replay, eta=0.5
    full Adjoint, no replay, eta=eta_*
    full Adjoint, replay, eta=eta_*

The first learned outer update can use a larger inner-training budget.
Subsequent updates are warm-started and use the standard inner budget.

Results are written incrementally so completed outer iterations survive
a wall-time termination.
"""

import argparse
import csv
from pathlib import Path

import jax
import numpy as np
from flax import nnx

import run_experiment as base


A = (
    1.0
    + np.log(base.TARGET_VARIANCE)
    / (1.0 - base.TARGET_VARIANCE)
)

ETA_OPTIMAL = 1.0 / (1.0 - A)

STOP_MEAN_ERROR = 100.0
STOP_VARIANCE = 1000.0
STOP_SAMPLE_ABS = 1000.0


TRACE_FIELDS = [
    "regime",
    "seed",
    "k",
    "mean",
    "variance",
    "mean_error",
    "replay",
    "eta",
    "inner_steps",
    "loss",
    "clip_fraction",
    "stop_reason",
]


def stopping_reason(
    mean,
    variance,
    terminal,
):
    if not np.isfinite(mean):
        return "nonfinite_mean"

    if not np.isfinite(variance):
        return "nonfinite_variance"

    values = np.asarray(
        terminal
    ).reshape(-1)

    if not np.all(
        np.isfinite(values)
    ):
        return "nonfinite_samples"

    if (
        abs(mean - base.MU)
        > STOP_MEAN_ERROR
    ):
        return (
            f"mean_error>{STOP_MEAN_ERROR:g}"
        )

    if variance > STOP_VARIANCE:
        return (
            f"variance>{STOP_VARIANCE:g}"
        )

    if (
        values.size
        and np.max(np.abs(values))
        > STOP_SAMPLE_ABS
    ):
        return (
            f"sample_abs>{STOP_SAMPLE_ABS:g}"
        )

    return ""


def append_trace(
    path,
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
            fieldnames=TRACE_FIELDS,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def trace_row(
    regime,
    seed,
    k,
    mean,
    variance,
    replay,
    eta,
    inner_steps,
    loss=np.nan,
    clip_fraction=np.nan,
    stop_reason="",
):
    return {
        "regime": regime,
        "seed": seed,
        "k": k,
        "mean": mean,
        "variance": variance,
        "mean_error": (
            mean - base.MU
        ),
        "replay": replay,
        "eta": (
            ""
            if eta is None
            else eta
        ),
        "inner_steps": inner_steps,
        "loss": loss,
        "clip_fraction": clip_fraction,
        "stop_reason": stop_reason,
    }


def run_oracle_long(
    outer_iterations,
    samples,
    steps,
    trace_path,
    terminal_dir,
):
    regime = "affine_oracle"

    config = base.build_config(
        seed=0,
        train_samples=128,
        batch_size=64,
        inner_steps=100,
    )

    model = base.AffineOracleSampler(
        config=config,
        steps=steps,
        outer_mean=base.INITIAL_MEAN,
        outer_variance=base.INITIAL_VARIANCE,
    )

    times = np.asarray(
        model._build_time_grid(),
        dtype=np.float64,
    )

    mean = base.INITIAL_MEAN
    variance = base.INITIAL_VARIANCE

    append_trace(
        trace_path,
        trace_row(
            regime=regime,
            seed=0,
            k=0,
            mean=mean,
            variance=variance,
            replay="",
            eta=None,
            inner_steps=0,
        ),
    )

    key = jax.random.PRNGKey(
        1000
    )

    terminal = None

    for outer in range(
        outer_iterations
    ):
        model.set_outer_state(
            mean,
            variance,
        )

        key, sample_key = (
            jax.random.split(key)
        )

        terminal = (
            model.generate_endpoints(
                samples,
                sample_key,
            )
        )

        mean, variance = (
            base.em_proposal_moments(
                mean,
                variance,
                times,
            )
        )

        reason = stopping_reason(
            mean,
            variance,
            terminal,
        )

        append_trace(
            trace_path,
            trace_row(
                regime=regime,
                seed=0,
                k=outer + 1,
                mean=mean,
                variance=variance,
                replay="",
                eta=None,
                inner_steps=0,
                stop_reason=reason,
            ),
        )

        np.save(
            terminal_dir
            / f"{regime}_seed0.npy",
            np.asarray(
                terminal
            ).reshape(-1),
        )

        print(
            f"{regime:34s} "
            f"k={outer + 1:3d} "
            f"mean={mean: .8f} "
            f"var={variance: .8f}"
            + (
                f" STOP={reason}"
                if reason
                else ""
            ),
            flush=True,
        )

        if reason:
            break

    return terminal


def run_learned_long(
    regime,
    mode,
    seed,
    outer_iterations,
    inner_steps,
    initial_inner_steps,
    train_samples,
    eval_samples,
    batch_size,
    steps,
    trace_path,
    terminal_dir,
    use_replay=False,
    damping_eta=None,
):
    model = base.build_model(
        seed=seed,
        train_samples=train_samples,
        batch_size=batch_size,
        inner_steps=inner_steps,
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

    persistent_replay = None

    if (
        mode == "full"
        and use_replay
    ):
        persistent_replay = (
            base._ReplayBuffer(
                dim=1,
                capacity=base.REPLAY_CAPACITY,
            )
        )

    # All full-Adjoin variants use the
    # same random stream for a clean
    # common-random-number comparison.
    key_offset = (
        20000
        if mode == "direct"
        else 30000
    )

    key = jax.random.PRNGKey(
        key_offset + int(seed)
    )

    key, p0_key = (
        jax.random.split(key)
    )

    training_endpoints = (
        base.sample_gaussian(
            p0_key,
            base.INITIAL_MEAN,
            base.INITIAL_VARIANCE,
            train_samples,
        )
    )

    mean = base.INITIAL_MEAN
    variance = (
        base.INITIAL_VARIANCE
    )

    append_trace(
        trace_path,
        trace_row(
            regime=regime,
            seed=seed,
            k=0,
            mean=mean,
            variance=variance,
            replay=(
                "on"
                if use_replay
                else "off"
            ),
            eta=damping_eta,
            inner_steps=0,
        ),
    )

    terminal = None

    for outer in range(
        outer_iterations
    ):
        current_inner = (
            initial_inner_steps
            if outer == 0
            else inner_steps
        )

        previous_params = None

        # p0 is externally prescribed.
        # The first update is therefore
        # left undamped.
        if (
            damping_eta is not None
            and outer > 0
        ):
            previous_params = (
                base.snapshot_params(
                    model
                )
            )

        if mode == "direct":
            loss, key = (
                base.train_direct(
                    model=model,
                    optimizer=optimizer,
                    metrics=metrics,
                    endpoints=training_endpoints,
                    mean=mean,
                    variance=variance,
                    inner_steps=current_inner,
                    batch_size=batch_size,
                    key=key,
                )
            )

            clip_fraction = 0.0

        elif mode == "full":
            if use_replay:
                replay = (
                    persistent_replay
                )
            else:
                # Fresh buffer each outer
                # iteration: same RAM code
                # path, but no memory of
                # previous endpoint sets.
                replay = (
                    base._ReplayBuffer(
                        dim=1,
                        capacity=max(
                            train_samples,
                            batch_size,
                        ),
                    )
                )

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

        else:
            raise ValueError(
                mode
            )

        if not np.isfinite(loss):
            reason = (
                "nonfinite_loss"
            )

            append_trace(
                trace_path,
                trace_row(
                    regime=regime,
                    seed=seed,
                    k=outer + 1,
                    mean=np.nan,
                    variance=np.nan,
                    replay=(
                        "on"
                        if use_replay
                        else "off"
                    ),
                    eta=damping_eta,
                    inner_steps=current_inner,
                    loss=loss,
                    clip_fraction=clip_fraction,
                    stop_reason=reason,
                ),
            )

            print(
                f"{regime:34s} "
                f"seed={seed} "
                f"k={outer + 1:3d} "
                f"STOP={reason}",
                flush=True,
            )

            break

        if previous_params is not None:
            base.relax_params(
                model,
                previous_params,
                damping_eta,
            )

        key, eval_key = (
            jax.random.split(key)
        )

        terminal = (
            model.generate_endpoints(
                eval_samples,
                eval_key,
            )
        )

        mean, variance = (
            base.empirical_moments(
                terminal
            )
        )

        reason = stopping_reason(
            mean,
            variance,
            terminal,
        )

        append_trace(
            trace_path,
            trace_row(
                regime=regime,
                seed=seed,
                k=outer + 1,
                mean=mean,
                variance=variance,
                replay=(
                    "on"
                    if use_replay
                    else "off"
                ),
                eta=damping_eta,
                inner_steps=current_inner,
                loss=loss,
                clip_fraction=clip_fraction,
                stop_reason=reason,
            ),
        )

        # Keep the latest terminal
        # population on disk after every
        # outer iteration.
        np.save(
            terminal_dir
            / f"{regime}_seed{seed}.npy",
            np.asarray(
                terminal
            ).reshape(-1),
        )

        applied_eta = (
            1.0
            if (
                damping_eta is None
                or outer == 0
            )
            else damping_eta
        )

        print(
            f"{regime:34s} "
            f"seed={seed} "
            f"k={outer + 1:3d} "
            f"eta={applied_eta:.8f} "
            f"mean={mean: .8f} "
            f"var={variance: .8f} "
            f"loss={loss:.4e} "
            f"clip={clip_fraction:.3f}"
            + (
                f" STOP={reason}"
                if reason
                else ""
            ),
            flush=True,
        )

        if reason:
            break

        if (
            outer + 1
            < outer_iterations
        ):
            key, train_key = (
                jax.random.split(key)
            )

            training_endpoints = (
                model.generate_endpoints(
                    train_samples,
                    train_key,
                )
            )

    return terminal


def write_summary(
    output,
    terminal_dir,
):
    summary_path = (
        output / "summary.csv"
    )

    fields = [
        "regime",
        "seed",
        "sample_count",
        "terminal_mean",
        "terminal_variance",
    ]

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for path in sorted(
            terminal_dir.glob(
                "*.npy"
            )
        ):
            name = path.stem

            regime, seed_text = (
                name.rsplit(
                    "_seed",
                    1,
                )
            )

            values = np.load(
                path
            ).reshape(-1)

            writer.writerow(
                {
                    "regime": regime,
                    "seed": int(
                        seed_text
                    ),
                    "sample_count": (
                        len(values)
                    ),
                    "terminal_mean": (
                        float(
                            values.mean()
                        )
                    ),
                    "terminal_variance": (
                        float(
                            values.var()
                        )
                    ),
                }
            )

    return summary_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outer",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--inner",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--initial-inner",
        type=int,
        default=5000,
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
        "--output",
        required=True,
    )

    args = parser.parse_args()

    output = Path(
        args.output
    ).expanduser()

    terminal_dir = (
        output
        / "terminal_samples"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    terminal_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trace_path = (
        output / "trace.csv"
    )

    if trace_path.exists():
        raise RuntimeError(
            f"{trace_path} already exists; "
            "use a fresh output directory."
        )

    print(
        "============================================================"
    )
    print(
        "Long Gaussian Adjoint instability study"
    )
    print(
        "============================================================"
    )
    print(
        f"target:          "
        f"N({base.MU}, {base.TARGET_VARIANCE})"
    )
    print(
        f"initial:         "
        f"N({base.INITIAL_MEAN}, "
        f"{base.INITIAL_VARIANCE})"
    )
    print(
        f"A:               {A:.10f}"
    )
    print(
        f"eta boundary:    0.5"
    )
    print(
        f"eta optimal:     "
        f"{ETA_OPTIMAL:.10f}"
    )
    print(
        f"outer max:       {args.outer}"
    )
    print(
        f"first inner:     "
        f"{args.initial_inner}"
    )
    print(
        f"later inner:     {args.inner}"
    )
    print(
        f"train samples:   "
        f"{args.train_samples}"
    )
    print(
        f"eval samples:    "
        f"{args.eval_samples}"
    )
    print(
        f"seeds:           {args.seeds}"
    )
    print(
        f"stop |m-mu|:     "
        f"{STOP_MEAN_ERROR}"
    )
    print(
        f"stop variance:   "
        f"{STOP_VARIANCE}"
    )
    print(
        f"stop |sample|:   "
        f"{STOP_SAMPLE_ABS}"
    )
    print(
        "============================================================",
        flush=True,
    )

    print(
        "\n===== AFFINE ORACLE =====",
        flush=True,
    )

    run_oracle_long(
        outer_iterations=args.outer,
        samples=args.oracle_samples,
        steps=args.steps,
        trace_path=trace_path,
        terminal_dir=terminal_dir,
    )

    regimes = [
        (
            "learned_affine",
            "direct",
            False,
            None,
        ),
        (
            "full_no_replay",
            "full",
            False,
            None,
        ),
        (
            "full_replay",
            "full",
            True,
            None,
        ),
        (
            "full_no_replay_damped_050",
            "full",
            False,
            0.5,
        ),
        (
            "full_replay_damped_050",
            "full",
            True,
            0.5,
        ),
        (
            "full_no_replay_damped_optimal",
            "full",
            False,
            ETA_OPTIMAL,
        ),
        (
            "full_replay_damped_optimal",
            "full",
            True,
            ETA_OPTIMAL,
        ),
    ]

    for (
        regime,
        mode,
        use_replay,
        eta,
    ) in regimes:
        print(
            f"\n===== {regime.upper()} =====",
            flush=True,
        )

        for seed in args.seeds:
            run_learned_long(
                regime=regime,
                mode=mode,
                seed=seed,
                outer_iterations=args.outer,
                inner_steps=args.inner,
                initial_inner_steps=args.initial_inner,
                train_samples=args.train_samples,
                eval_samples=args.eval_samples,
                batch_size=args.batch_size,
                steps=args.steps,
                trace_path=trace_path,
                terminal_dir=terminal_dir,
                use_replay=use_replay,
                damping_eta=eta,
            )

    summary_path = write_summary(
        output,
        terminal_dir,
    )

    print("")
    print(
        f"trace:   {trace_path}"
    )
    print(
        f"summary: {summary_path}"
    )
    print(
        "DONE",
        flush=True,
    )


if __name__ == "__main__":
    main()
