#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gaussian Adjoint instability study: damping v2.

Target:
    N(1.0, 0.02)

Initial outer endpoint law:
    N(1.05, 0.02)

New regimes only:
    mean/translation damping, eta=0.5, replay off/on
    mean/translation damping, eta=eta_*, replay off/on
    direct u damping, eta=0.5, replay off/on
    direct u damping, eta=eta_*, replay off/on

The first learned outer update is left undamped for all regimes so that
all trajectories begin from the same learned field. Damping begins at
the second learned outer update.

Direct u damping is implemented at the RAM regression-target level:

    u_target,damped
        = eta * u_target
          + (1 - eta) * u_previous.

Only one frozen lagged network is retained.

For u damping, an implied undamped proposal field

    u_hat
        = [u_new - (1 - eta) u_previous] / eta

is rolled out diagnostically using the same Brownian random numbers as
the damped field. This gives

    eta_eff
        = (m_new - m_old) / (m_hat - m_old),

and the logged mismatch is eta_eff - eta.

Mean damping applies the desired scalar mean recurrence directly:

    m_new
        = (1 - eta) m_old + eta m_hat,

by translating all proposal endpoints by a deterministic constant.
"""

import argparse
import csv
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
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
    "damping_scheme",
    "eta",
    "applied_eta",
    "inner_steps",
    "loss",
    "clip_fraction",
    "proposal_mean",
    "proposal_kind",
    "eta_effective",
    "eta_mismatch",
    "translation_delta",
    "stop_reason",
]


SUMMARY_FIELDS = [
    "regime",
    "seed",
    "n",
    "mean",
    "variance",
    "completed_k",
    "stop_reason",
]


def append_csv(path, fields, row):
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


def stopping_reason(mean, variance, terminal):
    if not np.isfinite(mean):
        return "nonfinite_mean"

    if not np.isfinite(variance):
        return "nonfinite_variance"

    values = np.asarray(
        terminal
    ).reshape(-1)

    if not np.all(np.isfinite(values)):
        return "nonfinite_samples"

    if abs(mean - base.MU) > STOP_MEAN_ERROR:
        return f"mean_error>{STOP_MEAN_ERROR:g}"

    if variance > STOP_VARIANCE:
        return f"variance>{STOP_VARIANCE:g}"

    if (
        values.size
        and np.max(np.abs(values)) > STOP_SAMPLE_ABS
    ):
        return f"sample_abs>{STOP_SAMPLE_ABS:g}"

    return ""


def clone_nnmodel(model):
    """Independent frozen copy of the current control network."""
    lagged = nnx.clone(
        model.nnmodel
    )
    lagged.eval()
    return lagged


def evaluate_network(
    model,
    network,
    x,
    t,
):
    """Evaluate a supplied time network with the model's time formatting."""
    nn_time = (
        model._format_network_time(
            t,
            x.shape[0],
        )
    )

    return network(
        x,
        nn_time,
    )


def train_full_adjoint_u_damped(
    model,
    optimizer,
    metrics,
    replay,
    endpoints,
    inner_steps,
    batch_size,
    key,
    previous_nn,
    eta,
):
    """
    Full RAM outer training block with direct function-space damping.

    If previous_nn is None, the update is undamped.

    Otherwise the regression label is

        eta * raw_RAM_target
        + (1 - eta) * u_previous(x,t).
    """
    gradients = (
        model.eval_terminal_gradient(
            endpoints
        )
    )

    gradients, clip_fraction = (
        model._clip_terminal_gradients(
            gradients,
            base.TARGET_CLIP,
        )
    )

    replay.add(
        endpoints,
        gradients,
    )

    if previous_nn is not None:
        previous_nn.eval()

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

        labels = batch["label"]

        if previous_nn is not None:
            inputs = batch["input"]

            x = inputs[
                :, : model.d
            ]

            t = inputs[
                :, model.d :
            ]

            previous_control = (
                jax.lax.stop_gradient(
                    previous_nn(
                        x,
                        t,
                    )
                )
            )

            labels = (
                eta * labels
                + (1.0 - eta)
                * previous_control
            )

        model.nnmodel.train()

        loss = base.train_step(
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
                jnp.asarray(
                    losses
                )
            )
        ),
        float(clip_fraction),
        key,
    )


def generate_endpoints_with_control(
    model,
    nsamples,
    subkey,
    control_fn,
):
    """
    Euler--Maruyama terminal rollout using an arbitrary control callable.

    Uses exactly the same discretization as AdjointSampler.generate_paths.
    """
    key = subkey

    x = jnp.zeros(
        (nsamples, model.d),
        dtype=jnp.float32,
    )

    for k in range(model.T):
        t = k * model.h

        control = control_fn(
            x,
            t,
        )

        sigma = jnp.asarray(
            model.eval_sigma(t),
            dtype=x.dtype,
        )

        while (
            sigma.ndim
            < control.ndim
        ):
            sigma = sigma[..., None]

        drift = (
            sigma * control
        )

        key, noise_key = (
            jax.random.split(key)
        )

        noise = jax.random.normal(
            noise_key,
            x.shape,
            dtype=x.dtype,
        )

        x = (
            x
            + model.h * drift
            + sigma
            * jnp.sqrt(model.h)
            * noise
        )

    return x


def generate_implied_proposal_endpoints(
    model,
    previous_nn,
    eta,
    nsamples,
    subkey,
):
    """
    Diagnostic rollout of the implied undamped proposal field

        u_hat
            = [u_new - (1-eta) u_previous] / eta.

    The result is diagnostic only; no extra proposal network is stored.
    """
    model.nnmodel.eval()
    previous_nn.eval()

    def implied_control(x, t):
        current_control = (
            model.eval_control(
                x,
                t,
            )
        )

        previous_control = (
            evaluate_network(
                model,
                previous_nn,
                x,
                t,
            )
        )

        return (
            current_control
            - (1.0 - eta)
            * previous_control
        ) / eta

    return generate_endpoints_with_control(
        model=model,
        nsamples=nsamples,
        subkey=subkey,
        control_fn=implied_control,
    )


def make_replay(
    use_replay,
    persistent_replay,
    train_samples,
    batch_size,
):
    if use_replay:
        return persistent_replay

    return base._ReplayBuffer(
        dim=1,
        capacity=max(
            train_samples,
            batch_size,
        ),
    )


def run_one(
    regime,
    scheme,
    eta,
    use_replay,
    seed,
    outer_iterations,
    inner_steps,
    initial_inner_steps,
    train_samples,
    eval_samples,
    batch_size,
    steps,
    trace_path,
    summary_path,
    terminal_dir,
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

    if use_replay:
        persistent_replay = (
            base._ReplayBuffer(
                dim=1,
                capacity=base.REPLAY_CAPACITY,
            )
        )

    # Same random stream for every full-Adjoin damping regime
    # at a given seed.
    key = jax.random.PRNGKey(
        30000 + int(seed)
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

    mean = float(
        base.INITIAL_MEAN
    )

    variance = float(
        base.INITIAL_VARIANCE
    )

    append_csv(
        trace_path,
        TRACE_FIELDS,
        {
            "regime": regime,
            "seed": seed,
            "k": 0,
            "mean": mean,
            "variance": variance,
            "mean_error": (
                mean - base.MU
            ),
            "replay": (
                "on"
                if use_replay
                else "off"
            ),
            "damping_scheme": scheme,
            "eta": eta,
            "applied_eta": "",
            "inner_steps": 0,
            "loss": "",
            "clip_fraction": "",
            "proposal_mean": "",
            "proposal_kind": "",
            "eta_effective": "",
            "eta_mismatch": "",
            "translation_delta": "",
            "stop_reason": "",
        },
    )

    terminal = None
    completed_k = 0
    final_reason = ""

    for outer in range(
        outer_iterations
    ):
        previous_mean = mean

        current_inner = (
            initial_inner_steps
            if outer == 0
            else inner_steps
        )

        applied_eta = (
            1.0
            if outer == 0
            else eta
        )

        replay = make_replay(
            use_replay=use_replay,
            persistent_replay=persistent_replay,
            train_samples=train_samples,
            batch_size=batch_size,
        )

        previous_nn = None

        if (
            scheme == "u"
            and outer > 0
        ):
            previous_nn = (
                clone_nnmodel(model)
            )

        if scheme == "u":
            (
                loss,
                clip_fraction,
                key,
            ) = (
                train_full_adjoint_u_damped(
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

        elif scheme == "mean":
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
                f"Unknown scheme: {scheme}"
            )

        if not np.isfinite(loss):
            final_reason = (
                "nonfinite_loss"
            )

            append_csv(
                trace_path,
                TRACE_FIELDS,
                {
                    "regime": regime,
                    "seed": seed,
                    "k": outer + 1,
                    "mean": np.nan,
                    "variance": np.nan,
                    "mean_error": np.nan,
                    "replay": (
                        "on"
                        if use_replay
                        else "off"
                    ),
                    "damping_scheme": scheme,
                    "eta": eta,
                    "applied_eta": applied_eta,
                    "inner_steps": current_inner,
                    "loss": loss,
                    "clip_fraction": clip_fraction,
                    "proposal_mean": "",
                    "proposal_kind": "",
                    "eta_effective": "",
                    "eta_mismatch": "",
                    "translation_delta": "",
                    "stop_reason": final_reason,
                },
            )

            print(
                f"{regime:42s} "
                f"seed={seed} "
                f"k={outer + 1:3d} "
                f"STOP={final_reason}",
                flush=True,
            )

            break

        key, eval_key = (
            jax.random.split(key)
        )

        proposal_mean = np.nan
        proposal_kind = ""
        eta_effective = np.nan
        eta_mismatch = np.nan
        translation_delta = 0.0

        if scheme == "mean":
            proposal_terminal = (
                model.generate_endpoints(
                    eval_samples,
                    eval_key,
                )
            )

            (
                proposal_mean,
                proposal_variance,
            ) = (
                base.empirical_moments(
                    proposal_terminal
                )
            )

            proposal_mean = float(
                proposal_mean
            )

            proposal_kind = "raw"

            if outer == 0:
                terminal = (
                    proposal_terminal
                )

            else:
                desired_mean = (
                    (1.0 - eta)
                    * previous_mean
                    + eta
                    * proposal_mean
                )

                translation_delta = (
                    desired_mean
                    - proposal_mean
                )

                terminal = (
                    proposal_terminal
                    + translation_delta
                )

            mean, variance = (
                base.empirical_moments(
                    terminal
                )
            )

            mean = float(mean)
            variance = float(variance)

            if outer > 0:
                denominator = (
                    proposal_mean
                    - previous_mean
                )

                if (
                    abs(denominator)
                    > 1.0e-6
                ):
                    eta_effective = (
                        mean
                        - previous_mean
                    ) / denominator

                    eta_mismatch = (
                        eta_effective
                        - eta
                    )

        else:
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

            mean = float(mean)
            variance = float(variance)

            if outer > 0:
                proposal_terminal = (
                    generate_implied_proposal_endpoints(
                        model=model,
                        previous_nn=previous_nn,
                        eta=eta,
                        nsamples=eval_samples,
                        subkey=eval_key,
                    )
                )

                (
                    proposal_mean,
                    _,
                ) = (
                    base.empirical_moments(
                        proposal_terminal
                    )
                )

                proposal_mean = float(
                    proposal_mean
                )

                proposal_kind = (
                    "implied_undamped"
                )

                denominator = (
                    proposal_mean
                    - previous_mean
                )

                if (
                    np.isfinite(
                        proposal_mean
                    )
                    and abs(denominator)
                    > 1.0e-6
                ):
                    eta_effective = (
                        mean
                        - previous_mean
                    ) / denominator

                    eta_mismatch = (
                        eta_effective
                        - eta
                    )

        final_reason = (
            stopping_reason(
                mean,
                variance,
                terminal,
            )
        )

        completed_k = (
            outer + 1
        )

        append_csv(
            trace_path,
            TRACE_FIELDS,
            {
                "regime": regime,
                "seed": seed,
                "k": completed_k,
                "mean": mean,
                "variance": variance,
                "mean_error": (
                    mean - base.MU
                ),
                "replay": (
                    "on"
                    if use_replay
                    else "off"
                ),
                "damping_scheme": scheme,
                "eta": eta,
                "applied_eta": applied_eta,
                "inner_steps": current_inner,
                "loss": loss,
                "clip_fraction": clip_fraction,
                "proposal_mean": (
                    ""
                    if not np.isfinite(
                        proposal_mean
                    )
                    else proposal_mean
                ),
                "proposal_kind": proposal_kind,
                "eta_effective": (
                    ""
                    if not np.isfinite(
                        eta_effective
                    )
                    else eta_effective
                ),
                "eta_mismatch": (
                    ""
                    if not np.isfinite(
                        eta_mismatch
                    )
                    else eta_mismatch
                ),
                "translation_delta": (
                    translation_delta
                    if scheme == "mean"
                    else ""
                ),
                "stop_reason": final_reason,
            },
        )

        np.save(
            terminal_dir
            / f"{regime}_seed{seed}.npy",
            np.asarray(
                terminal
            ).reshape(-1),
        )

        extra = ""

        if (
            outer > 0
            and np.isfinite(
                eta_effective
            )
        ):
            extra = (
                f" eta_eff="
                f"{eta_effective:.6f}"
                f" d_eta="
                f"{eta_mismatch:+.3e}"
            )

        if scheme == "mean":
            extra += (
                f" delta="
                f"{translation_delta:+.6f}"
            )

        print(
            f"{regime:42s} "
            f"seed={seed} "
            f"k={completed_k:3d} "
            f"eta={applied_eta:.8f} "
            f"mean={mean: .8f} "
            f"var={variance: .8f} "
            f"loss={loss:.4e} "
            f"clip={clip_fraction:.3f}"
            f"{extra}"
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
                jax.random.split(key)
            )

            training_endpoints = (
                model.generate_endpoints(
                    train_samples,
                    train_key,
                )
            )

            if (
                scheme == "mean"
                and outer > 0
            ):
                training_endpoints = (
                    training_endpoints
                    + translation_delta
                )

    if terminal is not None:
        append_csv(
            summary_path,
            SUMMARY_FIELDS,
            {
                "regime": regime,
                "seed": seed,
                "n": int(
                    np.asarray(
                        terminal
                    ).size
                ),
                "mean": mean,
                "variance": variance,
                "completed_k": completed_k,
                "stop_reason": final_reason,
            },
        )


def regime_name(
    scheme,
    eta_label,
    use_replay,
):
    replay_name = (
        "replay"
        if use_replay
        else "no_replay"
    )

    return (
        f"full_{replay_name}_"
        f"{scheme}_damped_"
        f"{eta_label}"
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
        type=Path,
        default=Path(
            "studies/"
            "adjoint_gaussian_instability/"
            "results/"
            "long_run/"
            "damping_v2_50"
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
        and any(out.iterdir())
    ):
        if args.overwrite:
            shutil.rmtree(out)
        else:
            raise SystemExit(
                f"{out} already exists and is not empty. "
                "Use --overwrite to replace it."
            )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    terminal_dir = (
        out / "terminal_samples"
    )

    terminal_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trace_path = (
        out / "trace.csv"
    )

    summary_path = (
        out / "summary.csv"
    )

    eta_settings = [
        (
            "050",
            0.5,
        ),
        (
            "optimal",
            ETA_OPTIMAL,
        ),
    ]

    print("=" * 72)
    print("Gaussian Adjoint damping-v2 study")
    print("=" * 72)
    print(
        f"target:        "
        f"N({base.MU}, "
        f"{base.TARGET_VARIANCE})"
    )
    print(
        f"initial:       "
        f"N({base.INITIAL_MEAN}, "
        f"{base.INITIAL_VARIANCE})"
    )
    print(
        f"A:             "
        f"{A:.10f}"
    )
    print(
        f"eta optimal:   "
        f"{ETA_OPTIMAL:.10f}"
    )
    print(
        f"outer:         "
        f"{args.outer}"
    )
    print(
        f"first inner:   "
        f"{args.initial_inner}"
    )
    print(
        f"later inner:   "
        f"{args.inner}"
    )
    print(
        f"train samples: "
        f"{args.train_samples}"
    )
    print(
        f"eval samples:  "
        f"{args.eval_samples}"
    )
    print(
        f"steps:         "
        f"{args.steps}"
    )
    print(
        f"seeds:         "
        f"{args.seeds}"
    )
    print("=" * 72)

    for scheme in (
        "mean",
        "u",
    ):
        for (
            eta_label,
            eta,
        ) in eta_settings:
            for use_replay in (
                False,
                True,
            ):
                regime = (
                    regime_name(
                        scheme,
                        eta_label,
                        use_replay,
                    )
                )

                print(
                    "\n===== "
                    + regime.upper()
                    + " =====",
                    flush=True,
                )

                for seed in args.seeds:
                    run_one(
                        regime=regime,
                        scheme=scheme,
                        eta=eta,
                        use_replay=use_replay,
                        seed=seed,
                        outer_iterations=args.outer,
                        inner_steps=args.inner,
                        initial_inner_steps=args.initial_inner,
                        train_samples=args.train_samples,
                        eval_samples=args.eval_samples,
                        batch_size=args.batch_size,
                        steps=args.steps,
                        trace_path=trace_path,
                        summary_path=summary_path,
                        terminal_dir=terminal_dir,
                    )

    print()
    print(
        f"trace:   {trace_path}"
    )
    print(
        f"summary: {summary_path}"
    )
    print("DONE")


if __name__ == "__main__":
    main()
