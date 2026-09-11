#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Single-step coverage diagnostic for the 1D Gaussian Adjoint experiment.

Compare exact-affine velocity regression on:

  1. RAM Brownian-bridge states;
  2. states from the exact affine-oracle controlled trajectory.

The target velocity is identical in both cases.  Only the x,t training
distribution changes.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from rmc.flax.trainer import train_step

from norm1D_Adjoint_instability import (
    AffineOracleSampler,
    GaussianTarget,
    INITIAL_MEAN,
    INITIAL_VARIANCE,
    MU,
    TARGET_VARIANCE,
    oracle_coefficients,
)

from norm1D_Adjoint_instability_learned import (
    build_model,
    empirical_moments,
    exact_affine_labels,
    sample_gaussian,
)


def relative_velocity_mse(model, x, t):
    inp = jnp.concatenate(
        [x, t[:, None]],
        axis=-1,
    )

    target = exact_affine_labels(
        inp,
        INITIAL_MEAN,
        INITIAL_VARIANCE,
    )

    model.nnmodel.eval()
    prediction = model.nnmodel(
        x,
        t[:, None],
    )

    mse = jnp.mean(
        (prediction - target) ** 2
    )

    energy = jnp.mean(
        target**2
    )

    return float(
        mse / jnp.maximum(energy, 1.0e-12)
    )


def bridge_validation(model, key, nsamples):
    endpoint_key, batch_key = jax.random.split(key)

    endpoints = sample_gaussian(
        endpoint_key,
        INITIAL_MEAN,
        INITIAL_VARIANCE,
        nsamples,
    )

    dummy = jnp.zeros_like(endpoints)

    batch = model.build_ram_batch(
        batch_key,
        endpoints,
        dummy,
        batch_size=None,
    )

    return (
        batch["input"][:, :1],
        batch["input"][:, 1],
    )


def oracle_path(model, key, nsamples):
    target = GaussianTarget(
        MU,
        TARGET_VARIANCE,
    )

    oracle = AffineOracleSampler(
        config=model.config,
        densitycl=target,
        h=model.h,
        T=model.T,
        outer_mean=INITIAL_MEAN,
        outer_variance=INITIAL_VARIANCE,
    )

    paths = jnp.stack(
        oracle.generate_paths(
            nsamples,
            key,
        ),
        axis=0,
    )

    times = oracle._build_time_grid()

    return paths, times


def train_bridge(
    model,
    optimizer,
    metrics,
    key,
    updates,
    train_samples,
    batch_size,
):
    endpoints_key, key = jax.random.split(key)

    endpoints = sample_gaussian(
        endpoints_key,
        INITIAL_MEAN,
        INITIAL_VARIANCE,
        train_samples,
    )

    dummy = jnp.zeros_like(endpoints)

    for step in range(updates):
        key, batch_key = jax.random.split(key)

        batch = model.build_ram_batch(
            batch_key,
            endpoints,
            dummy,
            batch_size=batch_size,
        )

        labels = exact_affine_labels(
            batch["input"],
            INITIAL_MEAN,
            INITIAL_VARIANCE,
        )

        model.nnmodel.train()

        train_step(
            model.nnmodel,
            model.compute_ram_loss,
            optimizer,
            metrics,
            batch["input"],
            labels,
            False,
        )

        metrics.reset()

    return key


def train_oracle_support(
    model,
    optimizer,
    metrics,
    key,
    paths,
    times,
    updates,
    batch_size,
):
    ntime = paths.shape[0] - 1
    nparticles = paths.shape[1]

    for step in range(updates):
        key, time_key, particle_key = jax.random.split(
            key,
            3,
        )

        time_indices = jax.random.randint(
            time_key,
            (batch_size,),
            minval=0,
            maxval=ntime,
        )

        particle_indices = jax.random.randint(
            particle_key,
            (batch_size,),
            minval=0,
            maxval=nparticles,
        )

        x = paths[
            time_indices,
            particle_indices,
            :,
        ]

        t = times[
            time_indices
        ]

        inp = jnp.concatenate(
            [
                x,
                t[:, None],
            ],
            axis=-1,
        )

        labels = exact_affine_labels(
            inp,
            INITIAL_MEAN,
            INITIAL_VARIANCE,
        )

        model.nnmodel.train()

        train_step(
            model.nnmodel,
            model.compute_ram_loss,
            optimizer,
            metrics,
            inp,
            labels,
            False,
        )

        metrics.reset()

    return key


def rollout_diagnostics(model, key, nsamples):
    endpoints = model.generate_endpoints(
        nsamples,
        key,
    )

    return empirical_moments(endpoints)


def path_relative_mse(model, paths, times, stride=5):
    selected_paths = paths[:-1:stride]
    selected_times = times[:-1:stride]

    ntime = selected_paths.shape[0]
    nparticles = selected_paths.shape[1]

    x = selected_paths.reshape(
        ntime * nparticles,
        1,
    )

    t = jnp.repeat(
        selected_times,
        nparticles,
    )

    return relative_velocity_mse(
        model,
        x,
        t,
    )



def print_local_slope_diagnostics(
    model,
    paths,
    times,
    name,
):
    """Compare learned and exact drift slopes along the oracle path."""
    print("")
    print(name)
    print(
        " t       oracle mean    exact alpha"
        "    learned du/dx    drift error at mean"
    )

    for fraction in [
        0.0,
        0.25,
        0.50,
        0.75,
        0.90,
        0.99,
    ]:
        index = int(
            round(
                fraction * (len(times) - 1)
            )
        )

        t = times[index]

        x_mean = jnp.mean(
            paths[index],
            axis=0,
        )

        alpha, beta = oracle_coefficients(
            t,
            INITIAL_MEAN,
            INITIAL_VARIANCE,
        )

        def scalar_control(x_scalar):
            x = x_scalar.reshape(
                1,
                1,
            )

            tt = jnp.asarray(
                [[t]],
                dtype=x.dtype,
            )

            return model.nnmodel(
                x,
                tt,
            )[0, 0]

        learned_slope = jax.grad(
            scalar_control
        )(
            x_mean[0]
        )

        exact_value = (
            alpha * x_mean[0]
            + beta
        )

        learned_value = scalar_control(
            x_mean[0]
        )

        print(
            f"{float(t):.2f}  "
            f"{float(x_mean[0]): .7f}  "
            f"{float(alpha): .7f}  "
            f"{float(learned_slope): .7f}  "
            f"{float(learned_value - exact_value): .7f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--updates",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--support-particles",
        type=int,
        default=2048,
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
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    root_key = jax.random.PRNGKey(
        50000 + args.seed
    )

    # Two identically initialized networks.
    bridge_model = build_model(
        seed=args.seed,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        inner_steps=args.updates,
        steps=args.steps,
    )

    oracle_model = build_model(
        seed=args.seed,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        inner_steps=args.updates,
        steps=args.steps,
    )

    bridge_optimizer = bridge_model._build_optimizer()
    oracle_optimizer = oracle_model._build_optimizer()

    bridge_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    oracle_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    (
        root_key,
        oracle_path_key,
        bridge_train_key,
        oracle_train_key,
        bridge_val_key,
        bridge_rollout_key,
        oracle_rollout_key,
    ) = jax.random.split(
        root_key,
        7,
    )

    paths, times = oracle_path(
        bridge_model,
        oracle_path_key,
        args.support_particles,
    )

    bridge_x, bridge_t = bridge_validation(
        bridge_model,
        bridge_val_key,
        4096,
    )

    bridge_train_key = train_bridge(
        bridge_model,
        bridge_optimizer,
        bridge_metrics,
        bridge_train_key,
        args.updates,
        args.train_samples,
        args.batch_size,
    )

    oracle_train_key = train_oracle_support(
        oracle_model,
        oracle_optimizer,
        oracle_metrics,
        oracle_train_key,
        paths,
        times,
        args.updates,
        args.batch_size,
    )

    print("")
    print("============================================================")
    print("VELOCITY ERRORS")
    print("============================================================")

    for name, model in [
        ("bridge-trained", bridge_model),
        ("oracle-support-trained", oracle_model),
    ]:
        bridge_error = relative_velocity_mse(
            model,
            bridge_x,
            bridge_t,
        )

        oracle_error = path_relative_mse(
            model,
            paths,
            times,
        )

        print("")
        print(name)
        print(
            f"  bridge relMSE:      "
            f"{bridge_error:.6e}"
        )
        print(
            f"  oracle-path relMSE: "
            f"{oracle_error:.6e}"
        )


    print("")
    print("============================================================")
    print("LOCAL SLOPE DIAGNOSTICS")
    print("============================================================")

    print_local_slope_diagnostics(
        bridge_model,
        paths,
        times,
        "bridge-trained",
    )

    print_local_slope_diagnostics(
        oracle_model,
        paths,
        times,
        "oracle-support-trained",
    )

    bridge_mean, bridge_variance = rollout_diagnostics(
        bridge_model,
        bridge_rollout_key,
        args.eval_samples,
    )

    oracle_mean, oracle_variance = rollout_diagnostics(
        oracle_model,
        oracle_rollout_key,
        args.eval_samples,
    )

    print("")
    print("============================================================")
    print("ROLLOUT TERMINAL MOMENTS")
    print("============================================================")
    print(
        "Exact 500-step oracle:"
        " mean ~12.5238, variance ~0.100996"
    )

    print("")
    print("bridge-trained:")
    print(f"  mean:     {bridge_mean:.9f}")
    print(f"  variance: {bridge_variance:.9f}")

    print("")
    print("oracle-support-trained:")
    print(f"  mean:     {oracle_mean:.9f}")
    print(f"  variance: {oracle_variance:.9f}")

    print("")
    print("============================================================")
    print("STATE COVERAGE")
    print("============================================================")

    for fraction in [0.25, 0.50, 0.75, 1.00]:
        index = int(
            round(
                fraction * args.steps
            )
        )

        x = np.asarray(
            paths[index]
        ).reshape(-1)

        bridge_mean_at_t = (
            fraction * INITIAL_MEAN
        )

        bridge_variance_at_t = (
            fraction * (1.0 - fraction)
            + fraction**2 * INITIAL_VARIANCE
        )

        print(
            f"t={fraction:.2f}: "
            f"bridge mean={bridge_mean_at_t:.4f}, "
            f"bridge std={np.sqrt(bridge_variance_at_t):.4f}; "
            f"oracle mean={np.mean(x):.4f}, "
            f"oracle std={np.std(x):.4f}"
        )


if __name__ == "__main__":
    main()
