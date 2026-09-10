#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of Adjoint Sampling for a 2D Gaussian mixture
======================================================

This script applies Adjoint Sampling to the standard nine-mode Gaussian
mixture used by the RMC examples.

Two training configurations are exposed:

    --mode naive
        Paper-style reciprocal adjoint matching.

    --mode repo
        Enables the practical repository-style features implemented by
        AdjointSampler.

The --smoke flag uses a small training configuration intended only to
verify the complete train/sample pipeline.
"""

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp

import numpy as np
from flax import nnx

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    ),
)

from utils.density_examples import NormMix2D

from rmc import (
    plot_quiver,
    plot_samples,
    plot_trajectories,
    save_plot,
)
from rmc.flax.nn_config_dict import NNConfigDict
from rmc.flax.trainer import load_model
from rmc.modules.adjoint import AdjointSampler


def compute_diagnostics(samples, means):
    """Compute simple diagnostics for the nine-mode mixture."""
    samples = np.asarray(samples)
    means = np.asarray(means)

    mean = np.mean(
        samples,
        axis=0,
    )

    covariance = np.cov(
        samples,
        rowvar=False,
        ddof=0,
    )

    squared_distances = np.sum(
        (samples[:, None, :] - means[None, :, :]) ** 2,
        axis=-1,
    )

    assignments = np.argmin(
        squared_distances,
        axis=1,
    )

    nearest_squared_distance = np.min(
        squared_distances,
        axis=1,
    )

    mode_masses = np.bincount(
        assignments,
        minlength=means.shape[0],
    ).astype(float)

    mode_masses /= samples.shape[0]

    mode_radius = 3.0 * np.sqrt(sigma2)
    near_mode_fraction = np.mean(nearest_squared_distance <= mode_radius**2)

    return {
        "mean": mean,
        "covariance": covariance,
        "mode_masses": mode_masses,
        "mean_nearest_mode_squared_distance": np.mean(nearest_squared_distance),
        "near_mode_fraction": near_mode_fraction,
    }


parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=("naive", "repo"),
    default="naive",
    help="Adjoint Sampling training configuration.",
)

parser.add_argument(
    "--smoke",
    action="store_true",
    help="Run a small end-to-end development test.",
)

parser.add_argument(
    "--outer",
    type=int,
    default=None,
    help="Number of outer Adjoint Sampling iterations.",
)

parser.add_argument(
    "--inner",
    type=int,
    default=None,
    help="Number of RAM optimizer steps per outer iteration.",
)

parser.add_argument(
    "--samples-per-outer",
    type=int,
    default=None,
    help="Number of controlled endpoints generated per outer iteration.",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=None,
    help="RAM regression batch size.",
)

parser.add_argument(
    "--nsamples",
    type=int,
    default=None,
    help="Number of final samples used for diagnostics.",
)

parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="Number of Euler--Maruyama integration steps.",
)

parser.add_argument(
    "--resume-model",
    type=str,
    default=None,
    help="Path to an nnx-state-adjoint.pkl model checkpoint.",
)

args = parser.parse_args()

print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

repo_features = args.mode == "repo"


# -------------------------------------------------------------------------
# Target distribution
# -------------------------------------------------------------------------

d = 2
grid = 1.0

means = jnp.array(
    [[grid * i, grid * j] for i in range(-1, 2) for j in range(-1, 2)],
    dtype=jnp.float32,
)

sigma2 = 0.012
weights = jnp.ones(means.shape[0])

Dcl = NormMix2D(
    means,
    sigma2,
    weights,
)


# -------------------------------------------------------------------------
# Training configuration
# -------------------------------------------------------------------------

if args.smoke:
    default_outer_iterations = 2
    default_outer_samples = 256
    default_inner_steps = 4
    default_batch_size = 128
    default_nsamples = 1000
else:
    if repo_features:
        # Match the practical scale of the released Cartesian experiments.
        # The finite replay buffer is useful only when each outer iteration
        # contributes substantially fewer samples than the buffer capacity.
        default_outer_iterations = 50
        default_outer_samples = 128
        default_inner_steps = None
        default_batch_size = 64
        default_nsamples = 10000
    else:
        default_outer_iterations = 5
        default_outer_samples = 4096
        default_inner_steps = None
        default_batch_size = 512
        default_nsamples = 10000

outer_iterations = args.outer if args.outer is not None else default_outer_iterations

outer_samples = (
    args.samples_per_outer if args.samples_per_outer is not None else default_outer_samples
)

inner_steps = args.inner if args.inner is not None else default_inner_steps

batch_size = args.batch_size if args.batch_size is not None else default_batch_size

nsamples = args.nsamples if args.nsamples is not None else default_nsamples


suffix = f"{args.mode}"
if args.smoke:
    suffix += "_smoke"

root_path = f"./results_adjoint_mix2D_{suffix}/"
os.makedirs(
    root_path,
    exist_ok=True,
)

nn_conf: NNConfigDict = {
    "seed": 0,
    "batch_size": batch_size,
    "dim": d,
    "layer_widths": [64, 64, 64],
    "activation_func": nnx.silu,
    "nn_type": "time_embed",
    "opt_type": "ADAM",
    "base_lr": 1.0e-4 if repo_features else 1.0e-3,
    "opt_grad_max_norm": 10.0,
    "max_samples": outer_samples,
    "nsamples": nsamples,
    "max_subiter": outer_iterations,
    "eval_every": 1,
    "has_aux": False,
    "root_path": root_path,
    "adjoint_repo_features": repo_features,
    "adjoint_outer_iterations": outer_iterations,
    "adjoint_outer_samples": outer_samples,
    "adjoint_batch_size": batch_size,
}

# During a smoke test, keep both modes computationally small.
# Otherwise each mode uses its own resolved inner-loop default.
if inner_steps is not None:
    nn_conf["adjoint_inner_steps"] = inner_steps


print("===================================================")
print("Adjoint Sampling 2D Gaussian-mixture example")
print("===================================================")
print(f"mode:          {args.mode}")
print(f"smoke:         {args.smoke}")
print(f"outer:         {outer_iterations}")
print(f"inner:         {inner_steps}")
print(f"samples/outer: {outer_samples}")
print(f"batch size:    {batch_size}")
print(f"base lr:       {nn_conf['base_lr']}")
print(f"final samples: {nsamples}")
print(f"integration:   {args.steps if args.steps is not None else 'mode default'}")
print(f"configuration: {nn_conf}")


# -------------------------------------------------------------------------
# Adjoint Sampler
# -------------------------------------------------------------------------

# Physical terminal time is one.
# The released Adjoint Sampling Cartesian configuration uses 500
# integration evaluations.  This is important for resolving the
# rapidly varying geometric diffusion schedule.
default_steps = 500 if repo_features else 50
T = args.steps if args.steps is not None else default_steps
h = 1.0 / T

model = AdjointSampler(
    config=nn_conf,
    densitycl=Dcl,
    h=h,
    T=T,
    verbose=True,
)

if args.resume_model is not None:
    resume_path = os.path.abspath(args.resume_model)

    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Adjoint model checkpoint not found: {resume_path}")

    resume_dir = os.path.dirname(resume_path)
    resume_name = os.path.basename(resume_path)

    if resume_name.endswith(".pkl"):
        resume_name = resume_name[:-4]

    load_model(
        model.nnmodel,
        resume_dir,
        resume_name,
    )

    print("")
    print(f"Loaded model checkpoint: {resume_path}")

print("")
print(
    "Reference terminal variance:",
    float(model.base_terminal_variance),
)

time_grid = model._build_time_grid()
dt_grid = jnp.diff(time_grid)

discrete_reference_variance = jnp.sum(model.eval_sigma(time_grid[:-1]) ** 2 * dt_grid)

print(
    "Euler reference terminal variance:",
    float(discrete_reference_variance),
)


# -------------------------------------------------------------------------
# Pre-training sampler diagnostics
# -------------------------------------------------------------------------

pretrain_key = jax.random.PRNGKey(nn_conf["seed"] + 100)

pretrain_path = model.sample(
    nn_conf["nsamples"],
    pretrain_key,
)

pretrain_terminal = jnp.asarray(pretrain_path)[-1]

pretrain_diagnostics = compute_diagnostics(
    pretrain_terminal,
    means,
)

print("")
print("===================================================")
print("PRE-TRAINING SAMPLE DIAGNOSTICS")
print("===================================================")
print("mean:")
print(pretrain_diagnostics["mean"])

print("")
print("covariance:")
print(pretrain_diagnostics["covariance"])

print("")
print("nearest-mode masses:")
print(
    pretrain_diagnostics["mode_masses"].reshape(
        3,
        3,
    )
)


# -------------------------------------------------------------------------
# Train
# -------------------------------------------------------------------------

history = model.train()

print("")
print("Training complete")

if history:
    print("Final training record:")
    print(history[-1])


# -------------------------------------------------------------------------
# Sample
# -------------------------------------------------------------------------

key = jax.random.PRNGKey(nn_conf["seed"] + 1)

particle_path = model.sample(
    nn_conf["nsamples"],
    key,
)

particles = jnp.asarray(particle_path)

terminal = particles[-1]

print("")
print(f"particles shape: {particles.shape}")


# -------------------------------------------------------------------------
# Diagnostics
# -------------------------------------------------------------------------

diagnostics = compute_diagnostics(
    terminal,
    means,
)

target_variance = 2.0 / 3.0 + sigma2

print("")
print("===================================================")
print("TERMINAL SAMPLE DIAGNOSTICS")
print("===================================================")

print("mean:")
print(diagnostics["mean"])

print("")
print("covariance:")
print(diagnostics["covariance"])

print("")
print(
    "target covariance diagonal:",
    target_variance,
)

print("")
print("nearest-mode masses:")
print(
    diagnostics["mode_masses"].reshape(
        3,
        3,
    )
)

print("")
print(
    "ideal mode mass:",
    1.0 / 9.0,
)

print("")
print(
    "mean squared distance to nearest mode:",
    diagnostics["mean_nearest_mode_squared_distance"],
)
print(
    "target scale for this diagnostic:",
    2.0 * sigma2,
)

print("")
print(
    "fraction within 3 component std of a mode:",
    diagnostics["near_mode_fraction"],
)


# -------------------------------------------------------------------------
# Save numerical results
# -------------------------------------------------------------------------

history_path = os.path.join(
    root_path,
    "history.json",
)

with open(
    history_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        history,
        file,
        indent=2,
    )

np.savez(
    os.path.join(
        root_path,
        "samples.npz",
    ),
    terminal=np.asarray(terminal),
    means=np.asarray(means),
    mean=diagnostics["mean"],
    covariance=diagnostics["covariance"],
    mode_masses=diagnostics["mode_masses"],
    mean_nearest_mode_squared_distance=diagnostics["mean_nearest_mode_squared_distance"],
    near_mode_fraction=diagnostics["near_mode_fraction"],
)


# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------

from matplotlib import pyplot as plt

plt.rcParams.update(
    {
        "font.size": 14,
    }
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 10),
)

ax1, ax2, ax3, ax4 = axes.ravel()

ax1 = plot_samples(
    particles[0],
    ax1,
    size=5,
    label="Initial samples",
)
ax1.set_title("Initial distribution")

ax2 = plot_samples(
    terminal,
    ax2,
    size=5,
    label="Adjoint samples",
)
ax2.set_title(f"Terminal samples ({args.mode})")

ntrajectories = min(
    20,
    particles.shape[1],
)

for i in range(ntrajectories):
    ax3 = plot_trajectories(
        particles[:, i, :],
        ax3,
        label="trajectory" if i == 0 else None,
    )

ax3.set_title("Sample trajectories")

control = model.eval_control(
    terminal,
    model.TT,
)

ax4 = plot_quiver(
    terminal,
    control,
    ax4,
)
ax4.set_title(r"Learned control $u_\theta(x,T)$")

fig.tight_layout()

save_plot(
    fig,
    os.path.join(
        root_path,
        "sampleAdjoint_Mix2D.png",
    ),
)

plt.close(fig)

print("")
print(f"Results written to {root_path}")
