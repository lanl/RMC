#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MU = 1.0
TARGET_VARIANCE = 0.02
INITIAL_MEAN = 1.05

REGIMES = {
    "affine_oracle": "Affine oracle",
    "learned_affine": "Learned affine",
    "full_adjoint": "Full Adjoint",
    "full_adjoint_damped": "Full Adjoint + damping",
}


def analytic_mean_trace(nouter):
    A = 1.0 + np.log(TARGET_VARIANCE) / (1.0 - TARGET_VARIANCE)

    e = INITIAL_MEAN - MU
    values = [INITIAL_MEAN]

    for _ in range(nouter):
        e = A * e
        values.append(MU + e)

    return np.asarray(values)


def load_traces():
    traces = defaultdict(lambda: defaultdict(list))

    with open(RESULTS / "trace.csv", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            regime = row["regime"]
            seed = int(row["seed"])

            traces[regime][seed].append(
                (
                    int(row["k"]),
                    float(row["mean"]),
                    float(row["variance"]),
                )
            )

    return traces


def load_terminal_samples():
    archive = np.load(RESULTS / "terminal_samples.npz")

    samples = {}

    for regime in REGIMES:
        if regime == "affine_oracle":
            samples[regime] = np.asarray(
                archive["affine_oracle"]
            ).reshape(-1)
            continue

        keys = sorted(
            key
            for key in archive.files
            if key.startswith(regime + "_seed")
        )

        samples[regime] = np.concatenate(
            [
                np.asarray(archive[key]).reshape(-1)
                for key in keys
            ]
        )

    return samples


def gaussian_density(x):
    return (
        np.exp(
            -0.5
            * (x - MU) ** 2
            / TARGET_VARIANCE
        )
        / np.sqrt(
            2.0 * np.pi * TARGET_VARIANCE
        )
    )


def plot_mean_trace(regime, seed_traces):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    nouter = max(
        k
        for trace in seed_traces.values()
        for k, _, _ in trace
    )

    theory = analytic_mean_trace(nouter)

    ax.plot(
        np.arange(len(theory)),
        theory,
        linestyle="--",
        marker="o",
        label="exact Gaussian recurrence",
    )

    if len(seed_traces) == 1:
        trace = next(iter(seed_traces.values()))
        k = np.asarray([row[0] for row in trace])
        means = np.asarray([row[1] for row in trace])

        ax.plot(
            k,
            means,
            marker="o",
            label="numerical result",
        )

    else:
        rows = []

        for seed, trace in sorted(seed_traces.items()):
            k = np.asarray([row[0] for row in trace])
            means = np.asarray([row[1] for row in trace])

            rows.append(means)

            ax.plot(
                k,
                means,
                marker="o",
                alpha=0.30,
            )

        rows = np.asarray(rows)

        ax.plot(
            k,
            rows.mean(axis=0),
            marker="o",
            linewidth=2.0,
            label="3-seed mean",
        )

    ax.axhline(
        MU,
        linewidth=1.0,
        label="target mean",
    )

    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Terminal mean")
    ax.set_title(REGIMES[regime])
    ax.grid(True, alpha=0.20)
    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        FIGURES / f"{regime}_mean.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_histogram(regime, samples):
    target_std = np.sqrt(TARGET_VARIANCE)

    lo = min(
        np.quantile(samples, 0.002),
        MU - 4.0 * target_std,
    )

    hi = max(
        np.quantile(samples, 0.998),
        MU + 4.0 * target_std,
    )

    x = np.linspace(lo, hi, 1200)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))

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

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title(REGIMES[regime])
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        FIGURES / f"{regime}_hist.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    traces = load_traces()
    samples = load_terminal_samples()

    for regime in REGIMES:
        plot_mean_trace(
            regime,
            traces[regime],
        )

        plot_histogram(
            regime,
            samples[regime],
        )

        print(
            f"{regime:24s} "
            f"mean={samples[regime].mean(): .8f} "
            f"variance={samples[regime].var(): .8f}"
        )


if __name__ == "__main__":
    main()
