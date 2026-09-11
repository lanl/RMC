#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibrate the inner training budget for the learned-affine Gaussian test.

Each experiment starts from scratch and performs exactly one outer update.
The learned rollout is compared with the discrete affine-oracle rollout.

Shared problem:
    target      N(1.0, 0.02)
    initial p0  N(1.05, 0.02)
    sigma(t)    1

Sweep:
    inner steps = 100, 500, 1000, 2000
    train samples = 512
    eval samples = 10000
    seeds = 0, 1, 2
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import run_experiment as exp


INNER_STEPS = [100, 500, 1000, 2000]
SEEDS = [0, 1, 2]

TRAIN_SAMPLES = 512
EVAL_SAMPLES = 10000
ORACLE_SAMPLES = 50000
BATCH_SIZE = 64
STEPS = 500


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "inner_calibration"
FIGURES = ROOT / "figures"


def main():
    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("============================================================")
    print("Learned-affine inner-step calibration")
    print("============================================================")
    print(f"target:           N({exp.MU}, {exp.TARGET_VARIANCE})")
    print(
        f"initial:          "
        f"N({exp.INITIAL_MEAN}, {exp.INITIAL_VARIANCE})"
    )
    print(f"inner steps:      {INNER_STEPS}")
    print(f"train samples:    {TRAIN_SAMPLES}")
    print(f"eval samples:     {EVAL_SAMPLES}")
    print(f"seeds:            {SEEDS}")
    print(f"EM steps:         {STEPS}")
    print("")

    # Same discrete oracle used by the report.
    oracle_trace, oracle_samples = exp.run_oracle(
        outer_iterations=1,
        samples=ORACLE_SAMPLES,
        steps=STEPS,
    )

    oracle_mean = oracle_trace[1]["mean"]
    oracle_variance = oracle_trace[1]["variance"]

    print("")
    print(
        f"DISCRETE ORACLE: mean={oracle_mean:.10f} "
        f"variance={oracle_variance:.10f}"
    )
    print("")

    rows = []

    for inner in INNER_STEPS:
        print(
            "============================================================"
        )
        print(f"INNER STEPS = {inner}")
        print(
            "============================================================"
        )

        for seed in SEEDS:
            trace, terminal = exp.run_learned(
                regime="direct",
                seed=seed,
                outer_iterations=1,
                inner_steps=inner,
                train_samples=TRAIN_SAMPLES,
                eval_samples=EVAL_SAMPLES,
                batch_size=BATCH_SIZE,
                steps=STEPS,
            )

            learned_mean = trace[1]["mean"]
            learned_variance = trace[1]["variance"]

            row = {
                "inner_steps": inner,
                "seed": seed,
                "mean": learned_mean,
                "variance": learned_variance,
                "oracle_mean": oracle_mean,
                "oracle_variance": oracle_variance,
                "mean_error_vs_oracle": (
                    learned_mean - oracle_mean
                ),
                "variance_error_vs_oracle": (
                    learned_variance - oracle_variance
                ),
                "abs_mean_error_vs_oracle": abs(
                    learned_mean - oracle_mean
                ),
                "abs_variance_error_vs_oracle": abs(
                    learned_variance - oracle_variance
                ),
            }

            rows.append(row)

            np.save(
                RESULTS
                / f"terminal_inner{inner}_seed{seed}.npy",
                np.asarray(terminal),
            )

            print(
                f"inner={inner:4d} "
                f"seed={seed} "
                f"mean={learned_mean: .8f} "
                f"var={learned_variance: .8f} "
                f"|dm|={row['abs_mean_error_vs_oracle']:.6f} "
                f"|dv|={row['abs_variance_error_vs_oracle']:.6f}"
            )

    csv_path = RESULTS / "inner_calibration.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("")
    print("============================================================")
    print("AVERAGES")
    print("============================================================")

    summary = []

    for inner in INNER_STEPS:
        subset = [
            row
            for row in rows
            if row["inner_steps"] == inner
        ]

        means = np.asarray(
            [row["mean"] for row in subset]
        )

        variances = np.asarray(
            [row["variance"] for row in subset]
        )

        mean_abs_errors = np.asarray(
            [
                row["abs_mean_error_vs_oracle"]
                for row in subset
            ]
        )

        variance_abs_errors = np.asarray(
            [
                row["abs_variance_error_vs_oracle"]
                for row in subset
            ]
        )

        record = {
            "inner_steps": inner,
            "mean_avg": means.mean(),
            "mean_std": means.std(ddof=0),
            "variance_avg": variances.mean(),
            "variance_std": variances.std(ddof=0),
            "mean_abs_error_avg": mean_abs_errors.mean(),
            "variance_abs_error_avg": (
                variance_abs_errors.mean()
            ),
        }

        summary.append(record)

        print(
            f"{inner:4d} steps: "
            f"mean={record['mean_avg']: .8f} "
            f"+/- {record['mean_std']:.6f}, "
            f"var={record['variance_avg']: .8f} "
            f"+/- {record['variance_std']:.6f}, "
            f"mean |error|={record['mean_abs_error_avg']:.6f}, "
            f"var |error|={record['variance_abs_error_avg']:.6f}"
        )

    summary_path = RESULTS / "inner_calibration_summary.csv"

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary)

    # Mean plot.
    fig, ax = plt.subplots(
        figsize=(6.2, 3.8)
    )

    for seed in SEEDS:
        seed_rows = [
            row
            for row in rows
            if row["seed"] == seed
        ]

        ax.plot(
            [row["inner_steps"] for row in seed_rows],
            [row["mean"] for row in seed_rows],
            marker="o",
            alpha=0.45,
            label=f"seed {seed}",
        )

    ax.axhline(
        oracle_mean,
        linestyle="--",
        label="discrete oracle",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Inner training steps")
    ax.set_ylabel("Mean after first outer update")
    ax.set_title("Learned-affine inner calibration")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        FIGURES / "inner_calibration_mean.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Variance plot.
    fig, ax = plt.subplots(
        figsize=(6.2, 3.8)
    )

    for seed in SEEDS:
        seed_rows = [
            row
            for row in rows
            if row["seed"] == seed
        ]

        ax.plot(
            [row["inner_steps"] for row in seed_rows],
            [row["variance"] for row in seed_rows],
            marker="o",
            alpha=0.45,
            label=f"seed {seed}",
        )

    ax.axhline(
        oracle_variance,
        linestyle="--",
        label="discrete oracle",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Inner training steps")
    ax.set_ylabel("Variance after first outer update")
    ax.set_title("Learned-affine inner calibration")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        FIGURES / "inner_calibration_variance.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("")
    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")
    print(
        f"  {FIGURES / 'inner_calibration_mean.png'}"
    )
    print(
        f"  {FIGURES / 'inner_calibration_variance.png'}"
    )


if __name__ == "__main__":
    main()
