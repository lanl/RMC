#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "long_run" / "eight_regime"
TERMINAL = RESULTS / "terminal_samples"
FIGURES = ROOT / "figures" / "long_run"

MU = 1.0
TARGET_VARIANCE = 0.02
TARGET_STD = np.sqrt(TARGET_VARIANCE)

PRIMARY = {
    "affine_oracle": "Affine oracle",
    "learned_affine": "Learned affine",
    "full_replay": "Full Adjoint",
    "full_replay_damped_050":
        r"Full Adjoint + damping, $\eta=0.5$",
    "full_replay_damped_optimal":
        r"Full Adjoint + mean-optimal damping",
}

REPLAY_PAIRS = [
    (
        "full_no_replay",
        "full_replay",
        "No damping",
    ),
    (
        "full_no_replay_damped_050",
        "full_replay_damped_050",
        r"$\eta=0.5$",
    ),
    (
        "full_no_replay_damped_optimal",
        "full_replay_damped_optimal",
        r"Mean-optimal $\eta$",
    ),
]


def colors():
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {
        0: cycle[0],
        1: cycle[1],
        2: cycle[2],
    }


def load_trace():
    rows = []

    with open(
        RESULTS / "trace.csv",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            rows.append(
                {
                    "regime": row["regime"],
                    "seed": int(row["seed"]),
                    "k": int(row["k"]),
                    "mean": float(row["mean"]),
                    "variance": float(row["variance"]),
                    "loss": (
                        float(row["loss"])
                        if row["loss"]
                        else np.nan
                    ),
                    "stop_reason": row.get(
                        "stop_reason",
                        "",
                    ),
                }
            )

    return rows


def group_trace(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[
            (row["regime"], row["seed"])
        ].append(row)

    for key in groups:
        groups[key].sort(
            key=lambda row: row["k"]
        )

    return groups


def target_density(x):
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


def load_samples(regime, seed):
    return np.load(
        TERMINAL / f"{regime}_seed{seed}.npy"
    ).reshape(-1)


def plot_mean_trace(
    groups,
    regime,
    title,
):
    c = colors()

    available = sorted(
        seed
        for (name, seed) in groups
        if name == regime
    )

    fig, ax = plt.subplots(
        figsize=(6.3, 3.7)
    )

    for seed in available:
        g = groups[(regime, seed)]

        k = np.asarray(
            [row["k"] for row in g]
        )

        means = np.asarray(
            [row["mean"] for row in g]
        )

        label = (
            "oracle"
            if regime == "affine_oracle"
            else f"seed {seed}"
        )

        ax.plot(
            k,
            means,
            marker=(
                "o"
                if len(g) <= 12
                else None
            ),
            linewidth=1.5,
            color=c.get(seed, c[0]),
            label=label,
        )

    ax.axhline(
        MU,
        color="black",
        linewidth=1.4,
        linestyle="--",
        label="target mean",
    )

    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Terminal-sample mean")
    ax.set_title(title)
    ax.grid(True, alpha=0.18)
    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        FIGURES / f"{regime}_mean.png",
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_oracle_histogram():
    samples = load_samples(
        "affine_oracle",
        0,
    )

    sample_mean = float(samples.mean())
    sample_std = float(samples.std())

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(6.3, 3.7),
        gridspec_kw={
            "width_ratios": [1.0, 1.0],
            "wspace": 0.08,
        },
    )

    left_lo = MU - 4.0 * TARGET_STD
    left_hi = MU + 4.0 * TARGET_STD

    right_lo = (
        sample_mean
        - 4.0 * sample_std
    )

    right_hi = (
        sample_mean
        + 4.0 * sample_std
    )

    x_left = np.linspace(
        left_lo,
        left_hi,
        800,
    )

    ax1.plot(
        x_left,
        target_density(x_left),
        color="black",
        linewidth=2.0,
        label=r"target $N(1,0.02)$",
    )

    bins = np.linspace(
        right_lo,
        right_hi,
        70,
    )

    ax2.hist(
        samples,
        bins=bins,
        density=True,
        alpha=0.28,
        color=colors()[0],
        label="oracle samples",
    )

    ax1.set_xlim(
        left_lo,
        left_hi,
    )

    ax2.set_xlim(
        right_lo,
        right_hi,
    )

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(left=False)

    d = 0.015

    ax1.plot(
        (1 - d, 1 + d),
        (-d, +d),
        transform=ax1.transAxes,
        color="black",
        clip_on=False,
    )

    ax1.plot(
        (1 - d, 1 + d),
        (1 - d, 1 + d),
        transform=ax1.transAxes,
        color="black",
        clip_on=False,
    )

    ax2.plot(
        (-d, +d),
        (-d, +d),
        transform=ax2.transAxes,
        color="black",
        clip_on=False,
    )

    ax2.plot(
        (-d, +d),
        (1 - d, 1 + d),
        transform=ax2.transAxes,
        color="black",
        clip_on=False,
    )

    ax1.set_ylabel("Density")
    ax1.set_xlabel("x")
    ax2.set_xlabel("x")

    ax1.legend(
        fontsize=8,
        loc="upper left",
    )

    ax2.legend(
        fontsize=8,
        loc="upper right",
    )

    fig.suptitle(
        "Affine oracle: terminal distribution at stopping iteration",
        y=0.98,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    fig.savefig(
        FIGURES / "affine_oracle_hist.png",
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_seed_histograms(
    regime,
    title,
):
    c = colors()

    seed_samples = {
        seed: load_samples(
            regime,
            seed,
        )
        for seed in [0, 1, 2]
    }

    combined = np.concatenate(
        list(seed_samples.values())
    )

    lo = min(
        float(
            np.quantile(
                combined,
                0.001,
            )
        ),
        MU - 4.0 * TARGET_STD,
    )

    hi = max(
        float(
            np.quantile(
                combined,
                0.999,
            )
        ),
        MU + 4.0 * TARGET_STD,
    )

    span = hi - lo
    lo -= 0.04 * span
    hi += 0.04 * span

    bins = np.linspace(
        lo,
        hi,
        85,
    )

    x = np.linspace(
        lo,
        hi,
        1200,
    )

    fig, ax = plt.subplots(
        figsize=(6.3, 3.7)
    )

    for seed, samples in seed_samples.items():
        ax.hist(
            samples,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.16,
            color=c[seed],
            label=f"seed {seed}",
        )

        ax.hist(
            samples,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.25,
            color=c[seed],
        )

    ax.plot(
        x,
        target_density(x),
        color="black",
        linewidth=2.0,
        label=r"target $N(1,0.02)$",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, alpha=0.12)
    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        FIGURES / f"{regime}_hist.png",
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_replay_ablation(groups):
    c = colors()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, 3.25),
        sharex=True,
    )

    for ax, (
        no_replay,
        replay,
        title,
    ) in zip(
        axes,
        REPLAY_PAIRS,
    ):
        for seed in [0, 1, 2]:
            nr = groups[
                (no_replay, seed)
            ]

            rp = groups[
                (replay, seed)
            ]

            ax.plot(
                [row["k"] for row in nr],
                [row["mean"] for row in nr],
                linestyle="--",
                linewidth=1.05,
                alpha=0.65,
                color=c[seed],
            )

            ax.plot(
                [row["k"] for row in rp],
                [row["mean"] for row in rp],
                linestyle="-",
                linewidth=1.35,
                color=c[seed],
            )

        ax.axhline(
            MU,
            color="black",
            linewidth=1.1,
            linestyle=":",
        )

        ax.set_title(title)
        ax.set_xlabel("Outer iteration")
        ax.grid(True, alpha=0.15)

    axes[0].set_ylabel(
        "Terminal-sample mean"
    )

    seed_handles = [
        Line2D(
            [0],
            [0],
            color=c[seed],
            lw=1.7,
            label=f"seed {seed}",
        )
        for seed in [0, 1, 2]
    ]

    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            lw=1.5,
            linestyle="-",
            label="replay",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            lw=1.5,
            linestyle="--",
            label="no replay",
        ),
    ]

    fig.legend(
        handles=(
            seed_handles
            + style_handles
        ),
        loc="upper center",
        ncol=5,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.91]
    )

    fig.savefig(
        FIGURES
        / "replay_ablation_mean.png",
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


def build_summary(groups):
    regimes = sorted(
        {
            regime
            for regime, _ in groups
            if regime != "affine_oracle"
        }
    )

    records = []

    for regime in regimes:
        final_means = []
        final_variances = []
        tail_abs_errors = []
        tail_variances = []

        for seed in [0, 1, 2]:
            g = groups[
                (regime, seed)
            ]

            final = g[-1]

            final_means.append(
                final["mean"]
            )

            final_variances.append(
                final["variance"]
            )

            tail = [
                row
                for row in g
                if row["k"] > 0
            ][-20:]

            tail_abs_errors.extend(
                abs(
                    row["mean"]
                    - MU
                )
                for row in tail
            )

            tail_variances.extend(
                row["variance"]
                for row in tail
            )

        final_means = np.asarray(
            final_means
        )

        final_variances = np.asarray(
            final_variances
        )

        record = {
            "regime": regime,
            "final_mean_avg":
                float(final_means.mean()),
            "final_mean_seed_sd":
                float(final_means.std(ddof=0)),
            "final_variance_avg":
                float(final_variances.mean()),
            "final_variance_seed_sd":
                float(
                    final_variances.std(
                        ddof=0
                    )
                ),
            "tail20_abs_mean_error":
                float(
                    np.mean(
                        tail_abs_errors
                    )
                ),
            "tail20_variance_avg":
                float(
                    np.mean(
                        tail_variances
                    )
                ),
        }

        records.append(record)

    output = (
        RESULTS
        / "report_summary.csv"
    )

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "regime",
                "final_mean_avg",
                "final_mean_seed_sd",
                "final_variance_avg",
                "final_variance_seed_sd",
                "tail20_abs_mean_error",
                "tail20_variance_avg",
            ],
        )

        writer.writeheader()
        writer.writerows(records)

    return records


def main():
    FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_trace()
    groups = group_trace(rows)

    for regime, title in PRIMARY.items():
        plot_mean_trace(
            groups,
            regime,
            title,
        )

        if regime == "affine_oracle":
            plot_oracle_histogram()
        else:
            plot_seed_histograms(
                regime,
                title,
            )

    plot_replay_ablation(
        groups
    )

    summary = build_summary(
        groups
    )

    print("")
    print("===== REPORT SUMMARY =====")

    header = (
        f"{'regime':34s} "
        f"{'mean':>9s} "
        f"{'seed_sd':>9s} "
        f"{'var':>9s} "
        f"{'tail|e|':>9s} "
        f"{'tailvar':>9s}"
    )

    print(header)

    for row in summary:
        print(
            f"{row['regime']:34s} "
            f"{row['final_mean_avg']:9.5f} "
            f"{row['final_mean_seed_sd']:9.5f} "
            f"{row['final_variance_avg']:9.5f} "
            f"{row['tail20_abs_mean_error']:9.5f} "
            f"{row['tail20_variance_avg']:9.5f}"
        )

    print("")
    print("Figures:")

    for path in sorted(
        FIGURES.glob("*.png")
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
