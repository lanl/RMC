#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate the manuscript figure for the 1D neural Adjoint Sampling
failure-mode suite.

Input
-----
studies/adjoint_gaussian_instability/results/multimodal_v1_50/trace.csv

Outputs
-------
studies/adjoint_gaussian_instability/results/multimodal_v1_50/generated/
    NNfailureModes.pdf
    NNfailureModes_summary.csv

The script validates that all 4 problems x 4 regimes x 3 seeds
completed 50 outer iterations before producing the figure.
"""

from __future__ import annotations

import csv
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


HERE = Path(__file__).resolve()
STUDY = HERE.parent.parent
RESULTS = STUDY / "results" / "multimodal_v1_50"
TRACE = RESULTS / "trace.csv"
OUT = RESULTS / "generated"

FIGURE = OUT / "NNfailureModes.pdf"
SUMMARY = OUT / "NNfailureModes_summary.csv"


PROBLEMS = [
    "single_gaussian",
    "displaced_modes",
    "missing_mode",
    "wrong_weights",
]

REGIMES = [
    "no_replay_undamped",
    "replay_undamped",
    "no_replay_u_damped",
    "replay_u_damped",
]

SEEDS = [0, 1, 2]
EXPECTED_OUTER = 50


REGIME_STYLE = {
    "no_replay_undamped": {
        "label": "No replay, undamped",
        "color": "#0072B2",
    },
    "replay_undamped": {
        "label": "Replay, undamped",
        "color": "#D55E00",
    },
    "no_replay_u_damped": {
        "label": "No replay, damped",
        "color": "#009E73",
    },
    "replay_u_damped": {
        "label": "Replay, damped",
        "color": "#CC79A7",
    },
}


PANEL_INFO = {
    "single_gaussian": {
        "field": "mean",
        "target": 1.0,
        "ylabel": r"terminal mean $m_k$",
        "log": False,
    },
    "displaced_modes": {
        "field": "half_separation",
        "target": 1.0,
        "ylabel": r"half-separation $\widehat{\nu}_k$",
        "log": False,
    },
    "missing_mode": {
        "field": "p_left",
        "target": 0.5,
        "ylabel": r"left-basin mass $P_k(X<0)$",
        "log": True,
    },
    "wrong_weights": {
        "field": "p_left",
        "target": 10.0 / 11.0,
        "ylabel": r"left-basin mass $P_k(X<0)$",
        "log": False,
    },
}


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_rows():
    if not TRACE.exists():
        raise FileNotFoundError(
            f"Missing experiment trace:\n{TRACE}"
        )

    with TRACE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Trace is empty: {TRACE}")

    return rows


def validate(rows):
    groups = defaultdict(set)

    for row in rows:
        problem = row["problem"]
        regime = row["regime"]
        seed = int(row["seed"])
        k = int(row["k"])

        if (
            problem in PROBLEMS
            and regime in REGIMES
            and seed in SEEDS
        ):
            groups[(problem, regime, seed)].add(k)

    expected = {
        (problem, regime, seed)
        for problem in PROBLEMS
        for regime in REGIMES
        for seed in SEEDS
    }

    missing = sorted(expected - set(groups))

    if missing:
        raise RuntimeError(
            "Missing trajectories:\n"
            + "\n".join(map(str, missing))
        )

    incomplete = []

    for key in sorted(expected):
        ks = groups[key]

        if EXPECTED_OUTER not in ks:
            incomplete.append(
                (key, max(ks) if ks else None)
            )

    if incomplete:
        raise RuntimeError(
            "Incomplete trajectories:\n"
            + "\n".join(
                f"{key}: max k={max_k}"
                for key, max_k in incomplete
            )
        )

    if len(expected) != 48:
        raise RuntimeError(
            f"Internal expected trajectory count is {len(expected)}, "
            "not 48."
        )

    print(
        "Validated 48 completed trajectories "
        f"through k={EXPECTED_OUTER}."
    )


def curve(rows, problem, regime, seed, field):
    selected = [
        row
        for row in rows
        if row["problem"] == problem
        and row["regime"] == regime
        and int(row["seed"]) == seed
    ]

    selected.sort(
        key=lambda row: int(row["k"])
    )

    ks = np.asarray(
        [int(row["k"]) for row in selected],
        dtype=int,
    )

    values = np.asarray(
        [as_float(row[field]) for row in selected],
        dtype=float,
    )

    return ks, values


def curves(rows, problem, regime, field):
    all_curves = []

    common_ks = None

    for seed in SEEDS:
        ks, values = curve(
            rows,
            problem,
            regime,
            seed,
            field,
        )

        if common_ks is None:
            common_ks = ks
        elif not np.array_equal(
            ks,
            common_ks,
        ):
            raise RuntimeError(
                f"Iteration mismatch for "
                f"{problem}/{regime}/seed{seed}"
            )

        all_curves.append(values)

    return (
        common_ks,
        np.asarray(all_curves),
    )


def nanmean_axis0(values):
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )
        return np.nanmean(
            values,
            axis=0,
        )


def make_figure(rows):
    # Match the manuscript PDE figure style in
    # randomPythonScripts/visualizePDEsolverFinal.py.
    plt.rcParams.update(
        {
            "font.family": "Serif",
            "font.size": 14,
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
    )
    axes = axes.reshape((4,))

    # Same plasma palette used by the PDE figure, sampled once for
    # each algorithmic regime.
    colors = cm.plasma(
        [0.85, 2.0 / 3.0, 1.0 / 3.0, 0.0]
    )

    for regime, color in zip(REGIMES, colors):
        REGIME_STYLE[regime]["color"] = color

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    # Chosen to avoid the dominant portions of each panel while
    # retaining the per-panel legend convention of the PDE figure.
    legend_locs = [
        "upper left",
        "upper left",
        "upper right",
        "upper left",
    ]

    for i, (ax, problem) in enumerate(zip(axes, PROBLEMS)):
        info = PANEL_INFO[problem]

        for regime in REGIMES:
            style = REGIME_STYLE[regime]

            ks, vals = curves(
                rows,
                problem,
                regime,
                info["field"],
            )

            plot_vals = vals.copy()

            if info["log"]:
                # With 10^4 diagnostic samples, 10^-4 is one observed
                # sample. Empirical zeros are placed slightly below
                # that level so they remain visible on a log axis.
                plot_vals = np.maximum(
                    plot_vals,
                    5.0e-5,
                )

            # Individual seeds.
            for seed_values in plot_vals:
                ax.plot(
                    ks,
                    seed_values,
                    color=style["color"],
                    alpha=0.25,
                    linewidth=1.0,
                )

            # Three-seed mean.
            mean_values = nanmean_axis0(vals)

            if info["log"]:
                mean_values = np.maximum(
                    mean_values,
                    5.0e-5,
                )

            ax.plot(
                ks,
                mean_values,
                color=style["color"],
                linewidth=2.5,
                label=style["label"],
            )

        # Target reference, styled analogously to the red dotted
        # target locations in the PDE figure.
        ax.axhline(
            info["target"],
            linestyle=":",
            color="red",
            linewidth=1.5,
            label="Target",
            zorder=0,
        )

        ax.set_xlabel("outer iteration")
        ax.set_ylabel(info["ylabel"])
        ax.set_xlim(0, EXPECTED_OUTER)

        # Leave deliberate vertical headroom for the legend rather
        # than shrinking the typography or placing it over the data.
        if problem == "single_gaussian":
            ax.set_ylim(-0.35, 2.55)

        elif problem == "displaced_modes":
            ax.set_ylim(0.15, 2.05)

        elif problem == "missing_mode":
            ax.set_yscale("log")
            ax.set_ylim(
                5.0e-5,
                100.0,
            )
            ax.axhline(
                1.0e-4,
                color="0.45",
                linestyle=":",
                linewidth=1.0,
                zorder=0,
            )

        elif problem == "wrong_weights":
            ax.set_ylim(0.0, 1.65)

        ax.legend(
            frameon=False,
            fontsize=11.5,
            loc=legend_locs[i],
            ncol=2,
            columnspacing=0.8,
            handlelength=1.6,
            handletextpad=0.5,
            borderaxespad=0.5,
        )

        ax.text(
            0,
            1.1,
            panel_labels[i],
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="right",
        )

    fig.tight_layout(
        pad=1.4,
        w_pad=2.6,
        h_pad=2.8,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def write_summary(rows):
    fields = [
        "problem",
        "regime",
        "target",
        "final_primary_mean",
        "final_primary_seed_sd",
        "tail10_abs_primary_error",
        "tail10_primary_defined_fraction",
        "tail10_w2",
        "final_variance_mean",
    ]

    with SUMMARY.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for problem in PROBLEMS:
            info = PANEL_INFO[problem]

            for regime in REGIMES:
                final_primary = []
                final_variance = []
                tail_primary = []
                tail_w2 = []

                for seed in SEEDS:
                    selected = [
                        row
                        for row in rows
                        if row["problem"] == problem
                        and row["regime"] == regime
                        and int(row["seed"]) == seed
                        and int(row["k"]) > 0
                    ]

                    selected.sort(
                        key=lambda row: int(
                            row["k"]
                        )
                    )

                    last = selected[-1]
                    tail = selected[-10:]

                    final_primary.append(
                        as_float(
                            last[
                                info["field"]
                            ]
                        )
                    )

                    final_variance.append(
                        as_float(
                            last["variance"]
                        )
                    )

                    tail_primary.extend(
                        as_float(
                            row[
                                info["field"]
                            ]
                        )
                        for row in tail
                    )

                    tail_w2.extend(
                        as_float(
                            row["w2"]
                        )
                        for row in tail
                    )

                final_primary = np.asarray(
                    final_primary,
                    dtype=float,
                )

                final_variance = np.asarray(
                    final_variance,
                    dtype=float,
                )

                tail_primary = np.asarray(
                    tail_primary,
                    dtype=float,
                )

                tail_w2 = np.asarray(
                    tail_w2,
                    dtype=float,
                )

                valid = np.isfinite(
                    tail_primary
                )

                if np.any(valid):
                    tail_error = float(
                        np.mean(
                            np.abs(
                                tail_primary[valid]
                                - info["target"]
                            )
                        )
                    )
                else:
                    tail_error = np.nan

                writer.writerow(
                    {
                        "problem": problem,
                        "regime": regime,
                        "target": (
                            info["target"]
                        ),
                        "final_primary_mean": float(
                            np.nanmean(
                                final_primary
                            )
                        ),
                        "final_primary_seed_sd": float(
                            np.nanstd(
                                final_primary
                            )
                        ),
                        "tail10_abs_primary_error": (
                            tail_error
                        ),
                        "tail10_primary_defined_fraction": float(
                            np.mean(valid)
                        ),
                        "tail10_w2": float(
                            np.nanmean(
                                tail_w2
                            )
                        ),
                        "final_variance_mean": float(
                            np.nanmean(
                                final_variance
                            )
                        ),
                    }
                )


def main():
    rows = load_rows()

    validate(rows)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    make_figure(rows)
    write_summary(rows)

    print()
    print(f"Figure:  {FIGURE}")
    print(f"Summary: {SUMMARY}")


if __name__ == "__main__":
    main()
