#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_multimodal_suite as mm


TITLES = {
    "single_gaussian": "Single Gaussian",
    "displaced_modes": "Displaced modes",
    "wrong_weights": "Wrong weights",
    "missing_mode": "Missing mode",
}


def mixture_pdf(x, means, weights, std):
    out = np.zeros_like(x, dtype=float)

    for mean, weight in zip(means, weights):
        out += weight * norm.pdf(
            x,
            loc=mean,
            scale=std,
        )

    return out


def reference_sample(
    rng,
    spec,
    std,
    n=200000,
):
    means = np.asarray(
        spec["initial_means"],
        dtype=float,
    )
    weights = np.asarray(
        spec["initial_weights"],
        dtype=float,
    )

    comp = rng.choice(
        len(means),
        size=n,
        p=weights,
    )

    return rng.normal(
        loc=means[comp],
        scale=std,
    )


def w2_1d(x, ref):
    x = np.sort(
        np.asarray(x).reshape(-1)
    )

    q = (
        np.arange(len(x), dtype=float)
        + 0.5
    ) / len(x)

    ref_q = np.quantile(
        ref,
        q,
    )

    return float(
        np.sqrt(
            np.mean(
                (x - ref_q) ** 2
            )
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inner",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=8192,
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
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    var = mm.COMPONENT_VARIANCE
    std = np.sqrt(var)

    out = Path(
        "studies/adjoint_gaussian_instability/results/"
        f"initial_reconstruction_"
        f"{args.train_samples}_{args.inner}"
    )
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7),
    )

    fig.subplots_adjust(
        top=0.88,
        hspace=0.42,
        wspace=0.16,
    )

    rng = np.random.default_rng(
        12345
    )

    print()
    print("=" * 92)
    print(
        "Initial-law reconstruction: "
        f"N_init={args.train_samples}, "
        "buffer=p0, RAM target=p0, "
        f"{args.inner} inner steps"
    )
    print("=" * 92)

    for ax, problem in zip(
        axes.flat,
        TITLES,
    ):
        spec = mm.PROBLEMS[
            problem
        ]

        initial_density = (
            mm.GaussianMixture1DTarget(
                means=spec[
                    "initial_means"
                ],
                weights=spec[
                    "initial_weights"
                ],
                variance=var,
            )
        )

        model = mm.make_model(
            target=initial_density,
            seed=args.seed,
            train_samples=args.train_samples,
            batch_size=args.batch_size,
            inner_steps=args.inner,
            steps=args.steps,
        )

        optimizer = (
            model._build_optimizer()
        )

        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average(
                "loss"
            ),
        )

        replay = (
            mm.base._ReplayBuffer(
                dim=1,
                capacity=args.train_samples,
            )
        )

        key = jax.random.PRNGKey(
            int(spec["key_offset"])
            + args.seed
        )

        (
            key,
            p0_train_key,
            _,
        ) = jax.random.split(
            key,
            3,
        )

        training_endpoints = (
            mm.sample_mixture_jax(
                key=p0_train_key,
                means=spec[
                    "initial_means"
                ],
                weights=spec[
                    "initial_weights"
                ],
                variance=var,
                n=args.train_samples,
            )
        )

        (
            loss,
            clip_fraction,
            key,
        ) = (
            mm.base.train_full_adjoint(
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                replay=replay,
                endpoints=training_endpoints,
                inner_steps=args.inner,
                batch_size=args.batch_size,
                key=key,
            )
        )

        key, rollout_key = (
            jax.random.split(key)
        )

        rollout = np.asarray(
            model.generate_endpoints(
                args.eval_samples,
                rollout_key,
            )
        ).reshape(-1)

        np.save(
            out
            / f"{problem}_rollout.npy",
            rollout,
        )

        ref = reference_sample(
            rng,
            spec,
            std,
        )

        w2 = w2_1d(
            rollout,
            ref,
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

        exact_var = float(
            var
            + np.sum(
                weights
                * (
                    means
                    - exact_mean
                ) ** 2
            )
        )

        exact_left = float(
            np.sum(
                weights
                * norm.cdf(
                    0.0,
                    loc=means,
                    scale=std,
                )
            )
        )

        print()
        print(problem)
        print(
            f"  W2 to p0:        "
            f"{w2:.6f}"
        )
        print(
            f"  mean:            "
            f"{np.mean(rollout):.6f} "
            f"(exact {exact_mean:.6f})"
        )
        print(
            f"  variance:        "
            f"{np.var(rollout):.6f} "
            f"(exact {exact_var:.6f})"
        )
        print(
            f"  left mass:       "
            f"{np.mean(rollout < 0):.6f} "
            f"(exact {exact_left:.6f})"
        )
        print(
            f"  RAM loss:        "
            f"{float(loss):.6e}"
        )
        print(
            f"  clip fraction:   "
            f"{float(clip_fraction):.6f}"
        )

        lo = min(
            np.quantile(
                rollout,
                0.001,
            ),
            means.min()
            - 4.0 * std,
        )

        hi = max(
            np.quantile(
                rollout,
                0.999,
            ),
            means.max()
            + 4.0 * std,
        )

        grid = np.linspace(
            lo,
            hi,
            1000,
        )

        ax.hist(
            rollout,
            bins=70,
            density=True,
            alpha=0.35,
            label=(
                "initialized-control "
                "rollout"
            ),
        )

        ax.plot(
            grid,
            mixture_pdf(
                grid,
                spec[
                    "initial_means"
                ],
                spec[
                    "initial_weights"
                ],
                std,
            ),
            linewidth=2.0,
            label=(
                r"analytic initial "
                r"$p^{(0)}$"
            ),
        )

        ax.set_title(
            TITLES[problem]
            + "\n"
            + rf"$W_2 \approx {w2:.3f}$"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("density")

    handles, labels = (
        axes.flat[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.985,
        ),
        ncol=2,
        frameon=False,
    )

    figure_path = (
        out
        / "initial_density_reconstruction.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("saved:", figure_path)


if __name__ == "__main__":
    main()
