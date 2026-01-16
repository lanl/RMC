# -*- coding: utf-8 -*-

"""Auxiliary plotting utilities."""


from jax.typing import ArrayLike


def plot_samples(
    samples: ArrayLike,
    ax,
    label: str,
    size: float = 8,
    marker: str = "o",
    alpha: float = 0.3,
    zorder: int = 0,
    **kwargs,
):
    """Plot samples in provided frame.

    Args:
        samples: Array of samples to plot. Assumes a 2D structure, with sample and coordinate components.
        ax: Figure frame to plot samples.
        label: Label to use for samples.
        size: Size of marker to use.
        marker: Marker style to use for plotting samples.
        alpha: Alpha blending value, between 0 (transparent) and 1 (opaque).
        zorder: Drawing order. Lower order values are drawn first.
        kwargs: Other properties.

    Returns:
        Figure frame with plotted samples.
    """
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        s=size,
        marker=marker,
        alpha=alpha,
        label=label,
        zorder=zorder,
        **kwargs,
    )
    ax.axis("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", frameon=False)

    return ax


def plot_trajectories(samples: ArrayLike, ax, label: str, linestyle=":", zorder: int = 0, **kwargs):
    """Plot trajectories of samples in provided frame.

    Args:
        samples: Array of samples to plot. Assumes a 2D structure, with sample and coordinate components.
        ax: Figure frame to plot trajectories.
        label: Label to use for trajectories.
        linestyle: Marker style to use for line plotting trajectories.
        zorder: Drawing order. Lower order values are drawn first.
        kwargs: Other properties.

    Returns:
        Figure frame with plotted trajectories.
    """
    ax.plot(
        samples[:, 0],
        samples[:, 1],
        linestyle=linestyle,
        label=label,
        zorder=zorder,
        **kwargs,
    )

    return ax
