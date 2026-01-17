# -*- coding: utf-8 -*-

"""Auxiliary plotting utilities."""

from pathlib import Path
from typing import Callable

from jax.typing import ArrayLike

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata


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
        samples: Array of sample coordinates.
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
    if label is not None:
        ax.legend(loc="best", frameon=False)

    return ax


def plot_trajectories(samples: ArrayLike, ax, label: str, linestyle=":", zorder: int = 0, **kwargs):
    """Plot trajectories of samples in provided frame.

    Args:
        samples: Array of sample coordinates.
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

    ax.axis("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if label is not None:
        ax.legend(loc="best", frameon=False)

    return ax


def plot_quiver(
    samples: ArrayLike,
    velocity: ArrayLike,
    ax,
    scale_units: str = "xy",
    angles: str = "xy",
    **kwargs,
):
    """Plot a 2D velocity in provided frame.

    Args:
        samples: Array of sample coordinates.
        velocity: Array of velocities to plot.
        ax: Figure frame to plot trajectories.
        scale_units: Physical image unit, which is used for rendering the scaled arrow representing the velocity.
        angles: Method for determining the angle of the arrows. For 'uv', arrow directions are based on display coordinates. For 'xy', arrow directions are based on data coordinates.
        kwargs: Other properties.

    Returns:
        Figure frame with plotted 2D field.
    """
    ax.quiver(
        samples[:, 0],
        samples[:, 1],
        velocity[:, 0],
        velocity[:, 1],
        angles=angles,
        scale_units=scale_units,
        **kwargs,
    )

    ax.axis("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return ax


def plot_func_contours(
    func: Callable,
    ax,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nx: int,
    ny: int,
    keepscale: bool = True,
    cbar: bool = False,
    **kwargs,
):
    """Plot contours of passed function.

    This assumes that the passed function is a function of two variables.

    Args:
        func: Function to evaluate contours of.
        ax: Figure frame to plot contours.
        xmin: Minimum x coordinate.
        xmax: Maximum x coordinate.
        ymin: Minimum y coordinate.
        ymax: Maximum y coordinate.
        nx: Number of partitions in x range.
        ny: Number or partitions in y range.
        keepscale: Flag to indicate it the function evaluation must be used directly (True) or if it must be exponentiated for cases where func corresponds to log-target (False).
        cbar: Flag to indicate if colorbar must be plotted.
        kwargs: Other properties.

    Returns:
        Figure frame with plotted function contours.
    """

    xcoord = np.linspace(xmin, xmax, nx)
    ycoord = np.linspace(ymin, ymax, ny)

    mesh = np.meshgrid(xcoord, ycoord)

    if keepscale:
        feval = np.array(
            [func(np.array([x, y])) for x, y in zip(mesh[0].flatten(), mesh[1].flatten())]
        )
    else:  # For log-target
        feval = np.array(
            [np.exp(func(np.array([x, y]))) for x, y in zip(mesh[0].flatten(), mesh[1].flatten())]
        )
    fplot = griddata((mesh[0].flatten(), mesh[1].flatten()), feval, mesh, method="nearest")

    contours = ax.contourf(mesh[0], mesh[1], fplot, **kwargs)
    if cbar:
        plt.colorbar(contours, ax=ax)

    return ax


def plot_func_xDim_contours(
    func: Callable,
    dim: int,
    ax,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nx: int,
    ny: int,
    keepscale: bool = True,
    cbar: bool = False,
    **kwargs,
):
    """Plot contours of passed function.

    This effectively explores function contours in the two first variables and sets the
    rest of variables to zero.

    Args:
        func: Function to evaluate contours of.
        dim: Dimension of the function domain.
        ax: Figure frame to plot contours.
        xmin: Minimum x coordinate.
        xmax: Maximum x coordinate.
        ymin: Minimum y coordinate.
        ymax: Maximum y coordinate.
        nx: Number of partitions in x range.
        ny: Number or partitions in y range.
        keepscale: Flag to indicate it the function evaluation must be used directly (True) or if it must be exponentiated for cases where func corresponds to log-target (False).
        cbar: Flag to indicate if colorbar must be plotted.
        kwargs: Other properties.

    Returns:
        Figure frame with plotted function contours.
    """

    xcoord = np.linspace(xmin, xmax, nx)
    ycoord = np.linspace(ymin, ymax, ny)

    mesh = np.meshgrid(xcoord, ycoord)

    if keepscale:
        feval = np.array(
            [
                func(np.concatenate([np.array([x, y]), np.zeros(dim - 2)]))
                for x, y in zip(mesh[0].flatten(), mesh[1].flatten())
            ]
        )
    else:  # For log-target
        feval = np.array(
            [
                np.exp(func(np.concatenate([np.array([x, y]), np.zeros(dim - 2)])))
                for x, y in zip(mesh[0].flatten(), mesh[1].flatten())
            ]
        )
    fplot = griddata((mesh[0].flatten(), mesh[1].flatten()), feval, mesh, method="nearest")

    contours = ax.contourf(mesh[0], mesh[1], fplot, **kwargs)
    if cbar:
        plt.colorbar(contours, ax=ax)

    return ax


def save_plot(
    fig: plt.Figure,
    path: Path | str,
    dpi: int = 300,
    close: bool = True,
) -> None:
    """
    Generic figure saver.

    Args:
        fig: Matplotlib figure to store.
        path: Path to store file.
        dpi: Resolution to save figure.
        close: Flag to indicate if figure should be closed after saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
