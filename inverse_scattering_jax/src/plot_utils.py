import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import jax.numpy as jnp
import numpy as np
from typing import Optional, Union, Tuple

class Plotting:
    """Class to handle plotting of physical domain and fields."""

    def __init__(
        self,
        nx: int,
        ny: int,
        h: float,
        npml: int,
        incident_directions: int,
        sampling_points: int,
    ):
        """Initializes the plotting object.

        Args:
            nx: Grid size in x.
            ny: Grid size in y.
            h: Grid spacing.
            npml: Number of PML layers.
            incident_directions: Number of incident directions.
            sampling_points: Number of sampling points.
        """
        self.nx = nx
        self.ny = ny
        self.h = h
        self.npml = npml
        self.incident_directions = incident_directions
        self.sampling_points = sampling_points

        # Physical extent calculation
        self.extent = [0, nx * h, 0, ny * h]
        
        # Domain limits (excluding PML)
        self.domain_x_min = npml * h
        self.domain_x_max = (nx - npml) * h
        self.domain_y_min = npml * h
        self.domain_y_max = (ny - npml) * h

    def _add_domain_limits(self, ax: plt.Axes):
        """Adds a rectangle indicating the physical domain limits."""
        rect = Rectangle(
            (self.domain_x_min, self.domain_y_min),
            self.domain_x_max - self.domain_x_min,
            self.domain_y_max - self.domain_y_min,
            linewidth=1,
            edgecolor='r',
            facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)

    def plot_perturbation(
        self,
        m: Union[jnp.ndarray, np.ndarray],
        title: str = "Perturbation",
        cmap: str = "seismic",
        show_limits: bool = True,
        ax: Optional[plt.Axes] = None,
    ):
        """Plots the perturbation model.

        Args:
            m: The model perturbation (flattened or 2D).
            title: Title of the plot.
            cmap: Colormap to use.
            show_limits: Whether to show physical domain limits.
            ax: Optional axes to plot on.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        
        m_plot = m.reshape((self.ny, self.nx))
        
        im = ax.imshow(
            m_plot,
            extent=self.extent,
            origin='lower',
            cmap=cmap
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)

        if show_limits:
            self._add_domain_limits(ax)

    def plot_scatterer_field(
        self,
        u: Union[jnp.ndarray, np.ndarray],
        title_prefix: str = "Scatterer Field",
        show_limits: bool = True,
        figsize: Tuple[int, int] = (18, 5),
    ):
        """Plots the scattered field (Real, Imaginary, Absolute).

        Args:
            u: The field (flattened or 2D).
            title_prefix: Prefix for the titles.
            show_limits: Whether to show physical domain limits.
            figsize: Figure size.
        """
        u_plot = u.reshape((self.ny, self.nx))
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Real part
        im0 = axes[0].imshow(
            np.real(u_plot),
            extent=self.extent,
            origin='lower',
            cmap='seismic'
        )
        axes[0].set_title(f"{title_prefix} (Real)")
        plt.colorbar(im0, ax=axes[0])
        
        # Imaginary part
        im1 = axes[1].imshow(
            np.imag(u_plot),
            extent=self.extent,
            origin='lower',
            cmap='seismic'
        )
        axes[1].set_title(f"{title_prefix} (Imaginary)")
        plt.colorbar(im1, ax=axes[1])
        
        # Absolute value
        im2 = axes[2].imshow(
            np.abs(u_plot),
            extent=self.extent,
            origin='lower',
            cmap='viridis'
        )
        axes[2].set_title(f"{title_prefix} (Absolute)")
        plt.colorbar(im2, ax=axes[2])

        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if show_limits:
                self._add_domain_limits(ax)
        
        plt.tight_layout()
