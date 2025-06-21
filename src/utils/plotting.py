import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from typing import Optional


__all__ = ["save_depth_3d_plot"]


def save_depth_3d_plot(depth_mm: np.ndarray,
                       out_path: Path,
                       title: Optional[str] = None,
                       dpi: int = 150,
                       show: bool = False,
                       mask_zero: bool = True,
                       invert_z: bool = True,
                       top_down: bool = False):
    """Save a 3-D surface plot visualisation of a metric depth image.

    Parameters
    ----------
    depth_mm : np.ndarray (H×W)
        Depth values in millimetres. NaNs or infs are treated as zeros.
    out_path : pathlib.Path
        Destination PNG file. Parent directory is created automatically.
    title : str, optional
        Title text for the plot.
    dpi : int
        Figure resolution.
    show : bool
        Whether to display the plot before saving.
    mask_zero : bool
        Whether to mask zero-depth pixels (e.g. background).
    invert_z : bool
        Whether to invert the Z axis so that larger depth is visually "lower" in the scene.
    top_down : bool
        Render with camera directly above looking down (elev=90°) for a height-map like view.
    """
    depth = depth_mm.astype(float).copy()

    # Optionally ignore zero-depth pixels (e.g. background)
    if mask_zero:
        depth[depth <= 0.0] = np.nan

    # Replace remaining invalid entries
    depth = np.nan_to_num(depth, nan=np.nan)

    h, w = depth.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, depth, cmap='viridis', linewidth=0, antialiased=False, rstride=1, cstride=1)

    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')
    ax.set_zlabel('Depth [mm]')
    ax.invert_yaxis()  # match image coordinates (origin at top-left)

    # Flip Z so that larger depth is visually "lower" in the scene
    if invert_z:
        ax.invert_zaxis()

    # Optionally change view to top-down
    if top_down:
        ax.view_init(elev=90, azim=-90)

    if title:
        ax.set_title(title)

    # Colour-bar only if data exists
    if np.isfinite(depth).any():
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Depth [mm]')

    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig) 