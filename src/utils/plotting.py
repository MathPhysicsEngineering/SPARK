import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from typing import Optional


def save_depth_3d_plot(depth_mm: np.ndarray,
                       out_path: Path,
                       title: Optional[str] = None,
                       dpi: int = 150,
                       show: bool = False):
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
    """
    depth = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)

    h, w = depth.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, depth, cmap='viridis', linewidth=0, antialiased=False)

    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')
    ax.set_zlabel('Depth [mm]')
    ax.invert_yaxis()  # match image coordinates (origin at top-left)

    if title:
        ax.set_title(title)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Depth [mm]')
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig) 