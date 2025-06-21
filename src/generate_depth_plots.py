import argparse
from pathlib import Path
import numpy as np
import tifffile as tiff
from typing import Optional

# Ensure project src/ is on PYTHONPATH when running as standalone
import sys
SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# Now safe to import internal utilities
from utils.plotting import save_depth_3d_plot

DEF_TITLE_TMPL = "{name} depth"


def process_directory(dir_path: Path, out_subdir: Optional[str] = None, overwrite: bool = False, show: bool = False):
    """Search `dir_path` recursively for `*_depth.tiff` files and generate 3-D plot PNGs.

    Parameters
    ----------
    dir_path : Path
        Root directory to search for depth TIFFs.
    out_subdir : str, optional
        If provided, saves plots into `file.parent/out_subdir/` rather than alongside the TIFF.
    overwrite : bool
        Re-create images even if the output PNG already exists.
    show : bool
        Display interactive 3-D plots while saving
    """
    depth_files = list(dir_path.rglob("*_depth.tiff"))
    if not depth_files:
        print(f"No depth TIFFs found in {dir_path}.")
        return

    print(f"Found {len(depth_files)} depth images in {dir_path} – generating 3-D plots…")

    for depth_path in depth_files:
        depth = tiff.imread(str(depth_path)).astype(np.float32)
        # Output path
        if out_subdir:
            out_dir = depth_path.parent / out_subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (depth_path.stem + "_3d.png")
        else:
            out_path = depth_path.with_name(depth_path.stem + "_3d.png")

        if out_path.exists() and not overwrite:
            print(f"[skip] {out_path} already exists")
            continue

        title = DEF_TITLE_TMPL.format(name=depth_path.stem)
        save_depth_3d_plot(depth, out_path, title, show=show)
        print(f"[OK] Saved {out_path.relative_to(dir_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3-D surface plots from depth TIFFs.")
    parser.add_argument("root", type=str, help="Directory containing *_depth.tiff files (searched recursively)")
    parser.add_argument("--out-subdir", type=str, default=None, help="Save plots into this subdirectory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing plot files")
    parser.add_argument("--show", action="store_true", help="Display interactive 3-D plots while saving")
    args = parser.parse_args()

    process_directory(Path(args.root), args.out_subdir, args.overwrite, args.show) 