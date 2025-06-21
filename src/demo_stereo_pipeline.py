import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from core.StereoCamera import CameraIntrinsics, StereoCamera
from utils.reprojection import warp_image, reconstruction_error, CameraIntrinsics as UtilsIntrinsics
from utils.plotting import save_depth_3d_plot


def _intrinsics_to_utils(intr: CameraIntrinsics) -> UtilsIntrinsics:
    """Helper to convert core intrinsics to utils dataclass (identical fields)."""
    return UtilsIntrinsics(intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height)


def main():
    # ---------------------------------------------------------------------
    # 1. Load mesh / point cloud from data folder
    # ---------------------------------------------------------------------
    mesh_path = Path("data/Hand.ply")

    def _build_mesh_from_pointcloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """Create a watertight mesh from point cloud via Poisson reconstruction (higher depth)."""
        print("Converting point cloud to mesh via Poisson reconstruction – this may take a moment…")

        # Estimate normals – use neighbour radius proportional to the bbox diagonal
        bbox = pcd.get_axis_aligned_bounding_box()
        diag_len = np.linalg.norm(bbox.get_extent())
        normal_radius = diag_len * 0.02  # 2 % of diagonal
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=50))

        # Reconstruct – depth 11 gives finer details; scale slightly >1 to capture full volume
        mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=11,
            scale=1.2,
            linear_fit=True,
        )
        # Remove low-density vertices to clean mesh
        densities = np.asarray(densities)
        density_threshold = np.percentile(densities, 2)
        vertices_to_keep = densities > density_threshold
        mesh_poisson.remove_vertices_by_mask(~vertices_to_keep)
        mesh_poisson.remove_degenerate_triangles()
        mesh_poisson.remove_duplicated_triangles()
        mesh_poisson.remove_duplicated_vertices()
        mesh_poisson.remove_non_manifold_edges()
        mesh_poisson.compute_vertex_normals()
        return mesh_poisson

    if mesh_path.exists():
        # Try loading as triangle mesh first
        tri_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if tri_mesh.has_triangles() and len(tri_mesh.triangles) > 0:
            mesh = tri_mesh
            mesh.compute_vertex_normals()
            geom_display = mesh
        else:
            # Load as point cloud then convert
            print("Loaded file appears to be a point cloud – reconstructing mesh…")
            pcd_raw = o3d.io.read_point_cloud(str(mesh_path))
            if not pcd_raw.has_points():
                raise RuntimeError("Hand.ply contains no points – cannot proceed.")
            mesh = _build_mesh_from_pointcloud(pcd_raw)
            geom_display = mesh
    else:
        raise FileNotFoundError("data/Hand.ply not found – ensure the file exists.")

    # Display the reconstructed mesh in a separate Open3D window (non-blocking if possible)
    try:
        o3d.visualization.draw_geometries([geom_display], window_name="Hand Mesh")
    except Exception as e:
        print(f"Warning: Failed to open Open3D viewer – {e}")

    # ---------------------------------------------------------------------
    # 2. Create stereo camera pair with slightly different intrinsics
    # ---------------------------------------------------------------------
    width, height = 640, 480
    fx_left, fy_left = 500.0, 500.0
    fx_right, fy_right = 502.0, 502.0  # small difference
    cx, cy = width / 2, height / 2

    left_intr = CameraIntrinsics(fx_left, fy_left, cx, cy, width, height)
    right_intr = CameraIntrinsics(fx_right, fy_right, cx, cy, width, height)

    # Small rotation (~0.2 degrees around Y)
    theta = np.deg2rad(0.2)
    rel_rot = o3d.geometry.get_rotation_matrix_from_xyz([0, theta, 0])

    stereo_cam = StereoCamera(left_intr, right_intr, baseline=6.0, relative_rotation=rel_rot)

    # ---------------------------------------------------------------------
    # 3. Position the stereo camera looking at the mesh centre
    # ---------------------------------------------------------------------
    center = np.asarray(mesh.get_center())
    cam_pos = center + np.array([0.0, 0.0, 300.0])  # 300mm in +Z
    # Rotate 180° around Y so optical axis points towards -Z (object)
    cam_rot = o3d.geometry.get_rotation_matrix_from_xyz([0.0, np.pi, 0.0])
    stereo_cam.set_left_pose(cam_pos, cam_rot)

    # ---------------------------------------------------------------------
    # 4. Capture stereo RGB-D
    # ---------------------------------------------------------------------
    data = stereo_cam.capture_stereo_images(mesh)
    left = data['left']
    right = data['right']

    # ---------------------------------------------------------------------
    # 5. Save images
    # ---------------------------------------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save RGB (PNG) and depth (TIFF with metric values in mm)
    # ------------------------------------------------------------------
    for side, d in data.items():
        # RGB is stored as 8-bit PNG (OpenCV expects BGR ordering)
        rgb_8u = (d['rgb'] * 255).astype(np.uint8)[..., ::-1]
        cv2.imwrite(str(out_dir / f"{side}.png"), rgb_8u)

        # Depth is already in millimetres → store unscaled as 16-bit TIFF
        depth_mm = np.nan_to_num(d['depth'], nan=0.0, posinf=0.0, neginf=0.0)

        # Save metric depth (mm) – convert to uint16 preserving absolute values
        depth_tiff = np.clip(depth_mm, 0, 65535).astype(np.uint16)

        # Using tifffile preserves the 16-bit data without compression artefacts
        try:
            import tifffile as tiff
            tiff.imwrite(out_dir / f"{side}_depth.tiff", depth_tiff)
        except ImportError:
            # Fallback: OpenCV can also write TIFF but ensure correct dtype
            cv2.imwrite(str(out_dir / f"{side}_depth.tiff"), depth_tiff)

        # Also write a quick-look PNG scaled to full 16-bit range for human viewing
        if np.max(depth_mm) > 0:
            depth_vis = (depth_mm / np.max(depth_mm) * 65535.0).astype(np.uint16)
            cv2.imwrite(str(out_dir / f"{side}_depth_vis.png"), depth_vis)

        # ------------------------------------------------------------------
        # 3-D surface plot (interactive + saved PNG)
        # ------------------------------------------------------------------
        save_depth_3d_plot(
            depth_mm,
            out_dir / f"{side}_depth_3d.png",
            title=f"{side.capitalize()} Depth 3D Surface",
            show=True,
        )

    # ---------------------------------------------------------------------
    # 6. Reconstruct right from left and depth & vice-versa
    # ---------------------------------------------------------------------
    left_utils_intr = _intrinsics_to_utils(left_intr)
    right_utils_intr = _intrinsics_to_utils(right_intr)

    recon_right = warp_image(
        left['rgb'],
        left['depth'],
        left_utils_intr,
        stereo_cam.left_camera.pose,
        right_utils_intr,
        stereo_cam.right_camera.pose,
    )

    recon_left = warp_image(
        right['rgb'],
        right['depth'],
        right_utils_intr,
        stereo_cam.right_camera.pose,
        left_utils_intr,
        stereo_cam.left_camera.pose,
    )

    # Save reconstructed images
    cv2.imwrite(str(out_dir / "recon_right.png"), (recon_right * 255).astype(np.uint8)[..., ::-1])
    cv2.imwrite(str(out_dir / "recon_left.png"), (recon_left * 255).astype(np.uint8)[..., ::-1])

    # Save diff visualisations (8-bit grayscale scaled to full range)
    diff_right_norm = np.clip(np.mean(np.abs(right['rgb'] - recon_right), axis=2), 0, 1)
    diff_left_norm  = np.clip(np.mean(np.abs(left['rgb'] - recon_left), axis=2), 0, 1)
    cv2.imwrite(str(out_dir / "diff_right.png"), (diff_right_norm*255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "diff_left.png"),  (diff_left_norm*255).astype(np.uint8))

    # ---------------------------------------------------------------------
    # 7. Report errors
    # ---------------------------------------------------------------------
    err_left_to_right = reconstruction_error(right['rgb'], recon_right)
    err_right_to_left = reconstruction_error(left['rgb'], recon_left)

    report_path = out_dir / "error_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Left -> Right MAE: {err_left_to_right:.6f}\n")
        f.write(f"Right -> Left MAE: {err_right_to_left:.6f}\n")

    print("Pipeline finished. Outputs saved to", out_dir)
    print(f"Left -> Right MAE: {err_left_to_right:.6f}")
    print(f"Right -> Left MAE: {err_right_to_left:.6f}")

    # ---------------------------------------------------------------------
    # 9. Visualise each image in its own window for easier inspection
    # ---------------------------------------------------------------------

    def _depth_norm(depth_img: np.ndarray) -> np.ndarray:
        """Normalise depth to [0,1] for display purposes."""
        max_d = np.nanmax(depth_img)
        if max_d <= 0 or np.isnan(max_d):
            max_d = 1.0
        return depth_img / max_d

    # Prepare diff heat-maps (per-pixel absolute error averaged over channels)
    diff_right = np.mean(np.abs(right['rgb'] - recon_right), axis=2)
    diff_left = np.mean(np.abs(left['rgb'] - recon_left), axis=2)

    def _diff_norm(diff_img: np.ndarray) -> np.ndarray:
        max_v = np.max(diff_img)
        return diff_img / max_v if max_v > 0 else diff_img

    visualisations = [
        ("Recon Right (from Left)", recon_right, None),
        ("Diff Right vs GT", _diff_norm(diff_right), 'inferno'),
        ("Recon Left (from Right)", recon_left, None),
        ("Diff Left vs GT", _diff_norm(diff_left), 'inferno'),
    ]

    for title, img, cmap in visualisations:
        plt.figure(title)
        if cmap:
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main() 