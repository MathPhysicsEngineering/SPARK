import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import tifffile as tiff

from core.StereoCamera import CameraIntrinsics, StereoCamera
from utils.reprojection import warp_image, CameraIntrinsics as UtilsIntrinsics, reconstruction_error


def _intrinsics_to_utils(intr: CameraIntrinsics) -> UtilsIntrinsics:
    return UtilsIntrinsics(intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height)


def save_stereo_dataset(scene: o3d.geometry.TriangleMesh, stereo: StereoCamera, pose_name: str, output_root: Path):
    """Capture, save RGB/Depth, reconstructions and diffs for the current pose."""
    data = stereo.capture_stereo_images(scene)
    left, right = data['left'], data['right']

    out_dir = output_root / pose_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save RGB + depth
    # ------------------------------------------------------------------
    for side, d in data.items():
        cv2.imwrite(str(out_dir / f"{side}.png"), (d['rgb'] * 255).astype(np.uint8)[..., ::-1])
        depth_u16 = np.clip(d['depth'], 0, 65535).astype(np.uint16)
        tiff.imwrite(out_dir / f"{side}_depth.tiff", depth_u16)

    # ------------------------------------------------------------------
    # Reconstructions
    # ------------------------------------------------------------------
    left_utils = _intrinsics_to_utils(stereo.left_camera.intrinsics)
    right_utils = _intrinsics_to_utils(stereo.right_camera.intrinsics)

    recon_right = warp_image(
        left['rgb'],
        left['depth'],
        left_utils,
        stereo.left_camera.pose,
        right_utils,
        stereo.right_camera.pose,
    )
    recon_left = warp_image(
        right['rgb'],
        right['depth'],
        right_utils,
        stereo.right_camera.pose,
        left_utils,
        stereo.left_camera.pose,
    )

    cv2.imwrite(str(out_dir / "recon_right.png"), (recon_right * 255).astype(np.uint8)[..., ::-1])
    cv2.imwrite(str(out_dir / "recon_left.png"), (recon_left * 255).astype(np.uint8)[..., ::-1])

    # Diff visualisations (grayscale error maps)
    diff_right = np.clip(np.mean(np.abs(right['rgb'] - recon_right), axis=2), 0, 1)
    diff_left = np.clip(np.mean(np.abs(left['rgb'] - recon_left), axis=2), 0, 1)
    cv2.imwrite(str(out_dir / "diff_right.png"), (diff_right * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "diff_left.png"), (diff_left * 255).astype(np.uint8))

    # Error metrics
    err_r = reconstruction_error(right['rgb'], recon_right)
    err_l = reconstruction_error(left['rgb'], recon_left)
    with open(out_dir / "error.txt", 'w') as f:
        f.write(f"Left -> Right MAE: {err_r:.6f}\n")
        f.write(f"Right -> Left MAE: {err_l:.6f}\n")

    print(f"[{pose_name}]  Left→Right MAE: {err_r:.4f} | Right→Left MAE: {err_l:.4f}")


def main():
    # ------------------------------------------------------------------
    # 1. Load mesh / point cloud
    # ------------------------------------------------------------------
    mesh_path = Path("data/Hand.ply")
    if not mesh_path.exists():
        raise FileNotFoundError("Hand.ply not found in data/")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        # Convert point cloud to mesh (quick Poisson)
        pcd = o3d.io.read_point_cloud(str(mesh_path))
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    mesh.compute_vertex_normals()

    # ------------------------------------------------------------------
    # 2. Stereo camera setup (shared across poses)
    # ------------------------------------------------------------------
    w, h = 640, 480
    fx = fy = 500.0
    cx, cy = w / 2, h / 2
    left_intr = CameraIntrinsics(fx, fy, cx, cy, w, h)
    right_intr = CameraIntrinsics(fx + 2, fy + 2, cx, cy, w, h)

    rel_rot = o3d.geometry.get_rotation_matrix_from_xyz([0, np.deg2rad(0.2), 0])
    stereo = StereoCamera(left_intr, right_intr, baseline=6.0, relative_rotation=rel_rot)

    # Shared camera position (looking along -Z)
    center = np.asarray(mesh.get_center())
    cam_pos = center + np.array([0.0, 0.0, 300.0])  # 300 mm in front
    base_rot = o3d.geometry.get_rotation_matrix_from_xyz([0.0, np.pi, 0.0])

    # ------------------------------------------------------------------
    # 3. Define poses
    # ------------------------------------------------------------------
    Rz_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])  # 180° about Z
    poses = {
        "pose_default": base_rot,
        "pose_rotZ180": base_rot @ Rz_180,
    }

    output_root = Path("output_poses")
    output_root.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Loop over poses and generate datasets
    # ------------------------------------------------------------------
    for name, rot in poses.items():
        stereo.set_left_pose(cam_pos, rot)
        save_stereo_dataset(mesh, stereo, name, output_root)


if __name__ == "__main__":
    main() 