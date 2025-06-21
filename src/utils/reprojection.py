import numpy as np
import cv2
from typing import Tuple

# Reuse shared intrinsics dataclass from core to avoid duplication
from core.StereoCamera import CameraIntrinsics


def _pixel_grid(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create meshgrid of pixel coordinates (u, v)."""
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    return u, v


def depth_to_points(depth: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """Convert a depth map to 3D points in camera coordinates (shape HxWx3, mm)."""
    u, v = _pixel_grid(intr.width, intr.height)
    z = depth
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    points = np.stack((x, y, z), axis=-1)  # H, W, 3
    return points


def warp_image(
    src_rgb: np.ndarray,
    src_depth: np.ndarray,
    src_intr: CameraIntrinsics,
    src_pose: np.ndarray,
    tgt_intr: CameraIntrinsics,
    tgt_pose: np.ndarray,
) -> np.ndarray:
    """Reconstruct target view from a source RGB-D image.

    Args:
        src_rgb: Source RGB image as float32 [0,1], shape (H, W, 3)
        src_depth: Source depth map in mm, shape (H, W)
        src_intr: Source camera intrinsics
        src_pose: 4x4 camera-to-world pose of the source camera
        tgt_intr: Target camera intrinsics
        tgt_pose: 4x4 camera-to-world pose of the target camera

    Returns:
        Synthesised target RGB image as float32 [0,1], shape (H, W, 3).
    """
    # Convert depth to 3D points in homogeneous coords (camera frame)
    pts_cam = depth_to_points(src_depth, src_intr)  # H, W, 3
    H, W, _ = pts_cam.shape
    pts_cam_flat = pts_cam.reshape(-1, 3)

    # Filter invalid depth (<=0)
    valid_mask = pts_cam_flat[:, 2] > 0
    pts_cam_valid = pts_cam_flat[valid_mask]

    # Transform to world coordinates
    pts_cam_h = np.concatenate([pts_cam_valid, np.ones((pts_cam_valid.shape[0], 1))], axis=1)
    pts_world = (src_pose @ pts_cam_h.T).T[:, :3]

    # Transform to target camera coordinates
    tgt_pose_inv = np.linalg.inv(tgt_pose)
    pts_world_h = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1)
    pts_tgt = (tgt_pose_inv @ pts_world_h.T).T[:, :3]

    # Project into target image plane
    u = (pts_tgt[:, 0] * tgt_intr.fx) / pts_tgt[:, 2] + tgt_intr.cx
    v = (pts_tgt[:, 1] * tgt_intr.fy) / pts_tgt[:, 2] + tgt_intr.cy

    # Round to nearest pixel for mapping (could use interpolation)
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Prepare maps for cv2.remap (float32)
    map_x = np.full((tgt_intr.height, tgt_intr.width), -1, dtype=np.float32)
    map_y = np.full((tgt_intr.height, tgt_intr.width), -1, dtype=np.float32)

    # Original pixel coordinates in source image
    src_u, src_v = _pixel_grid(src_intr.width, src_intr.height)
    src_u = src_u.flatten().astype(np.float32)
    src_v = src_v.flatten().astype(np.float32)

    # Fill maps where projection is valid and within bounds
    mask_bounds = (
        (u >= 0) & (u < tgt_intr.width) & (v >= 0) & (v < tgt_intr.height) & (pts_tgt[:, 2] > 0)
    )
    u_valid = u[mask_bounds]
    v_valid = v[mask_bounds]
    src_u_valid = src_u[valid_mask][mask_bounds]
    src_v_valid = src_v[valid_mask][mask_bounds]

    # map_x/ map_y is source coordinate lookup for each target pixel
    map_x[np.floor(v_valid).astype(int), np.floor(u_valid).astype(int)] = src_u_valid
    map_y[np.floor(v_valid).astype(int), np.floor(u_valid).astype(int)] = src_v_valid

    # Perform remap (bilinear)
    src_rgb_8u = (src_rgb * 255).astype(np.uint8)
    reconstructed = cv2.remap(
        src_rgb_8u,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    reconstructed = reconstructed.astype(np.float32) / 255.0
    return reconstructed


def reconstruction_error(gt_rgb: np.ndarray, recon_rgb: np.ndarray) -> float:
    """Compute mean absolute error between ground truth and reconstructed RGB images."""
    diff = np.abs(gt_rgb - recon_rgb)
    return float(np.mean(diff)) 