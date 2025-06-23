import numpy as np
import os
from src.core.mesh import Mesh
from src.core.StereoCamera import Camera, CameraIntrinsics, StereoCamera
from src.utils.validation import check_stereo_consistency, validate_xyz_consistency
import imageio.v2 as imageio
import tifffile
import cv2
from typing import Dict

def main():
    # Create output directory
    output_dir = "data/synthetic_gt"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load mesh
    mesh_path = "data/Hand.ply"
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at {mesh_path}")
        return
        
    mesh = Mesh(mesh_path)
    if not mesh.mesh:
        print("Error: Failed to load mesh")
        return
    
    # Scale mesh to reasonable size (around 200mm in largest dimension)
    min_bounds, max_bounds = mesh.get_bounds()
    mesh_size = np.linalg.norm(max_bounds - min_bounds)
    target_size = 200.0  # 200mm
    scale = target_size / mesh_size
    mesh.mesh.scale(scale, mesh.get_center())
    
    # Update bounds after scaling
    min_bounds, max_bounds = mesh.get_bounds()
    mesh_size = np.linalg.norm(max_bounds - min_bounds)
    print(f"Mesh size after scaling: {mesh_size:.1f}mm")
    
    # Create camera intrinsics (adjusted for mesh size)
    focal_length = 1.5 * mesh_size  # Focal length 1.5x the mesh size
    left_intrinsics = CameraIntrinsics(
        fx=focal_length,
        fy=focal_length,
        cx=640.0,
        cy=360.0,
        width=1280,
        height=720
    )
    
    right_intrinsics = CameraIntrinsics(
        fx=focal_length,
        fy=focal_length,
        cx=640.0,
        cy=360.0,
        width=1280,
        height=720
    )
    
    # Initialize stereo camera with wider baseline
    baseline = mesh_size * 0.1  # Baseline 10% of mesh size
    stereo = StereoCamera(
        left_intrinsics=left_intrinsics,
        right_intrinsics=right_intrinsics,
        baseline=baseline
    )
    
    # Set camera pose for good viewing angle
    mesh_center = mesh.get_center()
    
    # Position camera at 45 degree angle above and to the side
    distance = mesh_size * 2.0  # Camera distance 2x mesh size
    elevation = np.radians(30)  # 30 degrees up
    azimuth = np.radians(45)    # 45 degrees to the side
    
    position = mesh_center + np.array([
        distance * np.cos(elevation) * np.cos(azimuth),
        distance * np.cos(elevation) * np.sin(azimuth),
        distance * np.sin(elevation)
    ])
    
    # Create rotation matrix to look at mesh center
    forward = mesh_center - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0, 0, 1]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.stack([right, up, -forward], axis=1)
    
    print(f"Camera distance: {distance:.1f}mm")
    print(f"Camera baseline: {baseline:.1f}mm")
    
    stereo.set_left_pose(position, rotation)
    
    # Generate ground truth
    print("Generating ground truth...")
    gt_data = stereo.capture_stereo_images(mesh.mesh)
    
    # Save data
    print("Saving data...")
    save_data(gt_data, output_dir)
    
    # Validate consistency
    print("\nValidating consistency...")
    camera_params = {
        'fx': float(stereo.left_camera.intrinsics.fx),
        'fy': float(stereo.left_camera.intrinsics.fy),
        'cx': float(stereo.left_camera.intrinsics.cx),
        'cy': float(stereo.left_camera.intrinsics.cy),
        'baseline': float(stereo.baseline)
    }
    
    # Check stereo consistency
    stereo_consistent, stereo_error = check_stereo_consistency(
        gt_data['left'], gt_data['right'], camera_params
    )
    print(f"Stereo consistency: {stereo_consistent}")
    print(f"Stereo error: {stereo_error:.3f}mm")
    
    # Check XYZ consistency
    xyz_consistent, xyz_error = validate_xyz_consistency(
        gt_data['left'], gt_data['right'], camera_params
    )
    print(f"XYZ consistency: {xyz_consistent}")
    print(f"XYZ error: {xyz_error:.3f}mm")
    
    print("\nDone!")

def save_data(data: Dict[str, Dict[str, np.ndarray]], output_dir: str):
    """Save the captured data to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save left view
    left_xyz = data['left']['xyz']
    cv2.imwrite(os.path.join(output_dir, 'left_rgb.png'), (data['left']['rgb'] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, 'left_depth.tiff'), data['left']['depth'].astype(np.float32))
    cv2.imwrite(os.path.join(output_dir, 'left_x.tiff'), left_xyz[..., 0].astype(np.float32))
    cv2.imwrite(os.path.join(output_dir, 'left_y.tiff'), left_xyz[..., 1].astype(np.float32))
    cv2.imwrite(os.path.join(output_dir, 'left_z.tiff'), left_xyz[..., 2].astype(np.float32))
    np.save(os.path.join(output_dir, 'left_xyz.npy'), left_xyz)
    
    # Save right view
    right_xyz = data['right']['xyz']
    cv2.imwrite(os.path.join(output_dir, 'right_rgb.png'), (data['right']['rgb'] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, 'right_depth.tiff'), data['right']['depth'].astype(np.float32))
    cv2.imwrite(os.path.join(output_dir, 'right_x.tiff'), right_xyz[..., 0].astype(np.float32))
    cv2.imwrite(os.path.join(output_dir, 'right_y.tiff'), right_xyz[..., 1].astype(np.float32))
    cv2.imwrite(os.path.join(output_dir, 'right_z.tiff'), right_xyz[..., 2].astype(np.float32))
    np.save(os.path.join(output_dir, 'right_xyz.npy'), right_xyz)

if __name__ == "__main__":
    main() 