import numpy as np
import os
import tifffile
from src.core.StereoCamera import CameraIntrinsics, StereoCamera
from src.core.mesh import Mesh

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
        
    # Create camera intrinsics (different for each camera)
    left_intrinsics = CameraIntrinsics(
        fx=1000.0,  # Left camera focal length
        fy=1000.0,
        cx=640.0,   # Left camera principal point
        cy=360.0,
        width=1280,
        height=720
    )
    
    right_intrinsics = CameraIntrinsics(
        fx=1001.0,  # Slightly different focal length
        fy=1001.0,
        cx=639.0,   # Slightly different principal point
        cy=361.0,
        width=1280,
        height=720
    )
    
    # Create relative rotation matrix (small rotation around Y axis)
    angle = np.radians(0.1)  # 0.1 degrees
    relative_rotation = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    
    # Initialize stereo camera with proper relative rotation
    stereo = StereoCamera(
        left_intrinsics=left_intrinsics,
        right_intrinsics=right_intrinsics,
        baseline=6.0,  # 6mm baseline
        relative_rotation=relative_rotation
    )
    
    # Set camera pose (looking at mesh center from 500mm away)
    mesh_center = mesh.get_center()
    position = mesh_center + np.array([0, 0, 500])  # 500mm away
    
    # Create 180-degree rotation around Z axis
    angle = np.radians(180)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    # Set left camera pose (right camera pose will be automatically set)
    stereo.set_left_pose(position, rotation)
    
    # Generate ground truth
    print("Generating ground truth...")
    gt_data = stereo.capture_stereo_images(mesh.mesh)
    
    # Save data
    print("Saving data...")
    
    # Create output directories
    left_dir = os.path.join(output_dir, "left")
    right_dir = os.path.join(output_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    
    # Save left camera data
    tifffile.imwrite(os.path.join(left_dir, "rgb.tiff"), (gt_data['left']['rgb'] * 255).astype(np.uint8))
    tifffile.imwrite(os.path.join(left_dir, "depth.tiff"), gt_data['left']['depth'].astype(np.float32))
    tifffile.imwrite(os.path.join(left_dir, "xyz.tiff"), gt_data['left']['xyz'].astype(np.float32))
    
    # Save right camera data
    tifffile.imwrite(os.path.join(right_dir, "rgb.tiff"), (gt_data['right']['rgb'] * 255).astype(np.uint8))
    tifffile.imwrite(os.path.join(right_dir, "depth.tiff"), gt_data['right']['depth'].astype(np.float32))
    tifffile.imwrite(os.path.join(right_dir, "xyz.tiff"), gt_data['right']['xyz'].astype(np.float32))
    
    print("\nDone!")

if __name__ == "__main__":
    main() 