import numpy as np
import open3d as o3d
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

class Camera:
    """Single camera with intrinsics and pose."""
    def __init__(self, 
                 intrinsics: CameraIntrinsics,
                 name: str = "camera"):
        """
        Initialize camera.
        
        Args:
            intrinsics: Camera intrinsic parameters
            name: Camera name for identification
        """
        self.intrinsics = intrinsics
        self.name = name
        self.pose = np.eye(4)  # Identity matrix for initial pose
        
        # Create Open3D camera intrinsics
        self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height,
            intrinsics.fx, intrinsics.fy,
            intrinsics.cx, intrinsics.cy
        )
        
    def set_pose(self, position: np.ndarray, rotation: np.ndarray):
        """
        Set camera pose.
        
        Args:
            position: 3D position vector [x, y, z] in mm
            rotation: 3x3 rotation matrix
        """
        self.pose = np.eye(4)
        self.pose[:3, :3] = rotation
        self.pose[:3, 3] = position
        
    def get_camera_parameters(self) -> o3d.camera.PinholeCameraParameters:
        """Get Open3D camera parameters for rendering."""
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = self.o3d_intrinsics
        params.extrinsic = np.linalg.inv(self.pose)  # Convert from camera-to-world to world-to-camera
        return params
        
    def capture_image(self, scene: o3d.geometry.TriangleMesh) -> Dict[str, np.ndarray]:
        """
        Capture RGB, depth and XYZ images from the scene.
        
        Args:
            scene: Open3D triangle mesh to render
            
        Returns:
            Dictionary containing:
            - rgb: RGB image (H, W, 3) in range [0, 1]
            - depth: Depth image (H, W) in mm
            - xyz: XYZ coordinates (H, W, 3) in mm
        """
        # Create renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(
            self.intrinsics.width, self.intrinsics.height
        )
        
        # Set up scene
        material = o3d.visualization.rendering.MaterialRecord()
        # ------------------------------------------------------------------
        # Material setup – favour shaded rendering with vertex colours when
        # available so RGB images capture geometry details.
        # ------------------------------------------------------------------
        if scene.has_vertex_colors():
            # Shaded but still preserving per-vertex colours
            material.shader = "defaultLit"
            material.base_color = [1.0, 1.0, 1.0, 1.0]
            material.vertex_color_use = o3d.visualization.rendering.MaterialRecord.VertexColorUse.VertexColor
        else:
            # Fallback: simple grey shaded
            material.shader = "defaultLit"
            material.base_color = [0.7, 0.7, 0.7, 1.0]
        
        # Add mesh to scene
        renderer.scene.add_geometry("mesh", scene, material)
        
        # Optional: background & lighting (guard against API differences)
        try:
            renderer.scene.set_background([0, 0, 0, 1])
        except AttributeError:
            pass  # Older Open3D versions
        
        # Disable indirect light if available
        if hasattr(renderer.scene, "enable_indirect_light"):
            renderer.scene.enable_indirect_light(False)
        
        # Try adding simple directional light – skip if API differs
        if hasattr(renderer.scene, "add_directional_light"):
            try:
                renderer.scene.add_directional_light(
                    "dir_light",
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, -1.0],
                    0.5,
                    True,
                )
            except Exception:
                pass
        
        # Set camera parameters
        cam_params = self.get_camera_parameters()
        renderer.setup_camera(cam_params.intrinsic, cam_params.extrinsic)
        
        # Configure optional render settings
        if hasattr(renderer.scene, "set_max_ray_depth"):
            renderer.scene.set_max_ray_depth(3)
        if hasattr(renderer.scene, "enable_sun_light"):
            renderer.scene.enable_sun_light(True)
        
        # Render RGB image
        rgb = renderer.render_to_image()
        if rgb is None:
            print(f"Warning: Failed to render RGB image for camera {self.name}")
            rgb = np.zeros((self.intrinsics.height, self.intrinsics.width, 3))
        else:
            rgb = np.asarray(rgb) / 255.0  # Convert to float [0, 1]
        
        # Render linear depth (view-space meters) then convert to mm
        depth_img = renderer.render_to_depth_image(z_in_view_space=True)
        depth = np.asarray(depth_img)
        
        # Compute XYZ coordinates in camera space
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.cx, self.intrinsics.cy
        
        # Create pixel coordinates
        y, x = np.meshgrid(np.arange(self.intrinsics.height), np.arange(self.intrinsics.width), indexing='ij')
        
        # Convert to camera coordinates
        valid_depth = depth > 0
        X = np.zeros_like(depth)
        Y = np.zeros_like(depth)
        Z = depth
        
        X[valid_depth] = (x[valid_depth] - cx) * depth[valid_depth] / fx
        Y[valid_depth] = (y[valid_depth] - cy) * depth[valid_depth] / fy
        
        # Stack coordinates
        xyz = np.stack([X, Y, Z], axis=-1)
        
        # Transform to world coordinates
        xyz_homogeneous = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)
        xyz_world = (self.pose @ xyz_homogeneous.reshape(-1, 4).T).T
        xyz_world = xyz_world[..., :3].reshape(xyz.shape)
        
        # Set invalid points to zero in world coordinates
        xyz_world[~valid_depth] = 0
        
        return {
            'rgb': rgb,
            'depth': depth,
            'xyz': xyz_world
        }

class StereoCamera:
    """Stereo camera pair with relative transformation."""
    def __init__(self,
                 left_intrinsics: CameraIntrinsics,
                 right_intrinsics: CameraIntrinsics,
                 baseline: float = 6.0,  # mm
                 relative_rotation: Optional[np.ndarray] = None):
        """
        Initialize stereo camera pair.
        
        Args:
            left_intrinsics: Left camera intrinsics
            right_intrinsics: Right camera intrinsics
            baseline: Stereo baseline in mm
            relative_rotation: 3x3 rotation matrix for right camera relative to left
        """
        # Create cameras
        self.left_camera = Camera(left_intrinsics, name="left")
        self.right_camera = Camera(right_intrinsics, name="right")
        
        self.baseline = baseline
        
        # Default relative rotation (identity if none provided)
        self.relative_rotation = relative_rotation if relative_rotation is not None else np.eye(3)
            
    def set_left_pose(self, position: np.ndarray, rotation: np.ndarray):
        """
        Set the pose of the left camera and update right camera accordingly.
        
        Args:
            position: 3D position vector [x, y, z] in mm
            rotation: 3x3 rotation matrix
        """
        # Set left camera pose
        self.left_camera.set_pose(position, rotation)
        
        # Create relative transformation matrix
        relative_transform = np.eye(4)
        relative_transform[:3, :3] = self.relative_rotation
        relative_transform[0, 3] = self.baseline  # Translation along X axis
        
        # Right camera pose = left camera pose * relative transform
        right_pose = self.left_camera.pose @ relative_transform
        
        # Extract rotation and position for right camera
        right_rotation = right_pose[:3, :3]
        right_position = right_pose[:3, 3]
        
        # Set right camera pose
        self.right_camera.set_pose(right_position, right_rotation)
        
    def capture_stereo_images(self, scene: o3d.geometry.TriangleMesh) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Capture images from both cameras.
        
        Args:
            scene: Open3D triangle mesh to render
            
        Returns:
            Dictionary containing left and right camera images:
            {
                'left': {'rgb': ..., 'depth': ..., 'xyz': ...},
                'right': {'rgb': ..., 'depth': ..., 'xyz': ...}
            }
        """
        return {
            'left': self.left_camera.capture_image(scene),
            'right': self.right_camera.capture_image(scene)
        }
        
    def get_camera_parameters(self, is_left: bool = True) -> o3d.camera.PinholeCameraParameters:
        """
        Get Open3D camera parameters for rendering.
        
        Args:
            is_left: Whether to get parameters for left (True) or right (False) camera
        """
        camera = self.left_camera if is_left else self.right_camera
        return camera.get_camera_parameters()
        
    def get_intrinsics(self, is_left: bool = True) -> CameraIntrinsics:
        """
        Get camera intrinsics.
        
        Args:
            is_left: Whether to get intrinsics for left (True) or right (False) camera
        """
        camera = self.left_camera if is_left else self.right_camera
        return camera.intrinsics 

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------

    def capture_from_left_pose(self,
                               position: np.ndarray,
                               rotation: np.ndarray,
                               scene: o3d.geometry.TriangleMesh) -> Dict[str, Dict[str, np.ndarray]]:
        """Set the left-camera pose and immediately capture a stereo pair.

        Args:
            position: 3-vector (mm) for left-camera centre.
            rotation: 3×3 rotation matrix for left camera.
            scene: Mesh to render.

        Returns:
            Stereo data dict as for `capture_stereo_images`.
        """
        self.set_left_pose(position, rotation)
        return self.capture_stereo_images(scene) 