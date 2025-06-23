import numpy as np
import open3d as o3d
import os
from src.core.mesh import Mesh

def create_jet_color_map(values, vmin=None, vmax=None):
    """Create jet colormap for given values.
    
    Args:
        values: Array of values to map to colors
        vmin: Minimum value for color mapping (if None, uses min of values)
        vmax: Maximum value for color mapping (if None, uses max of values)
        
    Returns:
        Array of RGB colors matching jet colormap
    """
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)
    
    # Normalize values to [0,1]
    normalized = (values - vmin) / (vmax - vmin)
    
    # Create jet colormap
    colors = np.zeros((len(values), 3))
    
    # Red component
    colors[:, 0] = np.minimum(4 * normalized - 1.5, -4 * normalized + 4.5)
    
    # Green component
    colors[:, 1] = np.minimum(4 * normalized - 0.5, -4 * normalized + 3.5)
    
    # Blue component
    colors[:, 2] = np.minimum(4 * normalized + 0.5, -4 * normalized + 2.5)
    
    # Clip to [0,1]
    colors = np.clip(colors, 0, 1)
    
    return colors

def main():
    # Load mesh
    mesh_path = "data/Hand.ply"
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at {mesh_path}")
        return
        
    mesh = Mesh(mesh_path)
    if not mesh.mesh:
        print("Error: Failed to load mesh")
        return
    
    # Get vertices and compute normals if not present
    vertices = np.asarray(mesh.mesh.vertices)
    if not mesh.mesh.has_vertex_normals():
        mesh.mesh.compute_vertex_normals()
    
    # Calculate distances from origin for coloring
    distances = np.linalg.norm(vertices, axis=1)
    
    # Create jet colormap based on distances
    colors = create_jet_color_map(distances)
    
    # Create a new textured mesh
    textured_mesh = o3d.geometry.TriangleMesh()
    textured_mesh.vertices = mesh.mesh.vertices
    textured_mesh.triangles = mesh.mesh.triangles
    textured_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    textured_mesh.vertex_normals = mesh.mesh.vertex_normals
    
    # Ensure normals are oriented consistently
    textured_mesh.orient_triangles()
    textured_mesh.compute_vertex_normals()
    
    # Optional: Subdivide mesh for smoother appearance
    textured_mesh = textured_mesh.subdivide_midpoint(number_of_iterations=1)
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Textured Mesh Visualization", width=1280, height=720)
    
    # Add mesh
    vis.add_geometry(textured_mesh)
    
    # Set visualization options
    opt = vis.get_render_option()
    opt.mesh_show_back_face = False
    opt.light_on = True
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray background
    opt.point_size = 1.0
    
    # Create material for better rendering
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.base_roughness = 0.5
    material.base_reflectance = 0.8
    material.base_clearcoat = 0.2
    material.thickness = 1.0
    
    # Apply material to mesh
    textured_mesh.paint_uniform_color([1.0, 1.0, 1.0])  # Base color for material
    
    # Get view control
    vc = vis.get_view_control()
    
    # Set initial camera viewpoint
    cam = vc.convert_to_pinhole_camera_parameters()
    cam.extrinsic = np.array([
        [ 0.95892427, -0.12940952,  0.25115208,  73.83343506],
        [ 0.28318672,  0.55557023, -0.78177642, -292.47940063],
        [-0.01474465,  0.82139076,  0.57013515,  433.22457886],
        [ 0.,          0.,          0.,           1.        ]
    ])
    vc.convert_from_pinhole_camera_parameters(cam)
    
    # Run visualization
    print("\nVisualization controls:")
    print("- Left click + drag: Rotate")
    print("- Right click + drag: Pan")
    print("- Mouse wheel: Zoom")
    print("- Shift + Left click: Rotate around model")
    print("\nMesh colors show distance from origin using jet colormap:")
    print("- Blue: Closest points")
    print("- Green/Yellow: Medium distance")
    print("- Red: Farthest points")
    print("\nPress 'Q' or 'Escape' to exit...")
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main() 