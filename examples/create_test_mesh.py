import open3d as o3d
import numpy as np

def create_cube_mesh(size: float = 100.0) -> o3d.geometry.TriangleMesh:
    """Create a cube mesh with given size."""
    # Create vertices
    vertices = np.array([
        [0, 0, 0],  # 0
        [size, 0, 0],  # 1
        [size, size, 0],  # 2
        [0, size, 0],  # 3
        [0, 0, size],  # 4
        [size, 0, size],  # 5
        [size, size, size],  # 6
        [0, size, size],  # 7
    ])
    
    # Create triangles (each face has 2 triangles)
    triangles = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 2, 6], [1, 6, 5],  # right
    ])
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Compute vertex normals
    mesh.compute_vertex_normals()
    
    return mesh

def main():
    # Create output directory
    import os
    os.makedirs("data/processed/visualization", exist_ok=True)
    
    # Create cube mesh
    mesh = create_cube_mesh(size=100.0)  # 100mm cube
    
    # Save mesh
    output_path = "data/processed/visualization/mesh.ply"
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Saved test mesh to {output_path}")

if __name__ == "__main__":
    main() 