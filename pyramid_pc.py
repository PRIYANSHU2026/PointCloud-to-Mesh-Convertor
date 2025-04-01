import open3d as o3d
import numpy as np


def generate_tetrahedron_point_cloud(num_points=10000):
    # Define tetrahedron vertices
    vertices = np.array([
        [0, 0, 0],  # Base vertex 0
        [1, 0, 0],  # Base vertex 1
        [0, 1, 0],  # Base vertex 2
        [0.5, 0.5, 1]  # Apex
    ])

    # Define the four faces (vertex indices)
    faces = [
        [0, 1, 3],  # Side 1: 0-1-3
        [1, 2, 3],  # Side 2: 1-2-3
        [2, 0, 3],  # Side 3: 2-0-3
        [0, 2, 1]  # Base: 0-2-1
    ]

    # Number of points per face
    points_per_face = num_points // 4
    all_points = []

    # Sample points on each face
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        # Random barycentric coordinates
        u = np.random.uniform(0, 1, points_per_face)
        v = np.random.uniform(0, 1, points_per_face)
        # Ensure points stay inside triangle (u + v <= 1)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - u - v
        # Compute points as weighted sum of vertices
        points = v0 * w[:, np.newaxis] + v1 * u[:, np.newaxis] + v2 * v[:, np.newaxis]
        all_points.append(points)

    # Combine all points and adjust to exact num_points
    points = np.vstack(all_points)
    if len(points) > num_points:
        points = points[:num_points]
    elif len(points) < num_points:
        extra = num_points - len(points)
        points = np.vstack([points, points[:extra]])

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # No noise (0.0) as requested
    return pcd


# Generate and save
pcd = generate_tetrahedron_point_cloud(num_points=10000)
output_path = "/Users/srujanraj/Downloads/pyramid.pcd"
o3d.io.write_point_cloud(output_path, pcd)
print(f"Saved pyramid with {len(pcd.points)} points to {output_path}")