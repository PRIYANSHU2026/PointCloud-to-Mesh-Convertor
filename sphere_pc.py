import open3d as o3d
import numpy as np


def generate_sphere_point_cloud(radius=1.0, num_points=10000):
    """Generate a point cloud representing a sphere."""
    # Generate random spherical coordinates
    phi = np.random.uniform(0, 2 * np.pi, num_points)  # Azimuthal angle
    theta = np.arccos(np.random.uniform(-1, 1, num_points))  # Polar angle

    # Convert spherical to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Create point cloud
    points = np.stack((x, y, z), axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Optional: Add some noise to simulate real data
    noise = np.random.normal(0, 0.0, points.shape)  # Small noise
    pcd.points = o3d.utility.Vector3dVector(points + noise)

    return pcd


# Generate and save the sphere
pcd = generate_sphere_point_cloud(radius=1.0, num_points=10000)
output_path = "/Users/srujanraj/Downloads/sphere.pcd"
o3d.io.write_point_cloud(output_path, pcd)
print(f"Saved sphere point cloud with {len(pcd.points)} points to {output_path}")

# Optional: Visualize to confirm
o3d.visualization.draw_geometries([pcd], window_name="Sphere Point Cloud")