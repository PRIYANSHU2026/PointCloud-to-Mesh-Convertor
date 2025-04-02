import open3d as o3d
import numpy as np
import os


def load_point_cloud(file_path):
    print(f"Attempting to load file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist!")
    if file_path.endswith('.xyz'):
        point_cloud = np.loadtxt(file_path, delimiter=' ')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
    elif file_path.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            raise ValueError(f"Failed to load {file_path}: no points found!")
    else:
        raise ValueError("Unsupported file format. Use .xyz or .pcd")
    print(f"Loaded point cloud with {len(pcd.points)} points")
    return pcd

def preprocess_point_cloud(pcd):
    points = np.asarray(pcd.points)
    center = points.mean(axis=0)
    points -= center
    scale = np.max(np.linalg.norm(points, axis=1))
    points /= scale
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    print(f"Preprocessed to {len(pcd.points)} points")
    return pcd

def orient_normals(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)
    print("Normals oriented.")
    return pcd


def poisson_reconstruction(pcd, depth=12):
    print("Running Poisson reconstruction with depth=", depth)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    mesh.compute_vertex_normals()
    colors = np.full((len(mesh.vertices), 3), [0.0, 0.0, 1.0])  # Blue
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    print(f"Poisson mesh has {len(mesh.triangles)} triangles")
    return mesh


def ball_pivoting_reconstruction(pcd, radii=[0.1, 0.2, 0.5, 1.0, 2.0, 4.0]):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print(f"Average nearest neighbor distance: {avg_dist}")
    radii = [r * avg_dist for r in radii]  # Scale radii by avg distance
    print(f"Using radii: {radii}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    mesh.compute_vertex_normals()
    print(f"Ball Pivoting mesh has {len(mesh.triangles)} triangles")
    return mesh


def visualize_geometry(geometry, name="Geometry"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name)
    vis.add_geometry(geometry)
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.light_on = True
    render_option.point_show_normal = True
    vis.get_view_control().set_zoom(0.8)
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_lookat([0, 0, 0])
    vis.get_view_control().set_up([0, 1, 0])
    print(f"Displaying {name}. Close the window to continue...")
    print("Shortcuts: W (wireframe), S (solid), L (toggle light), +/- (size), Q (quit)")
    vis.run()
    vis.destroy_window()


def main():
    file_path = "/Users/srujanraj/Downloads/sphere.pcd"
    print(f"Starting with file: {file_path}")

    try:
        pcd = load_point_cloud(file_path)
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return

    print("Visualizing raw point cloud...")
    visualize_geometry(pcd, "Raw Point Cloud")

    pcd = orient_normals(pcd)
    print("Visualizing point cloud with normals...")
    visualize_geometry(pcd, "Point Cloud with Normals")

    poisson_mesh = poisson_reconstruction(pcd, depth=8)
    if poisson_mesh.has_triangles():
        o3d.io.write_triangle_mesh("poisson_mesh.ply", poisson_mesh)
        print("Saved poisson_mesh.ply")
        visualize_geometry(poisson_mesh, "Poisson Mesh")
    else:
        print("Poisson reconstruction failed: no triangles generated.")

    ball_pivot_mesh = ball_pivoting_reconstruction(pcd)
    if ball_pivot_mesh.has_triangles():
        o3d.io.write_triangle_mesh("ball_pivot_mesh.ply", ball_pivot_mesh)
        print("Saved ball_pivot_mesh.ply")
        visualize_geometry(ball_pivot_mesh, "Ball Pivoting Mesh")
    else:
        print("Ball Pivoting reconstruction failed: no triangles generated.")


if __name__ == "__main__":
    main()
