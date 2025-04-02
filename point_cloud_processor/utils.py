#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np
import os
import pyvista as pv


def generate_sphere_point_cloud(radius=1.0, num_points=10000):
    """Generate a point cloud representing a sphere."""
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.arccos(np.random.uniform(-1, 1, num_points))
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    points = np.stack((x, y, z), axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def generate_tetrahedron_point_cloud(num_points=10000):
    """Generate a point cloud representing a pyramid (tetrahedron)."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 1]
    ])
    faces = [[0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 2, 1]]
    points_per_face = num_points // 4
    all_points = []

    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        u = np.random.uniform(0, 1, points_per_face)
        v = np.random.uniform(0, 1, points_per_face)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - u - v
        points = v0 * w[:, np.newaxis] + v1 * u[:, np.newaxis] + v2 * v[:, np.newaxis]
        all_points.append(points)

    points = np.vstack(all_points)
    if len(points) > num_points:
        points = points[:num_points]
    elif len(points) < num_points:
        extra = num_points - len(points)
        points = np.vstack([points, points[:extra]])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def load_point_cloud(file_path):
    """Load point cloud from file."""
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
    return pcd


def orient_normals(pcd, k=30):
    """Estimate and orient normals."""
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=k))
    pcd.orient_normals_consistent_tangent_plane(k=k)
    return pcd


def poisson_reconstruction(pcd, depth=8):
    """Perform Poisson surface reconstruction."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    mesh.compute_vertex_normals()
    return mesh


def ball_pivoting_reconstruction(pcd, radii=[1.0, 2.0, 4.0, 8.0, 16.0]):
    """Perform Ball Pivoting surface reconstruction."""
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [r * avg_dist for r in radii]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    mesh.compute_vertex_normals()
    return mesh


def o3d_to_pyvista(o3d_mesh):
    """Convert Open3D mesh to PyVista mesh for visualization."""
    if isinstance(o3d_mesh, o3d.geometry.PointCloud):
        vertices = np.asarray(o3d_mesh.points)
        mesh = pv.PolyData(vertices)
        if o3d_mesh.has_normals():
            mesh.point_data["Normals"] = np.asarray(o3d_mesh.normals)
        if o3d_mesh.has_colors():
            colors = np.asarray(o3d_mesh.colors)
            if colors.shape[1] == 3:  # RGB format
                mesh.point_data["Colors"] = colors * 255
    else:
        vertices = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        # Add triangle count to faces array for PyVista format
        faces_pv = np.zeros((len(faces), 4), dtype=np.int64)
        faces_pv[:, 0] = 3  # Triangle count (always 3 for triangles)
        faces_pv[:, 1:4] = faces
        faces_pv = faces_pv.flatten()
        mesh = pv.PolyData(vertices, faces_pv)
        if o3d_mesh.has_vertex_normals():
            mesh.point_data["Normals"] = np.asarray(o3d_mesh.vertex_normals)
    return mesh
