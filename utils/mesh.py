import os
import re

import imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from utils.logger import logger


def get_output_path(
    file_path,
    output_type="",
    output_dir="out",
    file_type="png",
):
    """
    Generate output path based on input PLY file path.

    Args:
        file_path (str): Path to the input PLY file
        output_type (str): Type of output file
        output_dir (str): Directory to save output files
        output_file_type (str): Type of output file

    Returns:
        str: Path for the output file
    """
    if output_type:
        output_dir = os.path.join(output_dir, output_type)

    os.makedirs(output_dir, exist_ok=True)

    path_match = re.match(r".*/(.*)/([^.]+)\.ply", file_path)

    if path_match:
        folder_name = path_match.group(1)
        file_name = path_match.group(2)
        output_path = os.path.join(output_dir, f"{folder_name}_{file_name}.{file_type}")
    else:
        # Use filename as fallback
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.{file_type}")

    return output_path


def build_mesh(mesh: o3d.geometry.TriangleMesh):
    """Reconstructs a mesh from a point cloud using Poisson surface reconstruction."""
    logger.info(
        "Input is a point cloud. Reconstructing a mesh for curvature analysis..."
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals

    pcd.orient_normals_consistent_tangent_plane(100)

    logger.info("Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8
    )

    logger.info("Cropping mesh based on density to remove artifacts...")
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh


def load_mesh(file_path, is_build_mesh=True):
    """
    Load a mesh from a PLY file.

    Args:
        file_path (str): Path to the PLY file

    Returns:
        mesh: TriangleMesh
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(file_path)

    if mesh.is_empty():
        logger.error("Error: No vertices found in the final mesh. Aborting.")
        raise ValueError("No vertices found in the final mesh. Aborting.")

    if not mesh.has_triangles() and is_build_mesh:
        logger.info(
            "No triangles found in the mesh. Building a mesh from the point cloud..."
        )
        mesh = build_mesh(mesh)
    return mesh


def apply_rotation(mesh: o3d.geometry.TriangleMesh, x=0, y=0, z=0, in_degrees=True):
    """
    Apply rotation matrices to vertices and normals.

    Args:
        mesh: Mesh to rotate
        x, y, z: Rotation angles in degrees or radians
        in_degrees: Whether the angles are in degrees

    Returns:
        rotated_mesh
    """
    if in_degrees:
        x = np.radians(x)
        y = np.radians(y)
        z = np.radians(z)

    rotation_matrix = mesh.get_rotation_matrix_from_xyz([x, y, z])
    mesh.rotate(rotation_matrix)
    return mesh


def calculate_distance_from_plane(
    mesh: o3d.geometry.TriangleMesh, plane_a=0, plane_b=0, plane_c=1, plane_d=0
):
    """
    Calculate distances from vertices to a plane.

    Args:
        vertices (np.ndarray): Mesh vertices
        a, b, c, d: Plane equation coefficients (ax + by + cz + d = 0)

    Returns:
        np.ndarray: Array of distances from each vertex to the plane
    """
    vertices = np.asarray(mesh.vertices)
    normal_vector = np.array([plane_a, plane_b, plane_c])
    denominator = np.linalg.norm(normal_vector)
    # Avoid division by zero
    if denominator == 0:
        logger.warning("Plane normal vector is zero. Using default values.")
        normal_vector = np.array([0, 0, 1])
        denominator = 1

    # Normalize the normal vector for more efficient computation
    unit_normal = normal_vector / denominator

    # Vectorized computation using broadcasting
    # This avoids explicit dot product calculation for each vertex
    distances = np.abs(np.sum(vertices * unit_normal, axis=1) + plane_d / denominator)
    return distances


def save_img(fig, image_path, dpi=500, pad_inches=0, bbox_inches="tight"):
    """
    Save the visualization to an image file.

    Args:
        fig: Matplotlib figure
        image_path (str): Path where to save the image
        dpi (int): DPI for the output image
        pad_inches (float): Padding around the figure
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(image_path)) or ".", exist_ok=True)

    # Save the figure
    plt.savefig(
        image_path,
        dpi=dpi,
        pad_inches=pad_inches,
        bbox_inches=bbox_inches,
    )
    plt.close(fig)
    logger.success(f"Saved image to {image_path}")


def save_gif(frame_files, gif_path, duration=0.1):
    """
    Save a GIF animation from a list of image files.

    Args:
        frame_files (list): List of image file paths
        gif_path (str): Path where to save the GIF file
        duration (float): Duration of each frame in seconds
    """
    os.makedirs(os.path.dirname(os.path.abspath(gif_path)) or ".", exist_ok=True)

    images = []
    for file in frame_files:
        if os.path.exists(file):
            images.append(imageio.v2.imread(file))
            os.remove(file)

    imageio.mimsave(gif_path, images, duration=duration)
    logger.success(f"Animation saved as {gif_path}")
