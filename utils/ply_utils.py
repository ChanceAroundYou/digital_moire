import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from utils.logger import logger


def load_mesh(file_path):
    """
    Load a mesh from a PLY file.

    Args:
        file_path (str): Path to the PLY file

    Returns:
        tuple: (mesh, vertices, normals) arrays
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    mesh = o3d.io.read_triangle_mesh(file_path)
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    return mesh, vertices, normals


def create_rotation_matrices(x_angle=0, y_angle=0, z_angle=0):
    """
    Create rotation matrices for the given angles.

    Args:
        x_angle (float): Rotation angle around X axis in degrees
        y_angle (float): Rotation angle around Y axis in degrees
        z_angle (float): Rotation angle around Z axis in degrees

    Returns:
        tuple: (Rx, Ry, Rz) rotation matrices
    """
    # Convert to radians
    x_rad = np.radians(x_angle)
    y_rad = np.radians(y_angle)
    z_rad = np.radians(z_angle)

    # X-axis rotation matrix
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(x_rad), -np.sin(x_rad)],
            [0, np.sin(x_rad), np.cos(x_rad)],
        ]
    )

    # Y-axis rotation matrix
    Ry = np.array(
        [
            [np.cos(y_rad), 0, np.sin(y_rad)],
            [0, 1, 0],
            [-np.sin(y_rad), 0, np.cos(y_rad)],
        ]
    )

    # Z-axis rotation matrix
    Rz = np.array(
        [
            [np.cos(z_rad), -np.sin(z_rad), 0],
            [np.sin(z_rad), np.cos(z_rad), 0],
            [0, 0, 1],
        ]
    )

    return Rx, Ry, Rz


def apply_rotation(matrix, Rx=None, Ry=None, Rz=None):
    """
    Apply rotation matrices to vertices and normals.

    Args:
        matrix: Matrix to rotate
        Rx, Ry, Rz: Rotation matrices

    Returns:
        rotated_matrix
    """
    if Rx is not None:
        matrix = matrix @ Rx.T
    if Ry is not None:
        matrix = matrix @ Ry.T
    if Rz is not None:
        matrix = matrix @ Rz.T
    return matrix


def calculate_plane_distances(vertices, plane_a=0, plane_b=0, plane_c=1, plane_d=0):
    """
    Calculate distances from vertices to a plane.

    Args:
        vertices (np.ndarray): Mesh vertices
        a, b, c, d: Plane equation coefficients (ax + by + cz + d = 0)

    Returns:
        np.ndarray: Array of distances from each vertex to the plane
    """
    denominator = np.sqrt(plane_a**2 + plane_b**2 + plane_c**2)
    # Avoid division by zero
    if denominator == 0:
        logger.warning("Plane normal vector is zero. Using default values.")
        return np.zeros_like(vertices[:, 0])

    distances = (
        np.abs(
            plane_a * vertices[:, 0]
            + plane_b * vertices[:, 1]
            + plane_c * vertices[:, 2]
            + plane_d
        )
        / denominator
    )
    return distances


def save_img(fig, image_path, dpi=300, pad_inches=2):
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
    plt.savefig(image_path, dpi=dpi, pad_inches=pad_inches)
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

    images = [imageio.v2.imread(file) for file in frame_files]
    imageio.mimsave(gif_path, images, duration=duration)
    logger.success(f"Animation saved as {gif_path}")

    # Clean up temporary files
    for file in frame_files:
        os.remove(file)


def build_mesh(vertices, normals):
    """Reconstructs a mesh from a point cloud using Poisson surface reconstruction."""
    logger.info(
        "Input is a point cloud. Reconstructing a mesh for curvature analysis..."
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    if normals is None or len(normals) == 0:
        logger.info("No normals found, estimating normals for reconstruction...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    pcd.orient_normals_consistent_tangent_plane(100)

    logger.info("Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8
    )

    logger.info("Cropping mesh based on density to remove artifacts...")
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh
