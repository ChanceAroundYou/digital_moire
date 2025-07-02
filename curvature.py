import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import PolyCollection

from utils.logger import add_file_handler, logger
from utils.mesh import get_output_path, load_mesh, save_img


def calculate_curvature(mesh, curv_type="mean"):
    """Calculates the Gaussian curvature for each vertex using the efficient PyVista library."""
    # Convert open3d mesh to pyvista mesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if len(faces) == 0:
        logger.warning("Mesh has no triangles, cannot compute curvature.")
        return None

    # PyVista requires faces in a specific format: [n_points, p1, p2, p3, ...]
    faces_pyvista = np.hstack((np.full((len(faces), 1), 3), faces))
    pv_mesh = pv.PolyData(vertices, faces_pyvista)
    curvatures = pv_mesh.curvature(curv_type=curv_type)
    return curvatures


def visualize_curvature(mesh, curvatures, clip_range, output_path=None):
    """Creates and saves a 2D plot of mesh faces colored by vertex curvatures."""

    faces = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    if len(faces) == 0:
        logger.warning("Mesh has no triangles, cannot compute curvature.")
        return np.zeros(len(vertices))

    # Calculate per-face curvature by averaging vertex curvatures
    face_curvatures = np.mean(curvatures[faces], axis=1)

    original_range = (np.min(face_curvatures), np.max(face_curvatures))
    logger.info(f"Original face curvature range: {original_range}")

    if clip_range is not None:
        low_p, high_p = clip_range
        face_curvatures = np.clip(face_curvatures, low_p, high_p)
        logger.info(f"Clipping range: {clip_range} for color mapping.")

    min_c = np.min(face_curvatures)
    max_c = np.max(face_curvatures)

    if (max_c - min_c) > 1e-9:
        face_curvatures = (face_curvatures - min_c) / (max_c - min_c)
    else:
        logger.warning("Curvature range is too small, setting all curvatures to 0.")
        face_curvatures = np.zeros_like(face_curvatures)

    cmap = plt.get_cmap("jet")
    colors = cmap(face_curvatures)

    try:
        logger.info("Creating 2D face plot")
        fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")

        # Project vertices onto XY plane for each triangle
        polygons = vertices[faces][:, :, :2]

        # Set edge colors to match face colors (instead of 'none')
        poly_collection = PolyCollection(polygons, edgecolors=colors)
        poly_collection.set_facecolor(colors)

        ax.add_collection(poly_collection)
        ax.autoscale_view()

        # Configure axis appearance
        ax.set_aspect("equal")
        ax.axis("off")

        if output_path is not None:
            save_img(fig, output_path)

    except Exception as e:
        logger.error(f"Failed to create or save the plot: {e}")


def get_curvature_img(
    ply_path, clip_range=None, curv_type="mean", is_save=True, output_path=None
):
    """
    Main function to load a mesh, compute its curvature, and save a visualization.

    Args:
        file_path (str): Path to the input .ply file.
    """

    mesh = load_mesh(ply_path)
    curvatures = calculate_curvature(mesh, curv_type=curv_type)

    if output_path is None and is_save:
        output_path = get_output_path(ply_path, output_type=curv_type)

    visualize_curvature(
        mesh,
        curvatures,
        clip_range=clip_range,
        output_path=output_path,
    )


def get_curvature_imgs(
    directory="data",
    pattern=r".*\.ply$",
    clip_range=None,
    curv_type="mean",
    is_save=True,
):
    """
    Process multiple PLY files in a directory.

    Args:
        directory (str): Directory to search for PLY files
        pattern (str): Regex pattern to match files
    """
    file_pattern = re.compile(pattern)

    for root, _, files in os.walk(directory):
        for file in files:
            if file_pattern.match(file):
                file_path = os.path.join(root, file)
                get_curvature_img(
                    file_path,
                    clip_range=clip_range,
                    curv_type=curv_type,
                    is_save=is_save
                )


if __name__ == "__main__":
    add_file_handler()
    # ply_path = "data/21-10332/STD_fuse_mesh.ply"
    # ply_path = "data/16-10363/STD_fuse_mesh.ply"
    # ply_path = "data/21-10282/STD_fuse_mesh.ply"
    # ply_path = "data/23-10130/STD_fuse_mesh.ply"
    # ply_path = "data/22-10228/STD_fuse_mesh.ply"
    ply_path = "data/23-10673/STD_fuse_mesh.ply"
    get_curvature_img(
        ply_path,
        clip_range=(-0.015, 0.025),
        curv_type="mean",
    )
    get_curvature_img(
        ply_path,
        clip_range=(-0.00025, 0.00025),
        curv_type="gaussian",
    )
