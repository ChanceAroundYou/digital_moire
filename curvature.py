import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import PolyCollection

from utils.logger import add_file_handler, logger
from utils.ply_utils import build_mesh, load_mesh


def get_output_path(file_path, output_dir=os.path.join("out", "img")):
    """
    Generate output image path based on input PLY file path.

    Args:
        file_path (str): Path to the input PLY file
        output_dir (str): Directory to save output files

    Returns:
        str: Path for the output image file
    """
    gaussian_output_dir = os.path.join(output_dir, "gaussian")
    mean_output_dir = os.path.join(output_dir, "mean")
    os.makedirs(gaussian_output_dir, exist_ok=True)
    os.makedirs(mean_output_dir, exist_ok=True)

    # Try to extract information from the path
    path_match = re.match(r".*/(\d{2}-\d{5})/(STD|ATR)_([^.]+)\.ply", file_path)

    if path_match:
        project_id = path_match.group(1)
        scan_type = path_match.group(2)
        file_type = path_match.group(3)
        gaussian_output_path = os.path.join(
            gaussian_output_dir, f"{project_id}_{scan_type}_{file_type}.png"
        )
        mean_output_path = os.path.join(
            mean_output_dir, f"{project_id}_{scan_type}_{file_type}.png"
        )
    else:
        # Use filename as fallback
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        gaussian_output_path = os.path.join(gaussian_output_dir, f"{base_name}.png")
        mean_output_path = os.path.join(mean_output_dir, f"{base_name}.png")

    return gaussian_output_path, mean_output_path


def calculate_curvature(mesh, curv_type="mean"):
    """Calculates the Gaussian curvature for each vertex using the efficient PyVista library."""
    # Convert open3d mesh to pyvista mesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if len(faces) == 0:
        logger.warning("Mesh has no triangles, cannot compute curvature.")
        return np.zeros(len(vertices))

    # PyVista requires faces in a specific format: [n_points, p1, p2, p3, ...]
    faces_pyvista = np.hstack((np.full((len(faces), 1), 3), faces))
    pv_mesh = pv.PolyData(vertices, faces_pyvista)
    curvatures = pv_mesh.curvature(curv_type=curv_type)
    # pv_mesh.plot_curvature()
    return curvatures


def visualize_curvature(mesh, curvatures, clip_range, output_path):
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

    curvatures_to_plot = face_curvatures
    if clip_range is not None:
        low_p, high_p = clip_range
        curvatures_to_plot = np.clip(face_curvatures, low_p, high_p)
        logger.info(f"Clipping range: {clip_range} for color mapping.")

    min_c = np.min(curvatures_to_plot)
    max_c = np.max(curvatures_to_plot)

    if (max_c - min_c) > 1e-9:
        normalized_curvatures = (curvatures_to_plot - min_c) / (max_c - min_c)
    else:
        normalized_curvatures = np.zeros_like(curvatures_to_plot)

    cmap = plt.get_cmap("jet")
    colors = cmap(normalized_curvatures)

    logger.info(f"Creating 2D face plot and saving to '{output_path}'...")
    try:
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
        plt.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="black",
            dpi=300,
        )
        plt.close(fig)
        logger.success(f"Saved 2D curvature visualization to {output_path}.")
    except Exception as e:
        logger.error(f"Failed to create or save the plot: {e}")


def view_curvature(ply_path, gaussian_clip_range=None, mean_clip_range=None):
    """
    Main function to load a mesh, compute its curvature, and save a visualization.

    Args:
        file_path (str): Path to the input .ply file.
    """
    mesh, vertices, normals = load_mesh(ply_path)

    if not mesh.has_triangles():
        mesh = build_mesh(vertices, normals)

    if len(mesh.vertices) == 0:
        logger.error("Error: No vertices found in the final mesh. Aborting.")
        return
    gaussian_output_path, mean_output_path = get_output_path(ply_path)

    gaussian_curvatures = calculate_curvature(mesh, curv_type="gaussian")
    mean_curvatures = calculate_curvature(mesh, curv_type="mean")

    visualize_curvature(
        mesh,
        gaussian_curvatures,
        clip_range=gaussian_clip_range,
        output_path=gaussian_output_path,
    )
    visualize_curvature(
        mesh,
        mean_curvatures,
        clip_range=mean_clip_range,
        output_path=mean_output_path,
    )


if __name__ == "__main__":
    add_file_handler()
    # ply_path = "data/21-10332/STD_mesh.ply"
    # ply_path = "data/16-10363/STD_mesh.ply"
    # ply_path = "data/21-10282/STD_mesh.ply"
    # ply_path = "data/23-10130/STD_mesh.ply"
    # ply_path = "data/22-10228/STD_mesh.ply"
    ply_path = "data/23-10673/STD_mesh.ply"
    view_curvature(
        ply_path,
        gaussian_clip_range=(-0.00025, 0.00025),
        mean_clip_range=(-0.015, 0.025),
    )
