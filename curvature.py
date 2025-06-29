import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

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
    faces_pyvista = np.hstack((np.full((faces.shape[0], 1), 3), faces))
    pv_mesh = pv.PolyData(vertices, faces_pyvista)
    curvatures = pv_mesh.curvature(curv_type=curv_type)
    # pv_mesh.plot_curvature()
    return curvatures


def visualize_curvature(vertices, curvatures, clip_range, output_path):
    """Creates and saves a 2D scatter plot visualizing vertex curvatures."""
    original_range = np.min(curvatures), np.max(curvatures)
    logger.info(f"Original curvature range: {original_range}")

    if clip_range is None:
        # low_p, high_p = original_range
        pass
    else:
        low_p, high_p = clip_range
        curvatures = np.clip(curvatures, low_p, high_p)
        logger.info(f"Clipping range: {clip_range} for color mapping.")

    min_c = np.min(curvatures)
    max_c = np.max(curvatures)
    curvatures = (curvatures - min_c) / (max_c - min_c)

    cmap = plt.get_cmap("jet")
    colors = cmap(curvatures)[:, :3]

    logger.info(f"Creating 2D scatter plot and saving to '{output_path}'...")
    try:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
        ax.scatter(vertices[:, 0], vertices[:, 1], c=colors, s=1, alpha=0.8)
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
        
    final_vertices = np.asarray(mesh.vertices)
    if len(final_vertices) == 0:
        logger.error("Error: No vertices found in the final mesh. Aborting.")
        return
    gaussian_output_path, mean_output_path = get_output_path(ply_path)

    gaussian_curvatures = calculate_curvature(mesh, curv_type="gaussian")
    mean_curvatures = calculate_curvature(mesh, curv_type="mean")

    visualize_curvature(
        final_vertices,
        gaussian_curvatures,
        clip_range=gaussian_clip_range,
        output_path=gaussian_output_path,
    )
    visualize_curvature(
        final_vertices,
        mean_curvatures,
        clip_range=mean_clip_range,
        output_path=mean_output_path,
    )


if __name__ == "__main__":
    add_file_handler()
    ply_path = "data/21-10332/STD_mesh.ply"
    view_curvature(
        ply_path,
        gaussian_clip_range=(-0.00045, 0.00035),
        mean_clip_range=(-0.015, 0.02),
    )
