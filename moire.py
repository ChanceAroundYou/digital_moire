import os
import re

import matplotlib.pyplot as plt
import numpy as np

from utils.logger import add_file_handler, logger
from utils.mesh import (
    apply_rotation,
    calculate_distance_from_plane,
    get_output_path,
    load_mesh,
    save_img,
)


def visualize_moire(
    mesh,
    distances,
    levels=None,
    num_levels=100,
    background_color="gray",
    output_path=None,
    bbox_inches="tight"
):
    """
    Create a Moiré pattern visualization based on distance levels.

    Args:
        vertices (np.ndarray): Mesh vertices
        distances (np.ndarray): Distances from vertices to plane
        levels (np.ndarray, optional): Levels for the contour visualization
        num_levels (int): Number of levels for the contour visualization
        facecolor (str): Background color for the figure

    Returns:
        tuple: (fig, ax) matplotlib figure and axis
    """
    vertices = np.asarray(mesh.vertices)
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=background_color)

    # Calculate distance levels
    if levels is None:
        d_min, d_max = distances.min(), distances.max()
        levels = np.linspace(d_min, d_max, num_levels)

    # Draw contours as alternating black and white bands
    for i in range(len(levels) - 1):
        mask = (distances >= levels[i]) & (distances < levels[i + 1])
        color = "black" if i % 2 == 0 else "white"
        ax.scatter(vertices[mask, 0], vertices[mask, 1], c=color, s=1, alpha=0.5)

    # Configure axis appearance
    ax.set_aspect("equal")
    ax.axis("off")

    if output_path is not None:
        save_img(fig, output_path, bbox_inches=bbox_inches)


def get_moire_img(
    file_path,
    output_path=None,
    is_save=True,
    rotation_x_angle=0,
    rotation_y_angle=0,
    rotation_z_angle=0,
    plane_a=0,
    plane_b=0,
    plane_c=1,
    plane_d=20,
    num_levels=100,
    background_color="gray",
):
    """
    Load a PLY file and process it to create a Moiré pattern visualization.

    Args:
        file_path (str): Path to the PLY file
        image_path (str, optional): Path for the output image
        x_angle (float): Rotation angle around X axis in degrees
        y_angle (float): Rotation angle around Y axis in degrees
        z_angle (float): Rotation angle around Z axis in degrees
        a, b, c, d: Plane equation coefficients (ax + by + cz + d = 0)
        num_levels (int): Number of contour levels for the visualization
    """
    logger.info(f"Processing {file_path}")

    mesh = load_mesh(file_path, is_build_mesh=False)
    mesh = apply_rotation(mesh, rotation_x_angle, rotation_y_angle, rotation_z_angle)
    distances = calculate_distance_from_plane(mesh, plane_a, plane_b, plane_c, plane_d)

    if output_path is None and is_save:
        output_path = get_output_path(file_path, output_type="moire")

    visualize_moire(
        mesh,
        distances,
        num_levels=num_levels,
        background_color=background_color,
        output_path=output_path,
    )


def get_moire_imgs(
    directory="data",
    pattern=r".*\.ply$",
    is_save=True,
    rotation_x_angle=0,
    rotation_y_angle=0,
    rotation_z_angle=0,
    plane_a=0,
    plane_b=0,
    plane_c=1,
    plane_d=1,
    num_levels=100,
    background_color="gray",
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
                get_moire_img(
                    file_path,
                    is_save=is_save,
                    rotation_x_angle=rotation_x_angle,
                    rotation_y_angle=rotation_y_angle,
                    rotation_z_angle=rotation_z_angle,
                    plane_a=plane_a,
                    plane_b=plane_b,
                    plane_c=plane_c,
                    plane_d=plane_d,
                    num_levels=num_levels,
                    background_color=background_color,
                )


if __name__ == "__main__":
    # Process a single file
    # ply_path = "data/16-10072/STD_point_cloud.ply"
    # get_moire_img(ply_path, num_levels=200)

    # Uncomment to process all PLY files in the data directory
    add_file_handler()
    get_moire_imgs(
        directory="data",
        pattern=r".*mesh\.ply$",
        num_levels=100,
    )
