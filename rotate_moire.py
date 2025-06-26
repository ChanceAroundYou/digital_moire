import os
import re
import tempfile

import numpy as np

from moire import create_moire_visualization
from utils.logger import add_file_handler, logger
from utils.ply_utils import (
    apply_rotation,
    calculate_plane_distances,
    create_rotation_matrices,
    load_mesh,
    save_gif,
    save_img,
)


def get_angle(frame, total_frame_num):
    """
    Calculate angle for this frame (full 360 degree rotation)
    First quarter: 0 to -45 degrees
    Second quarter: -45 to 45 degrees
    Third quarter: 45 back to 0 degrees
    """
    if frame < total_frame_num / 4:
        angle = -45 * (frame / (total_frame_num / 4))
    elif frame < 3 * total_frame_num / 4:
        angle = -45 + 90 * ((frame - total_frame_num / 4) / (total_frame_num / 2))
    else:
        angle = 45 - 45 * ((frame - 3 * total_frame_num / 4) / (total_frame_num / 4))
    return angle


def get_output_path(file_path, output_dir="out/gif"):
    """
    Generate output gif path based on input PLY file path.

    Args:
        file_path (str): Path to the input PLY file
        output_dir (str): Directory to save output files

    Returns:
        str: Path for the output gif file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Try to extract information from the path
    path_match = re.match(r".*/(\d{2}-\d{5})/(STD|ATR)_([^.]+)\.ply", file_path)

    if path_match:
        project_id = path_match.group(1)
        scan_type = path_match.group(2)
        file_type = path_match.group(3)
        output_path = f"{output_dir}/{project_id}_{scan_type}_{file_type}.gif"
    else:
        # Use filename as fallback
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f"{output_dir}/{base_name}.gif"

    return output_path


def create_rotation_animation(
    file_path,
    output_gif=None,
    frame_num=36,
    num_levels=100,
    initial_x_angle=0,
    initial_y_angle=0,
    initial_z_angle=0,
    rotation_axis="y",
    duration=0.1,
):
    """Create an animation of the PLY model rotating around specified axis"""
    if output_gif is None:
        output_gif = get_output_path(file_path)

    logger.info(
        f"Creating animation with initial rotation: X={initial_x_angle}°, Y={initial_y_angle}°, Z={initial_z_angle}°"
    )

    # Create a temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    _, vertices, _ = load_mesh(file_path)

    # Create initial rotation matrices
    Rx, Ry, Rz = create_rotation_matrices(
        initial_x_angle, initial_y_angle, initial_z_angle
    )
    vertices = apply_rotation(vertices, Rx, Ry, Rz)
    distances = calculate_plane_distances(vertices)
    d_min, d_max = distances.min(), distances.max()
    fixed_levels = np.linspace(d_min, d_max, num_levels)

    # Generate frames for different rotation angles
    frame_files = []
    for i in range(frame_num):
        angle = get_angle(i, frame_num)
        # Create rotation matrix based on selected axis
        if rotation_axis == "x":
            Rx, _, _ = create_rotation_matrices(angle, 0, 0)
            rotated_vertices = apply_rotation(vertices, Rx=Rx)
        elif rotation_axis == "y":
            _, Ry, _ = create_rotation_matrices(0, angle, 0)
            rotated_vertices = apply_rotation(vertices, Ry=Ry)
        elif rotation_axis == "z":
            _, _, Rz = create_rotation_matrices(0, 0, angle)
            rotated_vertices = apply_rotation(vertices, Rz=Rz)

        fig, _ = create_moire_visualization(
            rotated_vertices, distances, levels=fixed_levels
        )
        frame_path = os.path.join(temp_dir, f".frame_{i:03d}.png")
        frame_files.append(frame_path)
        save_img(fig, frame_path)

    save_gif(frame_files, output_gif, duration)
    os.rmdir(temp_dir)


if __name__ == "__main__":
    add_file_handler()
    ply_path = "data/16-10072/STD_point_cloud.ply"
    create_rotation_animation(ply_path)
