import os
import tempfile

import numpy as np

from moire import visualize_moire
from utils.logger import add_file_handler, logger
from utils.mesh import (
    apply_rotation,
    calculate_distance_from_plane,
    get_output_path,
    load_mesh,
    save_gif,
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


def create_rotation_animation(
    file_path,
    output_gif=None,
    is_save=True,
    frame_num=36,
    rotation_axis="y",
    duration=0.1,
    initial_x_angle=0,
    initial_y_angle=0,
    initial_z_angle=0,
    plane_a=0,
    plane_b=0,
    plane_c=1,
    plane_d=1,
    num_levels=100,
    background_color="gray",
):
    """Create an animation of the PLY model rotating around specified axis"""
    logger.info(
        f"Creating animation with initial rotation: X={initial_x_angle}°, Y={initial_y_angle}°, Z={initial_z_angle}°"
    )

    # Create a temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    mesh = load_mesh(file_path, is_build_mesh=False)

    # Create initial rotation matrices
    initial_mesh = apply_rotation(mesh, initial_x_angle, initial_y_angle, initial_z_angle)
    distances = calculate_distance_from_plane(initial_mesh, plane_a, plane_b, plane_c, plane_d)
    d_min, d_max = distances.min(), distances.max()
    fixed_levels = np.linspace(d_min, d_max, num_levels)

    # Generate frames for different rotation angles
    frame_files = []
    for i in range(frame_num):
        angle = get_angle(i, frame_num)
        # Create rotation matrix based on selected axis
        if rotation_axis == "x":
            rotated_mesh = apply_rotation(initial_mesh, x=angle)
        elif rotation_axis == "y":
            rotated_mesh = apply_rotation(initial_mesh, y=angle)
        elif rotation_axis == "z":
            rotated_mesh = apply_rotation(initial_mesh, z=angle)

        frame_path = os.path.join(temp_dir, f".frame_{i:03d}.png")
        frame_files.append(frame_path)
        visualize_moire(
            rotated_mesh,
            distances,
            levels=fixed_levels,
            num_levels=num_levels,
            background_color=background_color,
            output_path=frame_path,
            bbox_inches=None,
        )

    if output_gif is None and is_save:
        output_gif = get_output_path(file_path, output_type="gif", file_type="gif")

    if is_save:
        save_gif(frame_files, output_gif, duration)

    os.rmdir(temp_dir)


if __name__ == "__main__":
    add_file_handler()
    ply_path = "data/16-10072/STD_fuse.ply"
    create_rotation_animation(ply_path)
