import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.collections import PolyCollection

from curvature import calculate_curvature
from utils.logger import add_file_handler, logger

# Reuse functions from the project's utils module
from utils.mesh import load_mesh


def clean_mesh_pipeline(
    ply_path: str,
    output_path: str,
    # Stage 1: Curvature thresholds
    clean_by_curvature: bool = True,
    curv_high_thresh: float = 0.05,
    curv_low_thresh: float = -0.1,
    # Stage 2: Curvature variance threshold
    clean_by_variance: bool = True,
    variance_thresh: float = 0.001,
    # Stage 3: Border removal
    clean_borders: bool = True,
    border_rings: int = 5,
    # Stage 4: Island removal
    remove_islands: bool = True,
):
    """
    A multi-stage pipeline to clean a 3D mesh with detailed visualization.
    """
    try:
        # --- PREPARATION ---
        logger.info(f"Loading mesh from {ply_path}...")
        mesh = load_mesh(ply_path)
        original_vertices = np.asarray(mesh.vertices)
        original_triangles = np.asarray(mesh.triangles)

        logger.info("Calculating mean curvature...")
        curvatures = calculate_curvature(mesh, "mean")
        if curvatures is None:
            return

        mesh.compute_adjacency_list()
        adjacency_list = [set(adj) for adj in mesh.adjacency_list]

        # --- COMPUTE REMOVAL MASKS FOR EACH STAGE ---

        # This array will store the reason for removal (0=kept, 1=curvature, 2=variance, etc.)
        removal_reason = np.zeros(len(original_vertices), dtype=int)

        # Stage 1: Curvature
        if clean_by_curvature:
            logger.info("Stage 1: Cleaning by absolute curvature...")
            mask = (curvatures > curv_high_thresh) | (curvatures < curv_low_thresh)
            removal_reason[mask] = 1
            logger.info(f"Marked {np.sum(mask)} vertices for 'Curvature'.")

        # Stage 2: Variance
        if clean_by_variance:
            logger.info("Stage 2: Cleaning by curvature variance...")
            variances = np.zeros_like(curvatures)
            for i in range(len(original_vertices)):
                neighbor_indices = list(adjacency_list[i])
                if len(neighbor_indices) > 0:
                    variances[i] = np.var(curvatures[neighbor_indices])
            mask = variances > variance_thresh
            removal_reason[mask & (removal_reason == 0)] = (
                2  # Only mark if not already marked
            )
            logger.info(
                f"Marked {np.sum(mask & (removal_reason == 0))} new vertices for 'Variance'."
            )

        # Stage 3: Borders
        if clean_borders:
            logger.info("Stage 3: Cleaning mesh borders...")
            border_mask = np.zeros_like(removal_reason, dtype=bool)
            boundary_vertices = mesh.get_non_manifold_vertices()
            for idx in boundary_vertices:
                border_mask[idx] = True

            for _ in range(border_rings):
                new_borders = np.copy(border_mask)
                for i, is_border in enumerate(border_mask):
                    if is_border:
                        for neighbor in adjacency_list[i]:
                            new_borders[neighbor] = True
                border_mask = new_borders

            removal_reason[border_mask & (removal_reason == 0)] = 3
            logger.info(
                f"Marked {np.sum(border_mask & (removal_reason == 0))} new vertices for 'Border'."
            )

        # --- VISUALIZATION & STAGE 4 (ISLANDS) ---
        logger.info("Preparing detailed visualization...")

        # Determine removal reason for each FACE
        # A face is removed if any of its vertices are marked for removal.
        face_removal_reason = np.max(removal_reason[original_triangles], axis=1)

        # Stage 4: Islands. This is a face-level operation.
        if remove_islands:
            logger.info("Stage 4: Identifying isolated islands...")
            preliminary_kept_faces = face_removal_reason == 0

            # Find clusters in the original mesh
            triangle_clusters, cluster_n_triangles, _ = (
                mesh.cluster_connected_triangles()
            )

            # Convert Open3D IntVector to numpy array to allow boolean mask indexing
            triangle_clusters_np = np.asarray(triangle_clusters)

            if len(cluster_n_triangles) > 0:
                # Find the largest cluster among the preliminarily kept faces
                kept_clusters = triangle_clusters_np[preliminary_kept_faces]
                if len(kept_clusters) > 0:
                    unique_clusters, counts = np.unique(
                        kept_clusters, return_counts=True
                    )
                    largest_kept_cluster_id = unique_clusters[np.argmax(counts)]

                    # Mark all faces not in this largest component as islands (reason 4)
                    island_mask = triangle_clusters_np != largest_kept_cluster_id
                    face_removal_reason[preliminary_kept_faces & island_mask] = 4
                    logger.info(
                        f"Marked {np.sum(preliminary_kept_faces & island_mask)} faces as 'Islands'."
                    )

        # Create color map for visualization
        color_map = {
            0: "#909090",  # Kept
            1: "#e63946",  # Red: Curvature
            2: "#f4a261",  # Orange: Variance
            3: "#e9c46a",  # Yellow: Border
            4: "#457b9d",  # Blue: Island
        }
        face_colors = [color_map[r] for r in face_removal_reason]

        fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")

        vertices_2d = original_vertices[:, :2]
        polygons = vertices_2d[original_triangles]

        collection = PolyCollection(
            polygons, facecolors=face_colors, edgecolors=face_colors, lw=0.1
        )
        ax.add_collection(collection)

        # Custom Legend
        legend_patches = [
            mpatches.Patch(color=color, label=label)
            for label, color in [
                ("Kept", color_map[0]),
                ("Removed: High Curvature", color_map[1]),
                ("Removed: High Variance", color_map[2]),
                ("Removed: Border Region", color_map[3]),
                ("Removed: Isolated Island", color_map[4]),
            ]
        ]
        ax.legend(
            handles=legend_patches, loc="lower right", frameon=False, labelcolor="white"
        )

        ax.set_aspect("equal", "box")
        ax.autoscale_view()
        ax.axis("off")
        ax.set_title("Multi-Stage Mesh Cleaning Analysis", color="white")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(
            output_path, dpi=300, bbox_inches="tight", pad_inches=0, facecolor="black"
        )
        plt.close(fig)
        logger.info(f"Detailed visualization saved to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)[:500]}", exc_info=True)


if __name__ == "__main__":
    add_file_handler()
    input_ply = "data/demo4/fuse_mesh.ply"
    output_image_path = "out/cleaned_mesh_analysis.png"

    clean_mesh_pipeline(ply_path=input_ply, output_path=output_image_path)
