import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyvista as pv
from matplotlib.collections import PolyCollection
from scipy.spatial import KDTree


# --- Reuse functions from the project ---
# It's better to import them, but for a self-contained script, we can redefine
def calculate_curvature(mesh, curv_type="mean"):
    """Calculates curvature using PyVista."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if len(faces) == 0:
        return None
    faces_pyvista = np.hstack((np.full((len(faces), 1), 3), faces))
    pv_mesh = pv.PolyData(vertices, faces_pyvista)
    return pv_mesh.curvature(curv_type=curv_type)


def load_mesh(file_path):
    """Loads a mesh using Open3D."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    mesh = o3d.io.read_triangle_mesh(file_path)
    if mesh.is_empty():
        raise ValueError("Mesh is empty.")
    if not mesh.has_triangles():
        print("Point cloud detected, attempting Poisson reconstruction...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8
        )
    return mesh


def _fit_curve_from_points(points, degree=4):
    """Helper function to fit a 3D curve to a set of points."""
    if len(points) < degree + 1:
        print(
            f"Warning: Cannot fit a polynomial of degree {degree} with only {len(points)} points."
        )
        return None

    centerline_np = np.array(points)
    # Sort points by y-coordinate to ensure correct fitting
    centerline_np = centerline_np[centerline_np[:, 1].argsort()]
    cl_x, cl_y, cl_z = centerline_np[:, 0], centerline_np[:, 1], centerline_np[:, 2]

    x_fit_coeffs = np.polyfit(cl_y, cl_x, degree)
    z_fit_coeffs = np.polyfit(cl_y, cl_z, degree)

    x_poly = np.poly1d(x_fit_coeffs)
    z_poly = np.poly1d(z_fit_coeffs)

    y_smooth = np.linspace(cl_y.min(), cl_y.max(), 200)
    x_smooth = x_poly(y_smooth)
    z_smooth = z_poly(y_smooth)

    return np.vstack((x_smooth, y_smooth, z_smooth)).T


def _get_binned_centroids(points, num_bins=100):
    """Helper function to get centroids from binned points."""
    if len(points) == 0:
        return []

    y_coords = points[:, 1]
    y_min, y_max = y_coords.min(), y_coords.max()

    bin_height = (y_max - y_min) / num_bins
    if bin_height < 1e-9:
        return [np.mean(points, axis=0)] if len(points) > 0 else []

    raw_centerline_points = []
    for i in range(num_bins):
        bin_y_min = y_min + i * bin_height
        bin_y_max = y_min + (i + 1) * bin_height
        points_in_bin = points[(y_coords >= bin_y_min) & (y_coords < bin_y_max)]
        if points_in_bin.shape[0] > 0:
            raw_centerline_points.append(np.mean(points_in_bin, axis=0))
    return raw_centerline_points


def find_spine_from_mesh(ply_path: str):
    """
    Analyzes a 3D mesh in a two-pass process to find the spinal column curve.
    """
    try:
        # --- Step 1: Load Data & Compute Curvature ---
        print(f"Loading mesh from {ply_path}...")
        mesh = load_mesh(ply_path)
        mesh.compute_vertex_normals()
        print("Calculating mean curvature...")
        curvatures = calculate_curvature(mesh, "mean")
        if curvatures is None:
            return
        vertices = np.asarray(mesh.vertices)

        # --- Step 2: Coarse Pass to Find Baseline Spine Curve ---
        print("Coarse Pass: Finding baseline spine...")
        low_curvature_mask = curvatures < np.quantile(curvatures, 0.15)

        x_coords = vertices[:, 0]
        x_min, x_max = x_coords.min(), x_coords.max()
        x_width = (
            x_max - x_min
        ) * 0.40  # Use a generous 40% of mesh width for the coarse pass
        roi_x_min, roi_x_max = (
            (x_min + x_max) / 2 - x_width / 2,
            (x_min + x_max) / 2 + x_width / 2,
        )
        spatial_mask_x = (x_coords >= roi_x_min) & (x_coords <= roi_x_max)

        y_coords_all = vertices[:, 1]
        y_min, y_max = y_coords_all.min(), y_coords_all.max()
        y_range = y_max - y_min
        roi_y_min, roi_y_max = y_min + 0.20 * y_range, y_max - 0.20 * y_range
        spatial_mask_y = (y_coords_all >= roi_y_min) & (y_coords_all <= roi_y_max)

        coarse_mask = low_curvature_mask & spatial_mask_x & spatial_mask_y
        coarse_candidate_vertices = vertices[coarse_mask]

        if len(coarse_candidate_vertices) == 0:
            print("Error: No candidate points found in coarse pass.")
            return

        coarse_centroids = _get_binned_centroids(coarse_candidate_vertices)
        baseline_curve = _fit_curve_from_points(coarse_centroids)
        if baseline_curve is None:
            print("Error: Failed to fit baseline curve in coarse pass.")
            return

        # --- Visualize Coarse Pass Results ---
        print("Creating 2D visualization for coarse pass...")
        fig_coarse, ax_coarse = plt.subplots(figsize=(8, 10), facecolor="black")
        vertices_2d = np.asarray(mesh.vertices)[:, :2]
        faces = np.asarray(mesh.triangles)
        polygons = vertices_2d[faces]
        ax_coarse.add_collection(
            PolyCollection(polygons, facecolors="#404040", edgecolors="#404040")
        )
        coarse_points_2d = coarse_candidate_vertices[:, :2]
        ax_coarse.scatter(
            coarse_points_2d[:, 0],
            coarse_points_2d[:, 1],
            c="green",
            s=3,
            alpha=0.5,
            label="Coarse Candidates",
        )
        baseline_2d = baseline_curve[:, :2]
        ax_coarse.plot(
            baseline_2d[:, 0],
            baseline_2d[:, 1],
            c="yellow",
            ls="--",
            lw=2.5,
            label="Baseline Curve",
        )
        ax_coarse.set_aspect("equal", "box")
        ax_coarse.autoscale_view()
        ax_coarse.axis("off")
        ax_coarse.legend(loc="upper right", labelcolor="white", frameon=False)
        output_path_coarse = "out/spine_2d_result_1_coarse.png"
        os.makedirs("out", exist_ok=True)
        fig_coarse.savefig(
            output_path_coarse,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="black",
        )
        plt.close(fig_coarse)
        print(f"Successfully saved coarse pass visualization to {output_path_coarse}")

        # --- Step 3: Fine Pass using the Baseline Curve as ROI ---
        print("Fine Pass: Refining spine curve...")
        # Create a KD-Tree for fast distance search to the baseline curve
        curve_kdtree = KDTree(baseline_curve)
        distances, _ = curve_kdtree.query(vertices, k=1)

        # Define a tubular ROI with a radius relative to the mesh's overall width
        mesh_width = x_max - x_min  # Calculated in coarse pass
        tube_radius = mesh_width * 0.05  # Radius is 5% of the total mesh width
        print(f"Using adaptive tube radius for fine pass: {tube_radius:.4f}")
        tubular_roi_mask = distances < tube_radius

        # Combine tubular ROI with the original low curvature mask
        fine_mask = tubular_roi_mask & low_curvature_mask
        fine_candidate_vertices = vertices[fine_mask]

        if len(fine_candidate_vertices) < 50:
            print(
                f"Warning: Found only {len(fine_candidate_vertices)} candidate points in fine pass."
            )
            if len(fine_candidate_vertices) == 0:
                print("Using coarse results as final.")
                fine_candidate_vertices = coarse_candidate_vertices

        fine_centroids = _get_binned_centroids(fine_candidate_vertices)
        final_curve = _fit_curve_from_points(fine_centroids)
        if final_curve is None:
            print("Error: Failed to fit final curve. Using baseline curve.")
            final_curve = baseline_curve

        # --- Step 4: 2D Visualization of Final Results ---
        print("Creating final 2D visualization...")
        fig, ax = plt.subplots(figsize=(8, 10), facecolor="black")

        # 1. Background mesh
        ax.add_collection(
            PolyCollection(polygons, facecolors="#404040", edgecolors="#404040")
        )

        # 2. Fine candidate points (cyan)
        fine_points_2d = fine_candidate_vertices[:, :2]
        ax.scatter(
            fine_points_2d[:, 0],
            fine_points_2d[:, 1],
            c="cyan",
            s=3,
            alpha=0.6,
            label="Fine Candidates",
        )

        # 3. Baseline curve (yellow, dashed)
        baseline_2d = baseline_curve[:, :2]
        ax.plot(
            baseline_2d[:, 0],
            baseline_2d[:, 1],
            c="yellow",
            ls="--",
            lw=2,
            label="Baseline Curve",
        )

        # 4. Final curve (red, solid)
        final_curve_2d = final_curve[:, :2]
        ax.plot(
            final_curve_2d[:, 0],
            final_curve_2d[:, 1],
            c="red",
            lw=2.5,
            label="Final Curve",
        )

        ax.set_aspect("equal", "box")
        ax.autoscale_view()
        ax.axis("off")
        ax.legend(loc="upper right", labelcolor="white", frameon=False)

        output_path_fine = "out/spine_2d_result_2_fine.png"
        os.makedirs("out", exist_ok=True)
        fig.savefig(
            output_path_fine,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="black",
        )
        plt.close(fig)

        print(f"Successfully saved final 2D visualization to {output_path_fine}")

    except Exception as e:
        print(f"An error occurred during the process: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    input_ply = "data/demo4/fuse_mesh.ply"
    if not os.path.exists(input_ply):
        print(f"Error: Input mesh not found at {input_ply}")
    else:
        find_spine_from_mesh(input_ply)
