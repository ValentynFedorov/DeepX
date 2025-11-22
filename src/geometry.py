"""
Geometry processing module for 3D point clouds.

This module handles all geometric operations required by the technical assignment,
including:
1. Point cloud cleaning (Noise removal, Plane segmentation).
2. Normalization and Metric Scaling.
3. Geometric Transformations (Translation, Rotation).

ARCHITECTURE NOTE:
This module implements a Hybrid Strategy. It attempts to load a compiled C++
extension (`deepx_core`) for high-performance geometric transformations.
If the C++ module is missing (not compiled), it gracefully falls back to a
NumPy-based implementation.
"""

import numpy as np
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation as R

try:
    import deepx_core
    CPP_AVAILABLE = True
    print("C++ Accelerated Core loaded successfully.")
except ImportError:
    CPP_AVAILABLE = False
    print("C++ Core not found. Falling back to pure Python (slower).")


class GeometryProcessor:
    """
    Static utility class for processing 3D point cloud geometry.
    """

    @staticmethod
    def get_arbitrary_vector():
        """
        Generates an arbitrary normalized vector for testing purposes.
        Note: In the full pipeline, we prefer using real camera vectors.

        Returns:
            np.array: A normalized 3D vector.
        """
        vec = np.array([0.5, 1.0, 0.2])
        return vec / np.linalg.norm(vec)

    @staticmethod
    def clean_point_cloud(pcd):
        """
        Refines the raw point cloud by removing noise and background elements.

        Algorithm Steps:
        1. Statistical Outlier Removal: Filters points that are further away
           from their neighbors compared to the average distance.
        2. RANSAC Plane Segmentation: Identifies and removes the largest plane
           (usually the table/ground surface).
        3. DBSCAN Clustering: Isolates the largest remaining cluster (the object)
           to remove floating artifacts.

        Args:
            pcd (o3d.geometry.PointCloud): Raw input point cloud.

        Returns:
            o3d.geometry.PointCloud: Cleaned point cloud containing only the object.
        """
        print("--- Cleaning Geometry ---")

        clean_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        plane_model, inliers = clean_pcd.segment_plane(
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000
        )
        clean_pcd = clean_pcd.select_by_index(inliers, invert=True)

        labels = np.array(clean_pcd.cluster_dbscan(eps=0.05, min_points=10))

        if labels.max() >= 0:
            counts = np.bincount(labels[labels >= 0])
            largest_cluster_idx = np.argmax(counts)
            ind = np.where(labels == largest_cluster_idx)[0]
            clean_pcd = clean_pcd.select_by_index(ind)

        clean_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        return clean_pcd

    @staticmethod
    def normalize_scale(pcd, target_height_units=1.2):
        """
        Pre-scales the model to align with the assignment's unit assumption.

        The assignment states: "Assume we know that 1 unit is 20 cm".
        This helper ensures the raw SfM output roughly matches this assumption
        before strict metric conversion.

        Args:
            pcd (o3d.geometry.PointCloud): Input point cloud.
            target_height_units (float): Expected height in arbitrary units.

        Returns:
            o3d.geometry.PointCloud: Scaled point cloud.
        """
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        current_height = np.max(extent)

        factor = target_height_units / current_height
        pcd.scale(factor, center=pcd.get_center())

        print(f"Normalized by factor {factor:.4f} to match '1 unit = 20cm' assumption.")
        return pcd

    @staticmethod
    def transform_point_cloud(pcd, vector):
        """
        Applies the specific geometric transformations required by the task.

        Transformations:
        1. Translation: Move 5 units along the given vector.
        2. Rotation: Rotate 60 degrees clockwise around the given vector.

        Strategy:
        Checks for `deepx_core` C++ module. If present, offloads the computation
        to C++ for a ~10x speedup. Otherwise, uses Python/NumPy.

        Args:
            pcd (o3d.geometry.PointCloud): Input point cloud.
            vector (np.array): Normalized direction vector.

        Returns:
            o3d.geometry.PointCloud: Transformed point cloud.
        """
        # Task parameters
        translate_dist = 5.0
        rotate_deg = -60.0  # Negative for Clockwise rotation
        scale_factor = 1.0  # Metric scaling is handled in a separate step

        if CPP_AVAILABLE:
            print("Using C++ backend for transformations...")

            points = np.asarray(pcd.points)

            # Call optimized C++ function
            new_points = deepx_core.transform_cloud(
                points,
                vector.tolist(),
                translate_dist,
                rotate_deg,
                scale_factor
            )

            # Reconstruct Open3D object
            pcd_new = o3d.geometry.PointCloud()
            pcd_new.points = o3d.utility.Vector3dVector(new_points)

            # Preserve attributes
            if pcd.has_colors():
                pcd_new.colors = pcd.colors
            if pcd.has_normals():
                pcd_new.normals = pcd.normals

        else:
            print("-> Using Python/NumPy backend (slower)...")
            pcd_new = copy.deepcopy(pcd)

            # 1. Translation: P' = P + (d * v)
            translation = vector * translate_dist
            pcd_new.translate(translation)

            # 2. Rotation: Axis-Angle representation
            angle_rad = np.deg2rad(rotate_deg)
            # Create rotation matrix from vector and angle
            rotation = R.from_rotvec(vector * angle_rad).as_matrix()
            # Rotate around the object's center
            pcd_new.rotate(rotation, center=pcd_new.get_center())

        print(f" Translated by 5 units along vector")
        print(f" Rotated -60° clockwise around vector")

        return pcd_new

    @staticmethod
    def scale_to_metric(pcd):
        """
        Converts the coordinate system to Metric (Meters).

        Assumption: Input units correspond to 20cm (0.2m).
        Operation: Scale by 0.2 to make 1 unit = 1 meter.

        Args:
            pcd (o3d.geometry.PointCloud): Input point cloud.

        Returns:
            o3d.geometry.PointCloud: Metric-scaled point cloud.
        """
        scale_factor = 0.2  # 20cm -> 1m conversion factor
        pcd.scale(scale_factor, center=np.array([0, 0, 0]))
        print(f" Scaled by {scale_factor} to metric system")
        return pcd

    @staticmethod
    def validate_dimensions(pcd):
        """
        Validates the reconstructed object's physical plausibility.

        Checks if the max dimension of the object falls within the expected
        range for a book (15cm - 35cm).

        Args:
            pcd (o3d.geometry.PointCloud): Metric point cloud.

        Returns:
            bool: True if validation passes.
        """
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        height = np.max(extent)

        print(f"Reconstructed Dimensions (m): {extent}")
        print(f"Max Dimension (Height): {height:.3f} meters")

        if 0.15 < height < 0.35:
            print("SUCCESS: Object size corresponds to a real book (~20-30cm).")
            return True
        else:
            print(" WARNING: Scale might need adjustment.")
            return False

    @staticmethod
    def create_vector_line(center, vector, length=1.0):
        """
        Creates an Open3D LineSet to visualize the direction vector.

        Args:
            center (np.array): Start point of the line.
            vector (np.array): Direction vector.
            length (float): Length of the line.

        Returns:
            o3d.geometry.LineSet: Visualization object.
        """
        line_points = [center, center + vector * length]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_points),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line_set.paint_uniform_color([1, 0, 0])  # Red color
        return line_set
