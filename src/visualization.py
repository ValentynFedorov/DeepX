"""
Visualization module for 3D geometric data using Open3D.

This module provides static utility methods to render point clouds,
geometric primitives (lines, vectors), and camera trajectories.
It abstracts the Open3D visualization logic to keep the main pipeline clean.
"""

import open3d as o3d
import numpy as np


class Visualizer:
    """
    A static utility class for rendering 3D scenes.
    """

    @staticmethod
    def visualize_point_cloud(pcd, window_name="Point Cloud", width=1024, height=768):
        """
        Basic viewer for a single point cloud.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud to render.
            window_name (str): Title of the window.
            width (int): Window width.
            height (int): Window height.
        """
        print(f"Opening {window_name}...")
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=window_name,
            width=width,
            height=height
        )

    @staticmethod
    def visualize_with_vector(pcd, vector, center=None, window_name="Model with Vector"):
        """
        Visualizes a point cloud alongside a specific 3D vector.

        Useful for debugging the "Arbitrary Camera Vector" selection step.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud.
            vector (np.array): Normalized direction vector [x, y, z].
            center (np.array, optional): Origin point of the vector.
                                         Defaults to point cloud center.
            window_name (str): Title of the window.
        """
        if center is None:
            center = pcd.get_center()

        line_points = [center, center + vector * 2.0]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_points),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line_set.paint_uniform_color([1, 0, 0])

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        o3d.visualization.draw_geometries(
            [pcd, line_set, axes],
            window_name=window_name,
            width=1024,
            height=768
        )

    @staticmethod
    def visualize_before_after(pcd_before, pcd_after, vector,
                               window_name="Before/After Transformation"):
        """
        Visualizes the 'Before' and 'After' states of the geometric transformation.

        This satisfies the assignment requirement to visualize the results of
        translation and rotation.

        Color Coding:
        - Grey: Original Model (Reference)
        - Gold: Transformed Model (Result)
        - Red Line: The vector axis used for rotation/translation

        Args:
            pcd_before (o3d.geometry.PointCloud): Original state.
            pcd_after (o3d.geometry.PointCloud): Transformed state.
            vector (np.array): The axis of rotation/translation.
            window_name (str): Title of the window.
        """
        pcd_before.paint_uniform_color([0.5, 0.5, 0.5])
        pcd_after.paint_uniform_color([1, 0.7, 0])


        center = pcd_before.get_center()
        line_points = [center, center + vector * 1.0]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_points),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line_set.paint_uniform_color([1, 0, 0])

        print(f"   Visualizing: {window_name}")
        print("   [Grey Object] = Original Position")
        print("   [Gold Object] = Transformed Position")
        print("   [Red Line]    = Axis Vector")
        print("   Controls: Left Click=Rotate, Ctrl+Click=Pan, Scroll=Zoom")

        o3d.visualization.draw_geometries(
            [pcd_before, pcd_after, line_set],
            window_name=window_name,
            width=1200,
            height=800
        )

    @staticmethod
    def visualize_cameras_and_model(pcd, camera_geometries,
                                    window_name="Cameras & Reconstruction"):
        """
        Visualizes the reconstructed point cloud along with camera frustums.

        This satisfies the requirement: "Visualize ... camera positions and camera vectors".

        Args:
            pcd (o3d.geometry.PointCloud): The fused point cloud.
            camera_geometries (list): List of Open3D meshes/lines representing cameras.
            window_name (str): Title of the window.
        """
        print("Visualizing Reconstruction with Camera Trajectory...")
        o3d.visualization.draw_geometries(
            [pcd, *camera_geometries],
            window_name=window_name,
            width=1024,
            height=768
        )

    @staticmethod
    def visualize_advanced(pcd, background_color=[0.1, 0.1, 0.1],
                           point_size=2.0, window_name="Advanced Visualization"):
        """
        Advanced renderer allowing control over background and point size.

        Unlike `draw_geometries`, this uses the explicitly managed `Visualizer` class
        to modify `RenderOption` before the window loop starts.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud.
            background_color (list): RGB list [0.0-1.0] for background. Default: Dark Grey.
            point_size (float): Size of rendered points. Default: 2.0.
            window_name (str): Title of the window.
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1024, height=768)

        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.background_color = np.asarray(background_color)
        opt.point_size = point_size

        print(f"Running advanced viewer. Point Size: {point_size}")
        vis.run()
        vis.destroy_window()