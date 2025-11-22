"""
Module for parsing and processing COLMAP camera data.

This module handles the interaction with COLMAP's sparse reconstruction output.
It converts binary models to text for easier parsing and extracts camera
extrinsics (position and orientation) to be used in geometric transformations.
"""

import numpy as np
import subprocess
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import open3d as o3d  # Imported at top level for type hinting if needed

class CameraParser:
    """
    Handles parsing of COLMAP 'images.txt' to extract camera poses and trajectories.
    """

    def __init__(self, sparse_path, colmap_bin=r"C:\COLMAP\bin\colmap.exe"):
        """
        Initialize the parser.

        Args:
            sparse_path (str | Path): Path to the COLMAP sparse output folder (containing images.bin/.txt).
            colmap_bin (str): Path to the COLMAP executable. Used for model conversion.
        """
        self.sparse_path = Path(sparse_path)
        self.colmap_bin = colmap_bin
        self.images_txt_path = self.sparse_path / "images.txt"

    def convert_to_text_format(self):
        """
        Converts COLMAP binary model files (.bin) to text files (.txt).

        COLMAP produces efficient binary files by default. This method wraps the
        CLI 'model_converter' command to generate human-readable text files
        that are easier to parse with Python without heavy dependencies.

        Returns:
            bool: True if conversion succeeded or text file already exists.
        """
        if not self.sparse_path.exists():
            print(f"Error: Sparse model not found at {self.sparse_path}")
            return False

        if self.images_txt_path.exists():
            print("Text model already exists. Skipping conversion.")
            return True

        print("Converting COLMAP model to text...")
        cmd = [
            self.colmap_bin, "model_converter",
            "--input_path", str(self.sparse_path),
            "--output_path", str(self.sparse_path),
            "--output_type", "TXT"
        ]

        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error during conversion: {e}")
            return False

    def parse_camera_poses(self):
        """
        Parses 'images.txt' to extract camera extrinsics.

        Mathematical Logic:
        COLMAP stores the transformation from World space to Camera space:
            P_cam = R * P_world + t
        Where 'R' is a quaternion and 't' is the translation vector.

        To visualize the cameras in 3D space, we need the inverse (Camera center):
            Center = -R^T * t

        We also extract the View Vector (Camera's Z-axis in World Space), which corresponds
        to the 3rd row of the Rotation Matrix.

        Returns:
            list[dict]: A sorted list of camera dictionaries containing:
                - 'center': np.array [x, y, z] (Camera position in world)
                - 'R': np.array 3x3 (Rotation matrix)
                - 'view_vec': np.array [x, y, z] (Direction camera is facing)
                - 'name': str (Image filename)
        """
        if not self.images_txt_path.exists():
            print(f"Error: {self.images_txt_path} not found.")
            return []

        poses = []
        print(f"Parsing {self.images_txt_path}...")

        with open(self.images_txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            parts = line.split()

            if len(parts) >= 9 and ("png" in parts[-1] or "jpg" in parts[-1]):

                qw, qx, qy, qz = map(float, parts[1:5])

                tx, ty, tz = map(float, parts[5:8])

                rotation = R.from_quat([qx, qy, qz, qw])
                rot_matrix = rotation.as_matrix()


                t = np.array([tx, ty, tz])
                center = -rot_matrix.T @ t
                view_vec = rot_matrix[2, :]

                poses.append({
                    "center": center,
                    "R": rot_matrix,
                    "view_vec": view_vec,
                    "name": parts[-1]
                })
        poses.sort(key=lambda x: x["name"])
        print(f"Found {len(poses)} camera positions.")

        return poses

    @staticmethod
    def create_camera_frustums(poses, frustum_size=0.05):
        """
        Generates Open3D geometries to visualize camera positions and trajectory.

        Args:
            poses (list): List of dicts parsed by `parse_camera_poses`.
            frustum_size (float): Scale factor for the camera axis visualization.
                                  Should be adjusted based on scene scale (e.g., 0.2 for scaled scenes).

        Returns:
            list[o3d.geometry.Geometry]: List containing the trajectory LineSet
                                         and coordinate frames for each camera.
        """
        import open3d as o3d

        geometries = []

        points = [p["center"] for p in poses]
        lines = [[i, i + 1] for i in range(len(points) - 1)]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.paint_uniform_color([1, 0, 0])
        geometries.append(line_set)

        for p in poses:
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=frustum_size,
                origin=[0, 0, 0]
            )
            mesh.rotate(p["R"].T, center=[0, 0, 0])

            mesh.translate(p["center"])
            geometries.append(mesh)

        return geometries