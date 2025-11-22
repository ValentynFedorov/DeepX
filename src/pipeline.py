"""
Primary pipeline module for video processing and COLMAP reconstruction.

This module orchestrates the end-to-end photogrammetry workflow:
1. Ingests raw video footage.
2. Extracts frames with uniform temporal sampling to maximize baseline.
3. Wraps the COLMAP CLI to perform Structure-from-Motion (SfM) and
   Multi-View Stereo (MVS) reconstruction.
"""

import cv2
import shutil
import subprocess
import os
from pathlib import Path


class Pipeline3D:
    """
    Manages the lifecycle of the 3D reconstruction process.

    Attributes:
        video_path (str): Path to the source video file.
        root (Path): Path to the workspace directory.
        max_frames (int): Frame limit constraint (from assignment requirements).
        colmap_bin (str): Path to the COLMAP executable.
    """

    def __init__(self, video_path, project_root="workspace", max_frames=100, colmap_bin=r"C:\COLMAP\bin\colmap.exe"):
        """
        Initialize the pipeline configuration.

        Args:
            video_path (str): Input video file path.
            project_root (str): Directory where intermediate and final results will be stored.
            max_frames (int): Maximum number of frames to extract (Default: 100).
            colmap_bin (str): Path to COLMAP binary. Defaults to a standard Windows path,
                              but ideally should be resolved via system PATH.
        """
        self.video_path = video_path
        self.max_frames = max_frames
        self.colmap_bin = colmap_bin
        self.root = Path(project_root)

        self.paths = {
            "images": self.root / "images",
            "db": self.root / "database.db",
            "sparse": self.root / "sparse",
            "dense": self.root / "dense",
            "fused": self.root / "dense" / "fused.ply"
        }

        self._clean_workspace()

    def _clean_workspace(self):
        """
        Resets the workspace directory to ensure a fresh reconstruction.

        Handles permission errors common on Windows when files are locked by other processes.
        """
        if self.root.exists():
            try:
                shutil.rmtree(self.root)
            except PermissionError:
                print("Warning: Could not delete old workspace. Please delete manually.")
            except Exception as e:
                print(f"Warning: Error cleaning workspace: {e}")

        self.root.mkdir(parents=True, exist_ok=True)
        self.paths["images"].mkdir(exist_ok=True)
        self.paths["dense"].mkdir(exist_ok=True)
        self.paths["sparse"].mkdir(exist_ok=True)

    def extract_frames(self):
        """
        Extracts frames from the video adhering to the 100-frame limit.

        Strategy:
        Uses uniform sampling (Strided Slice) rather than taking the first 100 frames.

        Why?
        Structure-from-Motion (SfM) requires a sufficient baseline (parallax) between frames
        to triangulate 3D points. Taking just the first few seconds of video usually results
        in low baseline and poor reconstruction. Uniform sampling covers the entire object rotation.

        Returns:
            int: Number of frames successfully saved.
        """
        print(f"Extracting frames from {self.video_path}...")

        if not os.path.exists(self.video_path):
             raise FileNotFoundError(f"Video file not found: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print("Error: Video appears to be empty or corrupted.")
            return 0

        step = max(1, total_frames // self.max_frames)

        count, saved = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % step == 0 and saved < self.max_frames:
                output_file = self.paths["images"] / f"frame_{saved:04d}.jpg"
                cv2.imwrite(str(output_file), frame)
                saved += 1
            count += 1

        cap.release()
        print(f"Extracted {saved} frames.")
        return saved

    def run_colmap_reconstruction(self):
        """
        Executes the full COLMAP pipeline: SfM (Sparse) + MVS (Dense).

        This method chains multiple COLMAP CLI commands.

        Pipeline Stages:
        1. Feature Extraction: Detects SIFT features in images.
        2. Matching: Finds correspondences between features across images.
        3. Mapper (SfM): Triangulates 3D points and solves for camera poses.
        4. Undistortion: Removes lens distortion to prepare for dense reconstruction.
        5. Patch Match Stereo: Computes depth maps and normal maps.
        6. Fusion: Merges depth maps into a single point cloud.
        """
        print("Running COLMAP Reconstruction...")

        commands = [
            [self.colmap_bin, "feature_extractor",
             "--database_path", str(self.paths["db"]),
             "--image_path", str(self.paths["images"]),
             "--ImageReader.camera_model", "SIMPLE_RADIAL"],

            [self.colmap_bin, "exhaustive_matcher",
             "--database_path", str(self.paths["db"])],

            [self.colmap_bin, "mapper",
             "--database_path", str(self.paths["db"]),
             "--image_path", str(self.paths["images"]),
             "--output_path", str(self.paths["sparse"])],

            [self.colmap_bin, "image_undistorter",
             "--image_path", str(self.paths["images"]),
             "--input_path", str(self.paths["sparse"] / "0"),
             "--output_path", str(self.paths["dense"]),
             "--output_type", "COLMAP"],

            [self.colmap_bin, "patch_match_stereo",
             "--workspace_path", str(self.paths["dense"])],

            [self.colmap_bin, "stereo_fusion",
             "--workspace_path", str(self.paths["dense"]),
             "--output_path", str(self.paths["fused"])]
        ]

        for cmd in commands:
            print(f"Running: {cmd[1]}...")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"CRITICAL ERROR running {cmd[1]}: {e}")
                print("Ensure COLMAP is installed and the input images are valid.")
                raise e

        print("Reconstruction complete.")

    def run_full_pipeline(self):
        """
        Convenience method to run the extraction and reconstruction sequentially.

        Returns:
            Path: Path to the final fused point cloud file.
        """
        self.extract_frames()
        self.run_colmap_reconstruction()
        return self.paths["fused"]