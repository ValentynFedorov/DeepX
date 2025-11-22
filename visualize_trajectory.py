import argparse
import open3d as o3d
from pathlib import Path
from src.cameras import CameraParser
from src.visualization import Visualizer

"""
Camera Trajectory Visualization Script.

This standalone module is designed to visualize the relationship between the
reconstructed 3D scene (Dense Point Cloud) and the camera poses (Sparse Reconstruction)
recovered by COLMAP.

Key Functionality:
1. **Data Fusion**: Loads the dense point cloud (`fused.ply`) and the sparse camera
   extrinsics (`images.txt/bin`).
2. **Trajectory Reconstruction**: Parses the position (center) and orientation (rotation)
   of every camera frame used in the reconstruction.
3. **Visual Rendering**: Generates 3D camera frustums (pyramids) and connects them
   with a trajectory line to visualize the motion path of the camera during video recording.

This script specifically addresses the Technical Assignment requirement:
"Visualize the point cloud reconstruction, camera positions, and camera vectors."

Usage:
    python visualize_trajectory.py --workspace workspace --colmap-bin /path/to/colmap
"""

def main():
    parser = argparse.ArgumentParser(description="Visualize Camera Trajectory")
    parser.add_argument("--workspace", default="workspace", help="Project workspace path")
    parser.add_argument("--colmap-bin", default=r"C:\COLMAP\bin\colmap.exe", help="Path to COLMAP")
    args = parser.parse_args()

    root = Path(args.workspace)
    sparse_path = root / "sparse" / "0"
    dense_path = root / "dense" / "fused.ply"

    if not dense_path.exists():
        print(f"Error: {dense_path} not found. Run main.py first.")
        return

    print(f"Loading point cloud from {dense_path}...")
    pcd = o3d.io.read_point_cloud(str(dense_path))

    print("Parsing camera poses...")
    cam_parser = CameraParser(sparse_path, colmap_bin=args.colmap_bin)
    cam_parser.convert_to_text_format()
    poses = cam_parser.parse_camera_poses()

    if not poses:
        print("No cameras found!")
        return


    camera_geoms = cam_parser.create_camera_frustums(poses, frustum_size=0.1)

    print("Left click + Drag to rotate. Scroll to zoom.")
    Visualizer.visualize_cameras_and_model(
        pcd,
        camera_geoms,
        window_name="DeepX: Camera Trajectory"
    )

if __name__ == "__main__":
    main()