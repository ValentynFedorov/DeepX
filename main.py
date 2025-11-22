"""
Main Entry Point (CLI Interface) for the DeepX 3D Reconstruction Pipeline.

This script acts as the orchestrator, glueing together the separate modules:
1. Pipeline (Video -> Dense Cloud)
2. Geometry (Cleaning -> Transformation -> Scaling)
3. Visualization (Rendering the final result)

Usage:
    python main.py --video input.mp4 --colmap-bin /usr/local/bin/colmap
"""

import argparse
import open3d as o3d
import copy
from pathlib import Path
from src.pipeline import Pipeline3D
from src.geometry import GeometryProcessor
from src.visualization import Visualizer


def main():
    """
    Main execution function.
    Parses command line arguments and executes the processing steps sequentially.
    """
    parser = argparse.ArgumentParser(
        description="DeepX 3D Book Reconstruction Pipeline"
    )

    # --- Arguments Setup ---
    parser.add_argument(
        "--video",
        type=str,
        default="input_video.mp4",
        help="Path to input video file (Target object)"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="workspace",
        help="Directory where COLMAP intermediate files will be stored"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Frame limit constraint (Requirement: max 100 frames)"
    )
    parser.add_argument(
        "--colmap-bin",
        type=str,
        default=r"C:\COLMAP\bin\colmap.exe",
        help="Path to the COLMAP executable on the system"
    )
    parser.add_argument(
        "--skip-reconstruction",
        action="store_true",
        help="Debug flag: Skips SfM/MVS if 'fused.ply' already exists"
    )

    args = parser.parse_args()

    try:

        pipeline = Pipeline3D(
            video_path=args.video,
            project_root=args.workspace,
            max_frames=args.max_frames,
            colmap_bin=args.colmap_bin
        )

        if not args.skip_reconstruction:
            print(f"Starting reconstruction pipeline for {args.video}...")
            point_cloud_path = pipeline.run_full_pipeline()
        else:
            point_cloud_path = Path(args.workspace) / "dense" / "fused.ply"
            print(f"Skipping reconstruction, using existing file: {point_cloud_path}")

        if not point_cloud_path.exists():
            raise FileNotFoundError(f"Point cloud not found at {point_cloud_path}")

        print("\nLoading point cloud...")
        pcd = o3d.io.read_point_cloud(str(point_cloud_path))

        print("\nGeometric Processing...")
        geo = GeometryProcessor()

        clean_pcd = geo.clean_point_cloud(pcd)

        normalized_pcd = geo.normalize_scale(clean_pcd)

        original_viz = copy.deepcopy(normalized_pcd)

        vector = geo.get_arbitrary_vector()
        print(f"Selected arbitrary vector: {vector}")

        transformed_pcd = geo.transform_point_cloud(normalized_pcd, vector)

        final_pcd = geo.scale_to_metric(transformed_pcd)

        original_viz.scale(0.2, center=[0, 0, 0])

        print("\nValidation & Visualization...")
        geo.validate_dimensions(final_pcd)

        visualizer = Visualizer()
        visualizer.visualize_before_after(
            pcd_before=original_viz,
            pcd_after=final_pcd,
            vector=vector,
            window_name="DeepX Task: Final Result"
        )

        print("\nPipeline completed successfully!")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Tip: Make sure the video file exists and COLMAP reconstruction completed successfully.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Tip: Check if COLMAP is installed and accessible via the specified path.")


if __name__ == "__main__":
    main()