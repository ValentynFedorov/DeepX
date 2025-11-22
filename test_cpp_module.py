"""
C++ Module Verification and Benchmarking Suite.

This script serves two purposes:
1. Unit Testing: Verifies that the compiled C++ extension (`deepx_core`)
   loads correctly and produces valid geometric transformations.
2. Performance Profiling: Benchmarks the C++ implementation against a pure
   Python/NumPy equivalent to quantify the speedup achieved (usually 10x-50x).

Usage:
    python test_cpp_module.py
"""

import numpy as np
import time
import sys

def test_cpp_available():
    """
    Checks if the compiled C++ extension is importable.

    Returns:
        bool: True if the module is found, False otherwise.
    """
    try:
        import deepx_core
        print("C++ module 'deepx_core' loaded successfully!")
        return True
    except ImportError:
        print(" C++ module not found.")
        return False


def test_transformation_correctness():
    """
    Sanity check for the C++ transformation logic.

    Verifies:
    1. Input/Output shape consistency.
    2. Mathematical validity (no NaNs or Infinite values).

    Returns:
        bool: True if all assertions pass.
    """
    try:
        import deepx_core

        print("\nTesting Transformation Correctness")

        points = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        vector = [0.0, 0.0, 1.0]
        translate_dist = 5.0
        rotate_deg = 90.0
        scale_factor = 2.0

        result = deepx_core.transform_cloud(
            points, vector, translate_dist, rotate_deg, scale_factor
        )

        # 4. Debug Output
        print(f"Input shape:  {points.shape}")
        print(f"Output shape: {result.shape}")

        # 5. Assertions
        assert result.shape == points.shape, "Output shape mismatch!"
        assert not np.any(np.isnan(result)), "NaNs detected in output!"
        assert not np.any(np.isinf(result)), "Infinite values detected!"

        print("Shape and Sanity checks passed.")
        return True

    except Exception as e:
        print(f"Test failed with error: {e}")
        return False


def benchmark_performance():
    """
    Runs a comparative benchmark between C++ and Python implementations.

    Methodology:
    - Generates random point clouds of increasing size ($10^3$ to $10^5$).
    - Measures execution time for the C++ kernel.
    - Measures execution time for an equivalent NumPy/SciPy implementation.
    - Calculates the Speedup Factor (Time_Python / Time_CPP).
    """
    try:
        import deepx_core
        from scipy.spatial.transform import Rotation as R

        print("Performance Benchmark")

        sizes = [1000, 10000, 100000]

        vector = np.array([0.5, 1.0, 0.2])
        vector = vector / np.linalg.norm(vector)

        for size in sizes:
            points = np.random.rand(size, 3)

            start = time.perf_counter()
            _ = deepx_core.transform_cloud(
                points, vector.tolist(), 5.0, -60.0, 1.0
            )
            cpp_time = time.perf_counter() - start

            start = time.perf_counter()

            angle_rad = np.deg2rad(-60.0)
            rotation_matrix = R.from_rotvec(vector * angle_rad).as_matrix()

            centroid = np.mean(points, axis=0)
            points_centered = points - centroid
            rotated = points_centered @ rotation_matrix.T
            translation = vector * 5.0
            _ = rotated + centroid + translation

            py_time = time.perf_counter() - start

            speedup = py_time / cpp_time

            print(f"\ndataset_size: {size:,} points")
            print(f"  C++ Kernel:   {cpp_time * 1000:.4f} ms")
            print(f"  Python/NumPy: {py_time * 1000:.4f} ms")
            print(f"  Speedup:   \033[92m{speedup:.1f}x\033[0m") # Green colored output

    except ImportError:
        print("\nBenchmark skipped: C++ module not installed.")
    except Exception as e:
        print(f"\Benchmark failed: {e}")


def main():
    """Main entry point for the test suite."""
    print("DeepX C++ Optimization Module Test Suite")

    # 1. Availability Check
    if not test_cpp_available():
        return

    # 2. Logic Verification
    if not test_transformation_correctness():
        print("Transformation logic failed. Aborting benchmark.")
        return

    # 3. Performance Profiling
    benchmark_performance()
    print("Test Suite Completed.")

if __name__ == "__main__":
    main()