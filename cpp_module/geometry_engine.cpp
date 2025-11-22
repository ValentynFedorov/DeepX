#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;

/**
 * Rodrigues' rotation formula implementation
 * Rotates point (x,y,z) around axis (kx,ky,kz) by angle theta
 * Formula: P_rot = P*cos(θ) + (k×P)*sin(θ) + k*(k·P)*(1-cos(θ))
 */
inline void rotate_point(double& x, double& y, double& z,
                         const double kx, const double ky, const double kz,
                         const double theta) {

    const double cos_t = std::cos(theta);
    const double sin_t = std::sin(theta);
    const double one_minus_cos = 1.0 - cos_t;

    // Dot product: k · P
    const double dot = kx * x + ky * y + kz * z;

    // Cross product: k × P
    const double cross_x = ky * z - kz * y;
    const double cross_y = kz * x - kx * z;
    const double cross_z = kx * y - ky * x;

    // Apply Rodrigues formula
    const double nx = x * cos_t + cross_x * sin_t + kx * dot * one_minus_cos;
    const double ny = y * cos_t + cross_y * sin_t + ky * dot * one_minus_cos;
    const double nz = z * cos_t + cross_z * sin_t + kz * dot * one_minus_cos;

    x = nx;
    y = ny;
    z = nz;
}

/**
 * High-performance point cloud transformation pipeline
 * Applies: Translation → Rotation → Scaling
 *
 * @param input_points: Nx3 numpy array of point coordinates
 * @param axis_vector: 3D vector defining rotation axis and translation direction
 * @param translate_dist: Translation distance along axis_vector
 * @param rotate_deg: Rotation angle in degrees (clockwise)
 * @param scale_factor: Uniform scale factor
 * @return: Transformed Nx3 numpy array
 */
py::array_t<double> transform_cloud(
    py::array_t<double> input_points,
    std::vector<double> axis_vector,
    double translate_dist,
    double rotate_deg,
    double scale_factor
) {
    // Validate input
    py::buffer_info buf = input_points.request();

    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Input must be Nx3 numpy array");
    }

    const size_t num_points = buf.shape[0];
    double* ptr = static_cast<double*>(buf.ptr);

    // Allocate output array
    auto result = py::array_t<double>({static_cast<long>(num_points), 3L});
    py::buffer_info res_buf = result.request();
    double* res_ptr = static_cast<double*>(res_buf.ptr);

    // Prepare transformation parameters
    const double theta = rotate_deg * M_PI / 180.0;

    // Normalize rotation axis
    const double vec_len = std::sqrt(
        axis_vector[0] * axis_vector[0] +
        axis_vector[1] * axis_vector[1] +
        axis_vector[2] * axis_vector[2]
    );
    const double kx = axis_vector[0] / vec_len;
    const double ky = axis_vector[1] / vec_len;
    const double kz = axis_vector[2] / vec_len;

    // Translation vector (along axis)
    const double tx = kx * translate_dist;
    const double ty = ky * translate_dist;
    const double tz = kz * translate_dist;

    // Compute centroid for rotation around center
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (size_t i = 0; i < num_points; ++i) {
        cx += ptr[i * 3 + 0];
        cy += ptr[i * 3 + 1];
        cz += ptr[i * 3 + 2];
    }
    cx /= num_points;
    cy /= num_points;
    cz /= num_points;

    // HIGH-PERFORMANCE TRANSFORMATION LOOP
    // This is the "inner loop" optimized for cache locality and vectorization
    for (size_t i = 0; i < num_points; ++i) {
        double x = ptr[i * 3 + 0];
        double y = ptr[i * 3 + 1];
        double z = ptr[i * 3 + 2];

        // 1. Translate to origin (for rotation around centroid)
        x -= cx;
        y -= cy;
        z -= cz;

        // 2. Apply rotation using Rodrigues formula
        rotate_point(x, y, z, kx, ky, kz, theta);

        // 3. Translate back + apply translation offset
        x += cx + tx;
        y += cy + ty;
        z += cz + tz;

        // 4. Apply uniform scaling
        x *= scale_factor;
        y *= scale_factor;
        z *= scale_factor;

        // Write result
        res_ptr[i * 3 + 0] = x;
        res_ptr[i * 3 + 1] = y;
        res_ptr[i * 3 + 2] = z;
    }

    return result;
}

/**
 * Python module binding
 */
PYBIND11_MODULE(deepx_core, m) {
    m.doc() = "C++ Accelerated Geometry Module for DeepX Assessment\n"
              "Provides high-performance point cloud transformations using optimized C++ code.";

    m.def("transform_cloud", &transform_cloud,
          "Apply rotation, translation and scale to point cloud",
          py::arg("input_points"),
          py::arg("axis_vector"),
          py::arg("translate_dist"),
          py::arg("rotate_deg"),
          py::arg("scale_factor"));
}