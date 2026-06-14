#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "cross_filter_capi.h"

namespace py = pybind11;

namespace {

const char* error_message(int code) {
    switch (code) {
        case CROSS_FILTER_OK:
            return "ok";
        case CROSS_FILTER_ERR_NULL:
            return "null image, output, or params";
        case CROSS_FILTER_ERR_SHAPE:
            return "unsupported image shape";
        case CROSS_FILTER_ERR_ALLOC:
            return "allocation failed";
        default:
            return "unknown cross filter backend error";
    }
}

}  // namespace

py::array_t<float> apply_cross_filter(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    int num_points,
    int length,
    float angle_deg,
    float threshold,
    float intensity,
    float spectral_strength,
    float line_thickness,
    int min_distance,
    float randomness,
    int speed_factor,
    bool debug_mode
) {
    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must be a 3D RGB float32 array");
    }

    std::vector<py::ssize_t> shape(in.shape.begin(), in.shape.end());
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    CrossFilterImageF32 input_image{
        static_cast<float*>(in.ptr),
        static_cast<int>(in.shape[1]),
        static_cast<int>(in.shape[0]),
        static_cast<int>(in.shape[2]),
        static_cast<int>(in.strides[0]),
    };
    CrossFilterImageF32 output_image{
        static_cast<float*>(out.ptr),
        static_cast<int>(out.shape[1]),
        static_cast<int>(out.shape[0]),
        static_cast<int>(out.shape[2]),
        static_cast<int>(out.strides[0]),
    };
    CrossFilterParams params{
        num_points,
        length,
        angle_deg,
        threshold,
        intensity,
        spectral_strength,
        line_thickness,
        min_distance,
        randomness,
        speed_factor,
        debug_mode ? 1 : 0,
    };

    int status = CROSS_FILTER_OK;
    {
        py::gil_scoped_release release;
        status = cross_filter_apply_v1(&input_image, &output_image, &params);
    }
    if (status != CROSS_FILTER_OK) {
        throw std::runtime_error(error_message(status));
    }

    return result;
}

PYBIND11_MODULE(_cross_filter_cpu, m) {
    m.doc() = "CPU CrossFilter backend for Platypus";
    m.def(
        "apply_cross_filter",
        &apply_cross_filter,
        py::arg("image"),
        py::arg("num_points") = 6,
        py::arg("length") = 100,
        py::arg("angle_deg") = 0.0f,
        py::arg("threshold") = 1.0f,
        py::arg("intensity") = 1.0f,
        py::arg("spectral_strength") = 0.2f,
        py::arg("line_thickness") = 1.0f,
        py::arg("min_distance") = 10,
        py::arg("randomness") = 0.0f,
        py::arg("speed_factor") = 4,
        py::arg("debug_mode") = false
    );
}
