#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "color_separation_capi.h"

namespace py = pybind11;

namespace {

const char* error_message(int code) {
    switch (code) {
        case COLOR_SEPARATION_OK:
            return "ok";
        case COLOR_SEPARATION_ERR_NULL:
            return "null image, output, or params";
        case COLOR_SEPARATION_ERR_SHAPE:
            return "unsupported image shape";
        case COLOR_SEPARATION_ERR_ALLOC:
            return "color separation backend allocation failed";
        default:
            return "unknown color separation backend error";
    }
}

}  // namespace

py::array_t<float> apply_color_separation(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    float shadow_chroma_clean,
    float shadow_threshold,
    float color_separation,
    float chroma_clarity,
    float color_density,
    float subtractive_saturation,
    float opponent_contrast
) {
    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must be a 3D RGB float32 array");
    }

    std::vector<py::ssize_t> shape(in.shape.begin(), in.shape.end());
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    ColorSeparationConstImageF32 input_image{
        static_cast<const float*>(in.ptr),
        static_cast<int>(in.shape[1]),
        static_cast<int>(in.shape[0]),
        static_cast<int>(in.shape[2]),
        static_cast<int>(in.strides[0]),
    };
    ColorSeparationImageF32 output_image{
        static_cast<float*>(out.ptr),
        static_cast<int>(out.shape[1]),
        static_cast<int>(out.shape[0]),
        static_cast<int>(out.shape[2]),
        static_cast<int>(out.strides[0]),
    };
    ColorSeparationParams params{
        shadow_chroma_clean,
        shadow_threshold,
        color_separation,
        chroma_clarity,
        color_density,
        subtractive_saturation,
        opponent_contrast,
    };

    int status = COLOR_SEPARATION_OK;
    {
        py::gil_scoped_release release;
        status = color_separation_apply_v1(&input_image, &output_image, &params);
    }
    if (status != COLOR_SEPARATION_OK) {
        throw std::runtime_error(error_message(status));
    }

    return result;
}

PYBIND11_MODULE(_color_separation_cpu, m) {
    m.doc() = "CPU color separation backend for Platypus";
    m.def(
        "apply_color_separation",
        &apply_color_separation,
        py::arg("image"),
        py::arg("shadow_chroma_clean") = 0.0f,
        py::arg("shadow_threshold") = 0.2f,
        py::arg("color_separation") = 0.0f,
        py::arg("chroma_clarity") = 0.0f,
        py::arg("color_density") = 0.0f,
        py::arg("subtractive_saturation") = 0.0f,
        py::arg("opponent_contrast") = 0.0f
    );
}
