/*
 * Python binding for the native fused display colour transform backend.
 *
 * License: GPL-3.0-or-later as part of Shade Wave / PLATYPUS.
 * Implements behaviour compatible with a subset of Colour Science for Python,
 * whose upstream project is BSD-3-Clause licensed.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "colour_functions_capi.h"

namespace py = pybind11;

namespace {

float seq_float_or_default(const py::object& obj, py::ssize_t index, float fallback, const char* name) {
    if (obj.is_none()) {
        return fallback;
    }
    py::sequence seq = py::cast<py::sequence>(obj);
    if (seq.size() <= index) {
        throw std::invalid_argument(std::string(name) + " is shorter than expected");
    }
    return py::cast<float>(seq[index]);
}

const char* error_message(int code) {
    switch (code) {
        case COLOUR_FUNCTIONS_OK:
            return "ok";
        case COLOUR_FUNCTIONS_ERR_NULL:
            return "null image, output, or params";
        case COLOUR_FUNCTIONS_ERR_SHAPE:
            return "unsupported image shape";
        case COLOUR_FUNCTIONS_ERR_ENCODING:
            return "unsupported display encoding";
        default:
            return "unknown colour functions backend error";
    }
}

}  // namespace

py::array_t<float> apply_display_color_transform(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    py::array_t<float, py::array::c_style | py::array::forcecast> basis,
    int encoding,
    const py::object& luminance_weights,
    float eps
) {
    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must be a 3D RGB float32 array");
    }

    py::buffer_info basis_info = basis.request();
    if (basis_info.ndim != 2 || basis_info.shape[0] != 3 || basis_info.shape[1] != 3) {
        throw std::invalid_argument("basis must be a 3x3 float32 array");
    }

    std::vector<py::ssize_t> shape(in.shape.begin(), in.shape.end());
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    ColourFunctionsConstImageF32 input_image{
        static_cast<const float*>(in.ptr),
        static_cast<int>(in.shape[1]),
        static_cast<int>(in.shape[0]),
        static_cast<int>(in.shape[2]),
        static_cast<int>(in.strides[0]),
    };
    ColourFunctionsImageF32 output_image{
        static_cast<float*>(out.ptr),
        static_cast<int>(out.shape[1]),
        static_cast<int>(out.shape[0]),
        static_cast<int>(out.shape[2]),
        static_cast<int>(out.strides[0]),
    };

    ColourFunctionsParams params{};
    const float* basis_ptr = static_cast<const float*>(basis_info.ptr);
    for (int i = 0; i < 9; ++i) {
        params.basis[i] = basis_ptr[i];
    }
    params.encoding = static_cast<ColourFunctionsEncoding>(encoding);
    params.luminance_weights[0] = seq_float_or_default(luminance_weights, 0, 0.2126f, "luminance_weights");
    params.luminance_weights[1] = seq_float_or_default(luminance_weights, 1, 0.7152f, "luminance_weights");
    params.luminance_weights[2] = seq_float_or_default(luminance_weights, 2, 0.0722f, "luminance_weights");
    params.eps = eps;

    int status = COLOUR_FUNCTIONS_OK;
    {
        py::gil_scoped_release release;
        status = colour_functions_transform_v1(&input_image, &output_image, &params);
    }
    if (status != COLOUR_FUNCTIONS_OK) {
        throw std::runtime_error(error_message(status));
    }

    return result;
}

PYBIND11_MODULE(_colour_functions_cpu, m) {
    m.doc() = "CPU colour functions backend for Platypus";
    m.def(
        "apply_display_color_transform",
        &apply_display_color_transform,
        py::arg("image"),
        py::arg("basis"),
        py::arg("encoding"),
        py::arg("luminance_weights") = py::none(),
        py::arg("eps") = 1.0e-12f
    );
}
