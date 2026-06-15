#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "subpixel_shift_capi.h"

namespace py = pybind11;

namespace {

const char* error_message(int code) {
    switch (code) {
        case SUBPIXEL_SHIFT_OK:
            return "ok";
        case SUBPIXEL_SHIFT_ERR_NULL:
            return "null image, output, or params";
        case SUBPIXEL_SHIFT_ERR_SHAPE:
            return "unsupported image shape";
        default:
            return "unknown subpixel shift backend error";
    }
}

SubpixelShiftConstImageF32 input_view(const py::buffer_info& info) {
    return SubpixelShiftConstImageF32{
        static_cast<const float*>(info.ptr),
        static_cast<int>(info.shape[1]),
        static_cast<int>(info.shape[0]),
        static_cast<int>(info.shape[2]),
        static_cast<int>(info.strides[0]),
    };
}

SubpixelShiftImageF32 output_view(const py::buffer_info& info) {
    return SubpixelShiftImageF32{
        static_cast<float*>(info.ptr),
        static_cast<int>(info.shape[1]),
        static_cast<int>(info.shape[0]),
        static_cast<int>(info.shape[2]),
        static_cast<int>(info.strides[0]),
    };
}

py::array_t<float> make_result_like(const py::buffer_info& in) {
    std::vector<py::ssize_t> shape(in.shape.begin(), in.shape.end());
    return py::array_t<float>(shape);
}

void validate_rgb(const py::buffer_info& in) {
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must be a 3D RGB float32 array");
    }
}

}  // namespace

py::array_t<float> subpixel_shift(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    float shift_x,
    float shift_y
) {
    py::buffer_info in = image.request();
    validate_rgb(in);
    py::array_t<float> result = make_result_like(in);
    py::buffer_info out = result.request();

    SubpixelShiftConstImageF32 input_image = input_view(in);
    SubpixelShiftImageF32 output_image = output_view(out);
    SubpixelShiftParams params{shift_x, shift_y};

    int status = SUBPIXEL_SHIFT_OK;
    {
        py::gil_scoped_release release;
        status = subpixel_shift_apply_v1(&input_image, &output_image, &params);
    }
    if (status != SUBPIXEL_SHIFT_OK) {
        throw std::runtime_error(error_message(status));
    }

    return result;
}

py::array_t<float> create_enhanced_image(
    py::array_t<float, py::array::c_style | py::array::forcecast> image
) {
    py::buffer_info in = image.request();
    validate_rgb(in);
    py::array_t<float> result = make_result_like(in);
    py::buffer_info out = result.request();

    SubpixelShiftConstImageF32 input_image = input_view(in);
    SubpixelShiftImageF32 output_image = output_view(out);

    int status = SUBPIXEL_SHIFT_OK;
    {
        py::gil_scoped_release release;
        status = subpixel_shift_enhance_v1(&input_image, &output_image);
    }
    if (status != SUBPIXEL_SHIFT_OK) {
        throw std::runtime_error(error_message(status));
    }

    return result;
}

PYBIND11_MODULE(_subpixel_shift_cpu, m) {
    m.doc() = "CPU subpixel shift backend for Platypus";
    m.def(
        "subpixel_shift",
        &subpixel_shift,
        py::arg("image"),
        py::arg("shift_x") = 0.5f,
        py::arg("shift_y") = 0.5f
    );
    m.def(
        "create_enhanced_image",
        &create_enhanced_image,
        py::arg("image")
    );
}
