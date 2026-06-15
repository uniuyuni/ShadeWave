#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "tone_capi.h"

namespace py = pybind11;

namespace {

const char* error_message(int code) {
    switch (code) {
        case TONE_OK:
            return "ok";
        case TONE_ERR_NULL:
            return "null image, output, or params";
        case TONE_ERR_SHAPE:
            return "unsupported image shape";
        case TONE_ERR_ALLOC:
            return "tone backend allocation failed";
        default:
            return "unknown tone backend error";
    }
}

}  // namespace

py::array_t<float> adjust_tone(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    float highlights,
    float shadows,
    float midtone,
    float white_level,
    float black_level,
    float disp_scale,
    float resolution_scale
) {
    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must be a 3D RGB float32 array");
    }

    std::vector<py::ssize_t> shape(in.shape.begin(), in.shape.end());
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    ToneConstImageF32 input_image{
        static_cast<const float*>(in.ptr),
        static_cast<int>(in.shape[1]),
        static_cast<int>(in.shape[0]),
        static_cast<int>(in.shape[2]),
        static_cast<int>(in.strides[0]),
    };
    ToneImageF32 output_image{
        static_cast<float*>(out.ptr),
        static_cast<int>(out.shape[1]),
        static_cast<int>(out.shape[0]),
        static_cast<int>(out.shape[2]),
        static_cast<int>(out.strides[0]),
    };
    ToneParams params{
        highlights,
        shadows,
        midtone,
        white_level,
        black_level,
        disp_scale,
        resolution_scale,
    };

    int status = TONE_OK;
    {
        py::gil_scoped_release release;
        status = tone_adjust_v1(&input_image, &output_image, &params);
    }
    if (status != TONE_OK) {
        throw std::runtime_error(error_message(status));
    }

    return result;
}

PYBIND11_MODULE(_tone_cpu, m) {
    m.doc() = "CPU tone backend for Platypus";
    m.def(
        "adjust_tone",
        &adjust_tone,
        py::arg("image"),
        py::arg("highlights") = 0.0f,
        py::arg("shadows") = 0.0f,
        py::arg("midtone") = 0.0f,
        py::arg("white_level") = 0.0f,
        py::arg("black_level") = 0.0f,
        py::arg("disp_scale") = 1.0f,
        py::arg("resolution_scale") = 1.0f
    );
}
