#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "film_process_capi.h"

namespace py = pybind11;

namespace {

const char* error_message(int code) {
    switch (code) {
        case FILM_PROCESS_OK:
            return "ok";
        case FILM_PROCESS_ERR_NULL:
            return "null image, output, or params";
        case FILM_PROCESS_ERR_SHAPE:
            return "unsupported image shape";
        case FILM_PROCESS_ERR_ALLOC:
            return "film process backend allocation failed";
        default:
            return "unknown film process backend error";
    }
}

}  // namespace

py::array_t<float> apply_film_process(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    int mode,
    float latitude,
    float contrast,
    float color_bias,
    float color_drift,
    float dye_purity,
    float crosstalk,
    float aging
) {
    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must be a 3D RGB float32 array");
    }

    std::vector<py::ssize_t> shape(in.shape.begin(), in.shape.end());
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    FilmProcessConstImageF32 input_image{
        static_cast<const float*>(in.ptr),
        static_cast<int>(in.shape[1]),
        static_cast<int>(in.shape[0]),
        static_cast<int>(in.shape[2]),
        static_cast<int>(in.strides[0]),
    };
    FilmProcessImageF32 output_image{
        static_cast<float*>(out.ptr),
        static_cast<int>(out.shape[1]),
        static_cast<int>(out.shape[0]),
        static_cast<int>(out.shape[2]),
        static_cast<int>(out.strides[0]),
    };
    FilmProcessParams params{
        mode,
        latitude,
        contrast,
        color_bias,
        color_drift,
        dye_purity,
        crosstalk,
        aging,
    };

    int status = FILM_PROCESS_OK;
    {
        py::gil_scoped_release release;
        status = film_process_apply_v1(&input_image, &output_image, &params);
    }
    if (status != FILM_PROCESS_OK) {
        throw std::runtime_error(error_message(status));
    }

    return result;
}

PYBIND11_MODULE(_film_process_cpu, m) {
    m.doc() = "CPU film process backend for Platypus";
    m.def(
        "apply_film_process",
        &apply_film_process,
        py::arg("image"),
        py::arg("mode") = 0,
        py::arg("latitude") = 0.55f,
        py::arg("contrast") = 0.50f,
        py::arg("color_bias") = 0.0f,
        py::arg("color_drift") = 0.0f,
        py::arg("dye_purity") = 0.75f,
        py::arg("crosstalk") = 0.30f,
        py::arg("aging") = 0.0f
    );
}
