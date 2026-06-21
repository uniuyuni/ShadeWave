#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "film_grain_capi.h"

namespace py = pybind11;

namespace {

const char* error_message(int code) {
    switch (code) {
        case FILM_GRAIN_OK:
            return "ok";
        case FILM_GRAIN_ERR_NULL:
            return "null image, output, or params";
        case FILM_GRAIN_ERR_SHAPE:
            return "unsupported image shape";
        case FILM_GRAIN_ERR_ALLOC:
            return "film grain backend allocation failed";
        default:
            return "unknown film grain backend error";
    }
}

}  // namespace

py::array_t<float> apply_film_grain(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    float amount,
    float grain_size,
    float roughness,
    float shadow,
    float highlight,
    float color,
    int seed
) {
    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] < 3) {
        throw std::invalid_argument("image must be a 3D RGB/RGBA float32 array");
    }

    std::vector<py::ssize_t> shape(in.shape.begin(), in.shape.end());
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    FilmGrainConstImageF32 input_image{
        static_cast<const float*>(in.ptr),
        static_cast<int>(in.shape[1]),
        static_cast<int>(in.shape[0]),
        static_cast<int>(in.shape[2]),
        static_cast<int>(in.strides[0]),
    };
    FilmGrainImageF32 output_image{
        static_cast<float*>(out.ptr),
        static_cast<int>(out.shape[1]),
        static_cast<int>(out.shape[0]),
        static_cast<int>(out.shape[2]),
        static_cast<int>(out.strides[0]),
    };
    FilmGrainParams params{
        amount,
        grain_size,
        roughness,
        shadow,
        highlight,
        color,
        seed,
    };

    int status = FILM_GRAIN_OK;
    {
        py::gil_scoped_release release;
        status = film_grain_apply_v1(&input_image, &output_image, &params);
    }
    if (status != FILM_GRAIN_OK) {
        throw std::runtime_error(error_message(status));
    }

    return result;
}

PYBIND11_MODULE(_film_grain_cpu, m) {
    m.doc() = "CPU film grain backend for Platypus";
    m.def(
        "apply_film_grain",
        &apply_film_grain,
        py::arg("image"),
        py::arg("amount") = 0.0f,
        py::arg("grain_size") = 2.0f,
        py::arg("roughness") = 50.0f,
        py::arg("shadow") = 60.0f,
        py::arg("highlight") = 30.0f,
        py::arg("color") = 10.0f,
        py::arg("seed") = 0
    );
}
