#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "vignette_capi.h"

namespace py = pybind11;

namespace {

float seq_float(const py::object& obj, py::ssize_t index, const char* name) {
    py::sequence seq = py::cast<py::sequence>(obj);
    if (seq.size() <= index) {
        throw std::invalid_argument(std::string(name) + " is shorter than expected");
    }
    return py::cast<float>(seq[index]);
}

const char* error_message(int code) {
    switch (code) {
        case VIGNETTE_OK:
            return "ok";
        case VIGNETTE_ERR_NULL:
            return "null image, output, or params";
        case VIGNETTE_ERR_SHAPE:
            return "unsupported image shape";
        default:
            return "unknown vignette backend error";
    }
}

}  // namespace

py::array_t<float> apply_vignette(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    float intensity,
    float radius_percent,
    const py::object& disp_info,
    const py::object& crop_rect,
    const py::object& offset,
    float gradient_softness
) {
    py::buffer_info in = image.request();
    if (in.ndim != 2 && in.ndim != 3) {
        throw std::invalid_argument("image must be a 2D grayscale or 3D RGB float32 array");
    }
    if (in.ndim == 3 && in.shape[2] != 3) {
        throw std::invalid_argument("3D image must have exactly 3 channels");
    }

    std::vector<py::ssize_t> shape(in.shape.begin(), in.shape.end());
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    VignetteImageF32 input_image{
        static_cast<float*>(in.ptr),
        static_cast<int>(in.shape[1]),
        static_cast<int>(in.shape[0]),
        in.ndim == 3 ? static_cast<int>(in.shape[2]) : 1,
        static_cast<int>(in.strides[0]),
    };
    VignetteImageF32 output_image{
        static_cast<float*>(out.ptr),
        static_cast<int>(out.shape[1]),
        static_cast<int>(out.shape[0]),
        out.ndim == 3 ? static_cast<int>(out.shape[2]) : 1,
        static_cast<int>(out.strides[0]),
    };
    VignetteParams params{};
    params.intensity = intensity;
    params.radius_percent = radius_percent;
    params.gradient_softness = gradient_softness;
    for (int i = 0; i < 5; ++i) {
        params.disp_info[i] = seq_float(disp_info, i, "disp_info");
    }
    for (int i = 0; i < 4; ++i) {
        params.crop_rect[i] = seq_float(crop_rect, i, "crop_rect");
    }
    for (int i = 0; i < 2; ++i) {
        params.offset[i] = seq_float(offset, i, "offset");
    }

    int status = VIGNETTE_OK;
    {
        py::gil_scoped_release release;
        status = vignette_apply_v1(&input_image, &output_image, &params);
    }
    if (status != VIGNETTE_OK) {
        throw std::runtime_error(error_message(status));
    }

    return result;
}

py::array_t<float> create_vignette_mask(
    int height,
    int width,
    float radius_percent,
    const py::object& disp_info,
    const py::object& crop_rect,
    const py::object& offset,
    float gradient_softness
) {
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument("height and width must be positive");
    }

    py::array_t<float> result({height, width});
    py::buffer_info out = result.request();

    VignetteImageF32 output_image{
        static_cast<float*>(out.ptr),
        width,
        height,
        1,
        static_cast<int>(out.strides[0]),
    };
    VignetteParams params{};
    params.intensity = 0.0f;
    params.radius_percent = radius_percent;
    params.gradient_softness = gradient_softness;
    for (int i = 0; i < 5; ++i) {
        params.disp_info[i] = seq_float(disp_info, i, "disp_info");
    }
    for (int i = 0; i < 4; ++i) {
        params.crop_rect[i] = seq_float(crop_rect, i, "crop_rect");
    }
    for (int i = 0; i < 2; ++i) {
        params.offset[i] = seq_float(offset, i, "offset");
    }

    int status = VIGNETTE_OK;
    {
        py::gil_scoped_release release;
        status = vignette_create_mask_v1(&output_image, &params);
    }
    if (status != VIGNETTE_OK) {
        throw std::runtime_error(error_message(status));
    }

    return result;
}

py::array_t<float> apply_vignette_mask(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    py::array_t<float, py::array::c_style | py::array::forcecast> mask,
    float intensity
) {
    py::buffer_info in = image.request();
    py::buffer_info mask_info = mask.request();
    if (in.ndim != 2 && in.ndim != 3) {
        throw std::invalid_argument("image must be a 2D grayscale or 3D RGB float32 array");
    }
    if (in.ndim == 3 && in.shape[2] != 3) {
        throw std::invalid_argument("3D image must have exactly 3 channels");
    }
    if (mask_info.ndim != 2 || mask_info.shape[0] != in.shape[0] || mask_info.shape[1] != in.shape[1]) {
        throw std::invalid_argument("mask must be a 2D float32 array matching image height and width");
    }

    std::vector<py::ssize_t> shape(in.shape.begin(), in.shape.end());
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    VignetteImageF32 input_image{
        static_cast<float*>(in.ptr),
        static_cast<int>(in.shape[1]),
        static_cast<int>(in.shape[0]),
        in.ndim == 3 ? static_cast<int>(in.shape[2]) : 1,
        static_cast<int>(in.strides[0]),
    };
    VignetteImageF32 mask_image{
        static_cast<float*>(mask_info.ptr),
        static_cast<int>(mask_info.shape[1]),
        static_cast<int>(mask_info.shape[0]),
        1,
        static_cast<int>(mask_info.strides[0]),
    };
    VignetteImageF32 output_image{
        static_cast<float*>(out.ptr),
        static_cast<int>(out.shape[1]),
        static_cast<int>(out.shape[0]),
        out.ndim == 3 ? static_cast<int>(out.shape[2]) : 1,
        static_cast<int>(out.strides[0]),
    };

    int status = VIGNETTE_OK;
    {
        py::gil_scoped_release release;
        status = vignette_apply_mask_v1(&input_image, &mask_image, &output_image, intensity);
    }
    if (status != VIGNETTE_OK) {
        throw std::runtime_error(error_message(status));
    }

    return result;
}

PYBIND11_MODULE(_vignette_cpu, m) {
    m.doc() = "CPU Vignette backend for Platypus";
    m.def(
        "apply_vignette",
        &apply_vignette,
        py::arg("image"),
        py::arg("intensity"),
        py::arg("radius_percent"),
        py::arg("disp_info"),
        py::arg("crop_rect"),
        py::arg("offset"),
        py::arg("gradient_softness") = 4.0f
    );
    m.def(
        "create_vignette_mask",
        &create_vignette_mask,
        py::arg("height"),
        py::arg("width"),
        py::arg("radius_percent"),
        py::arg("disp_info"),
        py::arg("crop_rect"),
        py::arg("offset"),
        py::arg("gradient_softness") = 4.0f
    );
    m.def(
        "apply_vignette_mask",
        &apply_vignette_mask,
        py::arg("image"),
        py::arg("mask"),
        py::arg("intensity")
    );
}
