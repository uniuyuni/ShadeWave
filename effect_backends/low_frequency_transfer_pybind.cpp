#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "low_frequency_transfer_capi.h"

namespace py = pybind11;

namespace {

const char* error_message(int code) {
    switch (code) {
        case LOW_FREQUENCY_TRANSFER_OK:
            return "ok";
        case LOW_FREQUENCY_TRANSFER_ERR_NULL:
            return "null image, output, or params";
        case LOW_FREQUENCY_TRANSFER_ERR_SHAPE:
            return "unsupported image shape";
        case LOW_FREQUENCY_TRANSFER_ERR_ALLOC:
            return "low frequency transfer allocation failed";
        default:
            return "unknown low frequency transfer backend error";
    }
}

void validate_pair(const py::buffer_info& restored, const py::buffer_info& reference) {
    if (restored.ndim != reference.ndim) {
        throw std::invalid_argument("restored and reference must have the same rank");
    }
    if (restored.ndim != 2 && restored.ndim != 3) {
        throw std::invalid_argument("image must be a 2D or 3D float32 array");
    }
    if (restored.shape[0] != reference.shape[0] || restored.shape[1] != reference.shape[1]) {
        throw std::invalid_argument("restored and reference must have the same height and width");
    }
    if (restored.ndim == 3) {
        if (restored.shape[2] != reference.shape[2] || (restored.shape[2] != 1 && restored.shape[2] != 3)) {
            throw std::invalid_argument("3D images must have 1 or 3 channels");
        }
    }
}

int channels_of(const py::buffer_info& info) {
    return info.ndim == 2 ? 1 : static_cast<int>(info.shape[2]);
}

std::vector<py::ssize_t> shape_of(const py::buffer_info& info) {
    return std::vector<py::ssize_t>(info.shape.begin(), info.shape.end());
}

LowFrequencyTransferConstImageF32 const_view(const py::buffer_info& info) {
    return LowFrequencyTransferConstImageF32{
        static_cast<const float*>(info.ptr),
        static_cast<int>(info.shape[1]),
        static_cast<int>(info.shape[0]),
        channels_of(info),
        static_cast<int>(info.strides[0]),
    };
}

LowFrequencyTransferImageF32 output_view(const py::buffer_info& info) {
    return LowFrequencyTransferImageF32{
        static_cast<float*>(info.ptr),
        static_cast<int>(info.shape[1]),
        static_cast<int>(info.shape[0]),
        channels_of(info),
        static_cast<int>(info.strides[0]),
    };
}

void validate_lowres(const py::buffer_info& low, int channels, const char* name) {
    if (low.ndim != 2 && low.ndim != 3) {
        throw std::invalid_argument(std::string(name) + " must be a 2D or 3D float32 array");
    }
    const int low_channels = low.ndim == 2 ? 1 : static_cast<int>(low.shape[2]);
    if (low.shape[0] <= 0 || low.shape[1] <= 0 || low_channels != channels) {
        throw std::invalid_argument(std::string(name) + " has incompatible shape");
    }
}

}  // namespace

py::array_t<float> apply_low_frequency_transfer(
    py::array_t<float, py::array::c_style | py::array::forcecast> restored,
    py::array_t<float, py::array::c_style | py::array::forcecast> reference,
    float sigma,
    bool use_highlight_protection,
    float highlight_threshold,
    float highlight_transition,
    float highlight_detail_strength,
    float luminance_transfer_strength
) {
    py::buffer_info restored_info = restored.request();
    py::buffer_info reference_info = reference.request();
    validate_pair(restored_info, reference_info);

    py::array_t<float> result(shape_of(restored_info));
    py::buffer_info output_info = result.request();

    LowFrequencyTransferConstImageF32 restored_image = const_view(restored_info);
    LowFrequencyTransferConstImageF32 reference_image = const_view(reference_info);
    LowFrequencyTransferImageF32 output_image = output_view(output_info);
    LowFrequencyTransferParams params{
        sigma,
        use_highlight_protection ? 1 : 0,
        highlight_threshold,
        highlight_transition,
        highlight_detail_strength,
        luminance_transfer_strength,
    };

    int status = LOW_FREQUENCY_TRANSFER_OK;
    {
        py::gil_scoped_release release;
        status = low_frequency_transfer_apply_v1(&restored_image, &reference_image, &output_image, &params);
    }
    if (status != LOW_FREQUENCY_TRANSFER_OK) {
        throw std::runtime_error(error_message(status));
    }

    return result;
}

py::array_t<float> compose_lowres(
    py::array_t<float, py::array::c_style | py::array::forcecast> restored,
    py::array_t<float, py::array::c_style | py::array::forcecast> reference,
    py::array_t<float, py::array::c_style | py::array::forcecast> low_diff,
    py::array_t<float, py::array::c_style | py::array::forcecast> low_restored,
    bool use_highlight_protection,
    float highlight_threshold,
    float highlight_transition,
    float highlight_detail_strength,
    float luminance_transfer_strength
) {
    py::buffer_info restored_info = restored.request();
    py::buffer_info reference_info = reference.request();
    validate_pair(restored_info, reference_info);
    const int channels = channels_of(restored_info);

    py::buffer_info low_diff_info = low_diff.request();
    py::buffer_info low_restored_info = low_restored.request();
    validate_lowres(low_diff_info, channels, "low_diff");
    if (use_highlight_protection) {
        validate_lowres(low_restored_info, channels, "low_restored");
    }

    py::array_t<float> result(shape_of(restored_info));
    py::buffer_info output_info = result.request();

    LowFrequencyTransferConstImageF32 restored_image = const_view(restored_info);
    LowFrequencyTransferConstImageF32 reference_image = const_view(reference_info);
    LowFrequencyTransferConstImageF32 low_diff_image = const_view(low_diff_info);
    LowFrequencyTransferConstImageF32 low_restored_image = const_view(low_restored_info);
    LowFrequencyTransferImageF32 output_image = output_view(output_info);
    LowFrequencyTransferParams params{
        0.0f,
        use_highlight_protection ? 1 : 0,
        highlight_threshold,
        highlight_transition,
        highlight_detail_strength,
        luminance_transfer_strength,
    };

    int status = LOW_FREQUENCY_TRANSFER_OK;
    {
        py::gil_scoped_release release;
        status = low_frequency_transfer_compose_lowres_v1(
            &restored_image,
            &reference_image,
            &low_diff_image,
            &low_restored_image,
            &output_image,
            &params
        );
    }
    if (status != LOW_FREQUENCY_TRANSFER_OK) {
        throw std::runtime_error(error_message(status));
    }

    return result;
}

PYBIND11_MODULE(_low_frequency_transfer_cpu, m) {
    m.doc() = "CPU low frequency transfer backend for Platypus";
    m.def(
        "apply_low_frequency_transfer",
        &apply_low_frequency_transfer,
        py::arg("restored"),
        py::arg("reference"),
        py::arg("sigma") = 30.0f,
        py::arg("use_highlight_protection") = false,
        py::arg("highlight_threshold") = 0.0f,
        py::arg("highlight_transition") = 0.35f,
        py::arg("highlight_detail_strength") = 0.25f,
        py::arg("luminance_transfer_strength") = 1.0f
    );
    m.def(
        "compose_lowres",
        &compose_lowres,
        py::arg("restored"),
        py::arg("reference"),
        py::arg("low_diff"),
        py::arg("low_restored"),
        py::arg("use_highlight_protection") = false,
        py::arg("highlight_threshold") = 0.0f,
        py::arg("highlight_transition") = 0.35f,
        py::arg("highlight_detail_strength") = 0.25f,
        py::arg("luminance_transfer_strength") = 1.0f
    );
}
