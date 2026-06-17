#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct LowFrequencyTransferMetalParams {
    int width;
    int height;
    int channels;
    int radius;
    int use_highlight_protection;
    float highlight_threshold;
    float highlight_transition;
    float highlight_detail_strength;
    float luminance_transfer_strength;
};

constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct LowFrequencyTransferMetalParams {
    int width;
    int height;
    int channels;
    int radius;
    int use_highlight_protection;
    float highlight_threshold;
    float highlight_transition;
    float highlight_detail_strength;
    float luminance_transfer_strength;
};

static inline int reflect101(int p, int len) {
    if (len <= 1) {
        return 0;
    }
    while (p < 0 || p >= len) {
        if (p < 0) {
            p = -p;
        } else {
            p = 2 * len - p - 2;
        }
    }
    return p;
}

kernel void low_frequency_diff_horizontal(
    const device float* restored [[buffer(0)]],
    const device float* reference [[buffer(1)]],
    device float* temp [[buffer(2)]],
    const device float* kernel_weights [[buffer(3)]],
    constant LowFrequencyTransferMetalParams& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int out_base = (y * p.width + x) * p.channels;
    for (int c = 0; c < p.channels; ++c) {
        float acc = 0.0f;
        for (int k = -p.radius; k <= p.radius; ++k) {
            int sx = reflect101(x + k, p.width);
            int base = (y * p.width + sx) * p.channels + c;
            acc += (reference[base] - restored[base]) * kernel_weights[k + p.radius];
        }
        temp[out_base + c] = acc;
    }
}

kernel void low_frequency_copy_horizontal(
    const device float* input [[buffer(0)]],
    device float* temp [[buffer(1)]],
    const device float* kernel_weights [[buffer(2)]],
    constant LowFrequencyTransferMetalParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int out_base = (y * p.width + x) * p.channels;
    for (int c = 0; c < p.channels; ++c) {
        float acc = 0.0f;
        for (int k = -p.radius; k <= p.radius; ++k) {
            int sx = reflect101(x + k, p.width);
            acc += input[(y * p.width + sx) * p.channels + c] * kernel_weights[k + p.radius];
        }
        temp[out_base + c] = acc;
    }
}

kernel void low_frequency_pair_horizontal(
    const device float* restored [[buffer(0)]],
    const device float* reference [[buffer(1)]],
    device float* temp_diff [[buffer(2)]],
    device float* temp_restored [[buffer(3)]],
    const device float* kernel_weights [[buffer(4)]],
    constant LowFrequencyTransferMetalParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int out_base = (y * p.width + x) * p.channels;
    for (int c = 0; c < p.channels; ++c) {
        float acc_diff = 0.0f;
        float acc_restored = 0.0f;
        for (int k = -p.radius; k <= p.radius; ++k) {
            int sx = reflect101(x + k, p.width);
            int base = (y * p.width + sx) * p.channels + c;
            float restored_v = restored[base];
            float weight = kernel_weights[k + p.radius];
            acc_diff += (reference[base] - restored_v) * weight;
            acc_restored += restored_v * weight;
        }
        temp_diff[out_base + c] = acc_diff;
        temp_restored[out_base + c] = acc_restored;
    }
}

kernel void low_frequency_vertical(
    const device float* temp [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* kernel_weights [[buffer(2)]],
    constant LowFrequencyTransferMetalParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int out_base = (y * p.width + x) * p.channels;
    for (int c = 0; c < p.channels; ++c) {
        float acc = 0.0f;
        for (int k = -p.radius; k <= p.radius; ++k) {
            int sy = reflect101(y + k, p.height);
            acc += temp[(sy * p.width + x) * p.channels + c] * kernel_weights[k + p.radius];
        }
        output[out_base + c] = acc;
    }
}

static inline float smoothstep01(float v) {
    float t = clamp(v, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

kernel void low_frequency_diff_vertical_compose(
    const device float* temp_diff [[buffer(0)]],
    const device float* restored [[buffer(1)]],
    device float* output [[buffer(2)]],
    const device float* kernel_weights [[buffer(3)]],
    constant LowFrequencyTransferMetalParams& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int base = (y * p.width + x) * p.channels;
    if (p.channels == 1) {
        float low_diff = 0.0f;
        for (int k = -p.radius; k <= p.radius; ++k) {
            int sy = reflect101(y + k, p.height);
            low_diff += temp_diff[sy * p.width + x] * kernel_weights[k + p.radius];
        }
        output[base] = restored[base] + low_diff * clamp(p.luminance_transfer_strength, 0.0f, 1.0f);
    } else {
        float3 low_diff = float3(0.0f);
        for (int k = -p.radius; k <= p.radius; ++k) {
            int sy = reflect101(y + k, p.height);
            int sample_base = (sy * p.width + x) * 3;
            float weight = kernel_weights[k + p.radius];
            low_diff.x += temp_diff[sample_base + 0] * weight;
            low_diff.y += temp_diff[sample_base + 1] * weight;
            low_diff.z += temp_diff[sample_base + 2] * weight;
        }
        float lum = dot(low_diff, float3(0.2126f, 0.7152f, 0.0722f));
        float luma_remove = 1.0f - clamp(p.luminance_transfer_strength, 0.0f, 1.0f);
        low_diff -= lum * luma_remove;
        output[base + 0] = restored[base + 0] + low_diff.x;
        output[base + 1] = restored[base + 1] + low_diff.y;
        output[base + 2] = restored[base + 2] + low_diff.z;
    }
}

kernel void low_frequency_pair_vertical_compose(
    const device float* temp_diff [[buffer(0)]],
    const device float* temp_restored [[buffer(1)]],
    const device float* restored [[buffer(2)]],
    const device float* reference [[buffer(3)]],
    device float* output [[buffer(4)]],
    const device float* kernel_weights [[buffer(5)]],
    constant LowFrequencyTransferMetalParams& p [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int base = (y * p.width + x) * p.channels;
    float luminance = reference[base];
    if (p.channels == 3) {
        luminance = max(luminance, reference[base + 1]);
        luminance = max(luminance, reference[base + 2]);
    }
    float transition = max(p.highlight_transition, 1.0e-6f);
    float mask = smoothstep01((luminance - p.highlight_threshold) / transition);
    float alpha = mask * (1.0f - p.highlight_detail_strength);

    if (p.channels == 1) {
        float low_diff = 0.0f;
        float low_restored = 0.0f;
        for (int k = -p.radius; k <= p.radius; ++k) {
            int sy = reflect101(y + k, p.height);
            int sample_base = sy * p.width + x;
            float weight = kernel_weights[k + p.radius];
            low_diff += temp_diff[sample_base] * weight;
            low_restored += temp_restored[sample_base] * weight;
        }
        float restored_v = restored[base];
        output[base] = restored_v + low_diff * clamp(p.luminance_transfer_strength, 0.0f, 1.0f) - alpha * (restored_v - low_restored);
    } else {
        float3 low_diff = float3(0.0f);
        float3 low_restored = float3(0.0f);
        for (int k = -p.radius; k <= p.radius; ++k) {
            int sy = reflect101(y + k, p.height);
            int sample_base = (sy * p.width + x) * 3;
            float weight = kernel_weights[k + p.radius];
            low_diff.x += temp_diff[sample_base + 0] * weight;
            low_diff.y += temp_diff[sample_base + 1] * weight;
            low_diff.z += temp_diff[sample_base + 2] * weight;
            low_restored.x += temp_restored[sample_base + 0] * weight;
            low_restored.y += temp_restored[sample_base + 1] * weight;
            low_restored.z += temp_restored[sample_base + 2] * weight;
        }
        float lum = dot(low_diff, float3(0.2126f, 0.7152f, 0.0722f));
        float luma_remove = 1.0f - clamp(p.luminance_transfer_strength, 0.0f, 1.0f);
        low_diff -= lum * luma_remove;
        float3 restored_v = float3(restored[base + 0], restored[base + 1], restored[base + 2]);
        float3 out_v = restored_v + low_diff - alpha * (restored_v - low_restored);
        output[base + 0] = out_v.x;
        output[base + 1] = out_v.y;
        output[base + 2] = out_v.z;
    }
}

kernel void low_frequency_compose(
    const device float* restored [[buffer(0)]],
    const device float* reference [[buffer(1)]],
    const device float* low_diff [[buffer(2)]],
    const device float* low_restored [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant LowFrequencyTransferMetalParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int base = (y * p.width + x) * p.channels;
    float alpha = 0.0f;
    if (p.use_highlight_protection != 0) {
        float luminance = reference[base];
        if (p.channels == 3) {
            luminance = max(luminance, reference[base + 1]);
            luminance = max(luminance, reference[base + 2]);
        }
        float transition = max(p.highlight_transition, 1.0e-6f);
        float mask = smoothstep01((luminance - p.highlight_threshold) / transition);
        alpha = mask * (1.0f - p.highlight_detail_strength);
    }
    if (p.channels == 1) {
        float restored_v = restored[base];
        float value = restored_v + low_diff[base] * clamp(p.luminance_transfer_strength, 0.0f, 1.0f);
        if (p.use_highlight_protection != 0) {
            value -= alpha * (restored_v - low_restored[base]);
        }
        output[base] = value;
    } else {
        float3 diff_v = float3(low_diff[base + 0], low_diff[base + 1], low_diff[base + 2]);
        float lum = dot(diff_v, float3(0.2126f, 0.7152f, 0.0722f));
        diff_v -= lum * (1.0f - clamp(p.luminance_transfer_strength, 0.0f, 1.0f));
        float3 restored_v = float3(restored[base + 0], restored[base + 1], restored[base + 2]);
        float3 value = restored_v + diff_v;
        if (p.use_highlight_protection != 0) {
            value -= alpha * (restored_v - float3(low_restored[base + 0], low_restored[base + 1], low_restored[base + 2]));
        }
        output[base + 0] = value.x;
        output[base + 1] = value.y;
        output[base + 2] = value.z;
    }
}
)METAL";

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) {
        throw std::runtime_error("Metal function not found");
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        std::string message = error ? [[error localizedDescription] UTF8String] : "unknown Metal pipeline error";
        throw std::runtime_error(message);
    }
    return pipeline;
}

struct MetalPipelines {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> diff_horizontal;
    id<MTLComputePipelineState> copy_horizontal;
    id<MTLComputePipelineState> pair_horizontal;
    id<MTLComputePipelineState> vertical;
    id<MTLComputePipelineState> diff_vertical_compose;
    id<MTLComputePipelineState> pair_vertical_compose;
    id<MTLComputePipelineState> compose;
};

MetalPipelines& metal_pipelines() {
    static MetalPipelines state{};
    static std::once_flag once;
    static std::string init_error;

    std::call_once(once, []() {
        @autoreleasepool {
            state.device = MTLCreateSystemDefaultDevice();
            if (!state.device) {
                init_error = "Metal device is unavailable";
                return;
            }

            NSError* error = nil;
            NSString* source = [NSString stringWithUTF8String:kMetalSource];
            state.library = [state.device newLibraryWithSource:source options:nil error:&error];
            if (!state.library) {
                init_error = error ? [[error localizedDescription] UTF8String] : "unknown Metal library error";
                return;
            }

            state.queue = [state.device newCommandQueue];
            if (!state.queue) {
                init_error = "Metal command queue is unavailable";
                return;
            }

            try {
                state.diff_horizontal = make_pipeline(state.device, state.library, @"low_frequency_diff_horizontal");
                state.copy_horizontal = make_pipeline(state.device, state.library, @"low_frequency_copy_horizontal");
                state.pair_horizontal = make_pipeline(state.device, state.library, @"low_frequency_pair_horizontal");
                state.vertical = make_pipeline(state.device, state.library, @"low_frequency_vertical");
                state.diff_vertical_compose = make_pipeline(state.device, state.library, @"low_frequency_diff_vertical_compose");
                state.pair_vertical_compose = make_pipeline(state.device, state.library, @"low_frequency_pair_vertical_compose");
                state.compose = make_pipeline(state.device, state.library, @"low_frequency_compose");
            } catch (const std::exception& exc) {
                init_error = exc.what();
            }
        }
    });

    if (!init_error.empty()) {
        throw std::runtime_error(init_error);
    }
    return state;
}

void dispatch_2d(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    NSUInteger width,
    NSUInteger height
) {
    [encoder setComputePipelineState:pipeline];
    NSUInteger tw = std::max<NSUInteger>(1, pipeline.threadExecutionWidth);
    NSUInteger th = std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / tw);
    if (th > 16) {
        th = 16;
    }
    MTLSize threads_per_group = MTLSizeMake(tw, th, 1);
    MTLSize grid = MTLSizeMake(width, height, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads_per_group];
}

std::vector<float> gaussian_kernel(float sigma) {
    if (sigma <= 0.0f) {
        return {1.0f};
    }
    int radius = static_cast<int>(std::ceil(static_cast<double>(sigma) * 3.0));
    radius = std::max(radius, 1);
    std::vector<float> kernel(static_cast<size_t>(radius * 2 + 1));
    double sum = 0.0;
    const double denom = 2.0 * static_cast<double>(sigma) * static_cast<double>(sigma);
    for (int i = -radius; i <= radius; ++i) {
        double v = std::exp(-(static_cast<double>(i) * static_cast<double>(i)) / denom);
        kernel[static_cast<size_t>(i + radius)] = static_cast<float>(v);
        sum += v;
    }
    for (float& v : kernel) {
        v = static_cast<float>(static_cast<double>(v) / sum);
    }
    return kernel;
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
    if (restored_info.ndim != reference_info.ndim || (restored_info.ndim != 2 && restored_info.ndim != 3)) {
        throw std::invalid_argument("restored and reference must be matching 2D or 3D float32 arrays");
    }
    if (restored_info.shape[0] != reference_info.shape[0] || restored_info.shape[1] != reference_info.shape[1]) {
        throw std::invalid_argument("restored and reference must have matching dimensions");
    }
    const int channels = restored_info.ndim == 2 ? 1 : static_cast<int>(restored_info.shape[2]);
    if (channels != (reference_info.ndim == 2 ? 1 : static_cast<int>(reference_info.shape[2])) || (channels != 1 && channels != 3)) {
        throw std::invalid_argument("images must have 1 or 3 channels");
    }

    const int width = static_cast<int>(restored_info.shape[1]);
    const int height = static_cast<int>(restored_info.shape[0]);
    std::vector<py::ssize_t> shape(restored_info.shape.begin(), restored_info.shape.end());
    py::array_t<float> result(shape);
    py::buffer_info output_info = result.request();
    std::vector<float> kernel = gaussian_kernel(sigma);
    const int radius = static_cast<int>(kernel.size() / 2);

    @autoreleasepool {
        MetalPipelines& pipelines = metal_pipelines();
        const size_t image_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels) * sizeof(float);
        const size_t kernel_bytes = kernel.size() * sizeof(float);

        id<MTLBuffer> restored_buffer = [pipelines.device newBufferWithBytes:restored_info.ptr length:image_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> reference_buffer = [pipelines.device newBufferWithBytes:reference_info.ptr length:image_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> temp_buffer = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> temp_restored_buffer = use_highlight_protection
            ? [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared]
            : output_buffer;
        id<MTLBuffer> kernel_buffer = [pipelines.device newBufferWithBytes:kernel.data() length:kernel_bytes options:MTLResourceStorageModeShared];

        if (!restored_buffer || !reference_buffer || !temp_buffer || !output_buffer || !temp_restored_buffer || !kernel_buffer) {
            throw std::runtime_error("Metal buffer allocation failed");
        }

        LowFrequencyTransferMetalParams params{
            width,
            height,
            channels,
            radius,
            use_highlight_protection ? 1 : 0,
            highlight_threshold,
            highlight_transition,
            highlight_detail_strength,
            luminance_transfer_strength,
        };
        id<MTLBuffer> params_buffer = [pipelines.device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];

        if (use_highlight_protection) {
            id<MTLComputeCommandEncoder> pair_h = [command_buffer computeCommandEncoder];
            [pair_h setBuffer:restored_buffer offset:0 atIndex:0];
            [pair_h setBuffer:reference_buffer offset:0 atIndex:1];
            [pair_h setBuffer:temp_buffer offset:0 atIndex:2];
            [pair_h setBuffer:temp_restored_buffer offset:0 atIndex:3];
            [pair_h setBuffer:kernel_buffer offset:0 atIndex:4];
            [pair_h setBuffer:params_buffer offset:0 atIndex:5];
            dispatch_2d(pair_h, pipelines.pair_horizontal, width, height);
            [pair_h endEncoding];

            id<MTLComputeCommandEncoder> pair_v = [command_buffer computeCommandEncoder];
            [pair_v setBuffer:temp_buffer offset:0 atIndex:0];
            [pair_v setBuffer:temp_restored_buffer offset:0 atIndex:1];
            [pair_v setBuffer:restored_buffer offset:0 atIndex:2];
            [pair_v setBuffer:reference_buffer offset:0 atIndex:3];
            [pair_v setBuffer:output_buffer offset:0 atIndex:4];
            [pair_v setBuffer:kernel_buffer offset:0 atIndex:5];
            [pair_v setBuffer:params_buffer offset:0 atIndex:6];
            dispatch_2d(pair_v, pipelines.pair_vertical_compose, width, height);
            [pair_v endEncoding];
        } else {
            id<MTLComputeCommandEncoder> diff_h = [command_buffer computeCommandEncoder];
            [diff_h setBuffer:restored_buffer offset:0 atIndex:0];
            [diff_h setBuffer:reference_buffer offset:0 atIndex:1];
            [diff_h setBuffer:temp_buffer offset:0 atIndex:2];
            [diff_h setBuffer:kernel_buffer offset:0 atIndex:3];
            [diff_h setBuffer:params_buffer offset:0 atIndex:4];
            dispatch_2d(diff_h, pipelines.diff_horizontal, width, height);
            [diff_h endEncoding];

            id<MTLComputeCommandEncoder> diff_v = [command_buffer computeCommandEncoder];
            [diff_v setBuffer:temp_buffer offset:0 atIndex:0];
            [diff_v setBuffer:restored_buffer offset:0 atIndex:1];
            [diff_v setBuffer:output_buffer offset:0 atIndex:2];
            [diff_v setBuffer:kernel_buffer offset:0 atIndex:3];
            [diff_v setBuffer:params_buffer offset:0 atIndex:4];
            dispatch_2d(diff_v, pipelines.diff_vertical_compose, width, height);
            [diff_v endEncoding];
        }

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer error]) {
            throw std::runtime_error([[[command_buffer error] localizedDescription] UTF8String]);
        }

        std::memcpy(output_info.ptr, [output_buffer contents], image_bytes);
    }

    return result;
}

PYBIND11_MODULE(_low_frequency_transfer_metal, m) {
    m.doc() = "Metal exact low frequency transfer backend for Platypus";
    m.def("metal_available", []() {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            return device != nil;
        }
    });
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
}
