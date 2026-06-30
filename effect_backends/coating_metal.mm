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

struct CoatingParams {
    int width;
    int height;
    int radius;
    float flare_factor;
    float contrast_factor;
    float saturation_factor;
    float m00;
    float m01;
    float m02;
    float m10;
    float m11;
    float m12;
    float m20;
    float m21;
    float m22;
};

constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct CoatingParams {
    int width;
    int height;
    int radius;
    float flare_factor;
    float contrast_factor;
    float saturation_factor;
    float m00;
    float m01;
    float m02;
    float m10;
    float m11;
    float m12;
    float m20;
    float m21;
    float m22;
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

kernel void coating_color_luma(
    const device float* input [[buffer(0)]],
    device float* colored [[buffer(1)]],
    device float* luma [[buffer(2)]],
    constant CoatingParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    int base = int(gid) * 3;
    float r = input[base];
    float g = input[base + 1];
    float b = input[base + 2];
    float cr = p.m00 * r + p.m01 * g + p.m02 * b;
    float cg = p.m10 * r + p.m11 * g + p.m12 * b;
    float cb = p.m20 * r + p.m21 * g + p.m22 * b;
    colored[base] = cr;
    colored[base + 1] = cg;
    colored[base + 2] = cb;
    luma[gid] = (cr + cg + cb) / 3.0f;
}

kernel void gaussian_plane_horizontal(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant CoatingParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    float sum = 0.0f;
    for (int k = -p.radius; k <= p.radius; ++k) {
        int sx = reflect101(x + k, p.width);
        sum += input[y * p.width + sx] * weights[k + p.radius];
    }
    output[y * p.width + x] = sum;
}

kernel void gaussian_plane_vertical(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant CoatingParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    float sum = 0.0f;
    for (int k = -p.radius; k <= p.radius; ++k) {
        int sy = reflect101(y + k, p.height);
        sum += input[sy * p.width + x] * weights[k + p.radius];
    }
    output[y * p.width + x] = sum;
}

kernel void coating_glare_luma(
    const device float* colored [[buffer(0)]],
    const device float* glow [[buffer(1)]],
    device float* glare [[buffer(2)]],
    device float* luma [[buffer(3)]],
    constant CoatingParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    int base = int(gid) * 3;
    float flare_intensity = p.flare_factor * 0.2f;
    float gv = glow[gid];
    float r = colored[base] + gv * flare_intensity;
    float g = colored[base + 1] + gv * 0.95f * flare_intensity;
    float b = colored[base + 2] + gv * 0.90f * flare_intensity;
    glare[base] = r;
    glare[base + 1] = g;
    glare[base + 2] = b;
    luma[gid] = (r + g + b) / 3.0f;
}

kernel void coating_micro_saturation(
    const device float* glare [[buffer(0)]],
    const device float* luma [[buffer(1)]],
    const device float* blurred_luma [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant CoatingParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    int base = int(gid) * 3;
    float y = luma[gid];
    float enhanced = blurred_luma[gid] + (y - blurred_luma[gid]) * p.contrast_factor;
    float ratio = enhanced / (y + 1.0e-6f);
    float r = glare[base] * ratio;
    float g = glare[base + 1] * ratio;
    float b = glare[base + 2] * ratio;
    float y2 = (r + g + b) / 3.0f;
    output[base] = y2 + (r - y2) * p.saturation_factor;
    output[base + 1] = y2 + (g - y2) * p.saturation_factor;
    output[base + 2] = y2 + (b - y2) * p.saturation_factor;
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
    id<MTLComputePipelineState> color_luma;
    id<MTLComputePipelineState> gaussian_h;
    id<MTLComputePipelineState> gaussian_v;
    id<MTLComputePipelineState> glare_luma;
    id<MTLComputePipelineState> micro_saturation;
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
                state.color_luma = make_pipeline(state.device, state.library, @"coating_color_luma");
                state.gaussian_h = make_pipeline(state.device, state.library, @"gaussian_plane_horizontal");
                state.gaussian_v = make_pipeline(state.device, state.library, @"gaussian_plane_vertical");
                state.glare_luma = make_pipeline(state.device, state.library, @"coating_glare_luma");
                state.micro_saturation = make_pipeline(state.device, state.library, @"coating_micro_saturation");
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

void dispatch_1d(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, int count) {
    NSUInteger tw = pipeline.threadExecutionWidth;
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
}

void dispatch_2d(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, int width, int height) {
    NSUInteger tw = pipeline.threadExecutionWidth;
    NSUInteger th = std::max<NSUInteger>(1, std::min<NSUInteger>(16, pipeline.maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(1, tw)));
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(width), static_cast<NSUInteger>(height), 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

std::vector<float> gaussian_weights(float sigma) {
    sigma = std::max(0.01f, sigma);
    int radius = std::max(1, static_cast<int>(std::lround(4.0f * sigma)));
    std::vector<float> weights(static_cast<size_t>(radius * 2 + 1));
    double sum = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        double v = std::exp(-(double(k) * double(k)) / (2.0 * double(sigma) * double(sigma)));
        weights[static_cast<size_t>(k + radius)] = static_cast<float>(v);
        sum += v;
    }
    for (float& v : weights) {
        v = static_cast<float>(double(v) / sum);
    }
    return weights;
}

}  // namespace

py::array_t<float> apply_coating(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    py::array_t<float, py::array::c_style | py::array::forcecast> matrix,
    float flare_factor,
    float contrast_factor,
    float saturation_factor,
    float resolution_scale
) {
    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must have shape (height, width, 3)");
    }
    py::buffer_info mat = matrix.request();
    if (mat.size != 9) {
        throw std::invalid_argument("matrix must contain 9 floats");
    }
    const int height = static_cast<int>(in.shape[0]);
    const int width = static_cast<int>(in.shape[1]);
    const int count = width * height;
    const size_t plane_bytes = static_cast<size_t>(count) * sizeof(float);
    const size_t image_bytes = plane_bytes * 3;
    const float* m = static_cast<const float*>(mat.ptr);

    py::array_t<float> result({height, width, 3});
    py::buffer_info out = result.request();

    {
        py::gil_scoped_release release;
        @autoreleasepool {
            MetalPipelines& pipelines = metal_pipelines();
            id<MTLBuffer> input_buffer = [pipelines.device newBufferWithBytes:in.ptr length:image_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> colored = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> luma1 = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> flare_tmp = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> flare_glow = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> glare = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> luma2 = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> micro_tmp = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> micro_blur = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buffer = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];

            CoatingParams base_params{
                width, height, 0,
                flare_factor, contrast_factor, saturation_factor,
                m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
            };
            id<MTLBuffer> base_params_buffer = [pipelines.device newBufferWithBytes:&base_params length:sizeof(base_params) options:MTLResourceStorageModeShared];
            std::vector<float> flare_weights = gaussian_weights(std::max(1.0f, 50.0f * resolution_scale));
            std::vector<float> micro_weights = gaussian_weights(std::max(1.0f, 10.0f * resolution_scale));
            CoatingParams flare_params = base_params;
            flare_params.radius = static_cast<int>(flare_weights.size() / 2);
            CoatingParams micro_params = base_params;
            micro_params.radius = static_cast<int>(micro_weights.size() / 2);
            id<MTLBuffer> flare_params_buffer = [pipelines.device newBufferWithBytes:&flare_params length:sizeof(flare_params) options:MTLResourceStorageModeShared];
            id<MTLBuffer> micro_params_buffer = [pipelines.device newBufferWithBytes:&micro_params length:sizeof(micro_params) options:MTLResourceStorageModeShared];
            id<MTLBuffer> flare_weights_buffer = [pipelines.device newBufferWithBytes:flare_weights.data() length:flare_weights.size() * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> micro_weights_buffer = [pipelines.device newBufferWithBytes:micro_weights.data() length:micro_weights.size() * sizeof(float) options:MTLResourceStorageModeShared];
            if (!input_buffer || !colored || !luma1 || !flare_tmp || !flare_glow || !glare || !luma2 || !micro_tmp || !micro_blur || !output_buffer || !base_params_buffer || !flare_params_buffer || !micro_params_buffer || !flare_weights_buffer || !micro_weights_buffer) {
                throw std::runtime_error("failed to allocate Metal coating buffers");
            }

            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.color_luma];
                [enc setBuffer:input_buffer offset:0 atIndex:0];
                [enc setBuffer:colored offset:0 atIndex:1];
                [enc setBuffer:luma1 offset:0 atIndex:2];
                [enc setBuffer:base_params_buffer offset:0 atIndex:3];
                dispatch_1d(enc, pipelines.color_luma, count);
                [enc endEncoding];
            }
            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.gaussian_h];
                [enc setBuffer:luma1 offset:0 atIndex:0];
                [enc setBuffer:flare_tmp offset:0 atIndex:1];
                [enc setBuffer:flare_weights_buffer offset:0 atIndex:2];
                [enc setBuffer:flare_params_buffer offset:0 atIndex:3];
                dispatch_2d(enc, pipelines.gaussian_h, width, height);
                [enc endEncoding];

                enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.gaussian_v];
                [enc setBuffer:flare_tmp offset:0 atIndex:0];
                [enc setBuffer:flare_glow offset:0 atIndex:1];
                [enc setBuffer:flare_weights_buffer offset:0 atIndex:2];
                [enc setBuffer:flare_params_buffer offset:0 atIndex:3];
                dispatch_2d(enc, pipelines.gaussian_v, width, height);
                [enc endEncoding];
            }
            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.glare_luma];
                [enc setBuffer:colored offset:0 atIndex:0];
                [enc setBuffer:flare_glow offset:0 atIndex:1];
                [enc setBuffer:glare offset:0 atIndex:2];
                [enc setBuffer:luma2 offset:0 atIndex:3];
                [enc setBuffer:base_params_buffer offset:0 atIndex:4];
                dispatch_1d(enc, pipelines.glare_luma, count);
                [enc endEncoding];
            }
            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.gaussian_h];
                [enc setBuffer:luma2 offset:0 atIndex:0];
                [enc setBuffer:micro_tmp offset:0 atIndex:1];
                [enc setBuffer:micro_weights_buffer offset:0 atIndex:2];
                [enc setBuffer:micro_params_buffer offset:0 atIndex:3];
                dispatch_2d(enc, pipelines.gaussian_h, width, height);
                [enc endEncoding];

                enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.gaussian_v];
                [enc setBuffer:micro_tmp offset:0 atIndex:0];
                [enc setBuffer:micro_blur offset:0 atIndex:1];
                [enc setBuffer:micro_weights_buffer offset:0 atIndex:2];
                [enc setBuffer:micro_params_buffer offset:0 atIndex:3];
                dispatch_2d(enc, pipelines.gaussian_v, width, height);
                [enc endEncoding];
            }
            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.micro_saturation];
                [enc setBuffer:glare offset:0 atIndex:0];
                [enc setBuffer:luma2 offset:0 atIndex:1];
                [enc setBuffer:micro_blur offset:0 atIndex:2];
                [enc setBuffer:output_buffer offset:0 atIndex:3];
                [enc setBuffer:base_params_buffer offset:0 atIndex:4];
                dispatch_1d(enc, pipelines.micro_saturation, count);
                [enc endEncoding];
            }

            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.error) {
                std::string message = [[command_buffer.error localizedDescription] UTF8String];
                throw std::runtime_error(message);
            }
            std::memcpy(out.ptr, [output_buffer contents], image_bytes);
        }
    }

    return result;
}

bool metal_available() {
    try {
        (void)metal_pipelines();
        return true;
    } catch (...) {
        return false;
    }
}

PYBIND11_MODULE(_coating_metal, m) {
    m.def("apply_coating", &apply_coating, py::arg("image"), py::arg("matrix"), py::arg("flare_factor"), py::arg("contrast_factor"), py::arg("saturation_factor"), py::arg("resolution_scale"));
    m.def("metal_available", &metal_available);
}
