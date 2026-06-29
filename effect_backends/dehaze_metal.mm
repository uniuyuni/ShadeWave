#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct DehazeParams {
    int width;
    int height;
    int radius;
    float strength;
    float a0;
    float a1;
    float a2;
    float beta0;
    float beta1;
    float beta2;
    float depth_lo;
    float depth_hi;
};

constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct DehazeParams {
    int width;
    int height;
    int radius;
    float strength;
    float a0;
    float a1;
    float a2;
    float beta0;
    float beta1;
    float beta2;
    float depth_lo;
    float depth_hi;
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

static inline float smoothstep_safe(float e0, float e1, float x) {
    float t = clamp((x - e0) / (e1 - e0 + 1.0e-12f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

kernel void dehaze_depth_raw(
    const device float* input [[buffer(0)]],
    device float* depth [[buffer(1)]],
    constant DehazeParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    int count = p.width * p.height;
    if (idx >= count) {
        return;
    }
    int base = idx * 3;
    float r = max(input[base], 0.0f);
    float g = max(input[base + 1], 0.0f);
    float b = max(input[base + 2], 0.0f);
    float max_v = max(max(r, g), b);
    float min_v = min(min(r, g), b);
    float gain = max(max_v, 1.0f);
    float l = max_v / gain;
    float s = max_v <= 1.0e-9f ? 0.0f : (max_v - min_v) / (max_v + 0.005f);
    depth[idx] = p.beta0 + p.beta1 * l + p.beta2 * s;
}

kernel void gaussian_horizontal(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant DehazeParams& p [[buffer(3)]],
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

kernel void gaussian_vertical_normalize_depth(
    const device float* input [[buffer(0)]],
    device float* depth [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant DehazeParams& p [[buffer(3)]],
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
    float d = (sum - p.depth_lo) / (p.depth_hi - p.depth_lo + 1.0e-8f);
    depth[y * p.width + x] = clamp(d, 0.0f, 1.0f);
}

kernel void dehaze_apply(
    const device float* input [[buffer(0)]],
    const device float* depth [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant DehazeParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    int count = p.width * p.height;
    if (idx >= count) {
        return;
    }
    int base = idx * 3;
    float d = depth[idx];
    float t = max(exp(-p.strength * d), 0.1f);
    float a[3] = {p.a0, p.a1, p.a2};
    float dehazed[3];
    for (int c = 0; c < 3; ++c) {
        dehazed[c] = (input[base + c] - a[c]) / t + a[c];
    }

    float r = max(input[base], 0.0f);
    float g = max(input[base + 1], 0.0f);
    float b = max(input[base + 2], 0.0f);
    float y = 0.299f * r + 0.587f * g + 0.114f * b;
    float shadow_end = 0.10f + 0.20f * clamp(p.strength, 0.0f, 1.0f);
    float amount = smoothstep_safe(0.005f, shadow_end, y);
    for (int c = 0; c < 3; ++c) {
        float v = input[base + c] + (dehazed[c] - input[base + c]) * amount;
        output[base + c] = max(v, 0.0f);
    }
}

kernel void haze_add(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant DehazeParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    int count = p.width * p.height;
    if (idx >= count) {
        return;
    }
    float haze_strength = -p.strength;
    float min_trans = 0.4f;
    float t = 1.0f - (1.0f - min_trans) * (haze_strength * haze_strength);
    float inv_t = 1.0f - t;
    int base = idx * 3;
    output[base] = input[base] * t + inv_t;
    output[base + 1] = input[base + 1] * t + inv_t;
    output[base + 2] = input[base + 2] * t + inv_t;
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
    id<MTLComputePipelineState> depth_raw;
    id<MTLComputePipelineState> gaussian_horizontal;
    id<MTLComputePipelineState> gaussian_vertical_normalize_depth;
    id<MTLComputePipelineState> dehaze_apply;
    id<MTLComputePipelineState> haze_add;
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
                state.depth_raw = make_pipeline(state.device, state.library, @"dehaze_depth_raw");
                state.gaussian_horizontal = make_pipeline(state.device, state.library, @"gaussian_horizontal");
                state.gaussian_vertical_normalize_depth = make_pipeline(state.device, state.library, @"gaussian_vertical_normalize_depth");
                state.dehaze_apply = make_pipeline(state.device, state.library, @"dehaze_apply");
                state.haze_add = make_pipeline(state.device, state.library, @"haze_add");
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

void dispatch_1d(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, NSUInteger count) {
    [encoder setComputePipelineState:pipeline];
    NSUInteger tpg = std::max<NSUInteger>(1, pipeline.threadExecutionWidth);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

void dispatch_2d(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, NSUInteger width, NSUInteger height) {
    [encoder setComputePipelineState:pipeline];
    NSUInteger tw = std::max<NSUInteger>(1, pipeline.threadExecutionWidth);
    NSUInteger th = std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / tw);
    if (th > 16) {
        th = 16;
    }
    [encoder dispatchThreads:MTLSizeMake(width, height, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

std::vector<float> gaussian_kernel(float sigma) {
    int ksize = static_cast<int>(std::round(static_cast<double>(sigma) * 6.0 + 1.0));
    if ((ksize & 1) == 0) {
        ++ksize;
    }
    ksize = std::max(3, ksize);
    int radius = ksize / 2;
    std::vector<float> kernel(static_cast<size_t>(ksize));
    double sum = 0.0;
    double denom = 2.0 * static_cast<double>(sigma) * static_cast<double>(sigma);
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

struct PreparedImage {
    py::array_t<float> result;
    py::buffer_info in;
    py::buffer_info out;
    int width;
    int height;
    int count;
    size_t image_bytes;
    size_t plane_bytes;
};

PreparedImage prepare_image(py::array_t<float, py::array::c_style | py::array::forcecast> image) {
    PreparedImage prepared{};
    prepared.in = image.request();
    if (prepared.in.ndim != 3 || prepared.in.shape[2] != 3) {
        throw std::invalid_argument("image must have shape (H, W, 3)");
    }
    prepared.width = static_cast<int>(prepared.in.shape[1]);
    prepared.height = static_cast<int>(prepared.in.shape[0]);
    prepared.count = prepared.width * prepared.height;
    prepared.image_bytes = static_cast<size_t>(prepared.count) * 3 * sizeof(float);
    prepared.plane_bytes = static_cast<size_t>(prepared.count) * sizeof(float);
    prepared.result = py::array_t<float>({prepared.height, prepared.width, 3});
    prepared.out = prepared.result.request();
    return prepared;
}

DehazeParams base_params(int width, int height, float strength) {
    const float beta0 = 0.121779f;
    const float beta1 = 0.959710f;
    const float beta2 = -0.780245f;
    const float depth_lo = beta0 + std::min(beta1, 0.0f) + std::min(beta2, 0.0f);
    const float depth_hi = beta0 + std::max(beta1, 0.0f) + std::max(beta2, 0.0f);
    return DehazeParams{
        width,
        height,
        1,
        strength,
        0.0f,
        0.0f,
        0.0f,
        beta0,
        beta1,
        beta2,
        depth_lo,
        depth_hi,
    };
}

id<MTLBuffer> make_params(id<MTLDevice> device, const DehazeParams& params) {
    return [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
}

std::array<float, 3> estimate_atmospheric_light_cpu(
    const float* image,
    const float* depth,
    int count
) {
    int n = static_cast<int>(static_cast<double>(count) * 0.001);
    if (n <= 0) {
        n = 1;
    }
    std::vector<int> indices(static_cast<size_t>(count));
    std::iota(indices.begin(), indices.end(), 0);
    auto by_depth = [depth](int lhs, int rhs) {
        return depth[lhs] > depth[rhs];
    };
    if (n < count) {
        std::nth_element(indices.begin(), indices.begin() + n, indices.end(), by_depth);
    }
    double acc0 = 0.0;
    double acc1 = 0.0;
    double acc2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const int base = indices[static_cast<size_t>(i)] * 3;
        acc0 += image[base];
        acc1 += image[base + 1];
        acc2 += image[base + 2];
    }
    const double inv = 1.0 / static_cast<double>(n);
    return {
        static_cast<float>(acc0 * inv),
        static_cast<float>(acc1 * inv),
        static_cast<float>(acc2 * inv),
    };
}

}  // namespace

bool metal_available() {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

py::array_t<float> dehaze_image(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    float strength
) {
    PreparedImage prepared = prepare_image(image);
    @autoreleasepool {
        MetalPipelines& pipelines = metal_pipelines();
        id<MTLBuffer> input = [pipelines.device newBufferWithBytes:prepared.in.ptr length:prepared.image_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> output = [pipelines.device newBufferWithLength:prepared.image_bytes options:MTLResourceStorageModeShared];
        if (!input || !output) {
            throw std::runtime_error("failed to allocate Metal dehaze image buffers");
        }

        DehazeParams params = base_params(prepared.width, prepared.height, strength);

        if (strength < 0.0f) {
            id<MTLBuffer> params_buffer = make_params(pipelines.device, params);
            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            [enc setBuffer:input offset:0 atIndex:0];
            [enc setBuffer:output offset:0 atIndex:1];
            [enc setBuffer:params_buffer offset:0 atIndex:2];
            dispatch_1d(enc, pipelines.haze_add, static_cast<NSUInteger>(prepared.count));
            [enc endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            std::memcpy(prepared.out.ptr, [output contents], prepared.image_bytes);
            return prepared.result;
        }

        std::vector<float> kernel = gaussian_kernel(0.5f);
        params.radius = static_cast<int>(kernel.size() / 2);
        id<MTLBuffer> depth_raw = [pipelines.device newBufferWithLength:prepared.plane_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> depth_temp = [pipelines.device newBufferWithLength:prepared.plane_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> depth = [pipelines.device newBufferWithLength:prepared.plane_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> kernel_buffer = [pipelines.device newBufferWithBytes:kernel.data() length:kernel.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> params_buffer = make_params(pipelines.device, params);
        if (!depth_raw || !depth_temp || !depth || !kernel_buffer || !params_buffer) {
            throw std::runtime_error("failed to allocate Metal dehaze depth buffers");
        }

        {
            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setBuffer:input offset:0 atIndex:0];
                [enc setBuffer:depth_raw offset:0 atIndex:1];
                [enc setBuffer:params_buffer offset:0 atIndex:2];
                dispatch_1d(enc, pipelines.depth_raw, static_cast<NSUInteger>(prepared.count));
                [enc endEncoding];
            }
            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setBuffer:depth_raw offset:0 atIndex:0];
                [enc setBuffer:depth_temp offset:0 atIndex:1];
                [enc setBuffer:kernel_buffer offset:0 atIndex:2];
                [enc setBuffer:params_buffer offset:0 atIndex:3];
                dispatch_2d(enc, pipelines.gaussian_horizontal, static_cast<NSUInteger>(prepared.width), static_cast<NSUInteger>(prepared.height));
                [enc endEncoding];
            }
            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setBuffer:depth_temp offset:0 atIndex:0];
                [enc setBuffer:depth offset:0 atIndex:1];
                [enc setBuffer:kernel_buffer offset:0 atIndex:2];
                [enc setBuffer:params_buffer offset:0 atIndex:3];
                dispatch_2d(enc, pipelines.gaussian_vertical_normalize_depth, static_cast<NSUInteger>(prepared.width), static_cast<NSUInteger>(prepared.height));
                [enc endEncoding];
            }
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
        }

        const float* image_ptr = static_cast<const float*>(prepared.in.ptr);
        const float* depth_ptr = static_cast<const float*>([depth contents]);
        std::array<float, 3> a = estimate_atmospheric_light_cpu(image_ptr, depth_ptr, prepared.count);
        params.a0 = a[0];
        params.a1 = a[1];
        params.a2 = a[2];
        id<MTLBuffer> apply_params_buffer = make_params(pipelines.device, params);
        if (!apply_params_buffer) {
            throw std::runtime_error("failed to allocate Metal dehaze apply params");
        }

        {
            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            [enc setBuffer:input offset:0 atIndex:0];
            [enc setBuffer:depth offset:0 atIndex:1];
            [enc setBuffer:output offset:0 atIndex:2];
            [enc setBuffer:apply_params_buffer offset:0 atIndex:3];
            dispatch_1d(enc, pipelines.dehaze_apply, static_cast<NSUInteger>(prepared.count));
            [enc endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
        }
        std::memcpy(prepared.out.ptr, [output contents], prepared.image_bytes);
    }
    return prepared.result;
}

PYBIND11_MODULE(_dehaze_metal, m) {
    m.def("metal_available", &metal_available);
    m.def("dehaze_image", &dehaze_image);
}
