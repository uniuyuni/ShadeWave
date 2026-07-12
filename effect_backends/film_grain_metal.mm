// Metal film grain backend.
// film_grain_cpu.c と同じステートレス整数ハッシュ乱数(mix_u32 / hash_noise /
// layer_noise)を GPU で再現する。mono ノイズの平均・分散はホスト側で double
// 累積して求め(CPU 版の OpenMP reduction と float 誤差内で一致)、適用パスを
// 2 本目のカーネルで行う。
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_buffer_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct FilmGrainMetalParams {
    int width;
    int height;
    int channels;
    unsigned int seed;
    float base_size;
    float fine_w;
    float mid_w;
    float coarse_w;
    float amount_scale;
    float shadow_gain;
    float highlight_gain;
    float color_gain;
    float mean;
    float inv_std;
};

constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float FG_KR = 0.2126f;
constant float FG_KG = 0.7152f;
constant float FG_KB = 0.0722f;

struct FilmGrainMetalParams {
    int width;
    int height;
    int channels;
    uint seed;
    float base_size;
    float fine_w;
    float mid_w;
    float coarse_w;
    float amount_scale;
    float shadow_gain;
    float highlight_gain;
    float color_gain;
    float mean;
    float inv_std;
};

static inline float fg_clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float smoothstep01(float x) {
    float t = fg_clamp(x, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static inline uint mix_u32(uint v) {
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

static inline float hash_noise(int x, int y, uint seed, uint salt) {
    uint h = seed ^ salt;
    h ^= uint(x) * 0x8da6b343u;
    h ^= uint(y) * 0xd8163841u;
    h = mix_u32(h);
    float u = float(h & 0x00FFFFFFu) * (1.0f / 16777215.0f);
    return (u * 2.0f - 1.0f) * 1.7320508075688772f;
}

static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

static inline float layer_noise(int x, int y, float grain_size, uint seed, uint salt) {
    grain_size = grain_size < 0.35f ? 0.35f : grain_size;
    if (grain_size <= 0.75f) {
        return hash_noise(x, y, seed, salt);
    }
    float sx = float(x) / grain_size;
    float sy = float(y) / grain_size;
    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    float fx = sx - float(x0);
    float fy = sy - float(y0);
    float wx = smoothstep01(fx);
    float wy = smoothstep01(fy);
    float n00 = hash_noise(x0, y0, seed, salt);
    float n10 = hash_noise(x0 + 1, y0, seed, salt);
    float n01 = hash_noise(x0, y0 + 1, seed, salt);
    float n11 = hash_noise(x0 + 1, y0 + 1, seed, salt);
    float nx0 = lerpf(n00, n10, wx);
    float nx1 = lerpf(n01, n11, wx);
    return lerpf(nx0, nx1, wy);
}

static inline float safe_luma(float r, float g, float b) {
    r = isfinite(r) ? r : (r > 0.0f ? 1.0f : 0.0f);
    g = isfinite(g) ? g : (g > 0.0f ? 1.0f : 0.0f);
    b = isfinite(b) ? b : (b > 0.0f ? 1.0f : 0.0f);
    return fg_clamp(FG_KR * r + FG_KG * g + FG_KB * b, 0.0f, 1.0f);
}

kernel void fg_mono(
    device float* mono [[buffer(0)]],
    constant FilmGrainMetalParams& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    int x = int(gid) % p.width;
    int y = int(gid) / p.width;
    float fine = layer_noise(x, y, p.base_size * 0.55f, p.seed, 0xA53A9D1Bu);
    float mid = layer_noise(x, y, p.base_size, p.seed, 0xC2B2AE35u);
    float coarse = layer_noise(x, y, p.base_size * 2.35f, p.seed, 0x9E3779B9u);
    mono[gid] = p.fine_w * fine + p.mid_w * mid + p.coarse_w * coarse;
}

kernel void fg_apply(
    const device float* input [[buffer(0)]],
    const device float* mono [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant FilmGrainMetalParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    int x = int(gid) % p.width;
    int y = int(gid) / p.width;
    int base = int(gid) * p.channels;
    float r0 = input[base];
    float g0 = input[base + 1];
    float b0 = input[base + 2];

    float luma = safe_luma(r0, g0, b0);
    float shadow_w = pow(1.0f - luma, 1.55f);
    float highlight_w = pow(luma, 1.75f);
    float midtone_w = 1.0f - pow(fabs(luma * 2.0f - 1.0f), 1.65f);
    float response = 0.50f * midtone_w
        + 0.42f * p.shadow_gain * shadow_w
        + 0.32f * p.highlight_gain * highlight_w;
    float headroom = fmin(luma, 1.0f - luma);
    float protect = 0.45f + 0.55f * fg_clamp(headroom * 5.0f, 0.0f, 1.0f);
    float amplitude = p.amount_scale * response * protect;
    float m = (mono[gid] - p.mean) * p.inv_std;

    float r = r0 + m * amplitude;
    float g = g0 + m * amplitude;
    float b = b0 + m * amplitude;

    if (p.color_gain > 0.0f) {
        float u = layer_noise(x, y, p.base_size * 1.35f, p.seed, 0x85EBCA6Bu);
        float v = layer_noise(x, y, p.base_size * 1.75f, p.seed, 0x27D4EB2Fu);
        float c_amp = amplitude * p.color_gain * 0.42f;
        r += (u * 0.82f + v * 0.28f) * c_amp;
        g += (u * -0.45f + v * 0.42f) * c_amp;
        b += (u * -0.37f + v * -0.70f) * c_amp;
    }

    output[base] = r;
    output[base + 1] = g;
    output[base + 2] = b;
    for (int c = 3; c < p.channels; ++c) {
        output[base + c] = input[base + c];
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
    id<MTLComputePipelineState> mono;
    id<MTLComputePipelineState> apply;
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
                state.mono = make_pipeline(state.device, state.library, @"fg_mono");
                state.apply = make_pipeline(state.device, state.library, @"fg_apply");
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
    NSUInteger tpg = std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace

bool metal_available() {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

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
        throw std::invalid_argument("image must be a 3D array with at least 3 channels");
    }
    const int height = static_cast<int>(in.shape[0]);
    const int width = static_cast<int>(in.shape[1]);
    const int channels = static_cast<int>(in.shape[2]);
    const int count = width * height;
    const size_t plane_bytes = static_cast<size_t>(count) * sizeof(float);
    const size_t image_bytes = static_cast<size_t>(count) * static_cast<size_t>(channels) * sizeof(float);

    std::vector<py::ssize_t> shape(in.shape.begin(), in.shape.end());
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    const float amount_clamped = clampf(amount, 0.0f, 100.0f);
    if (amount_clamped <= 0.0f) {
        std::memcpy(out.ptr, in.ptr, image_bytes);
        return result;
    }

    const float rough = clampf(roughness, 0.0f, 100.0f) / 100.0f;
    unsigned int seed_value = static_cast<unsigned int>(seed);
    if (seed_value == 0u) {
        seed_value = 0x6D2B79F5u;
    }
    seed_value ^= static_cast<unsigned int>(height) * 73856093u;
    seed_value ^= static_cast<unsigned int>(width) * 19349663u;

    FilmGrainMetalParams params{};
    params.width = width;
    params.height = height;
    params.channels = channels;
    params.seed = seed_value;
    params.base_size = grain_size > 0.35f ? grain_size : 0.35f;
    params.fine_w = 0.25f + 0.55f * rough;
    params.mid_w = 0.70f;
    params.coarse_w = 0.55f * (1.0f - rough);
    params.amount_scale = (amount_clamped / 100.0f) * 0.045f;
    params.shadow_gain = 0.35f + clampf(shadow, 0.0f, 100.0f) / 100.0f * 1.35f;
    params.highlight_gain = 0.15f + clampf(highlight, 0.0f, 100.0f) / 100.0f * 1.10f;
    params.color_gain = clampf(color, 0.0f, 100.0f) / 100.0f;

    {
        py::gil_scoped_release release;
        @autoreleasepool {
            MetalPipelines& pipelines = metal_pipelines();
            BufferBinding input_binding = make_buffer_for_input(pipelines.device, in.ptr, image_bytes);
            BufferBinding output_binding = make_buffer_for_output(pipelines.device, out.ptr, image_bytes);
            id<MTLBuffer> mono = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            if (!input_binding.buffer || !output_binding.buffer || !mono) {
                throw std::runtime_error("failed to allocate Metal film grain buffers");
            }

            {
                id<MTLBuffer> params_buffer = [pipelines.device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setBuffer:mono offset:0 atIndex:0];
                [enc setBuffer:params_buffer offset:0 atIndex:1];
                dispatch_1d(enc, pipelines.mono, static_cast<NSUInteger>(count));
                [enc endEncoding];
                [command_buffer commit];
                [command_buffer waitUntilCompleted];
                if (command_buffer.error) {
                    throw std::runtime_error([[command_buffer.error localizedDescription] UTF8String]);
                }
            }

            // 平均・分散は CPU の double 累積(OpenMP reduction と float 誤差内で一致)。
            const float* mono_ptr = static_cast<const float*>([mono contents]);
            double sum = 0.0;
            double sumsq = 0.0;
            for (int i = 0; i < count; ++i) {
                const double v = static_cast<double>(mono_ptr[i]);
                sum += v;
                sumsq += v * v;
            }
            const double mean = sum / static_cast<double>(count);
            double variance = sumsq / static_cast<double>(count) - mean * mean;
            if (variance < 1.0e-12) {
                variance = 1.0;
            }
            params.mean = static_cast<float>(mean);
            params.inv_std = static_cast<float>(1.0 / std::sqrt(variance));

            {
                id<MTLBuffer> params_buffer = [pipelines.device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setBuffer:input_binding.buffer offset:input_binding.offset atIndex:0];
                [enc setBuffer:mono offset:0 atIndex:1];
                [enc setBuffer:output_binding.buffer offset:output_binding.offset atIndex:2];
                [enc setBuffer:params_buffer offset:0 atIndex:3];
                dispatch_1d(enc, pipelines.apply, static_cast<NSUInteger>(count));
                [enc endEncoding];
                [command_buffer commit];
                [command_buffer waitUntilCompleted];
                if (command_buffer.error) {
                    throw std::runtime_error([[command_buffer.error localizedDescription] UTF8String]);
                }
            }
            finish_output_binding(output_binding, out.ptr, image_bytes);
        }
    }

    return result;
}

PYBIND11_MODULE(_film_grain_metal, m) {
    m.doc() = "Metal film grain backend for Platypus";
    m.def("metal_available", &metal_available);
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
