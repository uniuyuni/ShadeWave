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

struct CrossFilterMetalParams {
    int width;
    int height;
    int mini_width;
    int mini_height;
    int num_points;
    int length;
    float angle_deg;
    float threshold;
    float intensity;
    float spectral_strength;
    float line_thickness;
    int min_distance;
    float randomness;
    int speed_factor;
    int debug_mode;
};

constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct CrossFilterMetalParams {
    int width;
    int height;
    int mini_width;
    int mini_height;
    int num_points;
    int length;
    float angle_deg;
    float threshold;
    float intensity;
    float spectral_strength;
    float line_thickness;
    int min_distance;
    float randomness;
    int speed_factor;
    int debug_mode;
};

static inline float3 read_rgb(const device float* input, int width, int x, int y) {
    int base = (y * width + x) * 3;
    return float3(input[base + 0], input[base + 1], input[base + 2]);
}

static inline float luminance(float3 rgb) {
    return dot(rgb, float3(0.299f, 0.587f, 0.114f));
}

kernel void cross_filter_peak_impulse(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* impulse [[buffer(2)]],
    constant CrossFilterMetalParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }

    int base = (y * p.width + x) * 3;
    float3 rgb = float3(input[base + 0], input[base + 1], input[base + 2]);
    output[base + 0] = rgb.x;
    output[base + 1] = rgb.y;
    output[base + 2] = rgb.z;

    float lum = luminance(rgb);
    if (lum <= p.threshold) {
        return;
    }

    bool is_peak = true;
    int radius = max(p.min_distance, 0);
    for (int yy = max(0, y - radius); yy <= min(p.height - 1, y + radius) && is_peak; ++yy) {
        for (int xx = max(0, x - radius); xx <= min(p.width - 1, x + radius); ++xx) {
            if (luminance(read_rgb(input, p.width, xx, yy)) > lum) {
                is_peak = false;
                break;
            }
        }
    }
    if (!is_peak) {
        return;
    }

    if (p.debug_mode != 0) {
        output[base + 0] = 0.0f;
        output[base + 1] = 0.0f;
        output[base + 2] = 10.0f;
        return;
    }

    int sx = x / max(p.speed_factor, 1);
    int sy = y / max(p.speed_factor, 1);
    if (sx >= 0 && sx < p.mini_width && sy >= 0 && sy < p.mini_height) {
        float boost = float(max(p.speed_factor, 1)) * 1.5f;
        int ibase = (sy * p.mini_width + sx) * 3;
        impulse[ibase + 0] = rgb.x * boost;
        impulse[ibase + 1] = rgb.y * boost;
        impulse[ibase + 2] = rgb.z * boost;
    }
}

static inline float sample_channel_bilinear(const device float* img, int w, int h, float x, float y, int ch) {
    if (x < 0.0f || y < 0.0f || x > float(w - 1) || y > float(h - 1)) {
        return 0.0f;
    }
    int x0 = int(floor(x));
    int y0 = int(floor(y));
    int x1 = min(x0 + 1, w - 1);
    int y1 = min(y0 + 1, h - 1);
    float ax = x - float(x0);
    float ay = y - float(y0);
    float v00 = img[(y0 * w + x0) * 3 + ch];
    float v10 = img[(y0 * w + x1) * 3 + ch];
    float v01 = img[(y1 * w + x0) * 3 + ch];
    float v11 = img[(y1 * w + x1) * 3 + ch];
    float top = mix(v00, v10, ax);
    float bottom = mix(v01, v11, ax);
    return mix(top, bottom, ay);
}

static inline float sample_line_antialiased(
    const device float* img,
    int w,
    int h,
    float2 pos,
    float2 normal,
    float half_width,
    int ch
) {
    float center = sample_channel_bilinear(img, w, h, pos.x, pos.y, ch);
    float side_a = sample_channel_bilinear(img, w, h, pos.x + normal.x * half_width, pos.y + normal.y * half_width, ch);
    float side_b = sample_channel_bilinear(img, w, h, pos.x - normal.x * half_width, pos.y - normal.y * half_width, ch);
    return center * 0.5f + (side_a + side_b) * 0.25f;
}

kernel void cross_filter_streaks(
    const device float* impulse [[buffer(0)]],
    device float* streaks [[buffer(1)]],
    constant CrossFilterMetalParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.mini_width || y >= p.mini_height) {
        return;
    }

    int mini_length = max(p.length / max(p.speed_factor, 1), 1);
    int base_k_len = (mini_length % 2 == 0) ? mini_length + 1 : mini_length;
    bool symmetric = (p.num_points % 2) == 0;
    int num_passes = symmetric ? max(p.num_points / 2, 1) : max(p.num_points, 1);
    float rot_step = symmetric ? (180.0f / float(num_passes)) : (360.0f / float(num_passes));
    float spectral[3] = {1.0f + p.spectral_strength, 1.0f, 1.0f - p.spectral_strength};
    float pi = 3.14159265358979323846f;

    float3 accum = float3(0.0f);
    for (int pass = 0; pass < num_passes; ++pass) {
        float angle = (p.angle_deg + float(pass) * rot_step) * pi / 180.0f;
        float2 dir = float2(cos(angle), sin(angle));
        float2 normal = float2(-dir.y, dir.x);
        float aa_width = max(0.65f, (p.line_thickness - 1.0f) * 0.5f + 0.65f);
        for (int ch = 0; ch < 3; ++ch) {
            int ch_len = max(int(float(base_k_len) * spectral[ch]), 1);
            if ((ch_len % 2) == 0) {
                ch_len += 1;
            }
            int radius = max(ch_len / 2, 1);
            float value = sample_line_antialiased(
                impulse,
                p.mini_width,
                p.mini_height,
                float2(float(x), float(y)),
                normal,
                aa_width,
                ch
            );
            if (symmetric) {
                for (int d = 1; d < radius; ++d) {
                    float weight = exp(-8.0f * float(d) / float(radius));
                    float2 pos_a = float2(float(x), float(y)) + dir * float(d);
                    float2 pos_b = float2(float(x), float(y)) - dir * float(d);
                    value += weight * sample_line_antialiased(impulse, p.mini_width, p.mini_height, pos_a, normal, aa_width, ch);
                    value += weight * sample_line_antialiased(impulse, p.mini_width, p.mini_height, pos_b, normal, aa_width, ch);
                }
            } else {
                for (int d = 1; d < radius; ++d) {
                    float weight = exp(-8.0f * float(d) / float(radius));
                    float2 pos = float2(float(x), float(y)) + dir * float(d);
                    value += weight * sample_line_antialiased(impulse, p.mini_width, p.mini_height, pos, normal, aa_width, ch);
                }
            }
            accum[ch] += value;
        }
    }

    int base = (y * p.mini_width + x) * 3;
    streaks[base + 0] = accum.x;
    streaks[base + 1] = accum.y;
    streaks[base + 2] = accum.z;
}

static inline float sample_streak(const device float* img, int w, int h, float x, float y, int ch) {
    if (x < 0.0f || y < 0.0f || x > float(w - 1) || y > float(h - 1)) {
        return 0.0f;
    }
    int x0 = int(floor(x));
    int y0 = int(floor(y));
    int x1 = min(x0 + 1, w - 1);
    int y1 = min(y0 + 1, h - 1);
    float ax = x - float(x0);
    float ay = y - float(y0);
    float v00 = img[(y0 * w + x0) * 3 + ch];
    float v10 = img[(y0 * w + x1) * 3 + ch];
    float v01 = img[(y1 * w + x0) * 3 + ch];
    float v11 = img[(y1 * w + x1) * 3 + ch];
    float top = mix(v00, v10, ax);
    float bottom = mix(v01, v11, ax);
    return mix(top, bottom, ay);
}

kernel void cross_filter_composite(
    const device float* input [[buffer(0)]],
    const device float* streaks [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant CrossFilterMetalParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    float sx = (float(x) + 0.5f) * float(p.mini_width) / float(p.width) - 0.5f;
    float sy = (float(y) + 0.5f) * float(p.mini_height) / float(p.height) - 0.5f;
    int base = (y * p.width + x) * 3;
    output[base + 0] = input[base + 0] + sample_streak(streaks, p.mini_width, p.mini_height, sx, sy, 0) * p.intensity;
    output[base + 1] = input[base + 1] + sample_streak(streaks, p.mini_width, p.mini_height, sx, sy, 1) * p.intensity;
    output[base + 2] = input[base + 2] + sample_streak(streaks, p.mini_width, p.mini_height, sx, sy, 2) * p.intensity;
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
    id<MTLComputePipelineState> peak;
    id<MTLComputePipelineState> streak;
    id<MTLComputePipelineState> composite;
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
                state.peak = make_pipeline(state.device, state.library, @"cross_filter_peak_impulse");
                state.streak = make_pipeline(state.device, state.library, @"cross_filter_streaks");
                state.composite = make_pipeline(state.device, state.library, @"cross_filter_composite");
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

void encode_dispatch(
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

}  // namespace

py::array_t<float> apply_cross_filter(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    int num_points,
    int length,
    float angle_deg,
    float threshold,
    float intensity,
    float spectral_strength,
    float line_thickness,
    int min_distance,
    float randomness,
    int speed_factor,
    bool debug_mode
) {
    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must be a 3D RGB float32 array");
    }

    const int width = static_cast<int>(in.shape[1]);
    const int height = static_cast<int>(in.shape[0]);
    speed_factor = std::max(speed_factor, 1);
    int mini_width = width / speed_factor;
    int mini_height = height / speed_factor;
    if (mini_width < 1 || mini_height < 1) {
        mini_width = width;
        mini_height = height;
        speed_factor = 1;
    }

    std::vector<py::ssize_t> shape(in.shape.begin(), in.shape.end());
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    @autoreleasepool {
        MetalPipelines& pipelines = metal_pipelines();

        const size_t input_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3 * sizeof(float);
        const size_t mini_bytes = static_cast<size_t>(mini_width) * static_cast<size_t>(mini_height) * 3 * sizeof(float);

        BufferBinding input_binding = make_buffer_for_input(pipelines.device, in.ptr, input_bytes);
        BufferBinding output_binding = make_buffer_for_output(pipelines.device, out.ptr, input_bytes);
        id<MTLBuffer> input_buffer = input_binding.buffer;
        id<MTLBuffer> output_buffer = output_binding.buffer;
        id<MTLBuffer> impulse_buffer = [pipelines.device newBufferWithLength:mini_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> streak_buffer = [pipelines.device newBufferWithLength:mini_bytes options:MTLResourceStorageModeShared];
        memset([impulse_buffer contents], 0, mini_bytes);
        memset([streak_buffer contents], 0, mini_bytes);

        CrossFilterMetalParams params{
            width,
            height,
            mini_width,
            mini_height,
            num_points,
            length,
            angle_deg,
            threshold,
            intensity,
            spectral_strength,
            line_thickness,
            min_distance,
            randomness,
            speed_factor,
            debug_mode ? 1 : 0,
        };
        id<MTLBuffer> params_buffer = [pipelines.device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];

        id<MTLComputeCommandEncoder> peak_encoder = [command_buffer computeCommandEncoder];
        [peak_encoder setBuffer:input_buffer offset:input_binding.offset atIndex:0];
        [peak_encoder setBuffer:output_buffer offset:output_binding.offset atIndex:1];
        [peak_encoder setBuffer:impulse_buffer offset:0 atIndex:2];
        [peak_encoder setBuffer:params_buffer offset:0 atIndex:3];
        encode_dispatch(peak_encoder, pipelines.peak, width, height);
        [peak_encoder endEncoding];

        if (!debug_mode) {
            id<MTLComputeCommandEncoder> streak_encoder = [command_buffer computeCommandEncoder];
            [streak_encoder setBuffer:impulse_buffer offset:0 atIndex:0];
            [streak_encoder setBuffer:streak_buffer offset:0 atIndex:1];
            [streak_encoder setBuffer:params_buffer offset:0 atIndex:2];
            encode_dispatch(streak_encoder, pipelines.streak, mini_width, mini_height);
            [streak_encoder endEncoding];

            id<MTLComputeCommandEncoder> composite_encoder = [command_buffer computeCommandEncoder];
            [composite_encoder setBuffer:input_buffer offset:input_binding.offset atIndex:0];
            [composite_encoder setBuffer:streak_buffer offset:0 atIndex:1];
            [composite_encoder setBuffer:output_buffer offset:output_binding.offset atIndex:2];
            [composite_encoder setBuffer:params_buffer offset:0 atIndex:3];
            encode_dispatch(composite_encoder, pipelines.composite, width, height);
            [composite_encoder endEncoding];
        }

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer error]) {
            throw std::runtime_error([[[command_buffer error] localizedDescription] UTF8String]);
        }

        finish_output_binding(output_binding, out.ptr, input_bytes);
    }

    return result;
}

PYBIND11_MODULE(_cross_filter_metal, m) {
    m.doc() = "Metal CrossFilter backend for Platypus";
    m.def("metal_available", []() {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            return device != nil;
        }
    });
    m.def(
        "apply_cross_filter",
        &apply_cross_filter,
        py::arg("image"),
        py::arg("num_points") = 6,
        py::arg("length") = 100,
        py::arg("angle_deg") = 0.0f,
        py::arg("threshold") = 1.0f,
        py::arg("intensity") = 1.0f,
        py::arg("spectral_strength") = 0.2f,
        py::arg("line_thickness") = 1.0f,
        py::arg("min_distance") = 10,
        py::arg("randomness") = 0.0f,
        py::arg("speed_factor") = 4,
        py::arg("debug_mode") = false
    );
}
