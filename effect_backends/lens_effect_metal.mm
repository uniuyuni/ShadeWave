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

struct BokehFringeParams {
    int width;
    int height;
    int radius;
    float focus_depth;
    float strength;
};

struct ShapedBokehParams {
    int width;
    int height;
    int kernel_width;
    int kernel_height;
    float gain;
    float focus_depth;
    float strength;
};

struct SunstarParams {
    int width;
    int height;
    int source_count;
    int spike_count;
    float base_rot;
    float spacing;
};

struct SwirlParams {
    int width;
    int height;
    int use_depth;
    float center_x;
    float center_y;
    float focus_depth;
    float strength;
};

constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct BokehFringeParams {
    int width;
    int height;
    int radius;
    float focus_depth;
    float strength;
};

struct ShapedBokehParams {
    int width;
    int height;
    int kernel_width;
    int kernel_height;
    float gain;
    float focus_depth;
    float strength;
};

struct SunstarParams {
    int width;
    int height;
    int source_count;
    int spike_count;
    float base_rot;
    float spacing;
};

struct SwirlParams {
    int width;
    int height;
    int use_depth;
    float center_x;
    float center_y;
    float focus_depth;
    float strength;
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

static inline int reflect_edge(int p, int len) {
    if (len <= 1) {
        return 0;
    }
    while (p < 0 || p >= len) {
        if (p < 0) {
            p = -p - 1;
        } else {
            p = 2 * len - p - 1;
        }
    }
    return p;
}

static inline float3 sample_rgb_clamp(const device float* input, int width, int height, float x, float y) {
    x = clamp(x, 0.0f, float(width - 1));
    y = clamp(y, 0.0f, float(height - 1));
    int x0 = int(floor(x));
    int y0 = int(floor(y));
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    float tx = x - float(x0);
    float ty = y - float(y0);
    int b00 = (y0 * width + x0) * 3;
    int b10 = (y0 * width + x1) * 3;
    int b01 = (y1 * width + x0) * 3;
    int b11 = (y1 * width + x1) * 3;
    float3 c00 = float3(input[b00], input[b00 + 1], input[b00 + 2]);
    float3 c10 = float3(input[b10], input[b10 + 1], input[b10 + 2]);
    float3 c01 = float3(input[b01], input[b01 + 1], input[b01 + 2]);
    float3 c11 = float3(input[b11], input[b11 + 1], input[b11 + 2]);
    return mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
}

kernel void signed_abs_depth(
    const device float* depth [[buffer(0)]],
    device float* signed_depth [[buffer(1)]],
    device float* abs_depth [[buffer(2)]],
    constant BokehFringeParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    float s = depth[gid] - p.focus_depth;
    signed_depth[gid] = s;
    abs_depth[gid] = fabs(s);
}

kernel void gaussian_plane_horizontal(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant BokehFringeParams& p [[buffer(3)]],
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
    constant BokehFringeParams& p [[buffer(3)]],
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

kernel void gaussian_channel_horizontal(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant BokehFringeParams& p [[buffer(3)]],
    constant int& channel [[buffer(4)]],
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
        sum += input[(y * p.width + sx) * 3 + channel] * weights[k + p.radius];
    }
    output[y * p.width + x] = sum;
}

kernel void compose_bokeh_fringe(
    const device float* input [[buffer(0)]],
    const device float* signed_depth [[buffer(1)]],
    const device float* defocus [[buffer(2)]],
    const device float* blur_r [[buffer(3)]],
    const device float* blur_g [[buffer(4)]],
    const device float* blur_b [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant BokehFringeParams& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    int base = int(gid) * 3;
    float amount = clamp(defocus[gid] * (0.6f + 1.4f * p.strength), 0.0f, 1.0f);
    float front_w = signed_depth[gid] < 0.0f ? amount : 0.0f;
    float back_w = signed_depth[gid] > 0.0f ? amount : 0.0f;
    float r = input[base];
    float g = input[base + 1];
    float b = input[base + 2];
    output[base] = r * (1.0f - front_w) + blur_r[gid] * front_w;
    output[base + 1] = g * (1.0f - back_w) + blur_g[gid] * back_w;
    output[base + 2] = b * (1.0f - front_w) + blur_b[gid] * front_w;
}

kernel void shaped_no_depth_mono(
    const device float* input [[buffer(0)]],
    const device float* source [[buffer(1)]],
    const device float* kernel_data [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant ShapedBokehParams& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int krx = p.kernel_width / 2;
    int kry = p.kernel_height / 2;
    float3 acc = float3(0.0f);
    for (int ky = 0; ky < p.kernel_height; ++ky) {
        int sy = reflect_edge(y + ky - kry, p.height);
        for (int kx = 0; kx < p.kernel_width; ++kx) {
            int sx = reflect_edge(x + kx - krx, p.width);
            float kval = kernel_data[ky * p.kernel_width + kx];
            int sbase = (sy * p.width + sx) * 3;
            acc += float3(source[sbase], source[sbase + 1], source[sbase + 2]) * kval;
        }
    }
    int base = (y * p.width + x) * 3;
    output[base] = input[base] + acc.x * p.gain;
    output[base + 1] = input[base + 1] + acc.y * p.gain;
    output[base + 2] = input[base + 2] + acc.z * p.gain;
}

kernel void shaped_no_depth_color(
    const device float* input [[buffer(0)]],
    const device float* source [[buffer(1)]],
    const device float* kernel_data [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant ShapedBokehParams& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int krx = p.kernel_width / 2;
    int kry = p.kernel_height / 2;
    float3 acc = float3(0.0f);
    for (int ky = 0; ky < p.kernel_height; ++ky) {
        int sy = reflect_edge(y + ky - kry, p.height);
        for (int kx = 0; kx < p.kernel_width; ++kx) {
            int sx = reflect_edge(x + kx - krx, p.width);
            int kbase = (ky * p.kernel_width + kx) * 3;
            int sbase = (sy * p.width + sx) * 3;
            acc.x += source[sbase] * kernel_data[kbase];
            acc.y += source[sbase + 1] * kernel_data[kbase + 1];
            acc.z += source[sbase + 2] * kernel_data[kbase + 2];
        }
    }
    int base = (y * p.width + x) * 3;
    output[base] = input[base] + acc.x * p.gain;
    output[base + 1] = input[base + 1] + acc.y * p.gain;
    output[base + 2] = input[base + 2] + acc.z * p.gain;
}

kernel void shaped_depth_mono(
    const device float* input [[buffer(0)]],
    const device float* source [[buffer(1)]],
    const device float* depth [[buffer(2)]],
    const device float* kernel_data [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant ShapedBokehParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int krx = p.kernel_width / 2;
    int kry = p.kernel_height / 2;
    float3 acc = float3(0.0f);
    for (int ky = 0; ky < p.kernel_height; ++ky) {
        int sy = reflect_edge(y + ky - kry, p.height);
        for (int kx = 0; kx < p.kernel_width; ++kx) {
            int sx = reflect_edge(x + kx - krx, p.width);
            float kval = kernel_data[ky * p.kernel_width + kx];
            int sbase = (sy * p.width + sx) * 3;
            acc += float3(source[sbase], source[sbase + 1], source[sbase + 2]) * kval;
        }
    }
    int pix = y * p.width + x;
    int base = pix * 3;
    float w = clamp(fabs(depth[pix] - p.focus_depth) * 2.5f, 0.0f, 1.0f);
    w *= clamp(0.4f + 0.6f * p.strength, 0.0f, 1.0f);
    output[base] = input[base] * (1.0f - w) + acc.x * w;
    output[base + 1] = input[base + 1] * (1.0f - w) + acc.y * w;
    output[base + 2] = input[base + 2] * (1.0f - w) + acc.z * w;
}

kernel void shaped_depth_color(
    const device float* input [[buffer(0)]],
    const device float* source [[buffer(1)]],
    const device float* depth [[buffer(2)]],
    const device float* kernel_data [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant ShapedBokehParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int krx = p.kernel_width / 2;
    int kry = p.kernel_height / 2;
    float3 acc = float3(0.0f);
    for (int ky = 0; ky < p.kernel_height; ++ky) {
        int sy = reflect_edge(y + ky - kry, p.height);
        for (int kx = 0; kx < p.kernel_width; ++kx) {
            int sx = reflect_edge(x + kx - krx, p.width);
            int kbase = (ky * p.kernel_width + kx) * 3;
            int sbase = (sy * p.width + sx) * 3;
            acc.x += source[sbase] * kernel_data[kbase];
            acc.y += source[sbase + 1] * kernel_data[kbase + 1];
            acc.z += source[sbase + 2] * kernel_data[kbase + 2];
        }
    }
    int pix = y * p.width + x;
    int base = pix * 3;
    float w = clamp(fabs(depth[pix] - p.focus_depth) * 2.5f, 0.0f, 1.0f);
    w *= clamp(0.4f + 0.6f * p.strength, 0.0f, 1.0f);
    output[base] = input[base] * (1.0f - w) + acc.x * w;
    output[base + 1] = input[base + 1] * (1.0f - w) + acc.y * w;
    output[base + 2] = input[base + 2] * (1.0f - w) + acc.z * w;
}

kernel void sunstar_overlay(
    const device float* sources [[buffer(0)]],
    const device float* jitter [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant SunstarParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }

    float3 out = float3(0.0f);
    float2 pos = float2(float(x), float(y));
    float3 cool_bias = float3(0.92f, 0.97f, 1.10f);

    for (int si = 0; si < p.source_count; ++si) {
        int sbase = si * 9;
        float inten = sources[sbase];
        float cxs = sources[sbase + 1];
        float cys = sources[sbase + 2];
        float L = sources[sbase + 3];
        float spike_w0 = sources[sbase + 4];
        float core_sigma = sources[sbase + 5];
        float3 src_tint = float3(sources[sbase + 6], sources[sbase + 7], sources[sbase + 8]);

        float dx = pos.x - cxs;
        float dy = pos.y - cys;
        float r = sqrt(dx * dx + dy * dy) + 1.0e-3f;
        float theta = atan2(dy, dx);

        int k = int(round((theta - p.base_rot) / p.spacing));
        k = k % p.spike_count;
        if (k < 0) {
            k += p.spike_count;
        }
        int jbase = (si * p.spike_count + k) * 4;
        float ang_jit = jitter[jbase];
        float len_jit = jitter[jbase + 1];
        float wid_jit = jitter[jbase + 2];
        float amp_jit = jitter[jbase + 3];

        float a_k = p.base_rot + float(k) * p.spacing + ang_jit;
        float d_ang = theta - a_k;
        d_ang = atan2(sin(d_ang), cos(d_ang));
        float perp = r * sin(d_ang);
        float along = r * cos(d_ang);
        float Lm = max(1.5f, L * len_jit);
        float wm = max(0.5f, spike_w0 * wid_jit);
        float cross = exp(-(perp * perp) / (2.0f * wm * wm));
        float t = clamp(along / Lm, 0.0f, 1.0f);
        float prof = (1.0f - t) * (0.30f + 0.70f * exp(-along / (0.45f * Lm)));
        float ray = ((along > 0.0f) && (along < Lm)) ? (cross * prof * amp_jit) : 0.0f;
        float core = exp(-(r * r) / (2.0f * core_sigma * core_sigma));
        float scalar = (ray + 0.9f * core) * inten;

        float3 tip_tint = clamp(src_tint * cool_bias, float3(0.0f), float3(1.2f));
        float rn = clamp(r / max(L, 1.0f), 0.0f, 1.0f);
        float3 tint = src_tint * (1.0f - rn) + tip_tint * rn;
        out += scalar * tint;
    }

    int base = (y * p.width + x) * 3;
    output[base] = out.x;
    output[base + 1] = out.y;
    output[base + 2] = out.z;
}

kernel void swirl_bokeh_direct(
    const device float* input [[buffer(0)]],
    const device float* depth [[buffer(1)]],
    const device float* radial_norm [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant SwirlParams& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int pix = y * p.width + x;
    int base = pix * 3;
    float3 original = float3(input[base], input[base + 1], input[base + 2]);

    float max_angle = (12.0f * p.strength / 100.0f) * 0.017453292519943295f;
    if (max_angle < 1.0e-5f) {
        output[base] = original.x;
        output[base + 1] = original.y;
        output[base + 2] = original.z;
        return;
    }

    float dx = float(x) - p.center_x;
    float dy = float(y) - p.center_y;
    float r = sqrt(dx * dx + dy * dy);
    float theta = atan2(dy, dx);

    constexpr int sample_count = 17;
    float3 acc = float3(0.0f);
    float wsum = 0.0f;
    for (int i = 0; i < sample_count; ++i) {
        float u = (float(i) - 8.0f) / 8.0f;
        float a = theta + u * max_angle;
        float weight = exp(-2.0f * u * u);
        float sx = p.center_x + r * cos(a);
        float sy = p.center_y + r * sin(a);
        acc += sample_rgb_clamp(input, p.width, p.height, sx, sy) * weight;
        wsum += weight;
    }
    float3 swirl = acc / max(wsum, 1.0e-6f);

    float radial = 0.35f + 0.65f * clamp(radial_norm[pix], 0.0f, 1.0f);
    float defocus = 1.0f;
    if (p.use_depth != 0) {
        defocus = clamp(fabs(depth[pix] - p.focus_depth) * 2.5f, 0.0f, 1.0f);
    }
    float wm = clamp(radial * defocus, 0.0f, 1.0f);
    float3 out = original * (1.0f - wm) + swirl * wm;
    output[base] = out.x;
    output[base + 1] = out.y;
    output[base + 2] = out.z;
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
    id<MTLComputePipelineState> signed_abs;
    id<MTLComputePipelineState> gaussian_h;
    id<MTLComputePipelineState> gaussian_v;
    id<MTLComputePipelineState> channel_h;
    id<MTLComputePipelineState> compose;
    id<MTLComputePipelineState> shaped_no_depth_mono;
    id<MTLComputePipelineState> shaped_no_depth_color;
    id<MTLComputePipelineState> shaped_depth_mono;
    id<MTLComputePipelineState> shaped_depth_color;
    id<MTLComputePipelineState> sunstar_overlay;
    id<MTLComputePipelineState> swirl_bokeh;
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
                state.signed_abs = make_pipeline(state.device, state.library, @"signed_abs_depth");
                state.gaussian_h = make_pipeline(state.device, state.library, @"gaussian_plane_horizontal");
                state.gaussian_v = make_pipeline(state.device, state.library, @"gaussian_plane_vertical");
                state.channel_h = make_pipeline(state.device, state.library, @"gaussian_channel_horizontal");
                state.compose = make_pipeline(state.device, state.library, @"compose_bokeh_fringe");
                state.shaped_no_depth_mono = make_pipeline(state.device, state.library, @"shaped_no_depth_mono");
                state.shaped_no_depth_color = make_pipeline(state.device, state.library, @"shaped_no_depth_color");
                state.shaped_depth_mono = make_pipeline(state.device, state.library, @"shaped_depth_mono");
                state.shaped_depth_color = make_pipeline(state.device, state.library, @"shaped_depth_color");
                state.sunstar_overlay = make_pipeline(state.device, state.library, @"sunstar_overlay");
                state.swirl_bokeh = make_pipeline(state.device, state.library, @"swirl_bokeh_direct");
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

void encode_plane_blur(
    id<MTLCommandBuffer> command_buffer,
    MetalPipelines& pipelines,
    id<MTLBuffer> input,
    id<MTLBuffer> tmp,
    id<MTLBuffer> output,
    id<MTLBuffer> weights,
    id<MTLBuffer> params,
    int width,
    int height
) {
    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
    [enc setComputePipelineState:pipelines.gaussian_h];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:tmp offset:0 atIndex:1];
    [enc setBuffer:weights offset:0 atIndex:2];
    [enc setBuffer:params offset:0 atIndex:3];
    dispatch_2d(enc, pipelines.gaussian_h, width, height);
    [enc endEncoding];

    enc = [command_buffer computeCommandEncoder];
    [enc setComputePipelineState:pipelines.gaussian_v];
    [enc setBuffer:tmp offset:0 atIndex:0];
    [enc setBuffer:output offset:0 atIndex:1];
    [enc setBuffer:weights offset:0 atIndex:2];
    [enc setBuffer:params offset:0 atIndex:3];
    dispatch_2d(enc, pipelines.gaussian_v, width, height);
    [enc endEncoding];
}

}  // namespace

py::array_t<float> apply_shaped_bokeh_no_depth(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    py::array_t<float, py::array::c_style | py::array::forcecast> source,
    py::array_t<float, py::array::c_style | py::array::forcecast> kernel,
    bool colored_kernel,
    float gain
) {
    py::buffer_info in = image.request();
    py::buffer_info src = source.request();
    py::buffer_info ker = kernel.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must have shape (height, width, 3)");
    }
    if (src.ndim != 3 || src.shape[0] != in.shape[0] || src.shape[1] != in.shape[1] || src.shape[2] != 3) {
        throw std::invalid_argument("source must have shape (height, width, 3)");
    }
    if ((!colored_kernel && ker.ndim != 2) || (colored_kernel && (ker.ndim != 3 || ker.shape[2] != 3))) {
        throw std::invalid_argument("kernel must have shape (kh, kw) or (kh, kw, 3)");
    }
    const int height = static_cast<int>(in.shape[0]);
    const int width = static_cast<int>(in.shape[1]);
    const int kh = static_cast<int>(ker.shape[0]);
    const int kw = static_cast<int>(ker.shape[1]);
    const int count = width * height;
    const size_t image_bytes = static_cast<size_t>(count) * 3 * sizeof(float);
    const size_t kernel_bytes = static_cast<size_t>(ker.size) * sizeof(float);

    py::array_t<float> result({height, width, 3});
    py::buffer_info out = result.request();

    {
        py::gil_scoped_release release;
        @autoreleasepool {
            MetalPipelines& pipelines = metal_pipelines();
            id<MTLBuffer> input_buffer = [pipelines.device newBufferWithBytes:in.ptr length:image_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> source_buffer = [pipelines.device newBufferWithBytes:src.ptr length:image_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> kernel_buffer = [pipelines.device newBufferWithBytes:ker.ptr length:kernel_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buffer = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
            ShapedBokehParams params{width, height, kw, kh, gain, 0.0f, 0.0f};
            id<MTLBuffer> params_buffer = [pipelines.device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
            if (!input_buffer || !source_buffer || !kernel_buffer || !output_buffer || !params_buffer) {
                throw std::runtime_error("failed to allocate Metal shaped bokeh buffers");
            }
            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            id<MTLComputePipelineState> pipeline = colored_kernel ? pipelines.shaped_no_depth_color : pipelines.shaped_no_depth_mono;
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:input_buffer offset:0 atIndex:0];
            [enc setBuffer:source_buffer offset:0 atIndex:1];
            [enc setBuffer:kernel_buffer offset:0 atIndex:2];
            [enc setBuffer:output_buffer offset:0 atIndex:3];
            [enc setBuffer:params_buffer offset:0 atIndex:4];
            dispatch_2d(enc, pipeline, width, height);
            [enc endEncoding];
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

py::array_t<float> apply_shaped_bokeh_depth(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    py::array_t<float, py::array::c_style | py::array::forcecast> source,
    py::array_t<float, py::array::c_style | py::array::forcecast> depth,
    py::array_t<float, py::array::c_style | py::array::forcecast> kernel,
    bool colored_kernel,
    float focus_depth,
    float strength
) {
    py::buffer_info in = image.request();
    py::buffer_info src = source.request();
    py::buffer_info dep = depth.request();
    py::buffer_info ker = kernel.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must have shape (height, width, 3)");
    }
    if (src.ndim != 3 || src.shape[0] != in.shape[0] || src.shape[1] != in.shape[1] || src.shape[2] != 3) {
        throw std::invalid_argument("source must have shape (height, width, 3)");
    }
    if (dep.ndim != 2 || dep.shape[0] != in.shape[0] || dep.shape[1] != in.shape[1]) {
        throw std::invalid_argument("depth must have shape (height, width)");
    }
    if ((!colored_kernel && ker.ndim != 2) || (colored_kernel && (ker.ndim != 3 || ker.shape[2] != 3))) {
        throw std::invalid_argument("kernel must have shape (kh, kw) or (kh, kw, 3)");
    }
    const int height = static_cast<int>(in.shape[0]);
    const int width = static_cast<int>(in.shape[1]);
    const int kh = static_cast<int>(ker.shape[0]);
    const int kw = static_cast<int>(ker.shape[1]);
    const int count = width * height;
    const size_t plane_bytes = static_cast<size_t>(count) * sizeof(float);
    const size_t image_bytes = plane_bytes * 3;
    const size_t kernel_bytes = static_cast<size_t>(ker.size) * sizeof(float);

    py::array_t<float> result({height, width, 3});
    py::buffer_info out = result.request();

    {
        py::gil_scoped_release release;
        @autoreleasepool {
            MetalPipelines& pipelines = metal_pipelines();
            id<MTLBuffer> input_buffer = [pipelines.device newBufferWithBytes:in.ptr length:image_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> source_buffer = [pipelines.device newBufferWithBytes:src.ptr length:image_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> depth_buffer = [pipelines.device newBufferWithBytes:dep.ptr length:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> kernel_buffer = [pipelines.device newBufferWithBytes:ker.ptr length:kernel_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buffer = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
            ShapedBokehParams params{width, height, kw, kh, 0.0f, focus_depth, strength / 100.0f};
            id<MTLBuffer> params_buffer = [pipelines.device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
            if (!input_buffer || !source_buffer || !depth_buffer || !kernel_buffer || !output_buffer || !params_buffer) {
                throw std::runtime_error("failed to allocate Metal shaped bokeh buffers");
            }
            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            id<MTLComputePipelineState> pipeline = colored_kernel ? pipelines.shaped_depth_color : pipelines.shaped_depth_mono;
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:input_buffer offset:0 atIndex:0];
            [enc setBuffer:source_buffer offset:0 atIndex:1];
            [enc setBuffer:depth_buffer offset:0 atIndex:2];
            [enc setBuffer:kernel_buffer offset:0 atIndex:3];
            [enc setBuffer:output_buffer offset:0 atIndex:4];
            [enc setBuffer:params_buffer offset:0 atIndex:5];
            dispatch_2d(enc, pipeline, width, height);
            [enc endEncoding];
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

py::array_t<float> render_sunstar_overlay(
    py::array_t<float, py::array::c_style | py::array::forcecast> sources,
    py::array_t<float, py::array::c_style | py::array::forcecast> jitter,
    int width,
    int height,
    int source_count,
    int spike_count,
    float base_rot,
    float spacing
) {
    py::buffer_info src = sources.request();
    py::buffer_info jit = jitter.request();
    if (src.ndim != 2 || src.shape[1] != 9) {
        throw std::invalid_argument("sources must have shape (source_count, 9)");
    }
    if (jit.ndim != 3 || jit.shape[1] != spike_count || jit.shape[2] != 4) {
        throw std::invalid_argument("jitter must have shape (source_count, spike_count, 4)");
    }
    source_count = std::min<int>(source_count, static_cast<int>(src.shape[0]));
    if (source_count <= 0 || spike_count <= 0 || width <= 0 || height <= 0) {
        return py::array_t<float>({std::max(1, height), std::max(1, width), 3});
    }

    const size_t source_bytes = static_cast<size_t>(src.size) * sizeof(float);
    const size_t jitter_bytes = static_cast<size_t>(jit.size) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3 * sizeof(float);
    py::array_t<float> result({height, width, 3});
    py::buffer_info out = result.request();

    {
        py::gil_scoped_release release;
        @autoreleasepool {
            MetalPipelines& pipelines = metal_pipelines();
            id<MTLBuffer> sources_buffer = [pipelines.device newBufferWithBytes:src.ptr length:source_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> jitter_buffer = [pipelines.device newBufferWithBytes:jit.ptr length:jitter_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buffer = [pipelines.device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
            SunstarParams params{width, height, source_count, spike_count, base_rot, spacing};
            id<MTLBuffer> params_buffer = [pipelines.device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
            if (!sources_buffer || !jitter_buffer || !output_buffer || !params_buffer) {
                throw std::runtime_error("failed to allocate Metal sunstar buffers");
            }

            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            [enc setComputePipelineState:pipelines.sunstar_overlay];
            [enc setBuffer:sources_buffer offset:0 atIndex:0];
            [enc setBuffer:jitter_buffer offset:0 atIndex:1];
            [enc setBuffer:output_buffer offset:0 atIndex:2];
            [enc setBuffer:params_buffer offset:0 atIndex:3];
            dispatch_2d(enc, pipelines.sunstar_overlay, width, height);
            [enc endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.error) {
                std::string message = [[command_buffer.error localizedDescription] UTF8String];
                throw std::runtime_error(message);
            }
            std::memcpy(out.ptr, [output_buffer contents], output_bytes);
        }
    }

    return result;
}

py::array_t<float> apply_swirl_bokeh_direct(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    py::array_t<float, py::array::c_style | py::array::forcecast> depth,
    py::array_t<float, py::array::c_style | py::array::forcecast> radial_norm,
    bool use_depth,
    float center_x,
    float center_y,
    float focus_depth,
    float strength
) {
    py::buffer_info in = image.request();
    py::buffer_info dep = depth.request();
    py::buffer_info rad = radial_norm.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must have shape (height, width, 3)");
    }
    if (rad.ndim != 2 || rad.shape[0] != in.shape[0] || rad.shape[1] != in.shape[1]) {
        throw std::invalid_argument("radial_norm must have shape (height, width)");
    }
    if (use_depth && (dep.ndim != 2 || dep.shape[0] != in.shape[0] || dep.shape[1] != in.shape[1])) {
        throw std::invalid_argument("depth must have shape (height, width)");
    }
    const int height = static_cast<int>(in.shape[0]);
    const int width = static_cast<int>(in.shape[1]);
    const int count = width * height;
    const size_t plane_bytes = static_cast<size_t>(count) * sizeof(float);
    const size_t image_bytes = plane_bytes * 3;

    py::array_t<float> result({height, width, 3});
    py::buffer_info out = result.request();

    {
        py::gil_scoped_release release;
        @autoreleasepool {
            MetalPipelines& pipelines = metal_pipelines();
            id<MTLBuffer> input_buffer = [pipelines.device newBufferWithBytes:in.ptr length:image_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> depth_buffer = [pipelines.device newBufferWithBytes:dep.ptr length:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> radial_buffer = [pipelines.device newBufferWithBytes:rad.ptr length:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buffer = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
            SwirlParams params{width, height, use_depth ? 1 : 0, center_x, center_y, focus_depth, strength};
            id<MTLBuffer> params_buffer = [pipelines.device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
            if (!input_buffer || !depth_buffer || !radial_buffer || !output_buffer || !params_buffer) {
                throw std::runtime_error("failed to allocate Metal swirl buffers");
            }

            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            [enc setComputePipelineState:pipelines.swirl_bokeh];
            [enc setBuffer:input_buffer offset:0 atIndex:0];
            [enc setBuffer:depth_buffer offset:0 atIndex:1];
            [enc setBuffer:radial_buffer offset:0 atIndex:2];
            [enc setBuffer:output_buffer offset:0 atIndex:3];
            [enc setBuffer:params_buffer offset:0 atIndex:4];
            dispatch_2d(enc, pipelines.swirl_bokeh, width, height);
            [enc endEncoding];
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

py::array_t<float> apply_bokeh_color_fringe(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    py::array_t<float, py::array::c_style | py::array::forcecast> depth,
    float focus_depth,
    float strength,
    float resolution_scale
) {
    py::buffer_info in = image.request();
    py::buffer_info dep = depth.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must have shape (height, width, 3)");
    }
    if (dep.ndim != 2 || dep.shape[0] != in.shape[0] || dep.shape[1] != in.shape[1]) {
        throw std::invalid_argument("depth must have shape (height, width)");
    }
    const int height = static_cast<int>(in.shape[0]);
    const int width = static_cast<int>(in.shape[1]);
    const int count = width * height;
    const size_t plane_bytes = static_cast<size_t>(count) * sizeof(float);
    const size_t image_bytes = plane_bytes * 3;
    const float rs = std::max(1.0f, resolution_scale);
    const float s = strength / 100.0f;

    py::array_t<float> result({height, width, 3});
    py::buffer_info out = result.request();

    {
        py::gil_scoped_release release;
        @autoreleasepool {
            MetalPipelines& pipelines = metal_pipelines();
            id<MTLBuffer> input_buffer = [pipelines.device newBufferWithBytes:in.ptr length:image_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> depth_buffer = [pipelines.device newBufferWithBytes:dep.ptr length:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> signed_depth = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> abs_depth = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> defocus_tmp = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> defocus = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> chan_tmp = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> blur_r = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> blur_g = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> blur_b = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buffer = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];

            BokehFringeParams base_params{width, height, 0, focus_depth, s};
            std::vector<float> defocus_weights = gaussian_weights(std::max(0.5f, 2.0f * rs));
            std::vector<float> channel_weights = gaussian_weights((1.0f + 4.0f * s) * rs);
            BokehFringeParams defocus_params = base_params;
            defocus_params.radius = static_cast<int>(defocus_weights.size() / 2);
            BokehFringeParams channel_params = base_params;
            channel_params.radius = static_cast<int>(channel_weights.size() / 2);
            id<MTLBuffer> base_params_buffer = [pipelines.device newBufferWithBytes:&base_params length:sizeof(base_params) options:MTLResourceStorageModeShared];
            id<MTLBuffer> defocus_params_buffer = [pipelines.device newBufferWithBytes:&defocus_params length:sizeof(defocus_params) options:MTLResourceStorageModeShared];
            id<MTLBuffer> channel_params_buffer = [pipelines.device newBufferWithBytes:&channel_params length:sizeof(channel_params) options:MTLResourceStorageModeShared];
            id<MTLBuffer> defocus_weights_buffer = [pipelines.device newBufferWithBytes:defocus_weights.data() length:defocus_weights.size() * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> channel_weights_buffer = [pipelines.device newBufferWithBytes:channel_weights.data() length:channel_weights.size() * sizeof(float) options:MTLResourceStorageModeShared];
            int ch0 = 0, ch1 = 1, ch2 = 2;
            id<MTLBuffer> ch0_buffer = [pipelines.device newBufferWithBytes:&ch0 length:sizeof(ch0) options:MTLResourceStorageModeShared];
            id<MTLBuffer> ch1_buffer = [pipelines.device newBufferWithBytes:&ch1 length:sizeof(ch1) options:MTLResourceStorageModeShared];
            id<MTLBuffer> ch2_buffer = [pipelines.device newBufferWithBytes:&ch2 length:sizeof(ch2) options:MTLResourceStorageModeShared];

            if (!input_buffer || !depth_buffer || !signed_depth || !abs_depth || !defocus_tmp || !defocus || !chan_tmp || !blur_r || !blur_g || !blur_b || !output_buffer || !base_params_buffer || !defocus_params_buffer || !channel_params_buffer || !defocus_weights_buffer || !channel_weights_buffer || !ch0_buffer || !ch1_buffer || !ch2_buffer) {
                throw std::runtime_error("failed to allocate Metal bokeh fringe buffers");
            }

            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.signed_abs];
                [enc setBuffer:depth_buffer offset:0 atIndex:0];
                [enc setBuffer:signed_depth offset:0 atIndex:1];
                [enc setBuffer:abs_depth offset:0 atIndex:2];
                [enc setBuffer:base_params_buffer offset:0 atIndex:3];
                dispatch_1d(enc, pipelines.signed_abs, count);
                [enc endEncoding];
            }

            encode_plane_blur(command_buffer, pipelines, abs_depth, defocus_tmp, defocus, defocus_weights_buffer, defocus_params_buffer, width, height);

            auto encode_channel_blur = [&](id<MTLBuffer> channel_buffer, id<MTLBuffer> output) {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.channel_h];
                [enc setBuffer:input_buffer offset:0 atIndex:0];
                [enc setBuffer:chan_tmp offset:0 atIndex:1];
                [enc setBuffer:channel_weights_buffer offset:0 atIndex:2];
                [enc setBuffer:channel_params_buffer offset:0 atIndex:3];
                [enc setBuffer:channel_buffer offset:0 atIndex:4];
                dispatch_2d(enc, pipelines.channel_h, width, height);
                [enc endEncoding];

                enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.gaussian_v];
                [enc setBuffer:chan_tmp offset:0 atIndex:0];
                [enc setBuffer:output offset:0 atIndex:1];
                [enc setBuffer:channel_weights_buffer offset:0 atIndex:2];
                [enc setBuffer:channel_params_buffer offset:0 atIndex:3];
                dispatch_2d(enc, pipelines.gaussian_v, width, height);
                [enc endEncoding];
            };
            encode_channel_blur(ch0_buffer, blur_r);
            encode_channel_blur(ch1_buffer, blur_g);
            encode_channel_blur(ch2_buffer, blur_b);

            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.compose];
                [enc setBuffer:input_buffer offset:0 atIndex:0];
                [enc setBuffer:signed_depth offset:0 atIndex:1];
                [enc setBuffer:defocus offset:0 atIndex:2];
                [enc setBuffer:blur_r offset:0 atIndex:3];
                [enc setBuffer:blur_g offset:0 atIndex:4];
                [enc setBuffer:blur_b offset:0 atIndex:5];
                [enc setBuffer:output_buffer offset:0 atIndex:6];
                [enc setBuffer:base_params_buffer offset:0 atIndex:7];
                dispatch_1d(enc, pipelines.compose, count);
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

PYBIND11_MODULE(_lens_effect_metal, m) {
    m.def("apply_bokeh_color_fringe", &apply_bokeh_color_fringe, py::arg("image"), py::arg("depth"), py::arg("focus_depth"), py::arg("strength"), py::arg("resolution_scale"));
    m.def("apply_shaped_bokeh_no_depth", &apply_shaped_bokeh_no_depth, py::arg("image"), py::arg("source"), py::arg("kernel"), py::arg("colored_kernel"), py::arg("gain"));
    m.def("apply_shaped_bokeh_depth", &apply_shaped_bokeh_depth, py::arg("image"), py::arg("source"), py::arg("depth"), py::arg("kernel"), py::arg("colored_kernel"), py::arg("focus_depth"), py::arg("strength"));
    m.def("render_sunstar_overlay", &render_sunstar_overlay, py::arg("sources"), py::arg("jitter"), py::arg("width"), py::arg("height"), py::arg("source_count"), py::arg("spike_count"), py::arg("base_rot"), py::arg("spacing"));
    m.def("apply_swirl_bokeh_direct", &apply_swirl_bokeh_direct, py::arg("image"), py::arg("depth"), py::arg("radial_norm"), py::arg("use_depth"), py::arg("center_x"), py::arg("center_y"), py::arg("focus_depth"), py::arg("strength"));
    m.def("metal_available", &metal_available);
}
