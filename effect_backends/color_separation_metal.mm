// Metal color separation backend.
// color_separation_cpu.c と同じ計算(YCbCr 分解、shadow clean、chroma clarity の
// 分離ガウシアン、separation/density ステージ、subtractive/opponent 合成)を
// GPU で行う。CPU 版と float 誤差内で一致することを test_color_separation_effect
// 系のテストで担保する。
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

struct ColorSeparationMetalParams {
    int width;
    int height;
    int radius;
    float shadow_chroma_clean;
    float shadow_threshold;
    float color_separation;
    float chroma_clarity;
    float color_density;
    float subtractive_saturation;
    float opponent_contrast;
};

constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float CS_KR = 0.2126f;
constant float CS_KG = 0.7152f;
constant float CS_KB = 0.0722f;

struct ColorSeparationMetalParams {
    int width;
    int height;
    int radius;
    float shadow_chroma_clean;
    float shadow_threshold;
    float color_separation;
    float chroma_clarity;
    float color_density;
    float subtractive_saturation;
    float opponent_contrast;
};

static inline float cs_clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float cs_smoothstep(float e0, float e1, float x) {
    float t = cs_clamp((x - e0) / (e1 - e0 + 1.0e-12f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

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

static inline void subtractive_saturation_pixel(thread float& r, thread float& g, thread float& b, float amount) {
    amount = cs_clamp(amount, -1.0f, 1.0f);
    if (amount == 0.0f) {
        return;
    }
    float yy = CS_KR * r + CS_KG * g + CS_KB * b;
    float rv = r - yy;
    float gv = g - yy;
    float bv = b - yy;
    float chroma = sqrt(rv * rv + gv * gv + bv * bv);
    float relative_chroma = chroma / (fmax(yy, 0.0f) + 1.0e-4f);
    float chroma_gate = cs_smoothstep(0.025f, 0.42f, relative_chroma);
    float midtone_gate = cs_smoothstep(0.035f, 0.24f, yy) * (1.0f - cs_smoothstep(1.7f, 4.0f, yy));
    float sat_gain = 1.0f;
    float density = 1.0f;
    if (amount > 0.0f) {
        float vivid_rolloff = 1.0f - 0.45f * cs_smoothstep(0.95f, 2.20f, relative_chroma);
        sat_gain = 1.0f + amount * 0.55f * chroma_gate * midtone_gate * vivid_rolloff;
        density = 1.0f - amount * 0.18f * chroma_gate * midtone_gate;
    } else {
        float soften = -amount;
        sat_gain = 1.0f - soften * 0.42f * chroma_gate * midtone_gate;
        density = 1.0f + soften * 0.08f * chroma_gate * midtone_gate;
    }
    r = (yy + rv * sat_gain) * density;
    g = (yy + gv * sat_gain) * density;
    b = (yy + bv * sat_gain) * density;
}

static inline void opponent_contrast_pixel(thread float& r, thread float& g, thread float& b, float opponent_contrast) {
    float y_opp = CS_KR * r + CS_KG * g + CS_KB * b;
    float rg = r - g;
    float by = b - 0.5f * (r + g);
    float opponent_strength = (fabs(rg) + fabs(by)) / (fmax(y_opp, 0.0f) + 1.0e-4f);
    float midtone_mask = cs_smoothstep(0.05f, 0.24f, y_opp);
    float hdr_protect = 1.0f - cs_smoothstep(1.6f, 4.0f, y_opp);
    float vivid_rolloff = 1.0f - 0.70f * cs_smoothstep(0.70f, 1.80f, opponent_strength);
    float opponent_gain = 1.0f
        + cs_clamp(opponent_contrast, 0.0f, 1.0f) * 0.26f * midtone_mask * hdr_protect * vivid_rolloff;
    rg *= opponent_gain;
    by *= opponent_gain;
    float g_new = y_opp - (CS_KR + CS_KB * 0.5f) * rg - CS_KB * by;
    r = g_new + rg;
    g = g_new;
    b = g_new + 0.5f * rg + by;
}

// shadow_clean / clarity なしのときの全処理を 1 パスで行う。
kernel void cs_pointwise(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ColorSeparationMetalParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    int base = int(gid) * 3;
    float src_r = input[base];
    float src_g = input[base + 1];
    float src_b = input[base + 2];
    float yy = CS_KR * src_r + CS_KG * src_g + CS_KB * src_b;
    float cbv = 0.5389f * (src_b - yy);
    float crv = 0.6350f * (src_r - yy);

    if (p.color_separation > 0.0f) {
        float amount = cs_clamp(p.color_separation, 0.0f, 1.0f);
        float chroma = sqrt(cbv * cbv + crv * crv);
        float relative_chroma = chroma / (fmax(yy, 0.0f) + 1.0e-4f);
        float midtone_mask = cs_smoothstep(0.04f, 0.22f, yy);
        float hdr_protect = 1.0f - cs_smoothstep(1.6f, 4.0f, yy);
        float vivid_limit = 1.0f - 0.65f * cs_smoothstep(0.30f, 0.90f, relative_chroma);
        float sep_gain = 1.0f + amount * 0.35f * midtone_mask * hdr_protect * vivid_limit;
        cbv *= sep_gain;
        crv *= sep_gain;
    }

    if (p.color_density != 0.0f) {
        float density_value = cs_clamp(p.color_density, -1.0f, 1.0f);
        float chroma = sqrt(cbv * cbv + crv * crv);
        float relative_chroma = chroma / (fmax(yy, 0.0f) + 1.0e-4f);
        float midtone_mask = cs_smoothstep(0.06f, 0.24f, yy) * (1.0f - cs_smoothstep(1.4f, 3.2f, yy));
        float neutral_gate = cs_smoothstep(0.025f, 0.18f, relative_chroma);
        float density_gain = 1.0f;
        if (density_value > 0.0f) {
            float vivid_rolloff = 1.0f - 0.85f * cs_smoothstep(0.45f, 1.05f, relative_chroma);
            float density_amount = density_value * midtone_mask * neutral_gate * vivid_rolloff;
            float target_chroma = chroma + 0.10f * tanh(chroma / 0.10f);
            density_gain = 1.0f + density_amount * ((target_chroma / (chroma + 1.0e-6f)) - 1.0f);
        } else {
            float vivid_rolloff = 1.0f - 0.35f * cs_smoothstep(0.70f, 1.60f, relative_chroma);
            float density_amount = (-density_value) * midtone_mask * neutral_gate * vivid_rolloff;
            density_gain = 1.0f - 0.40f * density_amount;
        }
        cbv *= density_gain;
        crv *= density_gain;
    }

    float r = yy + 1.5748f * crv;
    float b = yy + 1.8556f * cbv;
    float g = yy - 0.1873f * cbv - 0.4681f * crv;

    if (p.subtractive_saturation != 0.0f) {
        subtractive_saturation_pixel(r, g, b, p.subtractive_saturation);
    }
    if (p.opponent_contrast > 0.0f) {
        opponent_contrast_pixel(r, g, b, p.opponent_contrast);
    }

    float lower_r = src_r < 0.0f ? src_r : 0.0f;
    float lower_g = src_g < 0.0f ? src_g : 0.0f;
    float lower_b = src_b < 0.0f ? src_b : 0.0f;
    output[base] = r > lower_r ? r : lower_r;
    output[base + 1] = g > lower_g ? g : lower_g;
    output[base + 2] = b > lower_b ? b : lower_b;
}

kernel void cs_to_ycbcr(
    const device float* input [[buffer(0)]],
    device float* y_plane [[buffer(1)]],
    device float* cb [[buffer(2)]],
    device float* cr [[buffer(3)]],
    constant ColorSeparationMetalParams& p [[buffer(4)]],
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
    float yy = CS_KR * r + CS_KG * g + CS_KB * b;
    y_plane[gid] = yy;
    cb[gid] = (b - yy) / 1.8556f;
    cr[gid] = (r - yy) / 1.5748f;
}

kernel void cs_shadow_clean(
    device float* cb [[buffer(0)]],
    device float* cr [[buffer(1)]],
    const device float* y_plane [[buffer(2)]],
    constant ColorSeparationMetalParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    float threshold = p.shadow_threshold > 1.0e-4f ? p.shadow_threshold : 1.0e-4f;
    float clean_amount = cs_clamp(p.shadow_chroma_clean, 0.0f, 1.0f) * 0.9f;
    float yy = y_plane[gid];
    float cbv = cb[gid];
    float crv = cr[gid];
    float chroma = sqrt(cbv * cbv + crv * crv);
    float relative_chroma = chroma / (fmax(yy, 0.0f) + 1.0e-4f);
    float shadow_mask = 1.0f - cs_smoothstep(threshold * 0.35f, threshold, yy);
    float vivid_protect = cs_smoothstep(0.12f, 0.45f, relative_chroma);
    float clean_scale = 1.0f - clean_amount * shadow_mask * (1.0f - vivid_protect);
    cb[gid] = cbv * clean_scale;
    cr[gid] = crv * clean_scale;
}

kernel void cs_clarity_weight(
    const device float* cb [[buffer(0)]],
    const device float* cr [[buffer(1)]],
    const device float* y_plane [[buffer(2)]],
    device float* weight [[buffer(3)]],
    constant ColorSeparationMetalParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    float yy = y_plane[gid];
    float cbv = cb[gid];
    float crv = cr[gid];
    float chroma = sqrt(cbv * cbv + crv * crv);
    float relative_chroma = chroma / (fmax(yy, 0.0f) + 1.0e-4f);
    float midtone_mask = cs_smoothstep(0.035f, 0.18f, yy);
    float hdr_protect = 1.0f - cs_smoothstep(1.6f, 4.0f, yy);
    float neutral_gate = cs_smoothstep(0.015f, 0.10f, relative_chroma);
    float vivid_limit = 1.0f - 0.45f * cs_smoothstep(0.80f, 1.80f, relative_chroma);
    weight[gid] = midtone_mask * hdr_protect * neutral_gate * vivid_limit;
}

kernel void cs_gauss_h(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant ColorSeparationMetalParams& p [[buffer(3)]],
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
        sum += src[y * p.width + sx] * weights[k + p.radius];
    }
    dst[y * p.width + x] = sum;
}

kernel void cs_gauss_v(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant ColorSeparationMetalParams& p [[buffer(3)]],
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
        sum += src[sy * p.width + x] * weights[k + p.radius];
    }
    dst[y * p.width + x] = sum;
}

kernel void cs_clarity_apply(
    device float* channel [[buffer(0)]],
    const device float* local_blur [[buffer(1)]],
    const device float* base_blur [[buffer(2)]],
    const device float* weight [[buffer(3)]],
    constant ColorSeparationMetalParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    float scale = cs_clamp(p.chroma_clarity, -1.0f, 1.0f) * 1.15f;
    channel[gid] = channel[gid] + (local_blur[gid] - base_blur[gid]) * scale * weight[gid];
}

// separation → density の 2 ステージ(いずれも pointwise)を1パスで適用。
kernel void cs_sep_density(
    device float* cb [[buffer(0)]],
    device float* cr [[buffer(1)]],
    const device float* y_plane [[buffer(2)]],
    constant ColorSeparationMetalParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    float yy = y_plane[gid];
    float cbv = cb[gid];
    float crv = cr[gid];

    if (p.color_separation > 0.0f) {
        float amount = cs_clamp(p.color_separation, 0.0f, 1.0f);
        float chroma = sqrt(cbv * cbv + crv * crv);
        float relative_chroma = chroma / (fmax(yy, 0.0f) + 1.0e-4f);
        float midtone_mask = cs_smoothstep(0.04f, 0.22f, yy);
        float hdr_protect = 1.0f - cs_smoothstep(1.6f, 4.0f, yy);
        float vivid_limit = 1.0f - 0.65f * cs_smoothstep(0.30f, 0.90f, relative_chroma);
        float sep_gain = 1.0f + amount * 0.35f * midtone_mask * hdr_protect * vivid_limit;
        cbv *= sep_gain;
        crv *= sep_gain;
    }

    if (p.color_density != 0.0f) {
        float density_value = cs_clamp(p.color_density, -1.0f, 1.0f);
        float chroma = sqrt(cbv * cbv + crv * crv);
        float relative_chroma = chroma / (fmax(yy, 0.0f) + 1.0e-4f);
        float midtone_mask = cs_smoothstep(0.06f, 0.24f, yy) * (1.0f - cs_smoothstep(1.4f, 3.2f, yy));
        float neutral_gate = cs_smoothstep(0.025f, 0.18f, relative_chroma);
        float density_gain = 1.0f;
        if (density_value > 0.0f) {
            float vivid_rolloff = 1.0f - 0.85f * cs_smoothstep(0.45f, 1.05f, relative_chroma);
            float density_amount = density_value * midtone_mask * neutral_gate * vivid_rolloff;
            float target_chroma = chroma + 0.10f * tanh(chroma / 0.10f);
            density_gain = 1.0f + density_amount * ((target_chroma / (chroma + 1.0e-6f)) - 1.0f);
        } else {
            float vivid_rolloff = 1.0f - 0.35f * cs_smoothstep(0.70f, 1.60f, relative_chroma);
            float density_amount = (-density_value) * midtone_mask * neutral_gate * vivid_rolloff;
            density_gain = 1.0f - 0.40f * density_amount;
        }
        cbv *= density_gain;
        crv *= density_gain;
    }

    cb[gid] = cbv;
    cr[gid] = crv;
}

kernel void cs_write_output(
    const device float* input [[buffer(0)]],
    const device float* y_plane [[buffer(1)]],
    const device float* cb [[buffer(2)]],
    const device float* cr [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant ColorSeparationMetalParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    int base = int(gid) * 3;
    float yy = y_plane[gid];
    float r = yy + 1.5748f * cr[gid];
    float b = yy + 1.8556f * cb[gid];
    float g = yy - 0.1873f * cb[gid] - 0.4681f * cr[gid];

    if (p.subtractive_saturation != 0.0f) {
        subtractive_saturation_pixel(r, g, b, p.subtractive_saturation);
    }
    if (p.opponent_contrast > 0.0f) {
        opponent_contrast_pixel(r, g, b, p.opponent_contrast);
    }

    float src_r = input[base];
    float src_g = input[base + 1];
    float src_b = input[base + 2];
    float lower_r = src_r < 0.0f ? src_r : 0.0f;
    float lower_g = src_g < 0.0f ? src_g : 0.0f;
    float lower_b = src_b < 0.0f ? src_b : 0.0f;
    output[base] = r > lower_r ? r : lower_r;
    output[base + 1] = g > lower_g ? g : lower_g;
    output[base + 2] = b > lower_b ? b : lower_b;
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
    id<MTLComputePipelineState> pointwise;
    id<MTLComputePipelineState> to_ycbcr;
    id<MTLComputePipelineState> shadow_clean;
    id<MTLComputePipelineState> clarity_weight;
    id<MTLComputePipelineState> gauss_h;
    id<MTLComputePipelineState> gauss_v;
    id<MTLComputePipelineState> clarity_apply;
    id<MTLComputePipelineState> sep_density;
    id<MTLComputePipelineState> write_output;
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
                state.pointwise = make_pipeline(state.device, state.library, @"cs_pointwise");
                state.to_ycbcr = make_pipeline(state.device, state.library, @"cs_to_ycbcr");
                state.shadow_clean = make_pipeline(state.device, state.library, @"cs_shadow_clean");
                state.clarity_weight = make_pipeline(state.device, state.library, @"cs_clarity_weight");
                state.gauss_h = make_pipeline(state.device, state.library, @"cs_gauss_h");
                state.gauss_v = make_pipeline(state.device, state.library, @"cs_gauss_v");
                state.clarity_apply = make_pipeline(state.device, state.library, @"cs_clarity_apply");
                state.sep_density = make_pipeline(state.device, state.library, @"cs_sep_density");
                state.write_output = make_pipeline(state.device, state.library, @"cs_write_output");
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

void dispatch_2d(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, NSUInteger width, NSUInteger height) {
    [encoder setComputePipelineState:pipeline];
    NSUInteger tw = std::max<NSUInteger>(1, pipeline.threadExecutionWidth);
    NSUInteger th = std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / tw);
    if (th > 16) {
        th = 16;
    }
    [encoder dispatchThreads:MTLSizeMake(width, height, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

// color_separation_cpu.c の make_gaussian_kernel と同一(float 累積を含めて一致させる)。
std::vector<float> gaussian_kernel(float sigma) {
    int ksize = static_cast<int>(std::floor(sigma * 6.0f + 1.0f + 0.5f));
    if ((ksize & 1) == 0) {
        ++ksize;
    }
    if (ksize < 3) {
        ksize = 3;
    }
    const int radius = ksize / 2;
    std::vector<float> kernel(static_cast<size_t>(radius) * 2 + 1);
    const float denom = 2.0f * sigma * sigma;
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        const float v = std::exp(-(static_cast<float>(i * i)) / denom);
        kernel[static_cast<size_t>(i + radius)] = v;
        sum += v;
    }
    const float inv_sum = sum != 0.0f ? 1.0f / sum : 1.0f;
    for (float& v : kernel) {
        v *= inv_sum;
    }
    return kernel;
}

}  // namespace

bool metal_available() {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

py::array_t<float> apply_color_separation(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    float shadow_chroma_clean,
    float shadow_threshold,
    float color_separation,
    float chroma_clarity,
    float color_density,
    float subtractive_saturation,
    float opponent_contrast
) {
    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must be a 3D RGB float32 array");
    }
    const int height = static_cast<int>(in.shape[0]);
    const int width = static_cast<int>(in.shape[1]);
    const int count = width * height;
    const size_t plane_bytes = static_cast<size_t>(count) * sizeof(float);
    const size_t image_bytes = plane_bytes * 3;

    std::vector<py::ssize_t> shape(in.shape.begin(), in.shape.end());
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    ColorSeparationMetalParams params{
        width, height, 0,
        shadow_chroma_clean, shadow_threshold, color_separation,
        chroma_clarity, color_density, subtractive_saturation, opponent_contrast,
    };

    {
        py::gil_scoped_release release;
        @autoreleasepool {
            MetalPipelines& pipelines = metal_pipelines();
            BufferBinding input_binding = make_buffer_for_input(pipelines.device, in.ptr, image_bytes);
            BufferBinding output_binding = make_buffer_for_output(pipelines.device, out.ptr, image_bytes);
            id<MTLBuffer> input_buffer = input_binding.buffer;
            id<MTLBuffer> output_buffer = output_binding.buffer;
            id<MTLBuffer> params_buffer = [pipelines.device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
            if (!input_buffer || !output_buffer || !params_buffer) {
                throw std::runtime_error("failed to allocate Metal color separation buffers");
            }

            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];

            const bool pointwise_only = shadow_chroma_clean == 0.0f && chroma_clarity == 0.0f;
            if (pointwise_only) {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setBuffer:input_buffer offset:input_binding.offset atIndex:0];
                [enc setBuffer:output_buffer offset:output_binding.offset atIndex:1];
                [enc setBuffer:params_buffer offset:0 atIndex:2];
                dispatch_1d(enc, pipelines.pointwise, static_cast<NSUInteger>(count));
                [enc endEncoding];
            } else {
                id<MTLBuffer> y_plane = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> cb = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> cr = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
                if (!y_plane || !cb || !cr) {
                    throw std::runtime_error("failed to allocate Metal color separation planes");
                }

                {
                    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                    [enc setBuffer:input_buffer offset:input_binding.offset atIndex:0];
                    [enc setBuffer:y_plane offset:0 atIndex:1];
                    [enc setBuffer:cb offset:0 atIndex:2];
                    [enc setBuffer:cr offset:0 atIndex:3];
                    [enc setBuffer:params_buffer offset:0 atIndex:4];
                    dispatch_1d(enc, pipelines.to_ycbcr, static_cast<NSUInteger>(count));
                    [enc endEncoding];
                }

                if (shadow_chroma_clean > 0.0f && shadow_threshold > 0.0f) {
                    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                    [enc setBuffer:cb offset:0 atIndex:0];
                    [enc setBuffer:cr offset:0 atIndex:1];
                    [enc setBuffer:y_plane offset:0 atIndex:2];
                    [enc setBuffer:params_buffer offset:0 atIndex:3];
                    dispatch_1d(enc, pipelines.shadow_clean, static_cast<NSUInteger>(count));
                    [enc endEncoding];
                }

                if (chroma_clarity != 0.0f) {
                    id<MTLBuffer> tmp = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
                    id<MTLBuffer> local_blur = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
                    id<MTLBuffer> base_blur = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
                    id<MTLBuffer> weight = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
                    if (!tmp || !local_blur || !base_blur || !weight) {
                        throw std::runtime_error("failed to allocate Metal clarity buffers");
                    }

                    std::vector<float> kernel_local = gaussian_kernel(1.2f);
                    std::vector<float> kernel_base = gaussian_kernel(7.0f);
                    id<MTLBuffer> local_weights = [pipelines.device newBufferWithBytes:kernel_local.data() length:kernel_local.size() * sizeof(float) options:MTLResourceStorageModeShared];
                    id<MTLBuffer> base_weights = [pipelines.device newBufferWithBytes:kernel_base.data() length:kernel_base.size() * sizeof(float) options:MTLResourceStorageModeShared];
                    ColorSeparationMetalParams local_params = params;
                    local_params.radius = static_cast<int>(kernel_local.size() / 2);
                    ColorSeparationMetalParams base_params = params;
                    base_params.radius = static_cast<int>(kernel_base.size() / 2);
                    id<MTLBuffer> local_params_buffer = [pipelines.device newBufferWithBytes:&local_params length:sizeof(local_params) options:MTLResourceStorageModeShared];
                    id<MTLBuffer> base_params_buffer = [pipelines.device newBufferWithBytes:&base_params length:sizeof(base_params) options:MTLResourceStorageModeShared];
                    if (!local_weights || !base_weights || !local_params_buffer || !base_params_buffer) {
                        throw std::runtime_error("failed to allocate Metal clarity kernel buffers");
                    }

                    auto encode_blur = [&](id<MTLBuffer> src, id<MTLBuffer> dst, id<MTLBuffer> weights, id<MTLBuffer> blur_params) {
                        {
                            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                            [enc setBuffer:src offset:0 atIndex:0];
                            [enc setBuffer:tmp offset:0 atIndex:1];
                            [enc setBuffer:weights offset:0 atIndex:2];
                            [enc setBuffer:blur_params offset:0 atIndex:3];
                            dispatch_2d(enc, pipelines.gauss_h, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height));
                            [enc endEncoding];
                        }
                        {
                            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                            [enc setBuffer:tmp offset:0 atIndex:0];
                            [enc setBuffer:dst offset:0 atIndex:1];
                            [enc setBuffer:weights offset:0 atIndex:2];
                            [enc setBuffer:blur_params offset:0 atIndex:3];
                            dispatch_2d(enc, pipelines.gauss_v, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height));
                            [enc endEncoding];
                        }
                    };
                    auto encode_clarity_apply = [&](id<MTLBuffer> channel) {
                        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                        [enc setBuffer:channel offset:0 atIndex:0];
                        [enc setBuffer:local_blur offset:0 atIndex:1];
                        [enc setBuffer:base_blur offset:0 atIndex:2];
                        [enc setBuffer:weight offset:0 atIndex:3];
                        [enc setBuffer:params_buffer offset:0 atIndex:4];
                        dispatch_1d(enc, pipelines.clarity_apply, static_cast<NSUInteger>(count));
                        [enc endEncoding];
                    };

                    {
                        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                        [enc setBuffer:cb offset:0 atIndex:0];
                        [enc setBuffer:cr offset:0 atIndex:1];
                        [enc setBuffer:y_plane offset:0 atIndex:2];
                        [enc setBuffer:weight offset:0 atIndex:3];
                        [enc setBuffer:params_buffer offset:0 atIndex:4];
                        dispatch_1d(enc, pipelines.clarity_weight, static_cast<NSUInteger>(count));
                        [enc endEncoding];
                    }
                    encode_blur(cb, local_blur, local_weights, local_params_buffer);
                    encode_blur(cb, base_blur, base_weights, base_params_buffer);
                    encode_clarity_apply(cb);
                    encode_blur(cr, local_blur, local_weights, local_params_buffer);
                    encode_blur(cr, base_blur, base_weights, base_params_buffer);
                    encode_clarity_apply(cr);
                }

                if (color_separation > 0.0f || color_density != 0.0f) {
                    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                    [enc setBuffer:cb offset:0 atIndex:0];
                    [enc setBuffer:cr offset:0 atIndex:1];
                    [enc setBuffer:y_plane offset:0 atIndex:2];
                    [enc setBuffer:params_buffer offset:0 atIndex:3];
                    dispatch_1d(enc, pipelines.sep_density, static_cast<NSUInteger>(count));
                    [enc endEncoding];
                }

                {
                    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                    [enc setBuffer:input_buffer offset:input_binding.offset atIndex:0];
                    [enc setBuffer:y_plane offset:0 atIndex:1];
                    [enc setBuffer:cb offset:0 atIndex:2];
                    [enc setBuffer:cr offset:0 atIndex:3];
                    [enc setBuffer:output_buffer offset:output_binding.offset atIndex:4];
                    [enc setBuffer:params_buffer offset:0 atIndex:5];
                    dispatch_1d(enc, pipelines.write_output, static_cast<NSUInteger>(count));
                    [enc endEncoding];
                }
            }

            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.error) {
                std::string message = [[command_buffer.error localizedDescription] UTF8String];
                throw std::runtime_error(message);
            }
            finish_output_binding(output_binding, out.ptr, image_bytes);
        }
    }

    return result;
}

PYBIND11_MODULE(_color_separation_metal, m) {
    m.doc() = "Metal color separation backend for Platypus";
    m.def("metal_available", &metal_available);
    m.def(
        "apply_color_separation",
        &apply_color_separation,
        py::arg("image"),
        py::arg("shadow_chroma_clean") = 0.0f,
        py::arg("shadow_threshold") = 0.2f,
        py::arg("color_separation") = 0.0f,
        py::arg("chroma_clarity") = 0.0f,
        py::arg("color_density") = 0.0f,
        py::arg("subtractive_saturation") = 0.0f,
        py::arg("opponent_contrast") = 0.0f
    );
}
