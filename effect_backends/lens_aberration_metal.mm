#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct LensAberrationParams {
    int width;
    int height;
    float strength;
    float resolution_scale;
};

struct LongitudinalParams {
    int width;
    int height;
    int radius;
    float strength;
    float focus_depth;
};

struct SphericalParams {
    int width;
    int height;
    int blur_radius;
    int glow_radius;
    int has_depth;
    float strength;
    float aperture;
    float focus_depth;
    float highlight_threshold;
    float resolution_scale;
    float total_blur_strength;
    float avg_blur_sigma;
    float glow_strength;
    float contrast_reduction;
    float pivot;
};

constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct LensAberrationParams {
    int width;
    int height;
    float strength;
    float resolution_scale;
};

struct LongitudinalParams {
    int width;
    int height;
    int radius;
    float strength;
    float focus_depth;
};

struct SphericalParams {
    int width;
    int height;
    int blur_radius;
    int glow_radius;
    int has_depth;
    float strength;
    float aperture;
    float focus_depth;
    float highlight_threshold;
    float resolution_scale;
    float total_blur_strength;
    float avg_blur_sigma;
    float glow_strength;
    float contrast_reduction;
    float pivot;
};

static inline int reflect_scipy(int p, int len) {
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

static inline float sample_channel(
    const device float* input,
    int width,
    int height,
    float fx,
    float fy,
    int ch
) {
    fx = clamp(fx, 0.0f, float(width - 1));
    fy = clamp(fy, 0.0f, float(height - 1));

    int x0 = clamp(int(floor(fx)), 0, width - 1);
    int y0 = clamp(int(floor(fy)), 0, height - 1);
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    float tx = fx - float(x0);
    float ty = fy - float(y0);

    float c00 = input[(y0 * width + x0) * 3 + ch];
    float c10 = input[(y0 * width + x1) * 3 + ch];
    float c01 = input[(y1 * width + x0) * 3 + ch];
    float c11 = input[(y1 * width + x1) * 3 + ch];
    float cx0 = mix(c00, c10, tx);
    float cx1 = mix(c01, c11, tx);
    return mix(cx0, cx1, ty);
}

kernel void lateral_ca(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant LensAberrationParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }

    float cx = float(p.width) * 0.5f;
    float cy = float(p.height) * 0.5f;
    float dx = float(x) - cx;
    float dy = float(y) - cy;
    float dist = sqrt(dx * dx + dy * dy);
    float dir_x = dist > 1.0e-6f ? dx / dist : 0.0f;
    float dir_y = dist > 1.0e-6f ? dy / dist : 0.0f;
    float norm = dist / max(cx, cy);
    float base_shift = p.strength * 2.0f * max(p.resolution_scale, 0.05f) * norm;

    int base = (y * p.width + x) * 3;
    output[base + 0] = sample_channel(input, p.width, p.height, float(x) - dir_x * base_shift * 0.5f, float(y) - dir_y * base_shift * 0.5f, 0);
    output[base + 1] = sample_channel(input, p.width, p.height, float(x) - dir_x * base_shift, float(y) - dir_y * base_shift, 1);
    output[base + 2] = sample_channel(input, p.width, p.height, float(x) - dir_x * base_shift * 1.5f, float(y) - dir_y * base_shift * 1.5f, 2);
}

kernel void longitudinal_defocus(
    const device float* depth [[buffer(0)]],
    device float* defocus [[buffer(1)]],
    constant LongitudinalParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    defocus[gid] = fabs(depth[gid] - p.focus_depth);
}

kernel void gaussian_plane_horizontal(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant LongitudinalParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    float sum = 0.0f;
    for (int k = -p.radius; k <= p.radius; ++k) {
        int sx = reflect_scipy(x + k, p.width);
        sum += input[y * p.width + sx] * weights[k + p.radius];
    }
    output[y * p.width + x] = sum;
}

kernel void gaussian_plane_vertical(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant LongitudinalParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    float sum = 0.0f;
    for (int k = -p.radius; k <= p.radius; ++k) {
        int sy = reflect_scipy(y + k, p.height);
        sum += input[sy * p.width + x] * weights[k + p.radius];
    }
    output[y * p.width + x] = sum;
}

kernel void gaussian_channel_horizontal(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant LongitudinalParams& p [[buffer(3)]],
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
        int sx = reflect_scipy(x + k, p.width);
        sum += input[(y * p.width + sx) * 3 + channel] * weights[k + p.radius];
    }
    output[y * p.width + x] = sum;
}

kernel void gaussian_plane_to_plane_vertical(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant LongitudinalParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    float sum = 0.0f;
    for (int k = -p.radius; k <= p.radius; ++k) {
        int sy = reflect_scipy(y + k, p.height);
        sum += input[sy * p.width + x] * weights[k + p.radius];
    }
    output[y * p.width + x] = sum;
}

kernel void longitudinal_compose(
    const device float* input [[buffer(0)]],
    const device float* defocus [[buffer(1)]],
    const device float* blur_r [[buffer(2)]],
    const device float* blur_b [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant LongitudinalParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    float s = clamp(p.strength, 0.0f, 2.0f);
    float weight = clamp(defocus[gid] * (0.5f + 0.25f * s), 0.0f, 1.0f);
    int base = int(gid) * 3;
    output[base + 0] = input[base + 0] * (1.0f - weight) + blur_r[gid] * weight;
    output[base + 1] = input[base + 1];
    output[base + 2] = input[base + 2] * (1.0f - weight) + blur_b[gid] * weight;
}

kernel void spherical_prepare(
    const device float* input [[buffer(0)]],
    const device float* depth [[buffer(1)]],
    device float* highlight [[buffer(2)]],
    device float* depth_weight [[buffer(3)]],
    device float* edge [[buffer(4)]],
    constant SphericalParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int idx = y * p.width + x;
    int base = idx * 3;
    float lum = (input[base] + input[base + 1] + input[base + 2]) / 3.0f;
    highlight[idx] = clamp((lum - p.highlight_threshold) / (1.0f - p.highlight_threshold), 0.0f, 1.0f);

    if (p.has_depth != 0) {
        float d = fabs(depth[idx] - p.focus_depth);
        float defocus = clamp(d * 3.0f, 0.0f, 1.0f);
        depth_weight[idx] = clamp(0.2f + defocus, 0.0f, 1.0f);
    } else {
        depth_weight[idx] = 1.0f;
    }

    float cx = float(p.width) * 0.5f;
    float cy = float(p.height) * 0.5f;
    float dx = float(x) - cx;
    float dy = float(y) - cy;
    float norm = sqrt(dx * dx + dy * dy) / max(cx, cy);
    edge[idx] = norm * norm;
}

kernel void spherical_blur_sigma(
    const device float* depth_weight [[buffer(0)]],
    const device float* edge_weight [[buffer(1)]],
    device float* blur_sigma [[buffer(2)]],
    constant SphericalParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    blur_sigma[gid] = p.total_blur_strength * depth_weight[gid] * (0.5f + 0.5f * edge_weight[gid]);
}

kernel void gaussian_rgb_horizontal(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant LongitudinalParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    for (int ch = 0; ch < 3; ++ch) {
        float sum = 0.0f;
        for (int k = -p.radius; k <= p.radius; ++k) {
            int sx = reflect_scipy(x + k, p.width);
            sum += input[(y * p.width + sx) * 3 + ch] * weights[k + p.radius];
        }
        output[(y * p.width + x) * 3 + ch] = sum;
    }
}

kernel void gaussian_rgb_vertical(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant LongitudinalParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    for (int ch = 0; ch < 3; ++ch) {
        float sum = 0.0f;
        for (int k = -p.radius; k <= p.radius; ++k) {
            int sy = reflect_scipy(y + k, p.height);
            sum += input[(sy * p.width + x) * 3 + ch] * weights[k + p.radius];
        }
        output[(y * p.width + x) * 3 + ch] = sum;
    }
}

kernel void spherical_glow_src(
    const device float* input [[buffer(0)]],
    const device float* highlight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant SphericalParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    float h = highlight[gid];
    int base = int(gid) * 3;
    output[base] = input[base] * h;
    output[base + 1] = input[base + 1] * h;
    output[base + 2] = input[base + 2] * h;
}

static inline float channel_blur3(const device float* rgb, int base, int ch, const device float* weights, int radius) {
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        int sc = reflect_scipy(ch + k, 3);
        sum += rgb[base + sc] * weights[k + radius];
    }
    return sum;
}

kernel void spherical_precontrast(
    const device float* input [[buffer(0)]],
    const device float* highlight [[buffer(1)]],
    const device float* blur_sigma [[buffer(2)]],
    const device float* blurred_spatial [[buffer(3)]],
    const device float* glow_spatial [[buffer(4)]],
    const device float* blur_weights [[buffer(5)]],
    const device float* glow_weights [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant SphericalParams& p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    int base = int(gid) * 3;
    float h = highlight[gid];
    float blend_ratio = clamp(blur_sigma[gid] / (p.total_blur_strength + 0.01f), 0.0f, 0.8f);
    for (int ch = 0; ch < 3; ++ch) {
        float blurred = channel_blur3(blurred_spatial, base, ch, blur_weights, p.blur_radius);
        float glow = channel_blur3(glow_spatial, base, ch, glow_weights, p.glow_radius);
        float v = input[base + ch] * (1.0f - h * p.glow_strength) + glow * p.glow_strength;
        output[base + ch] = v * (1.0f - blend_ratio) + blurred * blend_ratio;
    }
}

kernel void spherical_contrast(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SphericalParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height * 3;
    if (int(gid) >= count) {
        return;
    }
    output[gid] = (input[gid] - p.pivot) * p.contrast_reduction + p.pivot;
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
    id<MTLComputePipelineState> lateral_ca;
    id<MTLComputePipelineState> longitudinal_defocus;
    id<MTLComputePipelineState> gaussian_plane_horizontal;
    id<MTLComputePipelineState> gaussian_plane_vertical;
    id<MTLComputePipelineState> gaussian_channel_horizontal;
    id<MTLComputePipelineState> gaussian_plane_to_plane_vertical;
    id<MTLComputePipelineState> longitudinal_compose;
    id<MTLComputePipelineState> spherical_prepare;
    id<MTLComputePipelineState> spherical_blur_sigma;
    id<MTLComputePipelineState> gaussian_rgb_horizontal;
    id<MTLComputePipelineState> gaussian_rgb_vertical;
    id<MTLComputePipelineState> spherical_glow_src;
    id<MTLComputePipelineState> spherical_precontrast;
    id<MTLComputePipelineState> spherical_contrast;
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
                state.lateral_ca = make_pipeline(state.device, state.library, @"lateral_ca");
                state.longitudinal_defocus = make_pipeline(state.device, state.library, @"longitudinal_defocus");
                state.gaussian_plane_horizontal = make_pipeline(state.device, state.library, @"gaussian_plane_horizontal");
                state.gaussian_plane_vertical = make_pipeline(state.device, state.library, @"gaussian_plane_vertical");
                state.gaussian_channel_horizontal = make_pipeline(state.device, state.library, @"gaussian_channel_horizontal");
                state.gaussian_plane_to_plane_vertical = make_pipeline(state.device, state.library, @"gaussian_plane_to_plane_vertical");
                state.longitudinal_compose = make_pipeline(state.device, state.library, @"longitudinal_compose");
                state.spherical_prepare = make_pipeline(state.device, state.library, @"spherical_prepare");
                state.spherical_blur_sigma = make_pipeline(state.device, state.library, @"spherical_blur_sigma");
                state.gaussian_rgb_horizontal = make_pipeline(state.device, state.library, @"gaussian_rgb_horizontal");
                state.gaussian_rgb_vertical = make_pipeline(state.device, state.library, @"gaussian_rgb_vertical");
                state.spherical_glow_src = make_pipeline(state.device, state.library, @"spherical_glow_src");
                state.spherical_precontrast = make_pipeline(state.device, state.library, @"spherical_precontrast");
                state.spherical_contrast = make_pipeline(state.device, state.library, @"spherical_contrast");
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

void dispatch_2d(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, int width, int height) {
    NSUInteger tw = pipeline.threadExecutionWidth;
    NSUInteger th = std::max<NSUInteger>(1, std::min<NSUInteger>(16, pipeline.maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(1, tw)));
    MTLSize threads_per_group = MTLSizeMake(tw, th, 1);
    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(width), static_cast<NSUInteger>(height), 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads_per_group];
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

py::array_t<float> apply_lateral_ca(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    float strength,
    float resolution_scale
) {
    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must have shape (height, width, 3)");
    }
    const int height = static_cast<int>(in.shape[0]);
    const int width = static_cast<int>(in.shape[1]);
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument("image dimensions must be positive");
    }
    if (static_cast<size_t>(height) * static_cast<size_t>(width) > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("image is too large");
    }

    py::array_t<float> result({height, width, 3});
    py::buffer_info out = result.request();

    {
        py::gil_scoped_release release;
        @autoreleasepool {
            MetalPipelines& pipelines = metal_pipelines();
            const size_t bytes = static_cast<size_t>(height) * static_cast<size_t>(width) * 3 * sizeof(float);

            id<MTLBuffer> input_buffer = [pipelines.device newBufferWithBytes:in.ptr length:bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buffer = [pipelines.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            LensAberrationParams params{width, height, strength, resolution_scale};
            id<MTLBuffer> params_buffer = [pipelines.device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];

            if (!input_buffer || !output_buffer || !params_buffer) {
                throw std::runtime_error("failed to allocate Metal lens aberration buffers");
            }

            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:pipelines.lateral_ca];
            [encoder setBuffer:input_buffer offset:0 atIndex:0];
            [encoder setBuffer:output_buffer offset:0 atIndex:1];
            [encoder setBuffer:params_buffer offset:0 atIndex:2];
            dispatch_2d(encoder, pipelines.lateral_ca, width, height);
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];

            if (command_buffer.error) {
                std::string message = [[command_buffer.error localizedDescription] UTF8String];
                throw std::runtime_error(message);
            }

            std::memcpy(out.ptr, [output_buffer contents], bytes);
        }
    }

    return result;
}

py::array_t<float> apply_longitudinal_ca(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    py::array_t<float, py::array::c_style | py::array::forcecast> depth,
    float strength,
    float focus_depth,
    float resolution_scale
) {
    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must have shape (height, width, 3)");
    }
    py::buffer_info dm = depth.request();
    if (dm.ndim != 2 || dm.shape[0] != in.shape[0] || dm.shape[1] != in.shape[1]) {
        throw std::invalid_argument("depth must have shape (height, width)");
    }
    const int height = static_cast<int>(in.shape[0]);
    const int width = static_cast<int>(in.shape[1]);
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument("image dimensions must be positive");
    }

    py::array_t<float> result({height, width, 3});
    py::buffer_info out = result.request();

    {
        py::gil_scoped_release release;
        @autoreleasepool {
            MetalPipelines& pipelines = metal_pipelines();
            const int count = width * height;
            const size_t plane_bytes = static_cast<size_t>(count) * sizeof(float);
            const size_t image_bytes = plane_bytes * 3;

            id<MTLBuffer> input_buffer = [pipelines.device newBufferWithBytes:in.ptr length:image_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> depth_buffer = [pipelines.device newBufferWithBytes:dm.ptr length:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> defocus_raw = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> defocus_temp = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> defocus_blur = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> r_temp = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> r_blur = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_temp = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_blur = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buffer = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];

            const float rs = std::max(0.05f, resolution_scale);
            std::vector<float> defocus_weights = gaussian_weights(std::max(0.5f, 2.0f * rs));
            std::vector<float> fringe_weights = gaussian_weights((0.6f + 1.4f * std::clamp(strength, 0.0f, 2.0f)) * rs);
            LongitudinalParams defocus_params{width, height, static_cast<int>(defocus_weights.size() / 2), strength, focus_depth};
            LongitudinalParams fringe_params{width, height, static_cast<int>(fringe_weights.size() / 2), strength, focus_depth};
            id<MTLBuffer> defocus_params_buffer = [pipelines.device newBufferWithBytes:&defocus_params length:sizeof(defocus_params) options:MTLResourceStorageModeShared];
            id<MTLBuffer> fringe_params_buffer = [pipelines.device newBufferWithBytes:&fringe_params length:sizeof(fringe_params) options:MTLResourceStorageModeShared];
            id<MTLBuffer> defocus_weights_buffer = [pipelines.device newBufferWithBytes:defocus_weights.data() length:defocus_weights.size() * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> fringe_weights_buffer = [pipelines.device newBufferWithBytes:fringe_weights.data() length:fringe_weights.size() * sizeof(float) options:MTLResourceStorageModeShared];
            int channel_r = 0;
            int channel_b = 2;
            id<MTLBuffer> channel_r_buffer = [pipelines.device newBufferWithBytes:&channel_r length:sizeof(channel_r) options:MTLResourceStorageModeShared];
            id<MTLBuffer> channel_b_buffer = [pipelines.device newBufferWithBytes:&channel_b length:sizeof(channel_b) options:MTLResourceStorageModeShared];

            if (!input_buffer || !depth_buffer || !defocus_raw || !defocus_temp || !defocus_blur || !r_temp || !r_blur || !b_temp || !b_blur || !output_buffer || !defocus_params_buffer || !fringe_params_buffer || !defocus_weights_buffer || !fringe_weights_buffer || !channel_r_buffer || !channel_b_buffer) {
                throw std::runtime_error("failed to allocate Metal longitudinal CA buffers");
            }

            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];

            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.longitudinal_defocus];
                [enc setBuffer:depth_buffer offset:0 atIndex:0];
                [enc setBuffer:defocus_raw offset:0 atIndex:1];
                [enc setBuffer:defocus_params_buffer offset:0 atIndex:2];
                NSUInteger tw = pipelines.longitudinal_defocus.threadExecutionWidth;
                [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
                [enc endEncoding];
            }

            auto blur_plane = [&](id<MTLBuffer> src, id<MTLBuffer> tmp, id<MTLBuffer> dst, id<MTLBuffer> weights, id<MTLBuffer> params) {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.gaussian_plane_horizontal];
                [enc setBuffer:src offset:0 atIndex:0];
                [enc setBuffer:tmp offset:0 atIndex:1];
                [enc setBuffer:weights offset:0 atIndex:2];
                [enc setBuffer:params offset:0 atIndex:3];
                dispatch_2d(enc, pipelines.gaussian_plane_horizontal, width, height);
                [enc endEncoding];

                enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.gaussian_plane_vertical];
                [enc setBuffer:tmp offset:0 atIndex:0];
                [enc setBuffer:dst offset:0 atIndex:1];
                [enc setBuffer:weights offset:0 atIndex:2];
                [enc setBuffer:params offset:0 atIndex:3];
                dispatch_2d(enc, pipelines.gaussian_plane_vertical, width, height);
                [enc endEncoding];
            };

            blur_plane(defocus_raw, defocus_temp, defocus_blur, defocus_weights_buffer, defocus_params_buffer);

            auto blur_channel = [&](id<MTLBuffer> tmp, id<MTLBuffer> dst, id<MTLBuffer> channel_buffer) {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.gaussian_channel_horizontal];
                [enc setBuffer:input_buffer offset:0 atIndex:0];
                [enc setBuffer:tmp offset:0 atIndex:1];
                [enc setBuffer:fringe_weights_buffer offset:0 atIndex:2];
                [enc setBuffer:fringe_params_buffer offset:0 atIndex:3];
                [enc setBuffer:channel_buffer offset:0 atIndex:4];
                dispatch_2d(enc, pipelines.gaussian_channel_horizontal, width, height);
                [enc endEncoding];

                enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.gaussian_plane_to_plane_vertical];
                [enc setBuffer:tmp offset:0 atIndex:0];
                [enc setBuffer:dst offset:0 atIndex:1];
                [enc setBuffer:fringe_weights_buffer offset:0 atIndex:2];
                [enc setBuffer:fringe_params_buffer offset:0 atIndex:3];
                dispatch_2d(enc, pipelines.gaussian_plane_to_plane_vertical, width, height);
                [enc endEncoding];
            };

            blur_channel(r_temp, r_blur, channel_r_buffer);
            blur_channel(b_temp, b_blur, channel_b_buffer);

            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.longitudinal_compose];
                [enc setBuffer:input_buffer offset:0 atIndex:0];
                [enc setBuffer:defocus_blur offset:0 atIndex:1];
                [enc setBuffer:r_blur offset:0 atIndex:2];
                [enc setBuffer:b_blur offset:0 atIndex:3];
                [enc setBuffer:output_buffer offset:0 atIndex:4];
                [enc setBuffer:fringe_params_buffer offset:0 atIndex:5];
                NSUInteger tw = pipelines.longitudinal_compose.threadExecutionWidth;
                [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
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

py::array_t<float> apply_spherical_ca(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    py::array_t<float, py::array::c_style | py::array::forcecast> depth,
    bool has_depth,
    float strength,
    float aperture,
    float focus_depth,
    float highlight_threshold,
    float resolution_scale
) {
    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must have shape (height, width, 3)");
    }
    py::buffer_info dm = depth.request();
    if (dm.ndim != 2 || dm.shape[0] != in.shape[0] || dm.shape[1] != in.shape[1]) {
        throw std::invalid_argument("depth must have shape (height, width)");
    }
    const int height = static_cast<int>(in.shape[0]);
    const int width = static_cast<int>(in.shape[1]);
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument("image dimensions must be positive");
    }

    py::array_t<float> result({height, width, 3});
    py::buffer_info out = result.request();

    {
        py::gil_scoped_release release;
        @autoreleasepool {
            MetalPipelines& pipelines = metal_pipelines();
            const int count = width * height;
            const size_t plane_bytes = static_cast<size_t>(count) * sizeof(float);
            const size_t image_bytes = plane_bytes * 3;
            const float rs = std::max(0.05f, resolution_scale);
            const float aperture_factor = 2.8f / std::max(0.001f, aperture);
            const float total_blur_strength = strength * aperture_factor * 1.5f * rs;
            const float glow_strength = strength * 0.3f * aperture_factor;
            const float contrast_reduction = std::clamp(1.0f - strength * 0.1f * aperture_factor, 0.3f, 1.0f);

            id<MTLBuffer> input_buffer = [pipelines.device newBufferWithBytes:in.ptr length:image_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> depth_buffer = [pipelines.device newBufferWithBytes:dm.ptr length:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> highlight_raw = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> highlight_temp = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> highlight_blur = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> depth_weight = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> edge_raw = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> edge_temp = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> edge_blur = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> blur_sigma = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buffer = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];

            std::vector<float> highlight_weights = gaussian_weights(std::max(0.5f, 5.0f * rs));
            std::vector<float> edge_weights = gaussian_weights(std::max(0.5f, 10.0f * rs));
            LongitudinalParams highlight_params{width, height, static_cast<int>(highlight_weights.size() / 2), strength, focus_depth};
            LongitudinalParams edge_params{width, height, static_cast<int>(edge_weights.size() / 2), strength, focus_depth};
            SphericalParams params{
                width, height, 0, 0, has_depth ? 1 : 0,
                strength, aperture, focus_depth, highlight_threshold, rs,
                total_blur_strength, 0.0f, glow_strength, contrast_reduction, 0.0f,
            };

            id<MTLBuffer> params_buffer = [pipelines.device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
            id<MTLBuffer> highlight_params_buffer = [pipelines.device newBufferWithBytes:&highlight_params length:sizeof(highlight_params) options:MTLResourceStorageModeShared];
            id<MTLBuffer> edge_params_buffer = [pipelines.device newBufferWithBytes:&edge_params length:sizeof(edge_params) options:MTLResourceStorageModeShared];
            id<MTLBuffer> highlight_weights_buffer = [pipelines.device newBufferWithBytes:highlight_weights.data() length:highlight_weights.size() * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> edge_weights_buffer = [pipelines.device newBufferWithBytes:edge_weights.data() length:edge_weights.size() * sizeof(float) options:MTLResourceStorageModeShared];

            if (!input_buffer || !depth_buffer || !highlight_raw || !highlight_temp || !highlight_blur || !depth_weight || !edge_raw || !edge_temp || !edge_blur || !blur_sigma || !output_buffer || !params_buffer || !highlight_params_buffer || !edge_params_buffer || !highlight_weights_buffer || !edge_weights_buffer) {
                throw std::runtime_error("failed to allocate Metal spherical CA buffers");
            }

            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.spherical_prepare];
                [enc setBuffer:input_buffer offset:0 atIndex:0];
                [enc setBuffer:depth_buffer offset:0 atIndex:1];
                [enc setBuffer:highlight_raw offset:0 atIndex:2];
                [enc setBuffer:depth_weight offset:0 atIndex:3];
                [enc setBuffer:edge_raw offset:0 atIndex:4];
                [enc setBuffer:params_buffer offset:0 atIndex:5];
                dispatch_2d(enc, pipelines.spherical_prepare, width, height);
                [enc endEncoding];
            }

            auto blur_plane = [&](id<MTLBuffer> src, id<MTLBuffer> tmp, id<MTLBuffer> dst, id<MTLBuffer> weights, id<MTLBuffer> pbuf) {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.gaussian_plane_horizontal];
                [enc setBuffer:src offset:0 atIndex:0];
                [enc setBuffer:tmp offset:0 atIndex:1];
                [enc setBuffer:weights offset:0 atIndex:2];
                [enc setBuffer:pbuf offset:0 atIndex:3];
                dispatch_2d(enc, pipelines.gaussian_plane_horizontal, width, height);
                [enc endEncoding];

                enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.gaussian_plane_vertical];
                [enc setBuffer:tmp offset:0 atIndex:0];
                [enc setBuffer:dst offset:0 atIndex:1];
                [enc setBuffer:weights offset:0 atIndex:2];
                [enc setBuffer:pbuf offset:0 atIndex:3];
                dispatch_2d(enc, pipelines.gaussian_plane_vertical, width, height);
                [enc endEncoding];
            };

            blur_plane(highlight_raw, highlight_temp, highlight_blur, highlight_weights_buffer, highlight_params_buffer);
            blur_plane(edge_raw, edge_temp, edge_blur, edge_weights_buffer, edge_params_buffer);

            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.spherical_blur_sigma];
                [enc setBuffer:depth_weight offset:0 atIndex:0];
                [enc setBuffer:edge_blur offset:0 atIndex:1];
                [enc setBuffer:blur_sigma offset:0 atIndex:2];
                [enc setBuffer:params_buffer offset:0 atIndex:3];
                NSUInteger tw = pipelines.spherical_blur_sigma.threadExecutionWidth;
                [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
                [enc endEncoding];
            }
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.error) {
                std::string message = [[command_buffer.error localizedDescription] UTF8String];
                throw std::runtime_error(message);
            }

            const float* blur_sigma_ptr = static_cast<const float*>([blur_sigma contents]);
            double blur_sum = 0.0;
            for (int i = 0; i < count; ++i) {
                blur_sum += blur_sigma_ptr[i];
            }
            const float avg_blur_sigma = static_cast<float>(blur_sum / std::max(1, count));

            id<MTLBuffer> precontrast_buffer = input_buffer;
            id<MTLBuffer> composed_buffer = nil;
            std::vector<float> blur_weights;
            std::vector<float> glow_weights;
            id<MTLBuffer> blur_weights_buffer = nil;
            id<MTLBuffer> glow_weights_buffer = nil;

            if (avg_blur_sigma > 0.1f) {
                blur_weights = gaussian_weights(avg_blur_sigma);
                glow_weights = gaussian_weights(avg_blur_sigma * 2.0f);
                LongitudinalParams blur_params{width, height, static_cast<int>(blur_weights.size() / 2), strength, focus_depth};
                LongitudinalParams glow_params{width, height, static_cast<int>(glow_weights.size() / 2), strength, focus_depth};
                params.blur_radius = blur_params.radius;
                params.glow_radius = glow_params.radius;
                params.avg_blur_sigma = avg_blur_sigma;

                id<MTLBuffer> blur_params_buffer = [pipelines.device newBufferWithBytes:&blur_params length:sizeof(blur_params) options:MTLResourceStorageModeShared];
                id<MTLBuffer> glow_params_buffer = [pipelines.device newBufferWithBytes:&glow_params length:sizeof(glow_params) options:MTLResourceStorageModeShared];
                blur_weights_buffer = [pipelines.device newBufferWithBytes:blur_weights.data() length:blur_weights.size() * sizeof(float) options:MTLResourceStorageModeShared];
                glow_weights_buffer = [pipelines.device newBufferWithBytes:glow_weights.data() length:glow_weights.size() * sizeof(float) options:MTLResourceStorageModeShared];
                id<MTLBuffer> params2_buffer = [pipelines.device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
                id<MTLBuffer> blurred_temp = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> blurred_spatial = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> glow_src = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> glow_temp = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> glow_spatial = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
                composed_buffer = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
                if (!blur_params_buffer || !glow_params_buffer || !blur_weights_buffer || !glow_weights_buffer || !params2_buffer || !blurred_temp || !blurred_spatial || !glow_src || !glow_temp || !glow_spatial || !composed_buffer) {
                    throw std::runtime_error("failed to allocate Metal spherical CA blur buffers");
                }

                command_buffer = [pipelines.queue commandBuffer];
                auto blur_rgb = [&](id<MTLBuffer> src, id<MTLBuffer> tmp, id<MTLBuffer> dst, id<MTLBuffer> weights, id<MTLBuffer> pbuf) {
                    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                    [enc setComputePipelineState:pipelines.gaussian_rgb_horizontal];
                    [enc setBuffer:src offset:0 atIndex:0];
                    [enc setBuffer:tmp offset:0 atIndex:1];
                    [enc setBuffer:weights offset:0 atIndex:2];
                    [enc setBuffer:pbuf offset:0 atIndex:3];
                    dispatch_2d(enc, pipelines.gaussian_rgb_horizontal, width, height);
                    [enc endEncoding];

                    enc = [command_buffer computeCommandEncoder];
                    [enc setComputePipelineState:pipelines.gaussian_rgb_vertical];
                    [enc setBuffer:tmp offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    [enc setBuffer:weights offset:0 atIndex:2];
                    [enc setBuffer:pbuf offset:0 atIndex:3];
                    dispatch_2d(enc, pipelines.gaussian_rgb_vertical, width, height);
                    [enc endEncoding];
                };

                blur_rgb(input_buffer, blurred_temp, blurred_spatial, blur_weights_buffer, blur_params_buffer);
                {
                    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                    [enc setComputePipelineState:pipelines.spherical_glow_src];
                    [enc setBuffer:input_buffer offset:0 atIndex:0];
                    [enc setBuffer:highlight_blur offset:0 atIndex:1];
                    [enc setBuffer:glow_src offset:0 atIndex:2];
                    [enc setBuffer:params2_buffer offset:0 atIndex:3];
                    NSUInteger tw = pipelines.spherical_glow_src.threadExecutionWidth;
                    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
                    [enc endEncoding];
                }
                blur_rgb(glow_src, glow_temp, glow_spatial, glow_weights_buffer, glow_params_buffer);
                {
                    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                    [enc setComputePipelineState:pipelines.spherical_precontrast];
                    [enc setBuffer:input_buffer offset:0 atIndex:0];
                    [enc setBuffer:highlight_blur offset:0 atIndex:1];
                    [enc setBuffer:blur_sigma offset:0 atIndex:2];
                    [enc setBuffer:blurred_spatial offset:0 atIndex:3];
                    [enc setBuffer:glow_spatial offset:0 atIndex:4];
                    [enc setBuffer:blur_weights_buffer offset:0 atIndex:5];
                    [enc setBuffer:glow_weights_buffer offset:0 atIndex:6];
                    [enc setBuffer:composed_buffer offset:0 atIndex:7];
                    [enc setBuffer:params2_buffer offset:0 atIndex:8];
                    NSUInteger tw = pipelines.spherical_precontrast.threadExecutionWidth;
                    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
                    [enc endEncoding];
                }
                [command_buffer commit];
                [command_buffer waitUntilCompleted];
                if (command_buffer.error) {
                    std::string message = [[command_buffer.error localizedDescription] UTF8String];
                    throw std::runtime_error(message);
                }
                precontrast_buffer = composed_buffer;
            }

            const float* pre_ptr = static_cast<const float*>([precontrast_buffer contents]);
            double pivot_sum = 0.0;
            for (int i = 0; i < count * 3; ++i) {
                pivot_sum += pre_ptr[i];
            }
            params.pivot = static_cast<float>(pivot_sum / std::max(1, count * 3));
            params.avg_blur_sigma = avg_blur_sigma;
            id<MTLBuffer> contrast_params_buffer = [pipelines.device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
            if (!contrast_params_buffer) {
                throw std::runtime_error("failed to allocate Metal spherical CA contrast params");
            }

            command_buffer = [pipelines.queue commandBuffer];
            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setComputePipelineState:pipelines.spherical_contrast];
                [enc setBuffer:precontrast_buffer offset:0 atIndex:0];
                [enc setBuffer:output_buffer offset:0 atIndex:1];
                [enc setBuffer:contrast_params_buffer offset:0 atIndex:2];
                NSUInteger tw = pipelines.spherical_contrast.threadExecutionWidth;
                [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count * 3), 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
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

PYBIND11_MODULE(_lens_aberration_metal, m) {
    m.def("apply_lateral_ca", &apply_lateral_ca, py::arg("image"), py::arg("strength"), py::arg("resolution_scale"));
    m.def("apply_longitudinal_ca", &apply_longitudinal_ca, py::arg("image"), py::arg("depth"), py::arg("strength"), py::arg("focus_depth"), py::arg("resolution_scale"));
    m.def("apply_spherical_ca", &apply_spherical_ca, py::arg("image"), py::arg("depth"), py::arg("has_depth"), py::arg("strength"), py::arg("aperture"), py::arg("focus_depth"), py::arg("highlight_threshold"), py::arg("resolution_scale"));
    m.def("metal_available", &metal_available);
}
