#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_buffer_utils.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct LocalContrastParams {
    int width;
    int height;
    int radius;
    float strength;
    float eps;
    float factor;
};

constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct LocalContrastParams {
    int width;
    int height;
    int radius;
    float strength;
    float eps;
    float factor;
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

kernel void rgb_to_gray601(
    const device float* input [[buffer(0)]],
    device float* gray [[buffer(1)]],
    constant LocalContrastParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int base = (y * p.width + x) * 3;
    gray[y * p.width + x] = input[base] * 0.299f + input[base + 1] * 0.587f + input[base + 2] * 0.114f;
}

kernel void rgb_to_y709(
    const device float* input [[buffer(0)]],
    device float* y_plane [[buffer(1)]],
    constant LocalContrastParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int base = (y * p.width + x) * 3;
    y_plane[y * p.width + x] = input[base] * 0.2126f + input[base + 1] * 0.7152f + input[base + 2] * 0.0722f;
}

kernel void square_plane(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant LocalContrastParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    int idx = y * p.width + x;
    output[idx] = input[idx] * input[idx];
}


kernel void compute_guided_ab(
    const device float* mean [[buffer(0)]],
    const device float* mean_sq [[buffer(1)]],
    device float* a [[buffer(2)]],
    device float* b [[buffer(3)]],
    constant LocalContrastParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    float m = mean[gid];
    float var = max(mean_sq[gid] - m * m, 0.0f);
    float av = var / (var + p.eps);
    a[gid] = av;
    b[gid] = m - av * m;
}

kernel void detail_from_guided(
    const device float* source [[buffer(0)]],
    const device float* mean_a [[buffer(1)]],
    const device float* mean_b [[buffer(2)]],
    device float* detail [[buffer(3)]],
    constant LocalContrastParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    float base = mean_a[gid] * source[gid] + mean_b[gid];
    detail[gid] = source[gid] - base;
}

kernel void add_detail_from_guided(
    const device float* source [[buffer(0)]],
    const device float* mean_a [[buffer(1)]],
    const device float* mean_b [[buffer(2)]],
    device float* detail [[buffer(3)]],
    constant LocalContrastParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    float base = mean_a[gid] * source[gid] + mean_b[gid];
    detail[gid] += source[gid] - base;
}

kernel void compose_delta_rgb(
    const device float* input [[buffer(0)]],
    const device float* detail [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LocalContrastParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    float delta = detail[gid] * p.strength * p.factor;
    int base = int(gid) * 3;
    output[base] = input[base] + delta;
    output[base + 1] = input[base + 1] + delta;
    output[base + 2] = input[base + 2] + delta;
}

kernel void gaussian_horizontal(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant LocalContrastParams& p [[buffer(3)]],
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

kernel void gaussian_vertical(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant LocalContrastParams& p [[buffer(3)]],
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

kernel void compose_texture(
    const device float* input [[buffer(0)]],
    const device float* blur_small [[buffer(1)]],
    const device float* blur_large [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant LocalContrastParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    float delta = (blur_small[gid] - blur_large[gid]) * p.strength * p.factor;
    int base = int(gid) * 3;
    output[base] = input[base] + delta;
    output[base + 1] = input[base + 1] + delta;
    output[base + 2] = input[base + 2] + delta;
}
)METAL";

// box フィルタはスライディングウィンドウ(1スレッド=1行/1列)で O(1)/px にする。
// Kahan 補正付き移動和は加減算の再結合で壊れるため、このライブラリだけ
// fast math を無効にしてコンパイルする。
constexpr const char* kBoxMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct LocalContrastParams {
    int width;
    int height;
    int radius;
    float strength;
    float eps;
    float factor;
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

kernel void box_horizontal(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant LocalContrastParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    int y = int(gid);
    if (y >= p.height) {
        return;
    }
    const device float* row = input + y * p.width;
    device float* out_row = output + y * p.width;
    const int r = p.radius;
    const int w = p.width;
    const float norm = float(r * 2 + 1);
    float sum = 0.0f;
    float comp = 0.0f;
    for (int k = -r; k <= r; ++k) {
        float v = row[reflect101(k, w)];
        float t = v - comp;
        float s = sum + t;
        comp = (s - sum) - t;
        sum = s;
    }
    out_row[0] = sum / norm;
    for (int x = 1; x < w; ++x) {
        float v = row[reflect101(x + r, w)] - row[reflect101(x - 1 - r, w)];
        float t = v - comp;
        float s = sum + t;
        comp = (s - sum) - t;
        sum = s;
        out_row[x] = sum / norm;
    }
}

kernel void box_vertical(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant LocalContrastParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    int x = int(gid);
    if (x >= p.width) {
        return;
    }
    const int r = p.radius;
    const int h = p.height;
    const int w = p.width;
    const float norm = float(r * 2 + 1);
    float sum = 0.0f;
    float comp = 0.0f;
    for (int k = -r; k <= r; ++k) {
        float v = input[reflect101(k, h) * w + x];
        float t = v - comp;
        float s = sum + t;
        comp = (s - sum) - t;
        sum = s;
    }
    output[x] = sum / norm;
    for (int y = 1; y < h; ++y) {
        float v = input[reflect101(y + r, h) * w + x] - input[reflect101(y - 1 - r, h) * w + x];
        float t = v - comp;
        float s = sum + t;
        comp = (s - sum) - t;
        sum = s;
        output[y * w + x] = sum / norm;
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
    id<MTLComputePipelineState> rgb_to_gray601;
    id<MTLComputePipelineState> rgb_to_y709;
    id<MTLComputePipelineState> square_plane;
    id<MTLComputePipelineState> box_horizontal;
    id<MTLComputePipelineState> box_vertical;
    id<MTLComputePipelineState> compute_guided_ab;
    id<MTLComputePipelineState> detail_from_guided;
    id<MTLComputePipelineState> add_detail_from_guided;
    id<MTLComputePipelineState> compose_delta_rgb;
    id<MTLComputePipelineState> gaussian_horizontal;
    id<MTLComputePipelineState> gaussian_vertical;
    id<MTLComputePipelineState> compose_texture;
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
            NSString* box_source = [NSString stringWithUTF8String:kBoxMetalSource];
            MTLCompileOptions* box_options = [MTLCompileOptions new];
            box_options.fastMathEnabled = NO;
            id<MTLLibrary> box_library = [state.device newLibraryWithSource:box_source options:box_options error:&error];
            if (!box_library) {
                init_error = error ? [[error localizedDescription] UTF8String] : "unknown Metal box library error";
                return;
            }
            state.queue = [state.device newCommandQueue];
            if (!state.queue) {
                init_error = "Metal command queue is unavailable";
                return;
            }
            try {
                state.rgb_to_gray601 = make_pipeline(state.device, state.library, @"rgb_to_gray601");
                state.rgb_to_y709 = make_pipeline(state.device, state.library, @"rgb_to_y709");
                state.square_plane = make_pipeline(state.device, state.library, @"square_plane");
                state.box_horizontal = make_pipeline(state.device, box_library, @"box_horizontal");
                state.box_vertical = make_pipeline(state.device, box_library, @"box_vertical");
                state.compute_guided_ab = make_pipeline(state.device, state.library, @"compute_guided_ab");
                state.detail_from_guided = make_pipeline(state.device, state.library, @"detail_from_guided");
                state.add_detail_from_guided = make_pipeline(state.device, state.library, @"add_detail_from_guided");
                state.compose_delta_rgb = make_pipeline(state.device, state.library, @"compose_delta_rgb");
                state.gaussian_horizontal = make_pipeline(state.device, state.library, @"gaussian_horizontal");
                state.gaussian_vertical = make_pipeline(state.device, state.library, @"gaussian_vertical");
                state.compose_texture = make_pipeline(state.device, state.library, @"compose_texture");
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

void dispatch_2d(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, NSUInteger width, NSUInteger height) {
    [encoder setComputePipelineState:pipeline];
    NSUInteger tw = std::max<NSUInteger>(1, pipeline.threadExecutionWidth);
    NSUInteger th = std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / tw);
    if (th > 16) {
        th = 16;
    }
    [encoder dispatchThreads:MTLSizeMake(width, height, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

void dispatch_1d(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, NSUInteger count) {
    [encoder setComputePipelineState:pipeline];
    NSUInteger tpg = std::max<NSUInteger>(1, pipeline.threadExecutionWidth);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

std::vector<float> gaussian_kernel(float sigma) {
    if (sigma <= 0.0f) {
        return {1.0f};
    }
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

struct ImageBinding {
    py::array_t<float> result;
    py::buffer_info in;
    py::buffer_info out;
    int width;
    int height;
    int count;
    size_t image_bytes;
    size_t plane_bytes;
};

ImageBinding prepare_image(py::array_t<float, py::array::c_style | py::array::forcecast> image) {
    ImageBinding b{};
    b.in = image.request();
    if (b.in.ndim != 3 || b.in.shape[2] != 3) {
        throw std::invalid_argument("image must have shape (H, W, 3)");
    }
    b.width = static_cast<int>(b.in.shape[1]);
    b.height = static_cast<int>(b.in.shape[0]);
    b.count = b.width * b.height;
    b.image_bytes = static_cast<size_t>(b.count) * 3 * sizeof(float);
    b.plane_bytes = static_cast<size_t>(b.count) * sizeof(float);
    b.result = py::array_t<float>({b.height, b.width, 3});
    b.out = b.result.request();
    return b;
}

id<MTLBuffer> make_params(id<MTLDevice> device, const LocalContrastParams& params) {
    return [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
}

void encode_plane_kernel(
    id<MTLCommandBuffer> command_buffer,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> in0,
    id<MTLBuffer> out0,
    id<MTLBuffer> params,
    int width,
    int height,
    NSUInteger in0_offset = 0
) {
    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
    [enc setBuffer:in0 offset:in0_offset atIndex:0];
    [enc setBuffer:out0 offset:0 atIndex:1];
    [enc setBuffer:params offset:0 atIndex:2];
    dispatch_2d(enc, pipeline, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height));
    [enc endEncoding];
}

void encode_box(
    id<MTLCommandBuffer> command_buffer,
    MetalPipelines& pipelines,
    id<MTLBuffer> src,
    id<MTLBuffer> temp,
    id<MTLBuffer> dst,
    id<MTLBuffer> params,
    int width,
    int height
) {
    // スライディングウィンドウ版: horizontal は1スレッド=1行、vertical は1列。
    {
        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:temp offset:0 atIndex:1];
        [enc setBuffer:params offset:0 atIndex:2];
        dispatch_1d(enc, pipelines.box_horizontal, static_cast<NSUInteger>(height));
        [enc endEncoding];
    }
    {
        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
        [enc setBuffer:temp offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        [enc setBuffer:params offset:0 atIndex:2];
        dispatch_1d(enc, pipelines.box_vertical, static_cast<NSUInteger>(width));
        [enc endEncoding];
    }
}

void encode_guided_detail(
    id<MTLCommandBuffer> command_buffer,
    MetalPipelines& pipelines,
    id<MTLDevice> device,
    id<MTLBuffer> source,
    id<MTLBuffer> source_sq,
    id<MTLBuffer> detail,
    bool add_detail,
    LocalContrastParams params,
    int width,
    int height,
    size_t plane_bytes
) {
    id<MTLBuffer> params_buffer = make_params(device, params);
    id<MTLBuffer> mean = [device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> mean_sq = [device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> temp1 = [device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> temp2 = [device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> a = [device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> mean_a = [device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> mean_b = [device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
    if (!params_buffer || !mean || !mean_sq || !temp1 || !temp2 || !a || !b || !mean_a || !mean_b) {
        throw std::runtime_error("failed to allocate Metal guided-filter buffers");
    }

    encode_box(command_buffer, pipelines, source, temp1, mean, params_buffer, width, height);
    encode_box(command_buffer, pipelines, source_sq, temp1, mean_sq, params_buffer, width, height);

    {
        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
        [enc setBuffer:mean offset:0 atIndex:0];
        [enc setBuffer:mean_sq offset:0 atIndex:1];
        [enc setBuffer:a offset:0 atIndex:2];
        [enc setBuffer:b offset:0 atIndex:3];
        [enc setBuffer:params_buffer offset:0 atIndex:4];
        dispatch_1d(enc, pipelines.compute_guided_ab, static_cast<NSUInteger>(width * height));
        [enc endEncoding];
    }

    encode_box(command_buffer, pipelines, a, temp1, mean_a, params_buffer, width, height);
    encode_box(command_buffer, pipelines, b, temp2, mean_b, params_buffer, width, height);

    {
        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
        [enc setBuffer:source offset:0 atIndex:0];
        [enc setBuffer:mean_a offset:0 atIndex:1];
        [enc setBuffer:mean_b offset:0 atIndex:2];
        [enc setBuffer:detail offset:0 atIndex:3];
        [enc setBuffer:params_buffer offset:0 atIndex:4];
        dispatch_1d(enc, add_detail ? pipelines.add_detail_from_guided : pipelines.detail_from_guided, static_cast<NSUInteger>(width * height));
        [enc endEncoding];
    }
}

py::array_t<float> apply_guided_delta(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    float strength,
    bool use_y709,
    const std::vector<int>& radii,
    const std::vector<float>& eps_values,
    float factor
) {
    ImageBinding img = prepare_image(image);
    @autoreleasepool {
        MetalPipelines& pipelines = metal_pipelines();
        BufferBinding input_binding = make_buffer_for_input(pipelines.device, img.in.ptr, img.image_bytes);
        BufferBinding output_binding = make_buffer_for_output(pipelines.device, img.out.ptr, img.image_bytes);
        id<MTLBuffer> input = input_binding.buffer;
        id<MTLBuffer> output = output_binding.buffer;
        id<MTLBuffer> source = [pipelines.device newBufferWithLength:img.plane_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> source_sq = [pipelines.device newBufferWithLength:img.plane_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> detail = [pipelines.device newBufferWithLength:img.plane_bytes options:MTLResourceStorageModeShared];
        if (!input || !output || !source || !source_sq || !detail) {
            throw std::runtime_error("failed to allocate Metal local contrast buffers");
        }

        LocalContrastParams params{img.width, img.height, 1, strength, 0.0f, factor};
        id<MTLBuffer> params_buffer = make_params(pipelines.device, params);
        id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
        encode_plane_kernel(command_buffer, use_y709 ? pipelines.rgb_to_y709 : pipelines.rgb_to_gray601, input, source, params_buffer, img.width, img.height, input_binding.offset);
        encode_plane_kernel(command_buffer, pipelines.square_plane, source, source_sq, params_buffer, img.width, img.height);

        for (size_t i = 0; i < radii.size(); ++i) {
            LocalContrastParams guided_params{img.width, img.height, radii[i], strength, eps_values[i], factor};
            encode_guided_detail(
                command_buffer,
                pipelines,
                pipelines.device,
                source,
                source_sq,
                detail,
                i > 0,
                guided_params,
                img.width,
                img.height,
                img.plane_bytes
            );
        }

        LocalContrastParams compose_params{
            img.width,
            img.height,
            1,
            strength,
            0.0f,
            factor / static_cast<float>(std::max<size_t>(1, radii.size())),
        };
        id<MTLBuffer> compose_params_buffer = make_params(pipelines.device, compose_params);
        {
            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            [enc setBuffer:input offset:input_binding.offset atIndex:0];
            [enc setBuffer:detail offset:0 atIndex:1];
            [enc setBuffer:output offset:output_binding.offset atIndex:2];
            [enc setBuffer:compose_params_buffer offset:0 atIndex:3];
            dispatch_1d(enc, pipelines.compose_delta_rgb, static_cast<NSUInteger>(img.count));
            [enc endEncoding];
        }
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        finish_output_binding(output_binding, img.out.ptr, img.image_bytes);
    }
    return img.result;
}

void encode_gaussian(
    id<MTLCommandBuffer> command_buffer,
    MetalPipelines& pipelines,
    id<MTLBuffer> source,
    id<MTLBuffer> temp,
    id<MTLBuffer> dst,
    id<MTLBuffer> kernel,
    id<MTLBuffer> params,
    int width,
    int height
) {
    {
        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
        [enc setBuffer:source offset:0 atIndex:0];
        [enc setBuffer:temp offset:0 atIndex:1];
        [enc setBuffer:kernel offset:0 atIndex:2];
        [enc setBuffer:params offset:0 atIndex:3];
        dispatch_2d(enc, pipelines.gaussian_horizontal, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height));
        [enc endEncoding];
    }
    {
        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
        [enc setBuffer:temp offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        [enc setBuffer:kernel offset:0 atIndex:2];
        [enc setBuffer:params offset:0 atIndex:3];
        dispatch_2d(enc, pipelines.gaussian_vertical, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height));
        [enc endEncoding];
    }
}

}  // namespace

bool metal_available() {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

py::array_t<float> apply_clarity(py::array_t<float, py::array::c_style | py::array::forcecast> image, float strength) {
    py::buffer_info in = image.request();
    int width = static_cast<int>(in.shape[1]);
    int height = static_cast<int>(in.shape[0]);
    int radius = std::max(8, static_cast<int>(std::max(width, height) * 0.02f));
    return apply_guided_delta(image, strength, false, {radius}, {0.005f}, 1.0f);
}

py::array_t<float> apply_microcontrast(py::array_t<float, py::array::c_style | py::array::forcecast> image, float strength) {
    py::buffer_info in = image.request();
    int width = static_cast<int>(in.shape[1]);
    int height = static_cast<int>(in.shape[0]);
    float radius_scale = static_cast<float>(std::max(width, height)) / 4096.0f;
    int r1 = std::max(2, static_cast<int>(std::round(8.0f * radius_scale)));
    int r2 = std::max(3, static_cast<int>(std::round(20.0f * radius_scale)));
    r1 |= 1;
    r2 |= 1;
    return apply_guided_delta(image, strength, true, {r1, r2}, {0.01f, 0.02f}, 1.4f);
}

py::array_t<float> apply_texture(py::array_t<float, py::array::c_style | py::array::forcecast> image, float strength) {
    ImageBinding img = prepare_image(image);
    float radius_scale = static_cast<float>(std::max(img.width, img.height)) / 4096.0f;
    float sigma_small = std::max(0.3f, radius_scale);
    float sigma_large = std::max(0.6f, 4.0f * radius_scale);
    std::vector<float> kernel_small = gaussian_kernel(sigma_small);
    std::vector<float> kernel_large = gaussian_kernel(sigma_large);

    @autoreleasepool {
        MetalPipelines& pipelines = metal_pipelines();
        BufferBinding input_binding = make_buffer_for_input(pipelines.device, img.in.ptr, img.image_bytes);
        BufferBinding output_binding = make_buffer_for_output(pipelines.device, img.out.ptr, img.image_bytes);
        id<MTLBuffer> input = input_binding.buffer;
        id<MTLBuffer> output = output_binding.buffer;
        id<MTLBuffer> luma = [pipelines.device newBufferWithLength:img.plane_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> temp = [pipelines.device newBufferWithLength:img.plane_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> blur_small = [pipelines.device newBufferWithLength:img.plane_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> blur_large = [pipelines.device newBufferWithLength:img.plane_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> small_weights = [pipelines.device newBufferWithBytes:kernel_small.data() length:kernel_small.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> large_weights = [pipelines.device newBufferWithBytes:kernel_large.data() length:kernel_large.size() * sizeof(float) options:MTLResourceStorageModeShared];
        if (!input || !output || !luma || !temp || !blur_small || !blur_large || !small_weights || !large_weights) {
            throw std::runtime_error("failed to allocate Metal texture buffers");
        }

        LocalContrastParams params{img.width, img.height, 1, strength, 0.0f, 1.5f};
        id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
        id<MTLBuffer> luma_params = make_params(pipelines.device, params);
        encode_plane_kernel(command_buffer, pipelines.rgb_to_gray601, input, luma, luma_params, img.width, img.height, input_binding.offset);

        LocalContrastParams small_params{img.width, img.height, static_cast<int>(kernel_small.size() / 2), strength, 0.0f, 1.5f};
        LocalContrastParams large_params{img.width, img.height, static_cast<int>(kernel_large.size() / 2), strength, 0.0f, 1.5f};
        id<MTLBuffer> small_params_buffer = make_params(pipelines.device, small_params);
        id<MTLBuffer> large_params_buffer = make_params(pipelines.device, large_params);
        encode_gaussian(command_buffer, pipelines, luma, temp, blur_small, small_weights, small_params_buffer, img.width, img.height);
        encode_gaussian(command_buffer, pipelines, luma, temp, blur_large, large_weights, large_params_buffer, img.width, img.height);

        id<MTLBuffer> compose_params = make_params(pipelines.device, params);
        {
            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            [enc setBuffer:input offset:input_binding.offset atIndex:0];
            [enc setBuffer:blur_small offset:0 atIndex:1];
            [enc setBuffer:blur_large offset:0 atIndex:2];
            [enc setBuffer:output offset:output_binding.offset atIndex:3];
            [enc setBuffer:compose_params offset:0 atIndex:4];
            dispatch_1d(enc, pipelines.compose_texture, static_cast<NSUInteger>(img.count));
            [enc endEncoding];
        }
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        finish_output_binding(output_binding, img.out.ptr, img.image_bytes);
    }
    return img.result;
}

PYBIND11_MODULE(_local_contrast_metal, m) {
    m.def("metal_available", &metal_available);
    m.def("apply_clarity", &apply_clarity);
    m.def("apply_texture", &apply_texture);
    m.def("apply_microcontrast", &apply_microcontrast);
}
