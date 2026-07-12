// Metal HLS 色域別調整バックエンド。
// cores/core.py の _calculate_elliptical_weight / _apply_hls_adjust_map と
// bit-exact に近い形で一致することを test_hls_adjust_backend 系のテストで担保する。
// ガウシアンカーネルの重みは cv2.getGaussianKernel(ksize, 0) で生成した値を
// Python 側(hls_adjust_adapter)から渡してもらい、GPU 側では畳み込みのみ行う。
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

struct HlsAdjustMetalParams {
    int width;
    int height;
    int channels;
    int radius;
    float center;
    float width_left;
    float width_right;
    float fade_left;
    float fade_right;
    float l_min;
    float l_max;
    float s_min;
    float s_max;
    float adj_h;
    float adj_l;
    float adj_s;
};

constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct HlsAdjustMetalParams {
    int width;
    int height;
    int channels;
    int radius;
    float center;
    float width_left;
    float width_right;
    float fade_left;
    float fade_right;
    float l_min;
    float l_max;
    float s_min;
    float s_max;
    float adj_h;
    float adj_l;
    float adj_s;
};

static inline float hw_clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// cores/core.py の _smooth_step(x, 0.0, 1.0) と等価(edge0=0, edge1=1 固定)。
static inline float hw_smoothstep01(float x) {
    float t = hw_clamp(x, 0.0f, 1.0f);
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

// _calculate_elliptical_weight の完全移植。
kernel void hw_weight(
    const device float* hls_img [[buffer(0)]],
    device float* weight [[buffer(1)]],
    constant HlsAdjustMetalParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    int base = int(gid) * p.channels;
    float hue = hls_img[base];
    float l1 = hls_img[base + 1];
    float s = hls_img[base + 2];
    // 階調選択は実輝度(L×gain)で行う。L単体は正規化輝度で明暗を持たないため。
    float l = (p.channels > 3) ? (l1 * hls_img[base + 3]) : l1;

    // 1. Hue Excess Distance (Asymmetric)
    float signed_diff = hue - p.center;
    if (signed_diff > 180.0f) {
        signed_diff -= 360.0f;
    } else if (signed_diff < -180.0f) {
        signed_diff += 360.0f;
    }

    int side_idx = (signed_diff < 0.0f) ? 0 : 1;
    float abs_diff = fabs(signed_diff);
    float w_h = (side_idx == 0) ? p.width_left : p.width_right;
    float f_h = (side_idx == 0) ? p.fade_left : p.fade_right;

    float excess_h = 0.0f;
    if (abs_diff > w_h) {
        if (f_h > 1.0e-5f) {
            excess_h = (abs_diff - w_h) / f_h;
        } else {
            excess_h = 100.0f;  // Sharp cutoff
        }
    }

    // 2. L Excess Distance
    // l_max >= 1.0 は「上限なし」として扱う(HDR で実輝度が 1.0 を超えても
    // ハイライトが選択不能にならないようにするため)。
    const float fade_ls = 0.15f;
    float excess_l = 0.0f;
    if (l < p.l_min) {
        excess_l = (p.l_min - l) / fade_ls;
    } else if (p.l_max < 1.0f && l > p.l_max) {
        excess_l = (l - p.l_max) / fade_ls;
    }

    // 3. S Excess Distance
    float excess_s = 0.0f;
    if (s < p.s_min) {
        // STRICT FADE for Lower Bound (灰色/ノイズを除外するため fade_ls より鋭いフェード)
        const float strict_fade = 0.005f;
        excess_s = (p.s_min - s) / strict_fade;
    } else if (s > p.s_max) {
        excess_s = (s - p.s_max) / fade_ls;
    }

    // 4. Elliptical Combination (Euclidean Norm of Excess)
    float dist_sq = excess_h * excess_h + excess_l * excess_l + excess_s * excess_s;
    float dist = sqrt(dist_sq);

    // 5. Smooth Falloff
    weight[gid] = 1.0f - hw_smoothstep01(dist);
}

kernel void hw_gauss_h(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant HlsAdjustMetalParams& p [[buffer(3)]],
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

// 垂直ガウシアンの結果を出力せず、その場で total_adjust に融合する
// (lens_blur_metal.mm の lb_blur_v_accumulate と同じ発想)。
kernel void hw_gauss_v_accumulate(
    const device float* src [[buffer(0)]],
    const device float* weights [[buffer(1)]],
    device float* total_adjust [[buffer(2)]],
    constant HlsAdjustMetalParams& p [[buffer(3)]],
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
    int idx = y * p.width + x;
    int base = idx * 3;
    total_adjust[base] += sum * p.adj_h;
    total_adjust[base + 1] += sum * p.adj_l;
    total_adjust[base + 2] += sum * p.adj_s;
}

// _apply_hls_adjust_map の完全移植。
kernel void hw_apply(
    const device float* hls_img [[buffer(0)]],
    const device float* total_adjust [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant HlsAdjustMetalParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int count = p.width * p.height;
    if (int(gid) >= count) {
        return;
    }
    int base = int(gid) * p.channels;
    int abase = int(gid) * 3;

    float adj_h = total_adjust[abase];
    float adj_l = total_adjust[abase + 1];
    float adj_s = total_adjust[abase + 2];

    // --- 色相調整 ---
    // numba の `% 360.0` は常に非負を返す。fmod は符号を保持するため
    // floor ベースの剰余で再現する。
    float hue = hls_img[base];
    float new_h = hue + adj_h;
    new_h = new_h - 360.0f * floor(new_h / 360.0f);

    // --- 明度調整 (指数関数) ---
    float l_factor = exp2(adj_l * 2.0f);

    // --- 彩度調整 ---
    float sv = hls_img[base + 2];
    float new_s;
    if (adj_s > 0.0f) {
        // Vibrance Boost
        new_s = sv + sv * (1.0f - sv) * adj_s * 2.0f;
    } else {
        // Linear Desaturation
        new_s = sv * (1.0f + adj_s);
    }
    if (new_s < 0.0f) {
        new_s = 0.0f;
    }

    output[base] = new_h;
    output[base + 2] = new_s;

    if (p.channels > 3) {
        // gain(=実輝度)に明度調整を適用、L は保持
        output[base + 1] = hls_img[base + 1];
        output[base + 3] = hls_img[base + 3] * l_factor;
        for (int k = 4; k < p.channels; ++k) {
            output[base + k] = hls_img[base + k];
        }
    } else {
        // gain が無い(3ch)場合のフォールバック: 従来通り L を操作
        output[base + 1] = hls_img[base + 1] * l_factor;
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
    id<MTLComputePipelineState> weight;
    id<MTLComputePipelineState> gauss_h;
    id<MTLComputePipelineState> gauss_v_accumulate;
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
                state.weight = make_pipeline(state.device, state.library, @"hw_weight");
                state.gauss_h = make_pipeline(state.device, state.library, @"hw_gauss_h");
                state.gauss_v_accumulate = make_pipeline(state.device, state.library, @"hw_gauss_v_accumulate");
                state.apply = make_pipeline(state.device, state.library, @"hw_apply");
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

}  // namespace

bool metal_available() {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

py::array_t<float> apply_hls_adjust(
    py::array_t<float, py::array::c_style | py::array::forcecast> hls_img,
    py::array_t<float, py::array::c_style | py::array::forcecast> settings,
    py::array_t<float, py::array::c_style | py::array::forcecast> kernels,
    py::array_t<int, py::array::c_style | py::array::forcecast> kernel_offsets,
    py::array_t<int, py::array::c_style | py::array::forcecast> kernel_radii
) {
    py::buffer_info in = hls_img.request();
    if (in.ndim != 3 || in.shape[2] < 3) {
        throw std::invalid_argument("hls_img must be a (H, W, C>=3) float32 array");
    }
    py::buffer_info settings_info = settings.request();
    if (settings_info.ndim != 2 || settings_info.shape[1] != 12) {
        throw std::invalid_argument("settings must be a (N, 12) float32 array");
    }
    const int n_settings = static_cast<int>(settings_info.shape[0]);

    py::buffer_info offsets_info = kernel_offsets.request();
    py::buffer_info radii_info = kernel_radii.request();
    if (offsets_info.ndim != 1 || offsets_info.shape[0] != n_settings ||
        radii_info.ndim != 1 || radii_info.shape[0] != n_settings) {
        throw std::invalid_argument("kernel_offsets/kernel_radii must have shape (N,)");
    }

    const int height = static_cast<int>(in.shape[0]);
    const int width = static_cast<int>(in.shape[1]);
    const int channels = static_cast<int>(in.shape[2]);
    const int count = width * height;
    const size_t plane_bytes = static_cast<size_t>(count) * sizeof(float);
    const size_t image_bytes = static_cast<size_t>(count) * static_cast<size_t>(channels) * sizeof(float);
    const size_t adjust_bytes = plane_bytes * 3;

    std::vector<py::ssize_t> shape(in.shape.begin(), in.shape.end());
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    const float* settings_ptr = static_cast<const float*>(settings_info.ptr);
    const int* offsets_ptr = static_cast<const int*>(offsets_info.ptr);
    const int* radii_ptr = static_cast<const int*>(radii_info.ptr);
    py::buffer_info kernels_info = kernels.request();
    const size_t kernels_bytes = static_cast<size_t>(kernels_info.shape.empty() ? 0 : kernels_info.shape[0]) * sizeof(float);

    {
        py::gil_scoped_release release;
        @autoreleasepool {
            MetalPipelines& pipelines = metal_pipelines();
            BufferBinding input_binding = make_buffer_for_input(pipelines.device, in.ptr, image_bytes);
            BufferBinding output_binding = make_buffer_for_output(pipelines.device, out.ptr, image_bytes);
            id<MTLBuffer> input_buffer = input_binding.buffer;
            id<MTLBuffer> output_buffer = output_binding.buffer;
            if (!input_buffer || !output_buffer) {
                throw std::runtime_error("failed to allocate Metal HLS adjust image buffers");
            }

            BufferBinding kernels_binding{};
            id<MTLBuffer> kernels_buffer = nil;
            if (kernels_bytes > 0) {
                kernels_binding = make_buffer_for_input(pipelines.device, kernels_info.ptr, kernels_bytes);
                kernels_buffer = kernels_binding.buffer;
                if (!kernels_buffer) {
                    throw std::runtime_error("failed to allocate Metal HLS adjust kernel buffer");
                }
            }

            id<MTLBuffer> weight_plane = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> tmp_plane = [pipelines.device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> total_adjust = [pipelines.device newBufferWithLength:adjust_bytes options:MTLResourceStorageModeShared];
            if (!weight_plane || !tmp_plane || !total_adjust) {
                throw std::runtime_error("failed to allocate Metal HLS adjust working buffers");
            }
            std::memset([total_adjust contents], 0, adjust_bytes);

            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];

            // 各設定を weight -> horizontal blur -> vertical blur(累積) の順に処理する。
            std::vector<id<MTLBuffer>> per_setting_params_buffers;
            per_setting_params_buffers.reserve(static_cast<size_t>(n_settings) * 2);

            for (int i = 0; i < n_settings; ++i) {
                const float* row = settings_ptr + static_cast<size_t>(i) * 12;
                HlsAdjustMetalParams weight_params{
                    width, height, channels, 0,
                    row[0], row[1], row[2], row[3], row[4],
                    row[5], row[6], row[7], row[8],
                    0.0f, 0.0f, 0.0f,
                };
                id<MTLBuffer> weight_params_buffer = [pipelines.device newBufferWithBytes:&weight_params length:sizeof(weight_params) options:MTLResourceStorageModeShared];
                if (!weight_params_buffer) {
                    throw std::runtime_error("failed to allocate Metal HLS adjust weight params buffer");
                }
                per_setting_params_buffers.push_back(weight_params_buffer);

                {
                    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                    [enc setBuffer:input_buffer offset:input_binding.offset atIndex:0];
                    [enc setBuffer:weight_plane offset:0 atIndex:1];
                    [enc setBuffer:weight_params_buffer offset:0 atIndex:2];
                    dispatch_1d(enc, pipelines.weight, static_cast<NSUInteger>(count));
                    [enc endEncoding];
                }

                const int radius = radii_ptr[i];
                const int kernel_offset_floats = offsets_ptr[i];
                const NSUInteger kernel_byte_offset = kernels_binding.offset + static_cast<NSUInteger>(kernel_offset_floats) * sizeof(float);

                HlsAdjustMetalParams blur_params{
                    width, height, channels, radius,
                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    row[9], row[10], row[11],
                };
                id<MTLBuffer> blur_params_buffer = [pipelines.device newBufferWithBytes:&blur_params length:sizeof(blur_params) options:MTLResourceStorageModeShared];
                if (!blur_params_buffer) {
                    throw std::runtime_error("failed to allocate Metal HLS adjust blur params buffer");
                }
                per_setting_params_buffers.push_back(blur_params_buffer);

                {
                    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                    [enc setBuffer:weight_plane offset:0 atIndex:0];
                    [enc setBuffer:tmp_plane offset:0 atIndex:1];
                    [enc setBuffer:kernels_buffer offset:kernel_byte_offset atIndex:2];
                    [enc setBuffer:blur_params_buffer offset:0 atIndex:3];
                    dispatch_2d(enc, pipelines.gauss_h, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height));
                    [enc endEncoding];
                }
                {
                    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                    [enc setBuffer:tmp_plane offset:0 atIndex:0];
                    [enc setBuffer:kernels_buffer offset:kernel_byte_offset atIndex:1];
                    [enc setBuffer:total_adjust offset:0 atIndex:2];
                    [enc setBuffer:blur_params_buffer offset:0 atIndex:3];
                    dispatch_2d(enc, pipelines.gauss_v_accumulate, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height));
                    [enc endEncoding];
                }
            }

            HlsAdjustMetalParams apply_params{
                width, height, channels, 0,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f,
            };
            id<MTLBuffer> apply_params_buffer = [pipelines.device newBufferWithBytes:&apply_params length:sizeof(apply_params) options:MTLResourceStorageModeShared];
            if (!apply_params_buffer) {
                throw std::runtime_error("failed to allocate Metal HLS adjust apply params buffer");
            }

            {
                id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                [enc setBuffer:input_buffer offset:input_binding.offset atIndex:0];
                [enc setBuffer:total_adjust offset:0 atIndex:1];
                [enc setBuffer:output_buffer offset:output_binding.offset atIndex:2];
                [enc setBuffer:apply_params_buffer offset:0 atIndex:3];
                dispatch_1d(enc, pipelines.apply, static_cast<NSUInteger>(count));
                [enc endEncoding];
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

PYBIND11_MODULE(_hls_adjust_metal, m) {
    m.doc() = "Metal HLS per-color adjust backend for Platypus";
    m.def("metal_available", &metal_available);
    m.def(
        "apply_hls_adjust",
        &apply_hls_adjust,
        py::arg("hls_img"),
        py::arg("settings"),
        py::arg("kernels"),
        py::arg("kernel_offsets"),
        py::arg("kernel_radii")
    );
}
