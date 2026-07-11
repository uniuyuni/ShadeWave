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

struct LensBlurParams {
    int width;
    int height;
    int level;              // accumulate 対象のレベル番号
    int radius;             // ブラーパスのカーネル半径
    float chromatic;        // 色収差(coc_scale = {1+c, 1, 1-c})
    float inv_max_coc_lm1;  // (num_levels - 1) / max_coc_radius
};

// レンズブラー本体。
//  - accumulate: 現在のブラー平面 cur を、レベル別三角重みで acc/wsum に加算。
//  - blur_h / blur_v: 分離ガウシアン(reflect101 境界)を RGB 3ch まとめて適用。
//  - finalize: acc / wsum で正規化して出力。
// ガウシアンは合成可能なので、cur を逐次インクリメンタルにぼかしながら
// レベルごとに acc へ畳み込むことで、スタックをメモリ O(H*W) のまま構築する。
constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct LensBlurParams {
    int width;
    int height;
    int level;
    int radius;
    float chromatic;
    float inv_max_coc_lm1;
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

kernel void lb_blur_h(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant LensBlurParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    float3 sum = float3(0.0);
    for (int k = -p.radius; k <= p.radius; ++k) {
        int sx = reflect101(x + k, p.width);
        int base = (y * p.width + sx) * 3;
        float w = weights[k + p.radius];
        sum += float3(input[base], input[base + 1], input[base + 2]) * w;
    }
    int obase = (y * p.width + x) * 3;
    output[obase] = sum.x;
    output[obase + 1] = sum.y;
    output[obase + 2] = sum.z;
}

kernel void lb_blur_v(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant LensBlurParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.width || y >= p.height) {
        return;
    }
    float3 sum = float3(0.0);
    for (int k = -p.radius; k <= p.radius; ++k) {
        int sy = reflect101(y + k, p.height);
        int base = (sy * p.width + x) * 3;
        float w = weights[k + p.radius];
        sum += float3(input[base], input[base + 1], input[base + 2]) * w;
    }
    int obase = (y * p.width + x) * 3;
    output[obase] = sum.x;
    output[obase + 1] = sum.y;
    output[obase + 2] = sum.z;
}

kernel void lb_accumulate(
    const device float* cur [[buffer(0)]],
    const device float* coc [[buffer(1)]],
    device float* acc [[buffer(2)]],
    device float* wsum [[buffer(3)]],
    constant LensBlurParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    int count = p.width * p.height;
    if (idx >= count) {
        return;
    }
    float c = coc[idx];
    float3 scale = float3(1.0 + p.chromatic, 1.0, 1.0 - p.chromatic);
    float lvl = float(p.level);
    int base = idx * 3;
    for (int ch = 0; ch < 3; ++ch) {
        float blf = c * scale[ch] * p.inv_max_coc_lm1;
        float w = max(0.0f, 1.0f - fabs(blf - lvl));
        acc[base + ch] += cur[base + ch] * w;
        wsum[base + ch] += w;
    }
}

kernel void lb_finalize(
    const device float* acc [[buffer(0)]],
    const device float* wsum [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LensBlurParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    int count = p.width * p.height;
    if (idx >= count) {
        return;
    }
    int base = idx * 3;
    for (int ch = 0; ch < 3; ++ch) {
        float w = max(wsum[base + ch], 1.0e-6f);
        output[base + ch] = acc[base + ch] / w;
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
    id<MTLComputePipelineState> blur_h;
    id<MTLComputePipelineState> blur_v;
    id<MTLComputePipelineState> accumulate;
    id<MTLComputePipelineState> finalize;
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
                state.blur_h = make_pipeline(state.device, state.library, @"lb_blur_h");
                state.blur_v = make_pipeline(state.device, state.library, @"lb_blur_v");
                state.accumulate = make_pipeline(state.device, state.library, @"lb_accumulate");
                state.finalize = make_pipeline(state.device, state.library, @"lb_finalize");
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

// cv2.GaussianBlur(32F, ksize=0) 相当のカーネルサイズ(radius ~ 3*sigma)。
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

}  // namespace

bool metal_available() {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

py::array_t<float> apply_lensblur(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    py::array_t<float, py::array::c_style | py::array::forcecast> coc_radius,
    int num_levels,
    float max_coc_radius,
    float chromatic_aberration,
    float spherical_aberration
) {
    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must have shape (H, W, 3)");
    }
    py::buffer_info coc = coc_radius.request();
    const int height = static_cast<int>(in.shape[0]);
    const int width = static_cast<int>(in.shape[1]);
    if (coc.ndim != 2 || coc.shape[0] != height || coc.shape[1] != width) {
        throw std::invalid_argument("coc_radius must have shape (H, W) matching image");
    }
    if (num_levels < 1) {
        num_levels = 1;
    }
    if (max_coc_radius <= 0.0f) {
        max_coc_radius = 1.0f;
    }

    const int count = width * height;
    const size_t image_bytes = static_cast<size_t>(count) * 3 * sizeof(float);
    const size_t plane_bytes = static_cast<size_t>(count) * sizeof(float);

    py::array_t<float> result({height, width, 3});
    py::buffer_info out = result.request();

    LensBlurParams base{};
    base.width = width;
    base.height = height;
    base.chromatic = chromatic_aberration;
    base.inv_max_coc_lm1 = static_cast<float>(num_levels - 1) / max_coc_radius;

    @autoreleasepool {
        MetalPipelines& pipelines = metal_pipelines();

        id<MTLBuffer> cur = [pipelines.device newBufferWithBytes:in.ptr length:image_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> tmp = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> acc = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> wsum = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> coc_buf = [pipelines.device newBufferWithBytes:coc.ptr length:plane_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> output = [pipelines.device newBufferWithLength:image_bytes options:MTLResourceStorageModeShared];
        if (!cur || !tmp || !acc || !wsum || !coc_buf || !output) {
            throw std::runtime_error("failed to allocate Metal lens blur buffers");
        }
        std::memset([acc contents], 0, image_bytes);
        std::memset([wsum contents], 0, image_bytes);

        // レベルの持ち物(パラメータ / 重み)を command buffer 完了まで生かすため保持。
        std::vector<id<MTLBuffer>> retained;

        auto make_params = [&](int level, int radius) -> id<MTLBuffer> {
            LensBlurParams p = base;
            p.level = level;
            p.radius = radius;
            id<MTLBuffer> buf = [pipelines.device newBufferWithBytes:&p length:sizeof(p) options:MTLResourceStorageModeShared];
            retained.push_back(buf);
            return buf;
        };

        id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];

        auto encode_accumulate = [&](int level) {
            id<MTLBuffer> params = make_params(level, 0);
            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            [enc setBuffer:cur offset:0 atIndex:0];
            [enc setBuffer:coc_buf offset:0 atIndex:1];
            [enc setBuffer:acc offset:0 atIndex:2];
            [enc setBuffer:wsum offset:0 atIndex:3];
            [enc setBuffer:params offset:0 atIndex:4];
            dispatch_1d(enc, pipelines.accumulate, static_cast<NSUInteger>(count));
            [enc endEncoding];
        };

        // level 0: cur は原画像そのまま(sigma≈0)。
        encode_accumulate(0);

        double sigma_cur = 0.0;
        for (int level = 1; level < num_levels; ++level) {
            double sigma = static_cast<double>(level) * static_cast<double>(max_coc_radius)
                           / static_cast<double>(num_levels - 1) / 2.0;
            if (spherical_aberration > 0.0f && sigma > 1.0) {
                sigma *= (1.0 + static_cast<double>(spherical_aberration) * 0.2);
            }

            if (sigma >= 0.1) {
                double delta = std::sqrt(std::max(0.0, sigma * sigma - sigma_cur * sigma_cur));
                if (delta > 1.0e-4) {
                    std::vector<float> kernel = gaussian_kernel(static_cast<float>(delta));
                    int radius = static_cast<int>(kernel.size() / 2);
                    id<MTLBuffer> weights = [pipelines.device newBufferWithBytes:kernel.data()
                                                                          length:kernel.size() * sizeof(float)
                                                                         options:MTLResourceStorageModeShared];
                    retained.push_back(weights);
                    id<MTLBuffer> params = make_params(level, radius);

                    {
                        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                        [enc setBuffer:cur offset:0 atIndex:0];
                        [enc setBuffer:tmp offset:0 atIndex:1];
                        [enc setBuffer:weights offset:0 atIndex:2];
                        [enc setBuffer:params offset:0 atIndex:3];
                        dispatch_2d(enc, pipelines.blur_h, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height));
                        [enc endEncoding];
                    }
                    {
                        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
                        [enc setBuffer:tmp offset:0 atIndex:0];
                        [enc setBuffer:cur offset:0 atIndex:1];
                        [enc setBuffer:weights offset:0 atIndex:2];
                        [enc setBuffer:params offset:0 atIndex:3];
                        dispatch_2d(enc, pipelines.blur_v, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height));
                        [enc endEncoding];
                    }
                    sigma_cur = sigma;
                }
            }
            // sigma < 0.1 のレベルは reference が原画像を使う挙動に一致(cur は原画像のまま)。

            encode_accumulate(level);
        }

        {
            id<MTLBuffer> params = make_params(num_levels - 1, 0);
            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            [enc setBuffer:acc offset:0 atIndex:0];
            [enc setBuffer:wsum offset:0 atIndex:1];
            [enc setBuffer:output offset:0 atIndex:2];
            [enc setBuffer:params offset:0 atIndex:3];
            dispatch_1d(enc, pipelines.finalize, static_cast<NSUInteger>(count));
            [enc endEncoding];
        }

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        std::memcpy(out.ptr, [output contents], image_bytes);
    }

    return result;
}

PYBIND11_MODULE(_lens_blur_metal, m) {
    m.def("metal_available", &metal_available);
    m.def("apply_lensblur", &apply_lensblur);
}
