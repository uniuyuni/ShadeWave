#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_buffer_utils.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

// Host/device shared parameter block. All fields are 4-byte aligned scalars so
// the C++ and Metal struct layouts match without float3/array padding hazards.
struct LutMetalParams {
    int pixel_count;
    int size;
    float dmin0;
    float dmin1;
    float dmin2;
    float dmax0;
    float dmax1;
    float dmax2;
};

constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct LutMetalParams {
    int pixel_count;
    int size;
    float dmin0;
    float dmin1;
    float dmin2;
    float dmax0;
    float dmax1;
    float dmax2;
};

// table layout (size, size, size, 3) row-major.
static inline float table_at(const device float* table, int S, int i, int j, int k, int ch) {
    return table[(((i * S) + j) * S + k) * 3 + ch];
}

kernel void lut_trilinear(
    const device float* input [[buffer(0)]],
    const device float* table [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LutMetalParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    if (idx >= p.pixel_count) {
        return;
    }
    int base = idx * 3;
    int S = p.size;
    float sm1 = float(S - 1);

    float r = input[base + 0];
    float g = input[base + 1];
    float b = input[base + 2];

    // Clip to domain, normalize to [0,1], scale to grid coords.
    float gr = (clamp(r, p.dmin0, p.dmax0) - p.dmin0) / (p.dmax0 - p.dmin0) * sm1;
    float gg = (clamp(g, p.dmin1, p.dmax1) - p.dmin1) / (p.dmax1 - p.dmin1) * sm1;
    float gb = (clamp(b, p.dmin2, p.dmax2) - p.dmin2) / (p.dmax2 - p.dmin2) * sm1;

    // BGR index convention: axis0 (first table index) = gB, axis1 = gG, axis2 = gR.
    float ci = clamp(gb, 0.0f, sm1);
    int i0 = clamp(int(floor(ci)), 0, S - 2);
    int i1 = i0 + 1;
    float wi = ci - float(i0);

    float cj = clamp(gg, 0.0f, sm1);
    int j0 = clamp(int(floor(cj)), 0, S - 2);
    int j1 = j0 + 1;
    float wj = cj - float(j0);

    float ck = clamp(gr, 0.0f, sm1);
    int k0 = clamp(int(floor(ck)), 0, S - 2);
    int k1 = k0 + 1;
    float wk = ck - float(k0);

    for (int ch = 0; ch < 3; ++ch) {
        float c000 = table_at(table, S, i0, j0, k0, ch);
        float c001 = table_at(table, S, i0, j0, k1, ch);
        float c010 = table_at(table, S, i0, j1, k0, ch);
        float c011 = table_at(table, S, i0, j1, k1, ch);
        float c100 = table_at(table, S, i1, j0, k0, ch);
        float c101 = table_at(table, S, i1, j0, k1, ch);
        float c110 = table_at(table, S, i1, j1, k0, ch);
        float c111 = table_at(table, S, i1, j1, k1, ch);

        // Interpolate along axis0 (wi), then axis1 (wj), then axis2 (wk).
        float c00 = mix(c000, c100, wi);
        float c01 = mix(c001, c101, wi);
        float c10 = mix(c010, c110, wi);
        float c11 = mix(c011, c111, wi);

        float c0 = mix(c00, c10, wj);
        float c1 = mix(c01, c11, wj);

        output[base + ch] = mix(c0, c1, wk);
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
    id<MTLComputePipelineState> trilinear;
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
                state.trilinear = make_pipeline(state.device, state.library, @"lut_trilinear");
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

}  // namespace

py::array_t<float> apply_lut3d(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    py::array_t<float, py::array::c_style | py::array::forcecast> table,
    py::array_t<float, py::array::c_style | py::array::forcecast> domain,
    int size
) {
    py::buffer_info in = image.request();
    if (in.ndim < 2 || in.shape[in.ndim - 1] != 3) {
        throw std::invalid_argument("image must have last dimension 3 (float32 RGB)");
    }

    py::buffer_info tbl = table.request();
    const size_t expected_table = static_cast<size_t>(size) * size * size * 3;
    if (static_cast<size_t>(tbl.size) != expected_table) {
        throw std::invalid_argument("table size does not match size^3 * 3");
    }

    py::buffer_info dom = domain.request();
    if (static_cast<size_t>(dom.size) != 6) {
        throw std::invalid_argument("domain must contain 6 floats (2 x 3)");
    }

    size_t total = 1;
    for (int d = 0; d < in.ndim; ++d) {
        total *= static_cast<size_t>(in.shape[d]);
    }
    const int pixel_count = static_cast<int>(total / 3);

    std::vector<py::ssize_t> shape(in.shape.begin(), in.shape.end());
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    const float* dptr = static_cast<const float*>(dom.ptr);

    {
        py::gil_scoped_release release;
        @autoreleasepool {
            MetalPipelines& pipelines = metal_pipelines();

            const size_t image_bytes = static_cast<size_t>(pixel_count) * 3 * sizeof(float);
            const size_t table_bytes = expected_table * sizeof(float);

            BufferBinding input_binding = make_buffer_for_input(pipelines.device, in.ptr, image_bytes);
            BufferBinding table_binding = make_buffer_for_input(pipelines.device, tbl.ptr, table_bytes);
            BufferBinding output_binding = make_buffer_for_output(pipelines.device, out.ptr, image_bytes);
            id<MTLBuffer> input_buffer = input_binding.buffer;
            id<MTLBuffer> table_buffer = table_binding.buffer;
            id<MTLBuffer> output_buffer = output_binding.buffer;

            LutMetalParams params{
                pixel_count,
                size,
                dptr[0], dptr[1], dptr[2],
                dptr[3], dptr[4], dptr[5],
            };
            id<MTLBuffer> params_buffer = [pipelines.device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:pipelines.trilinear];
            [encoder setBuffer:input_buffer offset:input_binding.offset atIndex:0];
            [encoder setBuffer:table_buffer offset:table_binding.offset atIndex:1];
            [encoder setBuffer:output_buffer offset:output_binding.offset atIndex:2];
            [encoder setBuffer:params_buffer offset:0 atIndex:3];

            NSUInteger tpg = std::max<NSUInteger>(1, pipelines.trilinear.maxTotalThreadsPerThreadgroup);
            MTLSize threads_per_group = MTLSizeMake(tpg, 1, 1);
            MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(pixel_count), 1, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:threads_per_group];
            [encoder endEncoding];

            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if ([command_buffer error]) {
                throw std::runtime_error([[[command_buffer error] localizedDescription] UTF8String]);
            }

            finish_output_binding(output_binding, out.ptr, image_bytes);
        }
    }

    return result;
}

PYBIND11_MODULE(_lut_metal, m) {
    m.doc() = "Metal 3D LUT backend for Platypus";
    m.def("metal_available", []() {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            return device != nil;
        }
    });
    m.def(
        "apply_lut3d",
        &apply_lut3d,
        py::arg("image"),
        py::arg("table"),
        py::arg("domain"),
        py::arg("size")
    );
}
