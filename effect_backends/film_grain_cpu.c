#include "film_grain_capi.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define FG_KR 0.2126f
#define FG_KG 0.7152f
#define FG_KB 0.0722f

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static float smoothstep01(float x) {
    const float t = clampf(x, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static int validate_images(
    const FilmGrainConstImageF32* input,
    const FilmGrainImageF32* output,
    const FilmGrainParams* params
) {
    if (input == 0 || output == 0 || params == 0 || input->data == 0 || output->data == 0) {
        return FILM_GRAIN_ERR_NULL;
    }
    if (
        input->width <= 0
        || input->height <= 0
        || input->channels < 3
        || output->channels != input->channels
        || output->width != input->width
        || output->height != input->height
    ) {
        return FILM_GRAIN_ERR_SHAPE;
    }
    return FILM_GRAIN_OK;
}

static int checked_plane_size(int width, int height, size_t* out) {
    if (width <= 0 || height <= 0) {
        return 0;
    }
    const size_t wh = (size_t)width * (size_t)height;
    if (wh / (size_t)width != (size_t)height) {
        return 0;
    }
    *out = wh;
    return 1;
}

static uint32_t mix_u32(uint32_t v) {
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

static uint32_t grain_seed(int seed, int height, int width) {
    uint32_t s = (uint32_t)seed;
    if (s == 0u) {
        s = 0x6D2B79F5u;
    }
    s ^= ((uint32_t)height * 73856093u);
    s ^= ((uint32_t)width * 19349663u);
    return s;
}

static float hash_noise(int x, int y, uint32_t seed, uint32_t salt) {
    uint32_t h = seed ^ salt;
    h ^= (uint32_t)x * 0x8da6b343u;
    h ^= (uint32_t)y * 0xd8163841u;
    h = mix_u32(h);
    const float u = (float)(h & 0x00FFFFFFu) * (1.0f / 16777215.0f);
    return (u * 2.0f - 1.0f) * 1.7320508075688772f;
}

static float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

static float layer_noise(int x, int y, float grain_size, uint32_t seed, uint32_t salt) {
    grain_size = grain_size < 0.35f ? 0.35f : grain_size;
    if (grain_size <= 0.75f) {
        return hash_noise(x, y, seed, salt);
    }

    const float sx = (float)x / grain_size;
    const float sy = (float)y / grain_size;
    const int x0 = (int)floorf(sx);
    const int y0 = (int)floorf(sy);
    const float fx = sx - (float)x0;
    const float fy = sy - (float)y0;
    const float wx = smoothstep01(fx);
    const float wy = smoothstep01(fy);

    const float n00 = hash_noise(x0, y0, seed, salt);
    const float n10 = hash_noise(x0 + 1, y0, seed, salt);
    const float n01 = hash_noise(x0, y0 + 1, seed, salt);
    const float n11 = hash_noise(x0 + 1, y0 + 1, seed, salt);
    const float nx0 = lerpf(n00, n10, wx);
    const float nx1 = lerpf(n01, n11, wx);
    return lerpf(nx0, nx1, wy);
}

static float safe_luma(float r, float g, float b) {
    r = isfinite(r) ? r : (r > 0.0f ? 1.0f : 0.0f);
    g = isfinite(g) ? g : (g > 0.0f ? 1.0f : 0.0f);
    b = isfinite(b) ? b : (b > 0.0f ? 1.0f : 0.0f);
    return clampf(FG_KR * r + FG_KG * g + FG_KB * b, 0.0f, 1.0f);
}

int film_grain_apply_v1(
    const FilmGrainConstImageF32* input,
    FilmGrainImageF32* output,
    const FilmGrainParams* params
) {
    int status = validate_images(input, output, params);
    if (status != FILM_GRAIN_OK) {
        return status;
    }

    const float amount = clampf(params->amount, 0.0f, 100.0f);
    const int width = input->width;
    const int height = input->height;
    const int channels = input->channels;
    size_t count = 0;
    if (!checked_plane_size(width, height, &count)) {
        return FILM_GRAIN_ERR_SHAPE;
    }

    if (amount <= 0.0f) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < height; ++y) {
            const float* src_row = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
            float* dst_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);
            memcpy(dst_row, src_row, sizeof(float) * (size_t)width * (size_t)channels);
        }
        return FILM_GRAIN_OK;
    }

    float* mono = (float*)malloc(sizeof(float) * count);
    if (mono == 0) {
        return FILM_GRAIN_ERR_ALLOC;
    }

    const float rough = clampf(params->roughness, 0.0f, 100.0f) / 100.0f;
    const float shadow_gain = 0.35f + clampf(params->shadow, 0.0f, 100.0f) / 100.0f * 1.35f;
    const float highlight_gain = 0.15f + clampf(params->highlight, 0.0f, 100.0f) / 100.0f * 1.10f;
    const float color_gain = clampf(params->color, 0.0f, 100.0f) / 100.0f;
    const float base_size = params->grain_size > 0.35f ? params->grain_size : 0.35f;
    const uint32_t seed = grain_seed(params->seed, height, width);

    const float fine_size = base_size * 0.55f;
    const float mid_size = base_size;
    const float coarse_size = base_size * 2.35f;
    const float fine_w = 0.25f + 0.55f * rough;
    const float mid_w = 0.70f;
    const float coarse_w = 0.55f * (1.0f - rough);

    double sum = 0.0;
    double sumsq = 0.0;

    #pragma omp parallel for schedule(static) reduction(+:sum,sumsq)
    for (int y = 0; y < height; ++y) {
        const size_t row_base = (size_t)y * (size_t)width;
        for (int x = 0; x < width; ++x) {
            const float fine = layer_noise(x, y, fine_size, seed, 0xA53A9D1Bu);
            const float mid = layer_noise(x, y, mid_size, seed, 0xC2B2AE35u);
            const float coarse = layer_noise(x, y, coarse_size, seed, 0x9E3779B9u);
            const float v = fine_w * fine + mid_w * mid + coarse_w * coarse;
            mono[row_base + (size_t)x] = v;
            sum += (double)v;
            sumsq += (double)v * (double)v;
        }
    }

    const double mean = sum / (double)count;
    double variance = sumsq / (double)count - mean * mean;
    if (variance < 1.0e-12) {
        variance = 1.0;
    }
    const float inv_std = (float)(1.0 / sqrt(variance));
    const float amount_scale = (amount / 100.0f) * 0.045f;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const float* src_row = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
        float* dst_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);
        const size_t row_base = (size_t)y * (size_t)width;
        for (int x = 0; x < width; ++x) {
            const int base = x * channels;
            const float r0 = src_row[base + 0];
            const float g0 = src_row[base + 1];
            const float b0 = src_row[base + 2];

            const float luma = safe_luma(r0, g0, b0);
            const float shadow_w = powf(1.0f - luma, 1.55f);
            const float highlight_w = powf(luma, 1.75f);
            const float midtone_w = 1.0f - powf(fabsf(luma * 2.0f - 1.0f), 1.65f);
            const float response = 0.50f * midtone_w
                + 0.42f * shadow_gain * shadow_w
                + 0.32f * highlight_gain * highlight_w;
            const float headroom = fminf(luma, 1.0f - luma);
            const float protect = 0.45f + 0.55f * clampf(headroom * 5.0f, 0.0f, 1.0f);
            const float amplitude = amount_scale * response * protect;
            const float m = (mono[row_base + (size_t)x] - (float)mean) * inv_std;

            float r = r0 + m * amplitude;
            float g = g0 + m * amplitude;
            float b = b0 + m * amplitude;

            if (color_gain > 0.0f) {
                const float u = layer_noise(x, y, base_size * 1.35f, seed, 0x85EBCA6Bu);
                const float v = layer_noise(x, y, base_size * 1.75f, seed, 0x27D4EB2Fu);
                const float c_amp = amplitude * color_gain * 0.42f;
                r += (u * 0.82f + v * 0.28f) * c_amp;
                g += (u * -0.45f + v * 0.42f) * c_amp;
                b += (u * -0.37f + v * -0.70f) * c_amp;
            }

            dst_row[base + 0] = r;
            dst_row[base + 1] = g;
            dst_row[base + 2] = b;
            for (int c = 3; c < channels; ++c) {
                dst_row[base + c] = src_row[base + c];
            }
        }
    }

    free(mono);
    return FILM_GRAIN_OK;
}
