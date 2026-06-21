#include "color_separation_capi.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define CS_KR 0.2126f
#define CS_KG 0.7152f
#define CS_KB 0.0722f

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static float smoothstepf(float e0, float e1, float x) {
    const float t = clampf((x - e0) / (e1 - e0 + 1.0e-12f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static int validate_images(
    const ColorSeparationConstImageF32* input,
    const ColorSeparationImageF32* output,
    const ColorSeparationParams* params
) {
    if (input == 0 || output == 0 || params == 0 || input->data == 0 || output->data == 0) {
        return COLOR_SEPARATION_ERR_NULL;
    }
    if (
        input->width <= 0
        || input->height <= 0
        || input->channels != 3
        || output->channels != 3
        || input->width != output->width
        || input->height != output->height
    ) {
        return COLOR_SEPARATION_ERR_SHAPE;
    }
    return COLOR_SEPARATION_OK;
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

static int reflect101(int p, int len) {
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

static int gaussian_radius(float sigma) {
    int ksize = (int)floorf(sigma * 6.0f + 1.0f + 0.5f);
    if ((ksize & 1) == 0) {
        ++ksize;
    }
    if (ksize < 3) {
        ksize = 3;
    }
    return ksize / 2;
}

static int make_gaussian_kernel(float sigma, float** out_kernel, int* out_radius) {
    const int radius = gaussian_radius(sigma);
    const int size = radius * 2 + 1;
    float* kernel = (float*)malloc(sizeof(float) * (size_t)size);
    if (kernel == 0) {
        return 0;
    }

    const float denom = 2.0f * sigma * sigma;
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        const float v = expf(-((float)(i * i)) / denom);
        kernel[i + radius] = v;
        sum += v;
    }
    const float inv_sum = sum != 0.0f ? 1.0f / sum : 1.0f;
    for (int i = 0; i < size; ++i) {
        kernel[i] *= inv_sum;
    }
    *out_kernel = kernel;
    *out_radius = radius;
    return 1;
}

static void gaussian_horizontal(
    const float* src,
    float* dst,
    int width,
    int height,
    const float* kernel,
    int radius
) {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const size_t row_base = (size_t)y * (size_t)width;
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                const int sx = reflect101(x + k, width);
                sum += src[row_base + (size_t)sx] * kernel[k + radius];
            }
            dst[row_base + (size_t)x] = sum;
        }
    }
}

static void gaussian_vertical(
    const float* src,
    float* dst,
    int width,
    int height,
    const float* kernel,
    int radius
) {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const size_t row_base = (size_t)y * (size_t)width;
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                const int sy = reflect101(y + k, height);
                sum += src[(size_t)sy * (size_t)width + (size_t)x] * kernel[k + radius];
            }
            dst[row_base + (size_t)x] = sum;
        }
    }
}

static int gaussian_blur_plane(
    const float* src,
    float* tmp,
    float* dst,
    int width,
    int height,
    float sigma
) {
    float* kernel = 0;
    int radius = 0;
    if (!make_gaussian_kernel(sigma, &kernel, &radius)) {
        return 0;
    }
    gaussian_horizontal(src, tmp, width, height, kernel, radius);
    gaussian_vertical(tmp, dst, width, height, kernel, radius);
    free(kernel);
    return 1;
}

static void copy_input_to_ycbcr(
    const ColorSeparationConstImageF32* input,
    float* y_plane,
    float* cb_plane,
    float* cr_plane
) {
    const int width = input->width;
    const int height = input->height;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const float* src_row = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
        const size_t row_base = (size_t)y * (size_t)width;
        for (int x = 0; x < width; ++x) {
            const int base = x * 3;
            const float r = src_row[base + 0];
            const float g = src_row[base + 1];
            const float b = src_row[base + 2];
            const float yy = CS_KR * r + CS_KG * g + CS_KB * b;
            y_plane[row_base + (size_t)x] = yy;
            cb_plane[row_base + (size_t)x] = (b - yy) / 1.8556f;
            cr_plane[row_base + (size_t)x] = (r - yy) / 1.5748f;
        }
    }
}

static void apply_shadow_clean(
    float* cb,
    float* cr,
    const float* y_plane,
    int width,
    int height,
    float shadow_chroma_clean,
    float shadow_threshold
) {
    if (shadow_chroma_clean <= 0.0f || shadow_threshold <= 0.0f) {
        return;
    }
    const size_t count = (size_t)width * (size_t)height;
    const float threshold = shadow_threshold > 1.0e-4f ? shadow_threshold : 1.0e-4f;
    const float clean_amount = clampf(shadow_chroma_clean, 0.0f, 1.0f) * 0.9f;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        const float yy = y_plane[i];
        const float cbv = cb[i];
        const float crv = cr[i];
        const float chroma = sqrtf(cbv * cbv + crv * crv);
        const float relative_chroma = chroma / (fmaxf(yy, 0.0f) + 1.0e-4f);
        const float shadow_mask = 1.0f - smoothstepf(threshold * 0.35f, threshold, yy);
        const float vivid_protect = smoothstepf(0.12f, 0.45f, relative_chroma);
        const float clean_scale = 1.0f - clean_amount * shadow_mask * (1.0f - vivid_protect);
        cb[i] = cbv * clean_scale;
        cr[i] = crv * clean_scale;
    }
}

static void compute_clarity_weight(
    const float* cb,
    const float* cr,
    const float* y_plane,
    float* weight,
    size_t count
) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        const float yy = y_plane[i];
        const float cbv = cb[i];
        const float crv = cr[i];
        const float chroma = sqrtf(cbv * cbv + crv * crv);
        const float relative_chroma = chroma / (fmaxf(yy, 0.0f) + 1.0e-4f);
        const float midtone_mask = smoothstepf(0.035f, 0.18f, yy);
        const float hdr_protect = 1.0f - smoothstepf(1.6f, 4.0f, yy);
        const float neutral_gate = smoothstepf(0.015f, 0.10f, relative_chroma);
        const float vivid_limit = 1.0f - 0.45f * smoothstepf(0.80f, 1.80f, relative_chroma);
        weight[i] = midtone_mask * hdr_protect * neutral_gate * vivid_limit;
    }
}

static void apply_clarity_channel(
    float* channel,
    const float* local,
    const float* base,
    const float* weight,
    size_t count,
    float clarity_gain
) {
    const float scale = clarity_gain * 1.15f;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        channel[i] = channel[i] + (local[i] - base[i]) * scale * weight[i];
    }
}

static int apply_chroma_clarity(
    float* cb,
    float* cr,
    const float* y_plane,
    int width,
    int height,
    float chroma_clarity
) {
    if (chroma_clarity == 0.0f) {
        return 1;
    }
    size_t count = 0;
    if (!checked_plane_size(width, height, &count)) {
        return 0;
    }

    float* tmp = (float*)malloc(sizeof(float) * count);
    float* local = (float*)malloc(sizeof(float) * count);
    float* base = (float*)malloc(sizeof(float) * count);
    float* weight = (float*)malloc(sizeof(float) * count);
    if (tmp == 0 || local == 0 || base == 0 || weight == 0) {
        free(tmp);
        free(local);
        free(base);
        free(weight);
        return 0;
    }

    compute_clarity_weight(cb, cr, y_plane, weight, count);
    const float clarity_gain = clampf(chroma_clarity, -1.0f, 1.0f);
    if (
        !gaussian_blur_plane(cb, tmp, local, width, height, 1.2f)
        || !gaussian_blur_plane(cb, tmp, base, width, height, 7.0f)
    ) {
        free(tmp);
        free(local);
        free(base);
        free(weight);
        return 0;
    }
    apply_clarity_channel(cb, local, base, weight, count, clarity_gain);
    if (
        !gaussian_blur_plane(cr, tmp, local, width, height, 1.2f)
        || !gaussian_blur_plane(cr, tmp, base, width, height, 7.0f)
    ) {
        free(tmp);
        free(local);
        free(base);
        free(weight);
        return 0;
    }
    apply_clarity_channel(cr, local, base, weight, count, clarity_gain);

    free(tmp);
    free(local);
    free(base);
    free(weight);
    return 1;
}

static void apply_color_separation_stage(
    float* cb,
    float* cr,
    const float* y_plane,
    size_t count,
    float color_separation
) {
    if (color_separation <= 0.0f) {
        return;
    }
    const float amount = clampf(color_separation, 0.0f, 1.0f);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        const float yy = y_plane[i];
        const float cbv = cb[i];
        const float crv = cr[i];
        const float chroma = sqrtf(cbv * cbv + crv * crv);
        const float relative_chroma = chroma / (fmaxf(yy, 0.0f) + 1.0e-4f);
        const float midtone_mask = smoothstepf(0.04f, 0.22f, yy);
        const float hdr_protect = 1.0f - smoothstepf(1.6f, 4.0f, yy);
        const float vivid_limit = 1.0f - 0.65f * smoothstepf(0.30f, 0.90f, relative_chroma);
        const float sep_gain = 1.0f + amount * 0.35f * midtone_mask * hdr_protect * vivid_limit;
        cb[i] = cbv * sep_gain;
        cr[i] = crv * sep_gain;
    }
}

static void apply_color_density_stage(
    float* cb,
    float* cr,
    const float* y_plane,
    size_t count,
    float color_density
) {
    if (color_density == 0.0f) {
        return;
    }
    const float density_value = clampf(color_density, -1.0f, 1.0f);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        const float yy = y_plane[i];
        const float cbv = cb[i];
        const float crv = cr[i];
        const float chroma = sqrtf(cbv * cbv + crv * crv);
        const float relative_chroma = chroma / (fmaxf(yy, 0.0f) + 1.0e-4f);
        const float midtone_mask = smoothstepf(0.06f, 0.24f, yy) * (1.0f - smoothstepf(1.4f, 3.2f, yy));
        const float neutral_gate = smoothstepf(0.025f, 0.18f, relative_chroma);
        float density_gain = 1.0f;
        if (density_value > 0.0f) {
            const float vivid_rolloff = 1.0f - 0.85f * smoothstepf(0.45f, 1.05f, relative_chroma);
            const float density_amount = density_value * midtone_mask * neutral_gate * vivid_rolloff;
            const float target_chroma = chroma + 0.10f * tanhf(chroma / 0.10f);
            density_gain = 1.0f + density_amount * ((target_chroma / (chroma + 1.0e-6f)) - 1.0f);
        } else {
            const float vivid_rolloff = 1.0f - 0.35f * smoothstepf(0.70f, 1.60f, relative_chroma);
            const float density_amount = (-density_value) * midtone_mask * neutral_gate * vivid_rolloff;
            density_gain = 1.0f - 0.40f * density_amount;
        }
        cb[i] = cbv * density_gain;
        cr[i] = crv * density_gain;
    }
}

static void apply_subtractive_saturation_pixel(float* r, float* g, float* b, float amount) {
    amount = clampf(amount, -1.0f, 1.0f);
    if (amount == 0.0f) {
        return;
    }
    const float yy = CS_KR * *r + CS_KG * *g + CS_KB * *b;
    const float rv = *r - yy;
    const float gv = *g - yy;
    const float bv = *b - yy;
    const float chroma = sqrtf(rv * rv + gv * gv + bv * bv);
    const float relative_chroma = chroma / (fmaxf(yy, 0.0f) + 1.0e-4f);
    const float chroma_gate = smoothstepf(0.025f, 0.42f, relative_chroma);
    const float midtone_gate = smoothstepf(0.035f, 0.24f, yy) * (1.0f - smoothstepf(1.7f, 4.0f, yy));
    float sat_gain = 1.0f;
    float density = 1.0f;
    if (amount > 0.0f) {
        const float vivid_rolloff = 1.0f - 0.45f * smoothstepf(0.95f, 2.20f, relative_chroma);
        sat_gain = 1.0f + amount * 0.55f * chroma_gate * midtone_gate * vivid_rolloff;
        density = 1.0f - amount * 0.18f * chroma_gate * midtone_gate;
    } else {
        const float soften = -amount;
        sat_gain = 1.0f - soften * 0.42f * chroma_gate * midtone_gate;
        density = 1.0f + soften * 0.08f * chroma_gate * midtone_gate;
    }
    *r = (yy + rv * sat_gain) * density;
    *g = (yy + gv * sat_gain) * density;
    *b = (yy + bv * sat_gain) * density;
}

static void write_rgb_output(
    const ColorSeparationConstImageF32* input,
    ColorSeparationImageF32* output,
    const float* y_plane,
    const float* cb,
    const float* cr,
    const ColorSeparationParams* params
) {
    const int width = input->width;
    const int height = input->height;
    const float subtractive_saturation = params->subtractive_saturation;
    const float opponent_contrast = params->opponent_contrast;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const float* src_row = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
        float* dst_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);
        const size_t row_base = (size_t)y * (size_t)width;
        for (int x = 0; x < width; ++x) {
            const size_t idx = row_base + (size_t)x;
            const float yy = y_plane[idx];
            float r = yy + 1.5748f * cr[idx];
            float b = yy + 1.8556f * cb[idx];
            float g = yy - 0.1873f * cb[idx] - 0.4681f * cr[idx];

            if (subtractive_saturation != 0.0f) {
                apply_subtractive_saturation_pixel(&r, &g, &b, subtractive_saturation);
            }
            if (opponent_contrast > 0.0f) {
                const float y_opp = CS_KR * r + CS_KG * g + CS_KB * b;
                float rg = r - g;
                float by = b - 0.5f * (r + g);
                const float opponent_strength = (fabsf(rg) + fabsf(by)) / (fmaxf(y_opp, 0.0f) + 1.0e-4f);
                const float midtone_mask = smoothstepf(0.05f, 0.24f, y_opp);
                const float hdr_protect = 1.0f - smoothstepf(1.6f, 4.0f, y_opp);
                const float vivid_rolloff = 1.0f - 0.70f * smoothstepf(0.70f, 1.80f, opponent_strength);
                const float opponent_gain = 1.0f
                    + clampf(opponent_contrast, 0.0f, 1.0f) * 0.26f * midtone_mask * hdr_protect * vivid_rolloff;
                rg *= opponent_gain;
                by *= opponent_gain;
                const float g_new = y_opp - (CS_KR + CS_KB * 0.5f) * rg - CS_KB * by;
                r = g_new + rg;
                g = g_new;
                b = g_new + 0.5f * rg + by;
            }

            const int base = x * 3;
            const float lower_r = src_row[base + 0] < 0.0f ? src_row[base + 0] : 0.0f;
            const float lower_g = src_row[base + 1] < 0.0f ? src_row[base + 1] : 0.0f;
            const float lower_b = src_row[base + 2] < 0.0f ? src_row[base + 2] : 0.0f;
            dst_row[base + 0] = r > lower_r ? r : lower_r;
            dst_row[base + 1] = g > lower_g ? g : lower_g;
            dst_row[base + 2] = b > lower_b ? b : lower_b;
        }
    }
}

int color_separation_apply_v1(
    const ColorSeparationConstImageF32* input,
    ColorSeparationImageF32* output,
    const ColorSeparationParams* params
) {
    int status = validate_images(input, output, params);
    if (status != COLOR_SEPARATION_OK) {
        return status;
    }

    size_t count = 0;
    if (!checked_plane_size(input->width, input->height, &count)) {
        return COLOR_SEPARATION_ERR_SHAPE;
    }

    float* y_plane = (float*)malloc(sizeof(float) * count);
    float* cb = (float*)malloc(sizeof(float) * count);
    float* cr = (float*)malloc(sizeof(float) * count);
    if (y_plane == 0 || cb == 0 || cr == 0) {
        free(y_plane);
        free(cb);
        free(cr);
        return COLOR_SEPARATION_ERR_ALLOC;
    }

    copy_input_to_ycbcr(input, y_plane, cb, cr);
    apply_shadow_clean(cb, cr, y_plane, input->width, input->height, params->shadow_chroma_clean, params->shadow_threshold);
    if (!apply_chroma_clarity(cb, cr, y_plane, input->width, input->height, params->chroma_clarity)) {
        free(y_plane);
        free(cb);
        free(cr);
        return COLOR_SEPARATION_ERR_ALLOC;
    }
    apply_color_separation_stage(cb, cr, y_plane, count, params->color_separation);
    apply_color_density_stage(cb, cr, y_plane, count, params->color_density);
    write_rgb_output(input, output, y_plane, cb, cr, params);

    free(y_plane);
    free(cb);
    free(cr);
    return COLOR_SEPARATION_OK;
}
