#include "low_frequency_transfer_capi.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define LOW_FREQUENCY_TRANSFER_BOXES 4

static int validate_images(
    const LowFrequencyTransferConstImageF32* restored,
    const LowFrequencyTransferConstImageF32* reference,
    const LowFrequencyTransferImageF32* output
) {
    if (restored == 0 || reference == 0 || output == 0 || restored->data == 0 || reference->data == 0 || output->data == 0) {
        return LOW_FREQUENCY_TRANSFER_ERR_NULL;
    }
    if (
        restored->width <= 0
        || restored->height <= 0
        || restored->width != reference->width
        || restored->height != reference->height
        || restored->width != output->width
        || restored->height != output->height
    ) {
        return LOW_FREQUENCY_TRANSFER_ERR_SHAPE;
    }
    if (restored->channels != reference->channels || restored->channels != output->channels) {
        return LOW_FREQUENCY_TRANSFER_ERR_SHAPE;
    }
    if (restored->channels != 1 && restored->channels != 3) {
        return LOW_FREQUENCY_TRANSFER_ERR_SHAPE;
    }
    return LOW_FREQUENCY_TRANSFER_OK;
}

static int checked_count(int width, int height, int channels, size_t* out) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        return 0;
    }
    const size_t wh = (size_t)width * (size_t)height;
    if (wh / (size_t)width != (size_t)height) {
        return 0;
    }
    const size_t whc = wh * (size_t)channels;
    if (whc / wh != (size_t)channels) {
        return 0;
    }
    *out = whc;
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

static float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

static float smoothstep01(float v) {
    const float t = clamp01(v);
    return t * t * (3.0f - 2.0f * t);
}

static float clamp_strength(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

static float sample_bilinear_lowres(
    const LowFrequencyTransferConstImageF32* image,
    int full_width,
    int full_height,
    int x,
    int y,
    int c
) {
    const float sx = ((float)x + 0.5f) * (float)image->width / (float)full_width - 0.5f;
    const float sy = ((float)y + 0.5f) * (float)image->height / (float)full_height - 0.5f;
    const int x0_raw = (int)floorf(sx);
    const int y0_raw = (int)floorf(sy);
    const int x1_raw = x0_raw + 1;
    const int y1_raw = y0_raw + 1;
    const float wx = sx - (float)x0_raw;
    const float wy = sy - (float)y0_raw;
    const int x0 = x0_raw < 0 ? 0 : (x0_raw >= image->width ? image->width - 1 : x0_raw);
    const int x1 = x1_raw < 0 ? 0 : (x1_raw >= image->width ? image->width - 1 : x1_raw);
    const int y0 = y0_raw < 0 ? 0 : (y0_raw >= image->height ? image->height - 1 : y0_raw);
    const int y1 = y1_raw < 0 ? 0 : (y1_raw >= image->height ? image->height - 1 : y1_raw);
    const float* row0 = (const float*)((const unsigned char*)image->data + (size_t)y0 * image->stride_bytes);
    const float* row1 = (const float*)((const unsigned char*)image->data + (size_t)y1 * image->stride_bytes);
    const float v00 = row0[x0 * image->channels + c];
    const float v10 = row0[x1 * image->channels + c];
    const float v01 = row1[x0 * image->channels + c];
    const float v11 = row1[x1 * image->channels + c];
    const float top = v00 + (v10 - v00) * wx;
    const float bottom = v01 + (v11 - v01) * wx;
    return top + (bottom - top) * wy;
}

static void gaussian_box_sizes(float sigma, int* sizes, int count) {
    if (sigma < 0.01f) {
        sigma = 0.01f;
    }
    const float n = (float)count;
    float w_ideal = sqrtf((12.0f * sigma * sigma / n) + 1.0f);
    int wl = (int)floorf(w_ideal);
    if ((wl & 1) == 0) {
        wl -= 1;
    }
    if (wl < 1) {
        wl = 1;
    }
    const int wu = wl + 2;
    const float m_ideal = (12.0f * sigma * sigma - n * (float)(wl * wl) - 4.0f * n * (float)wl - 3.0f * n)
        / (-4.0f * (float)wl - 4.0f);
    int m = (int)roundf(m_ideal);
    if (m < 0) {
        m = 0;
    }
    if (m > count) {
        m = count;
    }
    for (int i = 0; i < count; ++i) {
        sizes[i] = i < m ? wl : wu;
    }
}

static void box_blur_horizontal(const float* src, float* dst, int width, int height, int channels, int radius) {
    const float norm = 1.0f / (float)(radius * 2 + 1);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const size_t row_base = (size_t)y * (size_t)width * (size_t)channels;
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                const int sx = reflect101(k, width);
                sum += src[row_base + (size_t)sx * (size_t)channels + (size_t)c];
            }
            for (int x = 0; x < width; ++x) {
                dst[row_base + (size_t)x * (size_t)channels + (size_t)c] = sum * norm;
                const int remove_x = reflect101(x - radius, width);
                const int add_x = reflect101(x + radius + 1, width);
                sum += src[row_base + (size_t)add_x * (size_t)channels + (size_t)c]
                    - src[row_base + (size_t)remove_x * (size_t)channels + (size_t)c];
            }
        }
    }
}

static void box_blur_vertical(const float* src, float* dst, int width, int height, int channels, int radius) {
    const float norm = 1.0f / (float)(radius * 2 + 1);

    #pragma omp parallel for schedule(static)
    for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                const int sy = reflect101(k, height);
                sum += src[((size_t)sy * (size_t)width + (size_t)x) * (size_t)channels + (size_t)c];
            }
            for (int y = 0; y < height; ++y) {
                dst[((size_t)y * (size_t)width + (size_t)x) * (size_t)channels + (size_t)c] = sum * norm;
                const int remove_y = reflect101(y - radius, height);
                const int add_y = reflect101(y + radius + 1, height);
                sum += src[((size_t)add_y * (size_t)width + (size_t)x) * (size_t)channels + (size_t)c]
                    - src[((size_t)remove_y * (size_t)width + (size_t)x) * (size_t)channels + (size_t)c];
            }
        }
    }
}

static void approximate_gaussian_inplace(float* image, float* scratch, int width, int height, int channels, float sigma) {
    if (sigma <= 0.01f) {
        return;
    }

    int sizes[LOW_FREQUENCY_TRANSFER_BOXES];
    gaussian_box_sizes(sigma, sizes, LOW_FREQUENCY_TRANSFER_BOXES);
    for (int i = 0; i < LOW_FREQUENCY_TRANSFER_BOXES; ++i) {
        const int radius = (sizes[i] - 1) / 2;
        if (radius < 1) {
            continue;
        }
        box_blur_horizontal(image, scratch, width, height, channels, radius);
        box_blur_vertical(scratch, image, width, height, channels, radius);
    }
}

static void fill_difference(
    const LowFrequencyTransferConstImageF32* restored,
    const LowFrequencyTransferConstImageF32* reference,
    float* out
) {
    const int width = restored->width;
    const int height = restored->height;
    const int channels = restored->channels;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const float* rest_row = (const float*)((const unsigned char*)restored->data + (size_t)y * restored->stride_bytes);
        const float* ref_row = (const float*)((const unsigned char*)reference->data + (size_t)y * reference->stride_bytes);
        float* dst_row = out + (size_t)y * (size_t)width * (size_t)channels;
        for (int x = 0; x < width * channels; ++x) {
            dst_row[x] = ref_row[x] - rest_row[x];
        }
    }
}

static void fill_restored_copy(const LowFrequencyTransferConstImageF32* restored, float* out) {
    const int width = restored->width;
    const int height = restored->height;
    const int channels = restored->channels;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const float* rest_row = (const float*)((const unsigned char*)restored->data + (size_t)y * restored->stride_bytes);
        float* dst_row = out + (size_t)y * (size_t)width * (size_t)channels;
        for (int x = 0; x < width * channels; ++x) {
            dst_row[x] = rest_row[x];
        }
    }
}

static void compose_without_highlight(
    const LowFrequencyTransferConstImageF32* restored,
    LowFrequencyTransferImageF32* output,
    const float* low_diff,
    const LowFrequencyTransferParams* params
) {
    const int width = restored->width;
    const int height = restored->height;
    const int channels = restored->channels;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const float* rest_row = (const float*)((const unsigned char*)restored->data + (size_t)y * restored->stride_bytes);
        float* out_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);
        const float* diff_row = low_diff + (size_t)y * (size_t)width * (size_t)channels;
        const float luma_strength = clamp_strength(params->luminance_transfer_strength);
        const float luma_remove = 1.0f - luma_strength;
        if (channels == 1) {
            for (int x = 0; x < width; ++x) {
                out_row[x] = rest_row[x] + diff_row[x] * luma_strength;
            }
        } else {
            for (int x = 0; x < width; ++x) {
                const int base = x * 3;
                const float dr = diff_row[base + 0];
                const float dg = diff_row[base + 1];
                const float db = diff_row[base + 2];
                const float lum = 0.2126f * dr + 0.7152f * dg + 0.0722f * db;
                out_row[base + 0] = rest_row[base + 0] + dr - lum * luma_remove;
                out_row[base + 1] = rest_row[base + 1] + dg - lum * luma_remove;
                out_row[base + 2] = rest_row[base + 2] + db - lum * luma_remove;
            }
        }
    }
}

static void compose_with_highlight(
    const LowFrequencyTransferConstImageF32* restored,
    const LowFrequencyTransferConstImageF32* reference,
    LowFrequencyTransferImageF32* output,
    const float* low_diff,
    const float* low_restored,
    const LowFrequencyTransferParams* params
) {
    const int width = restored->width;
    const int height = restored->height;
    const int channels = restored->channels;
    const float transition = params->highlight_transition <= 1.0e-6f ? 1.0e-6f : params->highlight_transition;
    const float detail_remove = 1.0f - params->highlight_detail_strength;
    const float luma_strength = clamp_strength(params->luminance_transfer_strength);
    const float luma_remove = 1.0f - luma_strength;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const float* rest_row = (const float*)((const unsigned char*)restored->data + (size_t)y * restored->stride_bytes);
        const float* ref_row = (const float*)((const unsigned char*)reference->data + (size_t)y * reference->stride_bytes);
        float* out_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);
        const size_t row_base = (size_t)y * (size_t)width * (size_t)channels;
        for (int x = 0; x < width; ++x) {
            float luminance = ref_row[x * channels];
            if (channels == 3) {
                const float g = ref_row[x * channels + 1];
                const float b = ref_row[x * channels + 2];
                luminance = luminance > g ? luminance : g;
                luminance = luminance > b ? luminance : b;
            }
            const float mask = smoothstep01((luminance - params->highlight_threshold) / transition);
            const float alpha = mask * detail_remove;
            const size_t base = row_base + (size_t)x * (size_t)channels;
            if (channels == 1) {
                const float restored_v = rest_row[x];
                const float high_restored = restored_v - low_restored[base];
                out_row[x] = restored_v + low_diff[base] * luma_strength - alpha * high_restored;
            } else {
                const int pix = x * 3;
                const float dr = low_diff[base + 0];
                const float dg = low_diff[base + 1];
                const float db = low_diff[base + 2];
                const float lum = 0.2126f * dr + 0.7152f * dg + 0.0722f * db;
                for (int c = 0; c < 3; ++c) {
                    const float restored_v = rest_row[pix + c];
                    const float high_restored = restored_v - low_restored[base + (size_t)c];
                    const float diff_v = low_diff[base + (size_t)c] - lum * luma_remove;
                    out_row[pix + c] = restored_v + diff_v - alpha * high_restored;
                }
            }
        }
    }
}

int low_frequency_transfer_apply_v1(
    const LowFrequencyTransferConstImageF32* restored,
    const LowFrequencyTransferConstImageF32* reference,
    LowFrequencyTransferImageF32* output,
    const LowFrequencyTransferParams* params
) {
    const int valid = validate_images(restored, reference, output);
    if (valid != LOW_FREQUENCY_TRANSFER_OK) {
        return valid;
    }
    if (params == 0) {
        return LOW_FREQUENCY_TRANSFER_ERR_NULL;
    }

    size_t count = 0;
    if (!checked_count(restored->width, restored->height, restored->channels, &count)) {
        return LOW_FREQUENCY_TRANSFER_ERR_SHAPE;
    }

    float* low_diff = (float*)output->data;
    float* scratch = (float*)malloc(sizeof(float) * count);
    if (scratch == 0) {
        return LOW_FREQUENCY_TRANSFER_ERR_ALLOC;
    }

    fill_difference(restored, reference, low_diff);
    approximate_gaussian_inplace(low_diff, scratch, restored->width, restored->height, restored->channels, params->sigma);

    if (!params->use_highlight_protection) {
        compose_without_highlight(restored, output, low_diff, params);
        free(scratch);
        return LOW_FREQUENCY_TRANSFER_OK;
    }

    float* low_restored = (float*)malloc(sizeof(float) * count);
    if (low_restored == 0) {
        free(scratch);
        return LOW_FREQUENCY_TRANSFER_ERR_ALLOC;
    }

    fill_restored_copy(restored, low_restored);
    approximate_gaussian_inplace(low_restored, scratch, restored->width, restored->height, restored->channels, params->sigma);
    compose_with_highlight(restored, reference, output, low_diff, low_restored, params);

    free(low_restored);
    free(scratch);
    return LOW_FREQUENCY_TRANSFER_OK;
}

int low_frequency_transfer_compose_lowres_v1(
    const LowFrequencyTransferConstImageF32* restored,
    const LowFrequencyTransferConstImageF32* reference,
    const LowFrequencyTransferConstImageF32* low_diff,
    const LowFrequencyTransferConstImageF32* low_restored,
    LowFrequencyTransferImageF32* output,
    const LowFrequencyTransferParams* params
) {
    const int valid = validate_images(restored, reference, output);
    if (valid != LOW_FREQUENCY_TRANSFER_OK) {
        return valid;
    }
    if (low_diff == 0 || low_diff->data == 0 || params == 0) {
        return LOW_FREQUENCY_TRANSFER_ERR_NULL;
    }
    if (low_diff->width <= 0 || low_diff->height <= 0 || low_diff->channels != restored->channels) {
        return LOW_FREQUENCY_TRANSFER_ERR_SHAPE;
    }
    if (params->use_highlight_protection) {
        if (low_restored == 0 || low_restored->data == 0) {
            return LOW_FREQUENCY_TRANSFER_ERR_NULL;
        }
        if (low_restored->width <= 0 || low_restored->height <= 0 || low_restored->channels != restored->channels) {
            return LOW_FREQUENCY_TRANSFER_ERR_SHAPE;
        }
    }

    const int width = restored->width;
    const int height = restored->height;
    const int channels = restored->channels;
    const float transition = params->highlight_transition <= 1.0e-6f ? 1.0e-6f : params->highlight_transition;
    const float detail_remove = 1.0f - params->highlight_detail_strength;
    const float luma_strength = clamp_strength(params->luminance_transfer_strength);
    const float luma_remove = 1.0f - luma_strength;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const float* rest_row = (const float*)((const unsigned char*)restored->data + (size_t)y * restored->stride_bytes);
        const float* ref_row = (const float*)((const unsigned char*)reference->data + (size_t)y * reference->stride_bytes);
        float* out_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);
        for (int x = 0; x < width; ++x) {
            float alpha = 0.0f;
            if (params->use_highlight_protection) {
                float luminance = ref_row[x * channels];
                if (channels == 3) {
                    const float g = ref_row[x * channels + 1];
                    const float b = ref_row[x * channels + 2];
                    luminance = luminance > g ? luminance : g;
                    luminance = luminance > b ? luminance : b;
                }
                const float mask = smoothstep01((luminance - params->highlight_threshold) / transition);
                alpha = mask * detail_remove;
            }
            if (channels == 1) {
                const float restored_v = rest_row[x];
                const float low_diff_v = sample_bilinear_lowres(low_diff, width, height, x, y, 0) * luma_strength;
                if (params->use_highlight_protection) {
                    const float low_restored_v = sample_bilinear_lowres(low_restored, width, height, x, y, 0);
                    out_row[x] = restored_v + low_diff_v - alpha * (restored_v - low_restored_v);
                } else {
                    out_row[x] = restored_v + low_diff_v;
                }
            } else {
                float diff_v[3];
                diff_v[0] = sample_bilinear_lowres(low_diff, width, height, x, y, 0);
                diff_v[1] = sample_bilinear_lowres(low_diff, width, height, x, y, 1);
                diff_v[2] = sample_bilinear_lowres(low_diff, width, height, x, y, 2);
                const float lum = 0.2126f * diff_v[0] + 0.7152f * diff_v[1] + 0.0722f * diff_v[2];
                const int pix = x * 3;
                for (int c = 0; c < 3; ++c) {
                    const float restored_v = rest_row[pix + c];
                    const float adjusted_diff = diff_v[c] - lum * luma_remove;
                    if (params->use_highlight_protection) {
                        const float low_restored_v = sample_bilinear_lowres(low_restored, width, height, x, y, c);
                        out_row[pix + c] = restored_v + adjusted_diff - alpha * (restored_v - low_restored_v);
                    } else {
                        out_row[pix + c] = restored_v + adjusted_diff;
                    }
                }
            }
        }
    }

    return LOW_FREQUENCY_TRANSFER_OK;
}
