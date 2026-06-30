#include "cross_filter_capi.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int x;
    float v;
} MaxNode;

static int checked_mul_size(int a, int b, int c, size_t* out) {
    if (a <= 0 || b <= 0 || c <= 0) {
        return 0;
    }
    const size_t ab = (size_t)a * (size_t)b;
    if (ab / (size_t)a != (size_t)b) {
        return 0;
    }
    const size_t abc = ab * (size_t)c;
    if (abc / ab != (size_t)c) {
        return 0;
    }
    *out = abc;
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

static float bilinear_sample(const float* img, int w, int h, int c, float x, float y) {
    if (x < 0.0f || y < 0.0f || x > (float)(w - 1) || y > (float)(h - 1)) {
        return 0.0f;
    }

    const int x0 = (int)floorf(x);
    const int y0 = (int)floorf(y);
    const int x1 = x0 + 1 < w ? x0 + 1 : x0;
    const int y1 = y0 + 1 < h ? y0 + 1 : y0;
    const float ax = x - (float)x0;
    const float ay = y - (float)y0;

    const float v00 = img[((y0 * w + x0) * 3) + c];
    const float v10 = img[((y0 * w + x1) * 3) + c];
    const float v01 = img[((y1 * w + x0) * 3) + c];
    const float v11 = img[((y1 * w + x1) * 3) + c];
    const float top = v00 + (v10 - v00) * ax;
    const float bottom = v01 + (v11 - v01) * ax;
    return top + (bottom - top) * ay;
}

static void rotate_image(const float* src, float* dst, int w, int h, float angle_deg) {
    const float radians = angle_deg * (float)M_PI / 180.0f;
    const float cs = cosf(radians);
    const float sn = sinf(radians);
    const float cx = (float)(w / 2);
    const float cy = (float)(h / 2);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float dx = (float)x - cx;
            const float dy = (float)y - cy;
            const float sx = cs * dx + sn * dy + cx;
            const float sy = -sn * dx + cs * dy + cy;
            const int base = (y * w + x) * 3;
            dst[base + 0] = bilinear_sample(src, w, h, 0, sx, sy);
            dst[base + 1] = bilinear_sample(src, w, h, 1, sx, sy);
            dst[base + 2] = bilinear_sample(src, w, h, 2, sx, sy);
        }
    }
}

static void horizontal_max_filter(const float* src, float* dst, int w, int h, int radius) {
    MaxNode* deque = (MaxNode*)malloc(sizeof(MaxNode) * (size_t)w);
    if (deque == NULL) {
        memcpy(dst, src, sizeof(float) * (size_t)w * (size_t)h);
        return;
    }

    for (int y = 0; y < h; ++y) {
        int head = 0;
        int tail = 0;
        const float* row = src + (size_t)y * w;
        float* out = dst + (size_t)y * w;

        for (int x = 0; x < w + radius; ++x) {
            const int add_x = x;
            if (add_x < w) {
                const float v = row[add_x];
                while (tail > head && deque[tail - 1].v <= v) {
                    tail--;
                }
                deque[tail].x = add_x;
                deque[tail].v = v;
                tail++;
            }

            const int out_x = x - radius;
            if (out_x >= 0 && out_x < w) {
                const int min_x = out_x - radius;
                while (tail > head && deque[head].x < min_x) {
                    head++;
                }
                out[out_x] = tail > head ? deque[head].v : row[out_x];
            }
        }
    }

    free(deque);
}

static void vertical_max_filter(const float* src, float* dst, int w, int h, int radius) {
    MaxNode* deque = (MaxNode*)malloc(sizeof(MaxNode) * (size_t)h);
    if (deque == NULL) {
        memcpy(dst, src, sizeof(float) * (size_t)w * (size_t)h);
        return;
    }

    for (int x = 0; x < w; ++x) {
        int head = 0;
        int tail = 0;
        for (int y = 0; y < h + radius; ++y) {
            const int add_y = y;
            if (add_y < h) {
                const float v = src[(size_t)add_y * w + x];
                while (tail > head && deque[tail - 1].v <= v) {
                    tail--;
                }
                deque[tail].x = add_y;
                deque[tail].v = v;
                tail++;
            }

            const int out_y = y - radius;
            if (out_y >= 0 && out_y < h) {
                const int min_y = out_y - radius;
                while (tail > head && deque[head].x < min_y) {
                    head++;
                }
                dst[(size_t)out_y * w + x] = tail > head ? deque[head].v : src[(size_t)out_y * w + x];
            }
        }
    }

    free(deque);
}

static void draw_debug_circle(float* dst, int w, int h, int cx, int cy) {
    const int r = 4;
    for (int y = cy - r; y <= cy + r; ++y) {
        if (y < 0 || y >= h) {
            continue;
        }
        for (int x = cx - r; x <= cx + r; ++x) {
            if (x < 0 || x >= w) {
                continue;
            }
            const int dx = x - cx;
            const int dy = y - cy;
            if (dx * dx + dy * dy <= r * r) {
                const int base = (y * w + x) * 3;
                dst[base + 0] = 0.0f;
                dst[base + 1] = 0.0f;
                dst[base + 2] = 10.0f;
            }
        }
    }
}

static float random_gain(float randomness) {
    if (randomness <= 0.0f) {
        return 1.0f;
    }
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    const float unit = (float)rand() / (float)RAND_MAX;
    return (1.0f - randomness) + unit * (2.0f * randomness);
}

static void gaussian_blur(float* img, float* scratch, int w, int h, float sigma) {
    const int ksize = ((int)(sigma * 6.0f)) | 1;
    if (ksize < 3) {
        return;
    }
    const int radius = ksize / 2;
    float* kernel = (float*)malloc(sizeof(float) * (size_t)ksize);
    if (kernel == NULL) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < ksize; ++i) {
        const int x = i - radius;
        const float v = expf(-((float)(x * x)) / (2.0f * sigma * sigma));
        kernel[i] = v;
        sum += v;
    }
    for (int i = 0; i < ksize; ++i) {
        kernel[i] /= sum;
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    const int sx = reflect101(x + k, w);
                    acc += img[((y * w + sx) * 3) + c] * kernel[k + radius];
                }
                scratch[((y * w + x) * 3) + c] = acc;
            }
        }
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    const int sy = reflect101(y + k, h);
                    acc += scratch[((sy * w + x) * 3) + c] * kernel[k + radius];
                }
                img[((y * w + x) * 3) + c] = acc;
            }
        }
    }

    free(kernel);
}

static void horizontal_exponential_filter_channel(
    const float* src,
    float* dst,
    int w,
    int h,
    int channel,
    int length,
    int symmetric
) {
    const int radius = length / 2;
    if (radius < 1) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                dst[((y * w + x) * 3) + channel] = src[((y * w + x) * 3) + channel];
            }
        }
        return;
    }

    const float a = expf(-8.0f / (float)radius);
    float* left = NULL;
    float* right = NULL;
    if (symmetric) {
        left = (float*)malloc(sizeof(float) * (size_t)w);
        right = (float*)malloc(sizeof(float) * (size_t)w);
        if (left == NULL || right == NULL) {
            free(left);
            free(right);
            return;
        }
    }

    for (int y = 0; y < h; ++y) {
        if (symmetric) {
            float acc = 0.0f;
            for (int x = 0; x < w; ++x) {
                const float v = src[((y * w + x) * 3) + channel];
                acc = v + a * acc;
                left[x] = acc;
            }
            acc = 0.0f;
            for (int x = w - 1; x >= 0; --x) {
                const float v = src[((y * w + x) * 3) + channel];
                acc = v + a * acc;
                right[x] = acc;
            }
            for (int x = 0; x < w; ++x) {
                const float center = src[((y * w + x) * 3) + channel];
                dst[((y * w + x) * 3) + channel] = left[x] + right[x] - center;
            }
        } else {
            float acc = 0.0f;
            for (int x = w - 1; x >= 0; --x) {
                const float v = src[((y * w + x) * 3) + channel];
                acc = v + a * acc;
                dst[((y * w + x) * 3) + channel] = acc;
            }
        }
    }

    free(left);
    free(right);
}

static void resize_bilinear(const float* src, int sw, int sh, float* dst, int dw, int dh) {
    const float scale_x = (float)sw / (float)dw;
    const float scale_y = (float)sh / (float)dh;
    for (int y = 0; y < dh; ++y) {
        const float sy = ((float)y + 0.5f) * scale_y - 0.5f;
        for (int x = 0; x < dw; ++x) {
            const float sx = ((float)x + 0.5f) * scale_x - 0.5f;
            const int base = (y * dw + x) * 3;
            dst[base + 0] = bilinear_sample(src, sw, sh, 0, sx, sy);
            dst[base + 1] = bilinear_sample(src, sw, sh, 1, sx, sy);
            dst[base + 2] = bilinear_sample(src, sw, sh, 2, sx, sy);
        }
    }
}

int cross_filter_apply_v1(
    const CrossFilterImageF32* input,
    CrossFilterImageF32* output,
    const CrossFilterParams* params
) {
    if (input == NULL || output == NULL || params == NULL || input->data == NULL || output->data == NULL) {
        return CROSS_FILTER_ERR_NULL;
    }
    if (input->width <= 0 || input->height <= 0 || input->channels != 3 ||
        output->width != input->width || output->height != input->height || output->channels != 3) {
        return CROSS_FILTER_ERR_SHAPE;
    }

    const int w = input->width;
    const int h = input->height;
    int speed_factor = params->speed_factor <= 0 ? 1 : params->speed_factor;
    int sh = h / speed_factor;
    int sw = w / speed_factor;
    if (sh < 1 || sw < 1) {
        sh = h;
        sw = w;
        speed_factor = 1;
    }

    size_t full_pixels = 0;
    size_t mini_pixels = 0;
    if (!checked_mul_size(w, h, 1, &full_pixels) || !checked_mul_size(sw, sh, 3, &mini_pixels)) {
        return CROSS_FILTER_ERR_ALLOC;
    }

    float* luminance = (float*)malloc(sizeof(float) * full_pixels);
    float* max_tmp = (float*)malloc(sizeof(float) * full_pixels);
    float* dilated = (float*)malloc(sizeof(float) * full_pixels);
    float* impulse_mini = (float*)calloc(mini_pixels, sizeof(float));
    if (luminance == NULL || max_tmp == NULL || dilated == NULL || impulse_mini == NULL) {
        free(luminance);
        free(max_tmp);
        free(dilated);
        free(impulse_mini);
        return CROSS_FILTER_ERR_ALLOC;
    }

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        const float* row = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
        float* out_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);
        for (int x = 0; x < w; ++x) {
            const int base = x * 3;
            const float r = row[base + 0];
            const float g = row[base + 1];
            const float b = row[base + 2];
            luminance[(size_t)y * w + x] = r * 0.299f + g * 0.587f + b * 0.114f;
            out_row[base + 0] = r;
            out_row[base + 1] = g;
            out_row[base + 2] = b;
        }
    }

    const int radius = params->min_distance < 0 ? 0 : params->min_distance;
    horizontal_max_filter(luminance, max_tmp, w, h, radius);
    vertical_max_filter(max_tmp, dilated, w, h, radius);

    int peak_count = 0;
    for (int y = 0; y < h; ++y) {
        const float* row = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
        for (int x = 0; x < w; ++x) {
            const size_t idx = (size_t)y * w + x;
            const float lum = luminance[idx];
            if (lum == dilated[idx] && lum > params->threshold) {
                ++peak_count;
                if (params->debug_mode) {
                    draw_debug_circle(output->data, w, h, x, y);
                } else {
                    const int sx = x / speed_factor;
                    const int sy = y / speed_factor;
                    if (sx >= 0 && sx < sw && sy >= 0 && sy < sh) {
                        const float boost = (float)speed_factor * 1.5f;
                        const float gain = random_gain(params->randomness);
                        const int mini_base = (sy * sw + sx) * 3;
                        const int base = x * 3;
                        impulse_mini[mini_base + 0] = row[base + 0] * gain * boost;
                        impulse_mini[mini_base + 1] = row[base + 1] * gain * boost;
                        impulse_mini[mini_base + 2] = row[base + 2] * gain * boost;
                    }
                }
            }
        }
    }

    free(luminance);
    free(max_tmp);
    free(dilated);

    if (params->debug_mode || peak_count == 0) {
        free(impulse_mini);
        return CROSS_FILTER_OK;
    }

    float* mini_scratch = (float*)malloc(sizeof(float) * mini_pixels);
    if (mini_scratch == NULL) {
        free(impulse_mini);
        return CROSS_FILTER_ERR_ALLOC;
    }
    if (params->line_thickness > 1.0f) {
        const float sigma = (params->line_thickness - 1.0f) * 0.5f;
        gaussian_blur(impulse_mini, mini_scratch, sw, sh, sigma);
    }

    int mini_length = params->length / speed_factor;
    if (mini_length < 1) {
        mini_length = 1;
    }
    const int pad_len = (int)((float)mini_length * 1.5f);
    const int pw = sw + pad_len * 2;
    const int ph = sh + pad_len * 2;
    size_t padded_values = 0;
    if (!checked_mul_size(pw, ph, 3, &padded_values)) {
        free(impulse_mini);
        free(mini_scratch);
        return CROSS_FILTER_ERR_ALLOC;
    }

    float* impulse_padded = (float*)calloc(padded_values, sizeof(float));
    float* rotated = (float*)calloc(padded_values, sizeof(float));
    float* filtered = (float*)calloc(padded_values, sizeof(float));
    float* unrotated = (float*)calloc(padded_values, sizeof(float));
    float* accumulated = (float*)calloc(padded_values, sizeof(float));
    if (impulse_padded == NULL || rotated == NULL || filtered == NULL || unrotated == NULL || accumulated == NULL) {
        free(impulse_mini);
        free(mini_scratch);
        free(impulse_padded);
        free(rotated);
        free(filtered);
        free(unrotated);
        free(accumulated);
        return CROSS_FILTER_ERR_ALLOC;
    }

    for (int y = 0; y < sh; ++y) {
        memcpy(
            impulse_padded + (((y + pad_len) * pw + pad_len) * 3),
            impulse_mini + ((y * sw) * 3),
            sizeof(float) * (size_t)sw * 3
        );
    }

    int base_k_len = mini_length;
    if (base_k_len % 2 == 0) {
        base_k_len += 1;
    }
    int num_passes;
    float rot_step;
    int use_symmetric;
    if (params->num_points % 2 == 0) {
        num_passes = params->num_points / 2;
        rot_step = num_passes > 0 ? 180.0f / (float)num_passes : 0.0f;
        use_symmetric = 1;
    } else {
        num_passes = params->num_points;
        rot_step = 360.0f / (float)num_passes;
        use_symmetric = 0;
    }
    if (num_passes == 0) {
        num_passes = 1;
    }

    const float spectral_scales[3] = {
        1.0f + params->spectral_strength,
        1.0f,
        1.0f - params->spectral_strength,
    };

    for (int pass = 0; pass < num_passes; ++pass) {
        const float current_angle = params->angle_deg + (float)pass * rot_step;
        rotate_image(impulse_padded, rotated, pw, ph, current_angle);
        memset(filtered, 0, sizeof(float) * padded_values);

        for (int ch = 0; ch < 3; ++ch) {
            int ch_len = (int)((float)base_k_len * spectral_scales[ch]);
            if (ch_len < 1) {
                ch_len = 1;
            }
            if (ch_len % 2 == 0) {
                ch_len += 1;
            }
            horizontal_exponential_filter_channel(rotated, filtered, pw, ph, ch, ch_len, use_symmetric);
        }

        rotate_image(filtered, unrotated, pw, ph, -current_angle);
        for (size_t i = 0; i < padded_values; ++i) {
            accumulated[i] += unrotated[i];
        }
    }

    float* streaks_mini = mini_scratch;
    for (int y = 0; y < sh; ++y) {
        memcpy(
            streaks_mini + ((y * sw) * 3),
            accumulated + (((y + pad_len) * pw + pad_len) * 3),
            sizeof(float) * (size_t)sw * 3
        );
    }

    float* streaks_full = (float*)malloc(sizeof(float) * full_pixels * 3);
    if (streaks_full == NULL) {
        free(impulse_mini);
        free(mini_scratch);
        free(impulse_padded);
        free(rotated);
        free(filtered);
        free(unrotated);
        free(accumulated);
        return CROSS_FILTER_ERR_ALLOC;
    }
    resize_bilinear(streaks_mini, sw, sh, streaks_full, w, h);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        const float* src_row = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
        float* out_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);
        for (int x = 0; x < w; ++x) {
            const int base = x * 3;
            const int full_base = (y * w + x) * 3;
            out_row[base + 0] = src_row[base + 0] + streaks_full[full_base + 0] * params->intensity;
            out_row[base + 1] = src_row[base + 1] + streaks_full[full_base + 1] * params->intensity;
            out_row[base + 2] = src_row[base + 2] + streaks_full[full_base + 2] * params->intensity;
        }
    }

    free(impulse_mini);
    free(mini_scratch);
    free(impulse_padded);
    free(rotated);
    free(filtered);
    free(unrotated);
    free(accumulated);
    free(streaks_full);
    return CROSS_FILTER_OK;
}
