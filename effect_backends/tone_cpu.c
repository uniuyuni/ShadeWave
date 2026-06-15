#include "tone_capi.h"

#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    const ToneConstImageF32* input;
    ToneImageF32* output;
    const float* y_orig;
    const float* y_in;
    float* y_blur;
    float* y_out;
    ToneParams params;
    float max_val;
    int y_begin;
    int y_end;
    int mode;
} ToneWorkerArgs;

enum {
    TONE_MODE_LUMINANCE_MID_SHADOW = 1,
    TONE_MODE_HIGH_POS_BLACK = 2,
    TONE_MODE_HIGH_NEG_BLACK = 3,
    TONE_MODE_WHITE_POS_FINAL = 4,
    TONE_MODE_WHITE_NEG_FINAL = 5,
    TONE_MODE_GAUSS_H = 6,
    TONE_MODE_GAUSS_V = 7,
};

static int tone_worker_count(int height, int width) {
    long detected = sysconf(_SC_NPROCESSORS_ONLN);
    int workers = detected > 0 ? (int)detected : 1;
    if (height * width < 65536) {
        workers = 1;
    }
    if (workers > height) {
        workers = height;
    }
    return workers < 1 ? 1 : workers;
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

static void gaussian_kernel3(float sigma, float kernel[3]) {
    if (sigma <= 0.0f) {
        kernel[0] = 0.25f;
        kernel[1] = 0.50f;
        kernel[2] = 0.25f;
        return;
    }
    const float side = expf(-1.0f / (2.0f * sigma * sigma));
    const float sum = side + 1.0f + side;
    kernel[0] = side / sum;
    kernel[1] = 1.0f / sum;
    kernel[2] = side / sum;
}

static float apply_midtones(float val, float midtone) {
    if (midtone == 0.0f) {
        return val;
    }
    if (midtone > 0.0f) {
        const float c = midtone / 100.0f * 16.0f;
        return logf(1.0f + val * c) / logf(1.0f + c);
    }

    const float c = -midtone / 100.0f * 16.0f;
    if (fabsf(c) < 1.0e-6f) {
        return val;
    }
    const float log1pc = logf(1.0f + c);
    const float normal_result = (expf(val * log1pc) - 1.0f) / c;
    const float derivative_at_1 = (1.0f + c) * log1pc / c;
    if (val <= 1.0f) {
        return normal_result;
    }
    return 1.0f + derivative_at_1 * (val - 1.0f);
}

static float apply_shadows(float val, float shadows) {
    if (shadows == 0.0f) {
        return val;
    }
    if (shadows > 0.0f) {
        const float factor = shadows / 100.0f * 6.0f;
        const float influence = expf(-5.0f * val);
        return val * (1.0f + factor * influence);
    }

    const float factor = shadows / 100.0f;
    const float influence = expf(-5.0f * val);
    const float raw_result = val * (1.0f + factor * influence);
    return fmaxf(raw_result, val * 0.1f);
}

static float apply_black(float val, float black_level) {
    if (black_level == 0.0f) {
        return val;
    }
    const float gamma = black_level > 0.0f
        ? expf(-(black_level / 100.0f) * 0.7f)
        : expf((-black_level / 100.0f) * 0.7f);
    return powf(fmaxf(val, 0.0f), gamma);
}

static float apply_highlight_pos(float val, float highlights) {
    return val * (1.0f + highlights / 100.0f * 2.0f);
}

static float apply_highlight_neg(float val, float base, float highlights) {
    const float factor = -highlights / 100.0f;
    const float detail = val - base;
    const float compressed_base = base / (1.0f + factor * fmaxf(base, 0.0f));
    float t = (base - 0.95f) / 0.4f;
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    const float smooth_mask = t * t * (3.0f - 2.0f * t);
    const float adaptive_factor = 1.0f / (1.0f + 10.0f * fabsf(detail));
    const float effective_boost = 1.17f * adaptive_factor;
    const float desired_boost = 1.0f + smooth_mask * factor * (effective_boost - 1.0f);
    return compressed_base + detail * desired_boost;
}

static float apply_white_pos(float val, float white_level, float max_val) {
    const float factor = white_level / 100.0f * 6.0f;
    const float base = max_val <= 1.0e-6f ? val : val / max_val;
    const float numer = logf(1.0f + logf(1.0f + base));
    const float denom = logf(1.0f + logf(1.0f + fmaxf(max_val, 2.0f)));
    const float denominator = denom == 0.0f ? 1.0f : 1.0f / denom;
    return val * (1.0f + factor * (numer * denominator));
}

static float apply_white_neg(float val, float base, float white_level, float max_val) {
    const float factor = -white_level / 100.0f;
    const float detail = val - base;
    const float safe_base = fmaxf(base, 0.0f);
    const float denom = logf(1.0f + logf(1.0f + fmaxf(max_val, 2.0f)));
    const float denominator = denom == 0.0f ? 1.0f : 1.0f / denom;
    const float target = logf(1.0f + logf(1.0f + safe_base)) * denominator;
    const float compressed_base = fminf(safe_base, base * (1.0f - factor) + target * factor);
    float t = (base - 0.95f) / 0.4f;
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    const float smooth_mask = t * t * (3.0f - 2.0f * t);
    const float adaptive_factor = 1.0f / (1.0f + 10.0f * fabsf(detail));
    const float desired_boost = 1.0f + smooth_mask * factor * (1.17f * adaptive_factor - 1.0f);
    const float safe_boost = detail < 0.0f
        ? fminf(desired_boost, compressed_base / fmaxf(-detail, 1.0e-8f))
        : desired_boost;
    return fmaxf(compressed_base + detail * safe_boost, 0.0f);
}

static void tone_luminance_mid_shadow_rows(const ToneWorkerArgs* args) {
    const ToneConstImageF32* input = args->input;
    const ToneParams* p = &args->params;
    const int width = input->width;
    for (int y = args->y_begin; y < args->y_end; ++y) {
        const float* src_row = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
        float* y_orig_row = args->y_out + (size_t)y * width;
        float* stage_row = args->y_blur + (size_t)y * width;
        for (int x = 0; x < width; ++x) {
            const int base = x * 3;
            const float lum = 0.2126f * src_row[base + 0]
                + 0.7152f * src_row[base + 1]
                + 0.0722f * src_row[base + 2];
            y_orig_row[x] = lum;
            stage_row[x] = apply_shadows(apply_midtones(lum, p->midtone), p->shadows);
        }
    }
}

static void tone_high_pos_black_rows(const ToneWorkerArgs* args) {
    const int width = args->input->width;
    const ToneParams* p = &args->params;
    for (int y = args->y_begin; y < args->y_end; ++y) {
        const float* in_row = args->y_in + (size_t)y * width;
        float* out_row = args->y_out + (size_t)y * width;
        for (int x = 0; x < width; ++x) {
            out_row[x] = apply_black(apply_highlight_pos(in_row[x], p->highlights), p->black_level);
        }
    }
}

static void tone_high_neg_black_rows(const ToneWorkerArgs* args) {
    const int width = args->input->width;
    const ToneParams* p = &args->params;
    for (int y = args->y_begin; y < args->y_end; ++y) {
        const float* in_row = args->y_in + (size_t)y * width;
        const float* blur_row = args->y_blur + (size_t)y * width;
        float* out_row = args->y_out + (size_t)y * width;
        for (int x = 0; x < width; ++x) {
            out_row[x] = apply_black(apply_highlight_neg(in_row[x], blur_row[x], p->highlights), p->black_level);
        }
    }
}

static void tone_white_pos_final_rows(const ToneWorkerArgs* args) {
    const ToneConstImageF32* input = args->input;
    ToneImageF32* output = args->output;
    const int width = input->width;
    const ToneParams* p = &args->params;
    for (int y = args->y_begin; y < args->y_end; ++y) {
        const float* src_row = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
        float* dst_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);
        const float* cur_row = args->y_in + (size_t)y * width;
        const float* orig_row = args->y_orig + (size_t)y * width;
        for (int x = 0; x < width; ++x) {
            const float val = apply_white_pos(cur_row[x], p->white_level, args->max_val);
            const float orig = orig_row[x];
            float gain = val / (orig >= 1.0e-6f ? orig : 1.0e-6f);
            if (orig < 1.0e-6f) {
                gain = 1.0f;
            }
            const int base = x * 3;
            dst_row[base + 0] = src_row[base + 0] * gain;
            dst_row[base + 1] = src_row[base + 1] * gain;
            dst_row[base + 2] = src_row[base + 2] * gain;
        }
    }
}

static void tone_white_neg_final_rows(const ToneWorkerArgs* args) {
    const ToneConstImageF32* input = args->input;
    ToneImageF32* output = args->output;
    const int width = input->width;
    const ToneParams* p = &args->params;
    for (int y = args->y_begin; y < args->y_end; ++y) {
        const float* src_row = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
        float* dst_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);
        const float* cur_row = args->y_in + (size_t)y * width;
        const float* blur_row = args->y_blur + (size_t)y * width;
        const float* orig_row = args->y_orig + (size_t)y * width;
        for (int x = 0; x < width; ++x) {
            const float val = apply_white_neg(cur_row[x], blur_row[x], p->white_level, args->max_val);
            const float orig = orig_row[x];
            float gain = val / (orig >= 1.0e-6f ? orig : 1.0e-6f);
            if (orig < 1.0e-6f) {
                gain = 1.0f;
            }
            const int base = x * 3;
            dst_row[base + 0] = src_row[base + 0] * gain;
            dst_row[base + 1] = src_row[base + 1] * gain;
            dst_row[base + 2] = src_row[base + 2] * gain;
        }
    }
}

static void gaussian_h_rows(const ToneWorkerArgs* args) {
    const int width = args->input->width;
    const float* src = args->y_in;
    float* dst = args->y_out;
    float kernel[3];
    gaussian_kernel3(args->params.resolution_scale * 0.5f, kernel);

    for (int y = args->y_begin; y < args->y_end; ++y) {
        for (int x = 0; x < width; ++x) {
            const int xm = reflect101(x - 1, width);
            const int xp = reflect101(x + 1, width);
            dst[(size_t)y * width + x] =
                src[(size_t)y * width + xm] * kernel[0]
                + src[(size_t)y * width + x] * kernel[1]
                + src[(size_t)y * width + xp] * kernel[2];
        }
    }
}

static void gaussian_v_rows(const ToneWorkerArgs* args) {
    const int width = args->input->width;
    const int height = args->input->height;
    const float* src = args->y_in;
    float* dst = args->y_out;
    float kernel[3];
    gaussian_kernel3(args->params.resolution_scale * 0.5f, kernel);

    for (int y = args->y_begin; y < args->y_end; ++y) {
        const int ym = reflect101(y - 1, height);
        const int yp = reflect101(y + 1, height);
        for (int x = 0; x < width; ++x) {
            dst[(size_t)y * width + x] =
                src[(size_t)ym * width + x] * kernel[0]
                + src[(size_t)y * width + x] * kernel[1]
                + src[(size_t)yp * width + x] * kernel[2];
        }
    }
}

static void tone_run_rows(const ToneWorkerArgs* args) {
    switch (args->mode) {
        case TONE_MODE_LUMINANCE_MID_SHADOW:
            tone_luminance_mid_shadow_rows(args);
            return;
        case TONE_MODE_HIGH_POS_BLACK:
            tone_high_pos_black_rows(args);
            return;
        case TONE_MODE_HIGH_NEG_BLACK:
            tone_high_neg_black_rows(args);
            return;
        case TONE_MODE_WHITE_POS_FINAL:
            tone_white_pos_final_rows(args);
            return;
        case TONE_MODE_WHITE_NEG_FINAL:
            tone_white_neg_final_rows(args);
            return;
        case TONE_MODE_GAUSS_H:
            gaussian_h_rows(args);
            return;
        case TONE_MODE_GAUSS_V:
            gaussian_v_rows(args);
            return;
    }
}

static void* tone_worker_main(void* raw_args) {
    tone_run_rows((const ToneWorkerArgs*)raw_args);
    return NULL;
}

static void tone_parallel_rows(ToneWorkerArgs base_args) {
    const int height = base_args.input->height;
    const int workers = tone_worker_count(height, base_args.input->width);
    if (workers <= 1) {
        base_args.y_begin = 0;
        base_args.y_end = height;
        tone_run_rows(&base_args);
        return;
    }

    pthread_t* threads = (pthread_t*)calloc((size_t)workers, sizeof(pthread_t));
    ToneWorkerArgs* args = (ToneWorkerArgs*)calloc((size_t)workers, sizeof(ToneWorkerArgs));
    int* started = (int*)calloc((size_t)workers, sizeof(int));
    if (threads == NULL || args == NULL || started == NULL) {
        free(threads);
        free(args);
        free(started);
        base_args.y_begin = 0;
        base_args.y_end = height;
        tone_run_rows(&base_args);
        return;
    }

    for (int i = 0; i < workers; ++i) {
        args[i] = base_args;
        args[i].y_begin = height * i / workers;
        args[i].y_end = height * (i + 1) / workers;
        if (pthread_create(&threads[i], NULL, tone_worker_main, &args[i]) == 0) {
            started[i] = 1;
        } else {
            tone_run_rows(&args[i]);
        }
    }

    for (int i = 0; i < workers; ++i) {
        if (started[i]) {
            pthread_join(threads[i], NULL);
        }
    }

    free(threads);
    free(args);
    free(started);
}

static void gaussian_blur3(const ToneConstImageF32* shape, const ToneParams* params, const float* src, float* scratch, float* dst) {
    ToneWorkerArgs args;
    memset(&args, 0, sizeof(args));
    args.input = shape;
    args.params = *params;
    args.y_in = src;
    args.y_out = scratch;
    args.mode = TONE_MODE_GAUSS_H;
    tone_parallel_rows(args);

    args.y_in = scratch;
    args.y_out = dst;
    args.mode = TONE_MODE_GAUSS_V;
    tone_parallel_rows(args);
}

static float plane_max(const float* plane, size_t count) {
    float max_val = -FLT_MAX;
    for (size_t i = 0; i < count; ++i) {
        if (plane[i] > max_val) {
            max_val = plane[i];
        }
    }
    return max_val;
}

int tone_adjust_v1(
    const ToneConstImageF32* input,
    ToneImageF32* output,
    const ToneParams* params
) {
    if (input == NULL || output == NULL || params == NULL || input->data == NULL || output->data == NULL) {
        return TONE_ERR_NULL;
    }
    if (input->width <= 0 || input->height <= 0 || input->width != output->width || input->height != output->height) {
        return TONE_ERR_SHAPE;
    }
    if (input->channels != 3 || output->channels != 3) {
        return TONE_ERR_SHAPE;
    }

    size_t plane_count = 0;
    if (!checked_plane_size(input->width, input->height, &plane_count)) {
        return TONE_ERR_SHAPE;
    }

    float* y_orig = (float*)malloc(sizeof(float) * plane_count);
    float* stage_a = (float*)malloc(sizeof(float) * plane_count);
    float* stage_b = (float*)malloc(sizeof(float) * plane_count);
    float* blur = (float*)malloc(sizeof(float) * plane_count);
    if (y_orig == NULL || stage_a == NULL || stage_b == NULL || blur == NULL) {
        free(y_orig);
        free(stage_a);
        free(stage_b);
        free(blur);
        return TONE_ERR_ALLOC;
    }

    ToneWorkerArgs args;
    memset(&args, 0, sizeof(args));
    args.input = input;
    args.output = output;
    args.params = *params;
    args.y_out = y_orig;
    args.y_blur = stage_a;
    args.mode = TONE_MODE_LUMINANCE_MID_SHADOW;
    tone_parallel_rows(args);

    args.y_in = stage_a;
    args.y_out = stage_b;
    if (params->highlights < 0.0f) {
        gaussian_blur3(input, params, stage_a, blur, stage_b);
        args.y_in = stage_a;
        args.y_blur = stage_b;
        args.y_out = blur;
        args.mode = TONE_MODE_HIGH_NEG_BLACK;
        tone_parallel_rows(args);
    } else {
        args.mode = TONE_MODE_HIGH_POS_BLACK;
        tone_parallel_rows(args);
    }

    const float* current_y = params->highlights < 0.0f ? blur : stage_b;
    if (params->white_level < 0.0f) {
        gaussian_blur3(input, params, current_y, stage_a, stage_b);
        args.y_orig = y_orig;
        args.y_in = current_y;
        args.y_blur = stage_b;
        args.max_val = plane_max(stage_b, plane_count);
        args.mode = TONE_MODE_WHITE_NEG_FINAL;
        tone_parallel_rows(args);
    } else {
        args.y_orig = y_orig;
        args.y_in = current_y;
        args.max_val = plane_max(current_y, plane_count);
        args.mode = TONE_MODE_WHITE_POS_FINAL;
        tone_parallel_rows(args);
    }

    free(y_orig);
    free(stage_a);
    free(stage_b);
    free(blur);
    return TONE_OK;
}
