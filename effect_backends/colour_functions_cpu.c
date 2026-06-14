#include "colour_functions_capi.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct {
    const ColourFunctionsConstImageF32* input;
    ColourFunctionsImageF32* output;
    ColourFunctionsParams params;
    int y_begin;
    int y_end;
} ColourFunctionsWorkerArgs;

static int colour_functions_worker_count(int height, int width) {
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

static float colour_functions_encode_gamma(float v, float gamma) {
    const float magnitude = powf(fabsf(v), 1.0f / gamma);
    return v < 0.0f ? -magnitude : magnitude;
}

static float colour_functions_encode_value(float v, ColourFunctionsEncoding encoding) {
    switch (encoding) {
        case COLOUR_FUNCTIONS_ENCODING_LINEAR:
            return v;
        case COLOUR_FUNCTIONS_ENCODING_SRGB:
            return v <= 0.0031308f ? 12.92f * v : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
        case COLOUR_FUNCTIONS_ENCODING_REC709:
            return v < 0.018f ? 4.5f * v : 1.099f * powf(v, 0.45f) - 0.099f;
        case COLOUR_FUNCTIONS_ENCODING_REC2020: {
            const float alpha = 1.09929682680944f;
            const float beta = 0.018053968510807f;
            return v < beta ? 4.5f * v : alpha * powf(v, 0.45f) - (alpha - 1.0f);
        }
        case COLOUR_FUNCTIONS_ENCODING_GAMMA_ADOBE_RGB:
            return colour_functions_encode_gamma(v, 563.0f / 256.0f);
        case COLOUR_FUNCTIONS_ENCODING_GAMMA_1_8:
            return colour_functions_encode_gamma(v, 1.8f);
        case COLOUR_FUNCTIONS_ENCODING_GAMMA_2_2:
            return colour_functions_encode_gamma(v, 2.2f);
        case COLOUR_FUNCTIONS_ENCODING_GAMMA_2_6:
            return colour_functions_encode_gamma(v, 2.6f);
        case COLOUR_FUNCTIONS_ENCODING_PROPHOTO:
            return v < (1.0f / 512.0f) ? 16.0f * v : powf(v, 1.0f / 1.8f);
    }
    return v;
}

static void colour_functions_compress_negative(
    float* r,
    float* g,
    float* b,
    const ColourFunctionsParams* p
) {
    if (*r >= 0.0f && *g >= 0.0f && *b >= 0.0f) {
        return;
    }

    const float lum = p->luminance_weights[0] * *r
        + p->luminance_weights[1] * *g
        + p->luminance_weights[2] * *b;

    if (lum > p->eps) {
        float scale = 1.0f;
        if (*r < 0.0f) {
            const float denom = fmaxf(lum - *r, p->eps);
            scale = fminf(scale, lum / denom);
        }
        if (*g < 0.0f) {
            const float denom = fmaxf(lum - *g, p->eps);
            scale = fminf(scale, lum / denom);
        }
        if (*b < 0.0f) {
            const float denom = fmaxf(lum - *b, p->eps);
            scale = fminf(scale, lum / denom);
        }
        scale = fminf(fmaxf(scale, 0.0f), 1.0f);
        *r = lum + scale * (*r - lum);
        *g = lum + scale * (*g - lum);
        *b = lum + scale * (*b - lum);
    }

    *r = fmaxf(*r, 0.0f);
    *g = fmaxf(*g, 0.0f);
    *b = fmaxf(*b, 0.0f);
}

static void colour_functions_transform_rows(const ColourFunctionsWorkerArgs* args) {
    const ColourFunctionsConstImageF32* input = args->input;
    ColourFunctionsImageF32* output = args->output;
    const ColourFunctionsParams* p = &args->params;
    const float* m = p->basis;

    for (int y = args->y_begin; y < args->y_end; ++y) {
        const float* src_row = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
        float* dst_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);

        for (int x = 0; x < input->width; ++x) {
            const int base = x * 3;
            const float r = src_row[base + 0];
            const float g = src_row[base + 1];
            const float b = src_row[base + 2];

            float out_r = r * m[0] + g * m[3] + b * m[6];
            float out_g = r * m[1] + g * m[4] + b * m[7];
            float out_b = r * m[2] + g * m[5] + b * m[8];

            colour_functions_compress_negative(&out_r, &out_g, &out_b, p);

            dst_row[base + 0] = colour_functions_encode_value(out_r, p->encoding);
            dst_row[base + 1] = colour_functions_encode_value(out_g, p->encoding);
            dst_row[base + 2] = colour_functions_encode_value(out_b, p->encoding);
        }
    }
}

static void* colour_functions_worker_main(void* raw_args) {
    colour_functions_transform_rows((const ColourFunctionsWorkerArgs*)raw_args);
    return NULL;
}

static int colour_functions_encoding_valid(ColourFunctionsEncoding encoding) {
    return encoding >= COLOUR_FUNCTIONS_ENCODING_LINEAR && encoding <= COLOUR_FUNCTIONS_ENCODING_PROPHOTO;
}

int colour_functions_transform_v1(
    const ColourFunctionsConstImageF32* input,
    ColourFunctionsImageF32* output,
    const ColourFunctionsParams* params
) {
    if (input == NULL || output == NULL || params == NULL || input->data == NULL || output->data == NULL) {
        return COLOUR_FUNCTIONS_ERR_NULL;
    }
    if (input->width <= 0 || input->height <= 0 || input->width != output->width || input->height != output->height) {
        return COLOUR_FUNCTIONS_ERR_SHAPE;
    }
    if (input->channels != 3 || output->channels != 3) {
        return COLOUR_FUNCTIONS_ERR_SHAPE;
    }
    if (!colour_functions_encoding_valid(params->encoding)) {
        return COLOUR_FUNCTIONS_ERR_ENCODING;
    }

    const int workers = colour_functions_worker_count(input->height, input->width);
    if (workers <= 1) {
        ColourFunctionsWorkerArgs args = {input, output, *params, 0, input->height};
        colour_functions_transform_rows(&args);
        return COLOUR_FUNCTIONS_OK;
    }

    pthread_t* threads = (pthread_t*)calloc((size_t)workers, sizeof(pthread_t));
    ColourFunctionsWorkerArgs* args = (ColourFunctionsWorkerArgs*)calloc((size_t)workers, sizeof(ColourFunctionsWorkerArgs));
    int* started = (int*)calloc((size_t)workers, sizeof(int));
    if (threads == NULL || args == NULL || started == NULL) {
        free(threads);
        free(args);
        free(started);
        ColourFunctionsWorkerArgs single_args = {input, output, *params, 0, input->height};
        colour_functions_transform_rows(&single_args);
        return COLOUR_FUNCTIONS_OK;
    }

    for (int i = 0; i < workers; ++i) {
        const int y_begin = input->height * i / workers;
        const int y_end = input->height * (i + 1) / workers;
        args[i].input = input;
        args[i].output = output;
        args[i].params = *params;
        args[i].y_begin = y_begin;
        args[i].y_end = y_end;
        if (pthread_create(&threads[i], NULL, colour_functions_worker_main, &args[i]) == 0) {
            started[i] = 1;
        } else {
            colour_functions_transform_rows(&args[i]);
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
    return COLOUR_FUNCTIONS_OK;
}
