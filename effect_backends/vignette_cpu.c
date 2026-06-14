#include "vignette_capi.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct {
    const VignetteImageF32* input;
    VignetteImageF32* output;
    VignetteParams params;
    int y_begin;
    int y_end;
} VignetteWorkerArgs;

static int vignette_worker_count(int height, int width) {
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

static void vignette_apply_rows(const VignetteWorkerArgs* args) {
    const VignetteImageF32* input = args->input;
    VignetteImageF32* output = args->output;
    const VignetteParams* p = &args->params;

    const float intensity = p->intensity / 100.0f;
    const float radius_percent = p->radius_percent / 100.0f;
    const float gradient_softness = fmaxf(0.1f, p->gradient_softness);

    const float dx = p->disp_info[0];
    const float dy = p->disp_info[1];
    const float scale = p->disp_info[4];
    const float x1 = p->crop_rect[0];
    const float y1 = p->crop_rect[1];
    const float x2 = p->crop_rect[2];
    const float y2 = p->crop_rect[3];
    const float offset_x = p->offset[0];
    const float offset_y = p->offset[1];

    const float center_x = (x1 + (x2 - x1) / 2.0f - dx) * scale + offset_x;
    const float center_y = (y1 + (y2 - y1) / 2.0f - dy) * scale + offset_y;
    const float mm_x = x2 - x1;
    const float mm_y = y2 - y1;
    const float mm = fmaxf(mm_x, mm_y) * scale;
    const float max_radius = sqrtf(mm * mm + mm * mm) / 2.0f;
    const float radius = max_radius * radius_percent;
    const int channels = input->channels;

    for (int y = args->y_begin; y < args->y_end; ++y) {
        const float* src_row = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
        float* dst_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);

        for (int x = 0; x < input->width; ++x) {
            const float x_delta = (float)x - center_x;
            const float y_delta = (float)y - center_y;
            float val;
            if (radius == 0.0f) {
                val = 1.0f;
            } else {
                val = sqrtf(x_delta * x_delta + y_delta * y_delta) / radius;
            }

            if (val > 1.0f) {
                val = 1.0f;
            } else if (val < 0.0f) {
                val = 0.0f;
            }

            float mask = powf(val, gradient_softness);
            mask = mask * mask * (3.0f - 2.0f * mask);

            const int base = x * channels;
            if (intensity < 0.0f) {
                const float vig = 1.0f + intensity * mask;
                for (int c = 0; c < channels; ++c) {
                    dst_row[base + c] = src_row[base + c] * vig;
                }
            } else {
                const float vig = 1.0f - intensity * mask;
                for (int c = 0; c < channels; ++c) {
                    const float v = src_row[base + c];
                    dst_row[base + c] = v + (1.0f - v) * (1.0f - vig);
                }
            }
        }
    }
}

static void* vignette_worker_main(void* raw_args) {
    vignette_apply_rows((const VignetteWorkerArgs*)raw_args);
    return NULL;
}

int vignette_apply_v1(
    const VignetteImageF32* input,
    VignetteImageF32* output,
    const VignetteParams* params
) {
    if (input == NULL || output == NULL || params == NULL || input->data == NULL || output->data == NULL) {
        return VIGNETTE_ERR_NULL;
    }
    if (input->width <= 0 || input->height <= 0 || input->width != output->width || input->height != output->height) {
        return VIGNETTE_ERR_SHAPE;
    }
    if (input->channels != output->channels || (input->channels != 1 && input->channels != 3)) {
        return VIGNETTE_ERR_SHAPE;
    }

    const int workers = vignette_worker_count(input->height, input->width);
    if (workers <= 1) {
        VignetteWorkerArgs args = {input, output, *params, 0, input->height};
        vignette_apply_rows(&args);
        return VIGNETTE_OK;
    }

    pthread_t* threads = (pthread_t*)calloc((size_t)workers, sizeof(pthread_t));
    VignetteWorkerArgs* args = (VignetteWorkerArgs*)calloc((size_t)workers, sizeof(VignetteWorkerArgs));
    int* started = (int*)calloc((size_t)workers, sizeof(int));
    if (threads == NULL || args == NULL || started == NULL) {
        free(threads);
        free(args);
        free(started);
        VignetteWorkerArgs single_args = {input, output, *params, 0, input->height};
        vignette_apply_rows(&single_args);
        return VIGNETTE_OK;
    }

    for (int i = 0; i < workers; ++i) {
        const int y_begin = input->height * i / workers;
        const int y_end = input->height * (i + 1) / workers;
        args[i].input = input;
        args[i].output = output;
        args[i].params = *params;
        args[i].y_begin = y_begin;
        args[i].y_end = y_end;
        if (pthread_create(&threads[i], NULL, vignette_worker_main, &args[i]) == 0) {
            started[i] = 1;
        } else {
            vignette_apply_rows(&args[i]);
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
    return VIGNETTE_OK;
}
