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

typedef struct {
    VignetteImageF32* output;
    VignetteParams params;
    int y_begin;
    int y_end;
} VignetteMaskWorkerArgs;

typedef struct {
    const VignetteImageF32* input;
    const VignetteImageF32* mask;
    VignetteImageF32* output;
    float intensity;
    int y_begin;
    int y_end;
} VignetteApplyMaskWorkerArgs;

typedef struct {
    float center_x;
    float center_y;
    float radius;
    float gradient_softness;
} VignetteMaskGeometry;

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

static VignetteMaskGeometry vignette_mask_geometry(const VignetteParams* p) {
    const float radius_percent = p->radius_percent / 100.0f;

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
    VignetteMaskGeometry geometry;
    geometry.center_x = center_x;
    geometry.center_y = center_y;
    geometry.radius = max_radius * radius_percent;
    geometry.gradient_softness = fmaxf(0.1f, p->gradient_softness);
    return geometry;
}

static float vignette_mask_value(const VignetteMaskGeometry* geometry, int x, int y) {
    const float x_delta = (float)x - geometry->center_x;
    const float y_delta = (float)y - geometry->center_y;
    float val = geometry->radius == 0.0f ? 1.0f : sqrtf(x_delta * x_delta + y_delta * y_delta) / geometry->radius;

    if (val > 1.0f) {
        val = 1.0f;
    } else if (val < 0.0f) {
        val = 0.0f;
    }

    float mask = powf(val, geometry->gradient_softness);
    return mask * mask * (3.0f - 2.0f * mask);
}

static void vignette_create_mask_rows(const VignetteMaskWorkerArgs* args) {
    VignetteImageF32* output = args->output;
    const VignetteMaskGeometry geometry = vignette_mask_geometry(&args->params);

    for (int y = args->y_begin; y < args->y_end; ++y) {
        float* dst_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);
        for (int x = 0; x < output->width; ++x) {
            dst_row[x] = vignette_mask_value(&geometry, x, y);
        }
    }
}

static void vignette_apply_mask_rows(const VignetteApplyMaskWorkerArgs* args) {
    const VignetteImageF32* input = args->input;
    const VignetteImageF32* mask = args->mask;
    VignetteImageF32* output = args->output;
    const float intensity = args->intensity / 100.0f;
    const int channels = input->channels;

    for (int y = args->y_begin; y < args->y_end; ++y) {
        const float* src_row = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
        const float* mask_row = (const float*)((const unsigned char*)mask->data + (size_t)y * mask->stride_bytes);
        float* dst_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);
        for (int x = 0; x < input->width; ++x) {
            const float mask_value = mask_row[x];
            const int base = x * channels;
            if (intensity < 0.0f) {
                const float vig = 1.0f + intensity * mask_value;
                for (int c = 0; c < channels; ++c) {
                    dst_row[base + c] = src_row[base + c] * vig;
                }
            } else {
                const float vig = 1.0f - intensity * mask_value;
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

static void* vignette_mask_worker_main(void* raw_args) {
    vignette_create_mask_rows((const VignetteMaskWorkerArgs*)raw_args);
    return NULL;
}

static void* vignette_apply_mask_worker_main(void* raw_args) {
    vignette_apply_mask_rows((const VignetteApplyMaskWorkerArgs*)raw_args);
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

int vignette_create_mask_v1(
    VignetteImageF32* output,
    const VignetteParams* params
) {
    if (output == NULL || params == NULL || output->data == NULL) {
        return VIGNETTE_ERR_NULL;
    }
    if (output->width <= 0 || output->height <= 0 || output->channels != 1) {
        return VIGNETTE_ERR_SHAPE;
    }

    const int workers = vignette_worker_count(output->height, output->width);
    if (workers <= 1) {
        VignetteMaskWorkerArgs args = {output, *params, 0, output->height};
        vignette_create_mask_rows(&args);
        return VIGNETTE_OK;
    }

    pthread_t* threads = (pthread_t*)calloc((size_t)workers, sizeof(pthread_t));
    VignetteMaskWorkerArgs* args = (VignetteMaskWorkerArgs*)calloc((size_t)workers, sizeof(VignetteMaskWorkerArgs));
    int* started = (int*)calloc((size_t)workers, sizeof(int));
    if (threads == NULL || args == NULL || started == NULL) {
        free(threads);
        free(args);
        free(started);
        VignetteMaskWorkerArgs single_args = {output, *params, 0, output->height};
        vignette_create_mask_rows(&single_args);
        return VIGNETTE_OK;
    }

    for (int i = 0; i < workers; ++i) {
        const int y_begin = output->height * i / workers;
        const int y_end = output->height * (i + 1) / workers;
        args[i].output = output;
        args[i].params = *params;
        args[i].y_begin = y_begin;
        args[i].y_end = y_end;
        if (pthread_create(&threads[i], NULL, vignette_mask_worker_main, &args[i]) == 0) {
            started[i] = 1;
        } else {
            vignette_create_mask_rows(&args[i]);
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

int vignette_apply_mask_v1(
    const VignetteImageF32* input,
    const VignetteImageF32* mask,
    VignetteImageF32* output,
    float intensity
) {
    if (input == NULL || mask == NULL || output == NULL || input->data == NULL || mask->data == NULL || output->data == NULL) {
        return VIGNETTE_ERR_NULL;
    }
    if (
        input->width <= 0 || input->height <= 0 ||
        input->width != output->width || input->height != output->height ||
        input->width != mask->width || input->height != mask->height ||
        input->channels != output->channels || mask->channels != 1 ||
        (input->channels != 1 && input->channels != 3)
    ) {
        return VIGNETTE_ERR_SHAPE;
    }

    const int workers = vignette_worker_count(input->height, input->width);
    if (workers <= 1) {
        VignetteApplyMaskWorkerArgs args = {input, mask, output, intensity, 0, input->height};
        vignette_apply_mask_rows(&args);
        return VIGNETTE_OK;
    }

    pthread_t* threads = (pthread_t*)calloc((size_t)workers, sizeof(pthread_t));
    VignetteApplyMaskWorkerArgs* args = (VignetteApplyMaskWorkerArgs*)calloc((size_t)workers, sizeof(VignetteApplyMaskWorkerArgs));
    int* started = (int*)calloc((size_t)workers, sizeof(int));
    if (threads == NULL || args == NULL || started == NULL) {
        free(threads);
        free(args);
        free(started);
        VignetteApplyMaskWorkerArgs single_args = {input, mask, output, intensity, 0, input->height};
        vignette_apply_mask_rows(&single_args);
        return VIGNETTE_OK;
    }

    for (int i = 0; i < workers; ++i) {
        const int y_begin = input->height * i / workers;
        const int y_end = input->height * (i + 1) / workers;
        args[i].input = input;
        args[i].mask = mask;
        args[i].output = output;
        args[i].intensity = intensity;
        args[i].y_begin = y_begin;
        args[i].y_end = y_end;
        if (pthread_create(&threads[i], NULL, vignette_apply_mask_worker_main, &args[i]) == 0) {
            started[i] = 1;
        } else {
            vignette_apply_mask_rows(&args[i]);
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
