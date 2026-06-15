#include "subpixel_shift_capi.h"

#include <math.h>
#include <stddef.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static int validate_images(
    const SubpixelShiftConstImageF32* input,
    const SubpixelShiftImageF32* output
) {
    if (input == 0 || output == 0 || input->data == 0 || output->data == 0) {
        return SUBPIXEL_SHIFT_ERR_NULL;
    }
    if (input->width <= 0 || input->height <= 0 || input->width != output->width || input->height != output->height) {
        return SUBPIXEL_SHIFT_ERR_SHAPE;
    }
    if (input->channels != 3 || output->channels != 3) {
        return SUBPIXEL_SHIFT_ERR_SHAPE;
    }
    return SUBPIXEL_SHIFT_OK;
}

static int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

int subpixel_shift_apply_v1(
    const SubpixelShiftConstImageF32* input,
    SubpixelShiftImageF32* output,
    const SubpixelShiftParams* params
) {
    const int valid = validate_images(input, output);
    if (valid != SUBPIXEL_SHIFT_OK) {
        return valid;
    }
    if (params == 0) {
        return SUBPIXEL_SHIFT_ERR_NULL;
    }

    const int width = input->width;
    const int height = input->height;
    const float shift_x = params->shift_x;
    const float shift_y = params->shift_y;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        float fy = (float)y - shift_y;
        int y0_raw = (int)floorf(fy);
        int y1_raw = y0_raw + 1;
        float wy1 = fy - (float)y0_raw;
        float wy0 = 1.0f - wy1;
        int y0 = clamp_int(y0_raw, 0, height - 1);
        int y1 = clamp_int(y1_raw, 0, height - 1);

        const float* row0 = (const float*)((const unsigned char*)input->data + (size_t)y0 * input->stride_bytes);
        const float* row1 = (const float*)((const unsigned char*)input->data + (size_t)y1 * input->stride_bytes);
        float* dst = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);

        for (int x = 0; x < width; ++x) {
            float fx = (float)x - shift_x;
            int x0_raw = (int)floorf(fx);
            int x1_raw = x0_raw + 1;
            float wx1 = fx - (float)x0_raw;
            float wx0 = 1.0f - wx1;
            int x0 = clamp_int(x0_raw, 0, width - 1);
            int x1 = clamp_int(x1_raw, 0, width - 1);

            const int b00 = x0 * 3;
            const int b10 = x1 * 3;
            const float w00 = wy0 * wx0;
            const float w10 = wy0 * wx1;
            const float w01 = wy1 * wx0;
            const float w11 = wy1 * wx1;
            const int out = x * 3;

            dst[out + 0] = row0[b00 + 0] * w00 + row0[b10 + 0] * w10 + row1[b00 + 0] * w01 + row1[b10 + 0] * w11;
            dst[out + 1] = row0[b00 + 1] * w00 + row0[b10 + 1] * w10 + row1[b00 + 1] * w01 + row1[b10 + 1] * w11;
            dst[out + 2] = row0[b00 + 2] * w00 + row0[b10 + 2] * w10 + row1[b00 + 2] * w01 + row1[b10 + 2] * w11;
        }
    }

    return SUBPIXEL_SHIFT_OK;
}

int subpixel_shift_enhance_v1(
    const SubpixelShiftConstImageF32* input,
    SubpixelShiftImageF32* output
) {
    const int valid = validate_images(input, output);
    if (valid != SUBPIXEL_SHIFT_OK) {
        return valid;
    }

    const int width = input->width;
    const int height = input->height;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const int ym = clamp_int(y - 1, 0, height - 1);
        const int yp = clamp_int(y + 1, 0, height - 1);
        const float* row_m = (const float*)((const unsigned char*)input->data + (size_t)ym * input->stride_bytes);
        const float* row_0 = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
        const float* row_p = (const float*)((const unsigned char*)input->data + (size_t)yp * input->stride_bytes);
        float* dst = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);

        for (int x = 0; x < width; ++x) {
            const int xm = clamp_int(x - 1, 0, width - 1) * 3;
            const int x0 = x * 3;
            const int xp = clamp_int(x + 1, 0, width - 1) * 3;
            const int out = x * 3;

            for (int c = 0; c < 3; ++c) {
                dst[out + c] =
                    (row_m[xm + c] + row_m[xp + c] + row_p[xm + c] + row_p[xp + c]) * 0.0625f
                    + (row_m[x0 + c] + row_0[xm + c] + row_0[xp + c] + row_p[x0 + c]) * 0.125f
                    + row_0[x0 + c] * 0.25f;
            }
        }
    }

    return SUBPIXEL_SHIFT_OK;
}
