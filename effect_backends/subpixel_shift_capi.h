#ifndef SUBPIXEL_SHIFT_CAPI_H
#define SUBPIXEL_SHIFT_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} SubpixelShiftConstImageF32;

typedef struct {
    float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} SubpixelShiftImageF32;

typedef struct {
    float shift_x;
    float shift_y;
} SubpixelShiftParams;

enum {
    SUBPIXEL_SHIFT_OK = 0,
    SUBPIXEL_SHIFT_ERR_NULL = 1,
    SUBPIXEL_SHIFT_ERR_SHAPE = 2,
};

int subpixel_shift_apply_v1(
    const SubpixelShiftConstImageF32* input,
    SubpixelShiftImageF32* output,
    const SubpixelShiftParams* params
);

int subpixel_shift_enhance_v1(
    const SubpixelShiftConstImageF32* input,
    SubpixelShiftImageF32* output
);

#ifdef __cplusplus
}
#endif

#endif
