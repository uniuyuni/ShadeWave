#ifndef VIGNETTE_CAPI_H
#define VIGNETTE_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} VignetteImageF32;

typedef struct {
    float intensity;
    float radius_percent;
    float gradient_softness;
    float disp_info[5];
    float crop_rect[4];
    float offset[2];
} VignetteParams;

enum {
    VIGNETTE_OK = 0,
    VIGNETTE_ERR_NULL = 1,
    VIGNETTE_ERR_SHAPE = 2,
};

int vignette_apply_v1(
    const VignetteImageF32* input,
    VignetteImageF32* output,
    const VignetteParams* params
);

int vignette_create_mask_v1(
    VignetteImageF32* output,
    const VignetteParams* params
);

int vignette_apply_mask_v1(
    const VignetteImageF32* input,
    const VignetteImageF32* mask,
    VignetteImageF32* output,
    float intensity
);

#ifdef __cplusplus
}
#endif

#endif
