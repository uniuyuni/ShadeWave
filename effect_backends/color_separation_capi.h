#ifndef COLOR_SEPARATION_CAPI_H
#define COLOR_SEPARATION_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} ColorSeparationConstImageF32;

typedef struct {
    float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} ColorSeparationImageF32;

typedef struct {
    float shadow_chroma_clean;
    float shadow_threshold;
    float color_separation;
    float chroma_clarity;
    float color_density;
    float subtractive_saturation;
    float opponent_contrast;
} ColorSeparationParams;

enum {
    COLOR_SEPARATION_OK = 0,
    COLOR_SEPARATION_ERR_NULL = 1,
    COLOR_SEPARATION_ERR_SHAPE = 2,
    COLOR_SEPARATION_ERR_ALLOC = 3,
};

int color_separation_apply_v1(
    const ColorSeparationConstImageF32* input,
    ColorSeparationImageF32* output,
    const ColorSeparationParams* params
);

#ifdef __cplusplus
}
#endif

#endif
