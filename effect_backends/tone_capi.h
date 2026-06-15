#ifndef TONE_CAPI_H
#define TONE_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} ToneConstImageF32;

typedef struct {
    float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} ToneImageF32;

typedef struct {
    float highlights;
    float shadows;
    float midtone;
    float white_level;
    float black_level;
    float disp_scale;
    float resolution_scale;
} ToneParams;

enum {
    TONE_OK = 0,
    TONE_ERR_NULL = 1,
    TONE_ERR_SHAPE = 2,
    TONE_ERR_ALLOC = 3,
};

int tone_adjust_v1(
    const ToneConstImageF32* input,
    ToneImageF32* output,
    const ToneParams* params
);

#ifdef __cplusplus
}
#endif

#endif
