#ifndef CROSS_FILTER_CAPI_H
#define CROSS_FILTER_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} CrossFilterImageF32;

typedef struct {
    int num_points;
    int length;
    float angle_deg;
    float threshold;
    float intensity;
    float spectral_strength;
    float line_thickness;
    int min_distance;
    float randomness;
    int speed_factor;
    int debug_mode;
} CrossFilterParams;

enum {
    CROSS_FILTER_OK = 0,
    CROSS_FILTER_ERR_NULL = 1,
    CROSS_FILTER_ERR_SHAPE = 2,
    CROSS_FILTER_ERR_ALLOC = 3,
};

int cross_filter_apply_v1(
    const CrossFilterImageF32* input,
    CrossFilterImageF32* output,
    const CrossFilterParams* params
);

#ifdef __cplusplus
}
#endif

#endif
