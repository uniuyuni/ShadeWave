#ifndef FILM_PROCESS_CAPI_H
#define FILM_PROCESS_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} FilmProcessConstImageF32;

typedef struct {
    float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} FilmProcessImageF32;

/* All scalar params are pre-normalized by the adapter:
 *   mode      : 0=Off, 1=Negative, 2=Slide, 3=B&W (Off must not reach the kernel)
 *   latitude  : 0..1
 *   contrast  : 0..1
 *   color_bias: -1..1
 *   color_drift: -1..1
 *   dye_purity: 0..1
 *   crosstalk : 0..1
 *   aging     : 0..1
 * Halation (a spatial op) is applied on the Python side before this kernel,
 * so it is intentionally absent here. The kernel is purely pointwise. */
typedef struct {
    int mode;
    float latitude;
    float contrast;
    float color_bias;
    float color_drift;
    float dye_purity;
    float crosstalk;
    float aging;
} FilmProcessParams;

enum {
    FILM_PROCESS_OK = 0,
    FILM_PROCESS_ERR_NULL = 1,
    FILM_PROCESS_ERR_SHAPE = 2,
    FILM_PROCESS_ERR_ALLOC = 3,
};

int film_process_apply_v1(
    const FilmProcessConstImageF32* input,
    FilmProcessImageF32* output,
    const FilmProcessParams* params
);

#ifdef __cplusplus
}
#endif

#endif
