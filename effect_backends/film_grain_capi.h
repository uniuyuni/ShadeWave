#ifndef FILM_GRAIN_CAPI_H
#define FILM_GRAIN_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} FilmGrainConstImageF32;

typedef struct {
    float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} FilmGrainImageF32;

typedef struct {
    float amount;
    float grain_size;
    float roughness;
    float shadow;
    float highlight;
    float color;
    int seed;
} FilmGrainParams;

enum {
    FILM_GRAIN_OK = 0,
    FILM_GRAIN_ERR_NULL = 1,
    FILM_GRAIN_ERR_SHAPE = 2,
    FILM_GRAIN_ERR_ALLOC = 3,
};

int film_grain_apply_v1(
    const FilmGrainConstImageF32* input,
    FilmGrainImageF32* output,
    const FilmGrainParams* params
);

#ifdef __cplusplus
}
#endif

#endif
