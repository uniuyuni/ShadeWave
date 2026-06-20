/*
 * Native fused display colour transform C API.
 *
 * License: GPL-3.0-or-later as part of Shade Wave / PLATYPUS.
 * Implements behaviour compatible with a subset of Colour Science for Python,
 * whose upstream project is BSD-3-Clause licensed.
 */

#ifndef COLOUR_FUNCTIONS_CAPI_H
#define COLOUR_FUNCTIONS_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    COLOUR_FUNCTIONS_ENCODING_LINEAR = 0,
    COLOUR_FUNCTIONS_ENCODING_SRGB = 1,
    COLOUR_FUNCTIONS_ENCODING_REC709 = 2,
    COLOUR_FUNCTIONS_ENCODING_REC2020 = 3,
    COLOUR_FUNCTIONS_ENCODING_GAMMA_ADOBE_RGB = 4,
    COLOUR_FUNCTIONS_ENCODING_GAMMA_1_8 = 5,
    COLOUR_FUNCTIONS_ENCODING_GAMMA_2_2 = 6,
    COLOUR_FUNCTIONS_ENCODING_GAMMA_2_6 = 7,
    COLOUR_FUNCTIONS_ENCODING_PROPHOTO = 8
} ColourFunctionsEncoding;

typedef struct {
    const float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} ColourFunctionsConstImageF32;

typedef struct {
    float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} ColourFunctionsImageF32;

typedef struct {
    float basis[9];
    ColourFunctionsEncoding encoding;
    float luminance_weights[3];
    float eps;
} ColourFunctionsParams;

enum {
    COLOUR_FUNCTIONS_OK = 0,
    COLOUR_FUNCTIONS_ERR_NULL = 1,
    COLOUR_FUNCTIONS_ERR_SHAPE = 2,
    COLOUR_FUNCTIONS_ERR_ENCODING = 3,
};

int colour_functions_transform_v1(
    const ColourFunctionsConstImageF32* input,
    ColourFunctionsImageF32* output,
    const ColourFunctionsParams* params
);

#ifdef __cplusplus
}
#endif

#endif
