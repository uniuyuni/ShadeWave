#ifndef IMAGE_TRANSFORM_CAPI_H
#define IMAGE_TRANSFORM_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    IMAGE_TRANSFORM_INTERPOLATION_NEAREST = 0,
    IMAGE_TRANSFORM_INTERPOLATION_LINEAR = 1,
    IMAGE_TRANSFORM_INTERPOLATION_AREA = 2,
    IMAGE_TRANSFORM_INTERPOLATION_CUBIC = 3,
    IMAGE_TRANSFORM_INTERPOLATION_LANCZOS4 = 4
} ImageTransformInterpolation;

typedef enum {
    IMAGE_TRANSFORM_BORDER_CONSTANT_ZERO = 0,
    IMAGE_TRANSFORM_BORDER_REFLECT = 1,
    IMAGE_TRANSFORM_BORDER_REPLICATE = 2
} ImageTransformBorderMode;

typedef struct {
    const float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} ImageTransformConstImageF32;

typedef struct {
    float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} ImageTransformImageF32;

typedef struct {
    int x;
    int y;
    int width;
    int height;
} ImageTransformRectI;

typedef struct {
    ImageTransformRectI source_rect;
    int canvas_width;
    int canvas_height;
    int draw_width;
    int draw_height;
    int offset_x;
    int offset_y;
    ImageTransformInterpolation interpolation;
    ImageTransformBorderMode border_mode;
} ImageTransformFitCropToCanvasParams;

int fit_crop_to_canvas_v1(
    const ImageTransformConstImageF32* input,
    ImageTransformImageF32* output,
    const ImageTransformFitCropToCanvasParams* params
);

#ifdef __cplusplus
}
#endif

#endif
