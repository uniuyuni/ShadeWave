#ifndef LUT_CAPI_H
#define LUT_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} LutConstImageF32;

typedef struct {
    float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} LutImageF32;

typedef struct {
    /* Cubic 3D LUT table, layout (size, size, size, 3) row-major. */
    const float* data;
    int size;
} LutTableF32;

typedef struct {
    /* Per-channel input domain: clip + normalize range. */
    float min[3];
    float max[3];
} LutDomainF32;

/* Apply a 3D LUT with trilinear interpolation.
 *
 * BGR index convention (input RGB grid coords g = (gR, gG, gB) = norm*(size-1)):
 *   table[a, b, c] addressed with a = floor(gB), b = floor(gG), c = floor(gR);
 *   flat = ((a*size + b)*size + c)*3.
 *   Interpolation weights: axis0 = frac(gB), axis1 = frac(gG), axis2 = frac(gR).
 */
int lut_apply_trilinear_v1(
    const LutConstImageF32* input,
    LutImageF32* output,
    const LutTableF32* table,
    const LutDomainF32* domain
);

#ifdef __cplusplus
}
#endif

#endif
