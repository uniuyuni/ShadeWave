#ifndef CONFIG_H
#define CONFIG_H

#define PROFILE "original"

#define PLATFORM_NVIDIA 0
#define PLATFORM_ATI    1
#define PLATFORM_INTEL  2
#define PLATFORM_APPLE  3

#define UNROLL 0

#define DISTANCE_IN_SPARSE 0

//#define BLOCK_SIZE 8
#define BLOCK_SIZE_HALF (BLOCK_SIZE / 2)
#define BLOCK_SIZE_SQ (BLOCK_SIZE * BLOCK_SIZE)

#define WINDOW_SIZE 38
#define WINDOW_SIZE_HALF 19

#define STEP_SIZE 3
// Multiple of STEP_SIZE
#define SPLIT_SIZE_X (3*STEP_SIZE)
#define SPLIT_SIZE_Y (3*STEP_SIZE)

#define WINDOW_STEP_SIZE_1 3
#define WINDOW_STEP_SIZE_2 3

#define MAX_BLOCK_COUNT_1 16
// 32 causes crash on CPU
#define MAX_BLOCK_COUNT_2 32

#define USE_KAISER_WINDOW 1

#define DCT_1D 0
#define HAAR_1D 1
#define TRANSFORM_METHOD_1D HAAR_1D

#define D_THRESHOLD_1 (3 * 2500)
#define D_THRESHOLD_2 (3 * 400)

// Default sigma value to use
#ifndef SIGMA
#   define SIGMA 1
#endif

//#define VARIANCE ((float)SIGMA*(float)SIGMA)
#define VARIANCE (0.02f * (float)SIGMA*(float)SIGMA)

#if 0 //(SIGMA > 40)
#   define USE_2D_THRESHOLD 1
#   define TAU_1D (2.8f * (float)SIGMA)
#else
#   define USE_2D_THRESHOLD 0
//#   define TAU_1D (2.7f * (float)SIGMA)
#   define TAU_1D   (0.12f * (float)SIGMA)
#endif

//#define TAU_2D (2.0f * (float)SIGMA)
#define TAU_2D (0.08f * (float)SIGMA)

#endif

