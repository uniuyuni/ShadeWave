#ifndef LOW_FREQUENCY_TRANSFER_CAPI_H
#define LOW_FREQUENCY_TRANSFER_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} LowFrequencyTransferConstImageF32;

typedef struct {
    float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} LowFrequencyTransferImageF32;

typedef struct {
    float sigma;
    int use_highlight_protection;
    float highlight_threshold;
    float highlight_transition;
    float highlight_detail_strength;
    float luminance_transfer_strength;
} LowFrequencyTransferParams;

enum {
    LOW_FREQUENCY_TRANSFER_OK = 0,
    LOW_FREQUENCY_TRANSFER_ERR_NULL = 1,
    LOW_FREQUENCY_TRANSFER_ERR_SHAPE = 2,
    LOW_FREQUENCY_TRANSFER_ERR_ALLOC = 3,
};

int low_frequency_transfer_apply_v1(
    const LowFrequencyTransferConstImageF32* restored,
    const LowFrequencyTransferConstImageF32* reference,
    LowFrequencyTransferImageF32* output,
    const LowFrequencyTransferParams* params
);

int low_frequency_transfer_compose_lowres_v1(
    const LowFrequencyTransferConstImageF32* restored,
    const LowFrequencyTransferConstImageF32* reference,
    const LowFrequencyTransferConstImageF32* low_diff,
    const LowFrequencyTransferConstImageF32* low_restored,
    LowFrequencyTransferImageF32* output,
    const LowFrequencyTransferParams* params
);

#ifdef __cplusplus
}
#endif

#endif
