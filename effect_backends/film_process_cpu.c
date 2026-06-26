#include "film_process_capi.h"

#include <math.h>
#include <stddef.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Rec.601 luma weights, matching cv2.COLOR_RGB2GRAY (== core.cvtColorRGB2Gray).
 * Used by both color_drift and dye-purity desaturation. Do not switch to Rec.709
 * or parity with the NumPy reference breaks. */
#define FP_KR 0.299f
#define FP_KG 0.587f
#define FP_KB 0.114f

enum {
    FP_MODE_OFF = 0,
    FP_MODE_NEGATIVE = 1,
    FP_MODE_SLIDE = 2,
    FP_MODE_BW = 3,
};

static int validate_images(
    const FilmProcessConstImageF32* input,
    const FilmProcessImageF32* output,
    const FilmProcessParams* params
) {
    if (input == 0 || output == 0 || params == 0 || input->data == 0 || output->data == 0) {
        return FILM_PROCESS_ERR_NULL;
    }
    if (
        input->width <= 0
        || input->height <= 0
        || input->channels != 3
        || output->channels != 3
        || input->width != output->width
        || input->height != output->height
    ) {
        return FILM_PROCESS_ERR_SHAPE;
    }
    return FILM_PROCESS_OK;
}

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* _soft_density_response for a single channel value. HDR-preserving: above the
 * shoulder knee the excess is re-added linearly (no hard ceiling). */
static inline float density_response(float x, float gamma, float toe, float shoulder, float headroom, float black) {
    const float exposed = x + toe > 0.0f ? x + toe : 0.0f;
    const float dl = powf(exposed, gamma);
    const float comp = dl / (dl + shoulder);
    const float excess = dl - shoulder > 0.0f ? dl - shoulder : 0.0f;
    const float density = comp + headroom * excess - black;
    return density > 0.0f ? density : 0.0f;
}

int film_process_apply_v1(
    const FilmProcessConstImageF32* input,
    FilmProcessImageF32* output,
    const FilmProcessParams* params
) {
    const int status = validate_images(input, output, params);
    if (status != FILM_PROCESS_OK) {
        return status;
    }

    const int width = input->width;
    const int height = input->height;
    const int mode = params->mode;
    const float latitude = params->latitude;
    const float contrast = params->contrast;
    const float color_bias = params->color_bias;
    const float color_drift = params->color_drift;
    const float dye_purity = params->dye_purity;
    const float crosstalk = params->crosstalk;
    const float aging = params->aging;

    /* Per-frame scalars (shared by every pixel). */
    const float wg_r = 1.0f + color_bias * 0.18f;
    const float wg_g = 1.0f - color_bias * 0.08f;
    const float wg_b = 1.0f - color_bias * 0.18f;
    const float ag_r = 1.0f + aging * 0.10f;
    const float ag_g = 1.0f - aging * 0.08f;
    const float ag_b = 1.0f - aging * 0.24f;
    float gain_r = wg_r * ag_r;
    float gain_g = wg_g * ag_g;
    float gain_b = wg_b * ag_b;
    gain_r = gain_r > 0.05f ? gain_r : 0.05f;
    gain_g = gain_g > 0.05f ? gain_g : 0.05f;
    gain_b = gain_b > 0.05f ? gain_b : 0.05f;

    /* mix_matrix rows (layers @ mix.T => out_i = sum_j layer_j * M[i][j]). */
    const float m00 = 1.0f - 0.60f * crosstalk, m01 = 0.36f * crosstalk, m02 = 0.24f * crosstalk;
    const float m10 = 0.30f * crosstalk, m11 = 1.0f - 0.54f * crosstalk, m12 = 0.24f * crosstalk;
    const float m20 = 0.24f * crosstalk, m21 = 0.36f * crosstalk, m22 = 1.0f - 0.60f * crosstalk;

    float gamma = 0.74f + contrast * 0.92f;
    if (mode == FP_MODE_SLIDE) {
        gamma += 0.22f;
    } else if (mode == FP_MODE_BW) {
        gamma += 0.08f;
    }
    const float toe = 0.045f + latitude * 0.13f;
    const float shoulder = 0.72f + latitude * 1.12f;
    const float headroom = 0.02f + latitude * 0.13f;
    const float toe_g = powf(toe, gamma);
    const float black = toe_g / (toe_g + shoulder);

    /* mode-specific constants */
    const float neg_k = 1.55f + contrast * 1.10f;
    const float slide_p = 0.72f + contrast * 0.48f;
    const float slide_mult = 1.06f + contrast * 0.22f;
    const float bw_p = 0.78f + contrast * 0.42f;

    /* color drift */
    const float drift_dir = color_drift > 0.0f ? 1.0f : -1.0f;
    const float drift_strength = fabsf(color_drift);
    const int drift_on = drift_strength > 1.0e-6f;

    /* dye purity / leak */
    const float purity = 0.42f + dye_purity * 0.88f;
    const float impurity = (1.0f - dye_purity) * 0.22f;
    const float l00 = 1.0f - impurity, l01 = impurity * 0.65f, l02 = impurity * 0.35f;
    const float l10 = impurity * 0.35f, l11 = 1.0f - impurity, l12 = impurity * 0.65f;
    const float l20 = impurity * 0.55f, l21 = impurity * 0.45f, l22 = 1.0f - impurity;

    /* fog / base stain (all modes) */
    const float fog = aging * 0.095f;
    const float bs_r = 1.0f + aging * 0.10f;
    const float bs_g = 1.0f + aging * 0.035f;
    const float bs_b = 1.0f - aging * 0.055f;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const float* src_row = (const float*)((const unsigned char*)input->data + (size_t)y * input->stride_bytes);
        float* dst_row = (float*)((unsigned char*)output->data + (size_t)y * output->stride_bytes);
        for (int x = 0; x < width; ++x) {
            const int base = x * 3;
            const float r = src_row[base + 0];
            const float g = src_row[base + 1];
            const float b = src_row[base + 2];

            /* per-channel exposure gain */
            const float lr = r * gain_r;
            const float lg = g * gain_g;
            const float lb = b * gain_b;

            /* layer cross-talk mix */
            const float mr = lr * m00 + lg * m01 + lb * m02;
            const float mg = lr * m10 + lg * m11 + lb * m12;
            const float mb = lr * m20 + lg * m21 + lb * m22;

            /* film density response (per channel) */
            const float dr = density_response(mr, gamma, toe, shoulder, headroom, black);
            const float dg = density_response(mg, gamma, toe, shoulder, headroom, black);
            const float db = density_response(mb, gamma, toe, shoulder, headroom, black);

            float pr, pg, pb;
            if (mode == FP_MODE_NEGATIVE) {
                float vr = 1.0f - expf(-dr * neg_k);
                float vg = 1.0f - expf(-dg * neg_k);
                float vb = 1.0f - expf(-db * neg_k);
                vr = powf(vr > 0.0f ? vr : 0.0f, 0.90f);
                vg = powf(vg > 0.0f ? vg : 0.0f, 0.90f);
                vb = powf(vb > 0.0f ? vb : 0.0f, 0.90f);
                pr = vr + (dr - 1.0f > 0.0f ? dr - 1.0f : 0.0f) * 0.45f;
                pg = vg + (dg - 1.0f > 0.0f ? dg - 1.0f : 0.0f) * 0.45f;
                pb = vb + (db - 1.0f > 0.0f ? db - 1.0f : 0.0f) * 0.45f;
            } else if (mode == FP_MODE_SLIDE) {
                pr = powf(dr > 0.0f ? dr : 0.0f, slide_p) * slide_mult;
                pg = powf(dg > 0.0f ? dg : 0.0f, slide_p) * slide_mult;
                pb = powf(db > 0.0f ? db : 0.0f, slide_p) * slide_mult;
            } else {
                float mono = dr * 0.28f + dg * 0.55f + db * 0.17f;
                mono = powf(mono > 0.0f ? mono : 0.0f, bw_p);
                pr = pg = pb = mono;
            }

            if (mode != FP_MODE_BW) {
                /* color drift (tonal split + opponent twist), per pixel */
                if (drift_on) {
                    const float sr = pr > 0.0f ? pr : 0.0f;
                    const float sg = pg > 0.0f ? pg : 0.0f;
                    const float sb = pb > 0.0f ? pb : 0.0f;
                    const float luma = FP_KR * sr + FP_KG * sg + FP_KB * sb;
                    const float shadow_w = powf(clampf(1.0f - luma, 0.0f, 1.0f), 1.45f);
                    const float highlight_w = powf(clampf(luma, 0.0f, 1.0f), 1.65f);
                    const float mid_w = 1.0f - powf(clampf(fabsf(luma * 2.0f - 1.0f), 0.0f, 1.0f), 1.8f);

                    const float tb_r = shadow_w * (-0.050f) + highlight_w * (0.075f) + mid_w * (0.020f);
                    const float tb_g = shadow_w * (0.026f) + highlight_w * (0.018f) + mid_w * (-0.014f);
                    const float tb_b = shadow_w * (0.064f) + highlight_w * (-0.050f) + mid_w * (0.010f);

                    const float rb_opp = sr - sb;
                    const float gm_opp = sg - (sr + sg + sb) / 3.0f;
                    const float tw_r = -0.030f * gm_opp;
                    const float tw_g = 0.022f * rb_opp;
                    const float tw_b = -0.026f * rb_opp + 0.018f * gm_opp;

                    pr += drift_dir * drift_strength * (tb_r + tw_r);
                    pg += drift_dir * drift_strength * (tb_g + tw_g);
                    pb += drift_dir * drift_strength * (tb_b + tw_b);
                    pr = pr > 0.0f ? pr : 0.0f;
                    pg = pg > 0.0f ? pg : 0.0f;
                    pb = pb > 0.0f ? pb : 0.0f;
                }

                /* dye purity (saturation toward/away from luma) */
                const float cl_r = pr > 0.0f ? pr : 0.0f;
                const float cl_g = pg > 0.0f ? pg : 0.0f;
                const float cl_b = pb > 0.0f ? pb : 0.0f;
                const float luma = FP_KR * cl_r + FP_KG * cl_g + FP_KB * cl_b;
                pr = luma + (pr - luma) * purity;
                pg = luma + (pg - luma) * purity;
                pb = luma + (pb - luma) * purity;

                /* dye layer leak (positive @ dye_leak.T) */
                const float or_ = pr * l00 + pg * l01 + pb * l02;
                const float og = pr * l10 + pg * l11 + pb * l12;
                const float ob = pr * l20 + pg * l21 + pb * l22;
                pr = or_;
                pg = og;
                pb = ob;
            }

            /* fog lift + base stain, then floor at 0 (HDR: no upper clip) */
            pr = (pr * (1.0f - fog) + fog) * bs_r;
            pg = (pg * (1.0f - fog) + fog) * bs_g;
            pb = (pb * (1.0f - fog) + fog) * bs_b;

            dst_row[base + 0] = pr > 0.0f ? pr : 0.0f;
            dst_row[base + 1] = pg > 0.0f ? pg : 0.0f;
            dst_row[base + 2] = pb > 0.0f ? pb : 0.0f;
        }
    }

    return FILM_PROCESS_OK;
}
