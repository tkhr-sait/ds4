/* NEON i8mm (FEAT_MATMUL_INT8) kernels for ds4.
 *
 * Adds an SMMLA-based 2x2 mini-GEMM for the Q4_K weights x Q8_K
 * activation dot product used by routed MoE experts.  One smmla
 * instruction consumes 8 K-elements per side and produces a 2x2 int32
 * outer-product accumulator, so 32 smmlas cover one 256-element Q4_K
 * block for the 4 dots, vs 4*16 = 64 vdotq for the same outputs.  The
 * front-end issue reduction translates to a real speedup on Apple M4.
 *
 * On hosts whose compiler does not understand `+i8mm`, the file
 * compiles as a stub (the SMMLA path is guarded by
 * __ARM_FEATURE_MATMUL_INT8) and ds4_has_i8mm() stays false. */

#include "ds4_neon_i8mm.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP2_I8MM
/* Mirrors Linux uapi <asm/hwcap.h>; older glibc headers may not expose this. */
#define HWCAP2_I8MM (1UL << 13)
#endif
#endif

static bool g_has_i8mm = false;
static bool g_initialized = false;

void ds4_neon_i8mm_init(void) {
    if (g_initialized) return;
    g_initialized = true;

    if (getenv("DS4_DISABLE_I8MM") != NULL) return;

#if defined(__ARM_FEATURE_MATMUL_INT8)
#if defined(__APPLE__)
    int has = 0;
    size_t size = sizeof(has);
    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &has, &size, NULL, 0) == 0 && has) {
        g_has_i8mm = true;
    }
#elif defined(__linux__)
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    if (hwcap2 & HWCAP2_I8MM) g_has_i8mm = true;
#endif
#endif
}

bool ds4_has_i8mm(void) {
    return g_has_i8mm;
}

#if defined(__ARM_FEATURE_MATMUL_INT8)

static inline float ds4_i8mm_f16_to_f32(uint16_t h) {
#if defined(__ARM_NEON)
    const float16x4_t hv = vreinterpret_f16_u16(vdup_n_u16(h));
    return vgetq_lane_f32(vcvt_f32_f16(hv), 0);
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 1;
            while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
            mant &= 0x03ff;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, sizeof(f)); return f;
#endif
}

void ds4_neon_i8mm_q4_K_q8_K_2x2(
        int                  n,
        float               *out,
        const block_q4_K    *x_row0,
        const block_q4_K    *x_row1,
        const block_q8_K    *y_col0,
        const block_q8_K    *y_col1) {
    const int nb = n / QK_K;

    /* fp_acc lanes hold [r0c0, r0c1, r1c0, r1c1] -- the SMMLA result layout. */
    float32x4_t fp_acc = vdupq_n_f32(0.0f);
    /* dmin contributions accumulated in scalar; folded in at the end. */
    float dmin_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    uint32_t utmp_0[4], utmp_1[4];

    for (int i = 0; i < nb; i++) {
        const float dx0 = ds4_i8mm_f16_to_f32(x_row0[i].d);
        const float dx1 = ds4_i8mm_f16_to_f32(x_row1[i].d);
        const float dy0 = y_col0[i].d;
        const float dy1 = y_col1[i].d;
        const float dminx0 = ds4_i8mm_f16_to_f32(x_row0[i].dmin);
        const float dminx1 = ds4_i8mm_f16_to_f32(x_row1[i].dmin);

        /* Unpack row 0 scales/mins. */
        memcpy(utmp_0, x_row0[i].scales, 12);
        utmp_0[3] = ((utmp_0[2] >> 4) & 0x0f0f0f0fu) | (((utmp_0[1] >> 6) & 0x03030303u) << 4);
        { uint32_t u = utmp_0[1] & 0x3f3f3f3fu;
          utmp_0[1] = (utmp_0[2] & 0x0f0f0f0fu) | (((utmp_0[0] >> 6) & 0x03030303u) << 4);
          utmp_0[2] = u; }
        utmp_0[0] &= 0x3f3f3f3fu;
        const uint8_t *scales_0 = (const uint8_t *)utmp_0;
        const uint8_t *mins_0   = (const uint8_t *)&utmp_0[2];

        /* Unpack row 1 scales/mins. */
        memcpy(utmp_1, x_row1[i].scales, 12);
        utmp_1[3] = ((utmp_1[2] >> 4) & 0x0f0f0f0fu) | (((utmp_1[1] >> 6) & 0x03030303u) << 4);
        { uint32_t u = utmp_1[1] & 0x3f3f3f3fu;
          utmp_1[1] = (utmp_1[2] & 0x0f0f0f0fu) | (((utmp_1[0] >> 6) & 0x03030303u) << 4);
          utmp_1[2] = u; }
        utmp_1[0] &= 0x3f3f3f3fu;
        const uint8_t *scales_1 = (const uint8_t *)utmp_1;
        const uint8_t *mins_1   = (const uint8_t *)&utmp_1[2];

        /* dmin correction (scalar): contribution per (row, col) per block. */
        int32_t s_0_c0 = 0, s_0_c1 = 0, s_1_c0 = 0, s_1_c1 = 0;
        for (int j = 0; j < 8; j++) {
            const int32_t bsum_c0 = (int32_t)y_col0[i].bsums[2*j] + (int32_t)y_col0[i].bsums[2*j + 1];
            const int32_t bsum_c1 = (int32_t)y_col1[i].bsums[2*j] + (int32_t)y_col1[i].bsums[2*j + 1];
            s_0_c0 += bsum_c0 * (int32_t)mins_0[j];
            s_0_c1 += bsum_c1 * (int32_t)mins_0[j];
            s_1_c0 += bsum_c0 * (int32_t)mins_1[j];
            s_1_c1 += bsum_c1 * (int32_t)mins_1[j];
        }
        dmin_acc[0] -= dy0 * dminx0 * (float)s_0_c0;
        dmin_acc[1] -= dy1 * dminx0 * (float)s_0_c1;
        dmin_acc[2] -= dy0 * dminx1 * (float)s_1_c0;
        dmin_acc[3] -= dy1 * dminx1 * (float)s_1_c1;

        const uint8_t *q4_0 = x_row0[i].qs;
        const uint8_t *q4_1 = x_row1[i].qs;
        const int8_t  *q8_0 = y_col0[i].qs;
        const int8_t  *q8_1 = y_col1[i].qs;

        /* 4 outer iters cover the 8 sub-blocks of one Q4_K block: each
         * iter handles one low/high nibble pair (sub-blocks 2j and 2j+1). */
        for (int j = 0; j < QK_K / 64; j++) {
            const uint8x16_t q4_0_a = vld1q_u8(q4_0);
            const uint8x16_t q4_0_b = vld1q_u8(q4_0 + 16);
            const uint8x16_t q4_1_a = vld1q_u8(q4_1);
            const uint8x16_t q4_1_b = vld1q_u8(q4_1 + 16);
            q4_0 += 32; q4_1 += 32;

            const int8x16_t q4_0_lo_a = vreinterpretq_s8_u8(vandq_u8(q4_0_a, m4b));
            const int8x16_t q4_0_lo_b = vreinterpretq_s8_u8(vandq_u8(q4_0_b, m4b));
            const int8x16_t q4_0_hi_a = vreinterpretq_s8_u8(vshrq_n_u8(q4_0_a, 4));
            const int8x16_t q4_0_hi_b = vreinterpretq_s8_u8(vshrq_n_u8(q4_0_b, 4));
            const int8x16_t q4_1_lo_a = vreinterpretq_s8_u8(vandq_u8(q4_1_a, m4b));
            const int8x16_t q4_1_lo_b = vreinterpretq_s8_u8(vandq_u8(q4_1_b, m4b));
            const int8x16_t q4_1_hi_a = vreinterpretq_s8_u8(vshrq_n_u8(q4_1_a, 4));
            const int8x16_t q4_1_hi_b = vreinterpretq_s8_u8(vshrq_n_u8(q4_1_b, 4));

            const int8x16_t q8_0_lo_a = vld1q_s8(q8_0);
            const int8x16_t q8_0_lo_b = vld1q_s8(q8_0 + 16);
            const int8x16_t q8_0_hi_a = vld1q_s8(q8_0 + 32);
            const int8x16_t q8_0_hi_b = vld1q_s8(q8_0 + 48);
            const int8x16_t q8_1_lo_a = vld1q_s8(q8_1);
            const int8x16_t q8_1_lo_b = vld1q_s8(q8_1 + 16);
            const int8x16_t q8_1_hi_a = vld1q_s8(q8_1 + 32);
            const int8x16_t q8_1_hi_b = vld1q_s8(q8_1 + 48);
            q8_0 += 64; q8_1 += 64;

            /* SMMLA expects A = (r0[0..7], r1[0..7]) and B = (c0[0..7], c1[0..7]).
             * We get there by vcombine_s8 of each row's low/high half. */
#define DS4_I8MM_SMMLA(ACC, R0V, R1V, C0V, C1V) do {                              \
    const int8x16_t A__ = vcombine_s8(vget_low_s8(R0V), vget_low_s8(R1V));        \
    const int8x16_t B__ = vcombine_s8(vget_low_s8(C0V), vget_low_s8(C1V));        \
    (ACC) = vmmlaq_s32((ACC), A__, B__);                                          \
    const int8x16_t A2_ = vcombine_s8(vget_high_s8(R0V), vget_high_s8(R1V));      \
    const int8x16_t B2_ = vcombine_s8(vget_high_s8(C0V), vget_high_s8(C1V));      \
    (ACC) = vmmlaq_s32((ACC), A2_, B2_);                                          \
} while (0)

            /* Sub-block 2j: 32 K-elements, 4 SMMLAs (8 K each).  Two from the
             * "_a" pair (K=0..15) and two from the "_b" pair (K=16..31). */
            int32x4_t int_acc_lo = vdupq_n_s32(0);
            DS4_I8MM_SMMLA(int_acc_lo, q4_0_lo_a, q4_1_lo_a, q8_0_lo_a, q8_1_lo_a);
            DS4_I8MM_SMMLA(int_acc_lo, q4_0_lo_b, q4_1_lo_b, q8_0_lo_b, q8_1_lo_b);
            const float s_lo_r0 = dx0 * (float)scales_0[2*j];
            const float s_lo_r1 = dx1 * (float)scales_1[2*j];
            const float32x4_t scale_lo = {
                s_lo_r0 * dy0, s_lo_r0 * dy1, s_lo_r1 * dy0, s_lo_r1 * dy1
            };
            fp_acc = vfmaq_f32(fp_acc, vcvtq_f32_s32(int_acc_lo), scale_lo);

            /* Sub-block 2j+1: high nibbles of the same Q4 bytes paired with
             * the second half of the activation block. */
            int32x4_t int_acc_hi = vdupq_n_s32(0);
            DS4_I8MM_SMMLA(int_acc_hi, q4_0_hi_a, q4_1_hi_a, q8_0_hi_a, q8_1_hi_a);
            DS4_I8MM_SMMLA(int_acc_hi, q4_0_hi_b, q4_1_hi_b, q8_0_hi_b, q8_1_hi_b);
            const float s_hi_r0 = dx0 * (float)scales_0[2*j + 1];
            const float s_hi_r1 = dx1 * (float)scales_1[2*j + 1];
            const float32x4_t scale_hi = {
                s_hi_r0 * dy0, s_hi_r0 * dy1, s_hi_r1 * dy0, s_hi_r1 * dy1
            };
            fp_acc = vfmaq_f32(fp_acc, vcvtq_f32_s32(int_acc_hi), scale_hi);

#undef DS4_I8MM_SMMLA
        }
    }

    /* Fold in dmin corrections and store. */
    out[0] = dmin_acc[0] + vgetq_lane_f32(fp_acc, 0);
    out[1] = dmin_acc[1] + vgetq_lane_f32(fp_acc, 1);
    out[2] = dmin_acc[2] + vgetq_lane_f32(fp_acc, 2);
    out[3] = dmin_acc[3] + vgetq_lane_f32(fp_acc, 3);
}

#else  /* !__ARM_FEATURE_MATMUL_INT8: stub. */

void ds4_neon_i8mm_q4_K_q8_K_2x2(int n, float *out,
                                 const block_q4_K *x_row0, const block_q4_K *x_row1,
                                 const block_q8_K *y_col0, const block_q8_K *y_col1) {
    (void)n; (void)out; (void)x_row0; (void)x_row1; (void)y_col0; (void)y_col1;
}

#endif
