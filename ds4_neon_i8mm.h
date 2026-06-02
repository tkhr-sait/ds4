#ifndef DS4_NEON_I8MM_H
#define DS4_NEON_I8MM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4_quant_blocks.h"

/* One-time runtime probe.  Safe to call before any other entrypoint.  Reads
 * CPU feature flags for ARMv8.6 FEAT_I8MM (Darwin: sysctlbyname FEAT_I8MM;
 * Linux: HWCAP2_I8MM).  Honors the DS4_DISABLE_I8MM env override. */
void ds4_neon_i8mm_init(void);

/* Returns the latched result of ds4_neon_i8mm_init().  False until init has
 * been called.  Safe to call from any thread after init. */
bool ds4_has_i8mm(void);

/* NEON i8mm (SMMLA) variant: a 2-row × 2-col mini-panel that computes
 * four Q4_K x Q8_K dot products at once via vmmlaq_s32.  One smmla
 * instruction consumes 8 K-elements per side and produces a 2x2 int32
 * accumulator; 32 smmla therefore cover one 256-element Q4_K block for
 * the 4 dots.  Compared with the per-dot vdotq path this cuts the
 * issued instruction count by ~4x on the hot path.
 *
 * out[r*2 + c] receives dot(x_row[r], y_col[c]) for r,c in {0,1}.
 * n must be a multiple of QK_K.  Must only be invoked when
 * ds4_has_i8mm() is true. */
void ds4_neon_i8mm_q4_K_q8_K_2x2(
        int                  n,
        float              * out,        /* [4] */
        const block_q4_K   * x_row0,
        const block_q4_K   * x_row1,
        const block_q8_K   * y_col0,
        const block_q8_K   * y_col1);

#endif
