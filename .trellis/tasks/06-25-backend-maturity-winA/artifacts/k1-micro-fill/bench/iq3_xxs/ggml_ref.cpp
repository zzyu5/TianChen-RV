// ggml's SHIPPED RVV iq3_xxs_q8_K kernel, _vl256 variant (k1/VLEN256 dispatch,
// case 256), lifted VERBATIM from llama.cpp arch/riscv/quants.c:5192, reframed
// to raw byte offsets + 4-arg adapter.
// block_iq3_xxs=98B (d@0 fp16, qs[96]@2: 64 grid-idx bytes + 32 metadata bytes@66).
// block_q8_K=292B. grid=iq3xxs_grid; keven_signs_q2xs[1024] lifted verbatim from
// quants.c (NOT in ggml-common.h). Uses vluxei16 grid+sign gather.
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <riscv_vector.h>

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "../keven_signs_q2xs.inc"

#define QK_K 256
#ifndef GGML_RESTRICT
#define GGML_RESTRICT __restrict
#endif
static inline float fp16_to_fp32(uint16_t h) { _Float16 f; memcpy(&f, &h, 2); return (float)f; }

static void iq3_xxs_vl256(int n, float * GGML_RESTRICT s, const uint8_t * GGML_RESTRICT vx, const uint8_t * GGML_RESTRICT vy) {
    const int WB = 98, YB = 292;
    const int nb = n / QK_K;

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;
    const uint32_t * grid32  = (const uint32_t *)iq3xxs_grid;

    // constants for unpacking logic
    const uint32_t shifts_val[8] = {0, 7, 14, 21, 0, 7, 14, 21};
    vuint32m1_t v_shifts = __riscv_vle32_v_u32m1(shifts_val, 8);

    const uint32_t gather_idx_val[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    vuint32m1_t v_gather_idx = __riscv_vle32_v_u32m1(gather_idx_val, 8);

    uint32_t aux32[2];
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const uint8_t * xb = vx + (size_t)i * WB;
        const uint8_t * yb = vy + (size_t)i * YB;
        uint16_t xd_h; memcpy(&xd_h, xb + 0, 2);
        float yd; memcpy(&yd, yb + 0, 4);
        const float d = fp16_to_fp32(xd_h) * yd;

        const uint8_t * GGML_RESTRICT q3_indices = xb + 2;            // x[i].qs
        const uint8_t * GGML_RESTRICT metadata   = xb + 2 + QK_K/4;   // x[i].qs + 64
        const int8_t  * GGML_RESTRICT q8         = (const int8_t *)(yb + 4); // y[i].qs

        float block_sum = 0.0f;

        for (int ib = 0; ib < QK_K / 64; ++ib) {
            // Load q8 (64 bytes)
            vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 64);
            q8 += 64;

            // load of metadata via memcpy
            memcpy(aux32, metadata, 2 * sizeof(uint32_t));
            metadata += 2 * sizeof(uint32_t);

            // Load q3 indices and gather magnitudes
            vuint8mf2_t v_q3_idx_u8 = __riscv_vle8_v_u8mf2(q3_indices, 16);
            q3_indices += 16;

            vuint16m1_t v_q3_idx_u16 = __riscv_vwmulu_vx_u16m1(v_q3_idx_u8, 4, 16);
            vuint32m2_t v_q3_magnitudes_u32 = __riscv_vluxei16_v_u32m2(grid32, v_q3_idx_u16, 16);
            vint8m2_t v_q3_magnitudes = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u32m2_u8m2(v_q3_magnitudes_u32));

            // --- Unpacking of Sign Indices ---

            // 1. Load the 2 auxiliary 32-bit integers into a vector
            vuint32m1_t v_aux = __riscv_vle32_v_u32m1(aux32, 2);

            // 2. Broadcast/Gather: replicate aux[0] to first 4 lanes, aux[1] to next 4 lanes
            vuint32m1_t v_aux_expanded = __riscv_vrgather_vv_u32m1(v_aux, v_gather_idx, 8);

            // 3. Apply Shifts and Mask: ((val >> shift) & 127)
            vuint32m1_t v_s_vals_raw = __riscv_vand_vx_u32m1(__riscv_vsrl_vv_u32m1(v_aux_expanded, v_shifts, 8), 127, 8);

            // 4. Narrow to u16 (required for vluxei index) and multiply by 8 (byte offset for u64 table)
            vuint16mf2_t sign_indices_byte_offset = __riscv_vsll_vx_u16mf2(__riscv_vncvt_x_x_w_u16mf2(v_s_vals_raw, 8), 3, 8);

            // 5. Gather Signs
            vuint64m2_t v_s_vals_u64 = __riscv_vluxei16_v_u64m2(signs64, sign_indices_byte_offset, 8);
            vint8m2_t v_s_vals = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(v_s_vals_u64));

            vint8m2_t v_q3_signed = __riscv_vmul_vv_i8m2(v_q3_magnitudes, v_s_vals, 64);
            vint16m4_t v_dot = __riscv_vwmul_vv_i16m4(v_q8, v_q3_signed, 64);

            vint16m2_t v_dot_1 = __riscv_vget_v_i16m4_i16m2(v_dot, 0);
            vint16m2_t v_dot_2 = __riscv_vget_v_i16m4_i16m2(v_dot, 1);

            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t v_sum_1 = __riscv_vwredsum_vs_i16m2_i32m1(v_dot_1, v_zero, 32);
            vint32m1_t v_sum_2 = __riscv_vwredsum_vs_i16m2_i32m1(v_dot_2, v_zero, 32);

            int32_t sum1_i = __riscv_vmv_x_s_i32m1_i32(v_sum_1);
            int32_t sum2_i = __riscv_vmv_x_s_i32m1_i32(v_sum_2);

            const float scale1_f = (float)(2 * (aux32[0] >> 28) + 1);
            const float scale2_f = (float)(2 * (aux32[1] >> 28) + 1);

            block_sum += sum1_i * scale1_f + sum2_i * scale2_f;
        }

        sumf += d * block_sum;
    }
    *s = 0.25f * sumf;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    iq3_xxs_vl256((int)n, s, vx, vy);
}
