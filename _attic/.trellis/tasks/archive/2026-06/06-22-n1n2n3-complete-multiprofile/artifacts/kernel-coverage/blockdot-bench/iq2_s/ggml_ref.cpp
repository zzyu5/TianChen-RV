// ggml's SHIPPED RVV iq2_s_q8_K kernel, _vl128 variant, lifted VERBATIM from
// llama.cpp arch/riscv/quants.c:3754, raw byte offsets + 4-arg adapter.
// block_iq2_s=82B (d@0 fp16, qs[64]@2 [grid-lo + embedded signs@qs+32], qh[8]@66,
// scales[8]@74). block_q8_K=292B. grid=iq2s_grid(1024). Signs are NOT a separate
// keven table here — they live in qs[32..63]; only iq2s_grid is used.
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <riscv_vector.h>

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#define QK_K 256
#ifndef GGML_RESTRICT
#define GGML_RESTRICT __restrict
#endif
static inline float fp16_to_fp32(uint16_t h) { _Float16 f; memcpy(&f, &h, 2); return (float)f; }

static void iq2_s_vl128(int n, float * GGML_RESTRICT s, const uint8_t * GGML_RESTRICT vx, const uint8_t * GGML_RESTRICT vy) {
    const int WB = 82, YB = 292;
    const int nb = n / QK_K;
    const uint64_t * grid64 = (const uint64_t *)iq2s_grid;

    vuint8m2_t v_ids = __riscv_vid_v_u8m2(32);
    vuint8m2_t v_sign_gather_indices = __riscv_vsrl_vx_u8m2(v_ids, 3, 32);
    vuint8m2_t v_ones = __riscv_vmv_v_x_u8m2(1, 32);
    vuint8m2_t v_shift_amts = __riscv_vand_vx_u8m2(v_ids, 7, 32);
    vuint8m2_t v_sign_masks = __riscv_vsll_vv_u8m2(v_ones, v_shift_amts, 32);
    uint16_t shift_qh_arr[4] = {11, 9, 7, 5};
    vuint16mf2_t v_shift_qh = __riscv_vle16_v_u16mf2(shift_qh_arr, 4);

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * xb = vx + (size_t)i * WB;
        const uint8_t * yb = vy + (size_t)i * YB;
        uint16_t xd_h; memcpy(&xd_h, xb + 0, 2);
        float yd; memcpy(&yd, yb + 0, 4);
        const float combined_scale = fp16_to_fp32(xd_h) * yd;

        const uint8_t * GGML_RESTRICT qs = xb + 2;        // x[i].qs
        const uint8_t * GGML_RESTRICT qh = xb + 66;       // x[i].qh
        const uint8_t * GGML_RESTRICT scales = xb + 74;   // x[i].scales
        const int8_t  * GGML_RESTRICT q8 = (const int8_t *)(yb + 4);

        const uint8_t * signs_ptr = qs + 32;
        float sum_block = 0.0f;
        for (int ib = 0; ib < 8; ++ib) {
            vuint8mf4_t v_qs_u8 = __riscv_vle8_v_u8mf4(qs, 4);
            qs += 4;
            uint8_t qh_val = *qh++;
            vuint8mf4_t v_qh_raw = __riscv_vmv_v_x_u8mf4(qh_val, 4);
            vuint16mf2_t v_qh_u16 = __riscv_vwcvtu_x_x_v_u16mf2(v_qh_raw, 4);
            vuint16mf2_t v_qh_mf2 = __riscv_vsll_vv_u16mf2(v_qh_u16, v_shift_qh, 4);
            v_qh_mf2 = __riscv_vand_vx_u16mf2(v_qh_mf2, 0x1800, 4);
            vuint16mf2_t v_qs_u16_mf2 = __riscv_vwcvtu_x_x_v_u16mf2(v_qs_u8, 4);
            vuint16mf2_t v_qs_u16 = __riscv_vsll_vx_u16mf2(v_qs_u16_mf2, 3, 4);
            vuint16mf2_t v_grid_offsets = __riscv_vor_vv_u16mf2(v_qs_u16, v_qh_mf2, 4);
            vint8m2_t v_grid_i8 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(__riscv_vluxei16_v_u64m2(grid64, v_grid_offsets, 4)));
            vuint8mf4_t v_signs_raw = __riscv_vle8_v_u8mf4(signs_ptr, 4);
            signs_ptr += 4;
            vuint8m2_t v_signs_source = __riscv_vlmul_ext_v_u8mf4_u8m2(v_signs_raw);
            vuint8m2_t v_signs_bcast = __riscv_vrgather_vv_u8m2(v_signs_source, v_sign_gather_indices, 32);
            vuint8m2_t v_sign_bits = __riscv_vand_vv_u8m2(v_signs_bcast, v_sign_masks, 32);
            vbool4_t m_negative = __riscv_vmsne_vx_u8m2_b4(v_sign_bits, 0, 32);
            vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 32);
            q8 += 32;
            vint8m2_t v_q8_signed = __riscv_vrsub_vx_i8m2_mu(m_negative, v_q8, v_q8, 0, 32);
            vint16m4_t v_dot = __riscv_vwmul_vv_i16m4(v_grid_i8, v_q8_signed, 32);
            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);
            int32_t s0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
                __riscv_vget_v_i16m4_i16m2(v_dot, 0), v_zero, 16));
            int32_t s1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
                __riscv_vget_v_i16m4_i16m2(v_dot, 1), v_zero, 16));
            uint8_t sc = *scales++;
            sum_block += s0 * (2 * (sc & 0xF) + 1);
            sum_block += s1 * (2 * (sc >> 4)  + 1);
        }
        sumf += sum_block * combined_scale;
    }
    *s = 0.125f * sumf;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    iq2_s_vl128((int)n, s, vx, vy);
}
