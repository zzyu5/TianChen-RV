// ggml's SHIPPED RVV iq2_s_q8_K kernel, _vl256 variant (k1/VLEN256 dispatch,
// case 256), lifted VERBATIM from llama.cpp arch/riscv/quants.c:3845, reframed
// to raw byte offsets + 4-arg adapter.
// block_iq2_s=82B (d@0 fp16, qs[64]@2 [grid-lo + embedded signs@qs+32=xb+34],
// qh[8]@66, scales[8]@74). block_q8_K=292B. grid=iq2s_grid(1024).
// Signs are EMBEDDED in qs[32..63] (NOT a separate keven table); only iq2s_grid
// is used. sign_gather_indices_arr/sign_bit_masks_arr lifted verbatim from
// quants.c:3744 (file-static, not in ggml-common.h).
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

static const uint8_t sign_gather_indices_arr[64] = {
    0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5, 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7
};

static const uint8_t sign_bit_masks_arr[64] = {
    1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,
    1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128
};

static void iq2_s_vl256(int n, float * GGML_RESTRICT s, const uint8_t * GGML_RESTRICT vx, const uint8_t * GGML_RESTRICT vy) {
    const int WB = 82, YB = 292;
    const int nb = n / QK_K;
    const uint64_t * grid64 = (const uint64_t *)iq2s_grid;

    // --- Pre-load Constants ---
    uint16_t gather_qh_arr[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    vuint16mf2_t v_gather_qh = __riscv_vle16_v_u16mf2(gather_qh_arr, 8);
    uint16_t shift_qh_arr[8] = {11, 9, 7, 5, 11, 9, 7, 5};
    vuint16mf2_t v_shift_qh = __riscv_vle16_v_u16mf2(shift_qh_arr, 8);

    // Constants for sign extraction
    vuint8m2_t v_sign_gather_indices = __riscv_vle8_v_u8m2(sign_gather_indices_arr, 64);
    vuint8m2_t v_sign_masks = __riscv_vle8_v_u8m2(sign_bit_masks_arr, 64);

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

        for (int ib = 0; ib < 4; ++ib) {
            // Combine low + high bits
            vuint8mf4_t v_qs_u8 = __riscv_vle8_v_u8mf4(qs, 8);
            qs += 8;
            uint16_t qh_val;
            memcpy(&qh_val, qh, 2);
            qh += 2;
            vuint8mf8_t v_qh_raw = __riscv_vle8_v_u8mf8((const uint8_t*)&qh_val, 2);
            vuint16mf4_t v_qh_u16 = __riscv_vwcvtu_x_x_v_u16mf4(v_qh_raw, 2);
            vuint16mf2_t v_qh_u16_ext = __riscv_vlmul_ext_v_u16mf4_u16mf2(v_qh_u16);
            vuint16mf2_t v_qh_expanded = __riscv_vrgather_vv_u16mf2(v_qh_u16_ext, v_gather_qh, 8);
            v_qh_expanded = __riscv_vsll_vv_u16mf2(v_qh_expanded, v_shift_qh, 8);

            // Mask: We want bits 11-12. 0x1800 = 0001 1000 0000 0000
            v_qh_expanded = __riscv_vand_vx_u16mf2(v_qh_expanded, 0x1800, 8);
            vuint16mf2_t v_qs_u16 = __riscv_vwcvtu_x_x_v_u16mf2(v_qs_u8, 8);

            // Multiply by 8 to get byte offset, instead of element offset
            v_qs_u16 = __riscv_vsll_vx_u16mf2(v_qs_u16, 3, 8);
            vuint16mf2_t v_grid_offsets = __riscv_vor_vv_u16mf2(v_qs_u16, v_qh_expanded, 8);

            // Lookup Grid using Byte Offsets
            vuint64m2_t v_grid_vals = __riscv_vluxei16_v_u64m2(grid64, v_grid_offsets, 8);

            vuint8m2_t v_grid_u8 = __riscv_vreinterpret_v_u64m2_u8m2(v_grid_vals);
            vint8m2_t v_grid_i8 = __riscv_vreinterpret_v_u8m2_i8m2(v_grid_u8);

            // Load signs and generate sign mask
            vuint8mf4_t v_signs_raw = __riscv_vle8_v_u8mf4(signs_ptr, 8);
            signs_ptr += 8;

            vuint8m2_t v_signs_source = __riscv_vlmul_ext_v_u8mf4_u8m2(v_signs_raw);
            vuint8m2_t v_signs_bcast = __riscv_vrgather_vv_u8m2(v_signs_source, v_sign_gather_indices, 64);

            vuint8m2_t v_sign_bits = __riscv_vand_vv_u8m2(v_signs_bcast, v_sign_masks, 64);
            vbool4_t m_negative = __riscv_vmsne_vx_u8m2_b4(v_sign_bits, 0, 64);

            vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 64);
            q8 += 64;

            vint8m2_t v_q8_signed = __riscv_vrsub_vx_i8m2_mu(m_negative, v_q8, v_q8, 0, 64);
            vint16m4_t v_dot = __riscv_vwmul_vv_i16m4(v_grid_i8, v_q8_signed, 64);

            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);

            int32_t s0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                __riscv_vget_v_i16m4_i16m1(v_dot, 0), v_zero, 16));
            int32_t s1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                __riscv_vget_v_i16m4_i16m1(v_dot, 1), v_zero, 16));
            int32_t s2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                __riscv_vget_v_i16m4_i16m1(v_dot, 2), v_zero, 16));
            int32_t s3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                __riscv_vget_v_i16m4_i16m1(v_dot, 3), v_zero, 16));

            uint8_t sc0 = scales[0];
            uint8_t sc1 = scales[1];
            scales += 2;

            sum_block += s0 * (2 * (sc0 & 0xF) + 1);
            sum_block += s1 * (2 * (sc0 >> 4)  + 1);
            sum_block += s2 * (2 * (sc1 & 0xF) + 1);
            sum_block += s3 * (2 * (sc1 >> 4)  + 1);
        }
        sumf += sum_block * combined_scale;
    }
    *s = 0.125f * sumf;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    iq2_s_vl256((int)n, s, vx, vy);
}
