// ggml's SHIPPED RVV iq3_s_q8_K kernel, _vl256 variant (k1/VLEN256 dispatch,
// case 256), lifted VERBATIM from llama.cpp arch/riscv/quants.c:4895, reframed
// to raw byte offsets + 4-arg adapter.
// block_iq3_s=110B (d@0 fp16, qs[64]@2, qh[8]@66, signs[32]@74, scales[4]@106).
// block_q8_K=292B. grid=iq3s_grid(512 u32). Signs are EMBEDDED in x.signs; NO
// keven table. sign_gather_indices_arr/sign_bit_masks_arr lifted verbatim from
// quants.c:3744 (file-static, not in ggml-common.h). Final scale is plain sumf.
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

static void iq3_s_vl256(int n, float * GGML_RESTRICT s, const uint8_t * GGML_RESTRICT vx, const uint8_t * GGML_RESTRICT vy) {
    const int WB = 110, YB = 292;
    const int nb = n / QK_K;

    const uint64_t * grid64 = (const uint64_t *)iq3s_grid;

    // --- Pre-load Constants ---
    const uint16_t qh_bit_shifts_arr[16] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };
    vuint8m2_t v_sign_gather_indices = __riscv_vle8_v_u8m2(sign_gather_indices_arr, 64);
    vuint8m2_t v_sign_masks = __riscv_vle8_v_u8m2(sign_bit_masks_arr, 64);
    vuint16m1_t v_qh_shifts = __riscv_vle16_v_u16m1(qh_bit_shifts_arr, 16);

    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const uint8_t * xb = vx + (size_t)i * WB;
        const uint8_t * yb = vy + (size_t)i * YB;
        uint16_t xd_h; memcpy(&xd_h, xb + 0, 2);
        float yd; memcpy(&yd, yb + 0, 4);
        const float d = fp16_to_fp32(xd_h);
        const float combined_scale = d * yd;

        const uint8_t * GGML_RESTRICT qs     = xb + 2;     // x[i].qs
        const uint8_t * GGML_RESTRICT qh     = xb + 66;    // x[i].qh
        const uint8_t * GGML_RESTRICT scales = xb + 106;   // x[i].scales
        const uint8_t * GGML_RESTRICT signs  = xb + 74;    // x[i].signs
        const int8_t  * GGML_RESTRICT q8     = (const int8_t *)(yb + 4);

        float sum_block = 0.0f;

        // Loop: Process 64 weights (16 mini-blocks of 4) per iteration
        for (int ib = 0; ib < 4; ++ib) {

            vuint8mf2_t v_qs_u8 = __riscv_vle8_v_u8mf2(qs, 16);
            qs += 16;

            uint16_t qh_val;
            memcpy(&qh_val, qh, 2);
            qh += 2;

            vuint16m1_t v_qh_val = __riscv_vmv_v_x_u16m1(qh_val, 16);
            // Extract bits: (qh >> i) & 1
            v_qh_val = __riscv_vsrl_vv_u16m1(v_qh_val, v_qh_shifts, 16);
            v_qh_val = __riscv_vand_vx_u16m1(v_qh_val, 1, 16);

            vuint16m1_t v_qs_u16 = __riscv_vwcvtu_x_x_v_u16m1(v_qs_u8, 16);
            v_qs_u16 = __riscv_vsll_vx_u16m1(v_qs_u16, 2, 16);
            v_qh_val = __riscv_vsll_vx_u16m1(v_qh_val, 10, 16);
            vuint16m1_t v_grid_offsets = __riscv_vor_vv_u16m1(v_qs_u16, v_qh_val, 16);

            // Grid value is 4xuint8
            vuint32m2_t v_grid_packed = __riscv_vluxei16_v_u32m2((const uint32_t *)grid64, v_grid_offsets, 16);
            vuint8m2_t v_grid_u8 = __riscv_vreinterpret_v_u32m2_u8m2(v_grid_packed);
            vuint8mf4_t v_signs_raw = __riscv_vle8_v_u8mf4(signs, 8);
            signs += 8;

            // Generate sign mask
            vuint8m2_t v_signs_source = __riscv_vlmul_ext_v_u8mf4_u8m2(v_signs_raw);
            vuint8m2_t v_signs_bcast = __riscv_vrgather_vv_u8m2(v_signs_source, v_sign_gather_indices, 64);
            vuint8m2_t v_sign_bits = __riscv_vand_vv_u8m2(v_signs_bcast, v_sign_masks, 64);
            vbool4_t m_negative = __riscv_vmsne_vx_u8m2_b4(v_sign_bits, 0, 64);

            vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 64);
            q8 += 64;

            // Apply Signs
            vint8m2_t v_q8_signed = __riscv_vrsub_vx_i8m2_mu(m_negative, v_q8, v_q8, 0, 64);
            vint16m4_t v_dot = __riscv_vwmulsu_vv_i16m4(v_q8_signed, v_grid_u8, 64);

            // Reduction
            vint16m2_t v_dot_lo = __riscv_vget_v_i16m4_i16m2(v_dot, 0);
            vint16m2_t v_dot_hi = __riscv_vget_v_i16m4_i16m2(v_dot, 1);
            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);

            int32_t s_lo = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(v_dot_lo, v_zero, 32));
            int32_t s_hi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(v_dot_hi, v_zero, 32));

            // Apply sub-scales
            uint8_t sc_byte = *scales++;
            int sc_lo = (sc_byte & 0xF) * 2 + 1;
            int sc_hi = (sc_byte >> 4)  * 2 + 1;

            sum_block += s_lo * sc_lo + s_hi * sc_hi;
        }
        sumf += sum_block * combined_scale;
    }
    *s = sumf;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    iq3_s_vl256((int)n, s, vx, vy);
}
