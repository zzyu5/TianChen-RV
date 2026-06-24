// ggml's SHIPPED RVV iq3_s_q8_K kernel, _vl128 variant, lifted VERBATIM from
// llama.cpp arch/riscv/quants.c:4800, raw byte offsets + 4-arg adapter.
// block_iq3_s=110B (d@0 fp16, qs[64]@2, qh[8]@66, signs[32]@74, scales[4]@106).
// block_q8_K=292B. grid=iq3s_grid(512 u32). Signs embedded in x.signs; NO keven table.
// Final scale is plain sumf (no 0.125/0.25).
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

static void iq3_s_vl128(int n, float * GGML_RESTRICT s, const uint8_t * GGML_RESTRICT vx, const uint8_t * GGML_RESTRICT vy) {
    const int WB = 110, YB = 292;
    const int nb = n / QK_K;
    const uint32_t * grid32 = (const uint32_t *)iq3s_grid;

    vuint8mf2_t v_id_8  = __riscv_vid_v_u8mf2(8);
    vuint8m2_t  v_id_32 = __riscv_vid_v_u8m2(32);
    vuint8m2_t v_sign_gather_indices, v_sign_masks;
    {
        vuint8m2_t v_shifts  = __riscv_vand_vx_u8m2(v_id_32, 7, 32);
        vuint8m2_t v_one_32  = __riscv_vmv_v_x_u8m2(1, 32);
        v_sign_gather_indices = __riscv_vsrl_vx_u8m2(v_id_32, 3, 32);
        v_sign_masks          = __riscv_vsll_vv_u8m2(v_one_32, v_shifts, 32);
    }

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
        const uint8_t * GGML_RESTRICT signs  = xb + 74;    // x[i].signs
        const uint8_t * GGML_RESTRICT scales = xb + 106;   // x[i].scales
        const int8_t  * GGML_RESTRICT q8     = (const int8_t *)(yb + 4);

        float sum_block = 0.0f;
        for (int ib = 0; ib < 8; ++ib) {
            vuint8m2_t v_grid_u8;
            {
                vuint8mf2_t v_qs_u8 = __riscv_vle8_v_u8mf2(qs, 8);
                qs += 8;
                uint8_t     qh_val   = *qh++;
                vuint8mf2_t v_qh_val = __riscv_vmv_v_x_u8mf2(qh_val, 8);
                v_qh_val = __riscv_vsrl_vv_u8mf2(v_qh_val, v_id_8, 8);
                v_qh_val = __riscv_vand_vx_u8mf2(v_qh_val, 1, 8);
                vuint16m1_t v_qs_u16 = __riscv_vwcvtu_x_x_v_u16m1(v_qs_u8, 8);
                v_qs_u16 = __riscv_vsll_vx_u16m1(v_qs_u16, 2, 8);
                vuint16m1_t v_qh_u16 = __riscv_vwcvtu_x_x_v_u16m1(v_qh_val, 8);
                v_qh_u16 = __riscv_vsll_vx_u16m1(v_qh_u16, 10, 8);
                vuint16m1_t v_grid_offsets = __riscv_vor_vv_u16m1(v_qs_u16, v_qh_u16, 8);
                vuint32m2_t v_grid_packed = __riscv_vluxei16_v_u32m2(grid32, v_grid_offsets, 8);
                v_grid_u8 = __riscv_vreinterpret_v_u32m2_u8m2(v_grid_packed);
            }
            __asm__ volatile ("" ::: "memory");
            int32_t s_val;
            {
                vuint8mf4_t v_signs_raw  = __riscv_vle8_v_u8mf4(signs, 4);
                signs += 4;
                vuint8m2_t v_signs_source = __riscv_vlmul_ext_v_u8mf4_u8m2(v_signs_raw);
                vuint8m2_t v_signs_bcast  = __riscv_vrgather_vv_u8m2(v_signs_source, v_sign_gather_indices, 32);
                vuint8m2_t v_sign_bits    = __riscv_vand_vv_u8m2(v_signs_bcast, v_sign_masks, 32);
                vbool4_t   m_negative     = __riscv_vmsne_vx_u8m2_b4(v_sign_bits, 0, 32);
                vint8m2_t v_q8        = __riscv_vle8_v_i8m2(q8, 32);
                q8 += 32;
                vint8m2_t  v_q8_signed = __riscv_vrsub_vx_i8m2_mu(m_negative, v_q8, v_q8, 0, 32);
                vint16m4_t v_dot       = __riscv_vwmulsu_vv_i16m4(v_q8_signed, v_grid_u8, 32);
                vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);
                s_val = __riscv_vmv_x_s_i32m1_i32(
                    __riscv_vwredsum_vs_i16m4_i32m1(v_dot, v_zero, 32));
            }
            __asm__ volatile ("" ::: "memory");
            {
                uint8_t sc_byte = scales[ib >> 1];
                int sc_val = (ib & 1) ? (sc_byte >> 4) : (sc_byte & 0xF);
                sc_val = sc_val * 2 + 1;
                sum_block += (float)(s_val * sc_val);
            }
        }
        sumf += sum_block * combined_scale;
    }
    *s = sumf;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    iq3_s_vl128((int)n, s, vx, vy);
}
