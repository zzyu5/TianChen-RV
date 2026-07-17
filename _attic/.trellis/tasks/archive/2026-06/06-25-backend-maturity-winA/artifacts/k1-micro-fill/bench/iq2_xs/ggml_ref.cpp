// ggml's SHIPPED RVV iq2_xs_q8_K kernel, _vl256 variant (k1/VLEN256 dispatch,
// case 256), lifted VERBATIM from llama.cpp arch/riscv/quants.c:4347, reframed
// to raw byte offsets + 4-arg adapter.
// block_iq2_xs=74B (d@0 fp16, qs[32]u16@2, scales[8]@66). block_q8_K=292B.
// grid=iq2xs_grid; keven_signs_q2xs[1024] lifted verbatim from quants.c (NOT in
// ggml-common.h). Uses vluxei16 grid+sign gather.
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

static void iq2_xs_vl256(int n, float * GGML_RESTRICT s, const uint8_t * GGML_RESTRICT vx, const uint8_t * GGML_RESTRICT vy) {
    const int WB = 74, YB = 292;
    const int nb = n / QK_K;
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;
    const uint64_t * grid64  = (const uint64_t *)iq2xs_grid;

    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const uint8_t * xb = vx + (size_t)i * WB;
        const uint8_t * yb = vy + (size_t)i * YB;
        uint16_t xd_h; memcpy(&xd_h, xb + 0, 2);
        float yd; memcpy(&yd, yb + 0, 4);
        const float d = fp16_to_fp32(xd_h) * yd;
        const uint16_t * GGML_RESTRICT qs = (const uint16_t *)(xb + 2);
        const int8_t   * GGML_RESTRICT q8 = (const int8_t *)(yb + 4);
        const uint8_t  * GGML_RESTRICT scales = xb + 66;

        int32_t sum_int = 0;

        for (int ib128 = 0; ib128 < 2; ++ib128) {

            vuint16m1_t v_qs = __riscv_vle16_v_u16m1(qs, 16);
            qs += 16;

            // Prepare offsets for grid and signs
            vuint16m1_t vidx_grid = __riscv_vsll_vx_u16m1(__riscv_vand_vx_u16m1(v_qs, 511, 16), 3, 16);
            vuint16m1_t vidx_sign = __riscv_vsll_vx_u16m1(__riscv_vsrl_vx_u16m1(v_qs, 9, 16), 3, 16);

            // Indexed load 128 weights (16 x 8-byte chunks)
            vuint64m4_t vq2_64 = __riscv_vluxei16_v_u64m4(grid64, vidx_grid, 16);
            vuint64m4_t vs2_64 = __riscv_vluxei16_v_u64m4(signs64, vidx_sign, 16);

            vint8m4_t q2u = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vq2_64));
            vint8m4_t q2s = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vs2_64));

            // Apply signs to get dequantized IQ2 values
            vint8m4_t q2_final = __riscv_vmul_vv_i8m4(q2u, q2s, 128);
            asm volatile("" ::: "memory");

            // Load corresponding Q8 weights
            vint8m4_t q8v = __riscv_vle8_v_i8m4(q8, 128);
            q8 += 128;

            vint16m8_t prod = __riscv_vwmul_vv_i16m8(q2_final, q8v, 128);
            asm volatile("" ::: "memory");

            uint8_t sc0 = scales[0];
            uint8_t sc1 = scales[1];
            uint8_t sc2 = scales[2];
            uint8_t sc3 = scales[3];
            scales += 4;

            vint32m1_t zero_vec = __riscv_vmv_v_x_i32m1(0, 1);

            // 9. Reduce each 16-element chunk and apply corresponding nibble scale

            int32_t s0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 0), zero_vec, 16));
            sum_int += s0 * ((sc0 & 0x0F) * 2 + 1);

            int32_t s1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 1), zero_vec, 16));
            sum_int += s1 * ((sc0 >> 4) * 2 + 1);

            int32_t s2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 2), zero_vec, 16));
            sum_int += s2 * ((sc1 & 0x0F) * 2 + 1);

            int32_t s3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 3), zero_vec, 16));
            sum_int += s3 * ((sc1 >> 4) * 2 + 1);

            int32_t s4 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 4), zero_vec, 16));
            sum_int += s4 * ((sc2 & 0x0F) * 2 + 1);

            int32_t s5 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 5), zero_vec, 16));
            sum_int += s5 * ((sc2 >> 4) * 2 + 1);

            int32_t s6 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 6), zero_vec, 16));
            sum_int += s6 * ((sc3 & 0x0F) * 2 + 1);

            int32_t s7 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 7), zero_vec, 16));
            sum_int += s7 * ((sc3 >> 4) * 2 + 1);
        }

        sumf += d * (float)sum_int;
    }
    *s = 0.125f * sumf;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    iq2_xs_vl256((int)n, s, vx, vy);
}
