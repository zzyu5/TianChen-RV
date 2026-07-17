// ggml's SHIPPED RVV iq2_xs_q8_K kernel, _vl128 variant, lifted VERBATIM from
// llama.cpp arch/riscv/quants.c:4270, raw byte offsets + 4-arg adapter.
// block_iq2_xs=74B (d@0 fp16, qs[32]u16@2, scales[8]@66). block_q8_K=292B.
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

static void iq2_xs_vl128(int n, float * GGML_RESTRICT s, const uint8_t * GGML_RESTRICT vx, const uint8_t * GGML_RESTRICT vy) {
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
        for (int ib64 = 0; ib64 < QK_K / 64; ++ib64) {
            vuint16m1_t v_qs = __riscv_vle16_v_u16m1(qs, 8);
            qs += 8;
            vuint16m1_t vidx_grid = __riscv_vsll_vx_u16m1(__riscv_vand_vx_u16m1(v_qs, 511, 8), 3, 8);
            vuint16m1_t vidx_sign = __riscv_vsll_vx_u16m1(__riscv_vsrl_vx_u16m1(v_qs, 9, 8), 3, 8);
            vuint64m4_t vq2_64 = __riscv_vluxei16_v_u64m4(grid64, vidx_grid, 8);
            vuint64m4_t vs2_64 = __riscv_vluxei16_v_u64m4(signs64, vidx_sign, 8);
            vint8m4_t q2u = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vq2_64));
            vint8m4_t q2s = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vs2_64));
            vint8m4_t q2_final = __riscv_vmul_vv_i8m4(q2u, q2s, 64);
            asm volatile("" ::: "memory");
            vint8m4_t q8v = __riscv_vle8_v_i8m4(q8, 64);
            q8 += 64;
            vint16m8_t prod = __riscv_vwmul_vv_i16m8(q2_final, q8v, 64);
            asm volatile("" ::: "memory");
            vint32m1_t zero_vec = __riscv_vmv_v_x_i32m1(0, 1);
            int32_t sum0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
                           __riscv_vget_v_i16m8_i16m2(prod, 0), zero_vec, 16));
            int32_t sum1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
                           __riscv_vget_v_i16m8_i16m2(prod, 1), zero_vec, 16));
            int32_t sum2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
                           __riscv_vget_v_i16m8_i16m2(prod, 2), zero_vec, 16));
            int32_t sum3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
                           __riscv_vget_v_i16m8_i16m2(prod, 3), zero_vec, 16));
            const uint8_t scale_byte_1 = scales[0];
            const uint8_t scale_byte_2 = scales[1];
            scales += 2;
            sum_int += sum0 * ((scale_byte_1 & 0x0F) * 2 + 1);
            sum_int += sum1 * ((scale_byte_1 >> 4)   * 2 + 1);
            sum_int += sum2 * ((scale_byte_2 & 0x0F) * 2 + 1);
            sum_int += sum3 * ((scale_byte_2 >> 4)   * 2 + 1);
        }
        sumf += d * sum_int;
    }
    *s = 0.125f * sumf;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    iq2_xs_vl128((int)n, s, vx, vy);
}
