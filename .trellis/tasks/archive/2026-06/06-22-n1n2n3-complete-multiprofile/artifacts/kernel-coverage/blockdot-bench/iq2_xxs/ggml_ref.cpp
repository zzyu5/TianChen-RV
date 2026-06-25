// ggml's SHIPPED RVV iq2_xxs_q8_K kernel, _vl128 variant, lifted VERBATIM from
// llama.cpp arch/riscv/quants.c:4523, expressed against raw byte offsets + 4-arg
// adapter. block_iq2_xxs=66B (d@0 fp16, qs[32]u16@2). block_q8_K=292B
// (d@0 float32, qs[256]int8@4, bsums[16]int16@260).
// Grids/ksigns come from ggml-common.h (GGML_COMMON_IMPL_C); keven_signs_q2xs[1024]
// is lifted verbatim from quants.c:4235 (NOT in ggml-common.h).
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

// VERBATIM body of ggml_vec_dot_iq2_xxs_q8_K_vl128, x/y reframed as byte pointers.
static void iq2_xxs_vl128(int n, float * GGML_RESTRICT s, const uint8_t * GGML_RESTRICT vx, const uint8_t * GGML_RESTRICT vy) {
    const int WB = 66, YB = 292;
    const int nb = n / QK_K;
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;
    const uint64_t * grid64  = (const uint64_t *)iq2xxs_grid;

    uint32_t shift_constants[4] = {0, 7, 14, 21};
    vuint32m1_t v_shifts = __riscv_vle32_v_u32m1(shift_constants, 4);

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * xb = vx + (size_t)i * WB;
        const uint8_t * yb = vy + (size_t)i * YB;
        uint16_t xd_h; memcpy(&xd_h, xb + 0, 2);
        float yd; memcpy(&yd, yb + 0, 4);
        const float combined_scale = fp16_to_fp32(xd_h) * yd;

        const uint8_t  * GGML_RESTRICT q2_ptr = xb + 2;            // x[i].qs
        const int8_t   * GGML_RESTRICT q8 = (const int8_t *)(yb + 4); // y[i].qs

        float sum = 0.0f;

        #pragma GCC unroll 1
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            vint8m2_t q8_1 = __riscv_vle8_v_i8m2(q8, 32); q8 += 32;
            vint8m2_t q8_2 = __riscv_vle8_v_i8m2(q8, 32); q8 += 32;

            vuint8mf4_t v_raw_q2_1 = __riscv_vle8_v_u8mf4(q2_ptr, 4);
            vuint8mf4_t v_raw_q2_2 = __riscv_vle8_v_u8mf4(q2_ptr + 8, 4);

            vuint16mf2_t vidx_q2_1 = __riscv_vwcvtu_x_x_v_u16mf2(v_raw_q2_1, 4);
            vuint16mf2_t vidx_q2_2 = __riscv_vwcvtu_x_x_v_u16mf2(v_raw_q2_2, 4);

            vidx_q2_1 = __riscv_vsll_vx_u16mf2(vidx_q2_1, 3, 4);
            vidx_q2_2 = __riscv_vsll_vx_u16mf2(vidx_q2_2, 3, 4);

            uint32_t s_packed_1, s_packed_2;
            memcpy(&s_packed_1, q2_ptr + 4, 4);
            memcpy(&s_packed_2, q2_ptr + 12, 4);

            vuint32m1_t v_s_1 = __riscv_vmv_v_x_u32m1(s_packed_1, 4);
            vuint32m1_t v_s_2 = __riscv_vmv_v_x_u32m1(s_packed_2, 4);
            v_s_1 = __riscv_vsrl_vv_u32m1(v_s_1, v_shifts, 4);
            v_s_2 = __riscv_vsrl_vv_u32m1(v_s_2, v_shifts, 4);

            v_s_1 = __riscv_vand_vx_u32m1(v_s_1, 127, 4);
            v_s_2 = __riscv_vand_vx_u32m1(v_s_2, 127, 4);

            vuint16mf2_t vidx_s2_1 = __riscv_vsll_vx_u16mf2(__riscv_vncvt_x_x_w_u16mf2(v_s_1, 4), 3, 4);
            vuint16mf2_t vidx_s2_2 = __riscv_vsll_vx_u16mf2(__riscv_vncvt_x_x_w_u16mf2(v_s_2, 4), 3, 4);

            vuint64m2_t vq2_64_1 = __riscv_vluxei16_v_u64m2(grid64, vidx_q2_1, 4);
            vuint64m2_t vq2_64_2 = __riscv_vluxei16_v_u64m2(grid64, vidx_q2_2, 4);

            vint8m2_t q2_1 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vq2_64_1));
            vint8m2_t q2_2 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vq2_64_2));

            vuint64m2_t vs2_64_1 = __riscv_vluxei16_v_u64m2(signs64, vidx_s2_1, 4);
            vuint64m2_t vs2_64_2 = __riscv_vluxei16_v_u64m2(signs64, vidx_s2_2, 4);
            vint8m2_t s2_1 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vs2_64_1));
            vint8m2_t s2_2 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vs2_64_2));

            vint8m2_t q8s_1 = __riscv_vmul_vv_i8m2(q8_1, s2_1, 32);
            vint8m2_t q8s_2 = __riscv_vmul_vv_i8m2(q8_2, s2_2, 32);

            vint16m4_t dot1 = __riscv_vwmul_vv_i16m4(q8s_1, q2_1, 32);
            vint16m4_t dot2 = __riscv_vwmul_vv_i16m4(q8s_2, q2_2, 32);

            vint32m1_t zero_vec = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t sumv1 = __riscv_vwredsum_vs_i16m4_i32m1(dot1, zero_vec, 32);
            vint32m1_t sumv2 = __riscv_vwredsum_vs_i16m4_i32m1(dot2, zero_vec, 32);

            int32_t scalar_sum1 = __riscv_vmv_x_s_i32m1_i32(sumv1);
            int32_t scalar_sum2 = __riscv_vmv_x_s_i32m1_i32(sumv2);

            int16_t scale1 = 2 * ((s_packed_1 >> 28) & 0xF) + 1;
            int16_t scale2 = 2 * ((s_packed_2 >> 28) & 0xF) + 1;

            sum += scalar_sum1 * scale1 + scalar_sum2 * scale2;
            q2_ptr += 16;
        }
        sumf += sum * combined_scale;
    }
    *s = 0.125f * sumf;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    iq2_xxs_vl128((int)n, s, vx, vy);
}
