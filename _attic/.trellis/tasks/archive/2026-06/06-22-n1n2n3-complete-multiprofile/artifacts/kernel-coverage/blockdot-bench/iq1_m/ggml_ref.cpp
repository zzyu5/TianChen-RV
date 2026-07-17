// ggml's SHIPPED RVV iq1_m_q8_K kernel, _vl128 variant, lifted VERBATIM from
// llama.cpp arch/riscv/quants.c:3161, raw byte offsets + 4-arg adapter.
// block_iq1_m=56B: qs[32]@0, qh[16]@32, scales[8]@48. NO d field (scale reconstructed
// from the 4 u16 in scales via iq1m_scale_t). block_q8_K=292B. iq1_m does NOT use bsums.
// grid=iq1s_grid. IQ1M_DELTA from ggml-common.h.
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <riscv_vector.h>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#define QK_K 256
#ifndef GGML_RESTRICT
#define GGML_RESTRICT __restrict
#endif
static inline float fp16_to_fp32(ggml_half h) { _Float16 f; memcpy(&f, &h, 2); return (float)f; }

static void iq1_m_vl128(int n, float * GGML_RESTRICT s, const uint8_t * GGML_RESTRICT vx, const uint8_t * GGML_RESTRICT vy) {
    const int WB = 56, YB = 292;
    const int nb = n / QK_K;
    iq1m_scale_t scale;
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * xb = vx + (size_t)i * WB;
        const uint8_t * yb = vy + (size_t)i * YB;
        float yd; memcpy(&yd, yb + 0, 4);
        const int8_t  * q8 = (const int8_t *)(yb + 4);
        const uint8_t * qs = xb + 0;        // x[i].qs
        const uint8_t * qh = xb + 32;       // x[i].qh
        const uint16_t * sc = (const uint16_t *)(xb + 48); // x[i].scales

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        vint32m4_t acc1 = __riscv_vmv_v_x_i32m4(0, 16);
        vint32m4_t acc2 = __riscv_vmv_v_x_i32m4(0, 16);
        #pragma GCC unroll 1
        for (int ib = 0; ib < QK_K/128; ib++) {
            const vuint8mf2_t qh_8 = __riscv_vle8_v_u8mf2(qh, 8);
            const vuint16m1_t qh_16_lo = __riscv_vzext_vf2_u16m1(qh_8, 8);
            const vuint16m1_t qh_16_hi = __riscv_vsll_vx_u16m1(qh_16_lo, 8, 8);
            const vuint16m2_t qhb = __riscv_vzext_vf2_u16m2(
                __riscv_vreinterpret_v_u16m1_u8m1(__riscv_vor_vv_u16m1(qh_16_lo, qh_16_hi, 8)), 16);
            qh += 8;
            const vuint16m2_t qsb = __riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(&qs[0], 16), 16);
            const vuint16m2_t shift = __riscv_vreinterpret_v_u32m2_u16m2(__riscv_vmv_v_x_u32m2(0x00040008, 8));
            vuint16m2_t index = __riscv_vor_vv_u16m2(qsb, __riscv_vand_vx_u16m2(__riscv_vsll_vv_u16m2(qhb, shift, 16), 0x700, 16), 16);
            index = __riscv_vsll_vx_u16m2(index, 3, 16);
            qs += 16;
            const vbool8_t mask = __riscv_vmsgtu_vx_u16m2_b8(
                __riscv_vand_vv_u16m2(qhb, __riscv_vreinterpret_v_u32m2_u16m2(__riscv_vmv_v_x_u32m2(0x00800008, 8)), 16), 0, 16);
            const vint64m8_t delta_pos = __riscv_vmv_v_x_i64m8(0x0101010101010101, 16);
            const vint8m8_t delta = __riscv_vreinterpret_v_i64m8_i8m8(
                __riscv_vmerge_vxm_i64m8(delta_pos, 0xffffffffffffffff, mask, 16));
            {
                const vint8m4_t iq1b = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vreinterpret_v_u64m4_i64m4(
                    __riscv_vluxei16_v_u64m4(iq1s_grid, __riscv_vget_v_u16m2_u16m1(index, 0), 8)));
                {
                    const vint8m2_t q8b = __riscv_vle8_v_i8m2(q8, 32); q8 += 32;
                    const vint16m4_t lsum1 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m4_i8m2(iq1b, 0), q8b, 32);
                    const vint16m4_t lsum2 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m8_i8m2(delta, 0), q8b, 32);
                    const int16_t ls_0 = 2*((sc[0] >> 0) & 0x7) + 1;
                    const int16_t ls_1 = 2*((sc[0] >> 3) & 0x7) + 1;
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_0, __riscv_vget_v_i16m4_i16m2(lsum1, 0), 16);
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_1, __riscv_vget_v_i16m4_i16m2(lsum1, 1), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_0, __riscv_vget_v_i16m4_i16m2(lsum2, 0), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_1, __riscv_vget_v_i16m4_i16m2(lsum2, 1), 16);
                }
                __asm__ __volatile__("" ::: "memory");
                {
                    const vint8m2_t q8b = __riscv_vle8_v_i8m2(q8, 32); q8 += 32;
                    const vint16m4_t lsum1 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m4_i8m2(iq1b, 1), q8b, 32);
                    const vint16m4_t lsum2 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m8_i8m2(delta, 1), q8b, 32);
                    const int16_t ls_0 = 2*((sc[0] >> 6) & 0x7) + 1;
                    const int16_t ls_1 = 2*((sc[0] >> 9) & 0x7) + 1;
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_0, __riscv_vget_v_i16m4_i16m2(lsum1, 0), 16);
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_1, __riscv_vget_v_i16m4_i16m2(lsum1, 1), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_0, __riscv_vget_v_i16m4_i16m2(lsum2, 0), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_1, __riscv_vget_v_i16m4_i16m2(lsum2, 1), 16);
                }
                sc += 1;
            }
            __asm__ __volatile__("" ::: "memory");
            {
                const vint8m4_t iq1b = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vreinterpret_v_u64m4_i64m4(
                    __riscv_vluxei16_v_u64m4(iq1s_grid, __riscv_vget_v_u16m2_u16m1(index, 1), 8)));
                {
                    const vint8m2_t q8b = __riscv_vle8_v_i8m2(q8, 32); q8 += 32;
                    const vint16m4_t lsum1 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m4_i8m2(iq1b, 0), q8b, 32);
                    const vint16m4_t lsum2 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m8_i8m2(delta, 2), q8b, 32);
                    const int16_t ls_0 = 2*((sc[0] >> 0) & 0x7) + 1;
                    const int16_t ls_1 = 2*((sc[0] >> 3) & 0x7) + 1;
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_0, __riscv_vget_v_i16m4_i16m2(lsum1, 0), 16);
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_1, __riscv_vget_v_i16m4_i16m2(lsum1, 1), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_0, __riscv_vget_v_i16m4_i16m2(lsum2, 0), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_1, __riscv_vget_v_i16m4_i16m2(lsum2, 1), 16);
                }
                __asm__ __volatile__("" ::: "memory");
                {
                    const vint8m2_t q8b = __riscv_vle8_v_i8m2(q8, 32); q8 += 32;
                    const vint16m4_t lsum1 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m4_i8m2(iq1b, 1), q8b, 32);
                    const vint16m4_t lsum2 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m8_i8m2(delta, 3), q8b, 32);
                    const int16_t ls_0 = 2*((sc[0] >> 6) & 0x7) + 1;
                    const int16_t ls_1 = 2*((sc[0] >> 9) & 0x7) + 1;
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_0, __riscv_vget_v_i16m4_i16m2(lsum1, 0), 16);
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_1, __riscv_vget_v_i16m4_i16m2(lsum1, 1), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_0, __riscv_vget_v_i16m4_i16m2(lsum2, 0), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_1, __riscv_vget_v_i16m4_i16m2(lsum2, 1), 16);
                }
                sc += 1;
            }
        }
        vint32m1_t one = __riscv_vmv_v_x_i32m1(0, 1);
        int sumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(acc1, one, 16));
        int sumi2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(acc2, one, 16));
        sumf += yd * fp16_to_fp32(scale.f16) * (sumi1 + IQ1M_DELTA * sumi2);
    }
    *s = sumf;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    iq1_m_vl128((int)n, s, vx, vy);
}
