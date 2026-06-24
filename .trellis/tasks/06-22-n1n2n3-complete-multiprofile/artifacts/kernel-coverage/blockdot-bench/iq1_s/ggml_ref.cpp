// ggml's SHIPPED RVV iq1_s_q8_K kernel, _vl128 variant, lifted VERBATIM from
// llama.cpp arch/riscv/quants.c:2747, raw byte offsets + 4-arg adapter.
// block_iq1_s=50B (d@0 fp16, qs[32]@2, qh[16]u16@34). block_q8_K=292B.
// iq1_s USES y.bsums (unlike tq2_0). grid=iq1s_grid. IQ1S_DELTA from ggml-common.h.
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

static void iq1_s_vl128(int n, float * GGML_RESTRICT s, const uint8_t * GGML_RESTRICT vx, const uint8_t * GGML_RESTRICT vy) {
    const int WB = 50, YB = 292;
    const int nb = n / QK_K;
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * xb = vx + (size_t)i * WB;
        const uint8_t * yb = vy + (size_t)i * YB;
        const uint16_t * x_qh = (const uint16_t *)(xb + 34);
        const uint8_t  * x_qs = xb + 2;
        const int8_t   * y_qs = (const int8_t *)(yb + 4);
        const int16_t  * y_bsums = (const int16_t *)(yb + 260);
        uint16_t xd_h; memcpy(&xd_h, xb + 0, 2);
        float yd; memcpy(&yd, yb + 0, 4);

        vuint16m1_t qh = __riscv_vle16_v_u16m1(x_qh, 8);
        vuint16m1_t temp = __riscv_vsrl_vx_u16m1(qh, 12, 8);
        temp = __riscv_vand_vx_u16m1(temp, 7, 8);
        vint32m2_t ls = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vwmulu_vx_u32m2(temp, 2, 8));
        ls = __riscv_vadd_vx_i32m2(ls, 1, 8);
        vbool16_t mask = __riscv_vmseq_vx_u16m1_b16(__riscv_vand_vx_u16m1(qh, 0x8000, 8), 0, 8);
        vint32m2_t delta_neg = __riscv_vmv_v_x_i32m2(-1, 8);
        vint32m2_t delta_pos = __riscv_vmv_v_x_i32m2(1, 8);
        vint32m2_t delta = __riscv_vmerge_vvm_i32m2(delta_neg, delta_pos, mask, 8);
        vuint8m2_t qs = __riscv_vle8_v_u8m2(x_qs, 32);
        const uint64_t shift = 0x0009000600030000;
        vuint16m4_t qh_shift = __riscv_vreinterpret_v_u64m4_u16m4(__riscv_vmv_v_x_u64m4(shift, 8));
        vuint16m4_t qh_gather_index = __riscv_vreinterpret_v_i16m4_u16m4(
            __riscv_vdiv_vx_i16m4(__riscv_vreinterpret_v_u16m4_i16m4(__riscv_vid_v_u16m4(32)), 4, 32));
        vuint16m4_t qh_ext = __riscv_vlmul_ext_v_u16m2_u16m4(__riscv_vlmul_ext_v_u16m1_u16m2(qh));
        vuint16m4_t qh_index = __riscv_vrgather_vv_u16m4(qh_ext, qh_gather_index, 32);
        qh_index = __riscv_vsrl_vv_u16m4(qh_index, qh_shift, 32);
        qh_index = __riscv_vand_vx_u16m4(qh_index, 7, 32);
        qh_index = __riscv_vsll_vx_u16m4(qh_index, 8, 32);
        qh_index = __riscv_vor_vv_u16m4(qh_index, __riscv_vzext_vf2_u16m4(qs, 32), 32);
        vuint16m4_t index = __riscv_vsll_vx_u16m4(qh_index, 3, 32);

        int32_t lsums_s[8];
        vint32m1_t one_scalar = __riscv_vmv_v_x_i32m1(0, 1);
        {
            vuint16m1_t gi = __riscv_vget_v_u16m4_u16m1(index, 0);
            vint8m4_t grid0 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t*)iq1s_grid, gi, 8));
            vint8m4_t q80 = __riscv_vle8_v_i8m4(&y_qs[0], 64);
            vint16m8_t lsum0 = __riscv_vwmul_vv_i16m8(grid0, q80, 128);
            lsums_s[0] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 0), one_scalar, 32));
            lsums_s[1] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 1), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        {
            vuint16m1_t gi = __riscv_vget_v_u16m4_u16m1(index, 1);
            vint8m4_t grid0 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t*)iq1s_grid, gi, 8));
            vint8m4_t q80 = __riscv_vle8_v_i8m4(&y_qs[64], 64);
            vint16m8_t lsum0 = __riscv_vwmul_vv_i16m8(grid0, q80, 128);
            lsums_s[2] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 0), one_scalar, 32));
            lsums_s[3] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 1), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        {
            vuint16m1_t gi = __riscv_vget_v_u16m4_u16m1(index, 2);
            vint8m4_t grid0 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t*)iq1s_grid, gi, 8));
            vint8m4_t q80 = __riscv_vle8_v_i8m4(&y_qs[128], 64);
            vint16m8_t lsum0 = __riscv_vwmul_vv_i16m8(grid0, q80, 128);
            lsums_s[4] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 0), one_scalar, 32));
            lsums_s[5] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 1), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        {
            vuint16m1_t gi = __riscv_vget_v_u16m4_u16m1(index, 3);
            vint8m4_t grid0 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t*)iq1s_grid, gi, 8));
            vint8m4_t q80 = __riscv_vle8_v_i8m4(&y_qs[192], 64);
            vint16m8_t lsum0 = __riscv_vwmul_vv_i16m8(grid0, q80, 128);
            lsums_s[6] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 0), one_scalar, 32));
            lsums_s[7] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 1), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        vint32m2_t lsums = __riscv_vle32_v_i32m2(&lsums_s[0], 8);

        vint16m2_t bsums_0 = __riscv_vle16_v_i16m2(y_bsums, 16);
        const vuint32m2_t bsums_i32 = __riscv_vreinterpret_v_u16m2_u32m2(__riscv_vreinterpret_v_i16m2_u16m2(bsums_0));
        const vint16m1_t bsums_i32_0 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vnsrl_wx_u16m1(bsums_i32, 0, 8));
        const vint16m1_t bsums_i32_1 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vnsrl_wx_u16m1(bsums_i32, 16, 8));
        const vint32m2_t bsums = __riscv_vwadd_vv_i32m2(bsums_i32_0, bsums_i32_1, 8);

        vint32m2_t sumi_v = __riscv_vmul_vv_i32m2(ls, lsums, 8);
        vint32m2_t sumi1_v = __riscv_vmul_vv_i32m2(__riscv_vmul_vv_i32m2(ls, delta, 8), bsums, 8);

        int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(sumi_v, __riscv_vmv_v_x_i32m1(0, 1), 8));
        int sumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(sumi1_v, __riscv_vmv_v_x_i32m1(0, 1), 8));
        sumf += fp16_to_fp32(xd_h) * yd * (sumi + IQ1S_DELTA * sumi1);
    }
    *s = sumf;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    iq1_s_vl128((int)n, s, vx, vy);
}
