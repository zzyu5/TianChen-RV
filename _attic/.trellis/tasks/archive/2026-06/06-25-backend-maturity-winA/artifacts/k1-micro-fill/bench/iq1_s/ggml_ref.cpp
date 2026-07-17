// ggml's SHIPPED RVV iq1_s_q8_K kernel, _vl256 variant (k1/VLEN256 dispatch,
// case 256), lifted VERBATIM from llama.cpp arch/riscv/quants.c:2859, reframed
// to raw byte offsets + 4-arg adapter.
// block_iq1_s=50B (d@0 fp16, qs[32]@2, qh[16]u16@34). block_q8_K=292B.
// iq1_s USES y.bsums (yb+260, int16). grid=iq1s_grid. IQ1S_DELTA from ggml-common.h.
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

static void iq1_s_vl256(int n, float * GGML_RESTRICT s, const uint8_t * GGML_RESTRICT vx, const uint8_t * GGML_RESTRICT vy) {
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

        // Load qh once for the entire superblock.
        vuint16mf2_t qh = __riscv_vle16_v_u16mf2(x_qh, 8);

        // Calculate ls.
        vuint16mf2_t temp = __riscv_vsrl_vx_u16mf2(qh, 12, 8);
        temp = __riscv_vand_vx_u16mf2(temp, 7, 8);
        vint32m1_t ls = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vwmulu_vx_u32m1(temp, 2, 8));
        ls = __riscv_vadd_vx_i32m1(ls, 1, 8);

        // Calculate delta.
        vbool32_t mask = __riscv_vmseq_vx_u16mf2_b32(__riscv_vand_vx_u16mf2(qh, 0x8000, 8), 0, 8);
        vint32m1_t delta_neg = __riscv_vmv_v_x_i32m1(-1, 8);
        vint32m1_t delta_pos = __riscv_vmv_v_x_i32m1(1, 8);
        vint32m1_t delta = __riscv_vmerge_vvm_i32m1(delta_neg, delta_pos, mask, 8);

        // Load qs.
        vuint8m1_t qs = __riscv_vle8_v_u8m1(x_qs, 32);

        // Prepare the indices.
        const uint64_t shift = 0x0009000600030000;
        vuint16m2_t qh_shift = __riscv_vreinterpret_v_u64m2_u16m2(__riscv_vmv_v_x_u64m2(shift, 8));
        vuint16m2_t qh_gather_index = __riscv_vreinterpret_v_i16m2_u16m2(
            __riscv_vdiv_vx_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vid_v_u16m2(32)), 4, 32));
        vuint16m2_t qh_ext = __riscv_vlmul_ext_v_u16m1_u16m2(__riscv_vlmul_ext_v_u16mf2_u16m1(qh));
        vuint16m2_t qh_index = __riscv_vrgather_vv_u16m2(qh_ext, qh_gather_index, 32);
        qh_index = __riscv_vsrl_vv_u16m2(qh_index, qh_shift, 32);
        qh_index = __riscv_vand_vx_u16m2(qh_index, 7, 32);
        qh_index = __riscv_vsll_vx_u16m2(qh_index, 8, 32);
        qh_index = __riscv_vor_vv_u16m2(qh_index, __riscv_vzext_vf2_u16m2(qs, 32), 32);
        vuint16m2_t index = __riscv_vsll_vx_u16m2(qh_index, 3, 32);

        // Final lsums.
        int32_t lsums_s[8];
        vint32m1_t one_scalar = __riscv_vmv_v_x_i32m1(0, 1);

        // Sub-blocks 1-4
        {
            vuint16m1_t grid_index0 = __riscv_vget_v_u16m2_u16m1(index, 0);
            vint8m4_t grid0 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t*)iq1s_grid, grid_index0, 16));
            vint8m4_t q80 = __riscv_vle8_v_i8m4(y_qs, 128);
            vint16m8_t lsum0 = __riscv_vwmul_vv_i16m8(grid0, q80, 128);
            lsums_s[0] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum0, 0), one_scalar, 32));
            lsums_s[1] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum0, 1), one_scalar, 32));
            lsums_s[2] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum0, 2), one_scalar, 32));
            lsums_s[3] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum0, 3), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        // Sub-blocks 5-8
        {
            vuint16m1_t grid_index1 = __riscv_vget_v_u16m2_u16m1(index, 1);
            vint8m4_t grid1 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t*)iq1s_grid, grid_index1, 16));
            vint8m4_t q81 = __riscv_vle8_v_i8m4(&y_qs[128], 128);
            vint16m8_t lsum1 = __riscv_vwmul_vv_i16m8(grid1, q81, 128);
            lsums_s[4] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum1, 0), one_scalar, 32));
            lsums_s[5] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum1, 1), one_scalar, 32));
            lsums_s[6] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum1, 2), one_scalar, 32));
            lsums_s[7] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum1, 3), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        vint32m1_t lsums = __riscv_vle32_v_i32m1(&lsums_s[0], 8);

        // Calculate the bsums.
        vint16m1_t bsums_0 = __riscv_vle16_v_i16m1(y_bsums, 16);
        const vuint32m1_t bsums_i32 = __riscv_vreinterpret_v_u16m1_u32m1(__riscv_vreinterpret_v_i16m1_u16m1(bsums_0));
        const vint16mf2_t bsums_i32_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(bsums_i32, 0, 8));
        const vint16mf2_t bsums_i32_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(bsums_i32, 16, 8));
        const vint32m1_t bsums = __riscv_vwadd_vv_i32m1(bsums_i32_0, bsums_i32_1, 8);

        // Accumulation.
        vint32m1_t sumi_v = __riscv_vmul_vv_i32m1(ls, lsums, 8);
        vint32m1_t sumi1_v = __riscv_vmul_vv_i32m1(__riscv_vmul_vv_i32m1(ls, delta, 8), bsums, 8);

        // Update sumf.
        int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(sumi_v, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
        int sumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(sumi1_v, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
        sumf += fp16_to_fp32(xd_h) * yd * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    iq1_s_vl256((int)n, s, vx, vy);
}
