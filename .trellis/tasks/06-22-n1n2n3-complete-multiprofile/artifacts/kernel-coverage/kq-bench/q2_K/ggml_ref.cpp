// ggml's SHIPPED RVV q2_K_q8_K kernel, VLEN256 variant (_vl256), verbatim from
// llama.cpp quants.c. 4-arg ABI adapter kern_ggml.
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <riscv_vector.h>

#define QK_K 256
typedef _Float16 ggml_half;

typedef struct {
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    struct { ggml_half d; ggml_half dmin; } dm;
} block_q2_K;
typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K;

static inline float fp16_to_fp32(ggml_half h){ return (float)h; }

static void ggml_vec_dot_q2_K_q8_K_vl256(int n, float *s, const void *vx, const void *vy) {
    const block_q2_K *x = (const block_q2_K *)vx;
    const block_q8_K *y = (const block_q8_K *)vy;
    const int nb = n / QK_K;

    float sumf = 0;
    uint8_t temp_01[32] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q2 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;
        const float dall = y[i].d * fp16_to_fp32(x[i].dm.d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dm.dmin);
        size_t vl = 16;
        vuint8m1_t scales = __riscv_vle8_v_u8m1(sc, vl);
        vuint8m1_t aux    = __riscv_vand_vx_u8m1(scales, 0x0F, vl);
        vint16m1_t q8sums = __riscv_vle16_v_i16m1(y[i].bsums, vl);
        vuint8mf2_t scales_2 = __riscv_vle8_v_u8mf2(sc, vl);
        vuint8mf2_t mins8    = __riscv_vsrl_vx_u8mf2(scales_2, 0x4, vl);
        vint16m1_t  mins     = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(mins8, vl));
        vint32m2_t  prod     = __riscv_vwmul_vv_i32m2(q8sums, mins, vl);
        vint32m1_t  vsums    = __riscv_vredsum_vs_i32m2_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);
        sumf += dmin * __riscv_vmv_x_s_i32m1_i32(vsums);
        vl = 32;
        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vuint8m1_t v_b   = __riscv_vle8_v_u8m1(temp_01, vl);
        uint8_t is = 0;
        int isum = 0;
        for (int j = 0; j < QK_K / 128; ++j) {
            vuint8m1_t q2_x = __riscv_vle8_v_u8m1(q2, vl);
            vuint8m1_t q2_0 = __riscv_vand_vx_u8m1(q2_x, 0x03, vl);
            vuint8m1_t q2_1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x2, vl), 0x03, vl);
            vuint8m1_t q2_2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x4, vl), 0x03, vl);
            vuint8m1_t q2_3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x6, vl), 0x03, vl);
            vuint8m1_t sc0 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 0 + is, vl), vl);
            vuint8m1_t sc1 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 2 + is, vl), vl);
            vuint8m1_t sc2 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 4 + is, vl), vl);
            vuint8m1_t sc3 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 6 + is, vl), vl);
            vint16m2_t p0 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_0, sc0, vl));
            vint16m2_t p1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_1, sc1, vl));
            vint16m2_t p2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_2, sc2, vl));
            vint16m2_t p3 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_3, sc3, vl));
            vint8m1_t q8_0 = __riscv_vle8_v_i8m1(q8, vl);
            vint8m1_t q8_1 = __riscv_vle8_v_i8m1(q8 + 32, vl);
            vint8m1_t q8_2 = __riscv_vle8_v_i8m1(q8 + 64, vl);
            vint8m1_t q8_3 = __riscv_vle8_v_i8m1(q8 + 96, vl);
            vint32m4_t s0 = __riscv_vwmul_vv_i32m4(p0, __riscv_vwcvt_x_x_v_i16m2(q8_0, vl), vl);
            vint32m4_t s1 = __riscv_vwmul_vv_i32m4(p1, __riscv_vwcvt_x_x_v_i16m2(q8_1, vl), vl);
            vint32m4_t s2 = __riscv_vwmul_vv_i32m4(p2, __riscv_vwcvt_x_x_v_i16m2(q8_2, vl), vl);
            vint32m4_t s3 = __riscv_vwmul_vv_i32m4(p3, __riscv_vwcvt_x_x_v_i16m2(q8_3, vl), vl);
            vint32m1_t isum0 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(s0, s1, vl), vzero, vl);
            vint32m1_t isum1 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(s2, s3, vl), isum0, vl);
            isum += __riscv_vmv_x_s_i32m1_i32(isum1);
            q2 += 32;
            q8 += 128;
            is = 8;
        }
        sumf += dall * isum;
    }
    *s = sumf;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    ggml_vec_dot_q2_K_q8_K_vl256((int)n, s, vx, vy);
}
