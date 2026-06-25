// ggml's SHIPPED RVV q5_K_q8_K kernel. q5_K has NO _vl256 split: the RVV path is
// inline under `#if defined __riscv_v` in the single ggml_vec_dot_q5_K_q8_K
// (quants.c). On K1 (__riscv_v defined) this IS the shipped optimized RVV kernel
// (it uses up to LMUL=8). Body lifted verbatim; 4-arg ABI adapter kern_ggml.
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <riscv_vector.h>

#define QK_K 256
#define K_SCALE_SIZE 12
typedef _Float16 ggml_half;

typedef struct {
    struct { ggml_half d; ggml_half dmin; } dm;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;
typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K;

static inline float fp16_to_fp32(ggml_half h){ return (float)h; }

static void ggml_vec_dot_q5_K_q8_K_rvv(int n, float *s, const void *vx, const void *vy) {
    assert(n % QK_K == 0);
    const block_q5_K *x = (const block_q5_K *)vx;
    const block_q8_K *y = (const block_q8_K *)vy;
    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;
    float sums = 0.0;
    size_t vl;
    for (int i = 0; i < nb; ++i) {
        vl = 8;
        const uint8_t *q5 = x[i].qs;
        const uint8_t *hm = x[i].qh;
        const  int8_t *q8 = y[i].qs;
        const float d = fp16_to_fp32(x[i].dm.d) * y[i].d;
        const float dmin = fp16_to_fp32(x[i].dm.dmin) * y[i].d;

        vint16m1_t q8sums_0 = __riscv_vlse16_v_i16m1(y[i].bsums, 4, vl);
        vint16m1_t q8sums_1 = __riscv_vlse16_v_i16m1(y[i].bsums+1, 4, vl);
        vint16m1_t q8sums = __riscv_vadd_vv_i16m1(q8sums_0, q8sums_1, vl);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        vuint8mf2_t mins8 = __riscv_vle8_v_u8mf2(mins, vl);
        vint16m1_t v_mins = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(mins8, vl));
        vint32m2_t prod = __riscv_vwmul_vv_i32m2(q8sums, v_mins, vl);
        vint32m1_t sumi = __riscv_vredsum_vs_i32m2_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);
        sumf -= dmin * __riscv_vmv_x_s_i32m1_i32(sumi);

        vl = 32;
        int32_t aux32 = 0;
        int is = 0;
        uint8_t m = 1;
        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vuint8m2_t vqh = __riscv_vle8_v_u8m2(hm, vl);
        for (int j = 0; j < QK_K/64; ++j) {
            vuint8m2_t q5_x = __riscv_vle8_v_u8m2(q5, vl);
            vint8m2_t  q8_y1 = __riscv_vle8_v_i8m2(q8, vl);
            vint8m2_t  q8_y2 = __riscv_vle8_v_i8m2(q8+32, vl);
            vint8m2_t q5_a = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vand_vx_u8m2(q5_x, 0x0F, vl));
            vuint8m2_t qh_m1 = __riscv_vand_vx_u8m2(vqh, m, vl);
            vbool4_t vmask_1 = __riscv_vmsne_vx_u8m2_b4(qh_m1, 0, vl);
            vint8m2_t q5_m1 = __riscv_vadd_vx_i8m2_mu(vmask_1, q5_a, q5_a, 16, vl);
            m <<= 1;
            vint8m2_t q5_l = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vsrl_vx_u8m2(q5_x, 0x04, vl));
            vuint8m2_t qh_m2 = __riscv_vand_vx_u8m2(vqh, m, vl);
            vbool4_t vmask_2 = __riscv_vmsne_vx_u8m2_b4(qh_m2, 0, vl);
            vint8m2_t q5_m2 = __riscv_vadd_vx_i8m2_mu(vmask_2, q5_l, q5_l, 16, vl);
            m <<= 1;
            vint16m4_t v0 = __riscv_vwmul_vv_i16m4(q5_m1, q8_y1, vl);
            vint16m4_t v1 = __riscv_vwmul_vv_i16m4(q5_m2, q8_y2, vl);
            vint32m8_t vs1 = __riscv_vwmul_vx_i32m8(v0, scales[is++], vl);
            vint32m8_t vs2 = __riscv_vwmul_vx_i32m8(v1, scales[is++], vl);
            vint32m1_t vacc1 = __riscv_vredsum_vs_i32m8_i32m1(vs1, vzero, vl);
            vint32m1_t vacc2 = __riscv_vredsum_vs_i32m8_i32m1(vs2, vacc1, vl);
            aux32 += __riscv_vmv_x_s_i32m1_i32(vacc2);
            q5 += 32;    q8 += 64;
        }
        sums += aux32 * d;
    }
    *s = sumf+sums;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    ggml_vec_dot_q5_K_q8_K_rvv((int)n, s, vx, vy);
}
