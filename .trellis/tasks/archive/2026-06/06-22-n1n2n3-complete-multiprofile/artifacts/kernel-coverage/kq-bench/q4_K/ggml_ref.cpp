// ggml's SHIPPED RVV q4_K_q8_K kernel, VLEN256 variant (_vl256), verbatim from
// llama.cpp ggml/src/ggml-cpu/arch/riscv/quants.c (the methodology-correct Win-B
// baseline on K1). Wrapped behind a 4-arg ABI adapter kern_ggml to match OUR
// emitted kernel's signature (size_t n, float*, const uint8_t*, const uint8_t*).
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
    uint8_t qs[QK_K/2];
} block_q4_K;
typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K;

static inline float fp16_to_fp32(ggml_half h){ return (float)h; }

static void ggml_vec_dot_q4_K_q8_K_vl256(int n, float *s, const void *vx, const void *vy) {
    assert(n % QK_K == 0);
    const block_q4_K *x = (const block_q4_K *)vx;
    const block_q8_K *y = (const block_q8_K *)vy;
    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
    const uint8_t *scales = (const uint8_t*)&utmp[0];
    const uint8_t *mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        size_t vl = 8;
        const float d = y[i].d * fp16_to_fp32(x[i].dm.d);
        const float dmin = y[i].d * fp16_to_fp32(x[i].dm.dmin);

        vint16mf2_t q8sums_0 = __riscv_vlse16_v_i16mf2(y[i].bsums, 4, vl);
        vint16mf2_t q8sums_1 = __riscv_vlse16_v_i16mf2(y[i].bsums+1, 4, vl);
        vint16mf2_t q8sums   = __riscv_vadd_vv_i16mf2(q8sums_0, q8sums_1, vl);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        vuint8mf4_t mins8  = __riscv_vle8_v_u8mf4(mins, vl);
        vint16mf2_t v_mins = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vzext_vf2_u16mf2(mins8, vl));
        vint32m1_t  prod   = __riscv_vwmul_vv_i32m1(q8sums, v_mins, vl);

        vint32m1_t sumi = __riscv_vredsum_vs_i32m1_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);
        sumf -= dmin * __riscv_vmv_x_s_i32m1_i32(sumi);

        const uint8_t *q4 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        vl = 32;
        int32_t sum_1 = 0;
        int32_t sum_2 = 0;
        vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);
        for (int j = 0; j < QK_K/64; ++j) {
            vuint8m1_t q4_x = __riscv_vle8_v_u8m1(q4, vl);
            vint8m1_t  q8_0 = __riscv_vle8_v_i8m1(q8, vl);
            vint8m1_t  q4_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q4_x, 0x0F, vl));
            vint16m2_t qv_0 = __riscv_vwmul_vv_i16m2(q4_0, q8_0, vl);
            vint16m1_t vs_0 = __riscv_vredsum_vs_i16m2_i16m1(qv_0, vzero, vl);
            sum_1 += __riscv_vmv_x_s_i16m1_i16(vs_0) * scales[2*j+0];
            vint8m1_t  q8_1 = __riscv_vle8_v_i8m1(q8+32, vl);
            vint8m1_t  q4_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(q4_x, 0x04, vl));
            vint16m2_t qv_1 = __riscv_vwmul_vv_i16m2(q4_1, q8_1, vl);
            vint16m1_t vs_1 = __riscv_vredsum_vs_i16m2_i16m1(qv_1, vzero, vl);
            sum_2 += __riscv_vmv_x_s_i16m1_i16(vs_1) * scales[2*j+1];
            q4 += 32;    q8 += 64;
        }
        sumf += d*(sum_1 + sum_2);
    }
    *s = sumf;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    ggml_vec_dot_q4_K_q8_K_vl256((int)n, s, vx, vy);
}
