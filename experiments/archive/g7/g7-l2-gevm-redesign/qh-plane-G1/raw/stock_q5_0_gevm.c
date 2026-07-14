#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
typedef uint16_t ggml_half;
#define QK5_0 32
#define QK8_0 32
typedef struct { ggml_half d; uint8_t qh[4]; uint8_t qs[QK5_0/2]; } block_q5_0;
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;
static inline float fp16(ggml_half h){ _Float16 f; __builtin_memcpy(&f,&h,2); return (float)f; }
#define GGML_RESTRICT __restrict
void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    int ib = 0;
    float sumf = 0;
    const block_q5_0 * GGML_RESTRICT x = (const block_q5_0*)vx;
    const block_q8_0 * GGML_RESTRICT y = (const block_q8_0*)vy;
    size_t vl;
    size_t vlenb = __riscv_vlenb();
    for (; ib < nb; ++ib) {
        vl = qk / 2;
        vuint8m1_t v0 = __riscv_vle8_v_u8m1(x[ib].qs, vl);
        vint8m1_t v0l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(v0, 0x0F, vl));
        vint8m1_t v0h = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(v0, 4, vl));
        vint8m2_t v0c;
        if (vlenb == 16) { v0c = __riscv_vcreate_v_i8m1_i8m2(v0l, v0h); }
        else { v0l = __riscv_vslideup_vx_i8m1(v0l, v0h, 16, 32); v0c = __riscv_vlmul_ext_v_i8m1_i8m2(v0l); }
        vl = qk;
        vbool4_t qh = __riscv_vlm_v_b4(x[ib].qh, vl);
        qh = __riscv_vmnand_mm_b4(qh, qh, vl);
        vint8m2_t v0f = __riscv_vsub_vx_i8m2_mu(qh, v0c, v0c, 0x10, vl);
        vint8m2_t v1 = __riscv_vle8_v_i8m2(y[ib].qs, vl);
        vint16m4_t mul = __riscv_vwmul_vv_i16m4(v0f, v1, vl);
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t sum = __riscv_vwredsum_vs_i16m4_i32m1(mul, zero, vl);
        int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);
        sumf += (fp16(x[ib].d) * fp16(y[ib].d)) * sumi;
    }
    *s = sumf;
}
