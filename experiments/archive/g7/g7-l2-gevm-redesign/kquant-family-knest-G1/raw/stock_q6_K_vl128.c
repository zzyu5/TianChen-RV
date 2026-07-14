#include <riscv_vector.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#define QK_K 256
#define K_SCALE_SIZE 12
typedef uint16_t ggml_half;
typedef struct { uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; ggml_half d; ggml_half dmin; } block_q2_K;
typedef struct { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; ggml_half d; } block_q3_K;
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d; } block_q6_K;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;
static inline float GGML_CPU_FP16_TO_FP32(ggml_half h){ return (float)h; }
#define GGML_RESTRICT __restrict
#define restrict __restrict
#define UNUSED(x) (void)(x)
#define NOINLINE __attribute__((noinline))
void ggml_vec_dot_q6_K_q8_K_vl128(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        __builtin_prefetch(&x[i + 1].d, 0, 1);

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

        const uint8_t * restrict q6 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;

        const int8_t * restrict scale = x[i].scales;

        int q6h;
        float ftmp;

        for (int j = 0; j < QK_K/128; ++j) {
            __asm__ __volatile__(
                "addi %[q6h], %[q6], 32\n\t"
                "ld t0, 0(%[scale])\n\t"
                "addi %[scale], %[scale], 8\n\t"
                "slli t6, t0, 1 * 8\n\t"
                "lb zero, 0(%[q6])\n\t"
                "slli t5, t0, 2 * 8\n\t"
                "slli t4, t0, 3 * 8\n\t"
                "lb zero, 0(%[q6h])\n\t"
                "slli t3, t0, 4 * 8\n\t"
                "slli t2, t0, 5 * 8\n\t"
                "lb zero, 0(%[qh])\n\t"
                "lb zero, 31(%[q6h])\n\t"
                "slli t1, t0, 6 * 8\n\t"
                "srai a7, t0, 56\n\t"
                "vsetvli zero, %[vl32], e8, m2\n\t"
                "vle8.v v8, (%[q6])\n\t"
                "srai t6, t6, 56\n\t"
                "srai t5, t5, 56\n\t"
                "srai t4, t4, 56\n\t"
                "srai t3, t3, 56\n\t"
                "vle8.v v10, (%[q6h])\n\t"
                "addi %[q6], %[q6], 64\n\t"
                "slli t0, t0, 7 * 8\n\t"
                "srai t2, t2, 56\n\t"
                "srai t1, t1, 56\n\t"
                "srai t0, t0, 56\n\t"
                "vle8.v v4, (%[qh])\n\t"
                "vsrl.vi v12, v8, 4\n\t"
                "vsrl.vi v14, v10, 4\n\t"
                "lb zero, 0(%[q8])\n\t"
                "vand.vi v8, v8, 0xF\n\t"
                "vand.vi v10, v10, 0xF\n\t"
                "lb zero, 32(%[q8])\n\t"
                "vsll.vi v0, v4, 4\n\t"
                "vsll.vi v2, v4, 2\n\t"
                "lb zero, 64(%[q8])\n\t"
                "vsrl.vi v6, v4, 2\n\t"
                "vand.vx v0, v0, %[mask]\n\t"
                "lb zero, 96(%[q8])\n\t"
                "vand.vx v2, v2, %[mask]\n\t"
                "vand.vx v4, v4, %[mask]\n\t"
                "vand.vx v6, v6, %[mask]\n\t"
                "vor.vv v8, v8, v0\n\t"
                "lb zero, 127(%[q8])\n\t"
                "vor.vv v10, v10, v2\n\t"
                "vor.vv v12, v12, v4\n\t"
                "vor.vv v14, v14, v6\n\t"
                "vsetvli zero, %[vl128], e8, m8\n\t"
                "vle8.v v0, (%[q8])\n\t"
                "vsub.vx v8, v8, %[vl32]\n\t"
                "vsetvli zero, %[vl64], e8, m4\n\t"
                "vwmul.vv v16, v0, v8\n\t"
                "vwmul.vv v24, v4, v12\n\t"
                "vsetivli zero, 16, e16, m2\n\t"
                "vmv.v.x v0, zero\n\t"
                "vwredsum.vs v10, v16, v0\n\t"
                "vwredsum.vs v9, v18, v0\n\t"
                "vwredsum.vs v8, v20, v0\n\t"
                "vwredsum.vs v7, v22, v0\n\t"
                "vwredsum.vs v11, v24, v0\n\t"
                "vwredsum.vs v12, v26, v0\n\t"
                "vwredsum.vs v13, v28, v0\n\t"
                "vwredsum.vs v14, v30, v0\n\t"
                "vsetivli zero, 4, e32, m1\n\t"
                "vmul.vx v0, v10, t0\n\t"
                "vmul.vx v1, v9, t1\n\t"
                "vmacc.vx v0, t2, v8\n\t"
                "vmacc.vx v1, t3, v7\n\t"
                "vmacc.vx v0, t4, v11\n\t"
                "vmacc.vx v1, t5, v12\n\t"
                "vmacc.vx v0, t6, v13\n\t"
                "vmacc.vx v1, a7, v14\n\t"
                "vadd.vv v0, v0, v1\n\t"
                "vfcvt.f.x.v v0, v0\n\t"
                "vfmv.f.s %[ftmp], v0\n\t"
                "fmadd.s %[sumf], %[d], %[ftmp], %[sumf]"
                : [q6] "+&r" (q6), [q6h] "=&r" (q6h)
                , [scale] "+&r" (scale)
                , [sumf] "+&f" (sumf), [ftmp] "=&f" (ftmp)
                : [qh] "r" (qh), [q8] "r" (q8)
                , [vl32] "r" (32), [vl64] "r" (64), [vl128] "r" (128)
                , [mask] "r" (0x30), [d] "f" (d)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                , "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a7"
                , "a6", "a5", "a4", "a3"
            );
            qh += 32;   q8 += 128;
        }
    }

    *s = sumf;
}
