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
void ggml_vec_dot_q3_K_q8_K_vl128(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    uint32_t utmp[4];
    float sumf = 0;
    uint32_t aux[3];

    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict qh = x[i].hmask;
        const  int8_t * restrict q8 = y[i].qs;

        int8_t * scale = (int8_t *)utmp;
        int tmp, t1, t2, t3, t4, t5, t6, t7;
        __asm__ __volatile__(
            "vsetivli zero, 12, e8, m1\n\t"
            "vle8.v v0, (%[s6b])\n\t"
            "vmv1r.v v2, v0\n\t"
            "vsetivli zero, 2, e64, m1\n\t"
            "vmv.v.x v9, %[sh]\n\t"\
            "vslidedown.vi v1, v0, 1\n\t"
            "vslide1up.vx v8, v9, zero\n\t" // {0, 0, 4, 4}
            "vslideup.vi v0, v2, 1\n\t" // {aux[0], aux[1], aux[0], aux[1]}
            "vsetivli zero, 4, e32, m1\n\t"
            "vid.v v9\n\t"
            "vmv.x.s %[tmp], v1\n\t"
            "vsll.vi v9, v9, 1\n\t" // {0, 2, 4, 6}
            "vmv.v.x v1, %[tmp]\n\t" // {aux[2], aux[2], aux[2], aux[2]}
            "vsrl.vv v4, v1, v9\n\t"
            "vsrl.vv v2, v0, v8\n\t"
            "vand.vx v5, v4, %[kmask1]\n\t"
            "vand.vx v3, v2, %[kmask2]\n\t"
            "vsll.vi v6, v5, 4\n\t"
            "vor.vv v7, v6, v3\n\t"
            "vsetivli zero, 16, e8, m1\n\t"
            "vsub.vx v0, v7, %[c]\n\t"
            "vse8.v v0, (%[scale])"
            : [tmp] "=&r" (tmp)
            : [sh] "r" (0x0000000400000004), [s6b] "r" (x[i].scales), [c] "r" (32)
            , [scale] "r" (scale), [kmask1] "r" (kmask1), [kmask2] "r" (kmask2)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );

        uint8_t m = 1;
        int isum = 0;
        for (int j = 0; j < QK_K; j += 128) {
            __asm__ __volatile__(
                "lb zero, 31(%[q3])\n\t"
                "vsetvli zero, %[vl32], e8, m2, ta, mu\n\t"
                "vle8.v v8, (%[q3])\n\t"
                "vsrl.vi v10, v8, 2\n\t"
                "vsrl.vi v12, v8, 4\n\t"
                "vsrl.vi v14, v8, 6\n\t"
                "lb zero, 64(%[q8])\n\t"
                "vand.vi v8, v8, 3\n\t"
                "vand.vi v10, v10, 3\n\t"
                "vand.vi v12, v12, 3\n\t"
                "vle8.v v2, (%[qh])\n\t"
                "lb zero, 127(%[q8])\n\t"
                "vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "vmseq.vx v0, v4, zero\n\t"
                "vadd.vi v8, v8, -4, v0.t\n\t"
                "lb zero, 0(%[q8])\n\t"
                "vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "vmseq.vx v0, v4, zero\n\t"
                "vadd.vi v10, v10, -4, v0.t\n\t"
                "vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "vmseq.vx v0, v4, zero\n\t"
                "vadd.vi v12, v12, -4, v0.t\n\t"
                "vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "vmseq.vx v0, v4, zero\n\t"
                "vadd.vi v14, v14, -4, v0.t\n\t"
                "vsetvli zero, %[vl128], e8, m8\n\t"
                "vle8.v v0, (%[q8])\n\t"
                "lb %[tmp], 0(%[scale])\n\t"
                "lb %[t1], 1(%[scale])\n\t"
                "lb %[t2], 2(%[scale])\n\t"
                "lb %[t3], 3(%[scale])\n\t"
                "vsetvli zero, %[vl64], e8, m4\n\t"
                "vwmul.vv v16, v0, v8\n\t"
                "vwmul.vv v24, v4, v12\n\t"
                "vsetivli zero, 16, e16, m2\n\t"
                "vmv.v.x v0, zero\n\t"
                "vwredsum.vs v8, v16, v0\n\t"
                "lb %[t4], 4(%[scale])\n\t"
                "lb %[t5], 5(%[scale])\n\t"
                "vwredsum.vs v9, v18, v0\n\t"
                "vwredsum.vs v10, v20, v0\n\t"
                "vwredsum.vs v11, v22, v0\n\t"
                "vwredsum.vs v12, v24, v0\n\t"
                "lb %[t6], 6(%[scale])\n\t"
                "lb %[t7], 7(%[scale])\n\t"
                "vwredsum.vs v13, v26, v0\n\t"
                "vwredsum.vs v14, v28, v0\n\t"
                "vwredsum.vs v15, v30, v0\n\t"
                "vsetivli zero, 4, e32, m1\n\t"
                "vmul.vx v0, v8, %[tmp]\n\t"
                "vmul.vx v1, v9, %[t1]\n\t"
                "vmacc.vx v0, %[t2], v10\n\t"
                "vmacc.vx v1, %[t3], v11\n\t"
                "vmacc.vx v0, %[t4], v12\n\t"
                "vmacc.vx v1, %[t5], v13\n\t"
                "vmacc.vx v0, %[t6], v14\n\t"
                "vmacc.vx v1, %[t7], v15\n\t"
                "vmv.x.s %[tmp], v0\n\t"
                "vmv.x.s %[t1], v1\n\t"
                "add %[isum], %[isum], %[tmp]\n\t"
                "add %[isum], %[isum], %[t1]"
                : [tmp] "=&r" (tmp), [t1] "=&r" (t1), [t2] "=&r" (t2), [t3] "=&r" (t3)
                , [t4] "=&r" (t4), [t5] "=&r" (t5), [t6] "=&r" (t6), [t7] "=&r" (t7)
                , [m] "+&r" (m), [isum] "+&r" (isum)
                : [vl128] "r" (128), [vl64] "r" (64), [vl32] "r" (32)
                , [q3] "r" (q3), [qh] "r" (qh), [scale] "r" (scale), [q8] "r" (q8)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            q3 += 32;    q8 += 128;   scale += 8;
        }

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        sumf += d * isum;
    }

    *s = sumf;
}
