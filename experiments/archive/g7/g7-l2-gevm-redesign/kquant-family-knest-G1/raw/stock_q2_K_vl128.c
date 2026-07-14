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
void ggml_vec_dot_q2_K_q8_K_vl128(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    uint8_t atmp[16];

    uint8_t temp_01[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    for (int i = 0; i < nb; ++i) {
        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;
        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);
        uint8_t *patmp = atmp;
        int vsums;
        int tmp, t1, t2, t3, t4, t5, t6, t7;
        __asm__ __volatile__(
            "vsetivli zero, 16, e8, m1\n\t"
            "vmv.v.x v8, zero\n\t"
            "lb zero, 15(%[sc])\n\t"
            "vle8.v v1, (%[sc])\n\t"
            "vle8.v v2, (%[bsums])\n\t"
            "addi %[tmp], %[bsums], 16\n\t"
            "vand.vi v0, v1, 0xF\n\t"
            "vsrl.vi v1, v1, 4\n\t"
            "vle8.v v3, (%[tmp])\n\t"
            "vse8.v v0, (%[scale])\n\t"
            "vsetivli zero, 16, e16, m2\n\t"
            "vzext.vf2 v0, v1\n\t"
            "vwmul.vv v4, v0, v2\n\t"
            "vsetivli zero, 16, e32, m4\n\t"
            "vredsum.vs v8, v4, v8\n\t"
            "vmv.x.s %[vsums], v8"
            : [tmp] "=&r" (tmp), [vsums] "=&r" (vsums)
            : [sc] "r" (sc), [scale] "r" (atmp), [bsums] "r" (y[i].bsums)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );
        sumf += dmin * vsums;
        int isum = 0;

        for (int j = 0; j < QK_K/128; ++j) {
            __asm__ __volatile__(
                "lb zero, 31(%[q2])\n\t"
                "addi %[tmp], %[q2], 16\n\t"
                "addi %[t1], %[q8], 16\n\t"
                "vsetivli zero, 16, e8, m1\n\t"
                "vle8.v v0, (%[q2])\n\t"
                "vle8.v v1, (%[tmp])\n\t"
                "vsrl.vi v2, v0, 2\n\t"
                "vsrl.vi v3, v1, 2\n\t"
                "vsrl.vi v4, v0, 4\n\t"
                "addi %[tmp], %[q8], 32\n\t"
                "vle8.v v8, (%[q8])\n\t"
                "vle8.v v9, (%[t1])\n\t"
                "addi %[t1], %[t1], 32\n\t"
                "vsrl.vi v5, v1, 4\n\t"
                "vsrl.vi v6, v0, 6\n\t"
                "vsrl.vi v7, v1, 6\n\t"
                "vle8.v v10, (%[tmp])\n\t"
                "vle8.v v11, (%[t1])\n\t"
                "addi %[tmp], %[tmp], 32\n\t"
                "addi %[t1], %[t1], 32\n\t"
                "vand.vi v0, v0, 0x3\n\t"
                "vand.vi v1, v1, 0x3\n\t"
                "vand.vi v2, v2, 0x3\n\t"
                "vle8.v v12, (%[tmp])\n\t"
                "vle8.v v13, (%[t1])\n\t"
                "addi %[tmp], %[tmp], 32\n\t"
                "addi %[t1], %[t1], 32\n\t"
                "vand.vi v3, v3, 0x3\n\t"
                "vand.vi v4, v4, 0x3\n\t"
                "vand.vi v5, v5, 0x3\n\t"
                "vle8.v v14, (%[tmp])\n\t"
                "vle8.v v15, (%[t1])\n\t"
                "vwmul.vv v16, v0, v8\n\t"
                "vwmul.vv v18, v1, v9\n\t"
                "vwmul.vv v20, v2, v10\n\t"
                "vwmul.vv v22, v3, v11\n\t"
                "vwmul.vv v24, v4, v12\n\t"
                "vwmul.vv v26, v5, v13\n\t"
                "vwmul.vv v28, v6, v14\n\t"
                "vwmul.vv v30, v7, v15\n\t"
                "vsetivli zero, 8, e16, m1\n\t"
                "vmv.v.x v0, zero\n\t"
                "lbu %[tmp], 0(%[scale])\n\t"
                "vwredsum.vs v8, v16, v0\n\t"
                "vwredsum.vs v9, v18, v0\n\t"
                "lbu %[t1], 1(%[scale])\n\t"
                "vwredsum.vs v10, v20, v0\n\t"
                "vwredsum.vs v11, v22, v0\n\t"
                "lbu %[t2], 2(%[scale])\n\t"
                "vwredsum.vs v12, v24, v0\n\t"
                "vwredsum.vs v13, v26, v0\n\t"
                "lbu %[t3], 3(%[scale])\n\t"
                "vwredsum.vs v14, v28, v0\n\t"
                "vwredsum.vs v15, v30, v0\n\t"
                "lbu %[t4], 4(%[scale])\n\t"
                "vwredsum.vs v8, v17, v8\n\t"
                "vwredsum.vs v9, v19, v9\n\t"
                "lbu %[t5], 5(%[scale])\n\t"
                "vwredsum.vs v10, v21, v10\n\t"
                "vwredsum.vs v11, v23, v11\n\t"
                "lbu %[t6], 6(%[scale])\n\t"
                "vwredsum.vs v12, v25, v12\n\t"
                "vwredsum.vs v13, v27, v13\n\t"
                "lbu %[t7], 7(%[scale])\n\t"
                "vwredsum.vs v14, v29, v14\n\t"
                "vwredsum.vs v15, v31, v15\n\t"
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
                , [isum] "+&r" (isum)
                : [q2] "r" (q2), [scale] "r" (patmp), [q8] "r" (q8)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            q2 += 32; q8 += 128; patmp += 8;
        }

        sumf += dall * isum;
    }

    *s = sumf;
}
