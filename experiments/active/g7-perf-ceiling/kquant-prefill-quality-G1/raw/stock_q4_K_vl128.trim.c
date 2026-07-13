#include <riscv_vector.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#define QK_K 256
#define K_SCALE_SIZE 12
typedef uint16_t ggml_half;
typedef struct { union { struct { ggml_half d; ggml_half dmin; }; ggml_half dm[2]; }; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } block_q4_K;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;
static inline float GGML_CPU_FP16_TO_FP32(ggml_half h){ return (float)h; }
#define GGML_RESTRICT __restrict
#define restrict __restrict
#define UNUSED(x) (void)(x)
#define NOINLINE
#define GGML_UNUSED(x) (void)(x)
static NOINLINE void ggml_vec_dot_q4_K_q8_K_vl128(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        float ftmp, ft2;
        const uint8_t * restrict q40;
        const uint8_t * restrict q41;
        const uint8_t * restrict q42;
        const uint8_t * restrict q43;
        const int8_t  * restrict q80;
        const int8_t  * restrict q81;
        const int8_t  * restrict q82;
        const int8_t  * restrict q83;
        int s0, s1, s2, s3;

        __asm__ __volatile__(
            "li %[s1], 8\n\t"
            "vsetivli zero, 4, e32, m1, ta, ma\n\t"
            "vle32.v v1, (%[s6b])\n\t"
            "vslide1down.vx v1, v1, zero\n\t"
            "vmv.v.x v16, zero\n\t"
            "vslidedown.vi v2, v1, 2\n\t"
            "vmv1r.v v3, v2\n\t"
            "vslideup.vi v2, v3, 1\n\t" // {aux[2], aux[2]}
            "vsetivli zero, 2, e32, m1, ta, ma\n\t"
            "vmv.v.i v4, 4\n\t"
            "vand.vx v8, v1, %[kmask1]\n\t"
            "vslide1up.vx v5, v4, zero\n\t" // {0, 4}
            "vsrl.vi v6, v1, 6\n\t"
            "vsrl.vv v7, v2, v5\n\t"
            "vsse32.v v8, (%[utmp]), %[s1]\n\t"
            "vand.vx v0, v6, %[kmask3]\n\t"
            "vand.vx v2, v7, %[kmask2]\n\t"
            "vsll.vi v6, v0, 4\n\t"
            "addi %[s0], %[utmp], 4\n\t"
            "vor.vv v1, v6, v2\n\t"
            "vsse32.v v1, (%[s0]), %[s1]\n\t"
            "vsetivli zero, 8, e16, m1, ta, ma\n\t"
            "vle32.v v2, (%[bsums])\n\t"
            "vnsrl.wi v0, v2, 0\n\t"
            "vnsrl.wi v1, v2, 16\n\t"
            "vadd.vv v2, v0, v1\n\t"
            "vle8.v v3, (%[mins])\n\t"
            "vzext.vf2 v4, v3\n\t"
            "vwmul.vv v6, v4, v2\n\t"
            "vsetivli zero, 4, e32, m1, ta, ma\n\t"
            "vredsum.vs v0, v6, v16\n\t"
            "vredsum.vs v0, v7, v0\n\t"
            "vfcvt.f.x.v v0, v0\n\t"
            "vfmv.f.s %[ftmp], v0\n\t"
            "vsetivli zero, 16, e8, m1, ta, ma\n\t"
            "vle8.v v0, (%[xs])\n\t"
            "fnmsub.s %[sumf], %[dmin], %[ftmp], %[sumf]\n\t"
            "addi %[q40], %[xs], 64\n\t"
            "addi %[q41], %[xs], 16\n\t"
            "addi %[q42], %[xs], 32\n\t"
            "addi %[q43], %[xs], 48\n\t"
            "addi %[q80], %[ys], 64\n\t"
            "vle8.v v1, (%[q41])\n\t"
            "vle8.v v2, (%[q42])\n\t"
            "addi %[q81], %[ys], 16\n\t"
            "addi %[q41], %[q41], 64\n\t"
            "addi %[q82], %[ys], 32\n\t"
            "vle8.v v3, (%[q43])\n\t"
            "vle8.v v8, (%[ys])\n\t"
            "addi %[q42], %[q42], 64\n\t"
            "addi %[q83], %[ys], 48\n\t"
            "addi %[q43], %[q43], 64\n\t"
            "vsrl.vi v4, v0, 4\n\t"
            "vle8.v v9, (%[q81])\n\t"
            "vle8.v v10, (%[q82])\n\t"
            "vand.vi v0, v0, 0xF\n\t"
            "addi %[q81], %[q81], 64\n\t"
            "vsrl.vi v5, v1, 4\n\t"
            "addi %[q82], %[q82], 64\n\t"
            "vle8.v v11, (%[q83])\n\t"
            "vle8.v v12, (%[q80])\n\t"
            "vand.vi v1, v1, 0xF\n\t"
            "addi %[q83], %[q83], 64\n\t"
            "vsrl.vi v6, v2, 4\n\t"
            "addi %[q80], %[q80], 64\n\t"
            "vle8.v v13, (%[q81])\n\t"
            "vle8.v v14, (%[q82])\n\t"
            "vand.vi v2, v2, 0xF\n\t"
            "addi %[q81], %[q81], 64\n\t"
            "vsrl.vi v7, v3, 4\n\t"
            "addi %[q82], %[q82], 64\n\t"
            "vwmul.vv v16, v0, v8\n\t"
            "vle8.v v15, (%[q83])\n\t"
            "vle8.v v0, (%[q40])\n\t"
            "vand.vi v3, v3, 0xF\n\t"
            "addi %[q83], %[q83], 64\n\t"
            "vwmul.vv v24, v2, v12\n\t"
            "vwmul.vv v20, v4, v10\n\t"
            "vwmul.vv v28, v6, v14\n\t"
            "vwmacc.vv v16, v1, v9\n\t"
            "vle8.v v1, (%[q41])\n\t"
            "vle8.v v2, (%[q42])\n\t"
            "vwmacc.vv v24, v3, v13\n\t"
            "vwmacc.vv v20, v5, v11\n\t"
            "vwmacc.vv v28, v7, v15\n\t"
            "addi %[q40], %[q80], 64\n\t"
            "addi %[q41], %[q81], 64\n\t"
            "vle8.v v3, (%[q43])\n\t"
            "vle8.v v8, (%[q80])\n\t"
            "addi %[q42], %[q82], 64\n\t"
            "addi %[q43], %[q83], 64\n\t"
            "vsrl.vi v4, v0, 4\n\t"
            "vle8.v v9, (%[q81])\n\t"
            "vle8.v v10, (%[q82])\n\t"
            "vand.vi v0, v0, 0xF\n\t"
            "vsrl.vi v5, v1, 4\n\t"
            "vsrl.vi v7, v3, 4\n\t"
            "vand.vi v3, v3, 0xF\n\t"
            "vle8.v v11, (%[q83])\n\t"
            "vle8.v v12, (%[q40])\n\t"
            "vand.vi v1, v1, 0xF\n\t"
            "vsrl.vi v6, v2, 4\n\t"
            "vand.vi v2, v2, 0xF\n\t"
            "vwmul.vv v18, v0, v8\n\t"
            "vle8.v v13, (%[q41])\n\t"
            "vle8.v v14, (%[q42])\n\t"
            "vwmul.vv v26, v2, v12\n\t"
            "vwmul.vv v22, v4, v10\n\t"
            "vwmul.vv v30, v6, v14\n\t"
            "vwmacc.vv v18, v1, v9\n\t"
            "vle8.v v15, (%[q43])\n\t"
            "vwmacc.vv v26, v3, v13\n\t"
            "vwmacc.vv v22, v5, v11\n\t"
            "vwmacc.vv v30, v7, v15\n\t"
            "vmv.v.x v0, zero\n\t"
            "vsetivli zero, 16, e16, m2, ta, ma\n\t"
            "vwredsum.vs v4, v16, v0\n\t"
            "lbu %[s0], 0(%[scale])\n\t"
            "vwredsum.vs v5, v20, v0\n\t"
            "lbu %[s1], 1(%[scale])\n\t"
            "vwredsum.vs v6, v24, v0\n\t"
            "lbu %[s2], 2(%[scale])\n\t"
            "vwredsum.vs v7, v28, v0\n\t"
            "lbu %[s3], 3(%[scale])\n\t"
            "vwredsum.vs v8, v18, v0\n\t"
            "lbu %[q40], 4(%[scale])\n\t"
            "vwredsum.vs v9, v22, v0\n\t"
            "lbu %[q41], 5(%[scale])\n\t"
            "vwredsum.vs v10, v26, v0\n\t"
            "lbu %[q42], 6(%[scale])\n\t"
            "vwredsum.vs v11, v30, v0\n\t"
            "lbu %[q43], 7(%[scale])\n\t"
            "vsetivli zero, 4, e32, m1, ta, ma\n\t"
            "vmul.vx v0, v4, %[s0]\n\t"
            "vmul.vx v1, v8, %[q40]\n\t"
            "vmacc.vx v0, %[s1], v5\n\t"
            "vmacc.vx v1, %[q41], v9\n\t"
            "vmacc.vx v0, %[s2], v6\n\t"
            "vmacc.vx v1, %[q42], v10\n\t"
            "vmacc.vx v0, %[s3], v7\n\t"
            "vmacc.vx v1, %[q43], v11\n\t"
            "vfcvt.f.x.v v0, v0\n\t"
            "vfcvt.f.x.v v1, v1\n\t"
            "vfmv.f.s %[ft2], v0\n\t"
            "vfmv.f.s %[ftmp], v1\n\t"
            "fadd.s %[ft2], %[ft2], %[ftmp]\n\t"
            "fmadd.s %[sumf], %[d], %[ft2], %[sumf]"
            : [ftmp] "=&f" (ftmp), [sumf] "+&f" (sumf), [ft2] "=&f" (ft2)
            , [s0] "=&r" (s0), [s1] "=&r" (s1), [s2] "=&r" (s2), [s3] "=&r" (s3)
            , [q40] "=&r" (q40), [q41] "=&r" (q41), [q42] "=&r" (q42), [q43] "=&r" (q43)
            , [q80] "=&r" (q80), [q81] "=&r" (q81), [q82] "=&r" (q82), [q83] "=&r" (q83)
            : [d] "f" (d), [ys] "r" (y[i].qs), [xs] "r" (x[i].qs), [scale] "r" (scales)
            , [bsums] "r" (y[i].bsums), [mins] "r" (mins), [utmp] "r" (utmp)
            , [s6b] "r" (&x[i]), [kmask1] "r" (kmask1), [dmin] "f" (dmin)
            , [kmask2] "r" (kmask2), [kmask3] "r" (kmask3)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );
    }

    *s = sumf;
}

