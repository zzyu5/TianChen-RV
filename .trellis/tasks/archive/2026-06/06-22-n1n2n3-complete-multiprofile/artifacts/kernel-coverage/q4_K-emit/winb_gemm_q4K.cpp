// q4_K repack GEMM — Win-B.micro: OUR compiler-emitted q4_K repack PREFILL GEMM
// (block-as-lane, M=4 rows, weight decode amortized once across the 4 rows) vs ggml's
// OWN VLEN128 prefill fallback. At VLEN128 ggml routes q4_K to nullptr (repack.cpp:4619
// case 128: break // TODO), so its prefill falls back to the plain per-row block-dot
// ggml_vec_dot_q4_K_q8_K, which on this part is the hand-written RVV inline-asm
// ggml_vec_dot_q4_K_q8_K_vl128 (quants.c:1770) — called PER-ROW over the M=4 activation
// rows × nc output columns. That is the methodology-correct Win-B baseline per
// N3-METHODOLOGY (ggml's REAL RVV kernel, NOT scalar/naive).
//
// THE HYPOTHESIS (the GEMM win the GEVM could not get): ours decodes each weight group's
// 6-bit unpack + nibbles ONCE and reuses it across the 4 activation rows, and streams the
// weights ONCE; ggml has no GEMM, so it re-streams + re-decodes the weights M=4× (one
// _vl128 call per (row, col)). The 4× weight-traffic + decode amortization is the source
// of any win — does it overcome the same mf2/half_lanes=8 VLEN128 shape-disadvantage that
// sank the q4_K GEVM Win-B (0.47-0.66×)? That is the decisive, genuinely uncertain test.
//
// Data construction lifted VERBATIM from the PASS-validated GEMM oracle oracle_gemm_q4K.cpp
// (distinct weights, 4 distinct rows, both layouts: vx=block_q4_Kx16 + vy=block_q8_Kx4 for
// ours; orig block_q4_K + per-row block_q8_K for ggml). Only the ref is swapped from the
// scalar oracle to ggml's real _vl128, plus timing. ratio = ggml/ours (>1 => ours WIN).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <random>
#include <riscv_vector.h>

#define QK_K 256
#define K_SCALE_SIZE 12

struct block_q4_K {
    _Float16 d;
    _Float16 dmin;
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qs[QK_K/2];
};
static_assert(sizeof(block_q4_K) == 4 + 12 + 128, "q4_K size");

struct block_q4_Kx16 {
    _Float16 d[16];
    _Float16 dmin[16];
    uint8_t  scales[192];
    uint8_t  qs[2048];
};
static_assert(sizeof(block_q4_Kx16) == 32 + 32 + 192 + 2048, "q4_Kx16 size 2304");

struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
};
static_assert(sizeof(block_q8_K) == 4 + 256 + 32, "q8_K size 292");

struct block_q8_Kx4 {
    float   d[4];
    int8_t  qs[QK_K*4];
    int16_t bsums[QK_K/4];
};
static_assert(sizeof(block_q8_Kx4) == 16 + 1024 + 128, "q8_Kx4 size 1168");

#define GGML_CPU_FP16_TO_FP32(x) ((float)(_Float16)(x))
#define GGML_RESTRICT __restrict__
#define UNUSED(x) (void)(x)
#define NOINLINE __attribute__((noinline))

// ================== ggml's OWN shipped RVV kernel for VLEN128 ==================
// Lifted VERBATIM from llama.cpp ggml/src/ggml-cpu/arch/riscv/quants.c:1770
// (ggml_vec_dot_q4_K_q8_K_vl128). The Win-B baseline.
static NOINLINE void ggml_vec_dot_q4_K_q8_K_vl128(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = (const block_q4_K *)vx;
    const block_q8_K * GGML_RESTRICT y = (const block_q8_K *)vy;

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
        const uint8_t * __restrict__ q40;
        const uint8_t * __restrict__ q41;
        const uint8_t * __restrict__ q42;
        const uint8_t * __restrict__ q43;
        const int8_t  * __restrict__ q80;
        const int8_t  * __restrict__ q81;
        const int8_t  * __restrict__ q82;
        const int8_t  * __restrict__ q83;
        int s0, s1, s2, s3;

        __asm__ __volatile__(
            "li %[s1], 8\n\t"
            "vsetivli zero, 4, e32, m1, ta, ma\n\t"
            "vle32.v v1, (%[s6b])\n\t"
            "vslide1down.vx v1, v1, zero\n\t"
            "vmv.v.x v16, zero\n\t"
            "vslidedown.vi v2, v1, 2\n\t"
            "vmv1r.v v3, v2\n\t"
            "vslideup.vi v2, v3, 1\n\t"
            "vsetivli zero, 2, e32, m1, ta, ma\n\t"
            "vmv.v.i v4, 4\n\t"
            "vand.vx v8, v1, %[kmask1]\n\t"
            "vslide1up.vx v5, v4, zero\n\t"
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

// ggml GEMM baseline: the FULL M*nc output by calling the real _vl128 per-row block-dot
// once per (row m, output column gcol). ggml has no q4_K GEMM at VLEN128, so it re-streams
// + re-decodes the weights M times. orig[gcol*nb] = column gcol's nb contiguous q4_K blocks;
// actrow[m] = row m's nb per-row block_q8_K.
static void ggml_gemm_q4K(int n, int M, int nc, int nb, float * out,
                          const block_q4_K * orig, const std::vector<std::vector<block_q8_K>> & actrow) {
    for (int m = 0; m < M; ++m)
        for (int gcol = 0; gcol < nc; ++gcol)
            ggml_vec_dot_q4_K_q8_K_vl128(n, &out[(size_t)m*nc + gcol], 0,
                (const void *)&orig[(size_t)gcol * nb], 0,
                (const void *)actrow[m].data(), 0, 1);
}

// ================== oracle data construction (verbatim from oracle_gemm_q4K.cpp) ==================
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); }
}

static block_q4_Kx16 make_block_q4_Kx16(const block_q4_K * in) {
    block_q4_Kx16 out;
    for (int i = 0; i < 16; i++) out.d[i]    = in[i].d;
    for (int i = 0; i < 16; i++) out.dmin[i] = in[i].dmin;
    const int blck_size_interleave = 1;
    const int end = QK_K * 8 / blck_size_interleave;
    for (int i = 0; i < end; ++i) { int src_id = i % 16; int src_offset = i / 16; out.qs[i] = in[src_id].qs[src_offset]; }
    uint8_t s[128], m[128];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 16; j++) {
        s[i * 16 + j] = in[j].scales[i] & 63; m[i * 16 + j] = in[j].scales[i + 4] & 63;
    }
    for (int i = 0; i < 4; i++) for (int j = 0; j < 16; j++) {
        s[64 + i * 16 + j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
        m[64 + i * 16 + j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
    }
    for (int i = 0; i < 128; i++) out.scales[i] = (s[i] & 15) | ((m[i] & 15) << 4);
    for (int i = 0; i < 64; i++) out.scales[128 + i] = ((s[i] & 48) >> 4) | ((m[i] & 48) >> 2) | (s[64 + i] & 48) | ((m[64 + i] & 48) << 2);
    return out;
}

static block_q8_Kx4 make_block_q8_Kx4(const block_q8_K rows[4]) {
    block_q8_Kx4 out;
    for (int m = 0; m < 4; m++) out.d[m] = rows[m].d;
    for (int j = 0; j < QK_K/4; j++) out.bsums[j] = 0;
    for (int j = 0; j < QK_K * 4; j++) {
        int src_id = j % 4; int src_offset = j / 4; int index = ((j >> 6) << 2) + (j & 3);
        out.qs[j] = rows[src_id].qs[src_offset];
        out.bsums[index] += out.qs[j];
    }
    return out;
}

extern "C" void tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs);

static void build_q4K_block(block_q4_K * b, std::mt19937 & rng, int col, int blk) {
    std::uniform_int_distribution<int> nib(0, 15);
    b->d    = (_Float16)(0.012f + 0.0007f * ((col + 3*blk) % 11));
    b->dmin = (_Float16)(0.008f + 0.0005f * ((2*col + blk) % 7));
    uint8_t ls[8], lm[8];
    for (int sb = 0; sb < 8; sb++) {
        ls[sb] = (uint8_t)( (7 + 6*sb + 2*col + blk) & 63 );
        lm[sb] = (uint8_t)( (5 + 5*sb + col + 3*blk) & 63 );
        if (ls[sb] == 0) ls[sb] = 1;
    }
    memset(b->scales, 0, K_SCALE_SIZE);
    for (int sb = 0; sb < 8; sb++) {
        uint8_t l_s = ls[sb], l_m = lm[sb];
        if (sb < 4) { b->scales[sb] = l_s; b->scales[sb+4] = l_m; }
        else { b->scales[sb+4] = (l_s & 0xF) | ((l_m & 0xF) << 4); b->scales[sb-4] |= ((l_s >> 4) << 6); b->scales[sb-0] |= ((l_m >> 4) << 6); }
    }
    for (int i = 0; i < QK_K/2; i++) b->qs[i] = (uint8_t)(nib(rng) | (nib(rng) << 4));
}

static void build_q8K_row(block_q8_K * a, std::mt19937 & rng, int row, int blk) {
    std::uniform_int_distribution<int> q8d(-90, 90);
    a->d = 0.015f + 0.0009f * ((blk + 5*row) % 13) + 0.0004f * row;
    for (int grp = 0; grp < QK_K/16; grp++) {
        int dc = ((grp * 7 + blk * 3 + row * 11) % 21) - 10;
        int sum = 0;
        for (int ii = 0; ii < 16; ii++) {
            int v = q8d(rng) + dc;
            if (v > 127) v = 127; if (v < -128) v = -128;
            a->qs[grp*16 + ii] = (int8_t)v; sum += v;
        }
        a->bsums[grp] = (int16_t)sum;
    }
}

static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }

int main(int /*argc*/, char** /*argv*/) {
    struct Shape { int nc; int n; };
    std::vector<Shape> shapes = {
        {16, 4096}, {16, 11008},
        {256, 4096}, {256, 11008},
    };
    const int M = 4;

    std::mt19937 rng(20260624u);
    printf("# q4_K repack PREFILL GEMM Win-B.micro: OUR repack GEMM (M=4, one call) vs ggml's real _vl128\n");
    printf("# baseline = ggml_vec_dot_q4_K_q8_K_vl128 called per-row x nc (M*nc calls; ggml has no VLEN128 q4_K GEMM)\n");
    printf("# %-12s %-12s %-12s %-12s %-10s %s\n", "shape(nc x n)", "ours_ns", "ggml_ns", "ratio", "agree_norm", "verdict");

    for (auto sh : shapes) {
        const int nc = sh.nc;
        const int n  = sh.n;
        const int nb = n / QK_K;
        const int ng = nc / 16;

        // ---- weights (column-contiguous originals): orig[gcol*nb + l] ----
        std::vector<block_q4_K> orig((size_t)nc * nb);
        for (int g = 0; g < ng; g++)
            for (int col = 0; col < 16; col++)
                for (int l = 0; l < nb; l++) {
                    int gcol = g*16 + col;
                    build_q4K_block(&orig[(size_t)gcol*nb + l], rng, gcol, l);
                }

        // ---- repack weights for our kernel: group-major, block-major (lane=col) ----
        std::vector<block_q4_Kx16> vx((size_t)ng * nb);
        std::vector<block_q4_K> tmp16(16);
        for (int g = 0; g < ng; g++)
            for (int l = 0; l < nb; l++) {
                for (int col = 0; col < 16; col++) tmp16[col] = orig[(size_t)(g*16+col)*nb + l];
                vx[(size_t)g*nb + l] = make_block_q4_Kx16(tmp16.data());
            }

        // ---- activation: 4 DISTINCT per-row block_q8_K, then interleave into block_q8_Kx4 ----
        std::vector<std::vector<block_q8_K>> actrow(M, std::vector<block_q8_K>(nb));
        for (int m = 0; m < M; m++)
            for (int l = 0; l < nb; l++)
                build_q8K_row(&actrow[m][l], rng, m, l);
        std::vector<block_q8_Kx4> vy(nb);
        for (int l = 0; l < nb; l++) {
            block_q8_K rows[4] = { actrow[0][l], actrow[1][l], actrow[2][l], actrow[3][l] };
            vy[l] = make_block_q8_Kx4(rows);
        }

        // ---- ours: ONE GEMM call produces the full M*nc output (bs=nc, nr=4) ----
        std::vector<float> out_ours((size_t)M*nc, 0.0f);
        tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
            (size_t)n, out_ours.data(),
            reinterpret_cast<const uint8_t*>(vx.data()),
            reinterpret_cast<const uint8_t*>(vy.data()),
            (size_t)M, (size_t)nc, (size_t)nc);

        // ---- ggml: M*nc calls of the real _vl128 over ORIGINAL weights + per-row q8_K ----
        std::vector<float> out_ggml((size_t)M*nc, 0.0f);
        ggml_gemm_q4K(n, M, nc, nb, out_ggml.data(), orig.data(), actrow);

        // ---- agreement norm over the FULL M*nc output: max|ours-ggml|/rms(ggml) ----
        double max_abs = 0.0, sumsq = 0.0;
        for (int i = 0; i < M*nc; i++) {
            double e = fabs((double)out_ours[i] - (double)out_ggml[i]);
            if (e > max_abs) max_abs = e;
            sumsq += (double)out_ggml[i]*(double)out_ggml[i];
        }
        double rms = sqrt(sumsq / (M*nc));
        double agree = (rms > 0) ? max_abs / rms : max_abs;

        // ---- timing: both produce the FULL M*nc output; best-of-reps min ----
        int iters = (n <= 4096) ? 200 : 80;
        if (nc >= 256) iters = (n <= 4096) ? 24 : 10;
        const int reps = 60;
        for (int w = 0; w < iters; w++) {
            tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
                (size_t)n, out_ours.data(),
                reinterpret_cast<const uint8_t*>(vx.data()),
                reinterpret_cast<const uint8_t*>(vy.data()),
                (size_t)M, (size_t)nc, (size_t)nc);
            ggml_gemm_q4K(n, M, nc, nb, out_ggml.data(), orig.data(), actrow);
        }
        double ours_best = 1e18, ggml_best = 1e18;
        for (int r = 0; r < reps; r++) {
            double t0 = now();
            for (int it = 0; it < iters; it++)
                tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
                    (size_t)n, out_ours.data(),
                    reinterpret_cast<const uint8_t*>(vx.data()),
                    reinterpret_cast<const uint8_t*>(vy.data()),
                    (size_t)M, (size_t)nc, (size_t)nc);
            double po = (now() - t0) / iters;
            if (po < ours_best) ours_best = po;

            t0 = now();
            for (int it = 0; it < iters; it++)
                ggml_gemm_q4K(n, M, nc, nb, out_ggml.data(), orig.data(), actrow);
            double pg = (now() - t0) / iters;
            if (pg < ggml_best) ggml_best = pg;
        }

        double ratio = ggml_best / ours_best;
        const char * verdict = (ratio > 1.0) ? "WIN" : "LOSS";
        char shapestr[32]; snprintf(shapestr, sizeof shapestr, "%dx%d", nc, n);
        printf("  %-12s %-12.1f %-12.1f %-12.3f %-10.3e %s%s\n",
               shapestr, ours_best, ggml_best, ratio, agree, verdict,
               ratio > 1.0 ? " (ours faster)" : " (ggml faster)");
    }
    return 0;
}
