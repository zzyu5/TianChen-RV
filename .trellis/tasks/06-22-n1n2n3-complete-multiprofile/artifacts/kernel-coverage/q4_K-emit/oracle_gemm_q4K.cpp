// q4_K x q8_K 16x1-REPACKED PREFILL GEMM numeric oracle (mf2 / VLEN128).
//
// Gate: does our compiler-emitted kernel
//   tcrv_emitc_..._ggml_repack_gemm_q4_K_q8_K(n, s, vx, vy, nr, nc, bs)
// compute the CORRECT q4_K dequant-MATMUL (M=4 activation rows x nc weight cols)
// result on the SG2044 (RVV1.0, VLEN128)?
//
// Reference = INDEPENDENT scalar dequant-matmul computed from the ORIGINAL (pre-repack)
// per-block q4_K weights, dotted PER-ROW with the per-row q8_K int8 quants. Strictly
// more robust than re-deriving the repack inversion in the oracle. The ref reads the
// 4 per-row block_q8_K DIRECTLY and never de-interleaves the block_q8_Kx4; the kernel
// reads the INTERLEAVED block_q8_Kx4 -> a kernel de-interleave bug cannot agree.
//
// The decisive NEW axis vs the GEVM oracle = the 4-ROW INTERLEAVE (block_q8_Kx4).
// We build 4 DISTINCT per-row block_q8_K (distinct d, distinct DC/qs per row) and
// interleave into block_q8_Kx4 via ggml's VERBATIM index math (ggml_quantize_mat_q8_K_4x1,
// repack.cpp:90): for j in 0..1023: m=j%4, e=j/4; qs_x4[j]=row[m].qs[e];
// index=((j>>6)<<2)+(j&3); bsums_x4[index]+=qs_x4[j]; d_x4[m]=row[m].d.
//
// Negative controls (perturb the REFERENCE; a SPIKE in norm-vs-kernel proves the
// corresponding structure is exercised):
//   NOMIN  - zero the min term            (min-term / bsums path)
//   PERM   - rotate per-sub-block scale+1  (sub-block-scale teeth)
//   ROWROT - dot activation row (m+1)%4    (the 4-row interleave mapping  <-- GEMM-new)
//
// out_mine is zero-init per trial: an unwritten output element reads 0 vs ref ~O(10)
// and spikes the norm, so a partial-write bug cannot hide.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// ---- ggml constants ----
#define QK_K 256
#define K_SCALE_SIZE 12

// ---- original per-block q4_K (ggml-common.h), 144 bytes ----
struct block_q4_K {
    _Float16 d;                 // +0
    _Float16 dmin;              // +2
    uint8_t  scales[K_SCALE_SIZE]; // +4
    uint8_t  qs[QK_K/2];        // +16
};
static_assert(sizeof(block_q4_K) == 4 + 12 + 128, "q4_K size");

// ---- repacked block_q4_Kx16 (repack.h), 2304 bytes ----
struct block_q4_Kx16 {
    _Float16 d[16];      // +0
    _Float16 dmin[16];   // +32
    uint8_t  scales[192];// +64
    uint8_t  qs[2048];   // +256
};
static_assert(sizeof(block_q4_Kx16) == 32 + 32 + 192 + 2048, "q4_Kx16 size 2304");

// ---- plain per-row block_q8_K (ggml-common.h), 292 bytes ----
struct block_q8_K {
    float   d;            // +0
    int8_t  qs[QK_K];     // +4
    int16_t bsums[QK_K/16];// +260 (16 * int16)
};
static_assert(sizeof(block_q8_K) == 4 + 256 + 32, "q8_K size 292");

// ---- interleaved block_q8_Kx4 (repack.h), 1168 bytes (the activation the kernel reads) ----
struct block_q8_Kx4 {
    float   d[4];           // +0   (16 B)
    int8_t  qs[QK_K*4];     // +16  (1024 B)
    int16_t bsums[QK_K/4];  // +1040 (64 * int16 = 128 B)
};
static_assert(sizeof(block_q8_Kx4) == 16 + 1024 + 128, "q8_Kx4 size 1168");

// ---- ggml get_scale_min_k4 (ggml-quants.c:822), verbatim ----
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// dequant modes: 0=NORMAL (true ref), 1=NOMIN (zero min term), 2=PERM (rotate per-sub-block
// scale by +1). NOMIN/PERM are NEGATIVE CONTROLS on the WEIGHT/min side (unchanged from GEVM).
enum DqMode { DQ_NORMAL = 0, DQ_NOMIN = 1, DQ_PERM = 2 };

// ---- ggml dequantize_row_q4_K (ggml-quants.c:1471), verbatim semantics, one block ----
static void dequantize_block_q4_K_mode(const block_q4_K * x, float * y, DqMode mode) {
    const uint8_t * q = x->qs;
    const float d   = (float) x->d;
    const float min = (float) x->dmin;

    uint8_t scv[8], mnv[8];
    for (int sb = 0; sb < 8; sb++) get_scale_min_k4(sb, x->scales, &scv[sb], &mnv[sb]);

    int sb = 0;
    for (int jj = 0; jj < QK_K; jj += 64) {
        int s0 = sb, s1 = sb + 1;
        if (mode == DQ_PERM) { s0 = (sb + 1) % 8; s1 = (sb + 2) % 8; }
        const float d1 = d * scv[s0]; float m1 = min * mnv[s0];
        const float d2 = d * scv[s1]; float m2 = min * mnv[s1];
        if (mode == DQ_NOMIN) { m1 = 0.0f; m2 = 0.0f; }
        for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
        q += 32; sb += 2;
    }
}

// ---- make_block_q4_Kx16 (repack.cpp:2913), verbatim, blck_size_interleave=1 ----
static block_q4_Kx16 make_block_q4_Kx16(const block_q4_K * in) {
    block_q4_Kx16 out;
    for (int i = 0; i < 16; i++) out.d[i]    = in[i].d;
    for (int i = 0; i < 16; i++) out.dmin[i] = in[i].dmin;

    const int blck_size_interleave = 1;
    const int end = QK_K * 8 / blck_size_interleave; // 2048

    for (int i = 0; i < end; ++i) {
        int src_id = i % 16;
        int src_offset = i / 16;
        int dst_offset = i;
        out.qs[dst_offset] = in[src_id].qs[src_offset];
    }

    uint8_t s[128], m[128];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 16; j++) {
            s[i * 16 + j] = in[j].scales[i] & 63;
            m[i * 16 + j] = in[j].scales[i + 4] & 63;
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 16; j++) {
            s[64 + i * 16 + j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
            m[64 + i * 16 + j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
        }
    }
    for (int i = 0; i < 128; i++) {
        out.scales[i] = (s[i] & 15) | ((m[i] & 15) << 4);
    }
    for (int i = 0; i < 64; i++) {
        out.scales[128 + i] = ((s[i] & 48) >> 4) | ((m[i] & 48) >> 2) | (s[64 + i] & 48) | ((m[64 + i] & 48) << 2);
    }
    return out;
}

// ---- interleave 4 per-row block_q8_K into block_q8_Kx4, VERBATIM ggml index math
//      (ggml_quantize_mat_q8_K_4x1, repack.cpp:90; here we already HAVE quantized
//      per-row qs, so we just gather them with ggml's identical qs/bsums index math). ----
static block_q8_Kx4 make_block_q8_Kx4(const block_q8_K rows[4]) {
    block_q8_Kx4 out;
    for (int m = 0; m < 4; m++) out.d[m] = rows[m].d;
    for (int j = 0; j < QK_K/4; j++) out.bsums[j] = 0;
    for (int j = 0; j < QK_K * 4; j++) {
        int src_id = j % 4;       // row m
        int src_offset = j / 4;   // q8_K element e (0..255)
        int index = ((j >> 6) << 2) + (j & 3); // bsums slot = g16*4 + m
        out.qs[j] = rows[src_id].qs[src_offset];
        out.bsums[index] += out.qs[j];
    }
    return out;
}

// ---- the compiler-emitted GEMM kernel under test ----
// signature: (size_t n, float* s, const uint8_t* vx, const uint8_t* vy,
//             size_t nr, size_t nc, size_t bs)
extern "C" void tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy,
    size_t nr, size_t nc, size_t bs);

// ---- build ONE adversarial original q4_K weight block (verbatim from GEVM oracle) ----
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
        if (sb < 4) {
            b->scales[sb]   = l_s;
            b->scales[sb+4] = l_m;
        } else {
            b->scales[sb+4] = (l_s & 0xF) | ((l_m & 0xF) << 4);
            b->scales[sb-4] |= ((l_s >> 4) << 6);
            b->scales[sb-0] |= ((l_m >> 4) << 6);
        }
    }

    for (int i = 0; i < QK_K/2; i++) {
        b->qs[i] = (uint8_t)(nib(rng) | (nib(rng) << 4));
    }
}

// ---- build ONE adversarial per-row block_q8_K (distinct per (row, block)) ----
// row in 0..3 selects a DISTINCT d + DC pattern so the 4 interleaved rows differ.
static void build_q8K_row(block_q8_K * a, std::mt19937 & rng, int row, int blk) {
    std::uniform_int_distribution<int> q8d(-90, 90);
    // distinct positive activation scale per (row, block)
    a->d = 0.015f + 0.0009f * ((blk + 5*row) % 13) + 0.0004f * row;
    for (int grp = 0; grp < QK_K/16; grp++) {
        // distinct nonzero DC per (row,16-group,block) so bsums != 0, vary across rows
        int dc = ((grp * 7 + blk * 3 + row * 11) % 21) - 10; // row-distinct
        int sum = 0;
        for (int ii = 0; ii < 16; ii++) {
            int v = q8d(rng) + dc;
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            a->qs[grp*16 + ii] = (int8_t)v;
            sum += v;
        }
        a->bsums[grp] = (int16_t)sum;
    }
}

// reference: output[m][gcol] for one column-group (16 cols) and one row m.
// orig: 16*nb original q4_K blocks for this group: orig[col*nb + l].
// actrows: nb per-row block_q8_K for ROW m: actrows[l]. mode perturbs the WEIGHT/min side.
// For ROWROT we instead pass a rotated activation pointer at the call site.
static void ref_group_row_mode(const block_q4_K * orig, const block_q8_K * actrow,
                               int nb, int m, float * out16, DqMode mode) {
    (void)m;
    for (int col = 0; col < 16; col++) {
        double acc = 0.0;
        std::vector<float> w(QK_K);
        for (int l = 0; l < nb; l++) {
            dequantize_block_q4_K_mode(&orig[col*nb + l], w.data(), mode);
            const block_q8_K & a = actrow[l];
            double dot = 0.0;
            for (int i = 0; i < QK_K; i++) dot += (double)w[i] * (double)a.qs[i];
            acc += (double)a.d * dot;
        }
        out16[col] = (float)acc;
    }
}

int main(int argc, char** argv) {
    bool controls = (argc > 1 && std::strcmp(argv[1], "--controls") == 0);
    struct Shape { int nc; int n; };
    std::vector<Shape> shapes = {
        {16, 256}, {16, 4096}, {16, 11008},
        {32, 256}, {32, 4096},
        {256, 256}, {256, 4096}, {256, 11008},
    };
    const int M = 4; // activation rows (one row-group, nr = 4)

    std::mt19937 rng(20260624u);
    printf("# q4_K GEMM scalar-oracle (M=4 rows, mf2/VLEN128). bar: norm < 1e-4 = PASS\n");
    printf("# %-14s %-14s %-12s %-12s %-12s %s\n", "shape(nc x n)", "max_abs_err", "rms(ref)", "norm", "rel", "verdict");

    double worst_norm = 0.0;
    bool any_bug = false;

    for (auto sh : shapes) {
        const int nc = sh.nc;
        const int n  = sh.n;
        const int nb = n / QK_K;
        const int ng = nc / 16;

        // ---- weights: per group, 16 columns x nb blocks, all distinct ----
        std::vector<block_q4_K> orig((size_t)nc * nb);
        for (int g = 0; g < ng; g++)
            for (int col = 0; col < 16; col++)
                for (int l = 0; l < nb; l++) {
                    int gcol = g*16 + col;
                    build_q4K_block(&orig[(size_t)gcol*nb + l], rng, gcol, l);
                }

        // ---- repack weights: group-major, block-major (lane=col) ----
        std::vector<block_q4_Kx16> vx((size_t)ng * nb);
        std::vector<block_q4_K> tmp16(16);
        for (int g = 0; g < ng; g++)
            for (int l = 0; l < nb; l++) {
                for (int col = 0; col < 16; col++)
                    tmp16[col] = orig[(size_t)(g*16+col)*nb + l];
                vx[(size_t)g*nb + l] = make_block_q4_Kx16(tmp16.data());
            }

        // ---- activation: 4 DISTINCT per-row block_q8_K per block (the GEMM-new axis) ----
        // actrow[m][l] = row m's block l. Then interleave into block_q8_Kx4 vy[l].
        std::vector<std::vector<block_q8_K>> actrow(M, std::vector<block_q8_K>(nb));
        for (int m = 0; m < M; m++)
            for (int l = 0; l < nb; l++)
                build_q8K_row(&actrow[m][l], rng, m, l);

        std::vector<block_q8_Kx4> vy(nb);
        for (int l = 0; l < nb; l++) {
            block_q8_K rows[4] = { actrow[0][l], actrow[1][l], actrow[2][l], actrow[3][l] };
            vy[l] = make_block_q8_Kx4(rows);
        }

        // ---- reference: out[m*nc + gcol], reading per-row block_q8_K DIRECTLY ----
        std::vector<float> ref((size_t)M * nc, 0.0f);
        for (int m = 0; m < M; m++)
            for (int g = 0; g < ng; g++)
                ref_group_row_mode(&orig[(size_t)g*16*nb], actrow[m].data(), nb, m,
                                   &ref[(size_t)m*nc + g*16], DQ_NORMAL);

        // ---- run the kernel (out zero-init), bs = nc, nr = 4, one row-group y=0 ----
        std::vector<float> out((size_t)M * nc, 0.0f);
        tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
            (size_t)n, out.data(),
            reinterpret_cast<const uint8_t*>(vx.data()),
            reinterpret_cast<const uint8_t*>(vy.data()),
            (size_t)M, (size_t)nc, (size_t)nc);

        // ---- compare over the full M*nc output ----
        double max_abs = 0.0, sumsq_ref = 0.0, max_rel = 0.0;
        int worst_idx = -1;
        for (int i = 0; i < M*nc; i++) {
            double e = fabs((double)out[i] - (double)ref[i]);
            if (e > max_abs) { max_abs = e; worst_idx = i; }
            sumsq_ref += (double)ref[i]*(double)ref[i];
            double denom = fabs((double)ref[i]);
            if (denom > 1e-6) { double r = e/denom; if (r > max_rel) max_rel = r; }
        }
        double rms = sqrt(sumsq_ref / (M*nc));
        double norm = (rms > 0) ? max_abs / rms : max_abs;
        if (norm > worst_norm) worst_norm = norm;
        const char * verdict = (norm < 1e-4) ? "PASS" : "BUG";
        if (norm >= 1e-4) any_bug = true;
        char shapestr[32]; snprintf(shapestr, sizeof shapestr, "%dx%d", nc, n);
        printf("  %-14s %-14.4e %-12.4e %-12.4e %-12.4e %s\n",
               shapestr, max_abs, rms, norm, max_rel, verdict);

        // ---- BUG localization: dump per-row / per-column divergence pattern ----
        if (norm >= 1e-4) {
            int wm = worst_idx / nc, wc = worst_idx % nc;
            printf("    !! BUG shape %s: worst idx=%d (row=%d col=%d) ours=%.6f ref=%.6f\n",
                   shapestr, worst_idx, wm, wc, (double)out[worst_idx], (double)ref[worst_idx]);
            // per-row max err -> a per-ROW(mod4) spike => qs i*4+m / bsums gsub*8+m interleave bug.
            printf("    per-row max|err|: ");
            for (int m = 0; m < M; m++) {
                double rm = 0.0;
                for (int c = 0; c < nc; c++) {
                    double e = fabs((double)out[m*nc+c]-(double)ref[m*nc+c]);
                    if (e > rm) rm = e;
                }
                printf("row%d=%.4e ", m, rm);
            }
            printf("\n");
            // per-column (mod16) max err over rows -> a per-COLUMN spike => weight lane map.
            int showc = nc < 16 ? nc : 16;
            printf("    per-col(0..%d) max|err| over rows: ", showc-1);
            for (int c = 0; c < showc; c++) {
                double cm = 0.0;
                for (int m = 0; m < M; m++) {
                    double e = fabs((double)out[m*nc+c]-(double)ref[m*nc+c]);
                    if (e > cm) cm = e;
                }
                printf("c%d=%.3e ", c, cm);
            }
            printf("\n");
        }

        // ---- NEGATIVE CONTROLS (one shape: 256x4096) ----
        if (controls && nc == 256 && n == 4096) {
            // bsums diagnostic on the interleaved activation.
            long bsum_abs = 0; int bsum_max = 0;
            for (int l = 0; l < nb; l++)
                for (int j = 0; j < QK_K/4; j++) {
                    int v = vy[l].bsums[j]; bsum_abs += (v<0?-v:v);
                    if ((v<0?-v:v) > bsum_max) bsum_max = (v<0?-v:v);
                }
            printf("  [control] q8_Kx4 bsums: mean|bsum|=%.1f max|bsum|=%d (nonzero => min term active)\n",
                   (double)bsum_abs / (nb * (QK_K/4)), bsum_max);

            // NOMIN + PERM: perturb the WEIGHT/min side of the reference.
            for (int cm = 1; cm <= 2; cm++) {
                DqMode mode = (cm == 1) ? DQ_NOMIN : DQ_PERM;
                std::vector<float> cref((size_t)M*nc, 0.0f);
                for (int m = 0; m < M; m++)
                    for (int g = 0; g < ng; g++)
                        ref_group_row_mode(&orig[(size_t)g*16*nb], actrow[m].data(), nb, m,
                                           &cref[(size_t)m*nc + g*16], mode);
                double cmax = 0.0;
                for (int i = 0; i < M*nc; i++) {
                    double e = fabs((double)out[i] - (double)cref[i]);
                    if (e > cmax) cmax = e;
                }
                double cnorm = (rms > 0) ? cmax / rms : cmax;
                const char * nm = (cm == 1) ? "NOMIN (zero min term)" : "PERM (sub-block scale +1 rotate)";
                printf("  [control] %-34s norm_vs_kernel = %.4e  (true-ref norm %.4e; teeth margin %.0fx)\n",
                       nm, cnorm, norm, cnorm / norm);
            }

            // ROWROT (GEMM-new): reference dots activation row (m+1)%4 instead of m.
            // A SPIKE proves the 4-row interleave mapping is genuinely exercised
            // (a kernel that swapped rows would match THIS perturbed ref, not the true one).
            {
                std::vector<float> cref((size_t)M*nc, 0.0f);
                for (int m = 0; m < M; m++) {
                    int mr = (m + 1) % M;
                    for (int g = 0; g < ng; g++)
                        ref_group_row_mode(&orig[(size_t)g*16*nb], actrow[mr].data(), nb, m,
                                           &cref[(size_t)m*nc + g*16], DQ_NORMAL);
                }
                double cmax = 0.0;
                for (int i = 0; i < M*nc; i++) {
                    double e = fabs((double)out[i] - (double)cref[i]);
                    if (e > cmax) cmax = e;
                }
                double cnorm = (rms > 0) ? cmax / rms : cmax;
                printf("  [control] %-34s norm_vs_kernel = %.4e  (true-ref norm %.4e; row-map margin %.0fx)\n",
                       "ROWROT (dot row (m+1)%4)", cnorm, norm, cnorm / norm);
            }
        }
    }

    printf("\nWORST_NORM %.4e  VERDICT %s\n", worst_norm, any_bug ? "BUG" : "PASS");
    return any_bug ? 1 : 0;
}
