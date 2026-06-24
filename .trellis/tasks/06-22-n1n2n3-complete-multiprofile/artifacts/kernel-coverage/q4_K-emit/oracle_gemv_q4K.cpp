// q4_K x q8_K 16x1-REPACKED GEVM numeric oracle (mf2 / VLEN128 re-derivation).
//
// Gate: does our compiler-emitted kernel
//   tcrv_emitc_..._ggml_repack_gemv_q4_K_q8_K(n, s, vx, vy, nc)
// compute the CORRECT q4_K dequant-matmul result on the SG2044 (RVV1.0, VLEN128)?
//
// Reference = INDEPENDENT scalar dequant-matmul computed from the ORIGINAL (pre-repack)
// per-block q4_K weights, dotted with the q8_K int8 quants. This is strictly more robust
// than re-deriving the repack inversion in the oracle (an inversion bug in the kernel
// cannot agree with an independent reference). The min/bsums correction is tested
// IMPLICITLY and EXACTLY: -dmin*m is baked into each dequantized weight float.
//
// Kernel input = make_block_q4_Kx16 (verbatim from ggml repack.cpp:2913, blck=1) over
// the SAME 16 original q4_K blocks per column-group.
//
// Adversarial input construction (per advisor):
//  (1) per-16-group DISTINCT nonzero DC offset on the activation -> large/varied bsums
//      -> the dmin*m*bsums min term is NOT multiplied by ~0 (catches min-term sign/index).
//  (2) per-sub-block DISTINCT scale on the weights -> the 8 sub-block 6-bit scales differ
//      clearly -> a sub-block permutation/offset bug does NOT cancel (catches "every 8th
//      element wrong" sub-block-offset bugs).
//
// out_mine is zero-init per trial: an unwritten output element reads 0 vs ref ~O(10) and
// spikes the norm, so a partial-write bug cannot hide.

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
// We store d/dmin as native _Float16 so the fp16 round-trip is IDENTICAL to what the
// kernel does (vle16_v_f16m1 + vfwcvt on the same bits).
struct block_q4_K {
    _Float16 d;                 // +0  super-block scale for quantized scales
    _Float16 dmin;              // +2  super-block scale for quantized mins
    uint8_t  scales[K_SCALE_SIZE]; // +4  6-bit scales+mins (12 bytes)
    uint8_t  qs[QK_K/2];        // +16 4-bit quants (128 bytes)
};
static_assert(sizeof(block_q4_K) == 4 + 12 + 128, "q4_K size");

// ---- repacked block_q4_Kx16 (repack.h), 2304 bytes ----
struct block_q4_Kx16 {
    _Float16 d[16];      // +0    (32 B)
    _Float16 dmin[16];   // +32   (32 B)
    uint8_t  scales[192];// +64   (192 B)
    uint8_t  qs[2048];   // +256  (2048 B)
};
static_assert(sizeof(block_q4_Kx16) == 32 + 32 + 192 + 2048, "q4_Kx16 size 2304");

// ---- plain block_q8_K (ggml-common.h), 292 bytes ----
struct block_q8_K {
    float   d;            // +0
    int8_t  qs[QK_K];     // +4
    int16_t bsums[QK_K/16];// +260 (16 * int16 = 32 B)
};
static_assert(sizeof(block_q8_K) == 4 + 256 + 32, "q8_K size 292");

// ---- ggml get_scale_min_k4 (ggml-quants.c:822), verbatim ----
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// dequant modes: 0=NORMAL (true ref), 1=NOMIN (zero the min term -> measures min-term
// magnitude as a fraction of output), 2=PERM (rotate per-sub-block scale by +1 -> measures
// sub-block-scale teeth). Modes 1/2 are NEGATIVE CONTROLS: they perturb the REFERENCE; if
// the resulting norm-vs-kernel SPIKES, the corresponding structure was genuinely exercised.
enum DqMode { DQ_NORMAL = 0, DQ_NOMIN = 1, DQ_PERM = 2 };

// ---- ggml dequantize_row_q4_K (ggml-quants.c:1471), verbatim semantics, one block ----
// y must hold QK_K floats. For DQ_PERM we collect the 8 sub-block scales first then rotate.
static void dequantize_block_q4_K_mode(const block_q4_K * x, float * y, DqMode mode) {
    const uint8_t * q = x->qs;
    const float d   = (float) x->d;
    const float min = (float) x->dmin;

    // collect the 8 sub-block scale/min codes (sb 0..7).
    uint8_t scv[8], mnv[8];
    for (int sb = 0; sb < 8; sb++) get_scale_min_k4(sb, x->scales, &scv[sb], &mnv[sb]);

    int sb = 0;
    for (int jj = 0; jj < QK_K; jj += 64) {
        // sub-block indices for this 64-group are sb and sb+1.
        int s0 = sb, s1 = sb + 1;
        if (mode == DQ_PERM) { s0 = (sb + 1) % 8; s1 = (sb + 2) % 8; } // rotate scale assignment
        const float d1 = d * scv[s0]; float m1 = min * mnv[s0];
        const float d2 = d * scv[s1]; float m2 = min * mnv[s1];
        if (mode == DQ_NOMIN) { m1 = 0.0f; m2 = 0.0f; }
        for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
        q += 32; sb += 2;
    }
}
static void dequantize_block_q4_K(const block_q4_K * x, float * y) {
    dequantize_block_q4_K_mode(x, y, DQ_NORMAL);
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

// ---- the compiler-emitted kernel under test ----
// signature: (size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc)
extern "C" void tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc);

// ---- build ONE adversarial original q4_K block ----
// col, blk index the column/block so every block is distinct.
static void build_q4K_block(block_q4_K * b, std::mt19937 & rng, int col, int blk) {
    std::uniform_int_distribution<int> nib(0, 15);
    // super-block scales: choose distinct, sane fp16 magnitudes.
    b->d    = (_Float16)(0.012f + 0.0007f * ((col + 3*blk) % 11)); // varies per col/blk
    b->dmin = (_Float16)(0.008f + 0.0005f * ((2*col + blk) % 7));

    // Build the 8 sub-block 6-bit scale (ls) and min (lm) codes DISTINCT per sub-block.
    // ls[sb], lm[sb] in [0,63]. Distinct-per-sub-block is the adversarial requirement:
    // it makes a sub-block permutation/offset bug NOT cancel.
    uint8_t ls[8], lm[8];
    for (int sb = 0; sb < 8; sb++) {
        ls[sb] = (uint8_t)( (7 + 6*sb + 2*col + blk) & 63 );  // spread across 0..63
        lm[sb] = (uint8_t)( (5 + 5*sb + col + 3*blk) & 63 );
        if (ls[sb] == 0) ls[sb] = 1; // avoid the all-zero "if(!d) continue" degenerate
    }

    // Pack ls/lm into the 12-byte q4_K scales[] exactly as quantize_row_q4_K_impl does.
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

    // Random 4-bit quants.
    for (int i = 0; i < QK_K/2; i++) {
        b->qs[i] = (uint8_t)(nib(rng) | (nib(rng) << 4));
    }
}

// reference dot for one output column-group (16 columns) and accumulate into ref[16].
// orig: pointer to 16*nb original q4_K blocks for this group (block-major within group:
//       orig[col*nb + l]). act: nb q8_K blocks.
static void ref_group_mode(const block_q4_K * orig, const block_q8_K * act, int nb, float * ref16, DqMode mode) {
    for (int col = 0; col < 16; col++) {
        double acc = 0.0;
        std::vector<float> w(QK_K);
        for (int l = 0; l < nb; l++) {
            dequantize_block_q4_K_mode(&orig[col*nb + l], w.data(), mode);
            const block_q8_K & a = act[l];
            double dot = 0.0;
            for (int i = 0; i < QK_K; i++) dot += (double)w[i] * (double)a.qs[i];
            acc += (double)a.d * dot;
        }
        ref16[col] = (float)acc;
    }
}
static void ref_group(const block_q4_K * orig, const block_q8_K * act, int nb, float * ref16) {
    ref_group_mode(orig, act, nb, ref16, DQ_NORMAL);
}

int main(int argc, char** argv) {
    bool controls = (argc > 1 && std::strcmp(argv[1], "--controls") == 0);
    // shapes: nc (output columns, multiple of 16) and n (contraction length, multiple of 256)
    struct Shape { int nc; int n; };
    std::vector<Shape> shapes = {
        {16, 256}, {16, 4096}, {16, 11008},
        {32, 256}, {32, 4096},
        {256, 256}, {256, 4096}, {256, 11008},
    };

    std::mt19937 rng(20260624u);
    printf("# q4_K GEVM scalar-oracle (mf2/VLEN128 re-derivation). bar: norm < 1e-4 = PASS\n");
    printf("# %-14s %-14s %-12s %-12s %-12s %s\n", "shape(nc x n)", "max_abs_err", "rms(ref)", "norm", "rel", "verdict");

    double worst_norm = 0.0;
    bool any_bug = false;

    for (auto sh : shapes) {
        const int nc = sh.nc;
        const int n  = sh.n;
        const int nb = n / QK_K;        // blocks along contraction
        const int ng = nc / 16;         // column-groups

        // ---- build original q4_K weights: for each group, 16 columns x nb blocks ----
        // Layout per group in 'orig': orig[g][col*nb + l].
        std::vector<block_q4_K> orig((size_t)nc * nb);
        for (int g = 0; g < ng; g++) {
            for (int col = 0; col < 16; col++) {
                for (int l = 0; l < nb; l++) {
                    int gcol = g*16 + col;
                    build_q4K_block(&orig[(size_t)(g*16+col)*nb + l], rng, gcol, l);
                }
            }
        }

        // ---- repack into block_q4_Kx16: group-major, block-major; the kernel reads
        //      vx + g*nb*2304 + l*2304, and within that block lane=col. make_block_q4_Kx16
        //      takes a contiguous array of 16 block_q4_K (one per lane/col). ----
        std::vector<block_q4_Kx16> vx((size_t)ng * nb);
        std::vector<block_q4_K> tmp16(16);
        for (int g = 0; g < ng; g++) {
            for (int l = 0; l < nb; l++) {
                for (int col = 0; col < 16; col++) {
                    tmp16[col] = orig[(size_t)(g*16+col)*nb + l];
                }
                vx[(size_t)g*nb + l] = make_block_q4_Kx16(tmp16.data());
            }
        }

        // ---- build the q8_K activation: nb blocks, with per-16-group DISTINCT DC offset
        //      so bsums are large + varied (exercises the min term). We construct the q8_K
        //      block DIRECTLY (we control d, qs, bsums) like quantize_row_q8_K but with an
        //      injected DC pattern; bsums computed from qs exactly as ggml does. ----
        std::vector<block_q8_K> act(nb);
        std::uniform_int_distribution<int> q8d(-90, 90);
        for (int l = 0; l < nb; l++) {
            block_q8_K & a = act[l];
            a.d = 0.015f + 0.0009f * (l % 13);   // positive activation scale (varies per block)
            for (int grp = 0; grp < QK_K/16; grp++) {
                // distinct nonzero DC per 16-group so bsums != 0 and vary across groups
                int dc = ((grp * 7 + l * 3) % 21) - 10; // in [-10,10], group-distinct
                int sum = 0;
                for (int ii = 0; ii < 16; ii++) {
                    int v = q8d(rng) + dc;
                    if (v > 127) v = 127;
                    if (v < -128) v = -128;
                    a.qs[grp*16 + ii] = (int8_t)v;
                    sum += v;
                }
                a.bsums[grp] = (int16_t)sum;
            }
        }

        // ---- reference ----
        std::vector<float> ref(nc, 0.0f);
        for (int g = 0; g < ng; g++) {
            ref_group(&orig[(size_t)g*16*nb], act.data(), nb, &ref[g*16]);
        }

        // ---- run the kernel (out zero-init) ----
        std::vector<float> out(nc, 0.0f);
        tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K(
            (size_t)n, out.data(),
            reinterpret_cast<const uint8_t*>(vx.data()),
            reinterpret_cast<const uint8_t*>(act.data()),
            (size_t)nc);

        // ---- compare ----
        double max_abs = 0.0, sumsq_ref = 0.0, max_rel = 0.0;
        int worst_idx = -1;
        for (int i = 0; i < nc; i++) {
            double e = fabs((double)out[i] - (double)ref[i]);
            if (e > max_abs) { max_abs = e; worst_idx = i; }
            sumsq_ref += (double)ref[i]*(double)ref[i];
            double denom = fabs((double)ref[i]);
            if (denom > 1e-6) { double r = e/denom; if (r > max_rel) max_rel = r; }
        }
        double rms = sqrt(sumsq_ref / nc);
        double norm = (rms > 0) ? max_abs / rms : max_abs;
        if (norm > worst_norm) worst_norm = norm;
        const char * verdict = (norm < 1e-4) ? "PASS" : "BUG";
        if (norm >= 1e-4) any_bug = true;
        char shapestr[32]; snprintf(shapestr, sizeof shapestr, "%dx%d", nc, n);
        printf("  %-14s %-14.4e %-12.4e %-12.4e %-12.4e %s\n",
               shapestr, max_abs, rms, norm, max_rel, verdict);

        // If BUG, dump the divergence pattern for localization.
        if (norm >= 1e-4) {
            printf("    !! BUG shape %s: worst idx=%d ours=%.6f ref=%.6f\n",
                   shapestr, worst_idx, (double)out[worst_idx], (double)ref[worst_idx]);
            int show = nc < 32 ? nc : 32;
            printf("    per-element (first %d): idx ours ref err\n", show);
            for (int i = 0; i < show; i++) {
                printf("      [%2d] %12.5f %12.5f %12.5f\n", i, (double)out[i], (double)ref[i],
                       (double)out[i]-(double)ref[i]);
            }
        }

        // ---- NEGATIVE CONTROLS (one shape: 256x4096) ----
        // Perturb the REFERENCE; a SPIKE in norm-vs-kernel proves the corresponding
        // structure was genuinely exercised (the correct test would catch a kernel bug there).
        if (controls && nc == 256 && n == 4096) {
            // bsums diagnostic: confirm the min term is non-trivial in the inputs.
            long bsum_abs = 0; int bsum_max = 0;
            for (int l = 0; l < nb; l++)
                for (int grp = 0; grp < QK_K/16; grp++) {
                    int v = act[l].bsums[grp]; bsum_abs += (v<0?-v:v);
                    if ((v<0?-v:v) > bsum_max) bsum_max = (v<0?-v:v);
                }
            printf("  [control] bsums: mean|bsum|=%.1f max|bsum|=%d (nonzero => min term active)\n",
                   (double)bsum_abs / (nb * (QK_K/16)), bsum_max);

            for (int cm = 1; cm <= 2; cm++) {
                DqMode mode = (cm == 1) ? DQ_NOMIN : DQ_PERM;
                std::vector<float> cref(nc, 0.0f);
                for (int g = 0; g < ng; g++)
                    ref_group_mode(&orig[(size_t)g*16*nb], act.data(), nb, &cref[g*16], mode);
                double cmax = 0.0;
                for (int i = 0; i < nc; i++) {
                    double e = fabs((double)out[i] - (double)cref[i]);
                    if (e > cmax) cmax = e;
                }
                double cnorm = (rms > 0) ? cmax / rms : cmax;
                const char * nm = (cm == 1) ? "NOMIN (zero min term)" : "PERM (sub-block scale +1 rotate)";
                printf("  [control] %-34s norm_vs_kernel = %.4e  (true-ref norm %.4e; teeth margin %.0fx)\n",
                       nm, cnorm, norm, cnorm / norm);
            }
        }
    }

    printf("\nWORST_NORM %.4e  VERDICT %s\n", worst_norm, any_bug ? "BUG" : "PASS");
    return any_bug ? 1 : 0;
}
