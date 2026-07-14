// q6_K x q8_K 16x1-REPACKED GEVM + GEMM numeric oracle.
//
// Gate: does the compiler-emitted q6_K repack kernel compute the CORRECT q6_K
// dequant-matmul result?
//
// The oracle is INDEPENDENT: the REFERENCE decodes the ORIGINAL (pre-repack)
// per-block q6_K exactly as ggml's canonical dequantize_row_q6_K / vec_dot_q6_K_q8_K
// (6-bit value = ((ql&0xF | ((qh>>shift)&3)<<4)) - 32, per-16-element SIGNED int8
// scale, SINGLE super-block d, NO min), while the EMITTER-MODEL reads the REPACKED
// block_q6_Kx16 and reproduces the emitter's EXACT lane-wise integer organization
// (block-as-lane strips, 8-position i16 chunk accumulate, i32 scale-weighted fold,
// single accumulator). The two paths use DIFFERENT data layouts and DIFFERENT
// arithmetic organization, so agreement is strong evidence.
//
// The integer contraction isum is compared BYTE-EXACT (int64 equality) between the
// reference and the emitter-model -- this is the core numeric certificate. The i16
// intermediate is tracked to certify no overflow occurred (the 8-position chunk stays
// within int16 for the test data). The fp result norm is reported informationally.
//
// NEGATIVE CONTROLS perturb the REFERENCE; an integer-mismatch SPIKE proves the
// corresponding q6_K structure was genuinely exercised by the emitter-model:
//   QH-OFF    : zero the high-2-bit qh plane in the ref -> weights lose bits 4-5.
//   SCALE-ROT : rotate the 16 signed scales by +1 -> a scale permutation bug cannot
//               cancel (distinct-per-sub-block scales required).
//   BIAS-OFF  : drop the -32 offset-binary bias in the ref -> the signed weight lane.
//
// Build:  g++ -O2 -std=c++17 oracle_repack_q6K.cpp -o /tmp/oracle_q6k && /tmp/oracle_q6k

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define QK_K 256

// ---- original per-block q6_K (ggml-common.h), 210 bytes ----
struct block_q6_K {
    uint8_t ql[QK_K / 2];      // +0    lower 4 bits (128 B)
    uint8_t qh[QK_K / 4];      // +128  upper 2 bits (64 B)
    int8_t  scales[QK_K / 16]; // +192  16 SIGNED int8 scales
    float   d;                 // super-block scale (we keep float; fp16 does not
                               // affect the byte-exact INTEGER isum certificate)
};

// ---- repacked block_q6_Kx16 (this project's block-as-lane convention), 3360 B ----
//   d[16]      @ +0     (32 B, fp16 in the real kernel; float here for the int test)
//   scales[256]@ +32    (256 B: signed int8, byte for scale s / column c at 32+s*16+c)
//   qh[1024]   @ +288   (1024 B: byte for qh index i / column c at 288+i*16+c)
//   ql[2048]   @ +1312  (2048 B: byte for ql index i / column c at 1312+i*16+c)
struct block_q6_Kx16 {
    float   d[16];
    int8_t  scales[256];
    uint8_t qh[1024];
    uint8_t ql[2048];
};

// ---- plain block_q8_K, 292 B (bsums UNUSED by q6_K) ----
struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};

// ---- make_block_q6_Kx16: interleave 16 original q6_K blocks (one per lane/col) ----
static block_q6_Kx16 make_block_q6_Kx16(const block_q6_K *in) {
    block_q6_Kx16 out;
    for (int c = 0; c < 16; ++c) out.d[c] = in[c].d;
    for (int s = 0; s < 16; ++s)
        for (int c = 0; c < 16; ++c)
            out.scales[s * 16 + c] = in[c].scales[s];
    for (int i = 0; i < 64; ++i)
        for (int c = 0; c < 16; ++c)
            out.qh[i * 16 + c] = in[c].qh[i];
    for (int i = 0; i < 128; ++i)
        for (int c = 0; c < 16; ++c)
            out.ql[i * 16 + c] = in[c].ql[i];
    return out;
}

enum DqMode { DQ_NORMAL = 0, DQ_QHOFF = 1, DQ_SCALEROT = 2, DQ_BIASOFF = 3 };

// ---- REFERENCE integer isum for ONE original q6_K block dotted with ONE q8_K block.
// Mirrors ggml's canonical decode; returns sum_i(scale_i * weight6_i * q8_i) as int64.
static int64_t ref_isum_block(const block_q6_K *x, const block_q8_K *a, DqMode mode) {
    int8_t sc[16];
    for (int i = 0; i < 16; ++i) sc[i] = x->scales[i];
    if (mode == DQ_SCALEROT)
        for (int i = 0; i < 16; ++i) sc[i] = x->scales[(i + 1) % 16];
    int bias = (mode == DQ_BIASOFF) ? 0 : 32;
    int qhmask = (mode == DQ_QHOFF) ? 0 : 3;

    int64_t isum = 0;
    for (int n = 0; n < QK_K; n += 128) {
        const uint8_t *ql = x->ql + (n / 128) * 64;
        const uint8_t *qh = x->qh + (n / 128) * 32;
        const int sb = (n / 128) * 8; // ggml advances sc += 8 per super-half
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            int q1 = ((ql[l +  0] & 0xF) | (((qh[l] >> 0) & qhmask) << 4)) - bias;
            int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & qhmask) << 4)) - bias;
            int q3 = ((ql[l +  0] >>  4) | (((qh[l] >> 4) & qhmask) << 4)) - bias;
            int q4 = ((ql[l + 32] >>  4) | (((qh[l] >> 6) & qhmask) << 4)) - bias;
            int base = n + l;
            isum += (int64_t)sc[sb + is + 0] * q1 * a->qs[base +  0];
            isum += (int64_t)sc[sb + is + 2] * q2 * a->qs[base + 32];
            isum += (int64_t)sc[sb + is + 4] * q3 * a->qs[base + 64];
            isum += (int64_t)sc[sb + is + 6] * q4 * a->qs[base + 96];
        }
    }
    return isum;
}

// The 4-quadrant metadata the emitter uses (must match the emitter exactly).
struct QuadInfo { int qlStream; int highNib; int qhShift; };
static const QuadInfo kQuads[4] = {{0, 0, 0}, {1, 0, 2}, {0, 1, 4}, {1, 1, 6}};

// ---- EMITTER-MODEL integer isum for column c of a repacked group, dotted with a
// per-column activation quant accessor `act(globalPos)`. Reproduces the emitter's
// EXACT lane-wise organization (j x sh x k-chunk-of-8 x 4 quads, i16 partials folded
// scale-weighted into i32). `maxAbsPartial` tracks the i16 stage magnitude.
template <typename ActFn>
static int64_t mine_isum_col(const block_q6_Kx16 *b, int c, ActFn act,
                             int &maxAbsPartial) {
    int64_t sumi = 0;
    for (int j = 0; j < 2; ++j) {
        for (int sh = 0; sh < 2; ++sh) {
            int scale[4];
            for (int q = 0; q < 4; ++q)
                scale[q] = b->scales[(j * 8 + q * 2 + sh) * 16 + c];
            for (int k = 0; k < 2; ++k) {
                int partial[4] = {0, 0, 0, 0};
                for (int p = 0; p < 8; ++p) {
                    int ll = sh * 16 + k * 8 + p;
                    uint8_t qlA = b->ql[(j * 64 + ll) * 16 + c];
                    uint8_t qlB = b->ql[(j * 64 + 32 + ll) * 16 + c];
                    uint8_t qh  = b->qh[(j * 32 + ll) * 16 + c];
                    for (int q = 0; q < 4; ++q) {
                        uint8_t qlSel = (kQuads[q].qlStream == 0) ? qlA : qlB;
                        int nib = kQuads[q].highNib ? (qlSel >> 4) : (qlSel & 0xF);
                        int raw = nib | (((qh >> kQuads[q].qhShift) & 3) << 4);
                        int w = raw - 32; // signed [-32,31]
                        int aq = act(j * 128 + q * 32 + ll);
                        partial[q] += aq * w; // i16 accumulate over 8 positions
                    }
                }
                for (int q = 0; q < 4; ++q) {
                    int ap = std::abs(partial[q]);
                    if (ap > maxAbsPartial) maxAbsPartial = ap;
                    sumi += (int64_t)scale[q] * partial[q];
                }
            }
        }
    }
    return sumi;
}

// ---- build ONE adversarial original q6_K block (distinct per col/blk) ----
static void build_q6K_block(block_q6_K *x, std::mt19937 &rng, int col, int blk) {
    std::uniform_int_distribution<int> byte(0, 255);
    x->d = 0.011f + 0.0007f * ((col + 3 * blk) % 11);
    // distinct-per-sub-block SIGNED scales in [-32,31] (avoid all-zero degenerate).
    for (int i = 0; i < 16; ++i) {
        int v = ((7 + 5 * i + 2 * col + 3 * blk) % 63) - 31;
        if (v == 0) v = 1;
        x->scales[i] = (int8_t)v;
    }
    for (int i = 0; i < QK_K / 2; ++i) x->ql[i] = (uint8_t)byte(rng); // full 8-bit ql
    for (int i = 0; i < QK_K / 4; ++i) x->qh[i] = (uint8_t)byte(rng); // full 8-bit qh
}

// ---- build a q8_K activation block with per-16-group distinct DC (varied values) ----
static void build_q8K_block(block_q8_K *a, std::mt19937 &rng, int blk) {
    std::uniform_int_distribution<int> q8d(-90, 90);
    a->d = 0.015f + 0.0009f * (blk % 13);
    for (int grp = 0; grp < QK_K / 16; ++grp) {
        int dc = ((grp * 7 + blk * 3) % 21) - 10;
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii) {
            int v = q8d(rng) + dc;
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            a->qs[grp * 16 + ii] = (int8_t)v;
            sum += v;
        }
        a->bsums[grp] = (int16_t)sum;
    }
}

int main() {
    printf("# q6_K x q8_K 16x1-REPACKED oracle (independent scalar).\n");
    printf("# BYTE-EXACT integer isum: reference (original q6_K) vs emitter-model "
           "(repacked block_q6_Kx16).\n\n");

    std::mt19937 rng(20260708u);
    struct Shape { int nc; int n; int nr; };
    std::vector<Shape> shapes = {
        {16, 256, 1}, {16, 4096, 1}, {32, 256, 4}, {32, 2560, 4},
        {256, 256, 1}, {256, 4096, 8}, {160, 2560, 4},
    };

    bool anyBug = false;
    int globalMaxPartial = 0;
    double worstNorm = 0.0;

    for (auto sh : shapes) {
        const int nc = sh.nc, n = sh.n, nr = sh.nr;
        const int nb = n / QK_K, ng = nc / 16;

        // original weights: [group][col][block]
        std::vector<block_q6_K> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_q6K_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);

        // repack: [group][block] -> block_q6_Kx16 (16 columns interleaved)
        std::vector<block_q6_Kx16> vx((size_t)ng * nb);
        std::vector<block_q6_K> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_q6_Kx16(tmp16.data());
            }

        // activations: nr rows x nb blocks (plain q8_K per row).
        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        // ---- BYTE-EXACT integer isum: reference vs emitter-model, all (row,col) ----
        long long mismatch = 0;
        int maxPartial = 0;
        double sumsqRef = 0, maxAbsErr = 0;
        for (int r = 0; r < nr; ++r) {
            for (int g = 0; g < ng; ++g) {
                for (int col = 0; col < 16; ++col) {
                    int gcol = g * 16 + col;
                    int64_t refI = 0, mineI = 0;
                    double refF = 0, mineF = 0;
                    for (int l = 0; l < nb; ++l) {
                        const block_q6_K *xo = &orig[(size_t)gcol * nb + l];
                        const block_q8_K *a = &act[(size_t)r * nb + l];
                        int64_t ri = ref_isum_block(xo, a, DQ_NORMAL);
                        const block_q6_Kx16 *b = &vx[(size_t)g * nb + l];
                        auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                        int64_t mi = mine_isum_col(b, col, actFn, maxPartial);
                        if (ri != mi) mismatch++;
                        refI += ri; mineI += mi;
                        refF  += (double)xo->d * (double)a->d * (double)ri;
                        mineF += (double)b->d[col] * (double)a->d * (double)mi;
                    }
                    (void)refI; (void)mineI;
                    double e = std::fabs(refF - mineF);
                    if (e > maxAbsErr) maxAbsErr = e;
                    sumsqRef += refF * refF;
                }
            }
        }
        int ncnt = nr * nc;
        double rms = std::sqrt(sumsqRef / ncnt);
        double norm = rms > 0 ? maxAbsErr / rms : maxAbsErr;
        if (norm > worstNorm) worstNorm = norm;
        if (maxPartial > globalMaxPartial) globalMaxPartial = maxPartial;
        bool bug = (mismatch != 0);
        if (bug) anyBug = true;
        printf("  shape nr=%-3d nc=%-4d n=%-6d : int-mismatch=%-6lld  "
               "maxAbsPartial=%-6d (i16 ok<=32767)  fp-norm=%.3e  %s\n",
               nr, nc, n, mismatch, maxPartial, norm,
               bug ? "INT-BUG" : "BYTE-EXACT");
    }

    // ---- GEMM interleaved-activation addressing: pack 4 plain q8_K rows into one
    // block_q8_Kx4 (qs at +16, byte for flat pos p / column c = p*4 + c) and confirm
    // the emitter-model's interleaved read is BYTE-EXACT to the per-row plain dot.
    // This exercises the ONE GEMM-specific piece the GEVM does not: the x4 interleave.
    {
        const int nc = 32, n = 2560;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_q6_K> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_q6K_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);
        std::vector<block_q6_Kx16> vx((size_t)ng * nb);
        std::vector<block_q6_K> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_q6_Kx16(tmp16.data());
            }
        // 4 activation rows (one q8_Kx4 group), nb blocks.
        std::vector<block_q8_K> row(4 * nb);
        for (int r = 0; r < 4; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8K_block(&row[(size_t)r * nb + l], rng, 900 + r * 7 + l);
        // interleaved qs: for block l, flat pos p, column c -> qsx4[l][p*4 + c].
        std::vector<int8_t> qsx4((size_t)nb * QK_K * 4);
        for (int l = 0; l < nb; ++l)
            for (int p = 0; p < QK_K; ++p)
                for (int c = 0; c < 4; ++c)
                    qsx4[(size_t)l * QK_K * 4 + p * 4 + c] =
                        row[(size_t)c * nb + l].qs[p];
        long long mism = 0, total = 0;
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col) {
                int gcol = g * 16 + col;
                for (int c = 0; c < 4; ++c)   // activation column within the x4 group
                    for (int l = 0; l < nb; ++l) {
                        int64_t ref = ref_isum_block(&orig[(size_t)gcol * nb + l],
                                                     &row[(size_t)c * nb + l],
                                                     DQ_NORMAL);
                        const int8_t *qsl = &qsx4[(size_t)l * QK_K * 4];
                        auto actFn = [&](int pos) { return (int)qsl[pos * 4 + c]; };
                        int dummy = 0;
                        int64_t mine = mine_isum_col(&vx[(size_t)g * nb + l], col,
                                                     actFn, dummy);
                        total++;
                        if (ref != mine) mism++;
                    }
            }
        printf("\n# GEMM interleaved block_q8_Kx4 (qs @+16, pos*4+c): "
               "int-mismatch = %lld / %lld  %s\n",
               mism, total, mism ? "INT-BUG" : "BYTE-EXACT");
        if (mism) anyBug = true;
    }

    // ---- NEGATIVE CONTROLS on one shape (nr=4, nc=32, n=2560) ----
    printf("\n# negative controls (perturb the REFERENCE; a large integer-mismatch\n"
           "# fraction proves the q6_K feature is genuinely exercised):\n");
    {
        const int nc = 32, n = 2560, nr = 4;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_q6_K> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_q6K_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);
        std::vector<block_q6_Kx16> vx((size_t)ng * nb);
        std::vector<block_q6_K> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_q6_Kx16(tmp16.data());
            }
        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        const char *names[3] = {"QH-OFF   (zero high-2-bit plane)",
                                "SCALE-ROT(rotate signed scales +1)",
                                "BIAS-OFF (drop -32 offset-binary)"};
        DqMode modes[3] = {DQ_QHOFF, DQ_SCALEROT, DQ_BIASOFF};
        for (int m = 0; m < 3; ++m) {
            long long total = 0, mism = 0;
            for (int r = 0; r < nr; ++r)
                for (int g = 0; g < ng; ++g)
                    for (int col = 0; col < 16; ++col) {
                        int gcol = g * 16 + col;
                        for (int l = 0; l < nb; ++l) {
                            const block_q6_K *xo = &orig[(size_t)gcol * nb + l];
                            const block_q8_K *a = &act[(size_t)r * nb + l];
                            int dummy = 0;
                            int64_t refP = ref_isum_block(xo, a, modes[m]);
                            auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                            int64_t mine = mine_isum_col(&vx[(size_t)g * nb + l],
                                                         col, actFn, dummy);
                            total++;
                            if (refP != mine) mism++;
                        }
                    }
            printf("  %-34s : perturbed-ref vs emitter int-mismatch = %lld / %lld "
                   "(%.0f%%)  %s\n",
                   names[m], mism, total, 100.0 * mism / total,
                   mism > 0 ? "EXERCISED" : "!! NOT EXERCISED");
        }
    }

    printf("\nWORST_FP_NORM %.3e   GLOBAL_MAX_i16_PARTIAL %d (bound 32767)   "
           "VERDICT %s\n",
           worstNorm, globalMaxPartial, anyBug ? "INT-BUG" : "BYTE-EXACT-INTEGER");
    return anyBug ? 1 : 0;
}
