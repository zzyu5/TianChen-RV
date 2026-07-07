// q2_K x q8_K 16x1-REPACKED GEVM + GEMM numeric oracle.
//
// Gate: does the compiler-emitted q2_K repack kernel compute the CORRECT q2_K
// dequant-matmul result?
//
// The oracle is INDEPENDENT: the REFERENCE decodes the ORIGINAL (pre-repack)
// per-block q2_K exactly as ggml's canonical ggml_vec_dot_q2_K_q8_K (2-bit weight
// = (qs[l] >> shift) & 3, per-16-element 4-bit packed scale/min: scale = sc & 0x0F,
// min = sc >> 4, DUAL super-block d/dmin, MIN term folded via the activation bsums),
// while the EMITTER-MODEL reads the REPACKED block_q2_Kx16 and reproduces the
// emitter's EXACT lane-wise integer organization (block-as-lane strips, 16-position
// i16 partial per sub-block, i32 scale-weighted fold, dual d/dmin + bsums-min fold).
// The two paths use DIFFERENT data layouts and DIFFERENT arithmetic organization, so
// agreement is strong evidence.
//
// q2_K is the MIN-TERM regression (back to the q4_K/q5_K dual d-dmin + bsums-min
// fold, NOT q6_K's single no-min accumulator):
//   * 2-BIT weight: 4 lanes per byte, (byte >> {0,2,4,6}) & 3, UNSIGNED [0,3], NO
//     offset-binary bias (the bias lives entirely in the per-sub-block 4-bit MIN).
//   * 4-BIT packed scale/min: sc & 0x0F is the sub-block scale, sc >> 4 the min
//     (q2_K-specific: a SINGLE byte per sub-block, low nibble scale / high nibble min
//     -- NOT q4_K's 6-bit two-byte bit-dance).
//   * 16 sub-blocks of 16 elements each (NOT q4_K's 8 of 32): each sub-block is
//     exactly ONE q8_K bsums group, so the MIN term reads a SINGLE bsum per sub-block
//     (NOT q4_K's paired bs0+bs1).
//
// TWO integer certificates are compared BYTE-EXACT (int64 equality) between the
// reference and the emitter-model:
//   isum  = sum_sub( scale_sub * sum_l(q8_l * weight2_l) )   -- the scale MAIN term.
//   summs = sum_sub( min_sub * bsum_sub )                    -- the MIN term.
// Both are pure integers (no fp reassociation), so this is a hard certificate. The
// i16 partial magnitude is tracked to certify no overflow (16*127*3 = 6096 < 32767).
// The fp result norm (dall*isum - dmin*summs) is reported informationally.
//
// NEGATIVE CONTROLS perturb the REFERENCE; a mismatch SPIKE proves the corresponding
// q2_K structure was genuinely exercised by the emitter-model:
//   2BIT      : mask the weight to its LOW bit only ((byte>>shift)&1) -> the 2-bit
//               assembly loses its high bit; isum diverges.
//   SCALE-ROT : rotate the 4-bit SCALE nibble by +1 sub-block (min untouched) -> a
//               scale permutation bug cannot cancel; isum diverges.
//   MIN-ROT   : rotate the 4-bit MIN nibble by +1 sub-block (scale untouched) -> the
//               bsums-min term is genuinely consumed; summs diverges.
//
// Build:  g++ -O2 -std=c++17 oracle_repack_q2K.cpp -o /tmp/oracle_q2k && /tmp/oracle_q2k

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define QK_K 256

// ---- original per-block q2_K (ggml-common.h), 84 bytes ----
struct block_q2_K {
    uint8_t scales[QK_K / 16]; // +0   16 packed 4-bit scale (low) / 4-bit min (high)
    uint8_t qs[QK_K / 4];      // +16  64 B: 2-bit quants (4 per byte)
    float   d;                 // super-block scale for quantized scales (fp16 in ggml;
                               // float here does not affect the byte-exact INTEGER cert)
    float   dmin;              // super-block scale for quantized mins
};

// ---- repacked block_q2_Kx16 (this project's block-as-lane convention), 1344 B ----
//   d[16]      @ +0    (32 B, fp16 in the real kernel; float here for the int test)
//   dmin[16]   @ +32   (32 B, fp16 in the real kernel; float here)
//   scales[256]@ +64   (256 B: packed scale/min byte for sub s / column c at 64+s*16+c)
//   qs[1024]   @ +320  (1024 B: 2-bit quant byte for qs index i / column c at 320+i*16+c)
struct block_q2_Kx16 {
    float   d[16];
    float   dmin[16];
    uint8_t scales[256];
    uint8_t qs[1024];
};

// ---- plain block_q8_K, 292 B ----
struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};

// ---- make_block_q2_Kx16: interleave 16 original q2_K blocks (one per lane/col) ----
static block_q2_Kx16 make_block_q2_Kx16(const block_q2_K *in) {
    block_q2_Kx16 out;
    for (int c = 0; c < 16; ++c) { out.d[c] = in[c].d; out.dmin[c] = in[c].dmin; }
    for (int s = 0; s < 16; ++s)
        for (int c = 0; c < 16; ++c)
            out.scales[s * 16 + c] = in[c].scales[s];
    for (int i = 0; i < 64; ++i)
        for (int c = 0; c < 16; ++c)
            out.qs[i * 16 + c] = in[c].qs[i];
    return out;
}

enum DqMode { DQ_NORMAL = 0, DQ_2BIT = 1, DQ_SCALEROT = 2, DQ_MINROT = 3 };

// ---- REFERENCE (ggml canonical ggml_vec_dot_q2_K_q8_K). Returns the two integer
// certificates: isum (scale main) and summs (min). Mirrors ggml's scalar loop. ----
static void ref_block(const block_q2_K *x, const block_q8_K *a, DqMode mode,
                      int64_t &isum, int64_t &summs) {
    // 4-bit scale / min per sub-block, with optional isolated rotation controls.
    int scale[16], mins[16];
    for (int s = 0; s < 16; ++s) {
        int scByte = x->scales[s];
        int scRotByte = x->scales[(s + 1) % 16];
        scale[s] = ((mode == DQ_SCALEROT) ? scRotByte : scByte) & 0x0F;
        mins[s]  = (((mode == DQ_MINROT)  ? scRotByte : scByte) >> 4) & 0x0F;
    }
    int wmask = (mode == DQ_2BIT) ? 1 : 3; // 2BIT control drops the high weight bit

    isum = 0;
    int is = 0;
    for (int k = 0; k < QK_K / 128; ++k) {           // super-half
        const uint8_t *q2 = x->qs + k * 32;
        const int8_t  *q8 = a->qs + k * 128;
        int shift = 0;
        for (int j = 0; j < 4; ++j) {                // shift = 0,2,4,6
            int d0 = scale[is];                      // elements l=0..15
            int isuml = 0;
            for (int l = 0; l < 16; ++l)
                isuml += q8[j * 32 + l] * ((q2[l] >> shift) & wmask);
            isum += (int64_t)d0 * isuml;
            int d1 = scale[is + 1];                  // elements l=16..31
            isuml = 0;
            for (int l = 16; l < 32; ++l)
                isuml += q8[j * 32 + l] * ((q2[l] >> shift) & wmask);
            isum += (int64_t)d1 * isuml;
            is += 2;
            shift += 2;
        }
    }
    summs = 0;
    for (int s = 0; s < 16; ++s) summs += (int64_t)a->bsums[s] * mins[s];
}

// ---- EMITTER-MODEL for column c of a repacked group, dotted with a per-column
// activation quant accessor `act(globalPos)` + a per-sub-block bsum accessor
// `bsum(sub)`. Reproduces the emitter's EXACT lane-wise organization: per super-half,
// per m-half (even/odd sub-blocks), a 16-position i16 partial per shift-j, folded
// scale-weighted into i32; the MIN term folds a SINGLE bsum per sub-block. ----
template <typename ActFn, typename BsumFn>
static void mine_col(const block_q2_Kx16 *b, int c, ActFn act, BsumFn bsum,
                     int64_t &sumi, int64_t &summs, int &maxAbsPartial) {
    sumi = 0;
    for (int k = 0; k < 2; ++k) {                 // super-half
        int scale[8];
        for (int sb = 0; sb < 8; ++sb)
            scale[sb] = b->scales[(k * 8 + sb) * 16 + c] & 0x0F;
        for (int mh = 0; mh < 2; ++mh) {          // m-half: 0 -> even sub, 1 -> odd
            int partial[4] = {0, 0, 0, 0};        // one per shift-j
            for (int mm = 0; mm < 16; ++mm) {
                int m = mh * 16 + mm;
                uint8_t byte = b->qs[(k * 32 + m) * 16 + c];
                for (int j = 0; j < 4; ++j) {
                    int w = (byte >> (2 * j)) & 3; // 2-bit lane, UNSIGNED
                    int aq = act(k * 128 + j * 32 + m);
                    partial[j] += aq * w;          // i16 accumulate over 16 positions
                }
            }
            for (int j = 0; j < 4; ++j) {
                int ap = std::abs(partial[j]);
                if (ap > maxAbsPartial) maxAbsPartial = ap;
                sumi += (int64_t)scale[2 * j + mh] * partial[j];
            }
        }
    }
    // MIN term: single bsum per 16-element sub-block, weighted by the 4-bit min.
    summs = 0;
    for (int sub = 0; sub < 16; ++sub) {
        int mn = b->scales[sub * 16 + c] >> 4;
        summs += (int64_t)bsum(sub) * mn;
    }
}

// ---- build ONE adversarial original q2_K block (distinct per col/blk) ----
static void build_q2K_block(block_q2_K *x, std::mt19937 &rng, int col, int blk) {
    std::uniform_int_distribution<int> byte(0, 255);
    x->d    = 0.011f + 0.0007f * ((col + 3 * blk) % 11);
    x->dmin = 0.006f + 0.0005f * ((col + 5 * blk) % 7);
    // distinct-per-sub-block packed scale (low nibble) + min (high nibble), non-degenerate.
    for (int s = 0; s < 16; ++s) {
        int sc = 1 + (3 + 5 * s + 2 * col + 3 * blk) % 15; // 1..15 scale
        int mn = 1 + (7 + 2 * s + 3 * col + 5 * blk) % 15; // 1..15 min
        x->scales[s] = (uint8_t)((mn << 4) | sc);
    }
    for (int i = 0; i < QK_K / 4; ++i) x->qs[i] = (uint8_t)byte(rng); // full 8-bit qs
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
    printf("# q2_K x q8_K 16x1-REPACKED oracle (independent scalar).\n");
    printf("# BYTE-EXACT integer certificates isum (scale main) + summs (bsums-min):\n");
    printf("#   reference (original q2_K) vs emitter-model (repacked block_q2_Kx16).\n\n");

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

        std::vector<block_q2_K> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_q2K_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);

        std::vector<block_q2_Kx16> vx((size_t)ng * nb);
        std::vector<block_q2_K> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_q2_Kx16(tmp16.data());
            }

        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        long long mismatch = 0;
        int maxPartial = 0;
        double sumsqRef = 0, maxAbsErr = 0;
        for (int r = 0; r < nr; ++r) {
            for (int g = 0; g < ng; ++g) {
                for (int col = 0; col < 16; ++col) {
                    int gcol = g * 16 + col;
                    double refF = 0, mineF = 0;
                    for (int l = 0; l < nb; ++l) {
                        const block_q2_K *xo = &orig[(size_t)gcol * nb + l];
                        const block_q8_K *a = &act[(size_t)r * nb + l];
                        int64_t ri, rs;
                        ref_block(xo, a, DQ_NORMAL, ri, rs);
                        const block_q2_Kx16 *b = &vx[(size_t)g * nb + l];
                        auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                        auto bsumFn = [&](int sub) { return (int)a->bsums[sub]; };
                        int64_t mi, ms;
                        mine_col(b, col, actFn, bsumFn, mi, ms, maxPartial);
                        if (ri != mi) mismatch++;
                        if (rs != ms) mismatch++;
                        refF  += (double)xo->d * (double)a->d * (double)ri
                               - (double)xo->dmin * (double)a->d * (double)rs;
                        mineF += (double)b->d[col] * (double)a->d * (double)mi
                               - (double)b->dmin[col] * (double)a->d * (double)ms;
                    }
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
    // block_q8_Kx4 (qs at +16, byte for flat pos p / column c = p*4 + c; bsums at
    // +1040, group16-major/column-minor index g16*4 + c) and confirm the emitter-
    // model's interleaved read is BYTE-EXACT to the per-row plain dot. Exercises the
    // ONE GEMM-specific piece the GEVM lacks: the x4 interleave (quant + bsums).
    {
        const int nc = 32, n = 2560;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_q2_K> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_q2K_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);
        std::vector<block_q2_Kx16> vx((size_t)ng * nb);
        std::vector<block_q2_K> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_q2_Kx16(tmp16.data());
            }
        std::vector<block_q8_K> row(4 * nb);
        for (int r = 0; r < 4; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8K_block(&row[(size_t)r * nb + l], rng, 900 + r * 7 + l);
        // interleaved qs: for block l, flat pos p, column c -> qsx4[l][p*4 + c].
        std::vector<int8_t> qsx4((size_t)nb * QK_K * 4);
        // interleaved bsums: group16 g16, column c -> bsx4[l][g16*4 + c].
        std::vector<int16_t> bsx4((size_t)nb * (QK_K / 16) * 4);
        for (int l = 0; l < nb; ++l) {
            for (int p = 0; p < QK_K; ++p)
                for (int c = 0; c < 4; ++c)
                    qsx4[(size_t)l * QK_K * 4 + p * 4 + c] =
                        row[(size_t)c * nb + l].qs[p];
            for (int g16 = 0; g16 < QK_K / 16; ++g16)
                for (int c = 0; c < 4; ++c)
                    bsx4[(size_t)l * (QK_K / 16) * 4 + g16 * 4 + c] =
                        row[(size_t)c * nb + l].bsums[g16];
        }
        long long mism = 0, total = 0;
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col) {
                int gcol = g * 16 + col;
                for (int c = 0; c < 4; ++c)   // activation column within the x4 group
                    for (int l = 0; l < nb; ++l) {
                        int64_t ri, rs;
                        ref_block(&orig[(size_t)gcol * nb + l],
                                  &row[(size_t)c * nb + l], DQ_NORMAL, ri, rs);
                        const int8_t *qsl = &qsx4[(size_t)l * QK_K * 4];
                        const int16_t *bsl = &bsx4[(size_t)l * (QK_K / 16) * 4];
                        auto actFn = [&](int pos) { return (int)qsl[pos * 4 + c]; };
                        auto bsumFn = [&](int sub) { return (int)bsl[sub * 4 + c]; };
                        int dummy = 0;
                        int64_t mi, ms;
                        mine_col(&vx[(size_t)g * nb + l], col, actFn, bsumFn, mi, ms,
                                 dummy);
                        total += 2;
                        if (ri != mi) mism++;
                        if (rs != ms) mism++;
                    }
            }
        printf("\n# GEMM interleaved block_q8_Kx4 (qs @+16 pos*4+c, bsums @+1040 "
               "g16*4+c): int-mismatch = %lld / %lld  %s\n",
               mism, total, mism ? "INT-BUG" : "BYTE-EXACT");
        if (mism) anyBug = true;
    }

    // ---- NEGATIVE CONTROLS on one shape (nr=4, nc=32, n=2560) ----
    printf("\n# negative controls (perturb the REFERENCE; a large mismatch fraction\n"
           "# proves the q2_K feature is genuinely exercised by the emitter-model):\n");
    {
        const int nc = 32, n = 2560, nr = 4;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_q2_K> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_q2K_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);
        std::vector<block_q2_Kx16> vx((size_t)ng * nb);
        std::vector<block_q2_K> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_q2_Kx16(tmp16.data());
            }
        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        const char *names[3] = {"2BIT     (mask weight to low bit)  [isum]",
                                "SCALE-ROT(rotate 4-bit scale +1)   [isum]",
                                "MIN-ROT  (rotate 4-bit min +1)     [summs]"};
        DqMode modes[3] = {DQ_2BIT, DQ_SCALEROT, DQ_MINROT};
        for (int m = 0; m < 3; ++m) {
            long long total = 0, mism = 0;
            for (int r = 0; r < nr; ++r)
                for (int g = 0; g < ng; ++g)
                    for (int col = 0; col < 16; ++col) {
                        int gcol = g * 16 + col;
                        for (int l = 0; l < nb; ++l) {
                            const block_q2_K *xo = &orig[(size_t)gcol * nb + l];
                            const block_q8_K *a = &act[(size_t)r * nb + l];
                            int dummy = 0;
                            int64_t rI, rS, mI, mS;
                            ref_block(xo, a, modes[m], rI, rS);
                            auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                            auto bsumFn = [&](int sub) { return (int)a->bsums[sub]; };
                            mine_col(&vx[(size_t)g * nb + l], col, actFn, bsumFn, mI,
                                     mS, dummy);
                            total++;
                            // 2BIT/SCALE perturb isum; MIN perturbs summs.
                            bool differ =
                                (modes[m] == DQ_MINROT) ? (rS != mS) : (rI != mI);
                            if (differ) mism++;
                        }
                    }
            printf("  %-42s : perturbed-ref vs emitter mismatch = %lld / %lld "
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
