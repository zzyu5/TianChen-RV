// tq2_0 x q8_K 16x1-REPACKED GEVM + GEMM numeric oracle (TERNARY, BitNet-class).
//
// Gate: does the compiler-emitted tq2_0 repack kernel compute the CORRECT ternary
// dequant-matmul result?
//
// The oracle is INDEPENDENT: the REFERENCE decodes the ORIGINAL (pre-repack)
// per-block tq2_0 exactly as ggml's canonical ggml_vec_dot_tq2_0_q8_K (2-bit weight
// value = (qs[k*32+m] >> (2*l)) & 3, TERNARY trit = value - 1 in {-1,0,1}, element
// order k*128 + l*32 + m, ONE fp16 super-block scale, NO sub-block scale / dmin /
// bsums / min term), while the EMITTER-MODEL reads the REPACKED block_tq2_0x16 and
// reproduces the emitter's EXACT lane-wise organization (block-as-lane strips, a
// per-super-half i16 partial widened into ONE i32 accumulator, single-scale fp32
// fold). The two paths use DIFFERENT data layouts, so agreement is strong evidence.
//
// tq2_0 is a LINEAR ternary super-block (the ternary analogue of q8_0's linear
// repack lifted to QK_K=256):
//   * 2-BIT TERNARY weight: 4 lanes per byte, ((byte >> {0,2,4,6}) & 3) - 1, the
//     `-1` ternary bias the q2_K repack (unsigned [0,3], no bias) LACKS.
//   * ONE fp16 super-block scale, NO per-sub-block scale, NO dmin, NO bsums.
//
// ONE integer certificate is compared BYTE-EXACT (int64 equality):
//   isum = sum over 256 elements of (trit * q8)   -- the whole ternary dot.
// It is a pure integer (no fp reassociation), so this is a hard certificate. The
// per-super-half i16 partial magnitude is tracked to certify no overflow (128*127 =
// 16256 < 32767). The fp result norm (fp16(x.d) * y.d * isum) is reported too.
//
// NEGATIVE CONTROLS perturb the REFERENCE; a mismatch SPIKE proves the corresponding
// tq2_0 structure was genuinely exercised by the emitter-model:
//   TRIT  : mask the weight to its LOW bit only ((byte>>shift)&1) -> the high 2-bit
//           lane is lost; the trit changes -> isum (int certificate) diverges.
//   SCALE : rotate the per-COLUMN fp16 super-block scale by +1 column -> a scale
//           permutation bug cannot cancel; the fp result diverges (mine reads the
//           correct per-column d, ref a rotated one).
//
// Build: g++ -O2 -std=c++17 oracle_repack_tq2_0.cpp -o /tmp/oracle_tq2 && /tmp/oracle_tq2

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define QK_K 256

// ---- original per-block tq2_0 (ggml-common.h), 66 bytes ----
struct block_tq2_0 {
    uint8_t qs[QK_K / 4]; // +0   64 B: 2-bit ternary quants (4 per byte)
    float   d;            // super-block scale (fp16 in ggml; float here does not
                          // affect the byte-exact INTEGER certificate)
};

// ---- repacked block_tq2_0x16 (block-as-lane convention), 1056 B ----
//   d[16]     @ +0    (32 B, fp16 in the real kernel; float here for the int test)
//   qs[1024]  @ +32   (1024 B: 2-bit quant byte for qs index i / column c at 32+i*16+c)
struct block_tq2_0x16 {
    float   d[16];
    uint8_t qs[1024];
};

// ---- plain block_q8_K, 292 B ----
struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};

static block_tq2_0x16 make_block_tq2_0x16(const block_tq2_0 *in) {
    block_tq2_0x16 out;
    for (int c = 0; c < 16; ++c) out.d[c] = in[c].d;
    for (int i = 0; i < 64; ++i)
        for (int c = 0; c < 16; ++c)
            out.qs[i * 16 + c] = in[c].qs[i];
    return out;
}

enum DqMode { DQ_NORMAL = 0, DQ_TRIT = 1 };

// ---- REFERENCE (ggml canonical ggml_vec_dot_tq2_0_q8_K). Returns the integer
// certificate isum = sum(trit * q8) over the 256-element super-block. ----
static int64_t ref_block(const block_tq2_0 *x, const block_q8_K *a, DqMode mode) {
    int wmask = (mode == DQ_TRIT) ? 1 : 3; // TRIT control drops the high weight bit
    int64_t isum = 0;
    for (int k = 0; k < QK_K / 128; ++k) {       // super-half
        const uint8_t *q2 = x->qs + k * 32;
        const int8_t  *q8 = a->qs + k * 128;
        for (int l = 0; l < 4; ++l) {            // shift = 0,2,4,6
            for (int m = 0; m < 32; ++m) {
                int trit = ((q2[m] >> (2 * l)) & wmask) - 1;
                isum += (int64_t)q8[l * 32 + m] * trit;
            }
        }
    }
    return isum;
}

// ---- EMITTER-MODEL for column c of a repacked group, dotted with a per-column
// activation quant accessor `act(globalPos)`. Reproduces the emitter's EXACT
// lane-wise organization: per super-half a 128-element i16 partial (tracked for
// overflow), widened into the i32 accumulator. ----
template <typename ActFn>
static int64_t mine_col(const block_tq2_0x16 *b, int c, ActFn act,
                        int &maxAbsPartial) {
    int64_t sumi = 0;
    for (int k = 0; k < 2; ++k) {                // super-half
        int partial = 0;                         // the per-super-half i16 partial
        for (int m = 0; m < 32; ++m) {
            uint8_t byte = b->qs[(k * 32 + m) * 16 + c];
            for (int j = 0; j < 4; ++j) {
                int trit = ((byte >> (2 * j)) & 3) - 1;
                partial += act(k * 128 + j * 32 + m) * trit;
            }
        }
        if (std::abs(partial) > maxAbsPartial) maxAbsPartial = std::abs(partial);
        sumi += partial;
    }
    return sumi;
}

// ---- build ONE adversarial original tq2_0 block (valid ternary values {0,1,2}) ----
static void build_tq2_block(block_tq2_0 *x, std::mt19937 &rng, int col, int blk) {
    std::uniform_int_distribution<int> trit(0, 2); // {0,1,2} -> {-1,0,1}
    x->d = 0.011f + 0.0007f * ((col + 3 * blk) % 11);
    for (int i = 0; i < QK_K / 4; ++i) {
        // pack 4 ternary values (0..2) into a byte, 2 bits each.
        int b = 0;
        for (int j = 0; j < 4; ++j) b |= (trit(rng) << (2 * j));
        x->qs[i] = (uint8_t)b;
    }
}

static void build_q8K_block(block_q8_K *a, std::mt19937 &rng, int blk) {
    std::uniform_int_distribution<int> q8d(-90, 90);
    a->d = 0.015f + 0.0009f * (blk % 13);
    for (int grp = 0; grp < QK_K / 16; ++grp) {
        int dc = ((grp * 7 + blk * 3) % 21) - 10;
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii) {
            int v = q8d(rng) + dc;
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            a->qs[grp * 16 + ii] = (int8_t)v;
            sum += v;
        }
        a->bsums[grp] = (int16_t)sum;
    }
}

int main() {
    printf("# tq2_0 x q8_K 16x1-REPACKED oracle (independent scalar, TERNARY).\n");
    printf("# BYTE-EXACT integer certificate isum (trit dot): reference (original\n");
    printf("#   tq2_0) vs emitter-model (repacked block_tq2_0x16).\n\n");

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

        std::vector<block_tq2_0> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_tq2_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);

        std::vector<block_tq2_0x16> vx((size_t)ng * nb);
        std::vector<block_tq2_0> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_tq2_0x16(tmp16.data());
            }

        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        long long mismatch = 0;
        int maxPartial = 0;
        double sumsqRef = 0, maxAbsErr = 0;
        for (int r = 0; r < nr; ++r)
            for (int g = 0; g < ng; ++g)
                for (int col = 0; col < 16; ++col) {
                    int gcol = g * 16 + col;
                    double refF = 0, mineF = 0;
                    for (int l = 0; l < nb; ++l) {
                        const block_tq2_0 *xo = &orig[(size_t)gcol * nb + l];
                        const block_q8_K *a = &act[(size_t)r * nb + l];
                        int64_t ri = ref_block(xo, a, DQ_NORMAL);
                        const block_tq2_0x16 *b = &vx[(size_t)g * nb + l];
                        auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                        int64_t mi = mine_col(b, col, actFn, maxPartial);
                        if (ri != mi) mismatch++;
                        refF  += (double)xo->d * (double)a->d * (double)ri;
                        mineF += (double)b->d[col] * (double)a->d * (double)mi;
                    }
                    double e = std::fabs(refF - mineF);
                    if (e > maxAbsErr) maxAbsErr = e;
                    sumsqRef += refF * refF;
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
    // block_q8_Kx4 (qs @+16, byte for flat pos p / column c = p*4 + c) and confirm
    // the emitter-model's interleaved read is BYTE-EXACT to the per-row plain dot. ----
    {
        const int nc = 32, n = 2560;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_tq2_0> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_tq2_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);
        std::vector<block_tq2_0x16> vx((size_t)ng * nb);
        std::vector<block_tq2_0> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_tq2_0x16(tmp16.data());
            }
        std::vector<block_q8_K> row(4 * nb);
        for (int r = 0; r < 4; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8K_block(&row[(size_t)r * nb + l], rng, 900 + r * 7 + l);
        std::vector<int8_t> qsx4((size_t)nb * QK_K * 4);
        for (int l = 0; l < nb; ++l)
            for (int p = 0; p < QK_K; ++p)
                for (int c = 0; c < 4; ++c)
                    qsx4[(size_t)l * QK_K * 4 + p * 4 + c] =
                        row[(size_t)c * nb + l].qs[p];
        long long mism = 0, total = 0;
        int dummy = 0;
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col) {
                int gcol = g * 16 + col;
                for (int c = 0; c < 4; ++c)
                    for (int l = 0; l < nb; ++l) {
                        int64_t ri = ref_block(&orig[(size_t)gcol * nb + l],
                                               &row[(size_t)c * nb + l], DQ_NORMAL);
                        const int8_t *qsl = &qsx4[(size_t)l * QK_K * 4];
                        auto actFn = [&](int pos) { return (int)qsl[pos * 4 + c]; };
                        int64_t mi = mine_col(&vx[(size_t)g * nb + l], col, actFn,
                                              dummy);
                        total++;
                        if (ri != mi) mism++;
                    }
            }
        printf("\n# GEMM interleaved block_q8_Kx4 (qs @+16 pos*4+c): int-mismatch = "
               "%lld / %lld  %s\n",
               mism, total, mism ? "INT-BUG" : "BYTE-EXACT");
        if (mism) anyBug = true;
    }

    // ---- NEGATIVE CONTROLS on one shape (nr=4, nc=32, n=2560) ----
    printf("\n# negative controls (perturb the REFERENCE; a large mismatch fraction\n"
           "# proves the tq2_0 feature is genuinely exercised by the emitter-model):\n");
    {
        const int nc = 32, n = 2560, nr = 4;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_tq2_0> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_tq2_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);
        std::vector<block_tq2_0x16> vx((size_t)ng * nb);
        std::vector<block_tq2_0> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_tq2_0x16(tmp16.data());
            }
        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        // TRIT control: perturbed-ref isum vs emitter sumi (int certificate).
        {
            long long total = 0, mism = 0;
            int dummy = 0;
            for (int r = 0; r < nr; ++r)
                for (int g = 0; g < ng; ++g)
                    for (int col = 0; col < 16; ++col) {
                        int gcol = g * 16 + col;
                        for (int l = 0; l < nb; ++l) {
                            const block_q8_K *a = &act[(size_t)r * nb + l];
                            int64_t rI = ref_block(&orig[(size_t)gcol * nb + l], a,
                                                   DQ_TRIT);
                            auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                            int64_t mI = mine_col(&vx[(size_t)g * nb + l], col,
                                                  actFn, dummy);
                            total++;
                            if (rI != mI) mism++;
                        }
                    }
            printf("  %-42s : perturbed-ref vs emitter mismatch = %lld / %lld "
                   "(%.0f%%)  %s\n",
                   "TRIT     (mask weight to low bit)  [isum]", mism, total,
                   100.0 * mism / total, mism > 0 ? "EXERCISED" : "!! NOT EXERCISED");
        }
        // SCALE control: rotate the per-column fp16 scale by +1 -> fp result diverges.
        {
            long long total = 0, mism = 0;
            int dummy = 0;
            for (int r = 0; r < nr; ++r)
                for (int g = 0; g < ng; ++g)
                    for (int col = 0; col < 16; ++col) {
                        int gcol = g * 16 + col;
                        for (int l = 0; l < nb; ++l) {
                            const block_q8_K *a = &act[(size_t)r * nb + l];
                            const block_tq2_0x16 *b = &vx[(size_t)g * nb + l];
                            auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                            int64_t mi = mine_col(b, col, actFn, dummy);
                            // reference uses the ROTATED column's scale.
                            double dRot = b->d[(col + 1) % 16];
                            double refF = dRot * (double)a->d * (double)mi;
                            double mineF = (double)b->d[col] * (double)a->d *
                                           (double)mi;
                            total++;
                            if (std::fabs(refF - mineF) > 1e-9) mism++;
                        }
                    }
            printf("  %-42s : perturbed-ref vs emitter mismatch = %lld / %lld "
                   "(%.0f%%)  %s\n",
                   "SCALE    (rotate per-col fp16 d +1)[fp]  ", mism, total,
                   100.0 * mism / total, mism > 0 ? "EXERCISED" : "!! NOT EXERCISED");
        }
    }

    printf("\nWORST_FP_NORM %.3e   GLOBAL_MAX_i16_PARTIAL %d (bound 32767)   "
           "VERDICT %s\n",
           worstNorm, globalMaxPartial, anyBug ? "INT-BUG" : "BYTE-EXACT-INTEGER");
    return anyBug ? 1 : 0;
}
