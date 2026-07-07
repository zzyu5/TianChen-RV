// tq1_0 x q8_K 16x1-REPACKED GEVM + GEMM numeric oracle (BASE-3 TERNARY).
//
// Gate: does the compiler-emitted tq1_0 repack kernel compute the CORRECT ternary
// dequant-matmul result?
//
// The oracle is INDEPENDENT: the REFERENCE decodes the ORIGINAL (pre-repack)
// per-block tq1_0 exactly as ggml's canonical base-3 unpack (trit(byte,l) =
// ((uint8_t)(byte * pow3[l]) * 3 >> 8) - 1, in {-1,0,1}; the 48-byte qs plane packs
// 5 trits/byte, the 4-byte qh plane 4 trits/byte, with ggml's element order), while
// the EMITTER-MODEL reads the REPACKED block_tq1_0x16 and reproduces the emitter's
// EXACT lane-wise organization (block-as-lane strips, per-region i16 partials widened
// into ONE i32 accumulator, single-scale fp32 fold). DIFFERENT data layouts ->
// agreement is strong evidence.
//
// tq1_0 is the LOWEST-bit LINEAR ternary super-block (1.6 bits/weight): ONE fp16
// super-block scale, NO sub-block scale / dmin / bsums / min term. The base-3 decode
// is BYTE-generatable: for ANY byte in [0,255], (byte*pow3[l]*3)>>8 is in {0,1,2}, so
// random qs/qh bytes yield valid trits and both paths decode identically.
//
// ONE integer certificate is compared BYTE-EXACT (int64 equality):
//   isum = sum over 256 elements of (trit * q8).
// Both the tail-vs-main split and the qh plane feed one i32 accumulator (order-free),
// so this is a hard certificate. The per-region i16 partial magnitude is tracked to
// certify no overflow (160*127 = 20320 < 32767).
//
// NEGATIVE CONTROLS perturb the REFERENCE:
//   TRIT  : the reference decodes each digit l with pow3[(l+1) % nDigits] (a base-3
//           DIGIT ROTATION) while keeping the q8 position -> the trit at each
//           position is scrambled; isum (int certificate) diverges.
//   SCALE : rotate the per-COLUMN fp16 super-block scale by +1 column -> fp diverges.
//
// Build: g++ -O2 -std=c++17 oracle_repack_tq1_0.cpp -o /tmp/oracle_tq1 && /tmp/oracle_tq1

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define QK_K 256

static const int pow3[5] = {1, 3, 9, 27, 81};

// decode base-3 digit l of `byte` -> {0,1,2} (ggml's exact unpack).
static inline int b3(uint8_t byte, int l) {
    uint8_t q = (uint8_t)(byte * (uint8_t)pow3[l]);
    return ((uint16_t)q * 3) >> 8;
}

// ---- original per-block tq1_0 (ggml-common.h), 54 bytes ----
struct block_tq1_0 {
    uint8_t qs[48]; // +0   48 B: base-3 (5 trits/byte) low plane
    uint8_t qh[4];  // +48   4 B: base-3 (4 trits/byte) high plane
    float   d;      // super-block scale (fp16 in ggml; float here for the int test)
};

// ---- repacked block_tq1_0x16 (block-as-lane convention) ----
//   d[16]    @ +0    (fp16 in the real kernel; float here for the int test)
//   qs[768]  @ +...  (base-3 qs byte for index i / column c at i*16+c)
//   qh[64]   @ +...  (base-3 qh byte for index i / column c at i*16+c)
struct block_tq1_0x16 {
    float   d[16];
    uint8_t qs[768];
    uint8_t qh[64];
};

struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};

static block_tq1_0x16 make_block_tq1_0x16(const block_tq1_0 *in) {
    block_tq1_0x16 out;
    for (int c = 0; c < 16; ++c) out.d[c] = in[c].d;
    for (int i = 0; i < 48; ++i)
        for (int c = 0; c < 16; ++c)
            out.qs[i * 16 + c] = in[c].qs[i];
    for (int i = 0; i < 4; ++i)
        for (int c = 0; c < 16; ++c)
            out.qh[i * 16 + c] = in[c].qh[i];
    return out;
}

enum DqMode { DQ_NORMAL = 0, DQ_TRIT = 1 };

// ---- REFERENCE (ggml canonical base-3 unpack over the ORIGINAL layout). ----
static int64_t ref_block(const block_tq1_0 *x, const block_q8_K *a, DqMode mode) {
    const int8_t *q8 = a->qs;
    int64_t isum = 0;
    // main qs: bytes 0..31, digits 0..4, q8 pos = l*32 + m.
    for (int m = 0; m < 32; ++m)
        for (int l = 0; l < 5; ++l) {
            int ld = (mode == DQ_TRIT) ? (l + 1) % 5 : l;
            int trit = b3(x->qs[m], ld) - 1;
            isum += (int64_t)q8[l * 32 + m] * trit;
        }
    // tail qs: bytes 32..47, digits 0..4, q8 pos = 160 + l*16 + m.
    for (int m = 0; m < 16; ++m)
        for (int l = 0; l < 5; ++l) {
            int ld = (mode == DQ_TRIT) ? (l + 1) % 5 : l;
            int trit = b3(x->qs[32 + m], ld) - 1;
            isum += (int64_t)q8[160 + l * 16 + m] * trit;
        }
    // qh: bytes 0..3, digits 0..3, q8 pos = 240 + l*4 + j.
    for (int j = 0; j < 4; ++j)
        for (int l = 0; l < 4; ++l) {
            int ld = (mode == DQ_TRIT) ? (l + 1) % 4 : l;
            int trit = b3(x->qh[j], ld) - 1;
            isum += (int64_t)q8[240 + l * 4 + j] * trit;
        }
    return isum;
}

// ---- EMITTER-MODEL for column c of a repacked group. Reproduces the emitter's
// per-region i16 partial fold (main / tail / qh), each widened into i32. ----
template <typename ActFn>
static int64_t mine_col(const block_tq1_0x16 *b, int c, ActFn act,
                        int &maxAbsPartial) {
    int64_t sumi = 0;
    // main region.
    int partial = 0;
    for (int bi = 0; bi < 32; ++bi) {
        uint8_t byte = b->qs[bi * 16 + c];
        for (int l = 0; l < 5; ++l)
            partial += act(l * 32 + bi) * (b3(byte, l) - 1);
    }
    if (std::abs(partial) > maxAbsPartial) maxAbsPartial = std::abs(partial);
    sumi += partial;
    // tail region.
    partial = 0;
    for (int bi = 32; bi < 48; ++bi) {
        uint8_t byte = b->qs[bi * 16 + c];
        for (int l = 0; l < 5; ++l)
            partial += act(160 + l * 16 + (bi - 32)) * (b3(byte, l) - 1);
    }
    if (std::abs(partial) > maxAbsPartial) maxAbsPartial = std::abs(partial);
    sumi += partial;
    // qh region.
    partial = 0;
    for (int bi = 0; bi < 4; ++bi) {
        uint8_t byte = b->qh[bi * 16 + c];
        for (int l = 0; l < 4; ++l)
            partial += act(240 + l * 4 + bi) * (b3(byte, l) - 1);
    }
    if (std::abs(partial) > maxAbsPartial) maxAbsPartial = std::abs(partial);
    sumi += partial;
    return sumi;
}

static void build_tq1_block(block_tq1_0 *x, std::mt19937 &rng, int col, int blk) {
    std::uniform_int_distribution<int> byte(0, 255);
    x->d = 0.011f + 0.0007f * ((col + 3 * blk) % 11);
    for (int i = 0; i < 48; ++i) x->qs[i] = (uint8_t)byte(rng);
    for (int i = 0; i < 4; ++i) x->qh[i] = (uint8_t)byte(rng);
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
    printf("# tq1_0 x q8_K 16x1-REPACKED oracle (independent scalar, BASE-3 TERNARY).\n");
    printf("# BYTE-EXACT integer certificate isum (trit dot): reference (original\n");
    printf("#   tq1_0) vs emitter-model (repacked block_tq1_0x16).\n\n");

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

        std::vector<block_tq1_0> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_tq1_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);

        std::vector<block_tq1_0x16> vx((size_t)ng * nb);
        std::vector<block_tq1_0> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_tq1_0x16(tmp16.data());
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
                        const block_tq1_0 *xo = &orig[(size_t)gcol * nb + l];
                        const block_q8_K *a = &act[(size_t)r * nb + l];
                        int64_t ri = ref_block(xo, a, DQ_NORMAL);
                        const block_tq1_0x16 *b = &vx[(size_t)g * nb + l];
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

    // ---- GEMM interleaved-activation addressing (qs @+16 pos*4+c). ----
    {
        const int nc = 32, n = 2560;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_tq1_0> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_tq1_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);
        std::vector<block_tq1_0x16> vx((size_t)ng * nb);
        std::vector<block_tq1_0> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_tq1_0x16(tmp16.data());
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

    // ---- NEGATIVE CONTROLS (nr=4, nc=32, n=2560) ----
    printf("\n# negative controls (perturb the REFERENCE; a large mismatch fraction\n"
           "# proves the tq1_0 feature is genuinely exercised by the emitter-model):\n");
    {
        const int nc = 32, n = 2560, nr = 4;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_tq1_0> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_tq1_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);
        std::vector<block_tq1_0x16> vx((size_t)ng * nb);
        std::vector<block_tq1_0> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_tq1_0x16(tmp16.data());
            }
        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        {   // TRIT: base-3 digit rotation in the reference -> isum diverges.
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
                   "TRIT     (base-3 digit rotation)   [isum]", mism, total,
                   100.0 * mism / total, mism > 0 ? "EXERCISED" : "!! NOT EXERCISED");
        }
        {   // SCALE: rotate the per-column fp16 scale by +1 -> fp result diverges.
            long long total = 0, mism = 0;
            int dummy = 0;
            for (int r = 0; r < nr; ++r)
                for (int g = 0; g < ng; ++g)
                    for (int col = 0; col < 16; ++col) {
                        int gcol = g * 16 + col;
                        for (int l = 0; l < nb; ++l) {
                            const block_q8_K *a = &act[(size_t)r * nb + l];
                            const block_tq1_0x16 *b = &vx[(size_t)g * nb + l];
                            auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                            int64_t mi = mine_col(b, col, actFn, dummy);
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
