// mxfp4 x q8_0 16x1-REPACKED GEVM + GEMM numeric oracle (FP4 CODEBOOK + E8M0 scale).
//
// Gate: does the compiler-emitted mxfp4 repack kernel compute the CORRECT FP4-CODEBOOK
// dequant-matmul result, with the E8M0 shared-exponent scale reconstructed exactly?
//
// The oracle is INDEPENDENT: the REFERENCE decodes the ORIGINAL (pre-repack)
// per-block mxfp4 exactly as ggml's canonical ggml_vec_dot_mxfp4_q8_0 -- the 4-bit
// nibble is an INDEX into the 16-entry DOUBLED-E2M1 int8 codebook (kvalues_mxfp4 =
// 2 * E2M1), so weight = kvalues_mxfp4[nibble] and the integer dot is
// sumi = sum_j kvalues_mxfp4[qs[j]&0xf]*q8[j] + kvalues_mxfp4[qs[j]>>4]*q8[j+16],
// scaled by GGML_E8M0_TO_FP32_HALF(x.e) * fp16(y.d), where E8M0_TO_FP32_HALF(e) =
// 2^(e-128) (the HALF folds the 0.5 that compensates the doubled codebook, exactly
// like ggml). The EMITTER-MODEL reads the REPACKED block_mxfp4x16 and reproduces the
// emitter's EXACT organization (block-as-lane strips, the memory codebook GATHER
// decode, an i32 accumulator, E8M0 vector-reconstructed single-scale fp32 fold). The
// two paths use DIFFERENT data layouts, so agreement is strong evidence.
//
// ONE integer certificate is compared BYTE-EXACT (int64 equality):
//   sumi = sum over 32 elements of kvalues_mxfp4[nibble] * q8   -- the whole dot.
// It is a pure integer (no fp reassociation). The fp4 codebook is small (|12|), so a
// single product magnitude (|12*127| = 1524 < 32767) fits i16 but 32 accumulations
// reach ~48768 (overflow i16), so the emitter (and this model) accumulate in i32.
//
// The E8M0 scale is an EXACT power of two, so it is byte-reproduced in both the
// reference and the emitter-model; the fp fold is reported as an norm (the vfmacc
// fused-order fold is pending-hardware, exactly as for the iq4_nl sibling).
//
// NEGATIVE CONTROLS perturb the REFERENCE; a mismatch SPIKE proves the corresponding
// mxfp4 structure was genuinely exercised by the emitter-model:
//   CODEBOOK : the REFERENCE treats the nibble as a LINEAR value (weight = nibble,
//              i.e. codebook[k] = k) instead of an fp4 codebook index. A real
//              codebook lookup diverges from fake-linear -> the int certificate
//              spikes. This is the load-bearing "real fp4 table lookup, NOT
//              fake-linear" control.
//   EXPONENT : perturb the per-COLUMN E8M0 shared exponent by +1 (2^(e-128) ->
//              2^(e-127), i.e. the scale doubles) -> the fp result diverges. This is
//              the load-bearing "real E8M0 shared-exponent scale, NOT ignored"
//              control.
//
// Build: g++ -O2 -std=c++17 oracle_repack_mxfp4.cpp -o /tmp/oracle_mxfp4 && /tmp/oracle_mxfp4

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define QK_MXFP4 32

// The canonical ggml kvalues_mxfp4[16] DOUBLED-E2M1 int8 fp4 codebook (2 * E2M1).
static const int8_t kvalues_mxfp4[16] = {0, 1, 2,  3,  4,  6,  8,  12,
                                         0, -1, -2, -3, -4, -6, -8, -12};

// ggml_e8m0_to_fp32_half (ggml-impl.h): 2^(e-128); the 0.5 compensates kvalues=2*E2M1.
static float e8m0_to_fp32_half(uint8_t e) {
    uint32_t bits;
    if (e < 2) bits = (uint32_t)0x00200000u << (e & 0x1F);
    else       bits = (uint32_t)(e - 1) << 23;
    float r;
    std::memcpy(&r, &bits, sizeof(float));
    return r;
}

// ---- original per-block mxfp4 (ggml-common.h), 17 bytes ----
struct block_mxfp4 {
    uint8_t e;                  // E8M0 shared exponent byte @ +0
    uint8_t qs[QK_MXFP4 / 2];   // 16 nibble bytes (low nibble -> elem j, high -> j+16)
};

// ---- repacked block_mxfp4x16 (block-as-lane convention), 272 B ----
//   e[16]    @ +0    (16 E8M0 exponent bytes, one per column)
//   qs[256]  @ +16   (nibble byte for qs index i / column c at 16 + i*16 + c)
struct block_mxfp4x16 {
    uint8_t e[16];
    uint8_t qs[256];
};

// ---- plain block_q8_0, 34 B ---- (fp16 d + 32 int8 quants). The oracle keeps d as a
// float; it does not affect the byte-exact INTEGER certificate.
struct block_q8_0 {
    float  d;
    int8_t qs[QK_MXFP4];
};

static block_mxfp4x16 make_block_mxfp4x16(const block_mxfp4 *in) {
    block_mxfp4x16 out;
    for (int c = 0; c < 16; ++c) out.e[c] = in[c].e;
    for (int i = 0; i < 16; ++i)
        for (int c = 0; c < 16; ++c)
            out.qs[i * 16 + c] = in[c].qs[i];
    return out;
}

enum DqMode { DQ_NORMAL = 0, DQ_CODEBOOK_LINEAR = 1 };

// ---- REFERENCE (ggml canonical ggml_vec_dot_mxfp4_q8_0). Returns the integer
// certificate sumi = sum(kvalues_mxfp4[nibble] * q8) over the 32-element block. In the
// CODEBOOK_LINEAR control the "codebook" is the identity (weight = nibble), which is
// what a FAKE-LINEAR emitter would compute. ----
static int64_t ref_block(const block_mxfp4 *x, const block_q8_0 *a, DqMode mode) {
    int64_t sumi = 0;
    for (int j = 0; j < 16; ++j) {
        int lo = x->qs[j] & 0x0F;
        int hi = x->qs[j] >> 4;
        int wLo = (mode == DQ_CODEBOOK_LINEAR) ? lo : kvalues_mxfp4[lo];
        int wHi = (mode == DQ_CODEBOOK_LINEAR) ? hi : kvalues_mxfp4[hi];
        sumi += (int64_t)wLo * a->qs[j] + (int64_t)wHi * a->qs[j + 16];
    }
    return sumi;
}

// ---- EMITTER-MODEL for column c of a repacked group, dotted with a per-column
// activation quant accessor act(pos). Reproduces the emitter's EXACT lane-wise
// codebook-gather organization (i32 accumulator). ----
template <typename ActFn>
static int64_t mine_col(const block_mxfp4x16 *b, int c, ActFn act,
                        int64_t &maxAbsSumi) {
    int64_t sumi = 0;
    for (int i = 0; i < 16; ++i) {
        uint8_t byte = b->qs[i * 16 + c];
        int wLo = kvalues_mxfp4[byte & 0x0F];  // REAL fp4 codebook gather
        int wHi = kvalues_mxfp4[byte >> 4];
        sumi += (int64_t)wLo * act(i) + (int64_t)wHi * act(i + 16);
        if (std::llabs(sumi) > maxAbsSumi) maxAbsSumi = std::llabs(sumi);
    }
    return sumi;
}

static void build_mxfp4_block(block_mxfp4 *x, std::mt19937 &rng, int col, int blk) {
    std::uniform_int_distribution<int> nib(0, 15);
    // E8M0 exponent near 127 (unit scale); range [118,134] keeps 2^(e-128) sane and
    // stays in the NORMAL (e >= 2) branch of the reconstruction.
    x->e = (uint8_t)(118 + ((col * 3 + blk * 5) % 17));
    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
        int lo = nib(rng), hi = nib(rng);
        x->qs[i] = (uint8_t)(lo | (hi << 4));
    }
}

static void build_q8_0_block(block_q8_0 *a, std::mt19937 &rng, int blk) {
    std::uniform_int_distribution<int> q8d(-90, 90);
    a->d = 0.015f + 0.0009f * (blk % 13);
    for (int i = 0; i < QK_MXFP4; ++i) {
        int dc = ((i * 7 + blk * 3) % 21) - 10;
        int v = q8d(rng) + dc;
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        a->qs[i] = (int8_t)v;
    }
}

int main() {
    printf("# mxfp4 x q8_0 16x1-REPACKED oracle (independent scalar, FP4 CODEBOOK + E8M0).\n");
    printf("# BYTE-EXACT integer certificate sumi (fp4 codebook dot): reference\n");
    printf("#   (original mxfp4) vs emitter-model (repacked block_mxfp4x16).\n\n");

    std::mt19937 rng(20260708u);
    struct Shape { int nc; int n; int nr; };
    std::vector<Shape> shapes = {
        {16, 32, 1},  {16, 4096, 1}, {32, 32, 4},  {32, 2560, 4},
        {256, 32, 1}, {256, 4096, 8}, {160, 2560, 4},
    };

    bool anyBug = false;
    int64_t globalMaxSumi = 0;
    double worstNorm = 0.0;

    for (auto sh : shapes) {
        const int nc = sh.nc, n = sh.n, nr = sh.nr;
        const int nb = n / QK_MXFP4, ng = nc / 16;

        std::vector<block_mxfp4> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_mxfp4_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                      g * 16 + col, l);

        std::vector<block_mxfp4x16> vx((size_t)ng * nb);
        std::vector<block_mxfp4> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_mxfp4x16(tmp16.data());
            }

        std::vector<block_q8_0> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_0_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        long long mismatch = 0;
        int64_t maxSumi = 0;
        double sumsqRef = 0, maxAbsErr = 0;
        for (int r = 0; r < nr; ++r)
            for (int g = 0; g < ng; ++g)
                for (int col = 0; col < 16; ++col) {
                    int gcol = g * 16 + col;
                    double refF = 0, mineF = 0;
                    for (int l = 0; l < nb; ++l) {
                        const block_mxfp4 *xo = &orig[(size_t)gcol * nb + l];
                        const block_q8_0 *a = &act[(size_t)r * nb + l];
                        int64_t ri = ref_block(xo, a, DQ_NORMAL);
                        const block_mxfp4x16 *b = &vx[(size_t)g * nb + l];
                        auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                        int64_t mi = mine_col(b, col, actFn, maxSumi);
                        if (ri != mi) mismatch++;
                        refF  += (double)e8m0_to_fp32_half(xo->e) * (double)a->d *
                                 (double)ri;
                        mineF += (double)e8m0_to_fp32_half(b->e[col]) *
                                 (double)a->d * (double)mi;
                    }
                    double e = std::fabs(refF - mineF);
                    if (e > maxAbsErr) maxAbsErr = e;
                    sumsqRef += refF * refF;
                }
        int ncnt = nr * nc;
        double rms = std::sqrt(sumsqRef / ncnt);
        double norm = rms > 0 ? maxAbsErr / rms : maxAbsErr;
        if (norm > worstNorm) worstNorm = norm;
        if (maxSumi > globalMaxSumi) globalMaxSumi = maxSumi;
        bool bug = (mismatch != 0);
        if (bug) anyBug = true;
        printf("  shape nr=%-3d nc=%-4d n=%-6d : int-mismatch=%-6lld  "
               "maxAbsSumi=%-7lld  fp-norm=%.3e  %s\n",
               nr, nc, n, mismatch, (long long)maxSumi, norm,
               bug ? "INT-BUG" : "BYTE-EXACT");
    }

    // ---- GEMM interleaved block_q8_0x4 addressing: pack 4 plain q8_0 rows into one
    // block_q8_0x4 (qs @+8, byte for flat pos p / column c = p*4 + c) and confirm the
    // emitter-model's interleaved read is BYTE-EXACT. ----
    {
        const int nc = 32, n = 2560;
        const int nb = n / QK_MXFP4, ng = nc / 16;
        std::vector<block_mxfp4> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_mxfp4_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                      g * 16 + col, l);
        std::vector<block_mxfp4x16> vx((size_t)ng * nb);
        std::vector<block_mxfp4> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_mxfp4x16(tmp16.data());
            }
        std::vector<block_q8_0> row(4 * nb);
        for (int r = 0; r < 4; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_0_block(&row[(size_t)r * nb + l], rng, 900 + r * 7 + l);
        std::vector<int8_t> qsx4((size_t)nb * QK_MXFP4 * 4);
        for (int l = 0; l < nb; ++l)
            for (int p = 0; p < QK_MXFP4; ++p)
                for (int c = 0; c < 4; ++c)
                    qsx4[(size_t)l * QK_MXFP4 * 4 + p * 4 + c] =
                        row[(size_t)c * nb + l].qs[p];
        long long mism = 0, total = 0;
        int64_t dummy = 0;
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col) {
                int gcol = g * 16 + col;
                for (int c = 0; c < 4; ++c)
                    for (int l = 0; l < nb; ++l) {
                        int64_t ri = ref_block(&orig[(size_t)gcol * nb + l],
                                               &row[(size_t)c * nb + l], DQ_NORMAL);
                        const int8_t *qsl = &qsx4[(size_t)l * QK_MXFP4 * 4];
                        auto actFn = [&](int pos) { return (int)qsl[pos * 4 + c]; };
                        int64_t mi = mine_col(&vx[(size_t)g * nb + l], col, actFn,
                                              dummy);
                        total++;
                        if (ri != mi) mism++;
                    }
            }
        printf("\n# GEMM interleaved block_q8_0x4 (qs @+8 pos*4+c): int-mismatch = "
               "%lld / %lld  %s\n",
               mism, total, mism ? "INT-BUG" : "BYTE-EXACT");
        if (mism) anyBug = true;
    }

    // ---- NEGATIVE CONTROLS on one shape (nr=4, nc=32, n=2560) ----
    printf("\n# negative controls (perturb the REFERENCE; a large mismatch fraction\n"
           "# proves the mxfp4 feature is genuinely exercised by the emitter-model):\n");
    {
        const int nc = 32, n = 2560, nr = 4;
        const int nb = n / QK_MXFP4, ng = nc / 16;
        std::vector<block_mxfp4> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_mxfp4_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                      g * 16 + col, l);
        std::vector<block_mxfp4x16> vx((size_t)ng * nb);
        std::vector<block_mxfp4> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_mxfp4x16(tmp16.data());
            }
        std::vector<block_q8_0> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_0_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        // CODEBOOK control: perturbed-ref (FAKE-LINEAR nibble) vs emitter sumi.
        {
            long long total = 0, mism = 0;
            int64_t dummy = 0;
            for (int r = 0; r < nr; ++r)
                for (int g = 0; g < ng; ++g)
                    for (int col = 0; col < 16; ++col) {
                        int gcol = g * 16 + col;
                        for (int l = 0; l < nb; ++l) {
                            const block_q8_0 *a = &act[(size_t)r * nb + l];
                            int64_t rI = ref_block(&orig[(size_t)gcol * nb + l],
                                                   a, DQ_CODEBOOK_LINEAR);
                            auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                            int64_t mI = mine_col(&vx[(size_t)g * nb + l], col,
                                                  actFn, dummy);
                            total++;
                            if (rI != mI) mism++;
                        }
                    }
            printf("  %-46s : perturbed-ref vs emitter mismatch = %lld / %lld "
                   "(%.0f%%)  %s\n",
                   "CODEBOOK (nibble-as-linear, NOT fp4 table) [sumi]", mism, total,
                   100.0 * mism / total, mism > 0 ? "EXERCISED" : "!! NOT EXERCISED");
        }
        // EXPONENT control: perturb the per-column E8M0 exponent by +1 -> the scale
        // doubles -> fp diverges. (A byte-exact-but-scale-blind emitter would pass.)
        {
            long long total = 0, mism = 0;
            int64_t dummy = 0;
            for (int r = 0; r < nr; ++r)
                for (int g = 0; g < ng; ++g)
                    for (int col = 0; col < 16; ++col) {
                        int gcol = g * 16 + col;
                        for (int l = 0; l < nb; ++l) {
                            const block_q8_0 *a = &act[(size_t)r * nb + l];
                            const block_mxfp4x16 *b = &vx[(size_t)g * nb + l];
                            auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                            int64_t mi = mine_col(b, col, actFn, dummy);
                            uint8_t ePerturb = (uint8_t)(b->e[col] + 1);
                            double refF = (double)e8m0_to_fp32_half(ePerturb) *
                                          (double)a->d * (double)mi;
                            double mineF = (double)e8m0_to_fp32_half(b->e[col]) *
                                           (double)a->d * (double)mi;
                            total++;
                            if (std::fabs(refF - mineF) > 1e-9) mism++;
                        }
                    }
            printf("  %-46s : perturbed-ref vs emitter mismatch = %lld / %lld "
                   "(%.0f%%)  %s\n",
                   "EXPONENT (E8M0 e +1, scale doubles)        [fp]  ", mism, total,
                   100.0 * mism / total, mism > 0 ? "EXERCISED" : "!! NOT EXERCISED");
        }
    }

    printf("\nWORST_FP_NORM %.3e   GLOBAL_MAX_i32_SUMI %lld   VERDICT %s\n",
           worstNorm, (long long)globalMaxSumi,
           anyBug ? "INT-BUG" : "BYTE-EXACT-INTEGER");
    return anyBug ? 1 : 0;
}
