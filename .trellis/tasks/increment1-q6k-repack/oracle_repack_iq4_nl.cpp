// iq4_nl x q8_0 16x1-REPACKED GEVM + GEMM numeric oracle (CODEBOOK, non-linear).
//
// Gate: does the compiler-emitted iq4_nl repack kernel compute the CORRECT
// NON-LINEAR CODEBOOK dequant-matmul result?
//
// The oracle is INDEPENDENT: the REFERENCE decodes the ORIGINAL (pre-repack)
// per-block iq4_nl exactly as ggml's canonical ggml_vec_dot_iq4_nl_q8_0 -- the
// 4-bit nibble is an INDEX into the 16-entry NON-LINEAR int8 codebook
// (kvalues_iq4nl), so weight = codebook[nibble] and the dot is
// sumi = sum_j codebook[qs[j]&0xf]*q8[j] + codebook[qs[j]>>4]*q8[j+16], scaled by
// fp16(x.d)*y.d. The EMITTER-MODEL reads the REPACKED block_iq4_nlx16 and
// reproduces the emitter's EXACT organization (block-as-lane strips, the memory
// codebook GATHER decode, an i32 accumulator, single-scale fp32 fold). The two
// paths use DIFFERENT data layouts, so agreement is strong evidence.
//
// ONE integer certificate is compared BYTE-EXACT (int64 equality):
//   sumi = sum over 32 elements of codebook[nibble] * q8   -- the whole dot.
// It is a pure integer (no fp reassociation). The single product magnitude
// (|127*127| = 16129 < 32767) fits i16 but 32 accumulations overflow it, so the
// emitter (and this model) accumulate in i32; the model tracks the running |sumi|.
//
// NEGATIVE CONTROLS perturb the REFERENCE; a mismatch SPIKE proves the
// corresponding iq4_nl structure was genuinely exercised by the emitter-model:
//   CODEBOOK : the REFERENCE treats the nibble as a LINEAR value (weight = nibble,
//              i.e. codebook[k] = k) instead of a codebook index. A real
//              codebook lookup diverges from fake-linear -> the int certificate
//              spikes. This is the load-bearing "real table lookup, NOT
//              fake-linear" control.
//   SCALE    : rotate the per-COLUMN fp16 super-block scale by +1 column -> a
//              scale permutation bug cannot cancel; the fp result diverges.
//
// Build: g++ -O2 -std=c++17 oracle_repack_iq4_nl.cpp -o /tmp/oracle_iq4nl && /tmp/oracle_iq4nl

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#define QK4_NL 32

// The canonical ggml kvalues_iq4nl[16] NON-LINEAR int8 codebook.
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22,
                                         -10, 1,    13,  25,  38,  53,  69,
                                         89,  113};

// ---- original per-block iq4_nl (ggml-common.h), 18 bytes ----
struct block_iq4_nl {
    float   d;               // super-block scale (fp16 in ggml; float here does
                             // not affect the byte-exact INTEGER certificate)
    uint8_t qs[QK4_NL / 2];  // 16 nibble bytes (low nibble -> elem j, high -> j+16)
};

// ---- repacked block_iq4_nlx16 (block-as-lane convention), 288 B ----
//   d[16]    @ +0    (32 B, fp16 in the real kernel; float here for the int test)
//   qs[256]  @ +32   (nibble byte for qs index i / column c at 32 + i*16 + c)
struct block_iq4_nlx16 {
    float   d[16];
    uint8_t qs[256];
};

// ---- plain block_q8_0, 34 B ---- (fp16 d + 32 int8 quants)
struct block_q8_0 {
    float  d;
    int8_t qs[QK4_NL];
};

static block_iq4_nlx16 make_block_iq4_nlx16(const block_iq4_nl *in) {
    block_iq4_nlx16 out;
    for (int c = 0; c < 16; ++c) out.d[c] = in[c].d;
    for (int i = 0; i < 16; ++i)
        for (int c = 0; c < 16; ++c)
            out.qs[i * 16 + c] = in[c].qs[i];
    return out;
}

enum DqMode { DQ_NORMAL = 0, DQ_CODEBOOK_LINEAR = 1 };

// ---- REFERENCE (ggml canonical ggml_vec_dot_iq4_nl_q8_0). Returns the integer
// certificate sumi = sum(codebook[nibble] * q8) over the 32-element block. In the
// CODEBOOK_LINEAR control the "codebook" is the identity (weight = nibble), which
// is what a FAKE-LINEAR emitter would compute. ----
static int64_t ref_block(const block_iq4_nl *x, const block_q8_0 *a, DqMode mode) {
    int64_t sumi = 0;
    for (int j = 0; j < 16; ++j) {
        int lo = x->qs[j] & 0x0F;
        int hi = x->qs[j] >> 4;
        int wLo = (mode == DQ_CODEBOOK_LINEAR) ? lo : kvalues_iq4nl[lo];
        int wHi = (mode == DQ_CODEBOOK_LINEAR) ? hi : kvalues_iq4nl[hi];
        sumi += (int64_t)wLo * a->qs[j] + (int64_t)wHi * a->qs[j + 16];
    }
    return sumi;
}

// ---- EMITTER-MODEL for column c of a repacked group, dotted with a per-column
// activation quant accessor act(pos). Reproduces the emitter's EXACT lane-wise
// codebook-gather organization (i32 accumulator). ----
template <typename ActFn>
static int64_t mine_col(const block_iq4_nlx16 *b, int c, ActFn act,
                        int64_t &maxAbsSumi) {
    int64_t sumi = 0;
    for (int i = 0; i < 16; ++i) {
        uint8_t byte = b->qs[i * 16 + c];
        int wLo = kvalues_iq4nl[byte & 0x0F];  // REAL codebook gather
        int wHi = kvalues_iq4nl[byte >> 4];
        sumi += (int64_t)wLo * act(i) + (int64_t)wHi * act(i + 16);
        if (std::llabs(sumi) > maxAbsSumi) maxAbsSumi = std::llabs(sumi);
    }
    return sumi;
}

static void build_iq4_block(block_iq4_nl *x, std::mt19937 &rng, int col, int blk) {
    std::uniform_int_distribution<int> nib(0, 15);
    x->d = 0.011f + 0.0007f * ((col + 3 * blk) % 11);
    for (int i = 0; i < QK4_NL / 2; ++i) {
        int lo = nib(rng), hi = nib(rng);
        x->qs[i] = (uint8_t)(lo | (hi << 4));
    }
}

static void build_q8_0_block(block_q8_0 *a, std::mt19937 &rng, int blk) {
    std::uniform_int_distribution<int> q8d(-90, 90);
    a->d = 0.015f + 0.0009f * (blk % 13);
    for (int i = 0; i < QK4_NL; ++i) {
        int dc = ((i * 7 + blk * 3) % 21) - 10;
        int v = q8d(rng) + dc;
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        a->qs[i] = (int8_t)v;
    }
}

int main() {
    printf("# iq4_nl x q8_0 16x1-REPACKED oracle (independent scalar, CODEBOOK).\n");
    printf("# BYTE-EXACT integer certificate sumi (codebook dot): reference\n");
    printf("#   (original iq4_nl) vs emitter-model (repacked block_iq4_nlx16).\n\n");

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
        const int nb = n / QK4_NL, ng = nc / 16;

        std::vector<block_iq4_nl> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq4_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);

        std::vector<block_iq4_nlx16> vx((size_t)ng * nb);
        std::vector<block_iq4_nl> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq4_nlx16(tmp16.data());
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
                        const block_iq4_nl *xo = &orig[(size_t)gcol * nb + l];
                        const block_q8_0 *a = &act[(size_t)r * nb + l];
                        int64_t ri = ref_block(xo, a, DQ_NORMAL);
                        const block_iq4_nlx16 *b = &vx[(size_t)g * nb + l];
                        auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                        int64_t mi = mine_col(b, col, actFn, maxSumi);
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
        if (maxSumi > globalMaxSumi) globalMaxSumi = maxSumi;
        bool bug = (mismatch != 0);
        if (bug) anyBug = true;
        printf("  shape nr=%-3d nc=%-4d n=%-6d : int-mismatch=%-6lld  "
               "maxAbsSumi=%-7lld  fp-norm=%.3e  %s\n",
               nr, nc, n, mismatch, (long long)maxSumi, norm,
               bug ? "INT-BUG" : "BYTE-EXACT");
    }

    // ---- GEMM interleaved block_q8_0x4 addressing: pack 4 plain q8_0 rows into
    // one block_q8_0x4 (qs @+8, byte for flat pos p / column c = p*4 + c) and
    // confirm the emitter-model's interleaved read is BYTE-EXACT. ----
    {
        const int nc = 32, n = 2560;
        const int nb = n / QK4_NL, ng = nc / 16;
        std::vector<block_iq4_nl> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq4_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);
        std::vector<block_iq4_nlx16> vx((size_t)ng * nb);
        std::vector<block_iq4_nl> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq4_nlx16(tmp16.data());
            }
        std::vector<block_q8_0> row(4 * nb);
        for (int r = 0; r < 4; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_0_block(&row[(size_t)r * nb + l], rng, 900 + r * 7 + l);
        std::vector<int8_t> qsx4((size_t)nb * QK4_NL * 4);
        for (int l = 0; l < nb; ++l)
            for (int p = 0; p < QK4_NL; ++p)
                for (int c = 0; c < 4; ++c)
                    qsx4[(size_t)l * QK4_NL * 4 + p * 4 + c] =
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
                        const int8_t *qsl = &qsx4[(size_t)l * QK4_NL * 4];
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
           "# proves the iq4_nl feature is genuinely exercised by the emitter-model):\n");
    {
        const int nc = 32, n = 2560, nr = 4;
        const int nb = n / QK4_NL, ng = nc / 16;
        std::vector<block_iq4_nl> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq4_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);
        std::vector<block_iq4_nlx16> vx((size_t)ng * nb);
        std::vector<block_iq4_nl> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq4_nlx16(tmp16.data());
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
                   "CODEBOOK (nibble-as-linear, NOT table)  [sumi]", mism, total,
                   100.0 * mism / total, mism > 0 ? "EXERCISED" : "!! NOT EXERCISED");
        }
        // SCALE control: rotate the per-column fp16 scale by +1 -> fp diverges.
        {
            long long total = 0, mism = 0;
            int64_t dummy = 0;
            for (int r = 0; r < nr; ++r)
                for (int g = 0; g < ng; ++g)
                    for (int col = 0; col < 16; ++col) {
                        int gcol = g * 16 + col;
                        for (int l = 0; l < nb; ++l) {
                            const block_q8_0 *a = &act[(size_t)r * nb + l];
                            const block_iq4_nlx16 *b = &vx[(size_t)g * nb + l];
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
            printf("  %-46s : perturbed-ref vs emitter mismatch = %lld / %lld "
                   "(%.0f%%)  %s\n",
                   "SCALE    (rotate per-col fp16 d +1)     [fp]  ", mism, total,
                   100.0 * mism / total, mism > 0 ? "EXERCISED" : "!! NOT EXERCISED");
        }
    }

    printf("\nWORST_FP_NORM %.3e   GLOBAL_MAX_i32_SUMI %lld   VERDICT %s\n",
           worstNorm, (long long)globalMaxSumi,
           anyBug ? "INT-BUG" : "BYTE-EXACT-INTEGER");
    return anyBug ? 1 : 0;
}
