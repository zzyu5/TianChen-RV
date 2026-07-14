// iq4_xs x q8_K 16x1-REPACKED GEVM + GEMM numeric oracle (SUPER-BLOCK CODEBOOK).
//
// Gate: does the compiler-emitted iq4_xs repack kernel compute the CORRECT
// super-block NON-LINEAR CODEBOOK + 6-bit SIGNED-SCALE dequant-matmul result?
//
// The oracle is INDEPENDENT: the REFERENCE decodes the ORIGINAL (pre-repack)
// per-super-block iq4_xs exactly as ggml's canonical dequantize/vec_dot -- 8
// sub-blocks of 32, each carrying its own 6-bit scale
//   ls = (scales_l[ib/2] >> 4*(ib%2) & 0xf) | (((scales_h >> 2*ib) & 3) << 4),
//   signed scale = ls - 32 (NO min),
// the 4-bit nibble is an INDEX into the 16-entry NON-LINEAR int8 codebook
// (kvalues_iq4nl), and
//   block_sumi = sum_ib (ls-32) * (sum_j codebook[qs&0xf]*q8[j] + codebook[qs>>4]*q8[j+16]),
//   result = fp16(x.d) * y.d * block_sumi.
// The EMITTER-MODEL reads the REPACKED block_iq4_xsx16 and reproduces the
// emitter's EXACT organization (block-as-lane strips, sh_lo/sh_hi split byte
// strips, scales_l pair strips, the memory codebook GATHER, an i32 sub-block dot
// weighted by the sign-extended scale via vmacc). DIFFERENT layouts -> agreement
// is strong evidence.
//
// ONE integer certificate is compared BYTE-EXACT (int64 equality):
//   block_sumi = sum_ib (ls-32) * sub_dot_ib   -- the whole scaled super-block dot.
//
// NEGATIVE CONTROLS perturb the REFERENCE; a mismatch SPIKE proves the structure
// was genuinely exercised by the emitter-model:
//   CODEBOOK   : the REFERENCE treats the nibble as a LINEAR value (weight =
//                nibble) instead of a codebook index -> real lookup diverges from
//                fake-linear. The load-bearing "real table, NOT fake-linear" gate.
//   SCALE_HI2  : the REFERENCE drops the 2 HIGH bits of the 6-bit scale (ls =
//                low4 - 32) -> the per-sub-block signed-scale assembly is proven
//                genuinely exercised (int certificate diverges).
//   SCALE_D    : rotate the per-COLUMN fp16 super-block d by +1 column (fp).
//
// Build: g++ -O2 -std=c++17 oracle_repack_iq4_xs.cpp -o /tmp/oracle_iq4xs && /tmp/oracle_iq4xs

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#define QK_K 256

static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22,
                                         -10, 1,    13,  25,  38,  53,  69,
                                         89,  113};

// ---- original per-super-block iq4_xs (ggml-common.h), 136 bytes ----
struct block_iq4_xs {
    float    d;              // super-block scale (fp16 in ggml; float for int test)
    uint16_t scales_h;       // 8x 2-bit high scale-bit pairs
    uint8_t  scales_l[QK_K / 64];   // 4 bytes: 8x 4-bit low scale nibbles
    uint8_t  qs[QK_K / 2];   // 128 nibble bytes (16 per sub-block)
};

// ---- repacked block_iq4_xsx16 (block-as-lane convention), 2176 B ----
//   d[16]        @ +0    (32 B)
//   sh_lo[16]    @ +32   (16 B: scales_h & 0xFF per column, sub-blocks 0..3)
//   sh_hi[16]    @ +48   (16 B: scales_h >> 8 per column, sub-blocks 4..7)
//   scales_l[64] @ +64   (pair p / column c at 64 + p*16 + c)
//   qs[2048]     @ +128  (sub-block ib, index i, column c at 128 + (ib*16+i)*16 + c)
struct block_iq4_xsx16 {
    float   d[16];
    uint8_t sh_lo[16];
    uint8_t sh_hi[16];
    uint8_t scales_l[64];
    uint8_t qs[2048];
};

static block_iq4_xsx16 make_block_iq4_xsx16(const block_iq4_xs *in) {
    block_iq4_xsx16 out;
    for (int c = 0; c < 16; ++c) {
        out.d[c] = in[c].d;
        out.sh_lo[c] = (uint8_t)(in[c].scales_h & 0xFF);
        out.sh_hi[c] = (uint8_t)((in[c].scales_h >> 8) & 0xFF);
        for (int p = 0; p < 4; ++p)
            out.scales_l[p * 16 + c] = in[c].scales_l[p];
    }
    for (int ib = 0; ib < 8; ++ib)
        for (int i = 0; i < 16; ++i)
            for (int c = 0; c < 16; ++c)
                out.qs[(ib * 16 + i) * 16 + c] = in[c].qs[ib * 16 + i];
    return out;
}

enum DqMode { DQ_NORMAL = 0, DQ_CODEBOOK_LINEAR = 1, DQ_SCALE_HI2 = 2 };

// ---- REFERENCE (ggml canonical iq4_xs). Returns block_sumi = sum_ib (ls-32) *
// sub_dot_ib over the 256-element super-block. ----
static int64_t ref_block(const block_iq4_xs *x, const int8_t *q8, DqMode mode) {
    int64_t block_sumi = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int low4 = (x->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF;
        int high2 = (x->scales_h >> (2 * ib)) & 0x3;
        int ls = (mode == DQ_SCALE_HI2) ? low4 : (low4 | (high2 << 4));
        int signed_scale = ls - 32;
        const uint8_t *qs = x->qs + ib * 16;
        const int8_t *q8s = q8 + ib * 32;
        int64_t sub = 0;
        for (int j = 0; j < 16; ++j) {
            int lo = qs[j] & 0x0F, hi = qs[j] >> 4;
            int wLo = (mode == DQ_CODEBOOK_LINEAR) ? lo : kvalues_iq4nl[lo];
            int wHi = (mode == DQ_CODEBOOK_LINEAR) ? hi : kvalues_iq4nl[hi];
            sub += (int64_t)wLo * q8s[j] + (int64_t)wHi * q8s[j + 16];
        }
        block_sumi += (int64_t)signed_scale * sub;
    }
    return block_sumi;
}

// ---- EMITTER-MODEL for column c of a repacked group, dotted with a per-column
// activation quant accessor act(pos). Reproduces the emitter's EXACT lane-wise
// codebook-gather + 6-bit signed-scale + i32 vmacc organization. ----
template <typename ActFn>
static int64_t mine_col(const block_iq4_xsx16 *b, int c, ActFn act,
                        int64_t &maxAbsSumi) {
    int64_t block_sumi = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int pair = ib / 2;
        uint8_t loByte = b->scales_l[pair * 16 + c];
        int low4 = (ib % 2 == 0) ? (loByte & 0x0F) : (loByte >> 4);
        uint8_t shByte = (ib < 4) ? b->sh_lo[c] : b->sh_hi[c];
        int high2 = (shByte >> (2 * (ib % 4))) & 0x3;
        int ls = low4 | (high2 << 4);
        int signed_scale = ls - 32;
        int64_t sub = 0;
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = b->qs[(ib * 16 + i) * 16 + c];
            int wLo = kvalues_iq4nl[byte & 0x0F];  // REAL codebook gather
            int wHi = kvalues_iq4nl[byte >> 4];
            sub += (int64_t)wLo * act(ib * 32 + i) +
                   (int64_t)wHi * act(ib * 32 + i + 16);
        }
        block_sumi += (int64_t)signed_scale * sub;
        if (std::llabs(block_sumi) > maxAbsSumi) maxAbsSumi = std::llabs(block_sumi);
    }
    return block_sumi;
}

static void build_iq4xs_block(block_iq4_xs *x, std::mt19937 &rng, int col, int blk) {
    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_int_distribution<int> s6(0, 63);
    x->d = 0.0021f + 0.00017f * ((col + 3 * blk) % 11);
    uint16_t sh = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int ls = s6(rng);                 // 6-bit 0..63
        if (ib % 2 == 0)
            x->scales_l[ib / 2] = (uint8_t)(ls & 0x0F);
        else
            x->scales_l[ib / 2] |= (uint8_t)((ls & 0x0F) << 4);
        sh |= (uint16_t)(((ls >> 4) & 0x3) << (2 * ib));
    }
    x->scales_h = sh;
    for (int i = 0; i < QK_K / 2; ++i) {
        int lo = nib(rng), hi = nib(rng);
        x->qs[i] = (uint8_t)(lo | (hi << 4));
    }
}

static void build_q8_block(int8_t *q8, float *d, std::mt19937 &rng, int blk) {
    std::uniform_int_distribution<int> q8d(-90, 90);
    *d = 0.015f + 0.0009f * (blk % 13);
    for (int grp = 0; grp < QK_K / 16; ++grp) {
        int dc = ((grp * 7 + blk * 3) % 21) - 10;
        for (int ii = 0; ii < 16; ++ii) {
            int v = q8d(rng) + dc;
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            q8[grp * 16 + ii] = (int8_t)v;
        }
    }
}

int main() {
    printf("# iq4_xs x q8_K 16x1-REPACKED oracle (independent scalar, SUPER-BLOCK\n");
    printf("#   CODEBOOK + 6-bit signed scale). BYTE-EXACT integer certificate\n");
    printf("#   block_sumi: reference (original iq4_xs) vs emitter-model.\n\n");

    std::mt19937 rng(20260708u);
    struct Shape { int nc; int n; int nr; };
    std::vector<Shape> shapes = {
        {16, 256, 1}, {16, 4096, 1}, {32, 256, 4}, {32, 2560, 4},
        {256, 256, 1}, {256, 4096, 8}, {160, 2560, 4},
    };

    bool anyBug = false;
    int64_t globalMaxSumi = 0;
    double worstNorm = 0.0;

    for (auto sh : shapes) {
        const int nc = sh.nc, n = sh.n, nr = sh.nr;
        const int nb = n / QK_K, ng = nc / 16;

        std::vector<block_iq4_xs> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq4xs_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                      g * 16 + col, l);

        std::vector<block_iq4_xsx16> vx((size_t)ng * nb);
        std::vector<block_iq4_xs> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq4_xsx16(tmp16.data());
            }

        std::vector<int8_t> aq((size_t)nr * nb * QK_K);
        std::vector<float> ad((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_block(&aq[((size_t)r * nb + l) * QK_K],
                               &ad[(size_t)r * nb + l], rng, r * 131 + l);

        long long mismatch = 0;
        int64_t maxSumi = 0;
        double sumsqRef = 0, maxAbsErr = 0;
        for (int r = 0; r < nr; ++r)
            for (int g = 0; g < ng; ++g)
                for (int col = 0; col < 16; ++col) {
                    int gcol = g * 16 + col;
                    double refF = 0, mineF = 0;
                    for (int l = 0; l < nb; ++l) {
                        const block_iq4_xs *xo = &orig[(size_t)gcol * nb + l];
                        const int8_t *q8 = &aq[((size_t)r * nb + l) * QK_K];
                        float dy = ad[(size_t)r * nb + l];
                        int64_t ri = ref_block(xo, q8, DQ_NORMAL);
                        const block_iq4_xsx16 *b = &vx[(size_t)g * nb + l];
                        auto actFn = [&](int pos) { return (int)q8[pos]; };
                        int64_t mi = mine_col(b, col, actFn, maxSumi);
                        if (ri != mi) mismatch++;
                        refF  += (double)xo->d * (double)dy * (double)ri;
                        mineF += (double)b->d[col] * (double)dy * (double)mi;
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
               "maxAbsSumi=%-9lld  fp-norm=%.3e  %s\n",
               nr, nc, n, mismatch, (long long)maxSumi, norm,
               bug ? "INT-BUG" : "BYTE-EXACT");
    }

    // ---- GEMM interleaved block_q8_Kx4 addressing: byte for flat pos p / column
    // c = p*4 + c. Confirm the emitter-model's interleaved read is BYTE-EXACT. ----
    {
        const int nc = 32, n = 2560;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_iq4_xs> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq4xs_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                      g * 16 + col, l);
        std::vector<block_iq4_xsx16> vx((size_t)ng * nb);
        std::vector<block_iq4_xs> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq4_xsx16(tmp16.data());
            }
        std::vector<int8_t> rowq((size_t)4 * nb * QK_K);
        std::vector<float> rowd((size_t)4 * nb);
        for (int r = 0; r < 4; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_block(&rowq[((size_t)r * nb + l) * QK_K],
                               &rowd[(size_t)r * nb + l], rng, 900 + r * 7 + l);
        std::vector<int8_t> qsx4((size_t)nb * QK_K * 4);
        for (int l = 0; l < nb; ++l)
            for (int p = 0; p < QK_K; ++p)
                for (int c = 0; c < 4; ++c)
                    qsx4[(size_t)l * QK_K * 4 + p * 4 + c] =
                        rowq[((size_t)c * nb + l) * QK_K + p];
        long long mism = 0, total = 0;
        int64_t dummy = 0;
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col) {
                int gcol = g * 16 + col;
                for (int c = 0; c < 4; ++c)
                    for (int l = 0; l < nb; ++l) {
                        int64_t ri = ref_block(&orig[(size_t)gcol * nb + l],
                                               &rowq[((size_t)c * nb + l) * QK_K],
                                               DQ_NORMAL);
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
           "# proves the iq4_xs feature is genuinely exercised by the emitter-model):\n");
    {
        const int nc = 32, n = 2560, nr = 4;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_iq4_xs> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq4xs_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                      g * 16 + col, l);
        std::vector<block_iq4_xsx16> vx((size_t)ng * nb);
        std::vector<block_iq4_xs> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq4_xsx16(tmp16.data());
            }
        std::vector<int8_t> aq((size_t)nr * nb * QK_K);
        std::vector<float> ad((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_block(&aq[((size_t)r * nb + l) * QK_K],
                               &ad[(size_t)r * nb + l], rng, r * 131 + l);

        auto runControl = [&](const char *label, DqMode mode, bool fp) {
            long long total = 0, mism = 0;
            int64_t dummy = 0;
            for (int r = 0; r < nr; ++r)
                for (int g = 0; g < ng; ++g)
                    for (int col = 0; col < 16; ++col) {
                        int gcol = g * 16 + col;
                        for (int l = 0; l < nb; ++l) {
                            const int8_t *q8 = &aq[((size_t)r * nb + l) * QK_K];
                            const block_iq4_xsx16 *b = &vx[(size_t)g * nb + l];
                            auto actFn = [&](int pos) { return (int)q8[pos]; };
                            int64_t mi = mine_col(b, col, actFn, dummy);
                            total++;
                            if (fp) {
                                double dy = ad[(size_t)r * nb + l];
                                double dRot = b->d[(col + 1) % 16];
                                double refF = dRot * dy * (double)mi;
                                double mineF = (double)b->d[col] * dy * (double)mi;
                                if (std::fabs(refF - mineF) > 1e-12) mism++;
                            } else {
                                int64_t rI = ref_block(&orig[(size_t)gcol * nb + l],
                                                       q8, mode);
                                if (rI != mi) mism++;
                            }
                        }
                    }
            printf("  %-48s : perturbed-ref vs emitter mismatch = %lld / %lld "
                   "(%.0f%%)  %s\n",
                   label, mism, total, 100.0 * mism / total,
                   mism > 0 ? "EXERCISED" : "!! NOT EXERCISED");
        };
        runControl("CODEBOOK  (nibble-as-linear, NOT table)  [sumi]",
                   DQ_CODEBOOK_LINEAR, false);
        runControl("SCALE_HI2 (drop 6-bit scale high 2 bits) [sumi]",
                   DQ_SCALE_HI2, false);
        runControl("SCALE_D   (rotate per-col fp16 d +1)      [fp]  ",
                   DQ_NORMAL, true);
    }

    printf("\nWORST_FP_NORM %.3e   GLOBAL_MAX_i32_SUMI %lld (bound 2^31)   "
           "VERDICT %s\n",
           worstNorm, (long long)globalMaxSumi,
           anyBug ? "INT-BUG" : "BYTE-EXACT-INTEGER");
    return anyBug ? 1 : 0;
}
