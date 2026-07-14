// iq2_xs x q8_K 16x1-REPACKED GEVM + GEMM numeric oracle (512-entry GRID codebook +
// ksigns SIGN plane, DUAL per-sub-block scale).
//
// Gate: does the compiler-emitted iq2_xs repack kernel compute the CORRECT
// super-block GRID + SIGN-plane dequant-matmul result?
//
// The oracle is INDEPENDENT: the REFERENCE decodes the ORIGINAL (pre-repack)
// per-super-block block_iq2_xs exactly as ggml's canonical
// ggml_vec_dot_iq2_xs_q8_K -- each 32-element sub-block has FOUR groups l=0..3;
// group l reads uint16 qs[ib32*4+l], whose low 9 bits (q2 & 511) index the FIXED
// 512-entry iq2xs_grid (each entry = 8 packed int8 grid bytes) and whose high 7
// bits (q2 >> 9) select a SIGN SELECTOR into ksigns_iq2xs (bit j flips grid[j]).
// The sub-block carries TWO scales from scales[ib32]: ls1 = 2*(sc&0xf)+1 weights
// groups 0-1, ls2 = 2*(sc>>4)+1 weights groups 2-3. block dot bsum =
// sum_ib32 (ls1*dot(groups 0-1) + ls2*dot(groups 2-3)), scaled by fp16(x.d)*y.d
// and finally by 0.125.
//
// The EMITTER-MODEL reads the REPACKED block_iq2_xsx16 and reproduces the
// emitter's EXACT organization: per (ib32, group l, column) it stores a uint16
// grid INDEX (q2 & 511) + a uint8 7-bit sign SELECTOR (q2 >> 9); per (ib32, half
// gh, column) a int8 ls scale (ls1 for gh=0, ls2 for gh=1). The model then
// GATHERS grid[idx*8+j] from a flat int8 grid table and GATHERS a +-1 sign from a
// flat sign plane signs64[sel*8+j] (exactly the vluxei16 grid + vluxei16 sign
// gathers + the vmul-onto-grid fold the emitter emits), accumulates the i32 dot,
// weights each group-half by its ls, and sums. The two paths use DIFFERENT data
// layouts, so agreement is strong evidence.
//
// ONE integer certificate is compared BYTE-EXACT (int64 equality):
//   bsum = sum_ib32 (ls1 * dot(groups 0-1) + ls2 * dot(groups 2-3))
// It is a pure integer (no fp reassociation).
//
// NEGATIVE CONTROLS perturb the REFERENCE; a mismatch SPIKE proves the
// corresponding iq2_xs structure was genuinely exercised by the emitter-model:
//   GRID   : the REFERENCE fetches a ROTATED grid entry (idx -> (idx+91)&511).
//   SIGN   : the REFERENCE IGNORES the sign plane (every sign forced +1).
//   SCALE1 : the REFERENCE perturbs ONLY ls1 (the groups-0-1 half scale).
//   SCALE2 : the REFERENCE perturbs ONLY ls2 (the groups-2-3 half scale).
// SCALE1/SCALE2 both spiking proves BOTH dual-scale halves are exercised.
//
// Build: g++ -O2 -std=c++17 oracle_repack_iq2_xs.cpp -o /tmp/oracle_iq2xs && \
//        /tmp/oracle_iq2xs

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include "iq2_grids.h"

#define QK_K 256

// ---- flat int8 grid byte table (512*8) and +-1 sign plane (128*8), EXACTLY the
// two tables the emitter emits (tcrv_iq2xs_grid viewed as int8 + the derived
// tcrv_iq2xs_signs64 from ksigns_iq2xs). ----
static int8_t grid_bytes[512 * 8];
static int8_t signs64[128 * 8];
static void build_tables() {
    for (int e = 0; e < 512; ++e) {
        uint64_t u = (uint64_t)iq2xs_grid[e];
        for (int j = 0; j < 8; ++j)
            grid_bytes[e * 8 + j] = (int8_t)((u >> (8 * j)) & 0xff);
    }
    for (int s = 0; s < 128; ++s)
        for (int j = 0; j < 8; ++j)
            signs64[s * 8 + j] = (ksigns_iq2xs[s] & (1 << j)) ? -1 : 1;
}

// ---- original per-super-block block_iq2_xs (ggml-common.h), 74 bytes ----
struct block_iq2_xs {
    float    d;                 // super-block scale (fp16 in ggml)
    uint16_t qs[QK_K / 8];      // 32 uint16: [9-bit grid idx | 7-bit sign sel]
    uint8_t  scales[QK_K / 32]; // 8: nibble pair (ls1 low, ls2 high) per sub-block
};

struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};

// ---- repacked block_iq2_xsx16 (block-as-lane convention), 1824 B ----
//   d[16]           @ +0     (32 B, fp16 in real kernel; float here)
//   ls[8][2][16]    @ +32    (256 B, int8: ls1 (gh=0) then ls2 (gh=1) per sub)
//   gidx[8][4][16]  @ +288   (1024 B, uint16 raw 9-bit grid index q2&511)
//   ssel[8][4][16]  @ +1312  (512 B, uint8 raw 7-bit sign selector q2>>9)
struct block_iq2_xsx16 {
    float    d[16];
    int8_t   ls[8][2][16];
    uint16_t gidx[8][4][16];
    uint8_t  ssel[8][4][16];
};

static block_iq2_xsx16 make_block_iq2_xsx16(const block_iq2_xs *in) {
    block_iq2_xsx16 out;
    for (int c = 0; c < 16; ++c) out.d[c] = in[c].d;
    for (int c = 0; c < 16; ++c)
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            int sc = in[c].scales[ib32];
            out.ls[ib32][0][c] = (int8_t)(2 * (sc & 0xf) + 1);  // ls1
            out.ls[ib32][1][c] = (int8_t)(2 * (sc >> 4) + 1);   // ls2
            for (int l = 0; l < 4; ++l) {
                uint16_t w = in[c].qs[4 * ib32 + l];
                out.gidx[ib32][l][c] = (uint16_t)(w & 511);
                out.ssel[ib32][l][c] = (uint8_t)(w >> 9);       // 7-bit
            }
        }
    return out;
}

enum DqMode { DQ_NORMAL = 0, DQ_GRID = 1, DQ_SIGN = 2, DQ_SCALE1 = 3, DQ_SCALE2 = 4 };

// ---- REFERENCE (ggml canonical ggml_vec_dot_iq2_xs_q8_K). ----
static int64_t ref_block(const block_iq2_xs *x, const int8_t *q8, DqMode mode) {
    int64_t bsum = 0;
    const uint16_t *q2 = x->qs;
    const uint8_t *sc = x->scales;
    int q8pos = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        int64_t ls1 = 2 * (sc[ib32] & 0xf) + 1;
        int64_t ls2 = 2 * (sc[ib32] >> 4) + 1;
        if (mode == DQ_SCALE1) ls1 += 2;
        if (mode == DQ_SCALE2) ls2 += 2;
        // groups 0-1 (ls1)
        int64_t sumi = 0;
        for (int l = 0; l < 2; ++l) {
            uint16_t w = q2[4 * ib32 + l];
            int idx = w & 511;
            if (mode == DQ_GRID) idx = (idx + 91) & 511;
            const int8_t *grid = &grid_bytes[idx * 8];
            uint8_t signs = ksigns_iq2xs[w >> 9];
            for (int j = 0; j < 8; ++j) {
                int sgn = (mode == DQ_SIGN) ? 1 : ((signs & (1 << j)) ? -1 : 1);
                sumi += (int64_t)q8[q8pos++] * grid[j] * sgn;
            }
        }
        bsum += sumi * ls1;
        // groups 2-3 (ls2)
        sumi = 0;
        for (int l = 2; l < 4; ++l) {
            uint16_t w = q2[4 * ib32 + l];
            int idx = w & 511;
            if (mode == DQ_GRID) idx = (idx + 91) & 511;
            const int8_t *grid = &grid_bytes[idx * 8];
            uint8_t signs = ksigns_iq2xs[w >> 9];
            for (int j = 0; j < 8; ++j) {
                int sgn = (mode == DQ_SIGN) ? 1 : ((signs & (1 << j)) ? -1 : 1);
                sumi += (int64_t)q8[q8pos++] * grid[j] * sgn;
            }
        }
        bsum += sumi * ls2;
    }
    return bsum;
}

// ---- EMITTER-MODEL for column c: reproduce the block-as-lane organization
// (u16 grid index gather + u8 sign selector gather + dual ls-weighted i32 dot). ----
template <typename ActFn>
static int64_t mine_col(const block_iq2_xsx16 *b, int c, ActFn act,
                        int64_t &maxAbsBsum) {
    int64_t bsum = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        for (int gh = 0; gh < 2; ++gh) {
            int64_t sumi = 0;
            for (int grp = gh * 2; grp < gh * 2 + 2; ++grp) {
                int idx = b->gidx[ib32][grp][c];   // u16 grid index
                int sel = b->ssel[ib32][grp][c];   // u8 7-bit sign selector
                for (int j = 0; j < 8; ++j) {
                    int k = ib32 * 32 + grp * 8 + j;
                    int gbv = grid_bytes[idx * 8 + j];  // REAL grid GATHER
                    int sgn = signs64[sel * 8 + j];     // REAL sign-plane GATHER
                    sumi += (int64_t)act(k) * gbv * sgn;
                }
            }
            bsum += sumi * (int64_t)b->ls[ib32][gh][c];
        }
        if (std::llabs(bsum) > maxAbsBsum) maxAbsBsum = std::llabs(bsum);
    }
    return bsum;
}

static void build_iq2xs_block(block_iq2_xs *x, std::mt19937 &rng, int col, int blk) {
    std::uniform_int_distribution<int> gi(0, 511);   // 9-bit grid index
    std::uniform_int_distribution<int> si(0, 127);   // 7-bit sign selector
    std::uniform_int_distribution<int> sc(0, 15);    // 4-bit scale nibble
    x->d = 0.010f + 0.0006f * ((col + 3 * blk) % 11);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        for (int l = 0; l < 4; ++l)
            x->qs[4 * ib32 + l] = (uint16_t)(gi(rng) | (si(rng) << 9));
        x->scales[ib32] = (uint8_t)(sc(rng) | (sc(rng) << 4));
    }
}

static void build_q8_K_block(block_q8_K *a, std::mt19937 &rng, int blk) {
    std::uniform_int_distribution<int> q8d(-90, 90);
    a->d = 0.015f + 0.0009f * (blk % 13);
    for (int i = 0; i < QK_K; ++i) {
        int dc = ((i * 7 + blk * 3) % 21) - 10;
        int v = q8d(rng) + dc;
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        a->qs[i] = (int8_t)v;
    }
    for (int i = 0; i < QK_K / 16; ++i) a->bsums[i] = 0;
}

int main() {
    build_tables();
    printf("# iq2_xs x q8_K 16x1-REPACKED oracle (512-grid + ksigns sign plane, dual scale).\n");
    printf("# BYTE-EXACT integer certificate bsum (grid*sign*q8, dual-ls-weighted):\n");
    printf("#   reference (original iq2_xs) vs emitter-model (block_iq2_xsx16).\n\n");

    std::mt19937 rng(20260708u);
    struct Shape { int nc; int n; int nr; };
    std::vector<Shape> shapes = {
        {16, 256, 1},   {16, 4096, 1}, {32, 256, 4},  {32, 2560, 4},
        {256, 256, 1},  {256, 4096, 8}, {160, 2560, 4},
    };

    bool anyBug = false;
    int64_t globalMaxBsum = 0;
    double worstNorm = 0.0;

    for (auto sh : shapes) {
        const int nc = sh.nc, n = sh.n, nr = sh.nr;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_iq2_xs> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq2xs_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                      g * 16 + col, l);
        std::vector<block_iq2_xsx16> vx((size_t)ng * nb);
        std::vector<block_iq2_xs> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq2_xsx16(tmp16.data());
            }
        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        long long mismatch = 0;
        int64_t maxBsum = 0;
        double sumsqRef = 0, maxAbsErr = 0;
        for (int r = 0; r < nr; ++r)
            for (int g = 0; g < ng; ++g)
                for (int col = 0; col < 16; ++col) {
                    int gcol = g * 16 + col;
                    double refF = 0, mineF = 0;
                    for (int l = 0; l < nb; ++l) {
                        const block_iq2_xs *xo = &orig[(size_t)gcol * nb + l];
                        const block_q8_K *a = &act[(size_t)r * nb + l];
                        int64_t ri = ref_block(xo, a->qs, DQ_NORMAL);
                        const block_iq2_xsx16 *b = &vx[(size_t)g * nb + l];
                        auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                        int64_t mi = mine_col(b, col, actFn, maxBsum);
                        if (ri != mi) mismatch++;
                        refF  += 0.125 * (double)xo->d * (double)a->d * (double)ri;
                        mineF += 0.125 * (double)b->d[col] * (double)a->d * (double)mi;
                    }
                    double e = std::fabs(refF - mineF);
                    if (e > maxAbsErr) maxAbsErr = e;
                    sumsqRef += refF * refF;
                }
        int ncnt = nr * nc;
        double rms = std::sqrt(sumsqRef / ncnt);
        double norm = rms > 0 ? maxAbsErr / rms : maxAbsErr;
        if (norm > worstNorm) worstNorm = norm;
        if (maxBsum > globalMaxBsum) globalMaxBsum = maxBsum;
        bool bug = (mismatch != 0);
        if (bug) anyBug = true;
        printf("  shape nr=%-3d nc=%-4d n=%-6d : int-mismatch=%-6lld  "
               "maxAbsBsum=%-9lld  fp-norm=%.3e  %s\n",
               nr, nc, n, mismatch, (long long)maxBsum, norm,
               bug ? "INT-BUG" : "BYTE-EXACT");
    }

    // ---- GEMM interleaved block_q8_Kx4 addressing (qs @+16 pos*4+c). ----
    {
        const int nc = 32, n = 2560;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_iq2_xs> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq2xs_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                      g * 16 + col, l);
        std::vector<block_iq2_xsx16> vx((size_t)ng * nb);
        std::vector<block_iq2_xs> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq2_xsx16(tmp16.data());
            }
        std::vector<block_q8_K> row(4 * nb);
        for (int r = 0; r < 4; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_K_block(&row[(size_t)r * nb + l], rng, 900 + r * 7 + l);
        std::vector<int8_t> qsx4((size_t)nb * QK_K * 4);
        for (int l = 0; l < nb; ++l)
            for (int p = 0; p < QK_K; ++p)
                for (int c = 0; c < 4; ++c)
                    qsx4[(size_t)l * QK_K * 4 + p * 4 + c] = row[(size_t)c * nb + l].qs[p];
        long long mism = 0, total = 0;
        int64_t dummy = 0;
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col) {
                int gcol = g * 16 + col;
                for (int c = 0; c < 4; ++c)
                    for (int l = 0; l < nb; ++l) {
                        int64_t ri = ref_block(&orig[(size_t)gcol * nb + l],
                                               row[(size_t)c * nb + l].qs, DQ_NORMAL);
                        const int8_t *qsl = &qsx4[(size_t)l * QK_K * 4];
                        auto actFn = [&](int pos) { return (int)qsl[pos * 4 + c]; };
                        int64_t mi = mine_col(&vx[(size_t)g * nb + l], col, actFn, dummy);
                        total++;
                        if (ri != mi) mism++;
                    }
            }
        printf("\n# GEMM interleaved block_q8_Kx4 (qs @+16 pos*4+c): int-mismatch = "
               "%lld / %lld  %s\n", mism, total, mism ? "INT-BUG" : "BYTE-EXACT");
        if (mism) anyBug = true;
    }

    // ---- NEGATIVE CONTROLS (nr=4, nc=32, n=2560). ----
    printf("\n# negative controls (perturb the REFERENCE; a large mismatch fraction\n"
           "# proves the iq2_xs feature is genuinely exercised by the emitter-model):\n");
    {
        const int nc = 32, n = 2560, nr = 4;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_iq2_xs> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq2xs_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                      g * 16 + col, l);
        std::vector<block_iq2_xsx16> vx((size_t)ng * nb);
        std::vector<block_iq2_xs> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq2_xsx16(tmp16.data());
            }
        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        struct Ctl { const char *name; DqMode mode; };
        Ctl ctls[4] = {
            {"GRID   (rotate grid index +91, wrong 512-entry)   [bsum]", DQ_GRID},
            {"SIGN   (ignore ksigns sign plane, all +1)         [bsum]", DQ_SIGN},
            {"SCALE1 (perturb ONLY ls1, groups-0-1 half)        [bsum]", DQ_SCALE1},
            {"SCALE2 (perturb ONLY ls2, groups-2-3 half)        [bsum]", DQ_SCALE2},
        };
        for (auto &ct : ctls) {
            long long total = 0, mism = 0;
            int64_t dummy = 0;
            for (int r = 0; r < nr; ++r)
                for (int g = 0; g < ng; ++g)
                    for (int col = 0; col < 16; ++col) {
                        int gcol = g * 16 + col;
                        for (int l = 0; l < nb; ++l) {
                            const block_q8_K *a = &act[(size_t)r * nb + l];
                            int64_t rI = ref_block(&orig[(size_t)gcol * nb + l], a->qs, ct.mode);
                            auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                            int64_t mI = mine_col(&vx[(size_t)g * nb + l], col, actFn, dummy);
                            total++;
                            if (rI != mI) mism++;
                        }
                    }
            printf("  %-52s : perturbed-ref vs emitter mismatch = %lld / %lld (%.0f%%)  %s\n",
                   ct.name, mism, total, 100.0 * mism / total,
                   mism > 0 ? "EXERCISED" : "!! NOT EXERCISED");
        }
    }

    printf("\nWORST_FP_NORM %.3e   GLOBAL_MAX_i32_BSUM %lld   VERDICT %s\n",
           worstNorm, (long long)globalMaxBsum,
           anyBug ? "INT-BUG" : "BYTE-EXACT-INTEGER");
    return anyBug ? 1 : 0;
}
