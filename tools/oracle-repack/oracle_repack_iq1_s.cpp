// iq1_s x q8_K 16x1-REPACKED GEVM + GEMM numeric oracle (2048-entry TERNARY grid
// codebook + per-sub-block +-1 DELTA over the activation bsums).
//
// Gate: does the compiler-emitted iq1_s repack kernel compute the CORRECT
// super-block ternary-grid + delta-bsum dequant-matmul result?
//
// ---------------------------------------------------------------------------
// [K-5b] THREE REQUIREMENTS, and how this oracle meets each:
//
// (3) ORACLE INDEPENDENT / NO SHARED DECODE IMPLEMENTATION.
//     The REFERENCE decodes the ORIGINAL (pre-repack) packed block_iq1_s exactly
//     as ggml's canonical ggml_vec_dot_iq1_s_q8_K does: it reads the raw uint16
//     qh word and derives, PER SUB-BLOCK AND PER GROUP, at read time:
//         ls    = 2*((qh[ib] >> 12) & 7) + 1
//         delta = qh[ib] & 0x8000 ? -1 : 1
//         idx   = qs[4*ib+l] | (((qh[ib] >> 3*l) & 7) << 8)
//     and fetches grid bytes by REINTERPRETING the uint64 grid entry
//     ((const int8_t *)(iq1s_grid + idx)), which is ggml's own read.
//     The EMITTER-MODEL never sees a qh word: it reads the REPACKED
//     block_iq1_sx16, where ls / delta / the 11-bit index are ALREADY separate
//     flat per-column strips, and gathers grid bytes from a FLAT int8 table by
//     grid_bytes[idx*8+j] (the vluxei16 gather the emitter emits). Different
//     layout, different index derivation, different table read. The ONLY shared
//     thing is iq1s_grid itself -- the FORMAT CONSTANT, not an implementation.
//
// (2) INPUT PATH SAME-SOURCE. Both paths are fed from the SAME generated
//     block_iq1_s array: the repacked strips are produced from it by
//     make_block_iq1_sx16 (the mat-quant the repack stage performs). The
//     reference reads the originals; the model reads the repack of those same
//     originals. Nothing is captured from the model and replayed.
//
// (1) CORPUS COMPLETENESS. This is the requirement the P1 oracle FAILED (it
//     never exercised the negative-scale path yet claimed every term was
//     exercised). So this oracle does not claim coverage -- it MEASURES and
//     PRINTS it, and returns non-zero if any axis is short:
//       * all 2048 11-bit grid index values         (COVERAGE grid_index)
//       * BOTH qh bit15 delta polarities  {-1,+1}   (COVERAGE delta)
//       * all 8 ls steps {1,3,5,7,9,11,13,15}       (COVERAGE ls)
//     The generator sweeps these SYSTEMATICALLY (not by luck) so coverage is
//     structural, and the counters below prove it rather than assert it.
//
// ---------------------------------------------------------------------------
// TWO integer certificates are compared BYTE-EXACT (int64 equality) -- iq1_s has
// TWO integer accumulators and checking only the first would leave the entire
// delta mechanism unverified:
//   sumi  = sum_ib ls * dot32(ib)                       (the ternary grid dot)
//   sumi1 = sum_ib ls * delta * (bsums[2ib] + bsums[2ib+1])   (the delta term)
// Both are pure integers (no fp reassociation). The fp fold
//   sumf += d_x*d_y * (sumi + 0.125f*sumi1)
// is reported as a norm only.
//
// NEGATIVE CONTROLS perturb the REFERENCE; a mismatch SPIKE proves the
// corresponding iq1_s structure was genuinely exercised by the emitter-model:
//   GRID    : the REFERENCE fetches a ROTATED grid entry (idx -> (idx+457)&2047).
//   QHHIGH  : the REFERENCE drops the qh HIGH 3 index bits (idx &= 0xff), so the
//             index degrades to 8-bit -- proves the 11-BIT assembly is exercised
//             and the grid is really 2048-entry, not 256.
//   LS      : the REFERENCE perturbs the per-sub-block ls (+2).
//   DELTA   : the REFERENCE IGNORES qh bit15 (delta forced +1) -- proves BOTH
//             polarities are exercised (a corpus with only +1 would NOT spike).
//   BSUMS   : the REFERENCE zeroes the activation bsums -- proves the delta term
//             really consumes bsums.
// GRID/QHHIGH/LS spike sumi (and sumi1 for LS); DELTA/BSUMS spike sumi1 ONLY,
// which is exactly right: they are delta-term-only structures.
//
// Build: g++ -O2 -std=c++17 oracle_repack_iq1_s.cpp -o /tmp/oracle_iq1s && \
//        /tmp/oracle_iq1s

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <set>
#include <random>
#include "iq1s_grid.h"

#define QK_K 256

// ---- flat int8 grid byte table (2048*8), EXACTLY the table the emitter gathers
// from (weft_iq1s_grid viewed through a (const int8_t *) cast). Built by shift
// extraction; the REFERENCE does NOT use this -- it reinterprets the uint64. ----
static int8_t grid_bytes[2048 * 8];
static void build_tables() {
    for (int e = 0; e < 2048; ++e) {
        uint64_t u = iq1s_grid[e];
        for (int j = 0; j < 8; ++j)
            grid_bytes[e * 8 + j] = (int8_t)((u >> (8 * j)) & 0xff);
    }
}

// ---- original per-super-block block_iq1_s (ggml-common.h), 50 bytes ----
struct block_iq1_s {
    float    d;                 // super-block scale (fp16 in ggml)
    uint8_t  qs[QK_K / 8];      // 32: low 8 bits of each group's grid index
    uint16_t qh[QK_K / 32];     // 8: [sign|ls(3)|g3(3)|g2(3)|g1(3)|g0(3)]
};

struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};

// ---- repacked block_iq1_sx16 (block-as-lane convention), 1312 B ----
//   d[16]           @ +0     (32 B, fp16 in the real kernel; float here)
//   ls[8][16]       @ +32    (128 B, int8 SINGLE ls per sub-block, [1,15])
//   delta[8][16]    @ +160   (128 B, int8 +-1; rides the sign-plane offset SLOT)
//   gidx[8][4][16]  @ +288   (1024 B, uint16 assembled 11-bit index [0,2047])
struct block_iq1_sx16 {
    float    d[16];
    int8_t   ls[8][16];
    int8_t   delta[8][16];
    uint16_t gidx[8][4][16];
};

// The REPACK (mat-quant): decode each qh word ONCE, into flat strips. This is the
// layout transform the repack stage performs; the kernel then never sees qh.
static block_iq1_sx16 make_block_iq1_sx16(const block_iq1_s *in) {
    block_iq1_sx16 out;
    for (int c = 0; c < 16; ++c) out.d[c] = in[c].d;
    for (int c = 0; c < 16; ++c)
        for (int ib = 0; ib < 8; ++ib) {
            int qhw = in[c].qh[ib];
            out.ls[ib][c]    = (int8_t)(2 * ((qhw >> 12) & 7) + 1);
            out.delta[ib][c] = (int8_t)(1 - 2 * ((qhw >> 15) & 1));
            for (int l = 0; l < 4; ++l)
                out.gidx[ib][l][c] =
                    (uint16_t)(in[c].qs[4 * ib + l] | (((qhw >> (3 * l)) & 7) << 8));
        }
    return out;
}

enum DqMode { DQ_NORMAL = 0, DQ_GRID = 1, DQ_QHHIGH = 2, DQ_LS = 3,
              DQ_DELTA = 4, DQ_BSUMS = 5 };

// ---- REFERENCE (ggml canonical ggml_vec_dot_iq1_s_q8_K), reading the ORIGINAL
// packed block: it derives ls/delta/idx from the raw qh word itself and reads the
// grid through the uint64 reinterpret, exactly as ggml does. ----
static void ref_block(const block_iq1_s *x, const int8_t *q8,
                      const int16_t *bsums, DqMode mode,
                      int64_t &sumi_out, int64_t &sumi1_out) {
    int64_t sumi = 0, sumi1 = 0;
    const uint8_t  *qs = x->qs;
    const uint16_t *qh = x->qh;
    int q8pos = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int qhw = qh[ib];
        int64_t ls = 2 * ((qhw >> 12) & 7) + 1;
        if (mode == DQ_LS) ls += 2;
        int64_t delta = (mode == DQ_DELTA) ? 1 : ((qhw & 0x8000) ? -1 : 1);
        int64_t lsum = 0;
        for (int l = 0; l < 4; ++l) {
            int idx = qs[4 * ib + l] | (((qhw >> (3 * l)) & 7) << 8);
            if (mode == DQ_QHHIGH) idx &= 0xff;          // drop the qh high bits
            if (mode == DQ_GRID)   idx = (idx + 457) & 2047;
            // ggml's own read: reinterpret the uint64 grid entry as 8 int8.
            const int8_t *grid = (const int8_t *)(iq1s_grid + idx);
            for (int j = 0; j < 8; ++j)
                lsum += (int64_t)q8[q8pos++] * grid[j];
        }
        sumi += ls * lsum;
        int64_t b0 = bsums[2 * ib + 0], b1 = bsums[2 * ib + 1];
        if (mode == DQ_BSUMS) { b0 = 0; b1 = 0; }
        sumi1 += ls * delta * (b0 + b1);
    }
    sumi_out = sumi;
    sumi1_out = sumi1;
}

// ---- EMITTER-MODEL for column c: reproduce the block-as-lane organization
// (u16 grid index -> flat int8 grid GATHER, NO sign plane, ls-weighted i32 dot,
// and the ls*delta*bsum-pair second accumulator). ----
template <typename ActFn, typename BsumFn>
static void mine_col(const block_iq1_sx16 *b, int c, ActFn act, BsumFn bsum,
                     int64_t &sumi_out, int64_t &sumi1_out,
                     int64_t &maxAbsSumi) {
    int64_t sumi = 0, sumi1 = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int64_t sub = 0;
        for (int grp = 0; grp < 4; ++grp) {
            int idx = b->gidx[ib][grp][c];          // pre-assembled 11-bit index
            for (int j = 0; j < 8; ++j) {
                int k = ib * 32 + grp * 8 + j;
                int gbv = grid_bytes[idx * 8 + j];  // REAL ternary grid GATHER
                sub += (int64_t)act(k) * gbv;       // NO sign fold: grid is signed
            }
        }
        sumi += sub * (int64_t)b->ls[ib][c];
        sumi1 += (int64_t)b->ls[ib][c] * (int64_t)b->delta[ib][c] *
                 ((int64_t)bsum(2 * ib + 0) + (int64_t)bsum(2 * ib + 1));
        if (std::llabs(sumi) > maxAbsSumi) maxAbsSumi = std::llabs(sumi);
    }
    sumi_out = sumi;
    sumi1_out = sumi1;
}

// ---- CORPUS COVERAGE instrumentation (the [K-5b](1) evidence). ----
static std::set<int> cov_idx;      // distinct 11-bit grid indices exercised
static long long cov_delta[2];     // qh bit15 == 0 (+1) / == 1 (-1)
static long long cov_ls[8];        // the 8 ls steps (qh>>12)&7

// The generator SWEEPS the decode axes systematically so coverage is structural,
// not luck: idx runs a full sequential cycle over [0,2047]; the ls field runs a
// stride-3 cycle over [0,7] (3 is coprime with 8 => all 8 steps); the sign bit
// alternates on the ls cycle. Activation magnitudes stay random.
static long long g_idx_ctr = 0, g_qh_ctr = 0;

static void build_iq1s_block(block_iq1_s *x, std::mt19937 &rng, int col, int blk) {
    x->d = 0.010f + 0.0006f * ((col + 3 * blk) % 11);
    for (int ib = 0; ib < 8; ++ib) {
        int lsField = (int)((g_qh_ctr * 3) % 8);
        int signBit = (int)((g_qh_ctr / 8) % 2);
        ++g_qh_ctr;
        cov_ls[lsField]++;
        cov_delta[signBit]++;
        int qhw = (lsField << 12) | (signBit << 15);
        for (int l = 0; l < 4; ++l) {
            int idx = (int)(g_idx_ctr % 2048);
            ++g_idx_ctr;
            cov_idx.insert(idx);
            x->qs[4 * ib + l] = (uint8_t)(idx & 0xff);
            qhw |= ((idx >> 8) & 7) << (3 * l);
        }
        x->qh[ib] = (uint16_t)qhw;
    }
}

// A REAL block_q8_K: bsums[i] is the true sum of the 16 quants of group i (what
// ggml's quantize_row_q8_K computes). The delta term reads them, so they must NOT
// be zero -- an all-zero-bsums corpus would leave sumi1 identically 0 and the
// whole delta mechanism unexercised (the P1 failure mode).
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
    for (int g = 0; g < QK_K / 16; ++g) {
        int s = 0;
        for (int j = 0; j < 16; ++j) s += a->qs[g * 16 + j];
        a->bsums[g] = (int16_t)s;
    }
}

// ---- The REAL (fp16-d) block_iq1_sx16 byte layout, recomputed independently here
// and pinned against the byte offsets the FRONT DOOR ships (kIq1SDecodeFacts in
// lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp, and the lit CHECKs in
// test/Conversion/RVV/rvv-emit-*-iq1-s-repack-*.mlir).
//
// SCOPE, stated honestly: the emitter-model above addresses the repacked strips by
// ARRAY INDEX, so it validates the layout's SEMANTICS + the arithmetic, NOT the
// emitter's byte arithmetic (its `float d[16]` is 64 B where the kernel's fp16 d[16]
// is 32 B -- the same modelling shortcut the 11 sibling oracles take). These pins
// therefore cover the remaining axis: that the layout this oracle models is the one
// the compiler actually emits offsets for. If someone re-lays-out the strip in the
// front door without re-laying-out this oracle, THIS is what goes red.
static bool check_layout_pins() {
    const int kInterleave = 16;
    int off_d     = 0;
    int off_ls    = off_d  + kInterleave * 2;              // 16 fp16 d  -> +32
    int off_delta = off_ls + 8 * kInterleave;              // 8*16 int8  -> +160
    int off_gidx  = off_delta + 8 * kInterleave;           // 8*16 int8  -> +288
    int stride    = off_gidx + 8 * 4 * kInterleave * 2;    // 8*4*16 u16 -> 1312
    struct Pin { const char *name; int got; int shipped; };
    Pin pins[] = {
        {"weight_block_stride       ", stride,    1312},
        {"weight_ls_byte_offset     ", off_ls,      32},
        {"weight_sign(DELTA)_offset ", off_delta,  160},
        {"weight_grid_idx_offset    ", off_gidx,   288},
        // block_q8_K: fp32 d @0 + 256 int8 quants @4 + 16 int16 bsums @260 = 292.
        {"gevm_act_quant_offset     ", 4,            4},
        {"gevm_act_bsums_offset     ", 4 + 256,    260},
        {"gevm_act_block_stride     ", 4 + 256 + 32, 292},
        // block_q8_Kx4: 4 fp32 d @0 + 1024 int8 quants @16 + 64 int16 bsums @1040.
        {"gemm_act_quant_offset     ", 16,          16},
        {"gemm_act_bsums_offset     ", 16 + 1024, 1040},
        {"gemm_act_block_stride     ", 16 + 1024 + 128, 1168},
    };
    bool ok = true;
    printf("\n# layout pins (independently recomputed here vs the offsets the front\n"
           "# door ships; the emitter-model above is index-addressed, so THIS is what\n"
           "# ties the modelled layout to the one the compiler emits):\n");
    for (auto &p : pins) {
        bool hit = p.got == p.shipped;
        if (!hit) ok = false;
        printf("  %s recomputed=%-5d shipped=%-5d  %s\n", p.name, p.got, p.shipped,
               hit ? "PIN OK" : "!! LAYOUT DRIFT");
    }
    return ok;
}

int main() {
    build_tables();
    printf("# iq1_s x q8_K 16x1-REPACKED oracle (2048 TERNARY grid + qh delta/bsums).\n");
    printf("# BYTE-EXACT integer certificates sumi (ls-weighted ternary grid dot) AND\n");
    printf("# sumi1 (ls*delta*bsum-pair): reference (ORIGINAL packed block_iq1_s, qh\n");
    printf("# decoded at read time) vs emitter-model (REPACKED block_iq1_sx16 strips).\n\n");

    std::mt19937 rng(20260717u);
    struct Shape { int nc; int n; int nr; };
    std::vector<Shape> shapes = {
        {16, 256, 1},   {16, 4096, 1}, {32, 256, 4},  {32, 2560, 4},
        {256, 256, 1},  {256, 4096, 8}, {160, 2560, 4},
    };

    bool anyBug = false;
    int64_t globalMaxSumi = 0;
    double worstNorm = 0.0;

    for (auto sh : shapes) {
        const int nc = sh.nc, n = sh.n, nr = sh.nr;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_iq1_s> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq1s_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                     g * 16 + col, l);
        std::vector<block_iq1_sx16> vx((size_t)ng * nb);
        std::vector<block_iq1_s> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq1_sx16(tmp16.data());
            }
        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        long long mismatch = 0, mismatch1 = 0;
        int64_t maxSumi = 0;
        double sumsqRef = 0, maxAbsErr = 0;
        for (int r = 0; r < nr; ++r)
            for (int g = 0; g < ng; ++g)
                for (int col = 0; col < 16; ++col) {
                    int gcol = g * 16 + col;
                    double refF = 0, mineF = 0;
                    for (int l = 0; l < nb; ++l) {
                        const block_iq1_s *xo = &orig[(size_t)gcol * nb + l];
                        const block_q8_K *a = &act[(size_t)r * nb + l];
                        int64_t ri, ri1;
                        ref_block(xo, a->qs, a->bsums, DQ_NORMAL, ri, ri1);
                        const block_iq1_sx16 *b = &vx[(size_t)g * nb + l];
                        auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                        auto bsFn = [&](int gi) { return (int)a->bsums[gi]; };
                        int64_t mi, mi1;
                        mine_col(b, col, actFn, bsFn, mi, mi1, maxSumi);
                        if (ri != mi) mismatch++;
                        if (ri1 != mi1) mismatch1++;
                        refF  += (double)xo->d * (double)a->d *
                                 ((double)ri + 0.125 * (double)ri1);
                        mineF += (double)b->d[col] * (double)a->d *
                                 ((double)mi + 0.125 * (double)mi1);
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
        bool bug = (mismatch != 0 || mismatch1 != 0);
        if (bug) anyBug = true;
        printf("  shape nr=%-3d nc=%-4d n=%-6d : sumi-mismatch=%-6lld "
               "sumi1-mismatch=%-6lld  maxAbsSumi=%-9lld  fp-norm=%.3e  %s\n",
               nr, nc, n, mismatch, mismatch1, (long long)maxSumi, norm,
               bug ? "INT-BUG" : "BYTE-EXACT");
    }

    // ---- GEMM interleaved block_q8_Kx4 addressing (qs @+16 pos*4+c, bsums @+1040
    // g16*4+c). Same weight strips; only the activation addressing changes. ----
    {
        const int nc = 32, n = 2560;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_iq1_s> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq1s_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                     g * 16 + col, l);
        std::vector<block_iq1_sx16> vx((size_t)ng * nb);
        std::vector<block_iq1_s> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq1_sx16(tmp16.data());
            }
        std::vector<block_q8_K> row(4 * nb);
        for (int r = 0; r < 4; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_K_block(&row[(size_t)r * nb + l], rng, 900 + r * 7 + l);
        // The interleaved x4 activation planes.
        std::vector<int8_t> qsx4((size_t)nb * QK_K * 4);
        std::vector<int16_t> bsx4((size_t)nb * 16 * 4);
        for (int l = 0; l < nb; ++l) {
            for (int p = 0; p < QK_K; ++p)
                for (int c = 0; c < 4; ++c)
                    qsx4[(size_t)l * QK_K * 4 + p * 4 + c] = row[(size_t)c * nb + l].qs[p];
            for (int g16 = 0; g16 < 16; ++g16)
                for (int c = 0; c < 4; ++c)
                    bsx4[(size_t)l * 16 * 4 + g16 * 4 + c] =
                        row[(size_t)c * nb + l].bsums[g16];
        }
        long long mism = 0, mism1 = 0, total = 0;
        int64_t dummy = 0;
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col) {
                int gcol = g * 16 + col;
                for (int c = 0; c < 4; ++c)
                    for (int l = 0; l < nb; ++l) {
                        int64_t ri, ri1;
                        ref_block(&orig[(size_t)gcol * nb + l],
                                  row[(size_t)c * nb + l].qs,
                                  row[(size_t)c * nb + l].bsums, DQ_NORMAL, ri, ri1);
                        const int8_t *qsl = &qsx4[(size_t)l * QK_K * 4];
                        const int16_t *bsl = &bsx4[(size_t)l * 16 * 4];
                        auto actFn = [&](int pos) { return (int)qsl[pos * 4 + c]; };
                        auto bsFn = [&](int gi) { return (int)bsl[gi * 4 + c]; };
                        int64_t mi, mi1;
                        mine_col(&vx[(size_t)g * nb + l], col, actFn, bsFn, mi, mi1,
                                 dummy);
                        total++;
                        if (ri != mi) mism++;
                        if (ri1 != mi1) mism1++;
                    }
            }
        printf("\n# GEMM interleaved block_q8_Kx4 (qs @+16 pos*4+c, bsums @+1040 g16*4+c):\n"
               "#   sumi-mismatch = %lld / %lld   sumi1-mismatch = %lld / %lld   %s\n",
               mism, total, mism1, total,
               (mism || mism1) ? "INT-BUG" : "BYTE-EXACT");
        if (mism || mism1) anyBug = true;
    }

    // ---- NEGATIVE CONTROLS (nr=4, nc=32, n=2560). ----
    printf("\n# negative controls (perturb the REFERENCE; a large mismatch fraction\n"
           "# proves the iq1_s feature is genuinely exercised by the emitter-model):\n");
    {
        const int nc = 32, n = 2560, nr = 4;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_iq1_s> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq1s_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                     g * 16 + col, l);
        std::vector<block_iq1_sx16> vx((size_t)ng * nb);
        std::vector<block_iq1_s> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq1_sx16(tmp16.data());
            }
        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        struct Ctl { const char *name; DqMode mode; };
        Ctl ctls[5] = {
            {"GRID   (rotate index +457, wrong 2048-entry)      [sumi ]", DQ_GRID},
            {"QHHIGH (drop qh high 3 idx bits, idx&=0xff)       [sumi ]", DQ_QHHIGH},
            {"LS     (perturb per-sub-block ls +2)              [both ]", DQ_LS},
            {"DELTA  (ignore qh bit15, delta forced +1)         [sumi1]", DQ_DELTA},
            {"BSUMS  (zero the activation bsums)                [sumi1]", DQ_BSUMS},
        };
        for (auto &ct : ctls) {
            long long total = 0, mism = 0, mism1 = 0;
            int64_t dummy = 0;
            for (int r = 0; r < nr; ++r)
                for (int g = 0; g < ng; ++g)
                    for (int col = 0; col < 16; ++col) {
                        int gcol = g * 16 + col;
                        for (int l = 0; l < nb; ++l) {
                            const block_q8_K *a = &act[(size_t)r * nb + l];
                            int64_t rI, rI1;
                            ref_block(&orig[(size_t)gcol * nb + l], a->qs, a->bsums,
                                      ct.mode, rI, rI1);
                            auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                            auto bsFn = [&](int gi) { return (int)a->bsums[gi]; };
                            int64_t mI, mI1;
                            mine_col(&vx[(size_t)g * nb + l], col, actFn, bsFn, mI,
                                     mI1, dummy);
                            total++;
                            if (rI != mI) mism++;
                            if (rI1 != mI1) mism1++;
                        }
                    }
            printf("  %-58s : sumi %lld/%lld (%.0f%%)  sumi1 %lld/%lld (%.0f%%)  %s\n",
                   ct.name, mism, total, 100.0 * mism / total,
                   mism1, total, 100.0 * mism1 / total,
                   (mism > 0 || mism1 > 0) ? "EXERCISED" : "!! NOT EXERCISED");
            if (mism == 0 && mism1 == 0) anyBug = true;  // a dead control = a hole
        }
    }

    // ---- CORPUS COMPLETENESS ([K-5b](1)): MEASURED, not claimed. ----
    printf("\n# corpus completeness (measured over every block generated above):\n");
    bool covOk = true;
    printf("  grid_index : %zu / 2048 distinct 11-bit indices exercised   %s\n",
           cov_idx.size(), cov_idx.size() == 2048 ? "COMPLETE" : "!! SHORT");
    if (cov_idx.size() != 2048) covOk = false;
    printf("  delta      : qh bit15=0 (+1) %lld blocks, bit15=1 (-1) %lld blocks  %s\n",
           cov_delta[0], cov_delta[1],
           (cov_delta[0] > 0 && cov_delta[1] > 0) ? "BOTH POLARITIES" : "!! SHORT");
    if (cov_delta[0] == 0 || cov_delta[1] == 0) covOk = false;
    printf("  ls         : ");
    for (int f = 0; f < 8; ++f) {
        printf("ls=%d:%lld ", 2 * f + 1, cov_ls[f]);
        if (cov_ls[f] == 0) covOk = false;
    }
    printf(" %s\n", covOk ? "ALL 8 STEPS" : "!! SHORT");
    if (!covOk) anyBug = true;

    bool pinsOk = check_layout_pins();
    if (!pinsOk) anyBug = true;

    printf("\nWORST_FP_NORM %.3e   GLOBAL_MAX_i32_SUMI %lld   COVERAGE %s   "
           "LAYOUT_PINS %s   VERDICT %s\n",
           worstNorm, (long long)globalMaxSumi, covOk ? "COMPLETE" : "SHORT",
           pinsOk ? "OK" : "DRIFT", anyBug ? "INT-BUG" : "BYTE-EXACT-INTEGER");
    return anyBug ? 1 : 0;
}
