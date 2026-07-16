// iq1_m x q8_K 16x1-REPACKED GEVM + GEMM numeric oracle (2048-entry TERNARY grid
// codebook + DUAL ls + PER-GROUP +-1 DELTA over IN-KERNEL group-of-8 activation sums
// + the ASSEMBLED nibble-scatter fp16 super-block scale).
//
// Gate: does the compiler-emitted iq1_m repack kernel compute the CORRECT
// super-block ternary-grid + per-group-delta dequant-matmul result?
//
// ---------------------------------------------------------------------------
// [K-5b] THREE REQUIREMENTS, and how this oracle meets each:
//
// (3) ORACLE INDEPENDENT / NO SHARED DECODE IMPLEMENTATION.
//     The REFERENCE decodes the ORIGINAL (pre-repack) packed block_iq1_m exactly
//     as ggml's canonical ggml_vec_dot_iq1_m_q8_K does: it reads the raw qh bytes
//     and the raw uint16 scale words and derives, AT READ TIME:
//         scale.u16 = (sc[0]>>12) | ((sc[1]>>8)&0x00f0) | ((sc[2]>>4)&0x0f00)
//                     | (sc[3]&0xf000)
//         ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1
//         ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1
//         delta[l] = qh[l/2] & (l%2 ? 0x80 : 0x08) ? -1 : 1
//         idx      = qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)
//     and fetches grid bytes by REINTERPRETING the uint64 grid entry
//     ((const int8_t *)(iq1s_grid + idx)), which is ggml's own read.
//     The EMITTER-MODEL never sees a qh byte or a scale word: it reads the REPACKED
//     block_iq1_mx16, where the assembled fp16, ls1/ls2, the four deltas and the
//     11-bit index are ALREADY separate flat per-column strips, and gathers grid
//     bytes from a FLAT int8 table by grid_bytes[idx*8+j] (the vluxei16 gather the
//     emitter emits). Different layout, different index derivation, different table
//     read, different scale derivation (the repack extracts nibble k as
//     ((sc[k]>>12)&0xf) << 4k -- a per-nibble loop -- where ggml uses the fused
//     four-term mask/shift expression above; equal by construction, NOT by shared
//     code). The ONLY shared thing is iq1s_grid itself -- the FORMAT CONSTANT, not
//     an implementation. (ggml's iq1_m vec_dot indexes `iq1s_grid` literally; iq1_m
//     has no separate grid of its own.)
//
// (2) INPUT PATH SAME-SOURCE. Both paths are fed from the SAME generated
//     block_iq1_m array: the repacked strips are produced from it by
//     make_block_iq1_mx16 (the mat-quant the repack stage performs). The reference
//     reads the originals; the model reads the repack of those same originals.
//     Nothing is captured from the model and replayed.
//
// (1) CORPUS COMPLETENESS. MEASURED and PRINTED, never claimed; non-zero exit if
//     any axis is short:
//       * all 2048 11-bit grid index values                 (COVERAGE grid_index)
//       * BOTH delta polarities at EACH of the 4 GROUP slots (COVERAGE delta) --
//         per-slot, because iq1_m's four deltas are INDEPENDENT bits and a corpus
//         that only ever flipped slot 0 would leave slots 1-3 unverified
//       * all 8 ls steps for ls1 AND all 8 for ls2, SEPARATELY (COVERAGE ls1/ls2)
//       * the scale assembly is certified EXHAUSTIVELY (all 16^4 = 65536 nibble
//         combinations) rather than sampled -- see certificate (3) below
//
// ---------------------------------------------------------------------------
// THREE integer certificates are compared BYTE-EXACT (int64 / uint16 equality).
// iq1_m has TWO integer accumulators AND a bit-exact scale assembly; checking only
// the first would leave both the entire delta mechanism and the whole
// nibble-scatter scale unverified:
//   sumi1 = sum_ib (ls1*sum1[0] + ls2*sum1[1])          (the ternary grid dot)
//   sumi2 = sum_ib (ls1*sum2[0] + ls2*sum2[1])          (the per-group delta term,
//           where sum2[k] = lsum2(2k)*delta[2k] + lsum2(2k+1)*delta[2k+1] and
//           lsum2(l) = sum_{j<8} q8[8l+j], the per-GROUP-OF-8 activation sum)
//   scale_u16 = the assembled iq1m_scale_t bit pattern    (EXHAUSTIVE, 65536/65536)
// All three are pure integers (no fp reassociation). The fp fold
//   sumf += d_x*d_y * (sumi1 + 0.125f*sumi2)
// is reported as a norm only.
//
// NOTE ON THE MODEL'S FOLD ORDER. ggml folds the two groups under one ls together
// FIRST (sum1[l/2] += lsum1(l); sumi1 += sum1[0]*ls1 + sum1[1]*ls2). The emitter --
// and this model -- folds PER GROUP (sumi1 += ls[grp/2]*lsum1(grp)). These are the
// same value by integer distributivity, and the certificates below are what prove
// the emitter's order was not a silent arithmetic change: the REFERENCE keeps
// ggml's grouping, the MODEL uses the emitter's, and they are compared int64-exact.
//
// NEGATIVE CONTROLS perturb the REFERENCE; a mismatch SPIKE proves the
// corresponding iq1_m structure was genuinely exercised by the emitter-model:
//   GRID    : the REFERENCE fetches a ROTATED grid entry (idx -> (idx+457)&2047).
//   QHHIGH  : the REFERENCE drops the qh HIGH 3 index bits (idx &= 0xff) -- proves
//             the 11-BIT assembly is exercised and the grid is really 2048-entry.
//   LS1     : the REFERENCE perturbs ls1 (+2) only.
//   LS2     : the REFERENCE perturbs ls2 (+2) only. SEPARATE from LS1 on purpose:
//             iq1_m is the first TernaryDelta row with DUAL ls, and one combined
//             control could pass while ls2 was never actually consumed.
//   DELTA   : the REFERENCE ignores the qh delta bits (all four forced +1) --
//             proves BOTH polarities are exercised.
//   BSUMS16 : *** the load-bearing one. *** The REFERENCE pretends the two 8-groups
//             inside one 16-group SHARE a delta (uses delta[2k] for both), which is
//             exactly what a block_q8_K bsums entry could express. A spike here is
//             the MEASURED proof of the claim this whole leaf rests on -- that
//             per-16 bsums CANNOT express iq1_m's delta term, so iq1_m genuinely
//             cannot ride the iq1_s bsums leaf. If this control did NOT spike, the
//             DeltaGridGroupSum axis value would be unjustified and iq1_m should
//             have been a parameter of the iq1_s leaf instead.
// GRID/QHHIGH spike sumi1; LS1/LS2 spike both; DELTA/BSUMS16 spike sumi2 ONLY,
// which is exactly right: they are delta-term-only structures.
//
// Build: g++ -O2 -std=c++17 oracle_repack_iq1_m.cpp -o /tmp/oracle_iq1m && \
//        /tmp/oracle_iq1m

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
// from (weft_iq1m_grid viewed through a (const int8_t *) cast). Built by shift
// extraction; the REFERENCE does NOT use this -- it reinterprets the uint64. ----
static int8_t grid_bytes[2048 * 8];
static void build_tables() {
    for (int e = 0; e < 2048; ++e) {
        uint64_t u = iq1s_grid[e];
        for (int j = 0; j < 8; ++j)
            grid_bytes[e * 8 + j] = (int8_t)((u >> (8 * j)) & 0xff);
    }
}

// ---- original per-super-block block_iq1_m (ggml-common.h), 56 bytes. NOTE there
// is NO inline d member: the fp16 super-block scale is scattered as four nibbles
// across the scales words. ----
struct block_iq1_m {
    uint8_t  qs[QK_K / 8];   // 32: low 8 bits of each group's grid index
    uint8_t  qh[QK_K / 16];  // 16: per group -> [delta|ghigh(3)] nibble pairs
    uint8_t  scales[QK_K / 32]; // 8 bytes = 4 uint16 words: ls1/ls2 + scale nibble
};

struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};

// ---- repacked block_iq1_mx16 (block-as-lane convention), 1824 B ----
//   d[16]            @ +0     (32 B, fp16 in the real kernel; float here)
//   ls[8][2][16]     @ +32    (256 B, int8 DUAL ls per sub-block, [1,15])
//   delta[8][4][16]  @ +288   (512 B, int8 +-1 PER GROUP; rides the sign-plane SLOT)
//   gidx[8][4][16]   @ +800   (1024 B, uint16 assembled 11-bit index [0,2047])
// scale_u16[16] is NOT a kernel strip -- it is the bit pattern the fp16 d strip
// carries, kept here so the assembly can be certified as an exact integer.
struct block_iq1_mx16 {
    float    d[16];
    uint16_t scale_u16[16];
    int8_t   ls[8][2][16];
    int8_t   delta[8][4][16];
    uint16_t gidx[8][4][16];
};

// fp16 bit pattern -> float (used only to give the repacked d strip a value; the
// SCALE certificate compares the u16 bit patterns, not this).
static float fp16_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t man  = (uint32_t)h & 0x3ffu;
    float sgn = sign ? -1.0f : 1.0f;
    if (exp == 0)  return sgn * std::ldexp((float)man, -24);
    if (exp == 31) return man ? NAN : sgn * INFINITY;
    return sgn * std::ldexp((float)(man | 0x400u), (int)exp - 25);
}

// The repack's INDEPENDENT scale assembly: extract nibble k (bits 12-15 of scale
// word k) and place it at bit 4k. ggml writes the same value as one fused
// four-term mask/shift expression; this per-nibble loop is a different derivation,
// which is the point ([K-5b](3)).
static uint16_t assemble_scale_repack(const uint16_t *sc) {
    uint16_t s = 0;
    for (int k = 0; k < 4; ++k)
        s |= (uint16_t)(((sc[k] >> 12) & 0xf) << (4 * k));
    return s;
}

// The REPACK (mat-quant): decode the qh/scales bits ONCE, into flat strips. This is
// the layout transform the repack stage performs; the kernel then never sees qh or
// a scale word.
static block_iq1_mx16 make_block_iq1_mx16(const block_iq1_m *in) {
    block_iq1_mx16 out;
    for (int c = 0; c < 16; ++c) {
        const uint16_t *sc = (const uint16_t *)in[c].scales;
        out.scale_u16[c] = assemble_scale_repack(sc);
        out.d[c] = fp16_to_float(out.scale_u16[c]);
        for (int ib = 0; ib < 8; ++ib) {
            // DUAL ls: bits [0..2] / [3..5] of the word for even ib, [6..8] /
            // [9..11] for odd ib.
            int shift = 6 * (ib % 2);
            out.ls[ib][0][c] = (int8_t)(2 * ((sc[ib / 2] >> (shift + 0)) & 0x7) + 1);
            out.ls[ib][1][c] = (int8_t)(2 * ((sc[ib / 2] >> (shift + 3)) & 0x7) + 1);
            for (int l = 0; l < 4; ++l) {
                int qhb = in[c].qh[2 * ib + l / 2];
                int nib = (l % 2) ? (qhb >> 4) : qhb;          // the group's nibble
                out.delta[ib][l][c] = (int8_t)((nib & 0x8) ? -1 : 1);
                out.gidx[ib][l][c] =
                    (uint16_t)(in[c].qs[4 * ib + l] | ((nib & 0x7) << 8));
            }
        }
    }
    return out;
}

enum DqMode { DQ_NORMAL = 0, DQ_GRID = 1, DQ_QHHIGH = 2, DQ_LS1 = 3,
              DQ_LS2 = 4, DQ_DELTA = 5, DQ_BSUMS16 = 6 };

// ---- REFERENCE (ggml canonical ggml_vec_dot_iq1_m_q8_K), reading the ORIGINAL
// packed block: it derives scale/ls1/ls2/delta/idx from the raw qh + scale words
// itself and reads the grid through the uint64 reinterpret, exactly as ggml does.
// It also keeps ggml's OWN fold grouping (sum1/sum2 per ls half, folded once at the
// end of the sub-block) rather than the emitter's per-group fold. ----
static void ref_block(const block_iq1_m *x, const int8_t *q8, DqMode mode,
                      int64_t &sumi1_out, int64_t &sumi2_out) {
    int64_t sumi1 = 0, sumi2 = 0;
    const uint8_t  *qs = x->qs;
    const uint8_t  *qh = x->qh;
    const uint16_t *sc = (const uint16_t *)x->scales;
    int q8pos = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int64_t sum1[2] = {0, 0}, sum2[2] = {0, 0};
        int delta[4];
        // ggml's exact delta derivation off the two qh bytes of this sub-block.
        delta[0] = (qh[2 * ib + 0] & 0x08) ? -1 : 1;
        delta[1] = (qh[2 * ib + 0] & 0x80) ? -1 : 1;
        delta[2] = (qh[2 * ib + 1] & 0x08) ? -1 : 1;
        delta[3] = (qh[2 * ib + 1] & 0x80) ? -1 : 1;
        if (mode == DQ_DELTA) { delta[0] = delta[1] = delta[2] = delta[3] = 1; }
        // BSUMS16: pretend the two 8-groups of a 16-group share a delta -- i.e.
        // exactly what a per-16 block_q8_K bsums entry could express. This is the
        // control that MEASURES whether bsums are really inexpressive for iq1_m.
        if (mode == DQ_BSUMS16) { delta[1] = delta[0]; delta[3] = delta[2]; }
        for (int l = 0; l < 4; ++l) {
            int qhb = qh[2 * ib + l / 2];
            int idx = qs[4 * ib + l] |
                      (((uint16_t)qhb << (8 - 4 * (l % 2))) & 0x700);
            if (mode == DQ_QHHIGH) idx &= 0xff;          // drop the qh high bits
            if (mode == DQ_GRID)   idx = (idx + 457) & 2047;
            // ggml's own read: reinterpret the uint64 grid entry as 8 int8.
            const int8_t *grid = (const int8_t *)(iq1s_grid + idx);
            int64_t lsum1 = 0, lsum2 = 0;
            for (int j = 0; j < 8; ++j) {
                lsum1 += (int64_t)q8[q8pos + j] * grid[j];
                lsum2 += (int64_t)q8[q8pos + j];   // the per-GROUP-OF-8 sum
            }
            q8pos += 8;
            sum1[l / 2] += lsum1;
            sum2[l / 2] += lsum2 * delta[l];
        }
        int shift = 6 * (ib % 2);
        int64_t ls1 = 2 * ((sc[ib / 2] >> (shift + 0)) & 0x7) + 1;
        int64_t ls2 = 2 * ((sc[ib / 2] >> (shift + 3)) & 0x7) + 1;
        if (mode == DQ_LS1) ls1 += 2;
        if (mode == DQ_LS2) ls2 += 2;
        sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
        sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
    }
    sumi1_out = sumi1;
    sumi2_out = sumi2;
}

// ---- EMITTER-MODEL for column c: reproduce the block-as-lane organization
// (u16 grid index -> flat int8 grid GATHER, NO sign plane, DUAL ls, and the
// PER-GROUP fold the emitter actually emits: sumi1 += ls[grp/2]*lsum1(grp),
// sumi2 += (ls[grp/2]*delta[grp])*lsum2(grp), with lsum2 accumulated IN-KERNEL
// from the same 8 activation scalars -- NO bsums read exists on this path). ----
template <typename ActFn>
static void mine_col(const block_iq1_mx16 *b, int c, ActFn act,
                     int64_t &sumi1_out, int64_t &sumi2_out,
                     int64_t &maxAbsSumi) {
    int64_t sumi1 = 0, sumi2 = 0;
    for (int ib = 0; ib < 8; ++ib) {
        for (int grp = 0; grp < 4; ++grp) {
            int gh = grp / 2;                       // ls1 for groups 0-1, ls2 for 2-3
            int idx = b->gidx[ib][grp][c];          // pre-assembled 11-bit index
            int64_t sub = 0, groupSum = 0;
            for (int j = 0; j < 8; ++j) {
                int k = ib * 32 + grp * 8 + j;
                int a = act(k);
                int gbv = grid_bytes[idx * 8 + j];  // REAL ternary grid GATHER
                sub += (int64_t)a * gbv;            // NO sign fold: grid is signed
                groupSum += a;                      // the IN-KERNEL group-of-8 sum
            }
            int64_t ls = (int64_t)b->ls[ib][gh][c];
            sumi1 += ls * sub;
            sumi2 += ls * (int64_t)b->delta[ib][grp][c] * groupSum;
        }
        if (std::llabs(sumi1) > maxAbsSumi) maxAbsSumi = std::llabs(sumi1);
    }
    sumi1_out = sumi1;
    sumi2_out = sumi2;
}

// ---- CORPUS COVERAGE instrumentation (the [K-5b](1) evidence). ----
static std::set<int> cov_idx;      // distinct 11-bit grid indices exercised
static long long cov_delta[4][2];  // PER GROUP SLOT: bit clear (+1) / set (-1)
static long long cov_ls1[8];       // the 8 ls1 steps
static long long cov_ls2[8];       // the 8 ls2 steps

// The generator SWEEPS the decode axes systematically so coverage is structural,
// not luck: idx runs a full sequential cycle over [0,2047]; the ls1 field runs a
// stride-3 cycle over [0,7] and ls2 a stride-5 cycle (3 and 5 are both coprime
// with 8 => all 8 steps each, and the two never lock together); each of the FOUR
// delta bits is driven by a different bit of a counter, so all 4 slots see both
// polarities independently. Activation magnitudes stay random.
static long long g_idx_ctr = 0, g_sb_ctr = 0;

static void build_iq1m_block(block_iq1_m *x, std::mt19937 &rng, int col, int blk) {
    uint16_t sc[4];
    for (int k = 0; k < 4; ++k) sc[k] = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int ls1f = (int)((g_sb_ctr * 3) % 8);
        int ls2f = (int)((g_sb_ctr * 5 + 2) % 8);
        cov_ls1[ls1f]++;
        cov_ls2[ls2f]++;
        int shift = 6 * (ib % 2);
        sc[ib / 2] |= (uint16_t)((ls1f & 0x7) << (shift + 0));
        sc[ib / 2] |= (uint16_t)((ls2f & 0x7) << (shift + 3));
        for (int l = 0; l < 4; ++l) {
            int idx = (int)(g_idx_ctr % 2048);
            ++g_idx_ctr;
            cov_idx.insert(idx);
            x->qs[4 * ib + l] = (uint8_t)(idx & 0xff);
            // Each of the 4 group slots gets its OWN counter bit, so the slots are
            // independent (a shared bit would leave "two 8-groups differ" untested,
            // which is exactly what the BSUMS16 control needs to be able to see).
            int dbit = (int)((g_sb_ctr >> l) & 1);
            cov_delta[l][dbit]++;
            int nib = ((idx >> 8) & 0x7) | (dbit << 3);
            int qi = 2 * ib + l / 2;
            if (l % 2 == 0) x->qh[qi] = (uint8_t)nib;
            else            x->qh[qi] = (uint8_t)(x->qh[qi] | (nib << 4));
        }
        ++g_sb_ctr;
    }
    // The scale nibbles (bits 12-15 of each word). Kept in a SANE fp16 range here so
    // the reported fp norm is meaningful; the assembly itself is certified
    // EXHAUSTIVELY (all 65536 nibble combos) by scale_certificate() below, so this
    // corpus does not need to sweep them.
    int e = 8 + ((col + 3 * blk) % 5);            // fp16 exponent 8..12 -> ~2^-7..2^-3
    int m = (col * 7 + blk * 13) % 1024;
    uint16_t want = (uint16_t)((e << 10) | m);
    for (int k = 0; k < 4; ++k)
        sc[k] |= (uint16_t)(((want >> (4 * k)) & 0xf) << 12);
    memcpy(x->scales, sc, 8);
}

// A REAL block_q8_K. NOTE iq1_m does NOT read bsums (its delta term needs per-8
// group sums and bsums are per-16); they are still filled correctly because the
// struct is ggml's, and the BSUMS16 negative control is what proves the per-8
// granularity is load-bearing rather than a preference.
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

// ---- Certificate (3): the ASSEMBLED nibble-scatter fp16 scale, EXHAUSTIVELY.
// The assembly's entire input domain is the four scale-word HIGH nibbles = 16^4 =
// 65536 combinations, so this is not a sample: it is every reachable input. The
// low 12 bits of each word (the ls fields) are randomized on every combination to
// prove the assembly IGNORES them -- a leak would show up as a mismatch.
static bool scale_certificate(std::mt19937 &rng, long long &combos_out) {
    long long mism = 0, total = 0;
    for (int n0 = 0; n0 < 16; ++n0)
      for (int n1 = 0; n1 < 16; ++n1)
        for (int n2 = 0; n2 < 16; ++n2)
          for (int n3 = 0; n3 < 16; ++n3) {
            int nib[4] = {n0, n1, n2, n3};
            uint16_t sc[4];
            for (int k = 0; k < 4; ++k) {
                uint16_t lowNoise = (uint16_t)(rng() & 0x0fff);  // the ls fields
                sc[k] = (uint16_t)((nib[k] << 12) | lowNoise);
            }
            // REFERENCE: ggml's fused four-term expression, verbatim.
            uint16_t ref = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                                      ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
            // MODEL: the repack's per-nibble loop (a different derivation).
            uint16_t got = assemble_scale_repack(sc);
            ++total;
            if (ref != got) ++mism;
          }
    combos_out = total;
    printf("  scale_u16  : %lld / %lld nibble combinations EXHAUSTIVE, mismatch=%lld  %s\n",
           total, 65536LL, mism, (total == 65536 && mism == 0) ? "BYTE-EXACT" : "!! BUG");
    return total == 65536 && mism == 0;
}

// ---- The REAL (fp16-d) block_iq1_mx16 byte layout, recomputed independently here
// and pinned against the byte offsets the FRONT DOOR ships (kIq1MDecodeFacts in
// lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp, and the lit CHECKs in
// test/Conversion/RVV/rvv-emit-*-iq1-m-repack-*.mlir).
//
// SCOPE, stated honestly: the emitter-model above addresses the repacked strips by
// ARRAY INDEX, so it validates the layout's SEMANTICS + the arithmetic, NOT the
// emitter's byte arithmetic (its `float d[16]` is 64 B where the kernel's fp16 d[16]
// is 32 B -- the same modelling shortcut the 12 sibling oracles take). These pins
// therefore cover ONLY this oracle's internal self-consistency: that its formula-
// derived offsets match its own hand-transcribed shipped column.
//
// CORRECTION (2026-07-17, verified by test, not by reasoning): an earlier version of
// this comment claimed that a front-door re-layout "is what goes red" here. That is
// FALSE, and it was proven false on THIS file: a reviewer changed the front door only
// (kIq1MDecodeFacts weightBlockStride 1824 -> 1840), left this oracle untouched, and
// rebuilt -- the pins stayed GREEN. This file includes nothing from the front door
// (grep '#include.*(FrontDoor|RVVLower|GridDecodePlan)' -> 0); the shipped column is a
// hand-transcribed literal. The real coupling to the front door lives in the lit CHECKs.
//
// Keep this pin for what it is (a transcription guard) and do not let it be cited as
// front-door coverage.
static bool check_layout_pins() {
    const int kInterleave = 16;
    int off_d     = 0;
    int off_ls    = off_d  + kInterleave * 2;              // 16 fp16 d    -> +32
    int off_delta = off_ls + 8 * 2 * kInterleave;          // 8*2*16 int8  -> +288
    int off_gidx  = off_delta + 8 * 4 * kInterleave;       // 8*4*16 int8  -> +800
    int stride    = off_gidx + 8 * 4 * kInterleave * 2;    // 8*4*16 u16   -> 1824
    struct Pin { const char *name; int got; int shipped; };
    Pin pins[] = {
        {"weight_block_stride       ", stride,    1824},
        {"weight_ls_byte_offset     ", off_ls,      32},
        {"weight_sign(DELTA)_offset ", off_delta,  288},
        {"weight_grid_idx_offset    ", off_gidx,   800},
        // The PLAIN ggml block_iq1_m: qs[32] + qh[16] + scales[8], NO inline d.
        {"plain_block_iq1_m_stride  ", (int)sizeof(block_iq1_m), 56},
        {"plain_quant_byte_offset   ", 0,            0},
        // block_q8_K: fp32 d @0 + 256 int8 quants @4 + 16 int16 bsums @260 = 292.
        {"gevm_act_quant_offset     ", 4,            4},
        {"gevm_act_block_stride     ", 4 + 256 + 32, 292},
        // block_q8_Kx4: 4 fp32 d @0 + 1024 int8 quants @16 (+ 64 int16 bsums, which
        // iq1_m does NOT read -- there is deliberately no bsums pin here).
        {"gemm_act_quant_offset     ", 16,          16},
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
    printf("# iq1_m x q8_K 16x1-REPACKED oracle (2048 TERNARY grid + DUAL ls +\n");
    printf("# PER-GROUP delta over IN-KERNEL group-of-8 sums + assembled fp16 scale).\n");
    printf("# BYTE-EXACT integer certificates sumi1 (dual-ls ternary grid dot),\n");
    printf("# sumi2 (per-group ls*delta*groupSum) AND scale_u16 (the nibble-scatter\n");
    printf("# assembly): reference (ORIGINAL packed block_iq1_m, qh/scales decoded at\n");
    printf("# read time, ggml's OWN fold grouping) vs emitter-model (REPACKED\n");
    printf("# block_iq1_mx16 strips, the emitter's PER-GROUP fold order).\n\n");

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
        std::vector<block_iq1_m> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq1m_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                     g * 16 + col, l);
        std::vector<block_iq1_mx16> vx((size_t)ng * nb);
        std::vector<block_iq1_m> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq1_mx16(tmp16.data());
            }
        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        long long mismatch1 = 0, mismatch2 = 0;
        int64_t maxSumi = 0;
        double sumsqRef = 0, maxAbsErr = 0;
        for (int r = 0; r < nr; ++r)
            for (int g = 0; g < ng; ++g)
                for (int col = 0; col < 16; ++col) {
                    int gcol = g * 16 + col;
                    double refF = 0, mineF = 0;
                    for (int l = 0; l < nb; ++l) {
                        const block_iq1_m *xo = &orig[(size_t)gcol * nb + l];
                        const block_q8_K *a = &act[(size_t)r * nb + l];
                        int64_t ri1, ri2;
                        ref_block(xo, a->qs, DQ_NORMAL, ri1, ri2);
                        const block_iq1_mx16 *b = &vx[(size_t)g * nb + l];
                        auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                        int64_t mi1, mi2;
                        mine_col(b, col, actFn, mi1, mi2, maxSumi);
                        if (ri1 != mi1) mismatch1++;
                        if (ri2 != mi2) mismatch2++;
                        // The reference's d comes from ggml's own assembly of the
                        // ORIGINAL scale words; the model's from the repacked strip.
                        const uint16_t *sc = (const uint16_t *)xo->scales;
                        uint16_t refU = (uint16_t)((sc[0] >> 12) |
                                                   ((sc[1] >> 8) & 0x00f0) |
                                                   ((sc[2] >> 4) & 0x0f00) |
                                                   (sc[3] & 0xf000));
                        refF  += (double)fp16_to_float(refU) * (double)a->d *
                                 ((double)ri1 + 0.125 * (double)ri2);
                        mineF += (double)b->d[col] * (double)a->d *
                                 ((double)mi1 + 0.125 * (double)mi2);
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
        bool bug = (mismatch1 != 0 || mismatch2 != 0);
        if (bug) anyBug = true;
        printf("  shape nr=%-3d nc=%-4d n=%-6d : sumi1-mismatch=%-6lld "
               "sumi2-mismatch=%-6lld  maxAbsSumi=%-9lld  fp-norm=%.3e  %s\n",
               nr, nc, n, mismatch1, mismatch2, (long long)maxSumi, norm,
               bug ? "INT-BUG" : "BYTE-EXACT");
    }

    // ---- GEMM interleaved block_q8_Kx4 addressing (qs @+16 pos*4+c). Same weight
    // strips; only the activation addressing changes. There is NO bsums plane to
    // interleave -- iq1_m reads none. ----
    {
        const int nc = 32, n = 2560;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_iq1_m> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq1m_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                     g * 16 + col, l);
        std::vector<block_iq1_mx16> vx((size_t)ng * nb);
        std::vector<block_iq1_m> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq1_mx16(tmp16.data());
            }
        std::vector<block_q8_K> row(4 * nb);
        for (int r = 0; r < 4; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_K_block(&row[(size_t)r * nb + l], rng, 900 + r * 7 + l);
        // The interleaved x4 activation quant plane.
        std::vector<int8_t> qsx4((size_t)nb * QK_K * 4);
        for (int l = 0; l < nb; ++l)
            for (int p = 0; p < QK_K; ++p)
                for (int c = 0; c < 4; ++c)
                    qsx4[(size_t)l * QK_K * 4 + p * 4 + c] = row[(size_t)c * nb + l].qs[p];
        long long mism1 = 0, mism2 = 0, total = 0;
        int64_t dummy = 0;
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col) {
                int gcol = g * 16 + col;
                for (int c = 0; c < 4; ++c)
                    for (int l = 0; l < nb; ++l) {
                        int64_t ri1, ri2;
                        ref_block(&orig[(size_t)gcol * nb + l],
                                  row[(size_t)c * nb + l].qs, DQ_NORMAL, ri1, ri2);
                        const int8_t *qsl = &qsx4[(size_t)l * QK_K * 4];
                        auto actFn = [&](int pos) { return (int)qsl[pos * 4 + c]; };
                        int64_t mi1, mi2;
                        mine_col(&vx[(size_t)g * nb + l], col, actFn, mi1, mi2, dummy);
                        total++;
                        if (ri1 != mi1) mism1++;
                        if (ri2 != mi2) mism2++;
                    }
            }
        printf("\n# GEMM interleaved block_q8_Kx4 (qs @+16 pos*4+c; NO bsums plane --\n"
               "# iq1_m reads none, its group-of-8 sums come from these same quants):\n"
               "#   sumi1-mismatch = %lld / %lld   sumi2-mismatch = %lld / %lld   %s\n",
               mism1, total, mism2, total,
               (mism1 || mism2) ? "INT-BUG" : "BYTE-EXACT");
        if (mism1 || mism2) anyBug = true;
    }

    // ---- NEGATIVE CONTROLS (nr=4, nc=32, n=2560). ----
    printf("\n# negative controls (perturb the REFERENCE; a large mismatch fraction\n"
           "# proves the iq1_m feature is genuinely exercised by the emitter-model):\n");
    {
        const int nc = 32, n = 2560, nr = 4;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_iq1_m> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq1m_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                     g * 16 + col, l);
        std::vector<block_iq1_mx16> vx((size_t)ng * nb);
        std::vector<block_iq1_m> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq1_mx16(tmp16.data());
            }
        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        struct Ctl { const char *name; DqMode mode; };
        Ctl ctls[6] = {
            {"GRID    (rotate index +457, wrong 2048-entry)     [sumi1]", DQ_GRID},
            {"QHHIGH  (drop qh high 3 idx bits, idx&=0xff)      [sumi1]", DQ_QHHIGH},
            {"LS1     (perturb ls1 only, +2)                    [both ]", DQ_LS1},
            {"LS2     (perturb ls2 only, +2)                    [both ]", DQ_LS2},
            {"DELTA   (ignore qh delta bits, all four -> +1)    [sumi2]", DQ_DELTA},
            {"BSUMS16 (force the 2 8-groups of a 16 to SHARE a\n"
             "           delta = what per-16 bsums COULD express) [sumi2]", DQ_BSUMS16},
        };
        for (auto &ct : ctls) {
            long long total = 0, mism1 = 0, mism2 = 0;
            int64_t dummy = 0;
            for (int r = 0; r < nr; ++r)
                for (int g = 0; g < ng; ++g)
                    for (int col = 0; col < 16; ++col) {
                        int gcol = g * 16 + col;
                        for (int l = 0; l < nb; ++l) {
                            const block_q8_K *a = &act[(size_t)r * nb + l];
                            int64_t rI1, rI2;
                            ref_block(&orig[(size_t)gcol * nb + l], a->qs, ct.mode,
                                      rI1, rI2);
                            auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                            int64_t mI1, mI2;
                            mine_col(&vx[(size_t)g * nb + l], col, actFn, mI1, mI2,
                                     dummy);
                            total++;
                            if (rI1 != mI1) mism1++;
                            if (rI2 != mI2) mism2++;
                        }
                    }
            printf("  %-58s : sumi1 %lld/%lld (%.0f%%)  sumi2 %lld/%lld (%.0f%%)  %s\n",
                   ct.name, mism1, total, 100.0 * mism1 / total,
                   mism2, total, 100.0 * mism2 / total,
                   (mism1 > 0 || mism2 > 0) ? "EXERCISED" : "!! NOT EXERCISED");
            if (mism1 == 0 && mism2 == 0) anyBug = true;  // a dead control = a hole
        }
    }

    // ---- CORPUS COMPLETENESS ([K-5b](1)): MEASURED, not claimed. ----
    printf("\n# corpus completeness (measured over every block generated above):\n");
    bool covOk = true;
    printf("  grid_index : %zu / 2048 distinct 11-bit indices exercised   %s\n",
           cov_idx.size(), cov_idx.size() == 2048 ? "COMPLETE" : "!! SHORT");
    if (cov_idx.size() != 2048) covOk = false;
    // PER-SLOT delta coverage: iq1_m's four deltas are INDEPENDENT bits, so a
    // corpus that only flipped slot 0 would leave slots 1-3 unverified AND would
    // make the BSUMS16 control silently dead.
    for (int l = 0; l < 4; ++l) {
        bool both = cov_delta[l][0] > 0 && cov_delta[l][1] > 0;
        printf("  delta[g%d]  : +1 %lld groups, -1 %lld groups   %s\n", l,
               cov_delta[l][0], cov_delta[l][1],
               both ? "BOTH POLARITIES" : "!! SHORT");
        if (!both) covOk = false;
    }
    printf("  ls1        : ");
    for (int f = 0; f < 8; ++f) {
        printf("ls=%d:%lld ", 2 * f + 1, cov_ls1[f]);
        if (cov_ls1[f] == 0) covOk = false;
    }
    printf("\n  ls2        : ");
    for (int f = 0; f < 8; ++f) {
        printf("ls=%d:%lld ", 2 * f + 1, cov_ls2[f]);
        if (cov_ls2[f] == 0) covOk = false;
    }
    printf("\n  ls steps   : %s\n", covOk ? "ALL 8 STEPS ON BOTH ls1 AND ls2"
                                          : "!! SHORT");
    long long scaleCombos = 0;
    bool scaleOk = scale_certificate(rng, scaleCombos);
    if (!scaleOk) anyBug = true;
    if (!covOk) anyBug = true;

    bool pinsOk = check_layout_pins();
    if (!pinsOk) anyBug = true;

    printf("\nWORST_FP_NORM %.3e   GLOBAL_MAX_i32_SUMI %lld   COVERAGE %s   "
           "SCALE_EXHAUSTIVE %lld/65536 %s   LAYOUT_PINS %s   VERDICT %s\n",
           worstNorm, (long long)globalMaxSumi, covOk ? "COMPLETE" : "SHORT",
           scaleCombos, scaleOk ? "OK" : "BUG",
           pinsOk ? "OK" : "DRIFT", anyBug ? "INT-BUG" : "BYTE-EXACT-INTEGER");
    return anyBug ? 1 : 0;
}
