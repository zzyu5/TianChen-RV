// iq3_s x q8_K 16x1-REPACKED GEVM + GEMM numeric oracle (512-entry UINT32 grid codebook +
// EXPLICIT per-group sign plane + single per-sub-block ls scale, NO store constant).
//
// Gate: does the compiler-emitted iq3_s repack kernel compute the CORRECT super-block
// DUAL-ENTRY grid dequant-matmul result?
//
// WHAT MAKES THIS ROW DIFFERENT, and therefore what this oracle is really guarding. iq3_s
// is the LAST grid sibling and the one that resembles iq3_xxs closely enough to ride its
// whole emitter leaf -- same `uint32_t` grid, so one entry supplies only 4 of an 8-element
// group's bytes and ggml reads TWO per group:
//     grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32] << (8-2*l)) & 256)));
//     grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32] << (7-2*l)) & 256)));
//     for (j = 0; j < 4; ++j) { sumi += grid1[j]*q8[j+0]*sign(j+0);
//                               sumi += grid2[j]*q8[j+4]*sign(j+4); }
// But on THREE facts it is NOT iq3_xxs, and each is a way a "close enough" kernel could be
// silently wrong. Each therefore gets a coverage axis AND a negative control:
//   (a) the index is 9 BITS, not 8: the low 8 come from qs and bit 8 from a qh bit. A
//       kernel that dropped the qh bit would read entry idx&255 -- a real grid entry, so
//       real-looking numbers.                               (COVERAGE grid_index / NC QHBIT)
//   (b) the signs are EXPLICIT bytes carried by the block and tested with kmask_iq2xs.
//       There is NO ksigns_iq2xs selector indirection (that is iq3_xxs's mechanism). A
//       kernel that truncated the byte to 7 bits, as a signs64 plane would, gets lane 7's
//       sign wrong and nothing else.                          (COVERAGE signs / NC SIGN7BIT)
//   (c) ls is SINGLE per 32-element sub-block. ggml names ls1/ls2 per iteration but steps
//       ib32 += 2 and spends one on EACH sub-block; two sub-blocks share a scales byte.
//       Reading it as iq2_s-style dual-ls-within-a-sub-block would apply the NEIGHBOUR
//       sub-block's scale to half the lanes.                       (COVERAGE ls / NC LSPAIR)
// A kernel that (mis)lowered this row onto the single-base leaf would also still produce
// numbers -- see the PAIR coverage axis and the SINGLEBASE negative control.
//
// ---------------------------------------------------------------------------
// [K-5b] THREE REQUIREMENTS, and how this oracle meets each:
//
// (3) ORACLE INDEPENDENT / NO SHARED DECODE IMPLEMENTATION.
//     The REFERENCE decodes the ORIGINAL (pre-repack) packed block_iq3_s exactly as ggml's
//     canonical ggml_vec_dot_iq3_s_q8_K does: it walks qs/qh/signs/scales as four separate
//     planes, ASSEMBLES the 9-bit index at read time with ggml's own
//     `qs[2*l+0] | ((qh[ib32] << (8-2*l)) & 256)` shift form, derives ls from a scales
//     NIBBLE, tests the raw sign byte with `signs[l] & kmask_iq2xs[j]`, and fetches grid
//     bytes by REINTERPRETING the uint32 entry ((const uint8_t *)(iq3s_grid + idx)) --
//     which is ggml's own read.
//     The EMITTER-MODEL never sees a qh bit or a nibble: it reads the REPACKED
//     block_iq3_sx16, where the ASSEMBLED 9-bit index / ls / sign byte are already separate
//     flat per-column strips, and gathers grid bytes from a FLAT int8 table by
//     grid_bytes[idx*4+j] against a PRE-EXPANDED signs256[b*8+j] +-1 plane (the two
//     vluxei16 gathers the emitter emits). Different layout, different index derivation
//     (assembled-at-repack vs assembled-at-read), different table read, different sign
//     representation (ggml tests a bit with kmask; the model multiplies by a materialized
//     +-1 byte). The ONLY shared things are iq3s_grid / kmask_iq2xs themselves -- the
//     FORMAT CONSTANTS, not an implementation (see iq3s_tables.h; both machine-verified
//     equal to ggml's tables, and the signs256 plane is not shared at all -- it is not a
//     ggml table, it is DERIVED independently on each side).
//
// (2) INPUT PATH SAME-SOURCE. Both paths are fed from the SAME generated block_iq3_s array:
//     the repacked strips are produced from it by make_block_iq3_sx16 (the mat-quant the
//     repack stage performs). The reference reads the originals; the model reads the repack
//     of those same originals. Nothing is captured from the model and replayed.
//
// (1) CORPUS COMPLETENESS. Every counter is incremented IN A DECODE PATH, at the point of
//     USE -- never in the generator, which counts what the generator INTENDED rather than
//     what the decode consumed (the C4a-3 defect C4a-4 fixed and this file inherits). There
//     are TWO INDEPENDENT SETS: [R] filled by the REFERENCE as it reads the packed block,
//     [M] filled by the EMITTER-MODEL as it reads the repacked strips. Both must be complete
//     AND they must AGREE, so a pack/repack fault that silently narrowed a range shows up as
//     a SHORT axis or a REF/MODEL DISAGREEMENT rather than a green "COMPLETE". Axes (each
//     exits non-zero if short):
//       * all 512 grid index values, MEASURED SEPARATELY AT BOTH ENTRY SLOTS
//                                                       (COVERAGE grid_index e0/e1)
//         -- 512, not 256: this axis is what proves the qh bit reached the decode. A kernel
//         that dropped it could only ever show 256 distinct indices.
//       * BOTH values of the qh bit, at BOTH entry slots            (COVERAGE qh_bit e0/e1)
//       * all 256 explicit sign byte values                              (COVERAGE sign_byte)
//         -- 256, not 128: a signs64-style 7-bit truncation cannot reach the upper half.
//       * all 16 ls steps {1,3,...,31}                                          (COVERAGE ls)
//       * ls PAIR-DISTINGUISHING: sub-blocks whose ls differs from its scales-byte partner's.
//         Without these the corpus could not tell SINGLE ls from a dual-ls misread, since
//         both nibbles would carry the same value. A coverage REQUIREMENT, not a statistic.
//                                                             (COVERAGE ls_pair_distinguishing)
//       * PAIR: groups whose two entry indices are DISTINGUISHING, i.e. idx1 !=
//         (idx0+1)&511 -- the exact condition under which a single-base mis-lowering would
//         read different bytes. A corpus without these could not tell the dual-entry leaf
//         from a broken one, so this too is a REQUIREMENT.      (COVERAGE pair_distinguishing)
//
// ---------------------------------------------------------------------------
// INTEGER CERTIFICATES -- how many does iq3_s actually have?
// Counted from ggml_vec_dot_iq3_s_q8_K rather than assumed: it carries ONE integer
// accumulator chain. `sumi` accumulates the signed grid dot within a sub-block and is then
// FOLDED AWAY by `bsum += sumi*ls`; `sumf += d*bsum` is float and `*s = sumf` is the store.
// So there is exactly ONE end-of-block integer quantity: bsum. (Same as iq3_xxs. Unlike
// iq1_m, which has three -- two accumulators plus an assembled scale word -- because it has
// a delta term and a scattered fp16 d. iq3_s has neither: no delta, no second accumulator,
// no bsums plane, and an ordinary inline fp16 d.)
// We compare bsum BYTE-EXACT (int64 equality), and ADDITIONALLY certify the ls-UNWEIGHTED
// sub-block dot total `sumi_raw = sum_ib sumi_ib`. That second quantity is not in ggml; it is
// here for FAULT SEPARATION -- bsum alone lets an ls fault and a grid fault alias (both move
// one number), whereas an ls fault moves bsum and leaves sumi_raw fixed. Both are pure
// integers (no fp reassociation).
//
// THE FP STORE IS REPORTED AS A NORM ONLY -- and for this row that is worth stating plainly,
// because the emitted kernel does something ggml does not. ggml ends `*s = sumf`; the
// emitter multiplies by the literal 1.0f (see GridDecodePlan::storeScaleLiteral for why the
// constant is carried as 1.0f rather than as an absent field). This oracle exercises that
// difference rather than asserting it away: the REFERENCE store below is ggml's bare
// `sumf`, the emitter-model's applies the 1.0f, and the reported norm is the measured
// disagreement between them. What that norm is NOT is a byte-exact certificate -- the
// byte-exact certificates here are the integers, exactly as they are for iq3_xxs's 0.25f.
//
// NEGATIVE CONTROLS perturb the REFERENCE; a mismatch SPIKE proves the corresponding iq3_s
// structure was genuinely exercised by the emitter-model:
//   GRID       : the REFERENCE fetches a ROTATED grid entry (idx -> (idx+37)&511).
//   SIGNS      : the REFERENCE ignores the sign plane (all signs +1).
//   LS         : the REFERENCE perturbs the per-sub-block ls (+2).
//   SINGLEBASE : the REFERENCE emulates the SINGLE-BASE mis-lowering the dual-entry leaf
//                exists to avoid -- it reads lanes 4-7 from grid1's bytes 4-7 (i.e. entry
//                idx0+1) instead of from grid2's bytes 0-3, exactly what a `for j<8` against
//                one base would do.
//   QHBIT      : the REFERENCE drops the qh 9th bit (idx &= 255), the iq3_xxs-shaped misread
//                -- proving the 9-bit assembly is exercised and that a 256-entry read is not
//                secretly equivalent.
//   SIGN7BIT   : the REFERENCE masks the sign byte to 7 bits (signs[l] & 127), the
//                signs64-shaped misread -- proving the EXPLICIT 8-bit sign plane is
//                exercised and that iq3_xxs's 128-selector plane would NOT do.
//   LSPAIR     : the REFERENCE reads the PARTNER sub-block's ls nibble (the dual-ls misread
//                of a Single row) -- proving the ls granularity is exercised.
// GRID/SIGNS/SINGLEBASE/QHBIT/SIGN7BIT spike sumi_raw and bsum; LS and LSPAIR spike bsum
// ONLY, which is exactly right -- ls is not part of the unweighted dot.
//
// Build: g++ -O2 -std=c++17 oracle_repack_iq3_s.cpp -o /tmp/oracle_iq3s && /tmp/oracle_iq3s

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <set>
#include <random>
#include "iq3s_tables.h"

#define QK_K 256

// ---- flat int8 grid byte table (512*4), EXACTLY the table the emitter gathers from
// (weft_iq3s_grid viewed through a (const int8_t *) cast). Built by shift extraction; the
// REFERENCE does NOT use this -- it reinterprets the uint32 entry. Note 4 bytes per entry,
// not 8: that single fact is what forces the dual-entry leaf. ----
static int8_t grid_bytes[512 * 4];
// ---- the DERIVED signs256 +-1 plane (256 sign-byte values * 8 bytes), EXACTLY what the
// emitter declares as weft_iq2s_signs256 and gathers. The REFERENCE does NOT use this -- it
// tests the raw sign byte with kmask, as ggml does. NOT a ggml table: DERIVED here, and
// derived again (independently) by the emitter's decl. ----
static int8_t signs256[256 * 8];
static void build_tables() {
    for (int e = 0; e < 512; ++e) {
        uint32_t u = iq3s_grid[e];
        for (int j = 0; j < 4; ++j)
            grid_bytes[e * 4 + j] = (int8_t)((u >> (8 * j)) & 0xff);
    }
    for (int v = 0; v < 256; ++v)
        for (int b = 0; b < 8; ++b)
            signs256[v * 8 + b] = ((v >> b) & 1) ? -1 : 1;
}

// ---- original per-super-block block_iq3_s (ggml-common.h), 110 bytes ----
//   d          : fp16 super-block scale (float here; see the layout-pin scope note)
//   qs[64]     : QK_K/4  -- low 8 bits of 64 grid indices (8 per sub-block)
//   qh[8]      : QK_K/32 -- bit 8 of each index: slot e of sub-block ib is (qh[ib]>>e)&1
//   signs[32]  : QK_K/8  -- ONE explicit sign byte per 8-lane group
//   scales[4]  : QK_K/64 -- TWO sub-blocks' ls nibbles per byte
// Unlike block_iq3_xxs there is NO double-duty region: each plane is its own field.
struct block_iq3_s {
    float   d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t signs[QK_K / 8];
    uint8_t scales[QK_K / 64];
};

struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};

// ---- repacked block_iq3_sx16 (block-as-lane convention), 2720 B ----
//   d[16]            @ +0     (32 B, fp16 in the real kernel; float here)
//   ls[8][16]        @ +32    (128 B, int8 SINGLE ls per sub-block, [1,31])
//   gidx[8][8][16]   @ +160   (2048 B, UINT16 assembled 9-bit index -- EIGHT per sub-block,
//                              TWO per 8-lane group. u16, not u8: 511 does not fit a byte.
//                              This is the ONE strip that differs from iq3_xxs's layout.)
//   signs[8][4][16]  @ +2208  (512 B, uint8 EXPLICIT sign byte -- FOUR per sub-block, ONE
//                              per group)
struct block_iq3_sx16 {
    float    d[16];
    int8_t   ls[8][16];
    uint16_t gidx[8][8][16];
    uint8_t  signs[8][4][16];
};

// The REPACK (mat-quant): ASSEMBLE the 9-bit index and SPLIT the scales nibble ONCE into
// flat strips. This is the layout transform the repack stage performs; the kernel then never
// sees a qh bit or a nibble.
static block_iq3_sx16 make_block_iq3_sx16(const block_iq3_s *in) {
    block_iq3_sx16 out;
    for (int c = 0; c < 16; ++c) out.d[c] = in[c].d;
    for (int c = 0; c < 16; ++c) {
        const uint8_t *qs = in[c].qs;
        for (int ib = 0; ib < 8; ++ib) {
            // SINGLE ls per sub-block: two sub-blocks share a scales byte (low nibble ->
            // even sub-block, high nibble -> odd).
            unsigned nib = (in[c].scales[ib >> 1] >> (4 * (ib & 1))) & 0xf;
            out.ls[ib][c] = (int8_t)(2 * nib + 1);
            for (int e = 0; e < 8; ++e) {
                unsigned lo  = qs[8 * ib + e];
                unsigned hi  = (in[c].qh[ib] >> e) & 1u;
                out.gidx[ib][e][c] = (uint16_t)(lo | (hi << 8));
            }
            for (int l = 0; l < 4; ++l)
                out.signs[ib][l][c] = in[c].signs[4 * ib + l];
        }
    }
    return out;
}

enum DqMode { DQ_NORMAL = 0, DQ_GRID = 1, DQ_SIGNS = 2, DQ_LS = 3, DQ_SINGLEBASE = 4,
              DQ_QHBIT = 5, DQ_SIGN7BIT = 6, DQ_LSPAIR = 7 };

// ---- CORPUS COVERAGE instrumentation (the [K-5b](1) evidence).
// TWO INDEPENDENT SETS, both filled IN THE DECODE PATH AT POINT OF USE -- never in the
// generator. [R] = what the REFERENCE consumed from the PACKED block; [M] = what the
// EMITTER-MODEL consumed from the REPACKED strips. ----
struct Cov {
    std::set<int> idx_e0, idx_e1;   // 9-bit grid indices, PER ENTRY SLOT
    std::set<int> qh_e0, qh_e1;     // the 9th bit's values, PER ENTRY SLOT
    std::set<int> sgn;              // explicit 8-bit sign byte values
    std::set<int> ls;               // ls steps actually used
    long long pair_distinguishing = 0;    // groups where idx1 != (idx0+1)&511
    long long pair_total = 0;
    long long ls_pair_distinguishing = 0; // sub-blocks whose ls != partner's ls
    long long ls_pair_total = 0;
};
static Cov covR, covM;

// ---- REFERENCE (ggml canonical ggml_vec_dot_iq3_s_q8_K), reading the ORIGINAL packed
// block: four separate planes, the 9-bit index ASSEMBLED at read time with ggml's own shift
// form, ls from a scales nibble, the raw sign byte tested with kmask, and the grid read
// through the uint32 reinterpret -- exactly as ggml does. ----
static void ref_block(const block_iq3_s *x, const int8_t *q8, DqMode mode,
                      int64_t &sumi_raw_out, int64_t &bsum_out, bool count) {
    int64_t sumi_raw = 0, bsum = 0;
    const uint8_t *qs    = x->qs;
    const uint8_t *signs = x->signs;
    int q8pos = 0;
    // ggml's loop steps ib32 += 2 and reads BOTH nibbles of scales[ib32/2], spending ls1 on
    // sub-block ib32 and ls2 on ib32+1. Written out here the same way, so the SINGLE-ls
    // fact is visible rather than assumed.
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
        const int lsv[2] = {2 * (x->scales[ib32 / 2] & 0xf) + 1,
                            2 * (x->scales[ib32 / 2] >> 4) + 1};
        if (count) {
            covR.ls_pair_total++;
            if (lsv[0] != lsv[1]) covR.ls_pair_distinguishing++;
        }
        for (int half = 0; half < 2; ++half) {
            int ib = ib32 + half;
            int64_t ls = lsv[half];
            if (count) covR.ls.insert((int)ls);
            if (mode == DQ_LS)     ls += 2;
            if (mode == DQ_LSPAIR) ls = lsv[1 - half]; // the dual-ls misread of a Single row
            int64_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                // ggml's own 9-bit assembly, verbatim in shift form.
                int i0 = qs[2 * l + 0] | ((x->qh[ib] << (8 - 2 * l)) & 256);
                int i1 = qs[2 * l + 1] | ((x->qh[ib] << (7 - 2 * l)) & 256);
                if (count) {
                    covR.idx_e0.insert(i0);
                    covR.idx_e1.insert(i1);
                    covR.qh_e0.insert(i0 >> 8);
                    covR.qh_e1.insert(i1 >> 8);
                    covR.pair_total++;
                    if (i1 != ((i0 + 1) & 511)) covR.pair_distinguishing++;
                }
                if (mode == DQ_QHBIT) { i0 &= 255; i1 &= 255; }
                if (mode == DQ_GRID)  { i0 = (i0 + 37) & 511; i1 = (i1 + 37) & 511; }
                unsigned sb = signs[l];
                if (count && mode == DQ_NORMAL) covR.sgn.insert((int)sb);
                if (mode == DQ_SIGN7BIT) sb &= 127; // the signs64-shaped truncation
                const uint8_t *grid1 = (const uint8_t *)(iq3s_grid + i0);
                const uint8_t *grid2 = (const uint8_t *)(iq3s_grid + i1);
                for (int j = 0; j < 4; ++j) {
                    int s0 = (mode == DQ_SIGNS) ? 1 : ((sb & kmask_iq2xs[j + 0]) ? -1 : 1);
                    int s1 = (mode == DQ_SIGNS) ? 1 : ((sb & kmask_iq2xs[j + 4]) ? -1 : 1);
                    // DQ_SINGLEBASE: lanes 4-7 read grid1's bytes 4-7 (= entry i0+1's bytes
                    // 0-3) instead of grid2's 0-3.
                    int g2 = (mode == DQ_SINGLEBASE) ? grid1[j + 4] : grid2[j];
                    sumi += (int64_t)grid1[j] * q8[q8pos + j + 0] * s0;
                    sumi += (int64_t)g2       * q8[q8pos + j + 4] * s1;
                }
                q8pos += 8;
            }
            qs    += 8;
            signs += 4;
            sumi_raw += sumi;
            bsum     += sumi * ls;
        }
    }
    sumi_raw_out = sumi_raw;
    bsum_out = bsum;
}

// ---- EMITTER-MODEL for column c: reproduce the block-as-lane organization (u16 grid index
// -> flat int8 grid GATHER at idx*4+j, u8 sign byte -> signs256 GATHER at b*8+j, vmul sign
// fold onto the grid byte, ls-weighted i32 dot). This is the DUAL-BASE nest: entry e = j/4,
// grid byte j%4. ----
template <typename ActFn>
static void mine_col(const block_iq3_sx16 *b, int c, ActFn act,
                     int64_t &sumi_raw_out, int64_t &bsum_out,
                     int64_t &maxAbsBsum, bool count) {
    int64_t sumi_raw = 0, bsum = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int64_t sub = 0;
        int lsv = b->ls[ib][c];
        if (count) {
            covM.ls.insert(lsv);
            if ((ib & 1) == 0) {
                covM.ls_pair_total++;
                if (lsv != b->ls[ib + 1][c]) covM.ls_pair_distinguishing++;
            }
        }
        for (int grp = 0; grp < 4; ++grp) {
            int i0 = b->gidx[ib][2 * grp + 0][c];
            int i1 = b->gidx[ib][2 * grp + 1][c];
            int sb = b->signs[ib][grp][c];
            if (count) {
                covM.idx_e0.insert(i0);
                covM.idx_e1.insert(i1);
                covM.qh_e0.insert(i0 >> 8);
                covM.qh_e1.insert(i1 >> 8);
                covM.sgn.insert(sb);
                covM.pair_total++;
                if (i1 != ((i0 + 1) & 511)) covM.pair_distinguishing++;
            }
            for (int j = 0; j < 8; ++j) {
                int k = ib * 32 + grp * 8 + j;
                int e   = j / 4;                    // THE ACTIVATION-RANGE SPLIT
                int gj  = j % 4;
                int idx = e ? i1 : i0;
                int gbv = grid_bytes[idx * 4 + gj]; // REAL grid GATHER (idx*4, not *8)
                int sgn = signs256[sb * 8 + j];     // REAL sign GATHER (byte*8, 256-entry)
                sub += (int64_t)act(k) * (gbv * sgn);
            }
        }
        sumi_raw += sub;
        bsum += sub * (int64_t)lsv;
        if (std::llabs(bsum) > maxAbsBsum) maxAbsBsum = std::llabs(bsum);
    }
    sumi_raw_out = sumi_raw;
    bsum_out = bsum;
}

// The generator SWEEPS the decode axes systematically so coverage is structural, not luck --
// but NOTHING here is counted as coverage: these counters are only cycle drivers. Coverage is
// MEASURED in the decode paths above, which is the whole point (a generator that intends 512
// indices but packs them wrong must not be able to report 512). The two entry slots are
// driven by DIFFERENT strides so idx1 is generally != idx0+1; the qh bit is driven per slot;
// the ls nibbles run a stride-3 cycle over [0,15] (3 coprime with 16 => all steps) and are
// driven so that a sub-block and its scales-byte partner usually DIFFER; the sign byte runs a
// stride-7 cycle over [0,255] (7 coprime with 256 => all 256). Activation magnitudes stay
// random.
static long long g_i0 = 0, g_i1 = 40, g_sb = 0, g_sgn = 0;

static void build_iq3s_block(block_iq3_s *x, std::mt19937 &rng, int col, int blk) {
    x->d = 0.010f + 0.0006f * ((col + 3 * blk) % 11);
    for (int ib = 0; ib < 8; ++ib) {
        unsigned qh = 0;
        for (int l = 0; l < 4; ++l) {
            // The two entry slots advance at DIFFERENT rates over the FULL 9-bit range, so
            // the pair is generally distinguishing. The decode-path axes verify it.
            unsigned v0 = (unsigned)(g_i0 % 512); g_i0 += 1;
            unsigned v1 = (unsigned)(g_i1 % 512); g_i1 += 3;
            x->qs[8 * ib + 2 * l + 0] = (uint8_t)(v0 & 255);
            x->qs[8 * ib + 2 * l + 1] = (uint8_t)(v1 & 255);
            qh |= ((v0 >> 8) & 1u) << (2 * l + 0);
            qh |= ((v1 >> 8) & 1u) << (2 * l + 1);
            x->signs[4 * ib + l] = (uint8_t)((g_sgn * 7) % 256);
            ++g_sgn;
        }
        x->qh[ib] = (uint8_t)qh;
        // Two sub-blocks per scales byte; drive the two nibbles independently so the pair is
        // usually distinguishing (the ls_pair axis verifies it).
        unsigned nib = (unsigned)((g_sb * 3) % 16);
        ++g_sb;
        if ((ib & 1) == 0) x->scales[ib >> 1]  = (uint8_t)nib;
        else               x->scales[ib >> 1] |= (uint8_t)(nib << 4);
    }
}

// A REAL block_q8_K. iq3_s does NOT read bsums (no delta term), but a real quantizer fills
// them, so we do too -- and their presence is exactly why the front door must NOT stamp an
// activation_bsums_byte_offset for this row.
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

// ---- The REAL (fp16-d) block_iq3_sx16 byte layout, recomputed independently here and
// pinned against the byte offsets the FRONT DOOR ships (kIq3SDecodeFacts in
// lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp, and the lit CHECKs in
// test/Conversion/RVV/rvv-emit-*-iq3-s-repack-*.mlir).
//
// SCOPE, stated honestly and up front (inherited verbatim from the iq3_xxs sibling, which
// inherited the C4a-3 correction): these pins are a TRANSCRIPTION GUARD, NOT front-door
// coverage. This file #includes NOTHING from the front door -- the "shipped" column below is
// a hand-transcribed literal. Re-laying-out the front door and leaving this file untouched
// leaves these pins GREEN; that was verified by test on the iq1_m sibling
// (weightBlockStride 1824 -> 1840 stayed green), not argued from reasoning. All these pins
// can catch is this oracle's own internal inconsistency: that its formula-derived offsets
// match its own transcribed column.
//
// The real coupling to the front door lives in the lit CHECKs (`CHECK: literal "2720"` /
// `"292"` / `"1168"` / `"2192"`), not here. Do not let this be cited as front-door coverage.
//
// The emitter-model above also addresses the repacked strips by ARRAY INDEX, so it validates
// the layout's SEMANTICS + the arithmetic, NOT the emitter's byte arithmetic (its
// `float d[16]` is 64 B where the kernel's fp16 d[16] is 32 B -- the same modelling shortcut
// all sibling oracles take).
static bool check_layout_pins() {
    const int kInterleave = 16;
    int off_d     = 0;
    int off_ls    = off_d + kInterleave * 2;                  // 16 fp16 d   -> +32
    int off_gidx  = off_ls + 8 * kInterleave;                 // 8*16 int8   -> +160
    int off_signs = off_gidx + 8 * 8 * kInterleave * 2;       // 8*8*16 u16  -> +2208
    int stride    = off_signs + 8 * 4 * kInterleave;          // 8*4*16 u8   -> 2720
    // The LAST grid-index byte offset the dual-entry nest reaches -- the lit files' 2192
    // discriminator, recomputed here from the layout rather than copied from the CHECK.
    int last_gidx = off_gidx + ((7 * 8 + 3 * 2 + 1) * 16 + 1 * 8) * 2;
    struct Pin { const char *name; int got; int shipped; };
    Pin pins[] = {
        {"weight_block_stride       ", stride,     2720},
        {"weight_ls_byte_offset     ", off_ls,       32},
        {"weight_grid_idx_offset    ", off_gidx,    160},
        {"weight_sign_byte_offset   ", off_signs,  2208},
        {"dual_entry_last_gidx_off  ", last_gidx,  2192},
        // block_q8_K: fp32 d @0 + 256 int8 quants @4 + 16 int16 bsums @260 = 292.
        {"gevm_act_quant_offset     ", 4,             4},
        {"gevm_act_block_stride     ", 4 + 256 + 32, 292},
        // block_q8_Kx4: 4 fp32 d @0 + 1024 int8 quants @16 + 64 int16 bsums @1040.
        {"gemm_act_quant_offset     ", 16,           16},
        {"gemm_act_block_stride     ", 16 + 1024 + 128, 1168},
        // The PLAIN block_iq3_s: 2 + 64 + 8 + 32 + 4 = 110 (ggml's static_assert spells it
        // sizeof(ggml_half) + 13*(QK_K/32) + IQ3S_N_SCALE = 2 + 104 + 4).
        {"plain_iq3_s_stride        ", 2 + QK_K/4 + QK_K/32 + QK_K/8 + QK_K/64, 110},
        {"plain_qh_byte_offset      ", 2 + QK_K/4,   66},
        {"plain_signs_byte_offset   ", 2 + QK_K/4 + QK_K/32, 74},
        {"plain_scales_byte_offset  ", 2 + QK_K/4 + QK_K/32 + QK_K/8, 106},
    };
    bool ok = true;
    printf("---- repack layout PINS (transcription guard; NOT front-door coverage) ----\n");
    for (const Pin &p : pins) {
        bool good = p.got == p.shipped;
        if (!good) ok = false;
        printf("  %s derived=%-6d shipped=%-6d %s\n", p.name, p.got, p.shipped,
               good ? "ok" : "!! MISMATCH");
    }
    return ok;
}

static bool report_cov(const char *tag, const Cov &cv) {
    bool ok = true;
    printf("  [%s] grid_index e0: %3zu/512 %-8s   e1: %3zu/512 %-8s\n", tag,
           cv.idx_e0.size(), cv.idx_e0.size() == 512 ? "COMPLETE" : "!! SHORT",
           cv.idx_e1.size(), cv.idx_e1.size() == 512 ? "COMPLETE" : "!! SHORT");
    if (cv.idx_e0.size() != 512 || cv.idx_e1.size() != 512) ok = false;
    printf("  [%s] qh_bit    e0: %3zu/2   %-8s   e1: %3zu/2   %-8s\n", tag,
           cv.qh_e0.size(), cv.qh_e0.size() == 2 ? "COMPLETE" : "!! SHORT",
           cv.qh_e1.size(), cv.qh_e1.size() == 2 ? "COMPLETE" : "!! SHORT");
    if (cv.qh_e0.size() != 2 || cv.qh_e1.size() != 2) ok = false;
    printf("  [%s] sign_byte     : %3zu/256 %s\n", tag, cv.sgn.size(),
           cv.sgn.size() == 256 ? "COMPLETE" : "!! SHORT");
    if (cv.sgn.size() != 256) ok = false;
    printf("  [%s] ls steps      : %3zu/16  %s   {", tag, cv.ls.size(),
           cv.ls.size() == 16 ? "COMPLETE" : "!! SHORT");
    for (int v : cv.ls) printf("%d ", v);
    printf("}\n");
    if (cv.ls.size() != 16) ok = false;
    // The DUAL-ENTRY axis: without distinguishing pairs this corpus could not tell the
    // dual-entry leaf from a single-base mis-lowering, so a shortfall is a FAILURE.
    double pct = cv.pair_total ? 100.0 * (double)cv.pair_distinguishing / (double)cv.pair_total : 0.0;
    printf("  [%s] pair_distinguishing (idx1 != idx0+1): %lld/%lld (%.1f%%) %s\n", tag,
           cv.pair_distinguishing, cv.pair_total, pct,
           cv.pair_distinguishing > 0 ? "PRESENT" : "!! ABSENT (SINGLEBASE undetectable)");
    if (cv.pair_distinguishing == 0) ok = false;
    // The SINGLE-ls axis: without sub-blocks whose ls differs from its scales-byte partner's,
    // a dual-ls misread would read the same number and be invisible.
    double lpct = cv.ls_pair_total ? 100.0 * (double)cv.ls_pair_distinguishing / (double)cv.ls_pair_total : 0.0;
    printf("  [%s] ls_pair_distinguishing (ls != partner): %lld/%lld (%.1f%%) %s\n", tag,
           cv.ls_pair_distinguishing, cv.ls_pair_total, lpct,
           cv.ls_pair_distinguishing > 0 ? "PRESENT" : "!! ABSENT (LSPAIR undetectable)");
    if (cv.ls_pair_distinguishing == 0) ok = false;
    return ok;
}

int main() {
    build_tables();
    std::mt19937 rng(0xC4A5);

    const int NCOLGROUPS = 3;      // 3 * 16 = 48 weight columns
    const int NBLK = 48;           // 48 super-blocks of 256 = 12288 contraction elements
    const int NCOL = NCOLGROUPS * 16;

    std::vector<block_iq3_s> W((size_t)NCOL * NBLK);
    std::vector<block_q8_K>  A(NBLK);
    for (int cg = 0; cg < NCOLGROUPS; ++cg)
        for (int c = 0; c < 16; ++c)
            for (int b = 0; b < NBLK; ++b)
                build_iq3s_block(&W[(size_t)(cg * 16 + c) * NBLK + b], rng, cg * 16 + c, b);
    for (int b = 0; b < NBLK; ++b) build_q8_K_block(&A[b], rng, b);

    // The REPACK of those SAME originals ([K-5b](2) input-path same-source).
    std::vector<block_iq3_sx16> R((size_t)NCOLGROUPS * NBLK);
    for (int cg = 0; cg < NCOLGROUPS; ++cg)
        for (int b = 0; b < NBLK; ++b) {
            block_iq3_s col[16];
            for (int c = 0; c < 16; ++c) col[c] = W[(size_t)(cg * 16 + c) * NBLK + b];
            R[(size_t)cg * NBLK + b] = make_block_iq3_sx16(col);
        }

    struct Res { long long mismatch_raw, mismatch_bsum; double worst_norm; };
    auto run = [&](DqMode mode, bool count) -> Res {
        Res r{0, 0, 0.0};
        int64_t maxAbsBsum = 0;
        for (int cg = 0; cg < NCOLGROUPS; ++cg)
            for (int c = 0; c < 16; ++c) {
                int col = cg * 16 + c;
                double sumf_ref = 0.0, sumf_mine = 0.0;
                for (int b = 0; b < NBLK; ++b) {
                    int64_t rRaw, rB, mRaw, mB;
                    ref_block(&W[(size_t)col * NBLK + b], A[b].qs, mode, rRaw, rB, count);
                    const block_iq3_sx16 *rb = &R[(size_t)cg * NBLK + b];
                    mine_col(rb, c, [&](int k) { return (int)A[b].qs[k]; }, mRaw, mB,
                             maxAbsBsum, count);
                    if (rRaw != mRaw) r.mismatch_raw++;
                    if (rB != mB) r.mismatch_bsum++;
                    sumf_ref  += (double)W[(size_t)col * NBLK + b].d * A[b].d * (double)rB;
                    sumf_mine += (double)rb->d[c] * A[b].d * (double)mB;
                }
                // THE STORE. REFERENCE = ggml's bare `*s = sumf` (no constant). MODEL = the
                // emitted kernel's `1.0f * sumf`. The norm below is the MEASURED
                // disagreement between those two, not an assertion that they agree.
                double a = sumf_ref, bb = 1.0f * sumf_mine;
                double den = std::fabs(a) > 1e-9 ? std::fabs(a) : 1.0;
                double nrm = std::fabs(a - bb) / den;
                if (nrm > r.worst_norm) r.worst_norm = nrm;
            }
        return r;
    };

    printf("=== iq3_s x q8_K 16x1-REPACK oracle (ZERO-MODEL, x86-native) ===\n");
    printf("corpus: %d cols x %d blocks (%d contraction elements/col)\n\n",
           NCOL, NBLK, NBLK * QK_K);

    bool pinsOk = check_layout_pins();

    // The NORMAL run is the one that MEASURES coverage, in both decode paths.
    Res base = run(DQ_NORMAL, /*count=*/true);

    printf("\n---- CORPUS COVERAGE, MEASURED IN THE DECODE PATHS ----\n");
    printf("(counted at POINT OF USE, never in the generator: [R] = what the REFERENCE\n");
    printf(" read out of the PACKED block, [M] = what the EMITTER-MODEL read out of the\n");
    printf(" REPACKED strips. Both must be complete AND agree.)\n");
    bool covOk = report_cov("R", covR);
    covOk &= report_cov("M", covM);
    bool agree = covR.idx_e0 == covM.idx_e0 && covR.idx_e1 == covM.idx_e1 &&
                 covR.qh_e0 == covM.qh_e0 && covR.qh_e1 == covM.qh_e1 &&
                 covR.sgn == covM.sgn && covR.ls == covM.ls &&
                 covR.pair_distinguishing == covM.pair_distinguishing &&
                 covR.pair_total == covM.pair_total &&
                 covR.ls_pair_distinguishing == covM.ls_pair_distinguishing &&
                 covR.ls_pair_total == covM.ls_pair_total;
    printf("  REF/MODEL coverage AGREEMENT: %s\n",
           agree ? "IDENTICAL (the repack preserved every measured axis)"
                 : "!! DISAGREE (the repack lost or altered a value the reference saw)");
    covOk &= agree;

    printf("\n---- BYTE-EXACT INTEGER CERTIFICATES (ref vs emitter-model) ----\n");
    printf("  sumi_raw (ls-UNWEIGHTED sub-block dot total) mismatches: %lld\n",
           base.mismatch_raw);
    printf("  bsum     (ggml's ONE integer accumulator)     mismatches: %lld\n",
           base.mismatch_bsum);
    printf("  worst fp norm (ggml `*s = sumf` vs emitted `1.0f * sumf`, reported only): %.3e\n",
           base.worst_norm);

    printf("\n---- NEGATIVE CONTROLS (perturb the REFERENCE; a SPIKE proves the\n");
    printf("     structure is genuinely exercised by the emitter-model) ----\n");
    struct NC { const char *name; DqMode mode; const char *what; };
    NC ncs[] = {
        {"GRID      ", DQ_GRID,       "rotated grid entry (idx+37)&511"},
        {"SIGNS     ", DQ_SIGNS,      "sign plane ignored (all +1)"},
        {"LS        ", DQ_LS,         "per-sub-block ls +2"},
        {"SINGLEBASE", DQ_SINGLEBASE, "lanes 4-7 from grid1[j+4] (the one-base bug)"},
        {"QHBIT     ", DQ_QHBIT,      "9th index bit dropped (idx&255, the iq3_xxs misread)"},
        {"SIGN7BIT  ", DQ_SIGN7BIT,   "sign byte masked to 7 bits (the signs64 misread)"},
        {"LSPAIR    ", DQ_LSPAIR,     "partner sub-block's ls nibble (the dual-ls misread)"},
    };
    bool ncOk = true;
    for (const NC &n : ncs) {
        Res r = run(n.mode, /*count=*/false);
        bool spiked = r.mismatch_raw > 0 || r.mismatch_bsum > 0;
        if (!spiked) ncOk = false;
        printf("  %s raw_mm=%-5lld bsum_mm=%-5lld %-8s  (%s)\n", n.name, r.mismatch_raw,
               r.mismatch_bsum, spiked ? "SPIKED" : "!! NULL", n.what);
    }
    // LS must move bsum but NOT the unweighted dot -- the fault-separation claim.
    Res lsr = run(DQ_LS, false);
    bool lsSep = lsr.mismatch_raw == 0 && lsr.mismatch_bsum > 0;
    printf("  LS fault separation (moves bsum, leaves sumi_raw fixed): %s\n",
           lsSep ? "CONFIRMED" : "!! NOT AS DOCUMENTED");
    if (!lsSep) ncOk = false;
    // LSPAIR is an ls-only fault too, so it must separate the same way.
    Res lpr = run(DQ_LSPAIR, false);
    bool lpSep = lpr.mismatch_raw == 0 && lpr.mismatch_bsum > 0;
    printf("  LSPAIR fault separation (bsum only, sumi_raw fixed)    : %s\n",
           lpSep ? "CONFIRMED" : "!! NOT AS DOCUMENTED");
    if (!lpSep) ncOk = false;

    bool pass = pinsOk && covOk && ncOk && base.mismatch_raw == 0 &&
                base.mismatch_bsum == 0;
    printf("\n%s  pins=%s coverage=%s certificates=%s neg_controls=%s\n",
           pass ? "PASS" : "FAIL", pinsOk ? "ok" : "BAD", covOk ? "ok" : "BAD",
           (base.mismatch_raw == 0 && base.mismatch_bsum == 0) ? "0-mismatch" : "MISMATCH",
           ncOk ? "all-spiked" : "BAD");
    return pass ? 0 : 1;
}
