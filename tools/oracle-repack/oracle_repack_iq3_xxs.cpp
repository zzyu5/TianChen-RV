// iq3_xxs x q8_K 16x1-REPACKED GEVM + GEMM numeric oracle (256-entry UINT32 grid
// codebook + ksigns_iq2xs sign plane + single per-sub-block ls scale, 0.25f store).
//
// Gate: does the compiler-emitted iq3_xxs repack kernel compute the CORRECT
// super-block DUAL-ENTRY grid dequant-matmul result?
//
// WHAT MAKES THIS ROW DIFFERENT, and therefore what this oracle is really guarding:
// iq3_xxs's grid is `uint32_t iq3xxs_grid[256]`, so ONE entry supplies only 4 of an
// 8-element group's grid bytes and ggml reads TWO per group --
//     grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*l+0]);
//     grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*l+1]);
//     for (j = 0; j < 4; ++j) { sumi += grid1[j]*q8[j+0]*sign(j+0);
//                               sumi += grid2[j]*q8[j+4]*sign(j+4); }
// Every OTHER registered grid row reads one 8-byte entry per group. A kernel that
// (mis)lowered this row onto the single-base leaf would still produce numbers -- it
// would just read entry idx0+1's bytes for lanes 4-7. See the PAIR coverage axis and
// the SINGLEBASE negative control below: both exist specifically so that failure
// cannot pass this oracle quietly.
//
// ---------------------------------------------------------------------------
// [K-5b] THREE REQUIREMENTS, and how this oracle meets each:
//
// (3) ORACLE INDEPENDENT / NO SHARED DECODE IMPLEMENTATION.
//     The REFERENCE decodes the ORIGINAL (pre-repack) packed block_iq3_xxs exactly as
//     ggml's canonical ggml_vec_dot_iq3_xxs_q8_K does: it splits the qs region itself
//     (`gas = qs + QK_K/4`), memcpy's the raw uint32 aux word, and derives PER SUB-BLOCK
//     at read time:
//         ls    = 2*(aux32 >> 28) + 1
//         signs = ksigns_iq2xs[(aux32 >> 7*l) & 127]   tested by kmask_iq2xs[j]
//     fetching grid bytes by REINTERPRETING the uint32 entry
//     ((const uint8_t *)(iq3xxs_grid + idx)), which is ggml's own read.
//     The EMITTER-MODEL never sees an aux word: it reads the REPACKED block_iq3_xxsx16,
//     where ls / the sign selector / the two grid indices are ALREADY separate flat
//     per-column strips, and gathers grid bytes from a FLAT int8 table by
//     grid_bytes[idx*4+j] against a PRE-EXPANDED signs64[sel*8+j] +-1 plane (the two
//     vluxei16 gathers the emitter emits). Different layout, different index derivation,
//     different table read, different sign representation (ggml tests a bit with kmask;
//     the model multiplies by a materialized +-1 byte). The ONLY shared things are
//     iq3xxs_grid / ksigns_iq2xs / kmask_iq2xs themselves -- the FORMAT CONSTANTS, not an
//     implementation (see iq3xxs_tables.h).
//
// (2) INPUT PATH SAME-SOURCE. Both paths are fed from the SAME generated block_iq3_xxs
//     array: the repacked strips are produced from it by make_block_iq3_xxsx16 (the
//     mat-quant the repack stage performs). The reference reads the originals; the model
//     reads the repack of those same originals. Nothing is captured from the model and
//     replayed.
//
// (1) CORPUS COMPLETENESS -- and the C4a-3 DEFECT THIS FILE FIXES.
//     The iq1_s/iq1_m oracles incremented their coverage counters INSIDE THE GENERATOR.
//     That counts what the generator INTENDED to write, not what the decode actually
//     consumed: had the packing dropped a bit, or had the repack mis-split a word, those
//     counters would still have printed COMPLETE. They measured intent and reported it as
//     coverage.
//     Here every counter is incremented IN A DECODE PATH, at the point of USE, and there
//     are TWO INDEPENDENT SETS -- one filled by the REFERENCE as it reads the packed
//     block, one by the EMITTER-MODEL as it reads the repacked strips. Both must be
//     complete AND they must AGREE. A pack/repack fault that silently narrowed the value
//     range now shows up as a SHORT axis or a REF/MODEL DISAGREEMENT, not as a green
//     "COMPLETE". Axes (each exits non-zero if short):
//       * all 256 grid index values, MEASURED SEPARATELY AT BOTH ENTRY SLOTS
//                                                       (COVERAGE grid_index e0/e1)
//       * all 128 sign selector values                  (COVERAGE sign_sel)
//       * all 16 ls steps {1,3,...,31}                  (COVERAGE ls)
//       * PAIR: groups whose two entry indices are DISTINGUISHING, i.e. idx1 !=
//         (idx0+1)&255 -- the exact condition under which a single-base mis-lowering
//         would read different bytes. A corpus without these could not tell the
//         dual-entry leaf from a broken one, so this is a coverage REQUIREMENT, not a
//         statistic.                                    (COVERAGE pair_distinguishing)
//
// ---------------------------------------------------------------------------
// INTEGER CERTIFICATES -- how many does iq3_xxs actually have?
// Counted from ggml_vec_dot_iq3_xxs_q8_K rather than assumed: it carries ONE integer
// accumulator chain. `sumi` accumulates the signed grid dot within a sub-block and is
// then FOLDED AWAY by `bsum += sumi*ls`; `sumf += d*bsum` is float, and `*s = 0.25f*sumf`
// is the store. So there is exactly ONE end-of-block integer quantity: bsum. (This is
// unlike iq1_m, which has three -- two accumulators plus the assembled scale word --
// because it has a delta term and a scattered fp16 d. iq3_xxs has neither: no delta, no
// second accumulator, no bsums plane, and an ordinary inline fp16 d.)
// We compare bsum BYTE-EXACT (int64 equality), and ADDITIONALLY certify the ls-UNWEIGHTED
// sub-block dot total `sumi_raw = sum_ib sumi_ib`. That second quantity is not in ggml; it
// is here for FAULT SEPARATION -- bsum alone lets an ls fault and a grid fault alias
// (both move one number), whereas an ls fault moves bsum and leaves sumi_raw fixed. Both
// are pure integers (no fp reassociation). The fp fold sumf += d_x*d_y*bsum and the 0.25f
// store are reported as a norm only.
//
// NEGATIVE CONTROLS perturb the REFERENCE; a mismatch SPIKE proves the corresponding
// iq3_xxs structure was genuinely exercised by the emitter-model:
//   GRID       : the REFERENCE fetches a ROTATED grid entry (idx -> (idx+37)&255).
//   SIGNS      : the REFERENCE ignores the sign plane (all signs +1).
//   LS         : the REFERENCE perturbs the per-sub-block ls (+2).
//   SINGLEBASE : the REFERENCE emulates the SINGLE-BASE mis-lowering this row's whole
//                leaf exists to avoid -- it reads lanes 4-7 from grid1's bytes 4-7 (i.e.
//                entry idx0+1) instead of from grid2's bytes 0-3, exactly what the
//                iq2_xxs leaf's `for j<8` against one base would do. This is the control
//                that proves the DUAL-ENTRY nest is real and exercised.
//   AUXSPLIT   : the REFERENCE reads the sign selector at the WRONG aux32 shift
//                (7*l -> 7*l+1), proving the aux-word bit split is exercised.
// GRID/SIGNS/SINGLEBASE/AUXSPLIT spike sumi_raw and bsum; LS spikes bsum ONLY, which is
// exactly right -- ls is not part of the unweighted dot.
//
// Build: g++ -O2 -std=c++17 oracle_repack_iq3_xxs.cpp -o /tmp/oracle_iq3xxs && \
//        /tmp/oracle_iq3xxs

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <set>
#include <random>
#include "iq3xxs_tables.h"

#define QK_K 256

// ---- flat int8 grid byte table (256*4), EXACTLY the table the emitter gathers from
// (weft_iq3xxs_grid viewed through a (const int8_t *) cast). Built by shift extraction;
// the REFERENCE does NOT use this -- it reinterprets the uint32 entry. Note 4 bytes per
// entry, not 8: that single fact is what forces the dual-entry leaf. ----
static int8_t grid_bytes[256 * 4];
// ---- the DERIVED signs64 +-1 plane (128 selectors * 8 bytes), EXACTLY what the emitter
// declares as weft_iq2xxs_signs64 and gathers. The REFERENCE does NOT use this -- it
// tests ksigns bits with kmask, as ggml does. ----
static int8_t signs64[128 * 8];
static void build_tables() {
    for (int e = 0; e < 256; ++e) {
        uint32_t u = iq3xxs_grid[e];
        for (int j = 0; j < 4; ++j)
            grid_bytes[e * 4 + j] = (int8_t)((u >> (8 * j)) & 0xff);
    }
    for (int s = 0; s < 128; ++s) {
        unsigned sel = ksigns_iq2xs[s];
        for (int b = 0; b < 8; ++b)
            signs64[s * 8 + b] = ((sel >> b) & 1u) ? -1 : 1;
    }
}

// ---- original per-super-block block_iq3_xxs (ggml-common.h), 98 bytes ----
//   d       : fp16 super-block scale (float here; see the layout-pin scope note)
//   qs[96]  : 3*QK_K/8. DOUBLE DUTY -- qs[0..63] are the raw grid indices, and
//             qs[64..95] are 8 uint32 aux words (ggml: `gas = x[i].qs + QK_K/4`).
struct block_iq3_xxs {
    float   d;
    uint8_t qs[3 * QK_K / 8];
};

struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};

// ---- repacked block_iq3_xxsx16 (block-as-lane convention), 1696 B ----
//   d[16]           @ +0     (32 B, fp16 in the real kernel; float here)
//   ls[8][16]       @ +32    (128 B, int8 SINGLE ls per sub-block, [1,31])
//   gidx[8][8][16]  @ +160   (1024 B, uint8 raw grid index -- EIGHT per sub-block,
//                             TWO per 8-lane group. This is the strip that DOUBLES.)
//   ssel[8][4][16]  @ +1184  (512 B, uint8 7-bit sign selector -- FOUR per sub-block,
//                             ONE per group, as iq2_xxs has)
struct block_iq3_xxsx16 {
    float   d[16];
    int8_t  ls[8][16];
    uint8_t gidx[8][8][16];
    uint8_t ssel[8][4][16];
};

// The REPACK (mat-quant): split each aux32 ONCE into flat strips. This is the layout
// transform the repack stage performs; the kernel then never sees an aux word.
static block_iq3_xxsx16 make_block_iq3_xxsx16(const block_iq3_xxs *in) {
    block_iq3_xxsx16 out;
    for (int c = 0; c < 16; ++c) out.d[c] = in[c].d;
    for (int c = 0; c < 16; ++c) {
        const uint8_t *q3  = in[c].qs;
        const uint8_t *gas = in[c].qs + QK_K / 4;   // ggml's own split point
        for (int ib = 0; ib < 8; ++ib) {
            uint32_t aux32;
            memcpy(&aux32, gas + 4 * ib, sizeof(uint32_t));
            out.ls[ib][c] = (int8_t)(2 * (aux32 >> 28) + 1);
            for (int l = 0; l < 4; ++l)
                out.ssel[ib][l][c] = (uint8_t)((aux32 >> (7 * l)) & 127);
            // EIGHT indices per sub-block (ggml advances q3 by 8 per sub-block).
            for (int e = 0; e < 8; ++e)
                out.gidx[ib][e][c] = q3[8 * ib + e];
        }
    }
    return out;
}

enum DqMode { DQ_NORMAL = 0, DQ_GRID = 1, DQ_SIGNS = 2, DQ_LS = 3,
              DQ_SINGLEBASE = 4, DQ_AUXSPLIT = 5 };

// ---- CORPUS COVERAGE instrumentation (the [K-5b](1) evidence).
// TWO INDEPENDENT SETS, both filled IN THE DECODE PATH AT POINT OF USE -- never in the
// generator. [R] = what the REFERENCE consumed from the PACKED block; [M] = what the
// EMITTER-MODEL consumed from the REPACKED strips. Both must be complete and must agree;
// a repack that silently narrowed a range shows up as a disagreement. ----
struct Cov {
    std::set<int> idx_e0, idx_e1;   // grid indices, PER ENTRY SLOT
    std::set<int> ssel;             // 7-bit sign selectors
    std::set<int> ls;               // ls steps actually used
    long long pair_distinguishing = 0;  // groups where idx1 != (idx0+1)&255
    long long pair_total = 0;
};
static Cov covR, covM;

// ---- REFERENCE (ggml canonical ggml_vec_dot_iq3_xxs_q8_K), reading the ORIGINAL packed
// block: it splits qs at QK_K/4 itself, memcpy's the aux word, derives ls/signs from it at
// read time, and reads the grid through the uint32 reinterpret, exactly as ggml does. ----
static void ref_block(const block_iq3_xxs *x, const int8_t *q8, DqMode mode,
                      int64_t &sumi_raw_out, int64_t &bsum_out, bool count) {
    int64_t sumi_raw = 0, bsum = 0;
    const uint8_t *q3  = x->qs;
    const uint8_t *gas = x->qs + QK_K / 4;
    int q8pos = 0;
    for (int ib = 0; ib < 8; ++ib) {
        uint32_t aux32;
        memcpy(&aux32, gas, sizeof(uint32_t)); gas += sizeof(uint32_t);
        int64_t ls = 2 * (aux32 >> 28) + 1;
        if (count) covR.ls.insert((int)ls);
        if (mode == DQ_LS) ls += 2;
        int64_t sumi = 0;
        for (int l = 0; l < 4; ++l) {
            int i0 = q3[2 * l + 0], i1 = q3[2 * l + 1];
            if (count) {
                covR.idx_e0.insert(i0);
                covR.idx_e1.insert(i1);
                covR.pair_total++;
                if (i1 != ((i0 + 1) & 255)) covR.pair_distinguishing++;
            }
            if (mode == DQ_GRID) { i0 = (i0 + 37) & 255; i1 = (i1 + 37) & 255; }
            int shift = (mode == DQ_AUXSPLIT) ? 7 * l + 1 : 7 * l;
            unsigned sel = (aux32 >> shift) & 127;
            if (count && mode == DQ_NORMAL) covR.ssel.insert((int)sel);
            const uint8_t signs = ksigns_iq2xs[sel];
            // ggml's own read: reinterpret the uint32 grid entries as 4 uint8 each.
            const uint8_t *grid1 = (const uint8_t *)(iq3xxs_grid + i0);
            const uint8_t *grid2 = (const uint8_t *)(iq3xxs_grid + i1);
            for (int j = 0; j < 4; ++j) {
                int s0 = (mode == DQ_SIGNS) ? 1 : ((signs & kmask_iq2xs[j + 0]) ? -1 : 1);
                int s1 = (mode == DQ_SIGNS) ? 1 : ((signs & kmask_iq2xs[j + 4]) ? -1 : 1);
                // DQ_SINGLEBASE emulates the single-base mis-lowering: lanes 4-7 read
                // grid1's bytes 4-7 (= entry i0+1's bytes 0-3) instead of grid2's 0-3.
                int g2 = (mode == DQ_SINGLEBASE) ? grid1[j + 4] : grid2[j];
                sumi += (int64_t)grid1[j] * q8[q8pos + j + 0] * s0;
                sumi += (int64_t)g2       * q8[q8pos + j + 4] * s1;
            }
            q8pos += 8;
        }
        q3 += 8;
        sumi_raw += sumi;
        bsum += sumi * ls;
    }
    sumi_raw_out = sumi_raw;
    bsum_out = bsum;
}

// ---- EMITTER-MODEL for column c: reproduce the block-as-lane organization (u8 grid
// index -> flat int8 grid GATHER at idx*4+j, u8 sign selector -> signs64 GATHER at
// sel*8+j, vmul sign fold onto the grid byte, ls-weighted i32 dot). This is the DUAL-BASE
// nest: entry e = j/4, grid byte j%4. ----
template <typename ActFn>
static void mine_col(const block_iq3_xxsx16 *b, int c, ActFn act,
                     int64_t &sumi_raw_out, int64_t &bsum_out,
                     int64_t &maxAbsBsum, bool count) {
    int64_t sumi_raw = 0, bsum = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int64_t sub = 0;
        int lsv = b->ls[ib][c];
        if (count) covM.ls.insert(lsv);
        for (int grp = 0; grp < 4; ++grp) {
            int i0  = b->gidx[ib][2 * grp + 0][c];
            int i1  = b->gidx[ib][2 * grp + 1][c];
            int sel = b->ssel[ib][grp][c];
            if (count) {
                covM.idx_e0.insert(i0);
                covM.idx_e1.insert(i1);
                covM.ssel.insert(sel);
                covM.pair_total++;
                if (i1 != ((i0 + 1) & 255)) covM.pair_distinguishing++;
            }
            for (int j = 0; j < 8; ++j) {
                int k = ib * 32 + grp * 8 + j;
                int e  = j / 4;                 // THE ACTIVATION-RANGE SPLIT
                int gj = j % 4;
                int idx = e ? i1 : i0;
                int gbv = grid_bytes[idx * 4 + gj];   // REAL grid GATHER (idx*4, not *8)
                int sgn = signs64[sel * 8 + j];       // REAL sign GATHER (sel*8)
                sub += (int64_t)act(k) * (gbv * sgn); // sign folded ONTO the grid byte
            }
        }
        sumi_raw += sub;
        bsum += sub * (int64_t)lsv;
        if (std::llabs(bsum) > maxAbsBsum) maxAbsBsum = std::llabs(bsum);
    }
    sumi_raw_out = sumi_raw;
    bsum_out = bsum;
}

// The generator SWEEPS the decode axes systematically so coverage is structural, not
// luck -- but note that NOTHING here is counted as coverage: these counters are only
// cycle drivers. Coverage is MEASURED in the decode paths above, which is the whole
// point (a generator that intends 256 indices but packs them wrong must not be able to
// report 256). idx runs a full sequential cycle over [0,255]; the two entry slots are
// driven by DIFFERENT strides so idx1 is generally != idx0+1 (the pair_distinguishing
// axis measures whether that actually held); the ls field runs a stride-3 cycle over
// [0,15] and the sign selector a stride-5 cycle over [0,127] (3 coprime with 16, 5 with
// 128 => all steps). Activation magnitudes stay random.
static long long g_i0 = 0, g_i1 = 40, g_sb = 0, g_sel = 0;

static void build_iq3xxs_block(block_iq3_xxs *x, std::mt19937 &rng, int col, int blk) {
    x->d = 0.010f + 0.0006f * ((col + 3 * blk) % 11);
    uint8_t *q3  = x->qs;
    uint8_t *gas = x->qs + QK_K / 4;
    for (int ib = 0; ib < 8; ++ib) {
        uint32_t aux32 = 0;
        int lsf = (int)((g_sb * 3) % 16);
        aux32 |= ((uint32_t)lsf) << 28;
        for (int l = 0; l < 4; ++l) {
            int sel = (int)((g_sel * 5) % 128);
            ++g_sel;
            aux32 |= ((uint32_t)sel) << (7 * l);
            // The two entry slots advance at DIFFERENT rates, so the pair is generally
            // distinguishing (idx1 != idx0+1). The decode-path axis verifies it.
            q3[8 * ib + 2 * l + 0] = (uint8_t)(g_i0 % 256); g_i0 += 1;
            q3[8 * ib + 2 * l + 1] = (uint8_t)(g_i1 % 256); g_i1 += 3;
        }
        memcpy(gas + 4 * ib, &aux32, sizeof(uint32_t));
        ++g_sb;
    }
}

// A REAL block_q8_K. iq3_xxs does NOT read bsums (no delta term), but a real quantizer
// fills them, so we do too -- and their presence is exactly why the front door must NOT
// stamp an activation_bsums_byte_offset for this row.
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

// ---- The REAL (fp16-d) block_iq3_xxsx16 byte layout, recomputed independently here and
// pinned against the byte offsets the FRONT DOOR ships (kIq3XxsDecodeFacts in
// lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp, and the lit CHECKs in
// test/Conversion/RVV/rvv-emit-*-iq3-xxs-repack-*.mlir).
//
// SCOPE, stated honestly and up front (this file inherits the C4a-3 correction rather
// than repeating its mistake): these pins are a TRANSCRIPTION GUARD, NOT front-door
// coverage. This file #includes NOTHING from the front door -- the "shipped" column below
// is a hand-transcribed literal. Re-laying-out the front door and leaving this file
// untouched leaves these pins GREEN; that was verified by test on the iq1_m sibling
// (weightBlockStride 1824 -> 1840 stayed green), not argued from reasoning. All these pins
// can catch is this oracle's own internal inconsistency: that its formula-derived offsets
// match its own transcribed column.
//
// The real coupling to the front door lives in the lit CHECKs (`CHECK: literal "1696"` /
// `"292"` / `"1168"`), not here. Do not let this be cited as front-door coverage.
//
// The emitter-model above also addresses the repacked strips by ARRAY INDEX, so it
// validates the layout's SEMANTICS + the arithmetic, NOT the emitter's byte arithmetic
// (its `float d[16]` is 64 B where the kernel's fp16 d[16] is 32 B -- the same modelling
// shortcut all 13 sibling oracles take).
static bool check_layout_pins() {
    const int kInterleave = 16;
    int off_d    = 0;
    int off_ls   = off_d + kInterleave * 2;               // 16 fp16 d   -> +32
    int off_gidx = off_ls + 8 * kInterleave;              // 8*16 int8   -> +160
    int off_ssel = off_gidx + 8 * 8 * kInterleave;        // 8*8*16 u8   -> +1184
    int stride   = off_ssel + 8 * 4 * kInterleave;        // 8*4*16 u8   -> 1696
    struct Pin { const char *name; int got; int shipped; };
    Pin pins[] = {
        {"weight_block_stride       ", stride,    1696},
        {"weight_ls_byte_offset     ", off_ls,      32},
        {"weight_grid_idx_offset    ", off_gidx,   160},
        {"weight_sign_sel_offset    ", off_ssel,  1184},
        // block_q8_K: fp32 d @0 + 256 int8 quants @4 + 16 int16 bsums @260 = 292.
        {"gevm_act_quant_offset     ", 4,            4},
        {"gevm_act_block_stride     ", 4 + 256 + 32, 292},
        // block_q8_Kx4: 4 fp32 d @0 + 1024 int8 quants @16 + 64 int16 bsums @1040.
        {"gemm_act_quant_offset     ", 16,          16},
        {"gemm_act_block_stride     ", 16 + 1024 + 128, 1168},
        // The PLAIN block_iq3_xxs: fp16 d + 3*(QK_K/8) qs = 2 + 96 = 98, and ggml's own
        // aux split point gas = qs + QK_K/4 = +64 within qs (absolute +66).
        {"plain_iq3_xxs_stride      ", 2 + 3 * QK_K / 8, 98},
        {"plain_gas_byte_offset     ", 2 + QK_K / 4,     66},
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
    printf("  [%s] grid_index e0: %3zu/256 %-8s   e1: %3zu/256 %-8s\n", tag,
           cv.idx_e0.size(), cv.idx_e0.size() == 256 ? "COMPLETE" : "!! SHORT",
           cv.idx_e1.size(), cv.idx_e1.size() == 256 ? "COMPLETE" : "!! SHORT");
    if (cv.idx_e0.size() != 256 || cv.idx_e1.size() != 256) ok = false;
    printf("  [%s] sign_sel      : %3zu/128 %s\n", tag, cv.ssel.size(),
           cv.ssel.size() == 128 ? "COMPLETE" : "!! SHORT");
    if (cv.ssel.size() != 128) ok = false;
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
    return ok;
}

int main() {
    build_tables();
    std::mt19937 rng(0xC4A4);

    const int NCOLGROUPS = 3;      // 3 * 16 = 48 weight columns
    const int NBLK = 24;           // 24 super-blocks of 256 = 6144 contraction elements
    const int NCOL = NCOLGROUPS * 16;

    std::vector<block_iq3_xxs> W((size_t)NCOL * NBLK);
    std::vector<block_q8_K>    A(NBLK);
    for (int cg = 0; cg < NCOLGROUPS; ++cg)
        for (int c = 0; c < 16; ++c)
            for (int b = 0; b < NBLK; ++b)
                build_iq3xxs_block(&W[(size_t)(cg * 16 + c) * NBLK + b], rng, cg * 16 + c, b);
    for (int b = 0; b < NBLK; ++b) build_q8_K_block(&A[b], rng, b);

    // The REPACK of those SAME originals ([K-5b](2) input-path same-source).
    std::vector<block_iq3_xxsx16> R((size_t)NCOLGROUPS * NBLK);
    for (int cg = 0; cg < NCOLGROUPS; ++cg)
        for (int b = 0; b < NBLK; ++b) {
            block_iq3_xxs col[16];
            for (int c = 0; c < 16; ++c) col[c] = W[(size_t)(cg * 16 + c) * NBLK + b];
            R[(size_t)cg * NBLK + b] = make_block_iq3_xxsx16(col);
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
                    const block_iq3_xxsx16 *rb = &R[(size_t)cg * NBLK + b];
                    mine_col(rb, c, [&](int k) { return (int)A[b].qs[k]; }, mRaw, mB,
                             maxAbsBsum, count);
                    if (rRaw != mRaw) r.mismatch_raw++;
                    if (rB != mB) r.mismatch_bsum++;
                    sumf_ref  += (double)W[(size_t)col * NBLK + b].d * A[b].d * (double)rB;
                    sumf_mine += (double)rb->d[c] * A[b].d * (double)mB;
                }
                // The 0.25f store (ggml: *s = 0.25f * sumf), reported as a norm only.
                double a = 0.25 * sumf_ref, bb = 0.25 * sumf_mine;
                double den = std::fabs(a) > 1e-9 ? std::fabs(a) : 1.0;
                double nrm = std::fabs(a - bb) / den;
                if (nrm > r.worst_norm) r.worst_norm = nrm;
            }
        return r;
    };

    printf("=== iq3_xxs x q8_K 16x1-REPACK oracle (ZERO-MODEL, x86-native) ===\n");
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
                 covR.ssel == covM.ssel && covR.ls == covM.ls &&
                 covR.pair_distinguishing == covM.pair_distinguishing &&
                 covR.pair_total == covM.pair_total;
    printf("  REF/MODEL coverage AGREEMENT: %s\n",
           agree ? "IDENTICAL (the repack preserved every measured axis)"
                 : "!! DISAGREE (the repack lost or altered a value the reference saw)");
    covOk &= agree;

    printf("\n---- BYTE-EXACT INTEGER CERTIFICATES (ref vs emitter-model) ----\n");
    printf("  sumi_raw (ls-UNWEIGHTED sub-block dot total) mismatches: %lld\n",
           base.mismatch_raw);
    printf("  bsum     (ggml's ONE integer accumulator)     mismatches: %lld\n",
           base.mismatch_bsum);
    printf("  worst fp norm (0.25f store, reported only)             : %.3e\n",
           base.worst_norm);

    printf("\n---- NEGATIVE CONTROLS (perturb the REFERENCE; a SPIKE proves the\n");
    printf("     structure is genuinely exercised by the emitter-model) ----\n");
    struct NC { const char *name; DqMode mode; const char *what; };
    NC ncs[] = {
        {"GRID      ", DQ_GRID,       "rotated grid entry (idx+37)&255"},
        {"SIGNS     ", DQ_SIGNS,      "sign plane ignored (all +1)"},
        {"LS        ", DQ_LS,         "per-sub-block ls +2"},
        {"SINGLEBASE", DQ_SINGLEBASE, "lanes 4-7 from grid1[j+4] (the one-base bug)"},
        {"AUXSPLIT  ", DQ_AUXSPLIT,   "sign selector read at aux32 shift 7l+1"},
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

    bool pass = pinsOk && covOk && ncOk && base.mismatch_raw == 0 &&
                base.mismatch_bsum == 0;
    printf("\n%s  pins=%s coverage=%s certificates=%s neg_controls=%s\n",
           pass ? "PASS" : "FAIL", pinsOk ? "ok" : "BAD", covOk ? "ok" : "BAD",
           (base.mismatch_raw == 0 && base.mismatch_bsum == 0) ? "0-mismatch" : "MISMATCH",
           ncOk ? "all-spiked" : "BAD");
    return pass ? 0 : 1;
}
