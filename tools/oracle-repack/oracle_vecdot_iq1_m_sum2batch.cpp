// ============================================================================
// oracle_vecdot_iq1_m_sum2batch.cpp
//
// P3-Q1 (2026-07-17). Gate for the *vec_dot block-dot* iq1_m sum2/delta axis.
//
// ---------------------------------------------------------------------------
// WHY THIS FILE EXISTS (read before citing it)
// ---------------------------------------------------------------------------
// The P3-Q1 tasking said: reuse tools/oracle-repack/oracle_repack_iq1_m.cpp,
// "该 oracle 就是为这条机制设的门 -- 用它". That premise is FALSE, for two
// independent reasons, each verified against the file and recorded here so the
// next round does not re-derive them:
//
//   (1) WRONG PATH. Its model (`mine_col`) addresses `block_iq1_mx16` -- the
//       1824 B *repacked*, block-as-lane strip set -- reading pre-assembled
//       b->gidx / b->ls / b->delta. Its own header says the kernel on that path
//       "never sees qh". The vec_dot block-dot path reads the raw 56 B
//       `block_iq1_m` (qs[32]+qh[16]+scales[8]) and derives idx/ls/delta from qh
//       itself. Different input path => [K-5b](3) "输入路径同源" fails.
//
//   (2) IT NEVER EXECUTES EMITTED CODE. Its own SCOPE paragraph states it
//       "validates the layout's SEMANTICS + the arithmetic, NOT the emitter's
//       byte arithmetic". It is a hand-written C++ model vs a hand-written C++
//       reference. Even for the repack path it is not an emission gate.
//
// So this file is the vec_dot-path oracle that did not previously exist.
//
// ---------------------------------------------------------------------------
// WHAT THIS FILE GATES -- AND WHAT IT DOES **NOT** (state this when citing)
// ---------------------------------------------------------------------------
// GATES: the FOLD ALGEBRA of the proposed "sum2 per-group 符号和向量化批处理"
// mechanism -- i.e. that replacing, per 16-element half, the TWO 8-lane tiny
// reductions
//        sum2[h] = lsum2(2h)*delta[2h] + lsum2(2h+1)*delta[2h+1]
// by ONE sign-weighted 16-lane widening multiply-then-reduce
//        sum2[h] = SUM_{i=0..15} q8[16h+i] * s[i],
//        s[0..7] = delta[2h],  s[8..15] = delta[2h+1]
// preserves the value EXACTLY, over a swept corpus, against ggml's canonical
// fold. delta is +-1, so this is pure integer distribution -- no rounding, no
// reassociation of floats.
//
//     CITATION CORRECTION (2026-07-17): an earlier version cited "emitter :279"
//     for this. That line is
//         lib/Conversion/RVV/RVVToEmitCTernaryBinary.cpp:279
//         // delta = 1 - 2*((qhw >> 15) & 1)   (= +1 if bit15==0, -1 if bit15==1)
//     -- which is iq1_S's delta (one bit per super-block half, off the u16 qh
//     word), NOT iq1_M's. iq1_m's delta is per-8-group, four independent bits.
//     Citing it here was a wrong-format citation (and a bare line number with no
//     filename). The conclusion happens to hold, but one of its two witnesses did
//     not support it.
//     The correct iq1_m witness, opened and checked:
//         tools/oracle-repack/oracle_repack_iq1_m.cpp:191
//         out.delta[ib][l][c] = (int8_t)((nib & 0x8) ? -1 : 1);
//     i.e. iq1_m's delta is derived from a nibble bit and is literally +-1.
//
// DOES **NOT** GATE (named gaps, do not let this file be cited for them):
//   * NOT emission. No RVV code is compiled or run here.
//
//     CORRECTION (2026-07-17, by test, not by reasoning): an earlier version of
//     this comment said the host has "no riscv64 cross-compiler, no clang
//     (checked)". That is FALSE and was refuted on this machine:
//         /usr/lib/llvm-20/bin/clang-20 --print-targets | grep -ci riscv   -> 2
//     clang-20 cross-compiles the emitted kernel to riscv64 successfully
//     (--target=riscv64-unknown-elf -march=rv64gcv -O3 -c  ->  exit 0).
//     The original check used `command -v clang`, which misses the versioned
//     binary -- a "compiler absent" conclusion drawn from the wrong probe.
//
//     The real blocker is EXECUTION, not compilation:
//         qemu-riscv64 / qemu-riscv64-static / qemu-system-riscv64 / spike
//         -> all ABSENT (the installed qemu-img/io/nbd are disk-image tools).
//     So the emitted RVV kernel can be BUILT here but not RUN here.
//     [K-5] byte-exact on the EMITTED kernel remains OPEN and needs the board.
//   * NOT decode independence in the derivation sense. REF and MODEL both read
//     the same qh bit positions, because those bit positions ARE the format.
//     What is independent here is the FOLD STRUCTURE: REF keeps ggml's own
//     grouping (per-ls-half sum1/sum2, folded once at sub-block end); MODEL_A is
//     the emitter's per-group fold; MODEL_B is the proposed batched fold. The
//     three fold structures are genuinely different code, and that difference is
//     exactly what is under test.
//   * NOT a claim that fewer reductions run faster. Instruction count is not
//     time. (PR-37 / PR-46(a): byte-exact levers that hit their structural
//     target and moved cold not at all.)
//
// Build + run (x86 host, self-contained, no board):
//   g++ -O2 -std=c++17 tools/oracle-repack/oracle_vecdot_iq1_m_sum2batch.cpp \
//       -o /tmp/oracle_vecdot_iq1m && /tmp/oracle_vecdot_iq1m
//   echo "exit=$?"     # 0 = PASS, 1 = FAIL
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "iq1s_grid.h"   // canonical ggml iq1s_grid[2048] (format constant table)

static const int QK_K = 256;

// ---- the RAW packed block the vec_dot path actually reads: 56 B ----
struct block_iq1_m {
    uint8_t qs[QK_K / 8];    // 32: low 8 bits of each group's grid index
    uint8_t qh[QK_K / 16];   // 16: per group-pair -> [delta|ghigh(3)] nibble pairs
    uint8_t scales[QK_K / 32]; // 8 B = 4 uint16 words: ls1/ls2 + scale nibble.
                               // NOTE: no inline d -- the fp16 super-block scale is
                               // scattered as four nibbles across these words.
};
static_assert(sizeof(block_iq1_m) == 56, "block_iq1_m must be 56 B");

// flat ternary grid, as the kernel gathers it
static int8_t grid_bytes[2048 * 8];
static void init_grid() {
    for (int e = 0; e < 2048; ++e) {
        uint64_t u = iq1s_grid[e];
        for (int j = 0; j < 8; ++j)
            grid_bytes[e * 8 + j] = (int8_t)((u >> (8 * j)) & 0xff);
    }
}

// ---------------------------------------------------------------------------
// Fault-injection modes. NONE is the real gate; the rest are the negative
// controls, each of which must SPIKE (or, where stated, must NOT spike).
// ---------------------------------------------------------------------------
enum Mode {
    M_NONE = 0,
    M_BSUMS16,     // pretend the two 8-groups of a 16-group share a delta
    M_SIGNSWAP,    // MODEL_B only: swap the two halves of the sign vector
    M_SIGNALL1,    // MODEL_B only: force sign vector to all +1 (drop delta)
    M_LANEROT,     // MODEL_B only: rotate sign vector by 1 lane (misalign)
    M_GRIDPERT,    // perturb grid index (should hit sumi1 only, NOT sumi2)
};

// ===========================================================================
// REFERENCE -- ggml canonical ggml_vec_dot_iq1_m_q8_K fold.
// Keeps ggml's OWN grouping: sum1/sum2 per ls half, folded once at the end of
// the sub-block. Reads the raw qh/qs/scales.
// ===========================================================================
static void ref_block(const block_iq1_m *x, const int8_t *q8, Mode mode,
                      int64_t &sumi1_out, int64_t &sumi2_out) {
    int64_t sumi1 = 0, sumi2 = 0;
    const uint8_t *qs = x->qs;
    const uint8_t *qh = x->qh;
    const uint16_t *sc = (const uint16_t *)x->scales;
    int q8pos = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int64_t sum1[2] = {0, 0}, sum2[2] = {0, 0};
        int delta[4];
        delta[0] = (qh[2 * ib + 0] & 0x08) ? -1 : 1;
        delta[1] = (qh[2 * ib + 0] & 0x80) ? -1 : 1;
        delta[2] = (qh[2 * ib + 1] & 0x08) ? -1 : 1;
        delta[3] = (qh[2 * ib + 1] & 0x80) ? -1 : 1;
        if (mode == M_BSUMS16) { delta[1] = delta[0]; delta[3] = delta[2]; }
        for (int l = 0; l < 4; ++l) {
            int qhb = qh[2 * ib + l / 2];
            int idx = qs[4 * ib + l] | (((uint16_t)qhb << (8 - 4 * (l % 2))) & 0x700);
            if (mode == M_GRIDPERT) idx = (idx + 457) & 2047;
            const int8_t *grid = grid_bytes + (size_t)idx * 8;
            int64_t lsum1 = 0, lsum2 = 0;
            for (int j = 0; j < 8; ++j) {
                lsum1 += (int64_t)q8[q8pos + j] * grid[j];
                lsum2 += (int64_t)q8[q8pos + j];
            }
            q8pos += 8;
            sum1[l / 2] += lsum1;
            sum2[l / 2] += lsum2 * delta[l];
        }
        int shift = 6 * (ib % 2);
        int64_t ls1 = 2 * ((sc[ib / 2] >> (shift + 0)) & 0x7) + 1;
        int64_t ls2 = 2 * ((sc[ib / 2] >> (shift + 3)) & 0x7) + 1;
        sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
        sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
    }
    sumi1_out = sumi1;
    sumi2_out = sumi2;
}

// ===========================================================================
// MODEL_A -- the fold the emitter EMITS TODAY (32x tiny 8-lane vwredsum).
// Mirrors lib/Conversion/RVV/RVVToEmitCTernaryBinary.cpp:
//   vsetvl_e8m1(8) ; vle8_v_i8m1 ; vmv_v_x_i16m1(0,1) ;
//   vwredsum_vs_i8m1_i16m1 ; vmv_x_s_i16m1_i16 ; sum2[l/2] += lsum2*delta[l]
// `tiny_reductions` counts the 8-lane reductions this fold requires.
// ===========================================================================
static void model_A(const block_iq1_m *x, const int8_t *q8, Mode mode,
                    int64_t &sumi1_out, int64_t &sumi2_out,
                    long long &tiny_reductions) {
    int64_t sumi1 = 0, sumi2 = 0;
    const uint8_t *qs = x->qs;
    const uint8_t *qh = x->qh;
    const uint16_t *sc = (const uint16_t *)x->scales;
    for (int ib = 0; ib < 8; ++ib) {
        int64_t sum1[2] = {0, 0}, sum2[2] = {0, 0};
        int delta[4];
        delta[0] = (qh[2 * ib + 0] & 0x08) ? -1 : 1;
        delta[1] = (qh[2 * ib + 0] & 0x80) ? -1 : 1;
        delta[2] = (qh[2 * ib + 1] & 0x08) ? -1 : 1;
        delta[3] = (qh[2 * ib + 1] & 0x80) ? -1 : 1;
        if (mode == M_BSUMS16) { delta[1] = delta[0]; delta[3] = delta[2]; }
        for (int l = 0; l < 4; ++l) {
            int qhb = qh[2 * ib + l / 2];
            int idx = qs[4 * ib + l] | (((uint16_t)qhb << (8 - 4 * (l % 2))) & 0x700);
            if (mode == M_GRIDPERT) idx = (idx + 457) & 2047;
            const int8_t *grid = grid_bytes + (size_t)idx * 8;
            const int8_t *g8 = q8 + (ib * 32 + l * 8);
            int64_t sub = 0;
            int16_t lsum2 = 0;           // i16 accumulator: |Σ| <= 8*127 = 1016
            for (int j = 0; j < 8; ++j) {
                sub += (int64_t)g8[j] * grid[j];
                lsum2 = (int16_t)(lsum2 + g8[j]);   // the tiny 8-lane vwredsum
            }
            ++tiny_reductions;
            sum1[l / 2] += sub;
            sum2[l / 2] += (int64_t)lsum2 * delta[l];
        }
        int shift = 6 * (ib % 2);
        int64_t ls1 = 2 * ((sc[ib / 2] >> (shift + 0)) & 0x7) + 1;
        int64_t ls2 = 2 * ((sc[ib / 2] >> (shift + 3)) & 0x7) + 1;
        sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
        sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
    }
    sumi1_out = sumi1;
    sumi2_out = sumi2;
}

// ===========================================================================
// MODEL_B -- the PROPOSED batched fold. Per 16-element half h:
//   build s[16] : s[0..7] = delta[2h], s[8..15] = delta[2h+1]      (i8, +-1)
//   prod[i]     = q8[16h+i] * s[i]                                 (vwmul i8->i16)
//   sum2[h]     = SUM_{i=0..15} prod[i]                            (vwredsum i16->i32)
// This is ONE 16-lane sign-weighted multiply-reduce in place of TWO 8-lane tiny
// reductions. `batched_reductions` counts the 16-lane reductions.
// ===========================================================================
static void model_B(const block_iq1_m *x, const int8_t *q8, Mode mode,
                    int64_t &sumi1_out, int64_t &sumi2_out,
                    long long &batched_reductions) {
    int64_t sumi1 = 0, sumi2 = 0;
    const uint8_t *qs = x->qs;
    const uint8_t *qh = x->qh;
    const uint16_t *sc = (const uint16_t *)x->scales;
    for (int ib = 0; ib < 8; ++ib) {
        int64_t sum1[2] = {0, 0}, sum2[2] = {0, 0};
        int delta[4];
        delta[0] = (qh[2 * ib + 0] & 0x08) ? -1 : 1;
        delta[1] = (qh[2 * ib + 0] & 0x80) ? -1 : 1;
        delta[2] = (qh[2 * ib + 1] & 0x08) ? -1 : 1;
        delta[3] = (qh[2 * ib + 1] & 0x80) ? -1 : 1;
        if (mode == M_BSUMS16) { delta[1] = delta[0]; delta[3] = delta[2]; }

        // --- grid dot (sumi1 axis): untouched by this mechanism ---
        for (int l = 0; l < 4; ++l) {
            int qhb = qh[2 * ib + l / 2];
            int idx = qs[4 * ib + l] | (((uint16_t)qhb << (8 - 4 * (l % 2))) & 0x700);
            if (mode == M_GRIDPERT) idx = (idx + 457) & 2047;
            const int8_t *grid = grid_bytes + (size_t)idx * 8;
            const int8_t *g8 = q8 + (ib * 32 + l * 8);
            int64_t sub = 0;
            for (int j = 0; j < 8; ++j) sub += (int64_t)g8[j] * grid[j];
            sum1[l / 2] += sub;
        }

        // --- BATCHED sum2 axis: one sign-weighted 16-lane reduce per half ---
        for (int h = 0; h < 2; ++h) {
            int8_t s[16];
            for (int i = 0; i < 8; ++i)  s[i]     = (int8_t)delta[2 * h + 0];
            for (int i = 8; i < 16; ++i) s[i]     = (int8_t)delta[2 * h + 1];
            if (mode == M_SIGNALL1) for (int i = 0; i < 16; ++i) s[i] = 1;
            if (mode == M_SIGNSWAP) { for (int i = 0; i < 8; ++i) { int8_t t = s[i]; s[i] = s[i + 8]; s[i + 8] = t; } }
            if (mode == M_LANEROT) { int8_t t = s[0]; for (int i = 0; i < 15; ++i) s[i] = s[i + 1]; s[15] = t; }

            const int8_t *g16 = q8 + (ib * 32 + h * 16);
            int32_t acc = 0;                  // i32 accumulator: |Σ| <= 16*127 = 2032
            for (int i = 0; i < 16; ++i) {
                int16_t prod = (int16_t)((int16_t)g16[i] * (int16_t)s[i]);  // vwmul
                acc += prod;                                                // vwredsum
            }
            ++batched_reductions;
            sum2[h] = acc;
        }

        int shift = 6 * (ib % 2);
        int64_t ls1 = 2 * ((sc[ib / 2] >> (shift + 0)) & 0x7) + 1;
        int64_t ls2 = 2 * ((sc[ib / 2] >> (shift + 3)) & 0x7) + 1;
        sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
        sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
    }
    sumi1_out = sumi1;
    sumi2_out = sumi2;
}

// ---------------------------------------------------------------------------
// CORPUS COVERAGE -- MEASURED (printed), not asserted. [K-5b](1).
// ---------------------------------------------------------------------------
static std::set<int> cov_idx;        // distinct 11-bit grid indices exercised
static long long cov_delta[4][2];    // PER GROUP SLOT: +1 seen / -1 seen
static long long cov_ls1[8], cov_ls2[8];

static void observe(const block_iq1_m *x) {
    const uint8_t *qs = x->qs, *qh = x->qh;
    const uint16_t *sc = (const uint16_t *)x->scales;
    for (int ib = 0; ib < 8; ++ib) {
        int d[4] = {
            (qh[2 * ib + 0] & 0x08) ? 1 : 0, (qh[2 * ib + 0] & 0x80) ? 1 : 0,
            (qh[2 * ib + 1] & 0x08) ? 1 : 0, (qh[2 * ib + 1] & 0x80) ? 1 : 0};
        for (int l = 0; l < 4; ++l) {
            ++cov_delta[l][d[l]];
            int qhb = qh[2 * ib + l / 2];
            cov_idx.insert(qs[4 * ib + l] | (((uint16_t)qhb << (8 - 4 * (l % 2))) & 0x700));
        }
        int shift = 6 * (ib % 2);
        ++cov_ls1[(sc[ib / 2] >> (shift + 0)) & 0x7];
        ++cov_ls2[(sc[ib / 2] >> (shift + 3)) & 0x7];
    }
}

// ---------------------------------------------------------------------------
// Generator: SWEEPS the decode axes so coverage is structural, not luck.
//   idx     : full sequential cycle over [0,2047]
//   ls1/ls2 : stride-3 / stride-5 cycles over [0,7] (both coprime with 8 =>
//             all 8 steps each, and the two never lock together)
//   delta   : each of the 4 bits driven by a different counter bit => all 4
//             slots see BOTH polarities independently
//   q8      : random magnitudes, incl. the +-127 rails
// ---------------------------------------------------------------------------
static void gen_block(block_iq1_m *x, int8_t *q8, long long n, std::mt19937 &rng) {
    std::uniform_int_distribution<int> act(-127, 127);
    for (int ib = 0; ib < 8; ++ib) {
        for (int l = 0; l < 4; ++l) {
            long long t = n * 32 + ib * 4 + l;
            int idx = (int)(t % 2048);
            x->qs[4 * ib + l] = (uint8_t)(idx & 0xff);
            int high = (idx >> 8) & 0x7;
            uint8_t &h = x->qh[2 * ib + l / 2];
            if (l % 2 == 0) h = (uint8_t)((h & 0xf0) | high);
            else            h = (uint8_t)((h & 0x0f) | (high << 4));
        }
        // four INDEPENDENT delta bits, from four different counter bits
        long long c = n * 8 + ib;
        if ((c >> 0) & 1) x->qh[2 * ib + 0] |= 0x08;
        if ((c >> 1) & 1) x->qh[2 * ib + 0] |= 0x80;
        if ((c >> 2) & 1) x->qh[2 * ib + 1] |= 0x08;
        if ((c >> 3) & 1) x->qh[2 * ib + 1] |= 0x80;
    }
    uint16_t *sc = (uint16_t *)x->scales;
    for (int w = 0; w < 4; ++w) sc[w] = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int shift = 6 * (ib % 2);
        int a = (int)((n * 8 + ib) * 3 % 8);   // ls1: stride 3
        int b = (int)((n * 8 + ib) * 5 % 8);   // ls2: stride 5
        sc[ib / 2] |= (uint16_t)(a << (shift + 0));
        sc[ib / 2] |= (uint16_t)(b << (shift + 3));
    }
    for (int i = 0; i < QK_K; ++i) q8[i] = (int8_t)act(rng);
    // pin the rails so the i16 product bound is exercised
    if (n % 7 == 0) { q8[0] = 127; q8[8] = -127; q8[16] = 127; q8[31] = -127; }
}

struct Res { long long mismatch, total; };

// ---------------------------------------------------------------------------
// Fault injection MUST be ASYMMETRIC.
//
// FIRST VERSION OF THIS FILE GOT THIS WRONG, and the controls caught it: a mode
// applied to REF *and* A *and* B moves all three together, they still agree, and
// the control goes SILENT. BSUMS16 and GRIDPERT both read 0/20000 that way. That
// is the textbook hollow control -- a gate that cannot fail is not a gate.
//
// So each control names WHICH side it perturbs, and the other sides stay clean.
// ---------------------------------------------------------------------------
static Res run(Mode refMode, Mode aMode, Mode bMode, bool quiet, long long NB) {
    std::mt19937 rng(0xC0FFEEu);   // fixed seed: same corpus for every mode
    block_iq1_m x;
    int8_t q8[QK_K];
    long long mism = 0;
    long long tinyA = 0, batchB = 0;
    const bool clean = (refMode == M_NONE && aMode == M_NONE && bMode == M_NONE);
    for (long long n = 0; n < NB; ++n) {
        std::memset(&x, 0, sizeof(x));
        gen_block(&x, q8, n, rng);
        if (clean) observe(&x);

        int64_t r1, r2, a1, a2, b1, b2;
        ref_block(&x, q8, refMode, r1, r2);
        model_A(&x, q8, aMode, a1, a2, tinyA);
        model_B(&x, q8, bMode, b1, b2, batchB);

        // The gate: all three folds must agree on BOTH integer certificates.
        if (r1 != a1 || r2 != a2 || r1 != b1 || r2 != b2) ++mism;
    }
    if (!quiet && clean) {
        std::printf("  fold-reduction cost over corpus: MODEL_A tiny(8-lane)=%lld"
                    "  MODEL_B batched(16-lane)=%lld  ratio=%.2fx\n",
                    tinyA, batchB, batchB ? (double)tinyA / (double)batchB : 0.0);
        std::printf("  per super-block:                 MODEL_A=%lld  MODEL_B=%lld\n",
                    tinyA / NB, batchB / NB);
    }
    return {mism, NB};
}

// Measure a FORMAT property: does forcing the two 8-groups of a 16-group to share
// a delta (exactly what a per-16 block_q8_K bsums entry can express) change the
// canonical answer? This is the C4a-3 insight, MEASURED rather than asserted.
// It compares REF against REF -- it is NOT a control on the batching mechanism.
static long long measure_bsums16_inexpressive(long long NB) {
    std::mt19937 rng(0xC0FFEEu);
    block_iq1_m x; int8_t q8[QK_K];
    long long differ = 0;
    for (long long n = 0; n < NB; ++n) {
        std::memset(&x, 0, sizeof(x));
        gen_block(&x, q8, n, rng);
        int64_t c1, c2, f1, f2;
        ref_block(&x, q8, M_NONE, c1, c2);
        ref_block(&x, q8, M_BSUMS16, f1, f2);
        if (c1 != f1 || c2 != f2) ++differ;
    }
    return differ;
}

int main(int argc, char **argv) {
    const long long NB = (argc > 1) ? atoll(argv[1]) : 20000;
    init_grid();

    std::printf("=== iq1_m vec_dot sum2-batching FOLD-ALGEBRA oracle ===\n");
    std::printf("blocks=%lld  (REF=ggml canonical fold | A=emitter per-group fold |"
                " B=proposed batched fold)\n\n", NB);

    bool ok = true;

    // ---- THE GATE ----
    Res g = run(M_NONE, M_NONE, M_NONE, false, NB);
    std::printf("\n[GATE   ] NONE      mismatch=%lld/%lld   %s\n",
                g.mismatch, g.total, g.mismatch == 0 ? "PASS" : "*** FAIL ***");
    if (g.mismatch != 0) ok = false;

    // ---- COVERAGE: MEASURED, printed ----
    std::printf("\n--- corpus coverage (MEASURED) ---\n");
    std::printf("  distinct grid idx exercised : %zu / 2048  %s\n", cov_idx.size(),
                cov_idx.size() == 2048 ? "(COMPLETE)" : "(INCOMPLETE)");
    bool dok = true;
    for (int l = 0; l < 4; ++l) {
        std::printf("  delta slot %d               : +1 seen %lld, -1 seen %lld %s\n",
                    l, cov_delta[l][0], cov_delta[l][1],
                    (cov_delta[l][0] && cov_delta[l][1]) ? "(BOTH)" : "(*** MISSING ***)");
        if (!cov_delta[l][0] || !cov_delta[l][1]) dok = false;
    }
    int ls1seen = 0, ls2seen = 0;
    for (int i = 0; i < 8; ++i) { if (cov_ls1[i]) ++ls1seen; if (cov_ls2[i]) ++ls2seen; }
    std::printf("  ls1 steps seen              : %d/8   ls2 steps seen: %d/8\n",
                ls1seen, ls2seen);
    if (cov_idx.size() != 2048 || !dok || ls1seen != 8 || ls2seen != 8) ok = false;

    // ---- NEGATIVE CONTROLS ----
    // Each must SPIKE. A control that does not spike means the gate is blind to
    // that fault class -- which is exactly what "空心检查" means.
    std::printf("\n--- negative controls (ASYMMETRIC; each MUST spike) ---\n");
    struct { Mode rm, am, bm; const char *name; const char *what; } ctl[] = {
        {M_NONE, M_NONE, M_SIGNSWAP, "SIGNSWAP",
         "B-side: sign vector halves swapped (group<->lane alignment)"},
        {M_NONE, M_NONE, M_SIGNALL1, "SIGNALL1",
         "B-side: sign vector forced all +1 (delta dropped from the batch)"},
        {M_NONE, M_NONE, M_LANEROT,  "LANEROT",
         "B-side: sign vector rotated one lane (off-by-one lane)"},
        {M_GRIDPERT, M_NONE, M_NONE, "GRIDPERT",
         "REF-side: grid index perturbed -- proves the gate also covers the "
         "sumi1 axis, i.e. it is not a sum2-only gate wearing a hat"},
        {M_NONE, M_BSUMS16, M_NONE, "A_BSUMS16",
         "A-side: emitter fold forced to share a delta across each 16-group -- "
         "proves the gate would catch a bsums-shaped regression on the OLD fold"},
        {M_NONE, M_NONE, M_BSUMS16, "B_BSUMS16",
         "B-side: batched fold forced to share a delta across each 16-group -- "
         "this is THE fault the batching could plausibly introduce, since "
         "batching is exactly what puts two 8-groups in one 16-lane reduce"},
    };
    for (auto &c : ctl) {
        Res r = run(c.rm, c.am, c.bm, true, NB);
        bool spiked = r.mismatch > 0;
        std::printf("  [%-9s] mismatch=%6lld/%lld  %s\n     ^ %s\n",
                    c.name, r.mismatch, r.total,
                    spiked ? "SPIKED (gate sees it)" : "*** SILENT -> BLIND GATE ***",
                    c.what);
        if (!spiked) ok = false;
    }

    // ---- FORMAT-PROPERTY MEASUREMENT (not a control on the mechanism) ----
    // C4a-3's insight, measured: block_q8_K's bsums are per-16, iq1_m's delta is
    // per-8 with four independent bits => one bsums entry spans two 8-groups that
    // can carry OPPOSITE signs. If that reshaping changed nothing, bsums would be
    // usable and the 32 tiny reductions would be unnecessary. Measure it.
    long long bs = measure_bsums16_inexpressive(NB);
    std::printf("\n--- format property: can per-16 bsums express iq1_m's delta? "
                "(MEASURED) ---\n");
    std::printf("  REF(clean) vs REF(two 8-groups share a delta): differ on "
                "%lld/%lld blocks\n", bs, NB);
    std::printf("  => per-16 bsums %s express iq1_m's per-8 delta term.\n",
                bs > 0 ? "CANNOT" : "CAN (!!)");
    std::printf("  This is WHY the 32 tiny per-group reductions exist. It is a\n"
                "  statement about the FORMAT, not about the batching mechanism.\n");
    if (bs == 0) ok = false;

    // ---- WHAT THIS GATE CANNOT CATCH (say it out loud) ----
    std::printf("\n--- this gate is BLIND to (named, not hidden) ---\n");
    std::printf("  * emission. No RVV instruction is built or run here. A "
                "correct-algebra\n    emitter bug (wrong vl, wrong LMUL, wrong "
                "intrinsic, bad vsetvl order)\n    passes this gate silently. "
                "[K-5] byte-exact needs the board.\n");
    std::printf("  * qh bit-position errors shared by REF and MODEL: both read "
                "the same\n    bit positions, because those positions ARE the "
                "format.\n");
    std::printf("  * anything on the fp32 d/scale-assembly path (not modelled).\n");

    std::printf("\n=== %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
