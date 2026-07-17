// grid4_gemm_prefill_p2.cpp — P2 board batch: {iq1_s,iq1_m,iq3_xxs,iq3_s} gemm_tile PREFILL nr=16.
//
// WHY THIS FILE EXISTS ([L-10] "deployed != proven"):
//   C4a proved these four formats byte-exact on x86 ONLY, and what it compared was
//   tools/oracle-repack/oracle_repack_<fmt>.cpp's ref_block() (ggml-canonical) against that
//   file's OWN mine_col() -- whose comment literally reads "EMITTER-MODEL". That is
//   "ggml reference" vs "our x86 hand-model of our own emitter". It does NOT prove the
//   RISC-V that weft-opt actually emits computes the right answer on real silicon.
//   This driver keeps ref_block() as the oracle and replaces mine_col() with THE REAL
//   BOARD-COMPILED LEAF. Same oracle, real DUT instead of a modelled DUT => strictly
//   stronger than the x86 pass, and it reuses already-audited reference code.
//
// TWO CORRECTNESS LAYERS (PREREG §3.3). Both must pass before ANY timing runs.
//   T1 = BOARD INTEGER-PATH BYTE-EXACT. The leaf only exports f32 (sumi/sumi1 never leave),
//        so a naive compare could only be a tolerance compare -- which is NOT byte-exact.
//        Fix: build a corpus on which the fp fold is PROVABLY EXACT: nb=1 (K=256) and
//        d_x*d_y = 1.0 (fp16 0x3C00 / f32 1.0f). Then out(r,c) == C*m exactly, where
//        m = INV_C*a + b is an integer and C in {0.125,0.125,0.25,1.0} is a power of two.
//        Verdict is f32 `==` (bit equality), never a tolerance.
//        *** The exactness PRECONDITION is machine-checked per (r,c) at runtime (|a|,|b|,|m|
//        < 2^24 and C a power of two). If it fails we print and exit
//        VOID-EXACTNESS-PRECONDITION. We do NOT silently fall back to a tolerance gate.
//        (Prereg risk #2: I hand-verified the bound for iq1_s ONLY; the machine checks all
//        four so no number here rests on my arithmetic.) ***
//   T2 = MEASUREMENT-SHAPE gate. T1's shape (nb=1) is NOT the shape we time (nb=64), so the
//        timed shape gets its own gate: K=2048, real random d => cross-block f32 reassociation
//        genuinely exists => principled forward-error tolerance
//        tol = TOLK*nb*FLT_EPSILON*amag, amag = oracle's sum of |per-block contribution| in
//        double. worst_tolratio is PRINTED (hollow gate ~1; live gate <<1, faults >>1).
//
// [K-5b] THREE REQUIREMENTS
//   (3) ORACLE INDEPENDENT: oracle reads the ORIGINAL plain block and derives ls/delta/idx/
//       signs from raw qh/aux/scale words, reading the grid via ggml's own uint64/uint32
//       reinterpret. The DUT reads REPACKED flat strips and gathers from a flat table with
//       vluxei16. Different layout, different index derivation, different table read. The
//       only shared thing is the grid CONSTANT -- and even that is TWO INDEPENDENT COPIES:
//       ours from tools/oracle-repack/*.h, the leaf's rebuilt by the compiler from the
//       registry as weft_<fmt>_grid. A wrong registry table is therefore CAUGHT, not shared.
//   (2) INPUT PATH SAME-SOURCE: ONE rng makes plain W + plain A. Every consumer is a pure
//       function of those two: packed = make_x16(W), apack = make_q8_Kx4(A), OPP-G/OPP-X eat
//       plain W/A, oracle reads plain W/A. Zero second RNG stream (that was P1's VOID cause).
//       Zero intermediate capture/replay.
//   (1) CORPUS COMPLETENESS: the T1 generator SWEEPS the decode axes systematically (idx runs
//       a full cycle; ls runs a stride-coprime cycle; sign/delta bits driven by independent
//       counter bits) and coverage COUNTERS print the result. Short on any axis => non-zero
//       exit => VOID-CORPUS. Completeness is claimed for T1 ONLY; T2 uses random d and makes
//       NO completeness claim. The two are reported separately and neither implies the other.
//
// ACTIVATION RANGE, stated honestly: T1 uses a per-format ACT_MAX so the integer certificate
//   provably stays under 2^24 (iq3_xxs's grid bytes reach 62 and its ls reaches 31, so the
//   full +-127 range would overflow the f32 mantissa and make byte-exactness unprovable).
//   Activation MAGNITUDE is not a coverage axis (the grid/ls/delta/sign axes are, and they
//   stay fully swept); the full random range is exercised by T2 under its tolerance gate.
//   The runtime precondition check runs REGARDLESS -- ACT_MAX is a way to satisfy the gate,
//   never a way to skip it.
//
// ABI (PREREG §3.5 + [GOV-8-B]) -- verified from the emitted C, not assumed, and NOT copied
//   from P1 (P1's iq4_nl leaf is (n,s,vx,vy,nr,nc,bs); this family is a DIFFERENT ORDER):
//     (size_t nr, size_t bs, size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc)
//        v1          v2          v3         v4              v5                v6           v7
//   Confirmed by the emitter's own callee= annotations: v11=v1/4 `row_group_count`,
//   v12=v7/16 `col_group_count`, v10=v3/256 `block_count`, v6+.. `act_group_base` (x1168),
//   v5+.. `weight_group_base` (x1312), and `output_addr` = v4 + (v13*4+r)*v2 + v17*16 + h*8.
//   Under LP64D ints and pointers share a0-a7, so a misbind need not crash -- it can silently
//   produce plausible numbers. Hence the ABI SENTINEL GATE below, run before T1, with a
//   deliberate bs != nc (P1 always passed bs==nc, which would mask a bs/nc swap).
//
// FAULT INJECTION (proves the gate is not hollow):
//   -DINJECT=1  oracle constant fault (grid index rotated in ref_block)  => ALL THREE RED
//   -DINJECT=2  DUT output fault      (ours[mid] += 1.0f)                => OURS ONLY RED
//   -DINJECT=3  no driver change; linked against a leaf whose weft_<fmt>_grid table had ONE
//               byte flipped (the run script proves `cmp -l | wc -l` == 1)
//                                                                        => OURS ONLY RED
//   INJECT=3 is the arm the x86 pass could not have: it proves this gate bites THE REAL
//   EMITTED RISC-V, not a model and not the driver.
//
// argv: <K(mult256)> <nr(mult4)> <nc(mult16)> <reps> <seed> [verify_only]
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <time.h>
#include <riscv_vector.h>

#ifndef INJECT
#define INJECT 0
#endif
#ifndef TOLK
#define TOLK 4.0
#endif
#ifndef FLUSH_MB
#define FLUSH_MB 224
#endif

#define QK_K 256
typedef _Float16 ggml_half;

// ===========================================================================
// Per-format selection
// ===========================================================================
#if defined(FMT_IQ1_S)
  #define FMT_NAME "iq1_s"
  #include "iq1s_grid.h"
  #define FOLD_C   0.125
  #define FOLD_INV 8
  #define ACT_MAX  127     // |m| <= 8*487680+487680 = 4,389,120 < 2^24 (machine-rechecked)
  #define USES_BSUMS 1
#elif defined(FMT_IQ1_M)
  #define FMT_NAME "iq1_m"
  #include "iq1s_grid.h"   // iq1_m shares the iq1_s ternary grid
  #define FOLD_C   0.125
  #define FOLD_INV 8
  #define ACT_MAX  127
  #define USES_BSUMS 0
#elif defined(FMT_IQ3_XXS)
  #define FMT_NAME "iq3_xxs"
  #include "iq3xxs_tables.h"
  #define FOLD_C   0.25
  #define FOLD_INV 4
  // grid bytes reach 62, ls reaches 31 => worst |bsum| <= 8*31*32*62*ACT_MAX.
  // ACT_MAX=30 => 14,760,960 < 2^24 = 16,777,216. Still machine-rechecked per (r,c).
  #define ACT_MAX  30
  #define USES_BSUMS 0
#elif defined(FMT_IQ3_S)
  #define FMT_NAME "iq3_s"
  #include "iq3s_tables.h"
  #define FOLD_C   1.0
  #define FOLD_INV 1
  // grid bytes reach 15, ls reaches 31 => worst |bsum| <= 8*31*32*15*ACT_MAX.
  // ACT_MAX=100 => 11,904,000 < 2^24.
  #define ACT_MAX  100
  #define USES_BSUMS 0
#else
  #error "define one of FMT_IQ1_S / FMT_IQ1_M / FMT_IQ3_XXS / FMT_IQ3_S"
#endif

// ===========================================================================
// PLAIN blocks — the REAL ggml-common.h layouts (fp16 d, not the oracles' float d)
// ===========================================================================
struct block_q8_K { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; };
static_assert(sizeof(block_q8_K)==292, "q8_K 292");

// INTERLEAVED activation: 4 fp32 d @0, 1024 int8 quants @16 as pos*4+c,
// 64 int16 bsums @1040 as g16*4+c. (Pinned by the fixture: stride 1168.)
struct block_q8_Kx4 { float d[4]; int8_t qs[1024]; int16_t bsums[64]; };
static_assert(sizeof(block_q8_Kx4)==1168, "q8_Kx4 1168");
static_assert(offsetof(block_q8_Kx4,qs)==16,     "q8_Kx4 qs@16");
static_assert(offsetof(block_q8_Kx4,bsums)==1040,"q8_Kx4 bsums@1040");

#if defined(FMT_IQ1_S)
struct block_iq1_s { ggml_half d; uint8_t qs[QK_K/8]; uint16_t qh[QK_K/32]; };
typedef block_iq1_s PlainW;
static_assert(sizeof(PlainW)==50, "iq1_s 50");
struct X16 { ggml_half d[16]; int8_t ls[8][16]; int8_t delta[8][16]; uint16_t gidx[8][4][16]; };
static_assert(sizeof(X16)==1312, "iq1_sx16 1312");
static_assert(offsetof(X16,ls)==32 && offsetof(X16,delta)==160 && offsetof(X16,gidx)==288, "iq1_sx16 planes");

#elif defined(FMT_IQ1_M)
struct block_iq1_m { uint8_t qs[QK_K/8]; uint8_t qh[QK_K/16]; uint8_t scales[QK_K/32]; };
typedef block_iq1_m PlainW;
static_assert(sizeof(PlainW)==56, "iq1_m 56");
struct X16 { ggml_half d[16]; int8_t ls[8][2][16]; int8_t delta[8][4][16]; uint16_t gidx[8][4][16]; };
static_assert(sizeof(X16)==1824, "iq1_mx16 1824");
static_assert(offsetof(X16,ls)==32 && offsetof(X16,delta)==288 && offsetof(X16,gidx)==800, "iq1_mx16 planes");

#elif defined(FMT_IQ3_XXS)
struct block_iq3_xxs { ggml_half d; uint8_t qs[3*QK_K/8]; };
typedef block_iq3_xxs PlainW;
static_assert(sizeof(PlainW)==98, "iq3_xxs 98");
struct X16 { ggml_half d[16]; int8_t ls[8][16]; uint8_t gidx[8][8][16]; uint8_t ssel[8][4][16]; };
static_assert(sizeof(X16)==1696, "iq3_xxsx16 1696");
static_assert(offsetof(X16,ls)==32 && offsetof(X16,gidx)==160 && offsetof(X16,ssel)==1184, "iq3_xxsx16 planes");

#elif defined(FMT_IQ3_S)
struct block_iq3_s { ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t signs[QK_K/8]; uint8_t scales[QK_K/64]; };
typedef block_iq3_s PlainW;
static_assert(sizeof(PlainW)==110, "iq3_s 110");
struct X16 { ggml_half d[16]; int8_t ls[8][16]; uint16_t gidx[8][8][16]; uint8_t signs[8][4][16]; };
static_assert(sizeof(X16)==2720, "iq3_sx16 2720");
static_assert(offsetof(X16,ls)==32 && offsetof(X16,gidx)==160 && offsetof(X16,signs)==2208, "iq3_sx16 planes");
#endif

// ===========================================================================
// THE REPACK (mat-quant) — the ONLY producer of the packed weight form, fed from plain W.
// Transcribed from tools/oracle-repack/oracle_repack_<fmt>.cpp's make_block_<fmt>x16, with
// the oracles' `float d[16]` modelling shortcut CORRECTED to the real fp16 d[16] strip the
// leaf actually loads (verified: leaf does __riscv_vle16_v_f16m1(v39,8) at +0 and +16).
// ===========================================================================
static X16 make_x16(const PlainW* in) {
    X16 o; memset(&o, 0, sizeof(o));
#if defined(FMT_IQ1_S)
    for (int c=0;c<16;++c) o.d[c] = in[c].d;
    for (int c=0;c<16;++c) for (int ib=0; ib<8; ++ib) {
        int qhw = in[c].qh[ib];
        o.ls[ib][c]    = (int8_t)(2*((qhw>>12)&7)+1);
        o.delta[ib][c] = (int8_t)(1-2*((qhw>>15)&1));
        for (int l=0;l<4;++l)
            o.gidx[ib][l][c] = (uint16_t)(in[c].qs[4*ib+l] | (((qhw>>(3*l))&7)<<8));
    }
#elif defined(FMT_IQ1_M)
    for (int c=0;c<16;++c) {
        const uint16_t* sc = (const uint16_t*)in[c].scales;
        uint16_t su = 0;
        for (int k=0;k<4;++k) su |= (uint16_t)(((sc[k]>>12)&0xf) << (4*k));
        ggml_half h; memcpy(&h, &su, 2); o.d[c] = h;   // the assembled fp16 scale strip
        for (int ib=0; ib<8; ++ib) {
            int shift = 6*(ib%2);
            o.ls[ib][0][c] = (int8_t)(2*((sc[ib/2]>>(shift+0))&0x7)+1);
            o.ls[ib][1][c] = (int8_t)(2*((sc[ib/2]>>(shift+3))&0x7)+1);
            for (int l=0;l<4;++l) {
                int qhb = in[c].qh[2*ib + l/2];
                int nib = (l%2) ? (qhb>>4) : qhb;
                o.delta[ib][l][c] = (int8_t)((nib & 0x8) ? -1 : 1);
                o.gidx[ib][l][c]  = (uint16_t)(in[c].qs[4*ib+l] | ((nib & 0x7)<<8));
            }
        }
    }
#elif defined(FMT_IQ3_XXS)
    for (int c=0;c<16;++c) o.d[c] = in[c].d;
    for (int c=0;c<16;++c) {
        const uint8_t* q3  = in[c].qs;
        const uint8_t* gas = in[c].qs + QK_K/4;
        for (int ib=0; ib<8; ++ib) {
            uint32_t aux32; memcpy(&aux32, gas + 4*ib, 4);
            o.ls[ib][c] = (int8_t)(2*(aux32>>28)+1);
            for (int l=0;l<4;++l) o.ssel[ib][l][c] = (uint8_t)((aux32>>(7*l)) & 127);
            for (int e=0;e<8;++e) o.gidx[ib][e][c] = q3[8*ib+e];
        }
    }
#elif defined(FMT_IQ3_S)
    for (int c=0;c<16;++c) o.d[c] = in[c].d;
    for (int c=0;c<16;++c) {
        const uint8_t* qs = in[c].qs;
        for (int ib=0; ib<8; ++ib) {
            unsigned nib = (in[c].scales[ib>>1] >> (4*(ib&1))) & 0xf;
            o.ls[ib][c] = (int8_t)(2*nib+1);
            for (int e=0;e<8;++e) {
                unsigned lo = qs[8*ib+e];
                unsigned hi = (in[c].qh[ib] >> e) & 1u;
                o.gidx[ib][e][c] = (uint16_t)(lo | (hi<<8));
            }
            for (int l=0;l<4;++l) o.signs[ib][l][c] = in[c].signs[4*ib+l];
        }
    }
#endif
    return o;
}

// The interleaved activation, fed from plain A. qs[pos*4+c], bsums[g16*4+c].
static block_q8_Kx4 make_q8_Kx4(const block_q8_K* in) {
    block_q8_Kx4 o; memset(&o, 0, sizeof(o));
    for (int r=0;r<4;++r) o.d[r] = in[r].d;
    for (int r=0;r<4;++r) {
        for (int p=0;p<QK_K;++p)   o.qs[p*4+r]    = in[r].qs[p];
        for (int g=0;g<QK_K/16;++g) o.bsums[g*4+r] = in[r].bsums[g];
    }
    return o;
}

// ===========================================================================
// ORACLE — ref_block(), transcribed from tools/oracle-repack/oracle_repack_<fmt>.cpp.
// Reads the ORIGINAL plain block and derives everything the ggml-canonical way. It never
// reads the packed forms, never calls the leaf, never calls ggml. Returns the TWO integer
// certificates (a,b); the fp fold is out = d_x*d_y*(a + FOLD_C*b).
// INJECT=1 rotates the grid index HERE (oracle-constant fault) => all three must go red.
// ===========================================================================
static void ref_block(const PlainW* x, const block_q8_K* y, int64_t& a_out, int64_t& b_out) {
    const int8_t* q8 = y->qs;
#if defined(FMT_IQ1_S)
    int64_t sumi=0, sumi1=0; int q8pos=0;
    const uint8_t* qs = x->qs; const uint16_t* qh = x->qh;
    for (int ib=0; ib<8; ++ib) {
        int qhw = qh[ib];
        int64_t ls = 2*((qhw>>12)&7)+1;
        int64_t delta = (qhw & 0x8000) ? -1 : 1;
        int64_t lsum = 0;
        for (int l=0;l<4;++l) {
            int idx = qs[4*ib+l] | (((qhw>>(3*l))&7)<<8);
#if INJECT==1
            idx = (idx + 457) & 2047;
#endif
            const int8_t* grid = (const int8_t*)(iq1s_grid + idx);   // ggml's own read
            for (int j=0;j<8;++j) lsum += (int64_t)q8[q8pos++] * grid[j];
        }
        sumi  += ls * lsum;
        sumi1 += ls * delta * ((int64_t)y->bsums[2*ib+0] + (int64_t)y->bsums[2*ib+1]);
    }
    a_out = sumi; b_out = sumi1;

#elif defined(FMT_IQ1_M)
    int64_t sumi1=0, sumi2=0; int q8pos=0;
    const uint8_t* qs = x->qs; const uint8_t* qh = x->qh;
    const uint16_t* sc = (const uint16_t*)x->scales;
    for (int ib=0; ib<8; ++ib) {
        int64_t sum1[2]={0,0}, sum2[2]={0,0};
        int delta[4];
        delta[0] = (qh[2*ib+0] & 0x08) ? -1 : 1;
        delta[1] = (qh[2*ib+0] & 0x80) ? -1 : 1;
        delta[2] = (qh[2*ib+1] & 0x08) ? -1 : 1;
        delta[3] = (qh[2*ib+1] & 0x80) ? -1 : 1;
        for (int l=0;l<4;++l) {
            int qhb = qh[2*ib + l/2];
            int idx = qs[4*ib+l] | (((uint16_t)qhb << (8 - 4*(l%2))) & 0x700);
#if INJECT==1
            idx = (idx + 457) & 2047;
#endif
            const int8_t* grid = (const int8_t*)(iq1s_grid + idx);
            int64_t lsum1=0, lsum2=0;
            for (int j=0;j<8;++j) { lsum1 += (int64_t)q8[q8pos+j]*grid[j]; lsum2 += (int64_t)q8[q8pos+j]; }
            q8pos += 8;
            sum1[l/2] += lsum1;
            sum2[l/2] += lsum2 * delta[l];
        }
        int shift = 6*(ib%2);
        int64_t ls1 = 2*((sc[ib/2]>>(shift+0))&0x7)+1;
        int64_t ls2 = 2*((sc[ib/2]>>(shift+3))&0x7)+1;
        sumi1 += sum1[0]*ls1 + sum1[1]*ls2;
        sumi2 += sum2[0]*ls1 + sum2[1]*ls2;
    }
    a_out = sumi1; b_out = sumi2;

#elif defined(FMT_IQ3_XXS)
    int64_t bsum=0; int q8pos=0;
    const uint8_t* q3 = x->qs; const uint8_t* gas = x->qs + QK_K/4;
    for (int ib=0; ib<8; ++ib) {
        uint32_t aux32; memcpy(&aux32, gas, 4); gas += 4;
        int64_t ls = 2*(aux32>>28)+1;
        int64_t sumi = 0;
        for (int l=0;l<4;++l) {
            int i0 = q3[2*l+0], i1 = q3[2*l+1];
#if INJECT==1
            i0 = (i0 + 37) & 255; i1 = (i1 + 37) & 255;
#endif
            const uint8_t signs = ksigns_iq2xs[(aux32 >> (7*l)) & 127];
            const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + i0);   // ggml's own read
            const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + i1);
            for (int j=0;j<4;++j) {
                int s0 = (signs & kmask_iq2xs[j+0]) ? -1 : 1;
                int s1 = (signs & kmask_iq2xs[j+4]) ? -1 : 1;
                sumi += (int64_t)grid1[j] * q8[q8pos+j+0] * s0;
                sumi += (int64_t)grid2[j] * q8[q8pos+j+4] * s1;
            }
            q8pos += 8;
        }
        q3 += 8;
        bsum += sumi * ls;
    }
    a_out = 0; b_out = bsum;

#elif defined(FMT_IQ3_S)
    int64_t bsum=0; int q8pos=0;
    const uint8_t* qs = x->qs; const uint8_t* signs = x->signs;
    for (int ib32=0; ib32 < QK_K/32; ib32 += 2) {
        const int lsv[2] = {2*(x->scales[ib32/2] & 0xf)+1, 2*(x->scales[ib32/2] >> 4)+1};
        for (int half=0; half<2; ++half) {
            int ib = ib32 + half;
            int64_t ls = lsv[half];
            int64_t sumi = 0;
            for (int l=0;l<4;++l) {
                int i0 = qs[2*l+0] | ((x->qh[ib] << (8 - 2*l)) & 256);
                int i1 = qs[2*l+1] | ((x->qh[ib] << (7 - 2*l)) & 256);
#if INJECT==1
                i0 = (i0 + 37) & 511; i1 = (i1 + 37) & 511;
#endif
                unsigned sb = signs[l];
                const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + i0);
                const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + i1);
                for (int j=0;j<4;++j) {
                    int s0 = (sb & kmask_iq2xs[j+0]) ? -1 : 1;
                    int s1 = (sb & kmask_iq2xs[j+4]) ? -1 : 1;
                    sumi += (int64_t)grid1[j] * q8[q8pos+j+0] * s0;
                    sumi += (int64_t)grid2[j] * q8[q8pos+j+4] * s1;
                }
                q8pos += 8;
            }
            qs += 8; signs += 4;
            bsum += sumi * ls;
        }
    }
    a_out = 0; b_out = bsum;
#endif
}

// the fp16 super-block scale, read the ggml way (the oracle's own read of d)
static double plain_scale(const PlainW* x) {
#if defined(FMT_IQ1_M)
    const uint16_t* sc = (const uint16_t*)x->scales;
    uint16_t su = 0;
    for (int k=0;k<4;++k) su |= (uint16_t)(((sc[k]>>12)&0xf) << (4*k));
    ggml_half h; memcpy(&h, &su, 2); return (double)(float)h;
#else
    return (double)(float)x->d;
#endif
}

// ===========================================================================
// CORPUS — systematic sweep + coverage counters ([K-5b](1)). Coverage is STRUCTURAL
// (a full cycle / coprime stride), and the counters PROVE it rather than assert it.
// ===========================================================================
static std::set<int> cov_idx, cov_sel, cov_lsset;
static long long cov_pol[2] = {0,0};
static long long g_idx=0, g_sb=0;

static void build_w_block(PlainW* x, std::mt19937& rng, int col, int blk, bool exact) {
    memset(x, 0, sizeof(*x));
    std::uniform_real_distribution<float> dd(0.001f, 0.05f);
    float dv = exact ? 1.0f : dd(rng);
#if defined(FMT_IQ1_S)
    x->d = (ggml_half)dv;
    for (int ib=0; ib<8; ++ib) {
        int lsF = (int)((g_sb*3)%8), sgn = (int)((g_sb/8)%2); ++g_sb;
        cov_lsset.insert(2*lsF+1); cov_pol[sgn]++;
        int qhw = (lsF<<12) | (sgn<<15);
        for (int l=0;l<4;++l) {
            int idx = (int)(g_idx%2048); ++g_idx; cov_idx.insert(idx);
            x->qs[4*ib+l] = (uint8_t)(idx & 0xff);
            qhw |= ((idx>>8)&7) << (3*l);
        }
        x->qh[ib] = (uint16_t)qhw;
    }
#elif defined(FMT_IQ1_M)
    uint16_t sc[4] = {0,0,0,0};
    for (int ib=0; ib<8; ++ib) {
        int ls1f = (int)((g_sb*3)%8), ls2f = (int)((g_sb*5+2)%8);
        cov_lsset.insert(2*ls1f+1); cov_lsset.insert(2*ls2f+1);
        int shift = 6*(ib%2);
        sc[ib/2] |= (uint16_t)((ls1f & 7) << (shift+0));
        sc[ib/2] |= (uint16_t)((ls2f & 7) << (shift+3));
        for (int l=0;l<4;++l) {
            int idx = (int)(g_idx%2048); ++g_idx; cov_idx.insert(idx);
            x->qs[4*ib+l] = (uint8_t)(idx & 0xff);
            int dbit = (int)((g_sb>>l)&1); cov_pol[dbit]++;
            int nib = ((idx>>8)&7) | (dbit<<3);
            int qi = 2*ib + l/2;
            if (l%2==0) x->qh[qi] = (uint8_t)nib;
            else        x->qh[qi] = (uint8_t)(x->qh[qi] | (nib<<4));
        }
        ++g_sb;
    }
    { // the fp16 scale nibbles live in bits 12-15 of each word, disjoint from the ls fields
      ggml_half h = (ggml_half)dv; uint16_t want; memcpy(&want, &h, 2);
      for (int k=0;k<4;++k) sc[k] |= (uint16_t)(((want >> (4*k)) & 0xf) << 12);
      memcpy(x->scales, sc, 8); }
#elif defined(FMT_IQ3_XXS)
    x->d = (ggml_half)dv;
    uint8_t* gas = x->qs + QK_K/4;
    for (int ib=0; ib<8; ++ib) {
        for (int e=0;e<8;++e) { int idx=(int)(g_idx%256); ++g_idx; cov_idx.insert(idx); x->qs[8*ib+e]=(uint8_t)idx; }
        int lsF = (int)((g_sb*3)%16); // 3 coprime with 16 => all 16 ls steps
        uint32_t aux32 = ((uint32_t)lsF) << 28;
        cov_lsset.insert(2*lsF+1);
        for (int l=0;l<4;++l) { int sel=(int)((g_sb*5+l*13)%128); cov_sel.insert(sel); aux32 |= ((uint32_t)sel) << (7*l); }
        ++g_sb;
        memcpy(gas + 4*ib, &aux32, 4);
    }
#elif defined(FMT_IQ3_S)
    x->d = (ggml_half)dv;
    for (int ib=0; ib<8; ++ib) {
        for (int e=0;e<8;++e) {
            int idx = (int)(g_idx%512); ++g_idx; cov_idx.insert(idx);
            x->qs[8*ib+e] = (uint8_t)(idx & 0xff);
            if (idx & 256) x->qh[ib] |= (uint8_t)(1u<<e);
        }
        for (int l=0;l<4;++l) { int sb=(int)((g_sb*7+l*29)%256); cov_sel.insert(sb); x->signs[4*ib+l]=(uint8_t)sb; }
        int nib = (int)((g_sb*3)%16); cov_lsset.insert(2*nib+1);
        x->scales[ib>>1] = (uint8_t)((ib&1) ? (x->scales[ib>>1] | (nib<<4)) : nib);
        ++g_sb;
    }
#endif
}

// A REAL block_q8_K: bsums[g] is the TRUE sum of its 16 quants (what quantize_row_q8_K
// computes). iq1_s's delta term reads them, so they must not be degenerate.
static void build_a_block(block_q8_K* a, std::mt19937& rng, int blk, bool exact) {
    std::uniform_real_distribution<float> dd(0.001f, 0.05f);
    std::uniform_int_distribution<int> q8d(-ACT_MAX, ACT_MAX);
    std::uniform_int_distribution<int> q8f(-127, 127);
    a->d = exact ? 1.0f : dd(rng);
    for (int i=0;i<QK_K;++i) a->qs[i] = (int8_t)(exact ? q8d(rng) : q8f(rng));
    for (int g=0; g<QK_K/16; ++g) { int s=0; for (int j=0;j<16;++j) s += a->qs[g*16+j]; a->bsums[g]=(int16_t)s; }
}

// ===========================================================================
// DUT — the front-door lowered repack GEMM leaf. ABI verified above.
// OPP-G = registered opponent (T3): ggml's `_generic` arch-fallback scalar reference.
// OPP-X = as-shipped real path (CROSSOP): dispatch thunk -> hand-tuned _vlNNN.
// OPP-S = same-operator repack GEMM: DOES NOT EXIST for these four (probed ABSENT on both
//         boards) => no hand-brick opponent => hand-brick win is structurally unavailable.
// ===========================================================================
#define XCAT(a,b) a##b
#define CAT(a,b) XCAT(a,b)
#if defined(FMT_IQ1_S)
  #define LEAF weft_emitc_ggml_repack_gemm_iq1_s_q8_K_kernel_ggml_repack_gemm_iq1_s_q8_K
  #define OPPG ggml_vec_dot_iq1_s_q8_K_generic
  #define OPPX ggml_vec_dot_iq1_s_q8_K
#elif defined(FMT_IQ1_M)
  #define LEAF weft_emitc_ggml_repack_gemm_iq1_m_q8_K_kernel_ggml_repack_gemm_iq1_m_q8_K
  #define OPPG ggml_vec_dot_iq1_m_q8_K_generic
  #define OPPX ggml_vec_dot_iq1_m_q8_K
#elif defined(FMT_IQ3_XXS)
  #define LEAF weft_emitc_ggml_repack_gemm_iq3_xxs_q8_K_kernel_ggml_repack_gemm_iq3_xxs_q8_K
  #define OPPG ggml_vec_dot_iq3_xxs_q8_K_generic
  #define OPPX ggml_vec_dot_iq3_xxs_q8_K
#elif defined(FMT_IQ3_S)
  #define LEAF weft_emitc_ggml_repack_gemm_iq3_s_q8_K_kernel_ggml_repack_gemm_iq3_s_q8_K
  #define OPPG ggml_vec_dot_iq3_s_q8_K_generic
  #define OPPX ggml_vec_dot_iq3_s_q8_K
#endif

extern "C" void LEAF(size_t nr, size_t bs, size_t n, float* s,
                     const uint8_t* vx, const uint8_t* vy, size_t nc);
extern "C" void OPPG(int n, float* s, size_t bs, const void* vx, size_t bx,
                     const void* vy, size_t by, int nrc);
extern "C" void OPPX(int n, float* s, size_t bs, const void* vx, size_t bx,
                     const void* vy, size_t by, int nrc);

// ===========================================================================
// timing helpers
// ===========================================================================
static const size_t FLUSH = (size_t)FLUSH_MB*1024*1024;
static uint8_t* g_flush=nullptr; static volatile uint64_t g_sink=0;
static void cold_flush(){ uint64_t a=0; for(size_t i=0;i<FLUSH;i+=64){ g_flush[i]=(uint8_t)(g_flush[i]+1); a+=g_flush[i]; } g_sink+=a; }
static double now_ns(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }
static double med(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
static double riqr(std::vector<double> v){ std::sort(v.begin(),v.end()); size_t n=v.size(); double m=v[n/2];
    return m>0 ? 100.0*(v[3*n/4]-v[n/4])/m : 0.0; }

int main(int argc, char** argv){
    if (argc < 6){ fprintf(stderr,"usage: %s K nr nc reps seed [verify_only]\n", argv[0]); return 2; }
    int K=atoi(argv[1]), nr=atoi(argv[2]), nc=atoi(argv[3]), reps=atoi(argv[4]);
    unsigned seed=(unsigned)strtoul(argv[5],0,0);
    bool vo = argc>6 && atoi(argv[6]);
    if (K%QK_K){ fprintf(stderr,"K must be mult of 256\n"); return 2; }
    if (nr%4 || nc%16){ fprintf(stderr,"nr mult4 / nc mult16\n"); return 2; }
    int nb=K/QK_K, ng=nc/16, nrg=nr/4, no=nr*nc;
    long vlen=(long)__riscv_vlenb()*8;
    printf("=== %s gemm_tile PREFILL | K=%d nr=%d nc=%d VLEN=%ld seed=0x%X INJECT=%d TOLK=%.1f ACT_MAX=%d\n",
           FMT_NAME,K,nr,nc,vlen,seed,INJECT,(double)TOLK,ACT_MAX);

    // ---------------- ABI SENTINEL GATE (PREREG §3.5) ----------------------
    // Deliberately bs != nc so a bs/nc swap cannot hide (P1 always passed bs==nc).
    {
        int anr=8, anc=32, abs_=anc+16, aK=QK_K;
        int anb=aK/QK_K, ang=anc/16, anrg=anr/4;
        std::mt19937 arng(0xABC1);
        std::vector<PlainW> W((size_t)anc*anb);
        for (int c=0;c<anc;++c) for (int l=0;l<anb;++l) build_w_block(&W[(size_t)c*anb+l], arng, c, l, true);
        std::vector<block_q8_K> A((size_t)anr*anb);
        for (int r=0;r<anr;++r) for (int l=0;l<anb;++l) build_a_block(&A[(size_t)r*anb+l], arng, l, true);
        std::vector<X16> P((size_t)ang*anb);
        for (int g=0;g<ang;++g) for (int l=0;l<anb;++l){ PlainW t[16];
            for (int c=0;c<16;++c) t[c]=W[(size_t)(g*16+c)*anb+l]; P[(size_t)g*anb+l]=make_x16(t); }
        std::vector<block_q8_Kx4> AP((size_t)anrg*anb);
        for (int y=0;y<anrg;++y) for (int l=0;l<anb;++l){ block_q8_K t[4];
            for (int r=0;r<4;++r) t[r]=A[(size_t)(y*4+r)*anb+l]; AP[(size_t)y*anb+l]=make_q8_Kx4(t); }
        int rows_alloc = anr+4;
        std::vector<float> s((size_t)rows_alloc*abs_);
        for (auto& v : s) v = NAN;
        LEAF((size_t)anr,(size_t)abs_,(size_t)aK,s.data(),(const uint8_t*)P.data(),(const uint8_t*)AP.data(),(size_t)anc);
        long wrote_in=0, miss_in=0, clobber_out=0;
        for (int r=0;r<rows_alloc;++r) for (int c=0;c<abs_;++c){
            bool inside = (r<anr && c<anc); float v = s[(size_t)r*abs_+c];
            if (inside){ if (std::isnan(v)) miss_in++; else wrote_in++; }
            else if (!std::isnan(v)) clobber_out++;
        }
        printf("  ABI-GATE nr=%d nc=%d bs=%d(!=nc) K=%d : wrote_in=%ld/%d miss_in=%ld clobber_out=%ld -> %s\n",
               anr,anc,abs_,aK,wrote_in,anr*anc,miss_in,clobber_out,
               (miss_in==0 && clobber_out==0) ? "PASS" : "VOID-ABI");
        if (miss_in || clobber_out){
            fprintf(stderr,"VOID-ABI: leaf ABI binding does not match (nr,bs,n,s,vx,vy,nc)\n"); return 7; }
        cov_idx.clear(); cov_sel.clear(); cov_lsset.clear(); cov_pol[0]=cov_pol[1]=0; g_idx=0; g_sb=0;
    }

    // ---------------- single source of truth: plain W, plain A ------------------
    bool exact = (nb==1);       // T1 exactness corpus iff nb==1
    std::mt19937 rng(seed);
    std::vector<PlainW> W((size_t)nc*nb);
    for (int c=0;c<nc;++c) for (int l=0;l<nb;++l) build_w_block(&W[(size_t)c*nb+l], rng, c, l, exact);
    std::vector<block_q8_K> A((size_t)nr*nb);
    for (int r=0;r<nr;++r) for (int l=0;l<nb;++l) build_a_block(&A[(size_t)r*nb+l], rng, l, exact);

    // ---------------- derived views (pure functions of W / A) -------------------
    std::vector<X16> packed((size_t)ng*nb);
    for (int g=0;g<ng;++g) for (int l=0;l<nb;++l){ PlainW t[16];
        for (int c=0;c<16;++c) t[c]=W[(size_t)(g*16+c)*nb+l]; packed[(size_t)g*nb+l]=make_x16(t); }
    std::vector<block_q8_Kx4> apack((size_t)nrg*nb);
    for (int y=0;y<nrg;++y) for (int l=0;l<nb;++l){ block_q8_K t[4];
        for (int r=0;r<4;++r) t[r]=A[(size_t)(y*4+r)*nb+l]; apack[(size_t)y*nb+l]=make_q8_Kx4(t); }

    std::vector<float> ours(no,0.f), oG(no,0.f), oX(no,0.f);
    LEAF((size_t)nr,(size_t)nc,(size_t)K,ours.data(),(const uint8_t*)packed.data(),
         (const uint8_t*)apack.data(),(size_t)nc);
#if INJECT==2
    ours[no/2] += 1.0f;   // *** DELIBERATE DUT-OUTPUT FAULT (anti-hollow proof) ***
#endif
    for (int r=0;r<nr;++r) for (int c=0;c<nc;++c)
        OPPG(K,&oG[(size_t)r*nc+c],0,W.data()+(size_t)c*nb,0,A.data()+(size_t)r*nb,0,1);
    for (int r=0;r<nr;++r) for (int c=0;c<nc;++c)
        OPPX(K,&oX[(size_t)r*nc+c],0,W.data()+(size_t)c*nb,0,A.data()+(size_t)r*nb,0,1);

    // ---------------- ORACLE per (r,c) from PLAIN W/A only ---------------------
    std::vector<double> ora(no,0.0), tol(no,0.0);
    std::vector<float>  exp1(no,0.f);
    const double C = FOLD_C;
    const int64_t LIM = 1LL<<24;
    long pre_bad = 0; double worst_m = 0;
    for (int r=0;r<nr;++r) for (int c=0;c<nc;++c){
        double s=0.0, amag=0.0; size_t i=(size_t)r*nc+c;
        int64_t A1=0,B1=0;
        for (int l=0;l<nb;++l){
            int64_t a,b; ref_block(&W[(size_t)c*nb+l], &A[(size_t)r*nb+l], a, b);
            double d = plain_scale(&W[(size_t)c*nb+l]) * (double)A[(size_t)r*nb+l].d;
            double t = d * ((double)a + C*(double)b);
            s += t; amag += fabs(t);
            if (nb==1){ A1=a; B1=b; }
        }
        ora[i]=s; tol[i]=(double)TOLK*(double)nb*(double)FLT_EPSILON*amag;
        if (exact){
            // *** machine-checked exactness precondition — no hand arithmetic is trusted ***
            int64_t m = (int64_t)FOLD_INV*A1 + B1;
            if (std::llabs(A1)>=LIM || std::llabs(B1)>=LIM || std::llabs(m)>=LIM) pre_bad++;
            if ((double)std::llabs(m) > worst_m) worst_m = (double)std::llabs(m);
            exp1[i] = (float)(C * (double)m);
        }
    }

    // ---------------- T1 : BOARD INTEGER-PATH BYTE-EXACT (f32 `==`) ------------
    int mism1=-1;
    if (exact){
        // C must be a power of two for C*m to be exact
        double cm = C; int cexp=0; bool cpow2 = (frexp(cm,&cexp)==0.5);
        printf("  T1-PRECOND fold_C=%.4f power_of_two=%s inv=%d worst|m|=%.0f limit=2^24=%lld violations=%ld -> %s\n",
               C, cpow2?"yes":"NO", FOLD_INV, worst_m, (long long)LIM, pre_bad,
               (cpow2 && pre_bad==0) ? "PASS" : "VOID-EXACTNESS-PRECONDITION");
        if (!cpow2 || pre_bad){
            fprintf(stderr,"VOID-EXACTNESS-PRECONDITION: fp fold not provably exact on this corpus;"
                           " refusing to downgrade to a tolerance gate\n"); return 8; }
        mism1 = 0;
        for (int i=0;i<no;++i) if (!(ours[i] == exp1[i])) mism1++;
        printf("  T1 BOARD-BYTE-EXACT (criterion: f32 == , nb=1, d_x*d_y=1.0): mism=%d/%d -> %s\n",
               mism1,no, mism1==0?"PASS":"VOID-CORRECTNESS");
        // coverage
        int nls = (int)cov_lsset.size();
        printf("  CORPUS grid_idx=%zu ls_steps=%d polarity=[%lld,%lld] sel=%zu\n",
               cov_idx.size(), nls, cov_pol[0], cov_pol[1], cov_sel.size());
        bool covok = true;
#if defined(FMT_IQ1_S)
        covok = (cov_idx.size()==2048 && nls==8 && cov_pol[0]>0 && cov_pol[1]>0);
        printf("  CORPUS-GATE idx %zu/2048 ls %d/8 delta both=%s -> %s\n", cov_idx.size(), nls,
               (cov_pol[0]>0&&cov_pol[1]>0)?"yes":"no", covok?"COMPLETE":"VOID-CORPUS");
#elif defined(FMT_IQ1_M)
        covok = (cov_idx.size()==2048 && nls==8 && cov_pol[0]>0 && cov_pol[1]>0);
        printf("  CORPUS-GATE idx %zu/2048 ls %d/8 delta both=%s -> %s\n", cov_idx.size(), nls,
               (cov_pol[0]>0&&cov_pol[1]>0)?"yes":"no", covok?"COMPLETE":"VOID-CORPUS");
#elif defined(FMT_IQ3_XXS)
        covok = (cov_idx.size()==256 && nls==16 && cov_sel.size()==128);
        printf("  CORPUS-GATE idx %zu/256 ls %d/16 sign_sel %zu/128 -> %s\n", cov_idx.size(), nls,
               cov_sel.size(), covok?"COMPLETE":"VOID-CORPUS");
#elif defined(FMT_IQ3_S)
        covok = (cov_idx.size()==512 && nls==16 && cov_sel.size()==256);
        printf("  CORPUS-GATE idx %zu/512 ls %d/16 sign_byte %zu/256 -> %s\n", cov_idx.size(), nls,
               cov_sel.size(), covok?"COMPLETE":"VOID-CORPUS");
#endif
        if (!covok){ fprintf(stderr,"VOID-CORPUS: a decode axis was not fully exercised\n"); return 9; }
    }

    // ---------------- T2 : measurement-shape tolerance gate --------------------
    int mism2=0, mismG=0, mismX=0; double wtr=0, wtrG=0, wtrX=0;
    for (int i=0;i<no;++i){
        double e=fabs((double)ours[i]-ora[i]); double tr=(tol[i]>0)?e/tol[i]:(e>0?1e30:0);
        if (tr>1.0) mism2++; if (tr>wtr) wtr=tr;
        double eg=fabs((double)oG[i]-ora[i]); double trg=(tol[i]>0)?eg/tol[i]:(eg>0?1e30:0);
        if (trg>1.0) mismG++; if (trg>wtrG) wtrG=trg;
        double ex=fabs((double)oX[i]-ora[i]); double trx=(tol[i]>0)?ex/tol[i]:(ex>0?1e30:0);
        if (trx>1.0) mismX++; if (trx>wtrX) wtrX=trx;
    }
    printf("  T2 ours  vs oracle: mism=%d/%d worst_tolratio=%.3e\n", mism2,no,wtr);
    printf("  T2 OPP-G vs oracle: mism=%d/%d worst_tolratio=%.3e\n", mismG,no,wtrG);
    printf("  T2 OPP-X vs oracle: mism=%d/%d worst_tolratio=%.3e\n", mismX,no,wtrX);
    printf("  GATE-OURS = %-4s | GATE-OPPG = %-4s | GATE-OPPX = %-4s\n",
           mism2==0?"PASS":"FAIL", mismG==0?"PASS":"FAIL", mismX==0?"PASS":"FAIL");

    if (vo) return (mism2==0 && (!exact || mism1==0)) ? 0 : 1;
    if (mism2 != 0 || (exact && mism1 != 0)){
        fprintf(stderr,"VOID-CORRECTNESS: ours mismatched => perf numbers forbidden\n"); return 1; }

    // ---------------- COLD paired A/B/C, flush before EACH timed region --------
    g_flush=(uint8_t*)aligned_alloc(64,FLUSH); if(!g_flush){ fprintf(stderr,"OOM flush\n"); return 3; }
    memset(g_flush,1,FLUSH);
    std::vector<double> tR,tG,tX;
    std::vector<float> bR(no),bG(no),bX(no);
    for (int p=0;p<reps;++p){
        cold_flush(); double t0=now_ns();
        LEAF((size_t)nr,(size_t)nc,(size_t)K,bR.data(),(const uint8_t*)packed.data(),
             (const uint8_t*)apack.data(),(size_t)nc);
        tR.push_back(now_ns()-t0);
        cold_flush(); double t1=now_ns();
        for (int r=0;r<nr;++r) for (int c=0;c<nc;++c)
            OPPG(K,&bG[(size_t)r*nc+c],0,W.data()+(size_t)c*nb,0,A.data()+(size_t)r*nb,0,1);
        tG.push_back(now_ns()-t1);
        cold_flush(); double t2=now_ns();
        for (int r=0;r<nr;++r) for (int c=0;c<nc;++c)
            OPPX(K,&bX[(size_t)r*nc+c],0,W.data()+(size_t)c*nb,0,A.data()+(size_t)r*nb,0,1);
        tX.push_back(now_ns()-t2);
    }
    double rM=med(tR), gM=med(tG), xM=med(tX);
    printf("REPS_OURS seed=0x%X :",seed); for(double v:tR) printf(" %.0f",v); printf("\n");
    printf("REPS_OPPG seed=0x%X :",seed); for(double v:tG) printf(" %.0f",v); printf("\n");
    printf("REPS_OPPX seed=0x%X :",seed); for(double v:tX) printf(" %.0f",v); printf("\n");
    printf("GEMM_PREFILL fmt=%s K=%d nr=%d nc=%d reps=%d VLEN=%ld seed=0x%X | ours_med_ns=%.0f(iqr%.2f%%) "
           "oppG_med_ns=%.0f(iqr%.2f%%) oppX_med_ns=%.0f(iqr%.2f%%) | ratio_cold_G=%.4f ratio_cold_X=%.4f sink=%llu\n",
           FMT_NAME,K,nr,nc,reps,vlen,seed,rM,riqr(tR),gM,riqr(tG),xM,riqr(tX),gM/rM,xM/rM,
           (unsigned long long)g_sink);
    free(g_flush); return 0;
}
