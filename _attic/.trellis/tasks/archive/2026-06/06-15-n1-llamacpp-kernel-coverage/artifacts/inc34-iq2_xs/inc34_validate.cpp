// INC-34 iq2_xs x q8_K byte-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.iq2_xs_q8_k_block_dot op (the
// COMPLETE ggml ggml_vec_dot_iq2_xs_q8_K GRID-codebook super-block kernel) computes the
// SAME fp32 result *s as ggml's OWN _generic iq2_xs reference, BIT-FOR-BIT (memcmp of
// the float bits), over random block_iq2_xs x block_q8_K arrays at n multiples of 256 +
// named edge cases (grid index range into [256,511], sign patterns, scale extremes incl
// distinct ls1/ls2, q8 +/-127). FOUR negative controls (a WRONG grid + WRONG signs +
// WRONG scale-bias + a SCALE-SPLIT-COLLAPSE that applies ls1 to all 4 groups) must FAIL
// -> proves the 512-grid lookup, the sign plane, the explicit scale array, AND the
// two-scale half split are the live, load-bearing mechanisms.
//
// iq2_xs is the SIBLING of iq2_xxs -- the SECOND member of the deep IQ tail. It REUSES
// iq2_xxs's grid + sign mechanism with THREE structural deltas: a 512-entry grid, a
// 9-bit index read DIRECTLY from each uint16 qs word (q2[l]&511 / q2[l]>>9), and an
// EXPLICIT per-sub-block scales[] array (ls1 = 2*(sc&0xf)+1 for groups 0,1; ls2 =
// 2*(sc>>4)+1 for groups 2,3).
//
// REFERENCE: ggml's VERBATIM _generic (quants.c:897-945): d = fp16(x.d)*y.d; for each
// of 8 sub-blocks, ls1/ls2 from sc[ib32]; two halves {0,1} scaled by ls1, {2,3} by ls2;
// per group grid = (uint8*)(iq2xs_grid + (q2[l]&511)), signs = ksigns[q2[l]>>9], sumi +=
// grid[j]*q8[j]*(signs&kmask?-1:1); bsum += sumi*ls; sumf += d*bsum; *s = 0.125f*sumf.
// FP16->FP32 mirrors the board's scalar _Float16 widening (lossless). Whole TU built
// with the same -ffp-contract.
//
// The kernel under test is the UNMODIFIED, compiler-emitted iq2_xs kernel
// (iq2_xs_emitted.cpp) -- every line tagged source_op=tcrv_rvv.iq2_xs_q8_k_block_dot.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>

#define QK_K 256
static const int IQ2XS_STRIDE = 74;   // sizeof block_iq2_xs (2 + 64 + 8)
static const int Q8K_STRIDE = 292;    // sizeof block_q8_K

// ---- block formats (ggml-common.h, QK_K = 256) -------------------------------
//   block_iq2_xs (74 bytes): d(fp16) @0 | qs[32](uint16) @2 | scales[8](uint8) @66
//   block_q8_K  (292 bytes): d(fp32) @0 | qs[256] @4 | bsums[16] @260 (unused)
typedef struct { uint16_t d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; } block_iq2_xs;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

// ---- ggml's GRID codebook (512) + SIGN plane + kmask (ggml-common.h) -----------
static const uint64_t iq2xs_grid[512] = {0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL, 0x0808080808192b19ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b1919ULL, 0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x080808081908192bULL, 0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL, 0x080808082b191908ULL, 0x080808082b192b19ULL, 0x080808082b2b0808ULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL, 0x080808190819082bULL, 0x0808081908191919ULL, 0x0808081908192b08ULL, 0x0808081908192b2bULL, 0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL, 0x080808191908082bULL, 0x0808081919081919ULL, 0x0808081919082b08ULL, 0x0808081919190819ULL, 0x0808081919191908ULL, 0x08080819192b0808ULL, 0x08080819192b2b08ULL, 0x080808192b080819ULL, 0x080808192b081908ULL, 0x080808192b190808ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b08081919ULL, 0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL, 0x0808082b19080819ULL, 0x0808082b19081908ULL, 0x0808082b19190808ULL, 0x0808082b19191919ULL, 0x0808082b2b080808ULL, 0x0808082b2b082b2bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x080819080808192bULL, 0x0808190808082b19ULL, 0x0808190808190808ULL, 0x080819080819082bULL, 0x0808190808191919ULL, 0x0808190808192b08ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL, 0x0808190819082b08ULL, 0x0808190819190819ULL, 0x0808190819191908ULL, 0x080819081919192bULL, 0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908081919ULL, 0x0808191908082b08ULL, 0x0808191908190819ULL, 0x0808191908191908ULL, 0x08081919082b0808ULL, 0x0808191919080819ULL, 0x0808191919081908ULL, 0x0808191919190808ULL, 0x08081919192b0819ULL, 0x080819192b080808ULL, 0x0808192b08080819ULL, 0x0808192b08081908ULL, 0x0808192b08190808ULL, 0x0808192b082b192bULL, 0x0808192b19080808ULL, 0x0808192b1908082bULL, 0x0808192b2b081908ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808082b2bULL, 0x08082b0808190819ULL, 0x08082b0808191908ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b0819192b08ULL, 0x08082b082b080808ULL, 0x08082b082b2b0808ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL, 0x08082b1908081908ULL, 0x08082b1908190808ULL, 0x08082b1919080808ULL, 0x08082b192b080819ULL, 0x08082b192b082b19ULL, 0x08082b2b08080808ULL, 0x08082b2b082b0808ULL, 0x08082b2b082b2b08ULL, 0x08082b2b2b19192bULL, 0x08082b2b2b2b0808ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x081908080808192bULL, 0x0819080808082b19ULL, 0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL, 0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL, 0x0819080819080808ULL, 0x081908081908082bULL, 0x0819080819081919ULL, 0x0819080819082b08ULL, 0x0819080819190819ULL, 0x0819080819191908ULL, 0x08190808192b0808ULL, 0x08190808192b2b2bULL, 0x081908082b080819ULL, 0x081908082b081908ULL, 0x081908082b190808ULL, 0x0819081908080808ULL, 0x081908190808082bULL, 0x0819081908081919ULL, 0x0819081908082b08ULL, 0x0819081908190819ULL, 0x0819081908191908ULL, 0x08190819082b0808ULL, 0x0819081919080819ULL, 0x0819081919081908ULL, 0x0819081919190808ULL, 0x081908192b080808ULL, 0x081908192b191908ULL, 0x081908192b19192bULL, 0x0819082b08080819ULL, 0x0819082b08081908ULL, 0x0819082b0808192bULL, 0x0819082b08190808ULL, 0x0819082b19080808ULL, 0x0819082b192b0808ULL, 0x0819190808080808ULL, 0x081919080808082bULL, 0x0819190808081919ULL, 0x0819190808082b08ULL, 0x0819190808190819ULL, 0x0819190808191908ULL, 0x08191908082b0808ULL, 0x0819190819080819ULL, 0x0819190819081908ULL, 0x0819190819082b19ULL, 0x0819190819190808ULL, 0x08191908192b1908ULL, 0x081919082b080808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL, 0x0819191908190808ULL, 0x0819191919080808ULL, 0x0819192b08080808ULL, 0x0819192b08191908ULL, 0x0819192b19082b19ULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b080819082bULL, 0x08192b0819080808ULL, 0x08192b0819191908ULL, 0x08192b082b08192bULL, 0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b19192b192bULL, 0x08192b2b19190819ULL, 0x08192b2b2b2b2b19ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL, 0x082b080808082b08ULL, 0x082b080808082b2bULL, 0x082b080808190819ULL, 0x082b080808191908ULL, 0x082b0808082b0808ULL, 0x082b080819080819ULL, 0x082b080819081908ULL, 0x082b080819190808ULL, 0x082b08082b080808ULL, 0x082b08082b2b0808ULL, 0x082b081908080819ULL, 0x082b081908081908ULL, 0x082b081908190808ULL, 0x082b081919080808ULL, 0x082b081919082b08ULL, 0x082b0819192b1919ULL, 0x082b082b08080808ULL, 0x082b082b082b082bULL, 0x082b082b2b080808ULL, 0x082b082b2b2b2b08ULL, 0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b1908082b2b19ULL, 0x082b190819080808ULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b19191919082bULL, 0x082b19192b192b19ULL, 0x082b192b08080819ULL, 0x082b192b08192b2bULL, 0x082b192b2b2b192bULL, 0x082b2b0808080808ULL, 0x082b2b0808082b08ULL, 0x082b2b0808082b2bULL, 0x082b2b08082b0808ULL, 0x082b2b0819191919ULL, 0x082b2b082b082b08ULL, 0x082b2b082b2b082bULL, 0x082b2b19192b2b08ULL, 0x082b2b192b190808ULL, 0x082b2b2b08082b08ULL, 0x082b2b2b082b0808ULL, 0x082b2b2b2b08082bULL, 0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL, 0x1908080808080819ULL, 0x1908080808081908ULL, 0x190808080808192bULL, 0x1908080808082b19ULL, 0x1908080808190808ULL, 0x190808080819082bULL, 0x1908080808191919ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x190808081908082bULL, 0x1908080819081919ULL, 0x1908080819082b08ULL, 0x1908080819082b2bULL, 0x1908080819190819ULL, 0x1908080819191908ULL, 0x19080808192b0808ULL, 0x19080808192b1919ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL, 0x1908081908080808ULL, 0x190808190808082bULL, 0x1908081908081919ULL, 0x1908081908082b08ULL, 0x1908081908190819ULL, 0x1908081908191908ULL, 0x19080819082b0808ULL, 0x1908081919080819ULL, 0x1908081919081908ULL, 0x1908081919190808ULL, 0x190808192b080808ULL, 0x190808192b081919ULL, 0x190808192b2b082bULL, 0x1908082b08080819ULL, 0x1908082b08081908ULL, 0x1908082b08190808ULL, 0x1908082b0819082bULL, 0x1908082b082b2b19ULL, 0x1908082b19080808ULL, 0x1908190808080808ULL, 0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL, 0x1908190808190819ULL, 0x1908190808191908ULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL, 0x1908190819080819ULL, 0x1908190819081908ULL, 0x1908190819190808ULL, 0x190819082b080808ULL, 0x190819082b191908ULL, 0x1908191908080819ULL, 0x1908191908081908ULL, 0x1908191908190808ULL, 0x19081919082b1908ULL, 0x1908191919080808ULL, 0x190819192b192b2bULL, 0x1908192b08080808ULL, 0x1908192b08082b2bULL, 0x1908192b19081908ULL, 0x1908192b19190808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b0819191908ULL, 0x19082b08192b082bULL, 0x19082b1908080808ULL, 0x19082b1908190819ULL, 0x19082b1919081908ULL, 0x19082b1919190808ULL, 0x19082b19192b2b19ULL, 0x19082b2b08081908ULL, 0x1919080808080808ULL, 0x191908080808082bULL, 0x1919080808081919ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808191908ULL, 0x19190808082b0808ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL, 0x1919080819081908ULL, 0x1919080819190808ULL, 0x191908082b080808ULL, 0x1919081908080819ULL, 0x1919081908081908ULL, 0x1919081908190808ULL, 0x1919081908191919ULL, 0x1919081919080808ULL, 0x191908191908082bULL, 0x1919082b08080808ULL, 0x1919082b19081908ULL, 0x1919082b2b2b2b2bULL, 0x1919190808080819ULL, 0x1919190808081908ULL, 0x1919190808190808ULL, 0x19191908082b0819ULL, 0x1919190819080808ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b2b0819ULL, 0x1919191908080808ULL, 0x1919191908082b08ULL, 0x191919192b080808ULL, 0x191919192b082b08ULL, 0x1919192b082b0819ULL, 0x1919192b192b2b08ULL, 0x1919192b2b2b0819ULL, 0x19192b0808080808ULL, 0x19192b0808191908ULL, 0x19192b0819080819ULL, 0x19192b0819190808ULL, 0x19192b082b192b19ULL, 0x19192b1908192b2bULL, 0x19192b1919080808ULL, 0x19192b191908082bULL, 0x19192b2b2b081919ULL, 0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b080819191908ULL, 0x192b0808192b082bULL, 0x192b08082b08192bULL, 0x192b08082b2b2b19ULL, 0x192b081908080808ULL, 0x192b082b082b1908ULL, 0x192b082b19082b2bULL, 0x192b082b2b19082bULL, 0x192b190808080808ULL, 0x192b19080819192bULL, 0x192b191908190808ULL, 0x192b191919080808ULL, 0x192b191919081919ULL, 0x192b19192b2b1908ULL, 0x192b2b0808080819ULL, 0x192b2b08192b2b2bULL, 0x192b2b19082b1919ULL, 0x192b2b2b0808192bULL, 0x192b2b2b19191908ULL, 0x192b2b2b192b082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL, 0x2b08080808190819ULL, 0x2b08080808191908ULL, 0x2b080808082b0808ULL, 0x2b080808082b2b2bULL, 0x2b08080819080819ULL, 0x2b08080819081908ULL, 0x2b08080819190808ULL, 0x2b0808082b080808ULL, 0x2b0808082b08082bULL, 0x2b0808082b2b2b08ULL, 0x2b0808082b2b2b2bULL, 0x2b08081908080819ULL, 0x2b08081908081908ULL, 0x2b0808190808192bULL, 0x2b08081908190808ULL, 0x2b08081919080808ULL, 0x2b08081919190819ULL, 0x2b08081919192b19ULL, 0x2b08082b08080808ULL, 0x2b08082b082b0808ULL, 0x2b08082b2b080808ULL, 0x2b08082b2b08082bULL, 0x2b08082b2b2b0808ULL, 0x2b08082b2b2b2b08ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b0819080819082bULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b0819082b082b19ULL, 0x2b08191908080808ULL, 0x2b08191919081908ULL, 0x2b0819192b2b1919ULL, 0x2b08192b08192b08ULL, 0x2b08192b192b2b2bULL, 0x2b082b0808080808ULL, 0x2b082b0808082b08ULL, 0x2b082b08082b1919ULL, 0x2b082b0819192b2bULL, 0x2b082b082b080808ULL, 0x2b082b082b08082bULL, 0x2b082b082b2b2b08ULL, 0x2b082b190808192bULL, 0x2b082b2b082b082bULL, 0x2b082b2b2b080808ULL, 0x2b082b2b2b082b08ULL, 0x2b082b2b2b19192bULL, 0x2b082b2b2b2b2b08ULL, 0x2b19080808080819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL, 0x2b19080819080808ULL, 0x2b1908081919192bULL, 0x2b1908082b081908ULL, 0x2b19081908080808ULL, 0x2b190819082b082bULL, 0x2b190819192b1908ULL, 0x2b19082b1919192bULL, 0x2b19082b2b082b19ULL, 0x2b19190808080808ULL, 0x2b19190808081919ULL, 0x2b19190819081908ULL, 0x2b19190819190808ULL, 0x2b19190819192b08ULL, 0x2b191919082b2b19ULL, 0x2b1919192b190808ULL, 0x2b1919192b19082bULL, 0x2b19192b19080819ULL, 0x2b192b0819190819ULL, 0x2b192b082b2b192bULL, 0x2b192b1919082b19ULL, 0x2b192b2b08191919ULL, 0x2b192b2b192b0808ULL, 0x2b2b080808080808ULL, 0x2b2b08080808082bULL, 0x2b2b080808082b08ULL, 0x2b2b080808082b2bULL, 0x2b2b0808082b0808ULL, 0x2b2b0808082b2b2bULL, 0x2b2b08082b2b0808ULL, 0x2b2b081919190819ULL, 0x2b2b081919192b19ULL, 0x2b2b08192b2b192bULL, 0x2b2b082b08080808ULL, 0x2b2b082b0808082bULL, 0x2b2b082b08082b08ULL, 0x2b2b082b082b2b2bULL, 0x2b2b082b2b080808ULL, 0x2b2b082b2b2b0808ULL, 0x2b2b190819080808ULL, 0x2b2b19082b191919ULL, 0x2b2b192b192b1919ULL, 0x2b2b192b2b192b08ULL, 0x2b2b2b0808082b2bULL, 0x2b2b2b08082b0808ULL, 0x2b2b2b08082b082bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b0808ULL, 0x2b2b2b082b2b2b08ULL, 0x2b2b2b1908081908ULL, 0x2b2b2b192b081908ULL, 0x2b2b2b192b08192bULL, 0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b082b2b2bULL, 0x2b2b2b2b2b190819ULL, 0x2b2b2b2b2b2b2b2bULL};
static const uint8_t ksigns_iq2xs[128] = {0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255};
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

// WRONG grid for negative control 1: each entry's bytes incremented by 1.
static uint64_t iq2xs_grid_wrong[512];
// WRONG signs for negative control 2: ksigns with the sign bit (bit 7) cleared.
static uint8_t ksigns_wrong[128];

// ---- board fp16 -> fp32 (mirrors riscv_compute_fp16_to_fp32, lossless) -------
static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

// ---- The kernel our compiler emitted (declared; defined in the linked .cpp) --
extern "C" void
tcrv_emitc_ggml_vec_dot_iq2_xs_q8_K_kernel_ggml_vec_dot_iq2_xs_q8_K(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

static void our_kernel(int n, float *s, const void *vx, const void *vy) {
  tcrv_emitc_ggml_vec_dot_iq2_xs_q8_K_kernel_ggml_vec_dot_iq2_xs_q8_K(
      (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);
}

// ---- ggml's VERBATIM _generic (quants.c:897-945) -----------------------------
// Parameterised by the grid table, the ksigns table, a scale-bias flag, AND a
// split-collapse flag so the negative controls can swap them. The structure is ggml's
// EXACT inner loops (the two-half ls1/ls2 split).
static void ggml_generic_param(int n, float *s, const void *vx, const void *vy,
                               const uint64_t *grid_tab, const uint8_t *ksign_tab,
                               int wrong_scale, int collapse_split) {
  const block_iq2_xs *x = (const block_iq2_xs *)vx;
  const block_q8_K *y = (const block_q8_K *)vy;
  const int nb = n / QK_K;
  float sumf = 0.f;
  for (int i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d) * y[i].d;
    const uint16_t *q2 = x[i].qs;
    const uint8_t *sc = x[i].scales;
    const int8_t *q8 = y[i].qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
      // wrong_scale: drop the "+1". collapse_split: force ls2 = ls1.
      const uint16_t ls1 = wrong_scale ? (uint16_t)(2*(sc[ib32] & 0xf))
                                       : (uint16_t)(2*(sc[ib32] & 0xf) + 1);
      uint16_t ls2 = wrong_scale ? (uint16_t)(2*(sc[ib32] >> 4))
                                  : (uint16_t)(2*(sc[ib32] >> 4) + 1);
      if (collapse_split) ls2 = ls1;
      int32_t sumi = 0;
      for (int l = 0; l < 2; ++l) {
        const uint8_t *grid = (const uint8_t *)(grid_tab + (q2[l] & 511));
        const uint8_t signs = ksign_tab[q2[l] >> 9];
        for (int j = 0; j < 8; ++j)
          sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        q8 += 8;
      }
      bsum += sumi * ls1;
      sumi = 0;
      for (int l = 2; l < 4; ++l) {
        const uint8_t *grid = (const uint8_t *)(grid_tab + (q2[l] & 511));
        const uint8_t signs = ksign_tab[q2[l] >> 9];
        for (int j = 0; j < 8; ++j)
          sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        q8 += 8;
      }
      bsum += sumi * ls2;
      q2 += 4;
    }
    sumf += d * bsum;
  }
  *s = 0.125f * sumf;
}

static void ggml_ref(int n, float *s, const void *vx, const void *vy) {
  ggml_generic_param(n, s, vx, vy, iq2xs_grid, ksigns_iq2xs, 0, 0);
}

// ---- RNG -----------------------------------------------------------------------
static uint64_t rng_state = 0x1234567890abcdefULL;
static uint32_t xrng() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (uint32_t)(rng_state >> 32);
}

// Fill a random iq2_xs weight block + q8_K activation block.
static void fill_random_blocks(uint8_t *xbuf, uint8_t *ybuf, int nb, int q8_mode) {
  for (int i = 0; i < nb; ++i) {
    block_iq2_xs *x = (block_iq2_xs *)(xbuf + i * IQ2XS_STRIDE);
    block_q8_K *y = (block_q8_K *)(ybuf + i * Q8K_STRIDE);
    float dx = 0.005f + (xrng() & 0xff) * 0.0007f;
    _Float16 hx = (_Float16)dx;
    std::memcpy(&x->d, &hx, 2);
    // qs: random uint16 covering the full 9-bit index range and 7-bit sign range.
    for (int k = 0; k < QK_K / 8; ++k) x->qs[k] = (uint16_t)(xrng() & 0xffff);
    // scales: random bytes (distinct low/high nibbles exercise ls1 != ls2).
    for (int k = 0; k < QK_K / 32; ++k) x->scales[k] = (uint8_t)(xrng() & 0xff);
    y->d = 0.01f + (xrng() & 0xff) * 0.0011f;
    for (int k = 0; k < QK_K; ++k) {
      int v;
      if (q8_mode == 1) v = 127;
      else if (q8_mode == 2) v = (k & 1) ? 127 : -127;
      else v = (int)(xrng() % 255) - 127;  // [-127,127]
      y->qs[k] = (int8_t)v;
    }
    for (int k = 0; k < QK_K / 16; ++k) y->bsums[k] = 0;
  }
}

static int g_fail = 0, g_total = 0, g_gen_delta = 0;

static int check_case(const char *name, int n, const uint8_t *xbuf,
                      const uint8_t *ybuf) {
  float s_ours = 0.f, s_ref = 0.f;
  our_kernel(n, &s_ours, xbuf, ybuf);
  ggml_ref(n, &s_ref, xbuf, ybuf);
  g_total++;
  if (std::memcmp(&s_ours, &s_ref, sizeof(float)) != 0) {
    g_fail++;
    g_gen_delta++;
    if (g_fail <= 8)
      std::printf("  FAIL %-28s n=%-6d ours=%.9g ref=%.9g (bits %08x vs %08x)\n",
                  name, n, s_ours, s_ref,
                  *(uint32_t *)&s_ours, *(uint32_t *)&s_ref);
    return 1;
  }
  return 0;
}

int main() {
  // Build the wrong-grid + wrong-signs tables (negative controls).
  for (int i = 0; i < 512; ++i) {
    uint64_t g = iq2xs_grid[i];
    uint64_t w = 0;
    for (int b = 0; b < 8; ++b) {
      uint8_t by = (uint8_t)((g >> (8 * b)) & 0xff);
      w |= (uint64_t)((uint8_t)(by + 1)) << (8 * b);
    }
    iq2xs_grid_wrong[i] = w;
  }
  for (int i = 0; i < 128; ++i) ksigns_wrong[i] = ksigns_iq2xs[i] & 0x7f;

  const int MAXB = 64;
  uint8_t *xbuf = (uint8_t *)std::malloc(MAXB * IQ2XS_STRIDE);
  uint8_t *ybuf = (uint8_t *)std::malloc(MAXB * Q8K_STRIDE);

  // ---- random multiples of 256 -------------------------------------------------
  int ns[] = {256, 512, 768, 1024, 2048, 4096, 8192, 16384};
  for (int rep = 0; rep < 300; ++rep) {
    for (int ni = 0; ni < 8; ++ni) {
      int n = ns[ni];
      int nb = n / QK_K;
      if (nb > MAXB) continue;
      fill_random_blocks(xbuf, ybuf, nb, 0);
      check_case("random", n, xbuf, ybuf);
    }
  }

  // ---- edge: q8 = +127 / q8 = +/-127 ------------------------------------------
  for (int rep = 0; rep < 50; ++rep) {
    fill_random_blocks(xbuf, ybuf, 8, 1);
    check_case("q8=+127", 8 * QK_K, xbuf, ybuf);
    fill_random_blocks(xbuf, ybuf, 8, 2);
    check_case("q8=+/-127", 8 * QK_K, xbuf, ybuf);
  }

  // ---- edge: marching grid indices into the FULL [0,511] range (the key iq2_xs
  // delta vs iq2_xxs -- a low-8-bits bug passes if indices stop at 255). Set every
  // q2 word's 9-bit index to march 0..511 and the 7-bit sign selector to march
  // 0..127, the scales to extremes / distinct ls1!=ls2 nibbles.
  for (int extreme = 0; extreme < 3; ++extreme) {
    int nb = 8;
    fill_random_blocks(xbuf, ybuf, nb, 0);
    int idxc = 0;
    for (int i = 0; i < nb; ++i) {
      block_iq2_xs *x = (block_iq2_xs *)(xbuf + i * IQ2XS_STRIDE);
      for (int k = 0; k < QK_K / 8; ++k) {
        // stride 2 so the 256 words/block SPAN the FULL [0,511] index range (a
        // low-8-bit-mask bug would diverge on every index >= 256).
        uint16_t idx = (uint16_t)((idxc * 2) & 511); // 9-bit index spanning 0..511
        uint16_t sgn = (uint16_t)((idxc) & 0x7f);    // 7-bit sign selector
        x->qs[k] = (uint16_t)(idx | (sgn << 9));
        idxc++;
      }
      for (int ib = 0; ib < QK_K / 32; ++ib) {
        if (extreme == 0) x->scales[ib] = 0x00;          // ls1=ls2=1
        else if (extreme == 1) x->scales[ib] = 0xff;     // ls1=ls2=31
        else x->scales[ib] = (uint8_t)((ib & 0xf) | (((15 - (ib & 0xf)) & 0xf) << 4)); // ls1!=ls2
      }
    }
    const char *nm = extreme == 0 ? "grid511-march+scale0"
                   : extreme == 1 ? "grid511-march+scale31"
                                  : "grid511-march+ls1!=ls2";
    check_case(nm, nb * QK_K, xbuf, ybuf);
  }

  std::printf("iq2_xs byte-exact: %d/%d passed, %d failed (generic delta %d/%d)\n",
              g_total - g_fail, g_total, g_fail, g_gen_delta, g_total);

  // ---- NEGATIVE CONTROLS: grid/signs/scale/split must be load-bearing ----------
  fill_random_blocks(xbuf, ybuf, 8, 0);
  // make scales distinct so the split-collapse control actually bites.
  for (int i = 0; i < 8; ++i) {
    block_iq2_xs *x = (block_iq2_xs *)(xbuf + i * IQ2XS_STRIDE);
    for (int ib = 0; ib < QK_K / 32; ++ib)
      x->scales[ib] = (uint8_t)((ib & 0xf) | (((15 - (ib & 0xf)) & 0xf) << 4));
  }
  int nbig = 8 * QK_K;
  float s_ours = 0.f;
  our_kernel(nbig, &s_ours, xbuf, ybuf);

  int controls_ok = 0;
  {
    float s_wrong = 0.f;
    ggml_generic_param(nbig, &s_wrong, xbuf, ybuf, iq2xs_grid_wrong, ksigns_iq2xs, 0, 0);
    int diverges = std::memcmp(&s_ours, &s_wrong, sizeof(float)) != 0;
    std::printf("NEG-CTRL wrong-grid:  ours=%.9g wrong=%.9g -> %s\n", s_ours,
                s_wrong, diverges ? "DIVERGES (PASS)" : "MATCHES (FAIL!)");
    controls_ok += diverges;
  }
  {
    float s_wrong = 0.f;
    ggml_generic_param(nbig, &s_wrong, xbuf, ybuf, iq2xs_grid, ksigns_wrong, 0, 0);
    int diverges = std::memcmp(&s_ours, &s_wrong, sizeof(float)) != 0;
    std::printf("NEG-CTRL wrong-signs: ours=%.9g wrong=%.9g -> %s\n", s_ours,
                s_wrong, diverges ? "DIVERGES (PASS)" : "MATCHES (FAIL!)");
    controls_ok += diverges;
  }
  {
    float s_wrong = 0.f;
    ggml_generic_param(nbig, &s_wrong, xbuf, ybuf, iq2xs_grid, ksigns_iq2xs, 1, 0);
    int diverges = std::memcmp(&s_ours, &s_wrong, sizeof(float)) != 0;
    std::printf("NEG-CTRL wrong-scale: ours=%.9g wrong=%.9g -> %s\n", s_ours,
                s_wrong, diverges ? "DIVERGES (PASS)" : "MATCHES (FAIL!)");
    controls_ok += diverges;
  }
  {
    float s_wrong = 0.f;
    ggml_generic_param(nbig, &s_wrong, xbuf, ybuf, iq2xs_grid, ksigns_iq2xs, 0, 1);
    int diverges = std::memcmp(&s_ours, &s_wrong, sizeof(float)) != 0;
    std::printf("NEG-CTRL collapse-split (ls2=ls1): ours=%.9g wrong=%.9g -> %s\n",
                s_ours, s_wrong, diverges ? "DIVERGES (PASS)" : "MATCHES (FAIL!)");
    controls_ok += diverges;
  }

  std::free(xbuf);
  std::free(ybuf);

  int ok = (g_fail == 0) && (controls_ok == 4);
  std::printf("RESULT: %s\n", ok ? "ALL PASS" : "FAILURES PRESENT");
  return ok ? 0 : 1;
}
