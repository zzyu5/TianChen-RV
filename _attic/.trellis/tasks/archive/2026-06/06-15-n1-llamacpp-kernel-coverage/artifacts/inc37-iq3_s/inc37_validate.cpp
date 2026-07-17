// INC-37 iq3_s x q8_K byte-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.iq3_s_q8_k_block_dot op (the
// COMPLETE ggml ggml_vec_dot_iq3_s_q8_K GRID-codebook super-block kernel) computes the
// SAME fp32 result *s as ggml's OWN _generic iq3_s reference, BIT-FOR-BIT (memcmp of
// the float bits), over random block_iq3_s x block_q8_K arrays at n multiples of 256 +
// named edge cases (grid index range to 511, qh injection, signs, scales, q8 +/-127).
// FOUR negative controls (a WRONG grid + WRONG signs + WRONG scale + WRONG qh) must
// FAIL -> proves the grid, the explicit signs, the explicit scales, AND the qh 9th-bit
// index plane are the live, load-bearing mechanisms.
//
// iq3_s is a RE-COMPOSITION of three already-built mechanisms:
//   (a) the iq3 GRID-of-4 codebook (iq3_xxs): iq3s_grid[512], each entry = 4 int8.
//   (b) the qh 9th-bit plane (iq2_s): the index gets ONE high bit (mask 256) injected
//       from a per-sub-block qh byte; the two grid indices of a group take DIFFERENT
//       shifts (8-2l for grid1, 7-2l for grid2).
//   (c) EXPLICIT signs (iq2_s): the sign byte is read DIRECTLY from a dedicated 32-byte
//       signs[] region (NOT ksigns). The explicit 4-bit scales[] two-nibble split is
//       also reused from iq2_s. NO trailing factor (*s = sumf).
//
// REFERENCE: ggml's VERBATIM _generic (quants.c:1043-1097): d = fp16(x.d)*y.d, per
// ib32+=2 two-half loop, ls1/ls2 = 2*(scales[ib32/2]{&0xf,>>4})+1, grid1/grid2 =
// (uint8_t*)(iq3s_grid + (qs[2l+k] | ((qh[ib32+half] << (8/7-2l)) & 256))), signs =
// x.signs (explicit), sumi += grid1[j]*q8[j]*(signs[l]&kmask[j]?-1:1) +
// grid2[j]*q8[j+4]*(signs[l]&kmask[j+4]?-1:1), bsum += sumi*ls, sumf += d*bsum, *s =
// sumf. FP16->FP32 mirrors the board's scalar _Float16 widening (lossless). Whole TU
// built with the same -ffp-contract.
//
// The kernel under test is the UNMODIFIED, compiler-emitted iq3_s kernel
// (tcrv_emitted_kernel.cpp) -- every line tagged source_op=tcrv_rvv.iq3_s_q8_k_block_dot.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>

#define QK_K 256
#define IQ3S_N_SCALE (QK_K/64)               // 4
static const int IQ3S_STRIDE = 110;          // sizeof block_iq3_s = 2+64+8+32+4
static const int Q8K_STRIDE = 292;           // sizeof block_q8_K

// ---- block formats (ggml-common.h, QK_K = 256) -------------------------------
//   block_iq3_s (110 bytes): d(fp16) @0 | qs[64] @2 | qh[8] @66 | signs[32] @74 |
//                            scales[4] @106
//   block_q8_K (292 bytes):  d(fp32) @0 | qs[256] @4 | bsums[16] @260 (unused)
typedef struct {
  uint16_t d;
  uint8_t qs[QK_K/4];
  uint8_t qh[QK_K/32];
  uint8_t signs[QK_K/8];
  uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

// ---- ggml's GRID-of-4 codebook (iq3s_grid[512]) + kmask (ggml-common.h) -------
static const uint32_t iq3s_grid[512] = { 0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305, 0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905, 0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09, 0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b, 0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b, 0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d, 0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03, 0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505, 0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03, 0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901, 0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d, 0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303, 0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501, 0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105, 0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505, 0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101, 0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707, 0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b, 0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01, 0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f, 0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305, 0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103, 0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509, 0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503, 0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b, 0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f, 0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f, 0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f, 0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109, 0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f, 0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509, 0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501, 0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303, 0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f, 0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907, 0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703, 0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03, 0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01, 0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01, 0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903, 0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505, 0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b, 0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107, 0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509, 0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303, 0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103, 0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05, 0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b, 0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f, 0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701, 0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909, 0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305, 0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d, 0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b, 0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d, 0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307, 0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09, 0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309, 0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709, 0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f, 0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303, 0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503, 0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b, 0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101, };
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

// WRONG grid for negative control 1: each entry's bytes incremented by 1.
static uint32_t iq3s_grid_wrong[512];

// ---- board fp16 -> fp32 (mirrors riscv_compute_fp16_to_fp32, lossless) -------
static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

// ---- The kernel our compiler emitted (declared; defined in the linked .cpp) --
extern "C" void
tcrv_emitc_ggml_vec_dot_iq3_s_q8_K_kernel_ggml_vec_dot_iq3_s_q8_K(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

// ---- ggml _generic reference (VERBATIM quants.c:1043-1097, parameterized so the
// negative controls can swap ONE mechanism: the grid table, the explicit signs, the
// scale +1, or the qh injection) ----------------------------------------------------
struct RefKnobs {
  const uint32_t *grid;     // grid table (real or wrong)
  bool clearSignBit;        // wrong-signs: clear bit 7 of every explicit sign byte
  bool dropScalePlusOne;    // wrong-scale: drop the +1 in 2*nibble+1
  bool dropQh;              // wrong-qh: drop the qh 9th-bit injection (idx = qs only)
};

static void ref_iq3_s(int n, float *s, const void *vx, const void *vy,
                      const RefKnobs &k) {
  const block_iq3_s *x = (const block_iq3_s *)vx;
  const block_q8_K *y = (const block_q8_K *)vy;
  const int nb = n / QK_K;
  float sumf = 0.f;
  for (int i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d) * y[i].d;
    const uint8_t *qs = x[i].qs;
    const uint8_t *qh = x[i].qh;
    const uint8_t *signs = x[i].signs;
    const int8_t *q8 = y[i].qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
      int sc0 = x[i].scales[ib32/2] & 0xf;
      int sc1 = x[i].scales[ib32/2] >> 4;
      const uint32_t ls1 = k.dropScalePlusOne ? (uint32_t)(2*sc0) : (uint32_t)(2*sc0 + 1);
      const uint32_t ls2 = k.dropScalePlusOne ? (uint32_t)(2*sc1) : (uint32_t)(2*sc1 + 1);
      int32_t sumi = 0;
      for (int l = 0; l < 4; ++l) {
        int hb1 = k.dropQh ? 0 : ((qh[ib32+0] << (8-2*l)) & 256);
        int hb2 = k.dropQh ? 0 : ((qh[ib32+0] << (7-2*l)) & 256);
        const uint8_t *grid1 = (const uint8_t *)(k.grid + (qs[2*l+0] | hb1));
        const uint8_t *grid2 = (const uint8_t *)(k.grid + (qs[2*l+1] | hb2));
        uint8_t sg = signs[l];
        if (k.clearSignBit) sg &= 0x7f;
        for (int j = 0; j < 4; ++j) {
          sumi += grid1[j] * q8[j+0] * (sg & kmask_iq2xs[j+0] ? -1 : 1);
          sumi += grid2[j] * q8[j+4] * (sg & kmask_iq2xs[j+4] ? -1 : 1);
        }
        q8 += 8;
      }
      qs += 8;
      signs += 4;
      bsum += sumi * ls1;
      sumi = 0;
      for (int l = 0; l < 4; ++l) {
        int hb1 = k.dropQh ? 0 : ((qh[ib32+1] << (8-2*l)) & 256);
        int hb2 = k.dropQh ? 0 : ((qh[ib32+1] << (7-2*l)) & 256);
        const uint8_t *grid1 = (const uint8_t *)(k.grid + (qs[2*l+0] | hb1));
        const uint8_t *grid2 = (const uint8_t *)(k.grid + (qs[2*l+1] | hb2));
        uint8_t sg = signs[l];
        if (k.clearSignBit) sg &= 0x7f;
        for (int j = 0; j < 4; ++j) {
          sumi += grid1[j] * q8[j+0] * (sg & kmask_iq2xs[j+0] ? -1 : 1);
          sumi += grid2[j] * q8[j+4] * (sg & kmask_iq2xs[j+4] ? -1 : 1);
        }
        q8 += 8;
      }
      qs += 8;
      signs += 4;
      bsum += sumi * ls2;
    }
    sumf += d * bsum;
  }
  *s = sumf;
}

// ---- deterministic PRNG ------------------------------------------------------
static uint64_t rng_state = 0x123456789abcdef0ULL;
static uint32_t xr() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (uint32_t)(rng_state >> 32);
}
static uint8_t rb() { return (uint8_t)(xr() & 0xff); }

// random fp16 in a benign range (avoid NaN/Inf exponents) ----------------------
static uint16_t rand_fp16() {
  uint16_t exp = (uint16_t)(9 + (xr() % 8));  // ~[2^-6, 2^1]
  uint16_t mant = (uint16_t)(xr() & 0x3ff);
  uint16_t sign = (uint16_t)(xr() & 1);
  return (uint16_t)((sign << 15) | (exp << 10) | mant);
}

static void fill_block(block_iq3_s *xb, block_q8_K *yb, int idxMode) {
  xb->d = rand_fp16();
  for (int j = 0; j < QK_K/4; ++j) xb->qs[j] = rb();
  for (int j = 0; j < QK_K/32; ++j) xb->qh[j] = rb();
  for (int j = 0; j < QK_K/8; ++j) xb->signs[j] = rb();
  for (int j = 0; j < IQ3S_N_SCALE; ++j) xb->scales[j] = rb();
  yb->d = (fp16_to_fp32(rand_fp16()));
  for (int j = 0; j < QK_K; ++j) {
    if (idxMode == 1) yb->qs[j] = 127;          // q8 = +127
    else if (idxMode == 2) yb->qs[j] = (j & 1) ? 127 : -127;  // q8 = +/-127
    else yb->qs[j] = (int8_t)rb();
  }
}

// edge: drive the 9-bit index to 511 (qs byte 0xff + qh bit set for that group), the
// scale extremes (nibble 0 -> ls=1; nibble 15 -> ls=31), distinct ls1!=ls2, and mark
// whether index 511 is hit. Returns max index seen across the super-block.
static int fill_edge_block(block_iq3_s *xb, block_q8_K *yb, int *hit511) {
  xb->d = rand_fp16();
  // qs all 0xff so the low byte is 255; qh all 0xff so every group's high bit is set.
  for (int j = 0; j < QK_K/4; ++j) xb->qs[j] = 0xff;
  for (int j = 0; j < QK_K/32; ++j) xb->qh[j] = 0xff;
  for (int j = 0; j < QK_K/8; ++j) xb->signs[j] = rb();
  // scales: byte 0 -> low nibble 0 (ls=1), high nibble 15 (ls=31). Distinct ls1!=ls2.
  xb->scales[0] = (uint8_t)((15 << 4) | 0);
  for (int j = 1; j < IQ3S_N_SCALE; ++j) xb->scales[j] = rb();
  yb->d = fp16_to_fp32(rand_fp16());
  for (int j = 0; j < QK_K; ++j) yb->qs[j] = (int8_t)rb();
  // compute max index + 511 hit over the super-block (mirror the index assembly).
  int maxIdx = 0;
  *hit511 = 0;
  const uint8_t *qs = xb->qs;
  const uint8_t *qh = xb->qh;
  for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
    for (int half = 0; half < 2; ++half) {
      for (int l = 0; l < 4; ++l) {
        int i1 = qs[2*l+0] | ((qh[ib32+half] << (8-2*l)) & 256);
        int i2 = qs[2*l+1] | ((qh[ib32+half] << (7-2*l)) & 256);
        if (i1 > maxIdx) maxIdx = i1;
        if (i2 > maxIdx) maxIdx = i2;
        if (i1 == 511 || i2 == 511) *hit511 = 1;
      }
      qs += 8;
    }
  }
  return maxIdx;
}

static int bits_equal(float a, float b) {
  uint32_t ua, ub;
  std::memcpy(&ua, &a, 4);
  std::memcpy(&ub, &b, 4);
  return ua == ub;
}

int main() {
  // build the wrong grid (each byte +1) for negative control 1.
  for (int i = 0; i < 512; ++i) {
    uint32_t v = iq3s_grid[i];
    uint32_t w = 0;
    for (int b = 0; b < 4; ++b) {
      uint8_t byte = (uint8_t)((v >> (8*b)) & 0xff);
      byte = (uint8_t)(byte + 1);
      w |= ((uint32_t)byte) << (8*b);
    }
    iq3s_grid_wrong[i] = w;
  }

  const int reps = 300;
  const int ns[] = {256, 512, 768, 1024, 2048, 4096, 8192, 16384};
  const int nns = sizeof(ns)/sizeof(ns[0]);

  int total = 0, fails = 0, genericDelta = 0;
  RefKnobs real = {iq3s_grid, false, false, false};

  for (int r = 0; r < reps; ++r) {
    for (int ni = 0; ni < nns; ++ni) {
      int n = ns[ni];
      int nb = n / QK_K;
      block_iq3_s *X = (block_iq3_s *)malloc((size_t)nb * sizeof(block_iq3_s));
      block_q8_K *Y = (block_q8_K *)malloc((size_t)nb * sizeof(block_q8_K));
      int idxMode = (r % 7 == 0) ? 1 : (r % 11 == 0) ? 2 : 0;
      for (int b = 0; b < nb; ++b) fill_block(&X[b], &Y[b], idxMode);

      float s_ours = 0.f, s_ref = 0.f;
      tcrv_emitc_ggml_vec_dot_iq3_s_q8_K_kernel_ggml_vec_dot_iq3_s_q8_K(
          (size_t)n, &s_ours, (const uint8_t *)X, (const uint8_t *)Y);
      ref_iq3_s(n, &s_ref, X, Y, real);

      total++;
      if (!bits_equal(s_ours, s_ref)) {
        fails++;
        genericDelta++;
        if (fails <= 8)
          printf("  FAIL n=%d rep=%d ours=%.6f ref=%.6f\n", n, r, s_ours, s_ref);
      }
      free(X);
      free(Y);
    }
  }

  // edge cases: the 9-bit index marched to 511 via the qh plane + scale extremes.
  int maxIdxSeen = 0, hit511Any = 0;
  for (int r = 0; r < 64; ++r) {
    int nb = 4;
    int n = nb * QK_K;
    block_iq3_s *X = (block_iq3_s *)malloc((size_t)nb * sizeof(block_iq3_s));
    block_q8_K *Y = (block_q8_K *)malloc((size_t)nb * sizeof(block_q8_K));
    for (int b = 0; b < nb; ++b) {
      int hit = 0;
      int mx = fill_edge_block(&X[b], &Y[b], &hit);
      if (mx > maxIdxSeen) maxIdxSeen = mx;
      if (hit) hit511Any = 1;
    }
    float s_ours = 0.f, s_ref = 0.f;
    tcrv_emitc_ggml_vec_dot_iq3_s_q8_K_kernel_ggml_vec_dot_iq3_s_q8_K(
        (size_t)n, &s_ours, (const uint8_t *)X, (const uint8_t *)Y);
    ref_iq3_s(n, &s_ref, X, Y, real);
    total++;
    if (!bits_equal(s_ours, s_ref)) {
      fails++;
      if (fails <= 8)
        printf("  EDGE FAIL rep=%d ours=%.6f ref=%.6f\n", r, s_ours, s_ref);
    }
    free(X);
    free(Y);
  }

  printf("iq3_s byte-exact: %d/%d passed, %d failed (generic delta %d/%d)\n",
         total - fails, total, fails, genericDelta, total);
  printf("grid-index coverage (edge): max idx = %d, hit 511 = %s\n", maxIdxSeen,
         hit511Any ? "YES" : "NO");
  if (maxIdxSeen < 511 || !hit511Any) {
    printf("  COVERAGE ASSERTION FAILED: the 9-bit qh index plane was not exercised\n");
    fails++;
  }

  // ---- negative controls: each WRONG-mechanism reference must DIVERGE from ours -
  {
    int nb = 8;
    int n = nb * QK_K;
    block_iq3_s *X = (block_iq3_s *)malloc((size_t)nb * sizeof(block_iq3_s));
    block_q8_K *Y = (block_q8_K *)malloc((size_t)nb * sizeof(block_q8_K));
    for (int b = 0; b < nb; ++b) {
      // ensure qh has the high bit set in some groups + nonzero signs so the
      // wrong-qh / wrong-signs controls have something to flip.
      fill_block(&X[b], &Y[b], 0);
      for (int j = 0; j < QK_K/32; ++j) X[b].qh[j] |= 0x55;
      for (int j = 0; j < QK_K/8; ++j) X[b].signs[j] |= 0x80;
    }
    float s_ours = 0.f;
    tcrv_emitc_ggml_vec_dot_iq3_s_q8_K_kernel_ggml_vec_dot_iq3_s_q8_K(
        (size_t)n, &s_ours, (const uint8_t *)X, (const uint8_t *)Y);

    struct { const char *name; RefKnobs k; } controls[] = {
      {"wrong-grid  (every grid byte +1)", {iq3s_grid_wrong, false, false, false}},
      {"wrong-signs (clear explicit sign bit)", {iq3s_grid, true, false, false}},
      {"wrong-scale (drop the +1)", {iq3s_grid, false, true, false}},
      {"wrong-qh    (drop the 9th-bit plane)", {iq3s_grid, false, false, true}},
    };
    int ctrlOk = 0;
    for (auto &c : controls) {
      float s_wrong = 0.f;
      ref_iq3_s(n, &s_wrong, X, Y, c.k);
      bool diverges = !bits_equal(s_ours, s_wrong);
      printf("NEG-CTRL %-40s ours=%.6f wrong=%.6f -> %s\n", c.name, s_ours,
             s_wrong, diverges ? "DIVERGES" : "MATCHES (BUG: mechanism not live!)");
      if (diverges) ctrlOk++;
    }
    if (ctrlOk != 4) {
      printf("  NEGATIVE CONTROL FAILURE: %d/4 controls diverged (want 4)\n", ctrlOk);
      fails++;
    }
    free(X);
    free(Y);
  }

  if (fails == 0) {
    printf("RESULT: PASS (byte-exact vs ggml iq3_s _generic + 4 negative controls live)\n");
    return 0;
  }
  printf("RESULT: FAIL (%d failures)\n", fails);
  return 1;
}
