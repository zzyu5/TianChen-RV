// G7 L1 P1 host ZERO-MODEL oracle for the NEW independent q4_K@rvv GEVM Emission
// Plan (the COLUMN-GROUP-TILED, block-STREAMING, register-resident-accumulator
// decode structure — [K-10] structural-level, [PAT-2]-P9 regime-plan). This oracle
// is the byte-exact-BY-CONSTRUCTION hard gate for the plan's ARITHMETIC: it proves
// the new GEVM iteration ENVELOPE produces output BIT-IDENTICAL to the existing
// per-column (GEMM-plan-M=1) envelope, and to an INDEPENDENT re-derivation of the
// same q4_K super-block fold from the raw packed bytes.
//
// WHY the new plan is byte-exact by construction (the load-bearing invariant this
// oracle mechanizes): a GEVM output column is an INDEPENDENT reduction. The new
// plan differs from the existing col-outer/block-inner GEVM ONLY in the iteration
// ENVELOPE — it TILES the column-group loop, moves the contraction-block loop to
// the MIDDLE (so ONE q8_K activation block-load is SHARED across all columns of the
// tile), and keeps a REGISTER-RESIDENT f32 accumulator BANK live across the block
// stream. For any FIXED output column, the per-block reduction order, the 8
// sub-block / super-half split, the split-32 integer dot, the bsums-min correction,
// and the dual d/dmin fp16 float fold are UNTOUCHED. Reordering + tiling INDEPENDENT
// column reductions can never change a single column's value => bit-exact.
//
// SCOPE (honest): this oracle mechanizes the ARITHMETIC + ENVELOPE invariance at the
// LOGICAL q4_K super-block level (native block_q4_K decode via canonical
// get_scale_min_k4 + raw nibble — the SAME logical scale/min/quant values the RVV
// repack lane-wise vand/vsrl bit-dance yields). The repacked block_q4_Kx16 BYTE
// layout + the lane-wise emit is the lit golden's concern
// (rvv-to-emitc-repack-gemv-q4-K-colgroup-tiled-q8-K.mlir); this oracle's concern is
// that the fold order is q4_K-faithful AND the new envelope is byte-neutral.
//
// The RVV GEVM per-column fold this models (lib/.../RVVToEmitCBlockQuantLinear.cpp
// emitRepackKQuantGemvBodyQ4K :5950-6493, VERIFIED against that source):
//   per block l:  d0 = d_x * d_y ; dmins_d = dmin_x * d_y   (d_y = q8_K float delta)
//     sumi(i32)=0 ; bsums(i32)=0
//     for super-half j in {0,1}:
//       MIN: for sb in {0..3}: g=j*4+sb; bsums += (bsum[2g]+bsum[2g+1]) * min_6bit[g]
//       MAIN: for pair in {0,1}: for kchunk in {0,1}:  (32-elem sub-block split 2x16)
//         sLo=sHi=0; for ii in 0..15: i=kchunk*16+ii
//           sLo += A[jLoBase+i]*nibbleLo ; sHi += A[jHiBase+i]*nibbleHi
//         sumi += sc_6bit[gLo]*sLo + sc_6bit[gHi]*sHi
//     sumf += (float)sumi * d0 ;  sumf -= dmins_d * (float)bsums     (vfmacc/vfnmsac)
//
// Build + run:
//   cc -O2 -std=c11 q4-K-q8-K-repack-gevm-colgroup-tiled-oracle.c -o /tmp/g7gevm
//   /tmp/g7gevm
// Exit 0 + "ORACLE PASS" iff (1) tiled==per-column bit-exact, (2) zero-model==tiled
// bit-exact, over the whole grid, AND the anti-hollow controls diverge.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// (A) Shared decode primitives — canonical q4_K, re-used by every path (the ONE
// correct 6-bit unpack + raw nibble; identical to ggml_ref / the IME q4_K oracle).
// ---------------------------------------------------------------------------

static float fp16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0u) {
    if (mant == 0u) {
      bits = sign;
    } else {
      exp = 127u - 15u + 1u;
      while ((mant & 0x400u) == 0u) { mant <<= 1; exp--; }
      mant &= 0x3FFu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}
static uint16_t load_fp16(const uint8_t *p) {
  return (uint16_t)((unsigned)p[0] | ((unsigned)p[1] << 8));
}

// native block_q4_K = { fp16 d @0, fp16 dmin @2, uint8 scales[12] @4, uint8 qs[128] @16 }
#define Q4K_BYTES 144

// canonical get_scale_min_k4 (one correct 6-bit unpack of the 12 packed scale bytes)
static void get_scale_min_k4(int j, const uint8_t *q, int *sc, int *m) {
  if (j < 4) {
    *sc = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}
// canonical raw nibble for element p in [0,256): sub-block b=p/32, pos pl=p%32.
static int q4k_nibble(const uint8_t *blk, int p) {
  int b = p / 32, pl = p % 32;
  const uint8_t *qs = blk + 16;
  uint8_t byte = qs[(b / 2) * 32 + pl];
  return (b & 1) ? (byte >> 4) : (byte & 0x0F);
}

// block_q8_K activation (single row / M=1): float d, int8 qs[256], int16 bsums[16].
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } q8k_t;

// ---------------------------------------------------------------------------
// (B) The per-block CONTRIBUTION of one output column — the EXACT RVV GEVM fold
// leaf (integer core + bsums-min + dual d/dmin), shared by BOTH envelopes so the
// ONLY thing under test between them is the iteration structure, not the math.
// ---------------------------------------------------------------------------
static void gevm_block_contrib(const uint8_t *blk, const q8k_t *act,
                               int32_t *out_sumi, int32_t *out_bsums,
                               float *out_d0, float *out_dmins_d) {
  int sc[8], mn[8];
  for (int b = 0; b < 8; ++b) get_scale_min_k4(b, blk + 4, &sc[b], &mn[b]);
  float d_x = fp16_to_f32(load_fp16(blk + 0));
  float dmin_x = fp16_to_f32(load_fp16(blk + 2));
  *out_d0 = d_x * act->d;
  *out_dmins_d = dmin_x * act->d;

  int32_t sumi = 0, bsums = 0;
  for (int j = 0; j < 2; ++j) {          // super-half
    // MIN term: paired activation bsums * per-sub-block 6-bit min.
    for (int sb = 0; sb < 4; ++sb) {
      int g = j * 4 + sb;
      int32_t bs_pair = (int32_t)act->bsums[2 * g] + (int32_t)act->bsums[2 * g + 1];
      bsums += bs_pair * mn[g];
    }
    // MAIN term: 4 sub-blocks as 2 (low,high) pairs; each 32-elem dot split 2x16.
    for (int pair = 0; pair < 2; ++pair) {
      int gLo = j * 4 + pair * 2, gHi = gLo + 1;
      for (int kchunk = 0; kchunk < 2; ++kchunk) {
        int32_t sLo = 0, sHi = 0;
        for (int ii = 0; ii < 16; ++ii) {
          int i = kchunk * 16 + ii;
          int nLo = q4k_nibble(blk, gLo * 32 + i);
          int nHi = q4k_nibble(blk, gHi * 32 + i);
          sLo += (int32_t)act->qs[gLo * 32 + i] * nLo;
          sHi += (int32_t)act->qs[gHi * 32 + i] * nHi;
        }
        sumi += (int32_t)sc[gLo] * sLo + (int32_t)sc[gHi] * sHi;
      }
    }
  }
  *out_sumi = sumi;
  *out_bsums = bsums;
}

// ---------------------------------------------------------------------------
// (C) ENVELOPE 1 — the EXISTING per-column GEVM (== GEMM-plan-M=1 arithmetic):
// column-OUTER, block-INNER; the q8_K activation block is (re)read per column.
// ---------------------------------------------------------------------------
static void gevm_per_column(const uint8_t *W, const q8k_t *act, long N, long nb,
                            float *out) {
  for (long c = 0; c < N; ++c) {
    float sumf = 0.0f;
    for (long l = 0; l < nb; ++l) {
      const uint8_t *blk = W + (c * nb + l) * Q4K_BYTES;
      int32_t sumi, bsums; float d0, dmins_d;
      gevm_block_contrib(blk, &act[l], &sumi, &bsums, &d0, &dmins_d);
      sumf += (float)sumi * d0;          // vfmacc main
      sumf -= dmins_d * (float)bsums;    // vfnmsac min
    }
    out[c] = sumf;
  }
}

// ---------------------------------------------------------------------------
// (D) ENVELOPE 2 — the NEW independent GEVM Emission Plan: column-group-TILED,
// block-MIDDLE streaming, ONE activation block-load SHARED across the tile, a
// REGISTER-RESIDENT f32 accumulator BANK live across the block stream, weight-strip
// PREFETCH cadence in the structure. TG = column-groups per tile, CGW = 16 columns
// per group. The per-block leaf is IDENTICAL (gevm_block_contrib) — only the loop
// nest / accumulator residency / activation-sharing differ ([K-10] structural).
// ---------------------------------------------------------------------------
#define CGW 16
static void gevm_colgroup_tiled(const uint8_t *W, const q8k_t *act, long N,
                                long nb, int TG, float *out) {
  long groups = N / CGW;              // total 16-column groups
  long tiles = groups / TG;           // tiles of TG groups
  long bank = (long)TG * CGW;         // register-resident accumulator bank width
  for (long t = 0; t < tiles; ++t) {
    float accbank[64];                // TG*16 f32 accumulators, resident across K
    for (long r = 0; r < bank; ++r) accbank[r] = 0.0f;
    for (long l = 0; l < nb; ++l) {              // STREAMING K-reduction (middle)
      const q8k_t *a = &act[l];                  // ONE activation block, SHARED
      // prefetch cadence in structure: hint the next block's weight strips (the
      // resident repack layout, [PAT-2]-P9). A pure hint — never touches values.
      if (l + 1 < nb)
        for (int cg = 0; cg < TG; ++cg) {
          long g = t * TG + cg;
          __builtin_prefetch(W + ((g * CGW) * nb + (l + 1)) * Q4K_BYTES, 0, 3);
        }
      for (int cg = 0; cg < TG; ++cg) {          // column-group inner (tile)
        long g = t * TG + cg;
        for (int lane = 0; lane < CGW; ++lane) { // 16 columns share the a-load
          long c = g * CGW + lane;
          const uint8_t *blk = W + (c * nb + l) * Q4K_BYTES;
          int32_t sumi, bsums; float d0, dmins_d;
          gevm_block_contrib(blk, a, &sumi, &bsums, &d0, &dmins_d);
          accbank[cg * CGW + lane] += (float)sumi * d0;
          accbank[cg * CGW + lane] -= dmins_d * (float)bsums;
        }
      }
    }
    for (long r = 0; r < bank; ++r) out[t * bank + r] = accbank[r];
  }
}

// ---------------------------------------------------------------------------
// (E) INDEPENDENT ZERO-MODEL — re-derive the WHOLE computation from the raw packed
// bytes through a SEPARATE code path: first materialize explicit LOGICAL arrays
// (dequant-free integer nibbles W_log, activation A_log, per-sub-block sc/m, d/dmin),
// THEN fold in the q4_K order. Zero reuse of the kernel-path intermediates.
// ---------------------------------------------------------------------------
static void gevm_zero_model(const uint8_t *W, const q8k_t *act, long N, long nb,
                            float *out) {
  for (long c = 0; c < N; ++c) {
    float sumf = 0.0f;
    for (long l = 0; l < nb; ++l) {
      const uint8_t *blk = W + (c * nb + l) * Q4K_BYTES;
      // Independent logical materialization.
      int W_log[256];
      for (int p = 0; p < 256; ++p) W_log[p] = q4k_nibble(blk, p);
      int scL[8], mnL[8];
      for (int b = 0; b < 8; ++b) get_scale_min_k4(b, blk + 4, &scL[b], &mnL[b]);
      float d_x = fp16_to_f32(load_fp16(blk + 0));
      float dmin_x = fp16_to_f32(load_fp16(blk + 2));
      // Fold in the q4_K super-half / sub-block / split-2x16 order (== kernel fold).
      int32_t sumi = 0, bsums = 0;
      for (int j = 0; j < 2; ++j) {
        for (int sb = 0; sb < 4; ++sb) {
          int g = j * 4 + sb;
          int32_t bp = (int32_t)act[l].bsums[2 * g] + (int32_t)act[l].bsums[2 * g + 1];
          bsums += bp * mnL[g];
        }
        for (int pair = 0; pair < 2; ++pair) {
          int gLo = j * 4 + pair * 2, gHi = gLo + 1;
          for (int kchunk = 0; kchunk < 2; ++kchunk) {
            int32_t sLo = 0, sHi = 0;
            for (int ii = 0; ii < 16; ++ii) {
              int i = kchunk * 16 + ii;
              sLo += (int32_t)act[l].qs[gLo * 32 + i] * W_log[gLo * 32 + i];
              sHi += (int32_t)act[l].qs[gHi * 32 + i] * W_log[gHi * 32 + i];
            }
            sumi += (int32_t)scL[gLo] * sLo + (int32_t)scL[gHi] * sHi;
          }
        }
      }
      sumf += (float)sumi * (d_x * act[l].d);
      sumf -= (dmin_x * act[l].d) * (float)bsums;
    }
    out[c] = sumf;
  }
}

// ---------------------------------------------------------------------------
// (F) Anti-hollow CONTROLS (cert-hardening: a hollow oracle must FAIL these).
//   naive: element-wise dequant then plain dot — DIFFERENT float associativity =>
//          bounded-norm close but NOT bit-exact (proves the fold order is load-bearing).
//   nomin: drop the min term — must DIVERGE hugely (proves the min fold is real).
// ---------------------------------------------------------------------------
static float col_naive(const uint8_t *W, const q8k_t *act, long c, long nb, int drop_min) {
  double sum = 0.0;
  for (long l = 0; l < nb; ++l) {
    const uint8_t *blk = W + (c * nb + l) * Q4K_BYTES;
    int scL[8], mnL[8];
    for (int b = 0; b < 8; ++b) get_scale_min_k4(b, blk + 4, &scL[b], &mnL[b]);
    float d_x = fp16_to_f32(load_fp16(blk + 0));
    float dmin_x = fp16_to_f32(load_fp16(blk + 2));
    for (int p = 0; p < 256; ++p) {
      int b = p / 32;
      float w = d_x * scL[b] * q4k_nibble(blk, p);
      if (!drop_min) w -= dmin_x * mnL[b];
      sum += (double)act[l].d * w * (double)act[l].qs[p];
    }
  }
  return (float)sum;
}

static uint16_t rnd_finite_half(unsigned exp_bias) {  // finite positive normal
  return (uint16_t)((rand() & 0x03FF) | (exp_bias << 10));
}

int main(void) {
  srand(20260713u);
  const int TGs[] = {1, 2, 4};        // tile widths: 1 group .. 4 groups (64 cols)
  const long Ns[] = {16, 32, 64};     // output-column counts (multiples of 16)
  const long nbs[] = {1, 2, 3};       // super-blocks (contraction depth K = nb*256)

  long exact_mismatch = 0, zm_mismatch = 0, total = 0;
  double worst_naive_rel = 0.0, best_nomin_rel = 1e30;

  for (unsigned si = 0; si < sizeof(Ns) / sizeof(Ns[0]); ++si) {
    long N = Ns[si];
    for (unsigned bi = 0; bi < sizeof(nbs) / sizeof(nbs[0]); ++bi) {
      long nb = nbs[bi];

      // Weight: N columns x nb super-blocks, native block_q4_K, column-major blocks.
      uint8_t *W = (uint8_t *)malloc((size_t)N * nb * Q4K_BYTES);
      for (long i = 0; i < N * nb; ++i) {
        uint8_t *b = W + i * Q4K_BYTES;
        for (int k = 4; k < Q4K_BYTES; ++k) b[k] = (uint8_t)(rand() & 0xFF);  // scales+qs random
        uint16_t d = rnd_finite_half(13), dm = rnd_finite_half(12);
        b[0] = (uint8_t)(d & 0xFF); b[1] = (uint8_t)(d >> 8);
        b[2] = (uint8_t)(dm & 0xFF); b[3] = (uint8_t)(dm >> 8);
      }
      // Activation: nb block_q8_K rows; bsums DERIVED from qs (canonical q8_K).
      q8k_t *act = (q8k_t *)malloc((size_t)nb * sizeof(q8k_t));
      for (long l = 0; l < nb; ++l) {
        act[l].d = 0.05f + (float)(rand() % 1000) / 8000.0f;
        // POSITIVE-BIASED quants: nonzero-mean => large bsums => the dmin/min fold
        // carries real weight (cert corpus completeness: BOTH fold terms exercised,
        // and the NOMIN anti-hollow control becomes a strong discriminator).
        for (int p = 0; p < 256; ++p) act[l].qs[p] = (int8_t)(rand() % 100);
        for (int g = 0; g < 16; ++g) {
          int32_t s = 0;
          for (int e = 0; e < 16; ++e) s += act[l].qs[g * 16 + e];
          act[l].bsums[g] = (int16_t)s;
        }
      }

      float *ref = (float *)malloc((size_t)N * 4);
      float *zm = (float *)malloc((size_t)N * 4);
      gevm_per_column(W, act, N, nb, ref);
      gevm_zero_model(W, act, N, nb, zm);

      for (unsigned ti = 0; ti < sizeof(TGs) / sizeof(TGs[0]); ++ti) {
        int TG = TGs[ti];
        if ((N / CGW) % TG != 0) continue;   // tile must divide the group count
        float *tiled = (float *)malloc((size_t)N * 4);
        gevm_colgroup_tiled(W, act, N, nb, TG, tiled);
        for (long c = 0; c < N; ++c) {
          total++;
          if (memcmp(&tiled[c], &ref[c], 4) != 0) exact_mismatch++;   // BIT-exact
          if (memcmp(&tiled[c], &zm[c], 4) != 0) zm_mismatch++;       // BIT-exact
        }
        free(tiled);
      }

      // Anti-hollow controls (one representative column set).
      for (long c = 0; c < N; ++c) {
        float naive = col_naive(W, act, c, nb, 0);
        float nomin = col_naive(W, act, c, nb, 1);
        float denom = fabsf(ref[c]) + 1e-6f;
        double nr = fabs((double)naive - ref[c]) / denom;
        double mr = fabs((double)nomin - ref[c]) / denom;
        if (nr > worst_naive_rel) worst_naive_rel = nr;
        if (mr < best_nomin_rel) best_nomin_rel = mr;
      }
      free(W); free(act); free(ref); free(zm);
    }
  }

  printf("grid: N in {16,32,64} x nb in {1,2,3} x TG in {1,2,4}\n");
  printf("checked columns (tiled vs ref, tiled vs zero-model): %ld each\n", total);
  printf("(1) tiled-envelope == per-column(GEMM-M=1) : mismatch=%ld  [BIT-exact]\n",
         exact_mismatch);
  printf("(2) tiled-envelope == independent zero-model: mismatch=%ld  [BIT-exact]\n",
         zm_mismatch);
  printf("control naive-associativity  worst_rel = %.3e  (>0: fold order load-bearing)\n",
         worst_naive_rel);
  printf("control NOMIN (drop min fold) best_rel  = %.3e  (>>0: min fold real)\n",
         best_nomin_rel);

  int ok = (exact_mismatch == 0) && (zm_mismatch == 0) && (total > 0) &&
           (worst_naive_rel > 1e-8) && (best_nomin_rel > 1e-2);
  printf(ok ? "\nORACLE PASS\n" : "\nORACLE FAIL\n");
  return ok ? 0 : 1;
}
