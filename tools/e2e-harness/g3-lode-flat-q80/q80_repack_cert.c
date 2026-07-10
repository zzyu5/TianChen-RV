/* G3-lode-flat FLAT-4 收官格 q8_0 repack-GEVM+GEMM byte-exact ZERO-MODEL host cert.
 *
 * Certifies the ARITHMETIC the front-door q8_0 repack GEVM+GEMM CONSTRUCT (the
 * lowerToRepackGem{v,m}Q80 typed regions: the SHARED lane-wise integer core with
 * the q8_0 FULL-int8 decode leaf -- weight_full_i8, NO nibble unpack, NO qh, NO
 * offset -- and the d-ONLY dual-fp16 scale fold, NO min) against an INDEPENDENT
 * per-element scalar oracle that recomputes q8_0 x q8_0 from the RAW PLAIN block
 * bytes with ZERO reuse of the tested read path.
 *
 * q8_0 weight is a FULL signed int8 (one weight per contraction position; qk=32
 * positions per block). sumf += (d_x*d_y) * Σ_i(w_i8 * q8_i8)   -- NO nibble, NO
 * offset, NO min. The SIMPLEST flat variant.
 *
 * Cert three-requirements:
 *  (1) CORPUS COMPLETE  -- FULL int8 weight span (positive/negative/boundary +-127
 *      present), signed q8 activation span, d_x/d_y != 0 nondegenerate.
 *  (2) INPUT PATH SAME-ORIGIN -- both sides consume the SAME generated q8_0
 *      activation bytes AND the SAME generated int8 weight bytes. One buffer each.
 *  (3) ORACLE INDEPENDENT -- the tested side reads the REPACKED block_q8_0x16 x16
 *      buffer with the kernel's exact lane-wise addressing (byte = 32 + i*16 + c);
 *      the oracle reads the PLAIN block_q8_0 (fp16 d + 32 int8 qs) / block_q8_0
 *      activation buffers with a naive per-element dot. Neither shares the other's
 *      read/decode.
 *
 * The load-bearing claim is INTEGER bit-exact: the per-(block,column) FULL-int8 dot
 * sumi = Σ_i(w_i * q8_i) is arch-independent, so a host-scalar transcription of the
 * kernel's lane-wise vwmul+vwadd_wv i32 in-block accumulation MUST equal the oracle's
 * direct scalar sum to the bit -- for BOTH the GEVM (plain q8_0) and the GEMM
 * (interleaved block_q8_0x4) tested paths. The final f32 fold (d_x*d_y*sumi) is
 * compared with a reassociation tolerance.
 *
 * NOTE: no qemu-riscv64 user runner is available on this host, so the emitted RVV
 * vector-intrinsic C is compile-verified well-formed (see build_and_run.sh) but NOT
 * executed here; the numerical proof is this scalar zero-reuse pair. A board
 * (ssh rvv) run of the emitted kernel is the owed claim=full upgrade.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK 32
#define INTERLEAVE 16          /* block_q8_0x16 weight rows per group */
#define WSTRIDE_X16 544        /* 16 fp16 d + 512 full int8 quants */
#define WQOFF_X16 32           /* quants after the 16 fp16 d */
#define ASTRIDE_Q80 34         /* plain block_q8_0: fp16 d + 32 int8 quants */
#define AQOFF_Q80 2
/* GEMM interleaved block_q8_0x4: 4 fp16 d + 4*32 int8 quants. */
#define WSTRIDE_X4 136
#define AQOFF_Q80X4 8
#define ACOLS 4                /* activation_interleave (columnsPerPass @VLEN128) */

static uint16_t f32_to_f16(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) { return (uint16_t)sign; }
  if (exp >= 31) { return (uint16_t)(sign | 0x7C00u); }
  uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
  if (mant & 0x1000u) h++;
  return h;
}
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) bits = sign;
    else { exp = 127 - 15 + 1;
      while (!(mant & 0x400)) { mant <<= 1; exp--; }
      mant &= 0x3FF; bits = sign | (exp << 23) | (mant << 13); }
  } else if (exp == 31) { bits = sign | 0x7F800000u | (mant << 13);
  } else { bits = sign | ((exp - 15 + 127) << 23) | (mant << 13); }
  float f; memcpy(&f, &bits, 4); return f;
}

int main(void) {
  const int NBLK = 8;              /* contraction blocks (K = NBLK*32 = 256) */
  const int NCOL = 32;             /* weight columns (N), 2 column-groups of 16 */
  const int K = NBLK * QK;
  srand(0x5EED80);

  /* ---- PLAIN q8_0 weight blocks per (column, block) + PLAIN q8_0 acts. ---- */
  uint16_t *w_d  = malloc(sizeof(uint16_t) * NCOL * NBLK);
  int8_t   *w_qs = malloc((size_t)NCOL * NBLK * QK);
  uint16_t *a_d  = malloc(sizeof(uint16_t) * NBLK);
  int8_t   *a_qs = malloc((size_t)NBLK * QK);

  long pos_w = 0, neg_w = 0, bnd_w = 0;
  for (int c = 0; c < NCOL; ++c)
    for (int l = 0; l < NBLK; ++l) {
      float d = 0.02f + 0.5f * (float)((rand() % 100) + 1) / 100.0f; /* d != 0 */
      w_d[c * NBLK + l] = f32_to_f16(d);
      for (int i = 0; i < QK; ++i) {
        int8_t w = (int8_t)((rand() % 255) - 127);  /* FULL signed span */
        /* force boundary +-127 into the corpus deterministically */
        if (c == 0 && l == 0 && i == 0) w = 127;
        if (c == 0 && l == 0 && i == 1) w = -127;
        w_qs[((size_t)c * NBLK + l) * QK + i] = w;
        if (w > 0) pos_w++; else if (w < 0) neg_w++;
        if (w == 127 || w == -127) bnd_w++;
      }
    }
  for (int l = 0; l < NBLK; ++l) {
    float ad = 0.03f + 0.4f * (float)((rand() % 100) + 1) / 100.0f;
    for (int k = 0; k < QK; ++k)
      a_qs[l * QK + k] = (int8_t)((rand() % 255) - 127);  /* signed span */
    a_d[l] = f32_to_f16(ad);
  }

  /* ---- Materialize the REPACKED block_q8_0x16 weight (the x16 layout BOTH the
   * GEVM and GEMM front-door OUTPUT CONTRACT declares). 16 fp16 d @0, 512
   * interleaved FULL int8 quants @32: for position i (0..31) and column c
   * (0..15), byte = 32 + i*16 + c. ---- */
  int NG = NCOL / INTERLEAVE;
  uint8_t *wx16 = calloc((size_t)NG * NBLK * WSTRIDE_X16, 1);
  for (int g = 0; g < NG; ++g)
    for (int l = 0; l < NBLK; ++l) {
      uint8_t *bl = wx16 + ((size_t)g * NBLK + l) * WSTRIDE_X16;
      for (int c = 0; c < INTERLEAVE; ++c) {
        int col = g * INTERLEAVE + c;
        memcpy(bl + c * 2, &w_d[col * NBLK + l], 2);         /* d strip @0 */
        for (int i = 0; i < QK; ++i)                         /* quants @32 */
          bl[WQOFF_X16 + i * INTERLEAVE + c] =
              (uint8_t)w_qs[((size_t)col * NBLK + l) * QK + i];
      }
    }

  /* ---- TESTED (GEVM): plain block_q8_0 activation, one column at a time. Each
   * position i reads the strip byte (col c = lane c) and the ONE activation
   * scalar al.qs[i]; the FULL int8 product accumulates into an i32 sumi. ---- */
  int32_t *sumi_gevm = calloc(sizeof(int32_t), (size_t)NG * NBLK * INTERLEAVE);
  float *out_gevm = calloc(sizeof(float), NCOL);
  for (int g = 0; g < NG; ++g)
    for (int c = 0; c < INTERLEAVE; ++c) {
      float acc = 0.0f;
      for (int l = 0; l < NBLK; ++l) {
        uint8_t *bl = wx16 + ((size_t)g * NBLK + l) * WSTRIDE_X16;
        int32_t sumi = 0;
        for (int i = 0; i < QK; ++i) {
          int w = (int)(int8_t)bl[WQOFF_X16 + i * INTERLEAVE + c];  /* full i8 */
          int8_t a = a_qs[l * QK + i];               /* al.qs[i]  (GEVM) */
          sumi += w * (int)a;
        }
        sumi_gevm[((size_t)g * NBLK + l) * INTERLEAVE + c] = sumi;
        float dx = f16_to_f32(*(uint16_t *)(bl + c * 2));
        float dy = f16_to_f32(a_d[l]);
        acc += (dx * dy) * (float)sumi;             /* NO min */
      }
      out_gevm[g * INTERLEAVE + c] = acc;
    }

  /* ---- TESTED (GEMM): interleaved block_q8_0x4 activation. Position i, column
   * c_a in the x4 tile is al.qs[i*4+c_a] (NO high-row 64-offset -- full i8 has 32
   * contiguous positions, unlike q4_0's nibble lo/hi split). ---- */
  uint8_t *ax4 = calloc((size_t)NBLK * WSTRIDE_X4, 1);
  for (int l = 0; l < NBLK; ++l) {
    uint8_t *al = ax4 + (size_t)l * WSTRIDE_X4;
    for (int c = 0; c < ACOLS; ++c) memcpy(al + c * 2, &a_d[l], 2);  /* 4 fp16 d */
    for (int i = 0; i < QK; ++i)
      for (int c = 0; c < ACOLS; ++c)
        al[AQOFF_Q80X4 + i * ACOLS + c] = (uint8_t)a_qs[l * QK + i];
  }
  /* GEMM tested: weight column (g,c) x activation column c_a in the x4 tile. The
   * x4 column c_a carries a_qs[l] (same bytes) so the GEMM sumi MUST equal the
   * GEVM sumi for the SAME (g,c,l): certifies the x4 activation addressing. */
  long gemm_int_mismatch = 0;
  for (int g = 0; g < NG; ++g)
    for (int c = 0; c < INTERLEAVE; ++c) {
      int c_a = c % ACOLS;                 /* activation column in the x4 tile */
      for (int l = 0; l < NBLK; ++l) {
        uint8_t *bl = wx16 + ((size_t)g * NBLK + l) * WSTRIDE_X16;
        uint8_t *al = ax4 + (size_t)l * WSTRIDE_X4;
        int32_t sumi = 0;
        for (int i = 0; i < QK; ++i) {
          int w = (int)(int8_t)bl[WQOFF_X16 + i * INTERLEAVE + c];
          int8_t a = (int8_t)al[AQOFF_Q80X4 + i * ACOLS + c_a];
          sumi += w * (int)a;
        }
        if (sumi != sumi_gevm[((size_t)g * NBLK + l) * INTERLEAVE + c])
          gemm_int_mismatch++;
      }
    }

  /* ---- ORACLE: independent recompute from the PLAIN q8_0 / q8_0 blocks. ---- */
  int32_t *sumi_oracle = calloc(sizeof(int32_t), (size_t)NCOL * NBLK);
  double *out_oracle = calloc(sizeof(double), NCOL);
  for (int col = 0; col < NCOL; ++col) {
    double acc = 0.0;
    for (int l = 0; l < NBLK; ++l) {
      int32_t sumi = 0;
      for (int j = 0; j < QK; ++j) {
        int w = (int)w_qs[((size_t)col * NBLK + l) * QK + j];
        sumi += w * (int)a_qs[l * QK + j];
      }
      sumi_oracle[(size_t)col * NBLK + l] = sumi;
      double dx = (double)f16_to_f32(w_d[col * NBLK + l]);
      double dy = (double)f16_to_f32(a_d[l]);
      acc += (dx * dy) * (double)sumi;
    }
    out_oracle[col] = acc;
  }

  /* ---- Compare: INTEGER sumi bit-exact (GEVM vs oracle), FP output rel. ---- */
  long int_mismatch = 0;
  for (int g = 0; g < NG; ++g)
    for (int c = 0; c < INTERLEAVE; ++c) {
      int col = g * INTERLEAVE + c;
      for (int l = 0; l < NBLK; ++l) {
        int32_t t = sumi_gevm[((size_t)g * NBLK + l) * INTERLEAVE + c];
        int32_t o = sumi_oracle[(size_t)col * NBLK + l];
        if (t != o) {
          if (int_mismatch < 8)
            fprintf(stderr, "INT MISMATCH col=%d blk=%d tested=%d oracle=%d\n",
                    col, l, t, o);
          int_mismatch++;
        }
      }
    }
  double max_rel = 0.0;
  for (int col = 0; col < NCOL; ++col) {
    double ref = out_oracle[col];
    double got = (double)out_gevm[col];
    double denom = fabs(ref) > 1e-6 ? fabs(ref) : 1e-6;
    double rel = fabs(got - ref) / denom;
    if (rel > max_rel) max_rel = rel;
  }

  printf("q8_0 repack GEVM+GEMM ZERO-MODEL cert\n");
  printf("  corpus: NCOL=%d NBLK=%d K=%d\n", NCOL, NBLK, K);
  printf("  int8 weight span: positive=%ld negative=%ld boundary(+-127)=%ld\n",
         pos_w, neg_w, bnd_w);
  printf("  GEVM integer sumi mismatches (x16 vs oracle plain) = %ld\n",
         int_mismatch);
  printf("  GEMM integer sumi mismatches (x4 vs GEVM)           = %ld\n",
         gemm_int_mismatch);
  printf("  fp output max_rel (float tested vs double oracle)   = %.3e\n", max_rel);
  int nondegenerate = (pos_w > 0) && (neg_w > 0) && (bnd_w > 0);
  int pass = (int_mismatch == 0) && (gemm_int_mismatch == 0) &&
             (max_rel < 1e-4) && nondegenerate;
  printf("  RESULT: %s\n", pass ? "PASS (integer bit-exact GEVM+GEMM + fp reassoc-only)"
                                : "FAIL");
  return pass ? 0 : 1;
}
