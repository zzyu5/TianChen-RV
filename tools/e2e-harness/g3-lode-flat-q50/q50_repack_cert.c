/* G3-lode-flat FLAT-2 q5_0 repack-GEVM+GEMM byte-exact ZERO-MODEL host cert.
 *
 * Certifies the ARITHMETIC the front-door q5_0 repack GEVM+GEMM CONSTRUCT (the
 * lowerToRepackGem{v,m}Q50 typed regions: the SHARED q4_0 lane-wise integer core
 * with the q5_0 5th-bit decode leaf -- weight_nibble_unsigned + a transposed qh
 * plane + the -16 offset-binary bias -- and the d-ONLY dual-fp16 scale fold, NO
 * min) against an INDEPENDENT per-element scalar oracle that recomputes q5_0 x
 * q8_0 from the RAW PLAIN block bytes with ZERO reuse of the tested read path.
 *
 * q5_0 weight reconstruct: w = ((qs_nibble) | (qh_bit << 4)) - 16, in [-16,15].
 * The low element j (j in [0,16)) takes qh bit j, the high element j+16 takes qh
 * bit j+16. sumf += (d_x*d_y) * Σ_i(w_i * q8_i)   -- NO min term (unlike q4_1).
 *
 * Cert three-requirements:
 *  (1) CORPUS COMPLETE  -- qh bits SET and CLEAR both exercised, POST-OFFSET
 *      NEGATIVE weights present (w < 0 after -16), full nibble span, signed q8
 *      span, d_x/d_y != 0 nondegenerate. Non-degenerate (not all-qh-set / clear).
 *  (2) INPUT PATH SAME-ORIGIN -- both sides consume the SAME generated q8_0
 *      activation bytes. One quantizer, one activation buffer.
 *  (3) ORACLE INDEPENDENT -- the tested side reads the REPACKED block_q5_0x16 x16
 *      buffer with the kernel's exact lane-wise addressing AND the transposed qh
 *      mask expansion; the oracle reads the PLAIN block_q5_0 (fp16 d + u32 qh + 16
 *      qs) / block_q8_0 buffers with a naive per-element dot off the qh bit index.
 *      Neither shares the other's read/decode.
 *
 * The load-bearing claim is INTEGER bit-exact: the per-(block,column) reconstructed
 * 5-bit-weight dot sumi = Σ_i(w_i * q8_i) (w in [-16,15]) is arch-independent, so a
 * host-scalar transcription of the kernel's lane-wise vwmacc accumulation MUST
 * equal the oracle's direct scalar sum to the bit -- for BOTH the GEVM (plain q8_0)
 * and the GEMM (interleaved block_q8_0x4, RUNTIME-strip qh) tested paths. The final
 * f32 fold (d_x*d_y*sumi) is compared with a reassociation tolerance.
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
#define HALF_NIB 16            /* qs bytes per block (QK/2) */
#define INTERLEAVE 16          /* block_q5_0x16 weight rows per group */
#define WSTRIDE_X16 352        /* 16 fp16 d + 256 nibbles + 64 transposed qh */
#define WQOFF_X16 32           /* nibbles after the 16 fp16 d */
#define WQHOFF_X16 288         /* transposed qh after the 256 nibbles */
#define ASTRIDE_Q80 34         /* plain block_q8_0: fp16 d + 32 int8 quants */
#define AQOFF_Q80 2
/* GEMM interleaved block_q8_0x4: 4 fp16 d + 4*32 int8 quants. */
#define AQOFF_Q80X4 8
#define AHIROW_Q80X4 64        /* activation_interleave(4) * nibbleBytes(16) */
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
  srand(0x5EED50);

  /* ---- PLAIN q5_0 weight blocks per (column, block) + PLAIN q8_0 acts. ---- */
  uint16_t *w_d  = malloc(sizeof(uint16_t) * NCOL * NBLK);
  uint32_t *w_qh = malloc(sizeof(uint32_t) * NCOL * NBLK);      /* 32 qh bits */
  uint8_t  *w_qs = malloc((size_t)NCOL * NBLK * HALF_NIB);
  uint16_t *a_d  = malloc(sizeof(uint16_t) * NBLK);
  int8_t   *a_qs = malloc((size_t)NBLK * QK);

  long qh_set = 0, qh_clear = 0, neg_w = 0;
  for (int c = 0; c < NCOL; ++c)
    for (int l = 0; l < NBLK; ++l) {
      float d = 0.02f + 0.5f * (float)((rand() % 100) + 1) / 100.0f; /* d != 0 */
      w_d[c * NBLK + l] = f32_to_f16(d);
      uint32_t qh = ((uint32_t)rand() & 0xFFFFu) | ((uint32_t)rand() << 16);
      w_qh[c * NBLK + l] = qh;
      for (int i = 0; i < HALF_NIB; ++i)
        w_qs[((size_t)c * NBLK + l) * HALF_NIB + i] =
            (uint8_t)((rand() & 0x0F) | ((rand() & 0x0F) << 4)); /* full [0,15] */
      /* corpus accounting: qh bit coverage + post-offset negative weights */
      for (int j = 0; j < QK; ++j) {
        int qhb = (int)((qh >> j) & 1u);
        if (qhb) qh_set++; else qh_clear++;
        int nib = (j < HALF_NIB)
                      ? (w_qs[((size_t)c * NBLK + l) * HALF_NIB + j] & 0x0F)
                      : (w_qs[((size_t)c * NBLK + l) * HALF_NIB + (j - HALF_NIB)] >> 4);
        if (((nib | (qhb << 4)) - 16) < 0) neg_w++;
      }
    }
  for (int l = 0; l < NBLK; ++l) {
    float ad = 0.03f + 0.4f * (float)((rand() % 100) + 1) / 100.0f;
    for (int k = 0; k < QK; ++k)
      a_qs[l * QK + k] = (int8_t)((rand() % 255) - 127);  /* signed span */
    a_d[l] = f32_to_f16(ad);
  }

  /* ---- Materialize the REPACKED block_q5_0x16 weight (the x16 layout BOTH the
   * GEVM and GEMM front-door OUTPUT CONTRACT declares). 16 fp16 d @0, 256
   * interleaved nibbles @32, 64 transposed qh bytes @288: for element step e
   * (0..31), ONE 16-bit mask whose bit b is block(=col in group) b's qh bit e. ---- */
  int NG = NCOL / INTERLEAVE;
  uint8_t *wx16 = calloc((size_t)NG * NBLK * WSTRIDE_X16, 1);
  for (int g = 0; g < NG; ++g)
    for (int l = 0; l < NBLK; ++l) {
      uint8_t *bl = wx16 + ((size_t)g * NBLK + l) * WSTRIDE_X16;
      for (int c = 0; c < INTERLEAVE; ++c) {
        int col = g * INTERLEAVE + c;
        memcpy(bl + c * 2, &w_d[col * NBLK + l], 2);         /* d strip @0 */
        for (int i = 0; i < HALF_NIB; ++i)                   /* nibbles @32 */
          bl[WQOFF_X16 + i * INTERLEAVE + c] =
              w_qs[((size_t)col * NBLK + l) * HALF_NIB + i];
      }
      /* transposed qh: mask[e] bit c = col's qh bit e. */
      for (int e = 0; e < QK; ++e) {
        uint16_t mask = 0;
        for (int c = 0; c < INTERLEAVE; ++c) {
          int col = g * INTERLEAVE + c;
          mask |= (uint16_t)(((w_qh[col * NBLK + l] >> e) & 1u) << c);
        }
        memcpy(bl + WQHOFF_X16 + e * 2, &mask, 2);
      }
    }

  /* ---- TESTED (GEVM): plain block_q8_0 activation, one column at a time. The
   * per-strip qh bit for column c is (mask[e] >> c) & 1 -- lane c of the strip. ---- */
  int32_t *sumi_gevm = calloc(sizeof(int32_t), (size_t)NG * NBLK * INTERLEAVE);
  float *out_gevm = calloc(sizeof(float), NCOL);
  for (int g = 0; g < NG; ++g)
    for (int c = 0; c < INTERLEAVE; ++c) {
      float acc = 0.0f;
      for (int l = 0; l < NBLK; ++l) {
        uint8_t *bl = wx16 + ((size_t)g * NBLK + l) * WSTRIDE_X16;
        int32_t sumi = 0;
        for (int i = 0; i < HALF_NIB; ++i) {
          uint8_t p = bl[WQOFF_X16 + i * INTERLEAVE + c];
          uint16_t mlo, mhi;
          memcpy(&mlo, bl + WQHOFF_X16 + i * 2, 2);
          memcpy(&mhi, bl + WQHOFF_X16 + (HALF_NIB + i) * 2, 2);
          int lo = (int)(((p & 0x0F) | (((mlo >> c) & 1u) << 4))) - 16;
          int hi = (int)(((p >> 4)   | (((mhi >> c) & 1u) << 4))) - 16;
          int8_t a_lo = a_qs[l * QK + i];             /* al.qs[i]    (GEVM) */
          int8_t a_hi = a_qs[l * QK + HALF_NIB + i];  /* al.qs[16+i] (GEVM) */
          sumi += lo * (int)a_lo + hi * (int)a_hi;
        }
        sumi_gevm[((size_t)g * NBLK + l) * INTERLEAVE + c] = sumi;
        float dx = f16_to_f32(*(uint16_t *)(bl + c * 2));
        float dy = f16_to_f32(a_d[l]);
        acc += (dx * dy) * (float)sumi;             /* NO min */
      }
      out_gevm[g * INTERLEAVE + c] = acc;
    }

  /* ---- TESTED (GEMM): interleaved block_q8_0x4 activation, RUNTIME-strip qh
   * (col c's qh bit = mask[e] >> c, exactly as the GEVM lane, but the activation
   * is now al.qs[i*4+c] (low) / al.qs[64+i*4+c] (high) in the x4 layout). ---- */
  /* Materialize block_q8_0x4 from the SAME a_qs bytes (4-column interleave). */
  uint8_t *ax4 = calloc((size_t)NBLK * 136, 1);
  for (int l = 0; l < NBLK; ++l) {
    uint8_t *al = ax4 + (size_t)l * 136;
    /* 4 fp16 d @0 (all 4 "columns" of this M-tile share a_d[l] here) */
    for (int c = 0; c < ACOLS; ++c) memcpy(al + c * 2, &a_d[l], 2);
    for (int i = 0; i < HALF_NIB; ++i)
      for (int c = 0; c < ACOLS; ++c) {
        al[AQOFF_Q80X4 + i * ACOLS + c] = (uint8_t)a_qs[l * QK + i];
        al[AQOFF_Q80X4 + AHIROW_Q80X4 + i * ACOLS + c] =
            (uint8_t)a_qs[l * QK + HALF_NIB + i];
      }
  }
  /* GEMM tested: weight column (g,c_w) x activation column c_a in the x4 tile.
   * Certify the weight-side 5-bit reconstruct + the x4 activation addressing on
   * the diagonal c_a == the weight strip lane (the shared integer dot). */
  long gemm_int_mismatch = 0;
  for (int g = 0; g < NG; ++g)
    for (int c = 0; c < INTERLEAVE; ++c) {
      int c_a = c % ACOLS;                 /* activation column in the x4 tile */
      for (int l = 0; l < NBLK; ++l) {
        uint8_t *bl = wx16 + ((size_t)g * NBLK + l) * WSTRIDE_X16;
        uint8_t *al = ax4 + (size_t)l * 136;
        int32_t sumi = 0;
        for (int i = 0; i < HALF_NIB; ++i) {
          uint8_t p = bl[WQOFF_X16 + i * INTERLEAVE + c];
          uint16_t mlo, mhi;
          memcpy(&mlo, bl + WQHOFF_X16 + i * 2, 2);
          memcpy(&mhi, bl + WQHOFF_X16 + (HALF_NIB + i) * 2, 2);
          int lo = (int)(((p & 0x0F) | (((mlo >> c) & 1u) << 4))) - 16;
          int hi = (int)(((p >> 4)   | (((mhi >> c) & 1u) << 4))) - 16;
          int8_t a_lo = (int8_t)al[AQOFF_Q80X4 + i * ACOLS + c_a];
          int8_t a_hi = (int8_t)al[AQOFF_Q80X4 + AHIROW_Q80X4 + i * ACOLS + c_a];
          sumi += lo * (int)a_lo + hi * (int)a_hi;
        }
        /* x4 activation col c_a carries a_qs[l] (same bytes) so the GEMM sumi
         * MUST equal the GEVM sumi for the SAME (g,c,l): certifies the qh
         * runtime-strip + x4 addressing. */
        if (sumi != sumi_gevm[((size_t)g * NBLK + l) * INTERLEAVE + c])
          gemm_int_mismatch++;
      }
    }

  /* ---- ORACLE: independent recompute from the PLAIN q5_0 / q8_0 blocks. ---- */
  int32_t *sumi_oracle = calloc(sizeof(int32_t), (size_t)NCOL * NBLK);
  double *out_oracle = calloc(sizeof(double), NCOL);
  for (int col = 0; col < NCOL; ++col) {
    double acc = 0.0;
    for (int l = 0; l < NBLK; ++l) {
      uint32_t qh = w_qh[col * NBLK + l];
      int32_t sumi = 0;
      for (int j = 0; j < QK; ++j) {
        int nib = (j < HALF_NIB)
                      ? (w_qs[((size_t)col * NBLK + l) * HALF_NIB + j] & 0x0F)
                      : (w_qs[((size_t)col * NBLK + l) * HALF_NIB + (j - HALF_NIB)] >> 4);
        int qhb = (int)((qh >> j) & 1u);
        int w = ((nib | (qhb << 4)) - 16);
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

  printf("q5_0 repack GEVM+GEMM ZERO-MODEL cert\n");
  printf("  corpus: NCOL=%d NBLK=%d K=%d\n", NCOL, NBLK, K);
  printf("  qh bits set=%ld clear=%ld  post-offset negative weights=%ld\n",
         qh_set, qh_clear, neg_w);
  printf("  GEVM integer sumi mismatches (x16 vs oracle plain) = %ld\n",
         int_mismatch);
  printf("  GEMM integer sumi mismatches (x4 runtime-strip vs GEVM) = %ld\n",
         gemm_int_mismatch);
  printf("  fp output max_rel (float tested vs double oracle)    = %.3e\n", max_rel);
  int nondegenerate = (qh_set > 0) && (qh_clear > 0) && (neg_w > 0);
  int pass = (int_mismatch == 0) && (gemm_int_mismatch == 0) &&
             (max_rel < 1e-4) && nondegenerate;
  printf("  RESULT: %s\n", pass ? "PASS (integer bit-exact GEVM+GEMM + fp reassoc-only)"
                                : "FAIL");
  return pass ? 0 : 1;
}
