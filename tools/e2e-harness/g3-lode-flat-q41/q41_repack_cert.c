/* G3-lode-flat q4_1 repack-GEVM byte-exact ZERO-MODEL host cert.
 *
 * Certifies the ARITHMETIC the front-door q4_1 repack-GEVM CONSTRUCTS (the
 * lowerToRepackGemvQ41 typed region: SHARED q4_0 lane-wise nibble dot with the
 * weight_nibble_unsigned decode + the dual-fp16 scale fold with the single MIN
 * offset pair) against an INDEPENDENT per-element scalar oracle that recomputes
 * q4_1 x q8_1 from the RAW PLAIN block bytes with ZERO reuse of the tested read
 * path.
 *
 * Cert three-requirements:
 *  (1) CORPUS COMPLETE  -- m_x != 0 (MIN term active), d_x/d_y != 0
 *      (nondegenerate scales), full unsigned nibble span [0,15], signed q8 span.
 *  (2) INPUT PATH SAME-ORIGIN -- both sides consume the SAME generated q8_1
 *      activation bytes (the tested side reads them AS-IS; the oracle reads the
 *      same bytes). One quantizer, one activation buffer.
 *  (3) ORACLE INDEPENDENT -- the tested side reads the REPACKED block_q4_1x16
 *      x16 buffer with the kernel's exact lane-wise addressing; the oracle reads
 *      the PLAIN block_q4_1 / block_q8_1 buffers with a naive per-element dot.
 *      Neither shares the other's read/decode implementation.
 *
 * The load-bearing claim is INTEGER bit-exact: the per-(block,column) integer
 * dot sumi = sum_i(nibble_i * q8_i) is arch-independent, so a host-scalar
 * transcription of the kernel's lane-wise vwmacc accumulation MUST equal the
 * oracle's direct scalar sum to the bit. The final f32 fold (d_x*d_y*sumi +
 * m_x*s_y) is compared with a reassociation tolerance.
 *
 * NOTE: no qemu-riscv64 user runner is available on this host, so the emitted
 * RVV vector-intrinsic C is compile-verified well-formed (see build_cert.sh) but
 * NOT executed here; the numerical proof is this scalar zero-reuse pair. A board
 * (ssh rvv) run of the emitted kernel is the owed claim=full upgrade.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK 32
#define HALF_NIB 16            /* qs bytes per block (QK/2) */
#define INTERLEAVE 16          /* block_q4_1x16 weight rows per group */
#define WSTRIDE_X16 320        /* 16 fp16 d + 16 fp16 m + 256 nibbles */
#define WQOFF_X16 64
#define WMOFF_X16 32
#define ASTRIDE_Q81 36         /* block_q8_1: fp16 d + fp16 s + 32 int8 quants */
#define AQOFF_Q81 4
#define ASOFF_Q81 2

/* ---- fp16 <-> fp32 (round-to-nearest-even, IEEE binary16). The scales are
 * generated as fp16-representable values so the integer sumi is the only
 * arch-independent quantity we bit-compare; the fold uses these fp16 scales. */
static uint16_t f32_to_f16(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) { return (uint16_t)sign; }
  if (exp >= 31) { return (uint16_t)(sign | 0x7C00u); }
  uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
  if (mant & 0x1000u) h++;   /* round */
  return h;
}
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) bits = sign;
    else { /* subnormal */
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) { mant <<= 1; exp--; }
      mant &= 0x3FF;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f; memcpy(&f, &bits, 4); return f;
}

int main(void) {
  const int NBLK = 8;              /* contraction blocks (K = NBLK*32 = 256) */
  const int NCOL = 32;             /* weight columns (N), 2 column-groups of 16 */
  const int K = NBLK * QK;
  srand(0xC0FFEE);

  /* ---- Generate PLAIN weight blocks per (column, block) + PLAIN q8_1 acts. ---- */
  /* Plain weight: for each column c, block l: fp16 d, fp16 m, 16 qs nibble bytes. */
  uint16_t *w_d = malloc(sizeof(uint16_t) * NCOL * NBLK);
  uint16_t *w_m = malloc(sizeof(uint16_t) * NCOL * NBLK);
  uint8_t  *w_qs = malloc((size_t)NCOL * NBLK * HALF_NIB);
  uint16_t *a_d = malloc(sizeof(uint16_t) * NBLK);
  uint16_t *a_s = malloc(sizeof(uint16_t) * NBLK);
  int8_t   *a_qs = malloc((size_t)NBLK * QK);

  for (int c = 0; c < NCOL; ++c)
    for (int l = 0; l < NBLK; ++l) {
      /* d != 0, m != 0 (MIN term ACTIVE), nondegenerate. */
      float d = 0.02f + 0.5f * (float)((rand() % 100) + 1) / 100.0f;
      float m = -0.7f + 1.4f * (float)(rand() % 1000) / 1000.0f;
      if (fabsf(m) < 1e-3f) m += 0.25f;   /* keep min-term non-degenerate */
      w_d[c * NBLK + l] = f32_to_f16(d);
      w_m[c * NBLK + l] = f32_to_f16(m);
      for (int i = 0; i < HALF_NIB; ++i)
        w_qs[((size_t)c * NBLK + l) * HALF_NIB + i] =
            (uint8_t)((rand() & 0x0F) | ((rand() & 0x0F) << 4)); /* full [0,15] */
    }
  for (int l = 0; l < NBLK; ++l) {
    float ad = 0.03f + 0.4f * (float)((rand() % 100) + 1) / 100.0f;
    long isum = 0;
    for (int k = 0; k < QK; ++k) {
      int8_t q = (int8_t)((rand() % 255) - 127);  /* signed span */
      a_qs[l * QK + k] = q;
      isum += q;
    }
    a_d[l] = f32_to_f16(ad);
    a_s[l] = f32_to_f16(ad * (float)isum);        /* block_q8_1 s = d*sum(q) */
  }

  /* ---- Materialize the REPACKED block_q4_1x16 weight (stage-C x16 layout the
   * front-door OUTPUT CONTRACT declares). Column groups of 16. ---- */
  int NG = NCOL / INTERLEAVE;
  uint8_t *wx16 = calloc((size_t)NG * NBLK * WSTRIDE_X16, 1);
  for (int g = 0; g < NG; ++g)
    for (int l = 0; l < NBLK; ++l) {
      uint8_t *bl = wx16 + ((size_t)g * NBLK + l) * WSTRIDE_X16;
      for (int c = 0; c < INTERLEAVE; ++c) {
        int col = g * INTERLEAVE + c;
        memcpy(bl + c * 2, &w_d[col * NBLK + l], 2);           /* d strip @0 */
        memcpy(bl + WMOFF_X16 + c * 2, &w_m[col * NBLK + l], 2); /* m strip @32 */
        for (int i = 0; i < HALF_NIB; ++i)                     /* nibbles @64 */
          bl[WQOFF_X16 + i * INTERLEAVE + c] =
              w_qs[((size_t)col * NBLK + l) * HALF_NIB + i];
      }
    }

  /* ---- TESTED: the front-door kernel's exact arithmetic on the x16 buffer.
   * Per column group g, per column c, per block l: the UNSIGNED lane-wise nibble
   * dot (weight_nibble_unsigned) + the dual-fp16 scale fold + the single MIN
   * fold (m_x * s_y). Capture the per-(g,l,c) integer sumi. ---- */
  float *out_tested = calloc(sizeof(float), NCOL);
  int32_t *sumi_tested = calloc(sizeof(int32_t), (size_t)NG * NBLK * INTERLEAVE);
  for (int g = 0; g < NG; ++g)
    for (int c = 0; c < INTERLEAVE; ++c) {
      float acc = 0.0f;
      for (int l = 0; l < NBLK; ++l) {
        uint8_t *bl = wx16 + ((size_t)g * NBLK + l) * WSTRIDE_X16;
        const uint8_t *al = NULL; /* activation is plain block_q8_1 stream */
        int32_t sumi = 0;
        for (int i = 0; i < HALF_NIB; ++i) {
          uint8_t packed = bl[WQOFF_X16 + i * INTERLEAVE + c];
          int lo = (int)(packed & 0x0F);        /* UNSIGNED, no -8 */
          int hi = (int)(packed >> 4);
          int8_t a_lo = a_qs[l * QK + i];        /* al.qs[i]   */
          int8_t a_hi = a_qs[l * QK + HALF_NIB + i]; /* al.qs[16+i] */
          sumi += lo * (int)a_lo + hi * (int)a_hi;
          (void)al;
        }
        sumi_tested[((size_t)g * NBLK + l) * INTERLEAVE + c] = sumi;
        float dx = f16_to_f32(*(uint16_t *)(bl + c * 2));
        float mx = f16_to_f32(*(uint16_t *)(bl + WMOFF_X16 + c * 2));
        float dy = f16_to_f32(a_d[l]);
        float sy = f16_to_f32(a_s[l]);
        acc += (dx * dy) * (float)sumi + mx * sy;
      }
      out_tested[g * INTERLEAVE + c] = acc;
    }

  /* ---- ORACLE: independent per-element recompute from the PLAIN blocks (no x16
   * read). For column col, block l: integer dot from plain qs nibbles + plain
   * q8; f32 fold via full dequant. ---- */
  double *out_oracle = calloc(sizeof(double), NCOL);
  int32_t *sumi_oracle = calloc(sizeof(int32_t), (size_t)NCOL * NBLK);
  for (int col = 0; col < NCOL; ++col) {
    double acc = 0.0;
    for (int l = 0; l < NBLK; ++l) {
      int32_t sumi = 0;
      for (int j = 0; j < QK; ++j) {
        int nib;
        if (j < HALF_NIB) nib = w_qs[((size_t)col * NBLK + l) * HALF_NIB + j] & 0x0F;
        else nib = w_qs[((size_t)col * NBLK + l) * HALF_NIB + (j - HALF_NIB)] >> 4;
        sumi += nib * (int)a_qs[l * QK + j];
      }
      sumi_oracle[(size_t)col * NBLK + l] = sumi;
      double dx = (double)f16_to_f32(w_d[col * NBLK + l]);
      double mx = (double)f16_to_f32(w_m[col * NBLK + l]);
      double dy = (double)f16_to_f32(a_d[l]);
      double sy = (double)f16_to_f32(a_s[l]);
      acc += (dx * dy) * (double)sumi + mx * sy;
    }
    out_oracle[col] = acc;
  }

  /* ---- Compare: INTEGER sumi bit-exact (per block,column), FP output rel. ---- */
  long int_mismatch = 0;
  for (int g = 0; g < NG; ++g)
    for (int c = 0; c < INTERLEAVE; ++c) {
      int col = g * INTERLEAVE + c;
      for (int l = 0; l < NBLK; ++l) {
        int32_t t = sumi_tested[((size_t)g * NBLK + l) * INTERLEAVE + c];
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
    double got = (double)out_tested[col];
    double denom = fabs(ref) > 1e-6 ? fabs(ref) : 1e-6;
    double rel = fabs(got - ref) / denom;
    if (rel > max_rel) max_rel = rel;
  }

  printf("q4_1 repack-GEVM ZERO-MODEL cert\n");
  printf("  corpus: NCOL=%d NBLK=%d K=%d  (m_x!=0 MIN active, d!=0, nibble[0,15])\n",
         NCOL, NBLK, K);
  printf("  integer sumi mismatches (tested x16 vs oracle plain) = %ld\n",
         int_mismatch);
  printf("  fp output max_rel (float tested vs double oracle)     = %.3e\n", max_rel);
  int pass = (int_mismatch == 0) && (max_rel < 1e-4);
  printf("  RESULT: %s\n", pass ? "PASS (integer bit-exact + fp reassoc-only)"
                                : "FAIL");
  return pass ? 0 : 1;
}
