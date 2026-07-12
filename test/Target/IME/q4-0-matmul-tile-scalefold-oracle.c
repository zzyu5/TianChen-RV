// G5-M3 host oracle for the format-keyed q4_0 IME GEMM tile SCALE-FOLD epilogue.
//
// This is the forward-bridge counterpart of q4-0-matmul-tile-int32-oracle.c. The
// int32 oracle validates only the int32-EXACT MAC accumulator (the seal object).
// A real ggml q4_0 mul_mat forward needs the PER-BLOCK fp16 scale fold applied on
// top of that int32 core: the ggml q4_0 x q8_0 dot over one 32-element block is
//   d_a * d_w * Sum_{k in block} qa[k] * qw[k]
// summed over the K/32 contraction blocks. The int32 kernel produces the inner
// Sum(qa*qw) per block; this file adds the tcrv_ime_q4_0_vmadot_matmul_f32
// scale-fold epilogue (bridge #1) driven at RUNTIME shape (bridge #2) and checks
// it against an INDEPENDENT (ZERO-MODEL) canonical q4_0 x q8_0 reference.
//
// The scale-fold kernel REUSES verbatim the seal-proven decode + batched vmadot
// MAC leaf (here the scalar-substitute MAC, exactly as the int32 oracle does; the
// real vmadot "how" is board-proven in q4-0-matmul-tile-scalefold-k1seal.c). The
// ONLY new arithmetic vs the sealed int32 core is the per-block d_a*d_w*partial
// float fold; everything integer is byte-identical to the M1b seal.
//
// Two checks (both over a multi-shape grid to exercise runtime shape):
//   (1) int32 core: per-block int32 partials from the kernel path == independent
//       partials re-derived from the packed bytes (ZERO-MODEL, int32-EXACT).
//   (2) scale fold: kernel f32 output == the canonical q4_0 x q8_0 f32 reference
//       (quantize the original f32 W/X independently, plain triple-loop with the
//       per-block scale fold), BOUNDED-ULP (f32 reassociation only).
//
// Build + run:
//   cc -O2 -std=c11 q4-0-matmul-tile-scalefold-oracle.c -lm -o /tmp/q40sf && /tmp/q40sf
// Exit 0 + "ORACLE PASS" iff (1) is 0-diff and (2) is within the f32 tolerance.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// (A) The emitted structured helpers. The int32 decode + batched MAC are the
// M1b-sealed helpers (reused verbatim). The scale-fold epilogue kernel
// tcrv_ime_q4_0_vmadot_matmul_f32 is the NEW forward-bridge helper (bridge #1).
// ---------------------------------------------------------------------------

// Scalar substitute for the batched register-resident vmadot MAC leaf (int32
// contract identical to the K1-sealed asm leaf; see q4-0-matmul-tile-int32-oracle.c).
static inline void tcrv_ime_vmadot_mac_kloop(const int8_t *A, const int8_t *B,
                                             long kt, int32_t *frag) {
  int32_t acc[16];
  for (int r = 0; r < 16; ++r) acc[r] = 0;
  for (long kf = 0; kf < kt; ++kf) {
    const int8_t *Af = A + kf * 32;
    const int8_t *Bf = B + kf * 32;
    for (int m = 0; m < 4; ++m)
      for (int n = 0; n < 4; ++n) {
        int32_t s = 0;
        for (int k = 0; k < 8; ++k)
          s += (int32_t)Af[m * 8 + k] * (int32_t)Bf[n * 8 + k];
        acc[m * 4 + n] += s;
      }
  }
  for (int r = 0; r < 16; ++r) frag[r] = acc[r];
}

// The q4_0 offset-binary nibble DECODE (identical to the emitted/sealed decode core).
static inline void tcrv_ime_q4_0_dequant_fragment(const uint8_t *blk,
                                                  int8_t *out) {
  const uint8_t *qs = blk + 2; // past the 2-byte fp16 d slot (unused here; the
                               // real per-column d lives in the parallel dW array)
  for (int j = 0; j < 16; ++j) {
    out[j] = (int8_t)((int)(qs[j] & 0x0F) - 8);
    out[j + 16] = (int8_t)((int)(qs[j] >> 4) - 8);
  }
}

// The q4_0 SCALE-FOLD f32 GEMM (bridge #1 + #2). Per 4x4 output tile it walks the
// K/32 contraction blocks; per block it (a) decodes the 4 fragment-major q4_0
// weight blocks into a 128-int8 buffer, (b) runs ONE register-resident batched MAC
// over the block's 4 fragments -> the int32 partial Sum(qa*qw) for this 32-block
// (int32-EXACT; the sealed core), then (c) folds d_a*d_w*partial into the f32
// accumulator. The weight nibble pack (Bnib) is fragment-major 18-byte blocks (the
// SAME layout the int32 seal uses); the per-(column,block) fp16 weight scale is
// carried in the parallel dW array; the per-(row,block) activation scale in dA.
// M/N/K are RUNTIME parameters (bridge #2: the fixed micro-tile is generalized to
// the tensor's real shape).
static void tcrv_ime_q4_0_vmadot_matmul_f32(const int8_t *Apack, const float *dA,
                                            const uint8_t *Bnib, const float *dW,
                                            float *Cf, long M, long N, long K) {
  const long mt = M / 4, nt = N / 4, nb = K / 32; // nb = # of 32-element blocks
  const long q40_block_bytes = 18;
  const long frags_per_block = 4;                 // 32 / 8 fragments per block
  const long kt = K / 8;                          // total fragments per col-tile
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Apack + (long)mi * 4 * K;
    for (long nj = 0; nj < nt; ++nj) {
      const uint8_t *Bcol = Bnib + (long)nj * kt * q40_block_bytes;
      for (long b = 0; b < nb; ++b) {
        int8_t Bdec[128];
        for (long f = 0; f < frags_per_block; ++f)
          tcrv_ime_q4_0_dequant_fragment(
              Bcol + (b * frags_per_block + f) * q40_block_bytes, Bdec + f * 32);
        int32_t frag[16];
        // A fragments for this block: global fragments b*4 .. b*4+3, contiguous.
        tcrv_ime_vmadot_mac_kloop(Arow + b * frags_per_block * 32, Bdec,
                                  frags_per_block, frag);
        for (long r = 0; r < 4; ++r)
          for (long c = 0; c < 4; ++c) {
            long m = mi * 4 + r, n = nj * 4 + c;
            Cf[m * N + n] += dA[m * nb + b] * dW[n * nb + b] * (float)frag[r * 4 + c];
          }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// (B) The INDEPENDENT reference (ZERO-MODEL). Canonical ggml q4_0 weight quant +
// canonical q8_0 activation quant of the ORIGINAL f32 matrices, then a plain
// triple-loop GEMM with the per-block scale fold. Different code path than the
// fragment kernel: it never touches the fragment-major packed bytes.
// ---------------------------------------------------------------------------

// Deterministic IEEE half<->float (matches the emitter's fp16 helpers).
static uint16_t f32_to_fp16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFF) == 0xFF) // inf/nan
    return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0));
  if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u); // overflow -> inf
  if (exp <= 0) {                                     // subnormal/zero
    if (exp < -10) return (uint16_t)sign;
    mant |= 0x800000u;
    uint32_t shift = (uint32_t)(14 - exp);
    uint32_t half = mant >> shift;
    if ((mant >> (shift - 1)) & 1) half += 1; // round to nearest
    return (uint16_t)(sign | half);
  }
  uint16_t half = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
  if ((mant >> 12) & 1) half += 1; // round to nearest even-ish
  return half;
}
static float fp16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0u) {
    if (mant == 0u) bits = sign;
    else {
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

// Canonical ggml q4_0 quant of one 32-element block: d = amax/-8 (fp16-rounded),
// nibble = clamp(round(x/d) + 8, 0, 15). Returns fp16 d; fills 16 nibble bytes.
static uint16_t ref_quant_q4_0_block(const float *x, uint8_t qs[16]) {
  float amax = 0.0f, max = 0.0f;
  for (int j = 0; j < 32; ++j) {
    float a = fabsf(x[j]);
    if (a > amax) { amax = a; max = x[j]; }
  }
  float d = max / -8.0f;
  uint16_t dh = f32_to_fp16(d);
  float dr = fp16_to_f32(dh);
  float id = dr ? 1.0f / dr : 0.0f;
  for (int j = 0; j < 16; ++j) {
    int x0 = (int)(x[j] * id + 8.5f);
    int x1 = (int)(x[j + 16] * id + 8.5f);
    if (x0 < 0) x0 = 0; if (x0 > 15) x0 = 15;
    if (x1 < 0) x1 = 0; if (x1 > 15) x1 = 15;
    qs[j] = (uint8_t)(x0 | (x1 << 4));
  }
  return dh;
}

// Canonical ggml q8_0 quant of one 32-element block: d = amax/127 (fp16-rounded),
// q = round(x/d) in [-127,127]. Returns fp16 d; fills 32 int8 quants.
static uint16_t ref_quant_q8_0_block(const float *x, int8_t qs[32]) {
  float amax = 0.0f;
  for (int j = 0; j < 32; ++j) {
    float a = fabsf(x[j]);
    if (a > amax) amax = a;
  }
  float d = amax / 127.0f;
  uint16_t dh = f32_to_fp16(d);
  float dr = fp16_to_f32(dh);
  float id = dr ? 1.0f / dr : 0.0f;
  for (int j = 0; j < 32; ++j) {
    int q = (int)roundf(x[j] * id);
    if (q < -127) q = -127; if (q > 127) q = 127;
    qs[j] = (int8_t)q;
  }
  return dh;
}

static int test_shape(long M, long N, long K, unsigned seed) {
  const long mt = M / 4, nt = N / 4, nb = K / 32, kt = K / 8;
  const long q40_block_bytes = 18;

  srand(seed);
  // Original f32 activation X[M][K] and weight W[N][K] (W is the transposed
  // ggml weight matrix: output column n is a row of K weights).
  float *X = (float *)malloc(sizeof(float) * M * K);
  float *W = (float *)malloc(sizeof(float) * N * K);
  for (long i = 0; i < M * K; ++i) X[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  for (long i = 0; i < N * K; ++i) W[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  // --- Canonical quantization (the ZERO-MODEL reference material) ------------
  // Activation: per (m, block) d_a + int8 quants qa[m][k].
  float *dA = (float *)malloc(sizeof(float) * M * nb);
  int8_t *qa = (int8_t *)malloc((size_t)M * K);
  for (long m = 0; m < M; ++m)
    for (long b = 0; b < nb; ++b) {
      int8_t qs[32];
      uint16_t dh = ref_quant_q8_0_block(X + m * K + b * 32, qs);
      dA[m * nb + b] = fp16_to_f32(dh);
      for (int j = 0; j < 32; ++j) qa[m * K + b * 32 + j] = qs[j];
    }
  // Weight: per (n, block) d_w + nibble bytes; also keep dequant nibble int qw.
  float *dW = (float *)malloc(sizeof(float) * N * nb);
  uint8_t *wq = (uint8_t *)malloc((size_t)N * nb * 16); // 16 nibble bytes / block
  int8_t *qw = (int8_t *)malloc((size_t)N * K);
  for (long n = 0; n < N; ++n)
    for (long b = 0; b < nb; ++b) {
      uint8_t qs[16];
      uint16_t dh = ref_quant_q4_0_block(W + n * K + b * 32, qs);
      dW[n * nb + b] = fp16_to_f32(dh);
      for (int j = 0; j < 16; ++j) wq[(n * nb + b) * 16 + j] = qs[j];
      for (int j = 0; j < 16; ++j) {
        qw[n * K + b * 32 + j] = (int8_t)((int)(qs[j] & 0x0F) - 8);
        qw[n * K + b * 32 + j + 16] = (int8_t)((int)(qs[j] >> 4) - 8);
      }
    }

  // --- Pack into the fragment-major kernel layouts (bridge #3 + #4) ----------
  // Apack: (M/4) row-tiles, kt fragments of 32B; A[m][k] at [mi*4*K+kf*32+ml*8+kl].
  long apack_bytes = mt * kt * 32;
  int8_t *Apack = (int8_t *)malloc(apack_bytes);
  for (long m = 0; m < M; ++m)
    for (long k = 0; k < K; ++k) {
      long mi = m / 4, ml = m % 4, kf = k / 8, kl = k % 8;
      Apack[mi * 4 * K + kf * 32 + ml * 8 + kl] = qa[m * K + k];
    }
  // Bnib: (N/4) col-tiles, kt fragment-major 18B blocks; frame idx (nl*8+kl) holds
  // W[n][k] nibble (low if <16 else high of [-16]). d slot filled with fp16 d_w of
  // the OWNING native block for provenance (decode ignores it; dW carries it).
  long bnib_bytes = nt * kt * q40_block_bytes;
  uint8_t *Bnib = (uint8_t *)calloc(bnib_bytes, 1);
  for (long n = 0; n < N; ++n)
    for (long k = 0; k < K; ++k) {
      long nj = n / 4, nl = n % 4, kf = k / 8, kl = k % 8, b = k / 32;
      uint8_t *blk = Bnib + (nj * kt + kf) * q40_block_bytes;
      // stamp the fp16 d_w into the 2-byte slot (provenance only)
      uint16_t dh = f32_to_fp16(dW[n * nb + b]);
      blk[0] = (uint8_t)(dh & 0xFF); blk[1] = (uint8_t)(dh >> 8);
      uint8_t *qs = blk + 2;
      long idx = nl * 8 + kl;
      // reconstruct the packed nibble byte for this (n,k)
      long nb_j = (b * 16); // start nibble-byte index within column's block region
      // native nibble layout: qs_native[j] low = weight[b*32 + j], high = weight[b*32 + j + 16]
      uint8_t native = wq[(n * nb + b) * 16 + (k % 16)];
      int is_high = (k % 32) >= 16;
      int val4 = is_high ? (native >> 4) : (native & 0x0F);
      (void)nb_j;
      if (idx < 16) qs[idx] = (uint8_t)((qs[idx] & 0xF0) | (val4 & 0x0F));
      else          qs[idx - 16] = (uint8_t)((qs[idx - 16] & 0x0F) | ((val4 & 0x0F) << 4));
    }

  // --- Check (1): int32 core partials vs ZERO-MODEL (int32-EXACT) ------------
  long p_mism = 0, p_total = 0;
  for (long mi = 0; mi < mt; ++mi)
    for (long nj = 0; nj < nt; ++nj)
      for (long b = 0; b < nb; ++b) {
        const int8_t *Arow = Apack + mi * 4 * K;
        const uint8_t *Bcol = Bnib + nj * kt * q40_block_bytes;
        int8_t Bdec[128];
        for (long f = 0; f < 4; ++f)
          tcrv_ime_q4_0_dequant_fragment(Bcol + (b * 4 + f) * q40_block_bytes,
                                         Bdec + f * 32);
        int32_t frag[16];
        tcrv_ime_vmadot_mac_kloop(Arow + b * 4 * 32, Bdec, 4, frag);
        for (long r = 0; r < 4; ++r)
          for (long c = 0; c < 4; ++c) {
            long m = mi * 4 + r, n = nj * 4 + c;
            int32_t ref = 0;
            for (long j = 0; j < 32; ++j)
              ref += (int32_t)qa[m * K + b * 32 + j] * (int32_t)qw[n * K + b * 32 + j];
            p_total++;
            if (frag[r * 4 + c] != ref) p_mism++;
          }
      }

  // --- Check (2): scale-fold f32 vs canonical reference (bounded ULP) ---------
  float *Cf = (float *)calloc((size_t)M * N, sizeof(float));
  tcrv_ime_q4_0_vmadot_matmul_f32(Apack, dA, Bnib, dW, Cf, M, N, K);

  double max_abs_err = 0.0, max_rel_err = 0.0, ref_max = 0.0;
  for (long m = 0; m < M; ++m)
    for (long n = 0; n < N; ++n) {
      double ref = 0.0;
      for (long b = 0; b < nb; ++b) {
        int32_t part = 0;
        for (long j = 0; j < 32; ++j)
          part += (int32_t)qa[m * K + b * 32 + j] * (int32_t)qw[n * K + b * 32 + j];
        ref += (double)dA[m * nb + b] * (double)dW[n * nb + b] * (double)part;
      }
      double got = (double)Cf[m * N + n];
      double ae = fabs(got - ref);
      double re = fabs(ref) > 1e-6 ? ae / fabs(ref) : ae;
      if (ae > max_abs_err) max_abs_err = ae;
      if (re > max_rel_err) max_rel_err = re;
      if (fabs(ref) > ref_max) ref_max = fabs(ref);
    }

  printf("  shape M=%ld N=%ld K=%ld nb=%ld: int32-core %ld/%ld exact (mism=%ld) | "
         "scalefold max_abs=%.3e max_rel=%.3e (|C|max=%.3f)\n",
         M, N, K, nb, p_total - p_mism, p_total, p_mism, max_abs_err, max_rel_err,
         ref_max);

  int ok = (p_mism == 0) && (max_rel_err < 1e-4);
  free(X); free(W); free(dA); free(qa); free(dW); free(wq); free(qw);
  free(Apack); free(Bnib); free(Cf);
  return ok;
}

int main(void) {
  printf("G5-M3 q4_0@ime scale-fold epilogue oracle (bridge #1 fold + #2 runtime shape)\n");
  int ok = 1;
  ok &= test_shape(8, 8, 64, 1234567u);
  ok &= test_shape(4, 8, 96, 2244668u);
  ok &= test_shape(12, 16, 128, 9988776u);
  ok &= test_shape(8, 32, 256, 5551234u);
  if (ok) {
    printf("ORACLE PASS: q4_0 scale-fold f32 == canonical q4_0xq8_0 reference "
           "(int32 core exact, fold within f32 tol) across all shapes\n");
    return 0;
  }
  printf("ORACLE FAIL\n");
  return 1;
}
