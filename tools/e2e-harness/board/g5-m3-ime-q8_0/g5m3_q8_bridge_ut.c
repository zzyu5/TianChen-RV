// G5-M3 IME q8_0 forward-bridge INTEGRATION UT (k1 · IME harts 0-3).
//
// The FLAT int8-DIRECT sibling of g5m3_bridge_ut.c (q4_0). q8_0 weight is stored
// as native int8 (no nibble unpack), so bridge #4 is a straight int8 gather into
// the fragment-major layout; bridge #3 (f32 activation -> q8_0) is byte-identical
// to the q4_0 UT; the scale-fold GEMM is the same flat epilogue but the B fragment
// is int8-DIRECT (no dequant_fragment step). Validated against the REAL ggml
// quantizer/dequantizer (linked from libggml-base.so), not a re-implementation.
//
//   bridge #4 (weight repack): ggml NATIVE q8_0 (row-major N x K/32 34B blocks, as
//     produced by ggml quantize_row_q8_0_ref) -> tcrv fragment-major Bpack (int8
//     DIRECT) + dW. MIRAGE de-risk: reconstruct(Bpack,dW) MUST equal ggml
//     dequantize_row_q8_0 of the native bytes, element-exact (0 ULP).
//
//   bridge #3 (activation quant+pack): ggml f32 activation -> q8_0 (ggml
//     quantize_row_q8_0_ref) -> tcrv fragment-major Apack + dA. MIRAGE de-risk:
//     reconstruct(Apack,dA) MUST equal the native q8_0 int8/scale, byte-exact.
//
//   FULL A==B (correctness gate, the single-tensor mul_mat): A = our full bridge
//     (native q8_0 weight + f32 act -> #4 -> #3 -> scale-fold GEMM with the REAL
//     vmadot asm leaf on silicon). B = the ZERO-MODEL block-dot reference == exactly
//     the ggml q8_0 x q8_0 dot the stock RVV kernel computes (re-derived from the
//     same native bytes in double, a different code path). A==B bounded-ULP (f32
//     reassociation only). Real vmadot -> silicon seal of the integrated path.
//
// Build (SpacemiT/board toolchain that assembles `vmadot`, link ggml-base):
//   gcc -O2 -std=c11 -march=rv64gcv_xsmtvdotii1p0 -mabi=lp64d g5m3_q8_bridge_ut.c \
//       -L <build-off/bin> -lggml-base -lm -Wl,-rpath,<build-off/bin> -o ut
//   taskset -c 0-3 ./ut
// objdump golden: the MAC leaf disassembles to `vmadot v2,v0,v1` = 0xe210312b.
// Exit 0 + "Q8-BRIDGE-UT PASS" iff #4/#3 are exact and full A==B within f32 tol.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- REAL ggml symbols (libggml-base.so; ggml is C -> plain extern) ----------
extern void  quantize_row_q8_0_ref(const float *x, void *y, int64_t k);
extern void  dequantize_row_q8_0(const void *x, float *y, int64_t k);
extern float ggml_fp16_to_fp32(uint16_t h);

// native block byte size (ggml-common.h: block_q8_0 = 34 = 2B fp16 d + 32 int8)
#define Q80_NATIVE_BYTES 34
#define QK 32

// ---------------------------------------------------------------------------
// (A) Emitter-verbatim kernel leaves (identical to the sealed scale-fold seal;
//     the vmadot MAC leaf is byte-identical to q4_0 -- shared VmadotMacLeafOp).
// ---------------------------------------------------------------------------

// EMITTER-VERBATIM batched register-resident `vmadot` MAC leaf (0xe210312b).
static inline void tcrv_ime_vmadot_mac_kloop(const int8_t *A, const int8_t *B,
                                             long kt, int32_t *frag) {
  __asm__ volatile(
      "vsetvli   t0, zero, e8, m1, ta, ma   \n\t"
      "vmv.v.i   v2, 0                       \n\t"
      "vmv.v.i   v3, 0                       \n\t"
      "mv        t2, %[kt]                   \n\t"
      "mv        t3, %[pa]                   \n\t"
      "mv        t4, %[pb]                   \n\t"
      "1:                                    \n\t"
      "vle8.v    v0, (t3)                    \n\t"
      "vle8.v    v1, (t4)                    \n\t"
      "vmadot    v2, v0, v1                 \n\t"
      "addi      t3, t3, 32                  \n\t"
      "addi      t4, t4, 32                  \n\t"
      "addi      t2, t2, -1                  \n\t"
      "bnez      t2, 1b                      \n\t"
      "vsetvli   t0, zero, e32, m1, ta, ma   \n\t"
      "vse32.v   v2, (%[pf])                 \n\t"
      "addi      t5, %[pf], 32               \n\t"
      "vse32.v   v3, (t5)                    \n\t"
      :
      : [pa] "r"(A), [pb] "r"(B), [kt] "r"(kt), [pf] "r"(frag)
      : "t0", "t2", "t3", "t4", "t5", "v0", "v1", "v2", "v3", "memory");
}

// q8_0 SCALE-FOLD f32 GEMM (int8-DIRECT B; the flat sibling of the q4_0 GEMM, but
// the B fragment is read straight from int8 -- NO dequant_fragment step).
static void tcrv_ime_q8_0_vmadot_matmul_f32(const int8_t *Apack, const float *dA,
                                            const int8_t *Bpack, const float *dW,
                                            float *Cf, long M, long N, long K) {
  const long mt = M / 4, nt = N / 4, nb = K / 32;
  const long frags_per_block = 4; // 32 k / 8 k-per-fragment
  const long kt = K / 8;
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Apack + (long)mi * 4 * K;
    for (long nj = 0; nj < nt; ++nj) {
      const int8_t *Bcol = Bpack + (long)nj * kt * 32; // int8-DIRECT fragments
      for (long b = 0; b < nb; ++b) {
        int32_t frag[16];
        tcrv_ime_vmadot_mac_kloop(Arow + b * frags_per_block * 32,
                                  Bcol + b * frags_per_block * 32,
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
// (B) INTEGRATION BRIDGES: repack the REAL ggml native formats into the
// fragment-major kernel layouts.
// ---------------------------------------------------------------------------

// bridge #4: ggml NATIVE q8_0 weight (row-major, N rows x nb 34B blocks) ->
// fragment-major Bpack (col-tiles of kt 32B int8 fragments, INT8 DIRECT) +
// per-(col,block) dW. No nibble unpack -- the int8 qs are copied verbatim.
static void tcrv_bridge4_repack_q8_0(const uint8_t *wq_native, int8_t *Bpack,
                                     float *dW, long N, long K) {
  const long nb = K / 32, kt = K / 8;
  memset(Bpack, 0, (size_t)(N / 4) * kt * 32);
  for (long n = 0; n < N; ++n)
    for (long b = 0; b < nb; ++b) {
      const uint8_t *src = wq_native + (n * nb + b) * Q80_NATIVE_BYTES;
      uint16_t dh = (uint16_t)(src[0] | (src[1] << 8));
      dW[n * nb + b] = ggml_fp16_to_fp32(dh);
      const int8_t *qs = (const int8_t *)(src + 2); // 32 int8, DIRECT
      for (long kk = 0; kk < 32; ++kk) {
        long k = b * 32 + kk;
        long nj = n / 4, nl = n % 4, kf = k / 8, kl = k % 8;
        Bpack[(nj * kt + kf) * 32 + nl * 8 + kl] = qs[kk];
      }
    }
}

// bridge #3: ggml f32 activation (M rows x K) -> q8_0 (ggml quantize_row_q8_0_ref)
// -> fragment-major Apack (row-tiles of kt 32B fragments) + per-(row,block) dA.
// Byte-identical to the q4_0 UT bridge #3 (shared q8_0 activation path).
static void tcrv_bridge3_quant_pack_q8_0(const float *X, int8_t *Apack, float *dA,
                                         long M, long K, uint8_t *scratch) {
  const long nb = K / 32;
  for (long m = 0; m < M; ++m) {
    quantize_row_q8_0_ref(X + m * K, scratch, K); // REAL ggml q8_0 quant
    for (long b = 0; b < nb; ++b) {
      const uint8_t *blk = scratch + b * Q80_NATIVE_BYTES;
      uint16_t dh = (uint16_t)(blk[0] | (blk[1] << 8));
      dA[m * nb + b] = ggml_fp16_to_fp32(dh);
      const int8_t *qs = (const int8_t *)(blk + 2);
      for (long kk = 0; kk < 32; ++kk) {
        long k = b * 32 + kk;
        long mi = m / 4, ml = m % 4, kf = k / 8, kl = k % 8;
        Apack[mi * 4 * K + kf * 32 + ml * 8 + kl] = qs[kk];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

static int g_fail = 0;

static void frand_fill(float *p, long n, unsigned seed) {
  srand(seed);
  for (long i = 0; i < n; ++i) p[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

// --- bridge #4 UT: Bpack+dW reconstruct == ggml native dequant (0 ULP) -------
static int ut_bridge4(long N, long K, unsigned seed) {
  const long nb = K / 32, kt = K / 8;
  float *W = malloc(sizeof(float) * N * K);
  frand_fill(W, N * K, seed);
  uint8_t *wq = malloc((size_t)N * nb * Q80_NATIVE_BYTES);
  for (long n = 0; n < N; ++n)
    quantize_row_q8_0_ref(W + n * K, wq + (size_t)n * nb * Q80_NATIVE_BYTES, K);

  int8_t *Bpack = malloc((size_t)(N / 4) * kt * 32);
  float *dW = malloc(sizeof(float) * N * nb);
  tcrv_bridge4_repack_q8_0(wq, Bpack, dW, N, K);

  // ggml native dequant reference
  float *wdeq = malloc(sizeof(float) * N * K);
  for (long n = 0; n < N; ++n)
    dequantize_row_q8_0(wq + (size_t)n * nb * Q80_NATIVE_BYTES, wdeq + n * K, K);

  long mism = 0;
  double maxabs = 0.0;
  for (long n = 0; n < N; ++n)
    for (long k = 0; k < K; ++k) {
      long nj = n / 4, nl = n % 4, kf = k / 8, kl = k % 8, b = k / 32;
      int q = Bpack[(nj * kt + kf) * 32 + nl * 8 + kl]; // int8 direct
      float got = (float)q * dW[n * nb + b];
      float ref = wdeq[n * K + k];
      double ae = fabs((double)got - (double)ref);
      if (got != ref) mism++;
      if (ae > maxabs) maxabs = ae;
    }
  printf("  #4 weight-repack N=%ld K=%ld: dequant %ld/%ld exact vs ggml native "
         "(mism=%ld max_abs=%.3e)\n",
         N, K, N * K - mism, N * K, mism, maxabs);
  if (mism) g_fail = 1;
  free(W); free(wq); free(Bpack); free(dW); free(wdeq);
  return mism == 0;
}

// --- bridge #3 UT: Apack+dA reconstruct == native q8_0 (byte-exact) ---------
static int ut_bridge3(long M, long K, unsigned seed) {
  const long nb = K / 32, kt = K / 8;
  float *X = malloc(sizeof(float) * M * K);
  frand_fill(X, M * K, seed);

  int8_t *Apack = malloc((size_t)(M / 4) * kt * 32);
  float *dA = malloc(sizeof(float) * M * nb);
  uint8_t *scratch = malloc((size_t)nb * Q80_NATIVE_BYTES);
  tcrv_bridge3_quant_pack_q8_0(X, Apack, dA, M, K, scratch);

  // native reference: re-quantize per row, compare int8 + scale byte-exact
  long q_mism = 0, d_mism = 0;
  uint8_t *nat = malloc((size_t)nb * Q80_NATIVE_BYTES);
  for (long m = 0; m < M; ++m) {
    quantize_row_q8_0_ref(X + m * K, nat, K);
    for (long b = 0; b < nb; ++b) {
      const uint8_t *blk = nat + b * Q80_NATIVE_BYTES;
      uint16_t dh = (uint16_t)(blk[0] | (blk[1] << 8));
      if (dA[m * nb + b] != ggml_fp16_to_fp32(dh)) d_mism++;
      const int8_t *qs = (const int8_t *)(blk + 2);
      for (long kk = 0; kk < 32; ++kk) {
        long k = b * 32 + kk, mi = m / 4, ml = m % 4, kf = k / 8, kl = k % 8;
        if (Apack[mi * 4 * K + kf * 32 + ml * 8 + kl] != qs[kk]) q_mism++;
      }
    }
  }
  printf("  #3 act-quant+pack M=%ld K=%ld: int8 %ld/%ld exact (mism=%ld) | scale "
         "%ld/%ld exact (mism=%ld)\n",
         M, K, M * K - q_mism, M * K, q_mism, M * nb - d_mism, M * nb, d_mism);
  if (q_mism || d_mism) g_fail = 1;
  free(X); free(Apack); free(dA); free(scratch); free(nat);
  return (q_mism == 0) && (d_mism == 0);
}

// --- FULL A==B: our bridge (real vmadot) vs ZERO-MODEL block-dot stock math --
static int ut_full(long M, long N, long K, unsigned seed) {
  const long nb = K / 32, kt = K / 8;
  float *W = malloc(sizeof(float) * N * K);
  float *X = malloc(sizeof(float) * M * K);
  frand_fill(W, N * K, seed);
  frand_fill(X, M * K, seed ^ 0x5a5a5a5au);

  // REAL ggml native quantization (the on-disk / forward-input format)
  uint8_t *wq = malloc((size_t)N * nb * Q80_NATIVE_BYTES);
  for (long n = 0; n < N; ++n)
    quantize_row_q8_0_ref(W + n * K, wq + (size_t)n * nb * Q80_NATIVE_BYTES, K);
  uint8_t *aq = malloc((size_t)M * nb * Q80_NATIVE_BYTES);
  for (long m = 0; m < M; ++m)
    quantize_row_q8_0_ref(X + m * K, aq + (size_t)m * nb * Q80_NATIVE_BYTES, K);

  // A: our full integrated bridge (#4 + #3 + scale-fold GEMM, real vmadot)
  int8_t *Bpack = malloc((size_t)(N / 4) * kt * 32);
  float *dW = malloc(sizeof(float) * N * nb);
  tcrv_bridge4_repack_q8_0(wq, Bpack, dW, N, K);
  int8_t *Apack = malloc((size_t)(M / 4) * kt * 32);
  float *dA = malloc(sizeof(float) * M * nb);
  uint8_t *scratch = malloc((size_t)nb * Q80_NATIVE_BYTES);
  tcrv_bridge3_quant_pack_q8_0(X, Apack, dA, M, K, scratch);
  float *Cf = calloc((size_t)M * N, sizeof(float));
  tcrv_ime_q8_0_vmadot_matmul_f32(Apack, dA, Bpack, dW, Cf, M, N, K);

  // B: ZERO-MODEL block dot == exactly ggml_vec_dot_q8_0_q8_0 stock math,
  // re-derived from the SAME native bytes, computed TWO ways for a rigorous read:
  //   ref_d = DOUBLE accumulation (ground truth, order-independent).
  //   ref_f = FLOAT accumulation in the SAME per-block fold order as kernel A.
  // Gate = NORMWISE relative error max_abs/||C|| (per-cell rel is meaningless where
  // |C[m,n]|~0 by random sign cancellation; reported but NOT gated on).
  double max_abs = 0.0, max_rel_cell = 0.0, ref_max = 0.0, max_abs_vs_float = 0.0;
  for (long m = 0; m < M; ++m)
    for (long n = 0; n < N; ++n) {
      double ref_d = 0.0;
      float ref_f = 0.0f;
      for (long b = 0; b < nb; ++b) {
        const uint8_t *wblk = wq + (size_t)(n * nb + b) * Q80_NATIVE_BYTES;
        const uint8_t *ablk = aq + (size_t)(m * nb + b) * Q80_NATIVE_BYTES;
        float dw = ggml_fp16_to_fp32((uint16_t)(wblk[0] | (wblk[1] << 8)));
        float da = ggml_fp16_to_fp32((uint16_t)(ablk[0] | (ablk[1] << 8)));
        const int8_t *wqs = (const int8_t *)(wblk + 2);
        const int8_t *aqs = (const int8_t *)(ablk + 2);
        int32_t part = 0;
        for (long j = 0; j < 32; ++j)
          part += (int32_t)aqs[j] * (int32_t)wqs[j]; // int8 x int8 (both direct)
        ref_d += (double)da * (double)dw * (double)part;
        ref_f += da * dw * (float)part; // same fold order as kernel A
      }
      double got = (double)Cf[m * N + n];
      double ae = fabs(got - ref_d);
      double re = fabs(ref_d) > 1e-6 ? ae / fabs(ref_d) : ae;
      double avf = fabs(got - (double)ref_f);
      if (ae > max_abs) max_abs = ae;
      if (re > max_rel_cell) max_rel_cell = re;
      if (fabs(ref_d) > ref_max) ref_max = fabs(ref_d);
      if (avf > max_abs_vs_float) max_abs_vs_float = avf;
    }
  double normwise = max_abs / (ref_max > 1e-9 ? ref_max : 1.0);
  int ok = normwise < 1e-5; // f32 GEMM accuracy floor (reassociation only)
  printf("  A==B  M=%ld N=%ld K=%ld: normwise=%.3e (max_abs=%.3e |C|max=%.3f) "
         "max_abs_vs_float_order=%.3e per_cell_rel=%.3e %s\n",
         M, N, K, normwise, max_abs, ref_max, max_abs_vs_float, max_rel_cell,
         ok ? "OK" : "FAIL");
  if (!ok) g_fail = 1;
  free(W); free(X); free(wq); free(aq); free(Bpack); free(dW);
  free(Apack); free(dA); free(scratch); free(Cf);
  return ok;
}

int main(void) {
  printf("G5-M3 IME q8_0 forward-bridge INTEGRATION UT "
         "(real ggml native formats; real vmadot leaf; int8-DIRECT weight)\n");
  printf("[bridge #4 weight repack: native q8_0 -> fragment-major Bpack(int8)+dW]\n");
  ut_bridge4(8, 64, 111u);
  ut_bridge4(16, 128, 222u);
  ut_bridge4(32, 256, 333u);
  printf("[bridge #3 activation quant+pack: f32 -> q8_0 -> fragment-major Apack+dA]\n");
  ut_bridge3(8, 64, 444u);
  ut_bridge3(12, 128, 555u);
  ut_bridge3(8, 256, 666u);
  printf("[FULL single-tensor mul_mat A==B: our bridge (real vmadot) vs stock "
         "q8_0xq8_0 block-dot]\n");
  ut_full(8, 8, 64, 1234567u);
  ut_full(4, 8, 96, 2244668u);
  ut_full(12, 16, 128, 9988776u);
  ut_full(8, 32, 256, 5551234u);
  ut_full(64, 64, 512, 4242424u);
  if (!g_fail) {
    printf("Q8-BRIDGE-UT PASS: #4 int8-direct repack + #3 quant/pack exact vs ggml "
           "native; full single-tensor mul_mat A==B within f32 tol (real vmadot on K1)\n");
    return 0;
  }
  printf("Q8-BRIDGE-UT FAIL\n");
  return 1;
}
