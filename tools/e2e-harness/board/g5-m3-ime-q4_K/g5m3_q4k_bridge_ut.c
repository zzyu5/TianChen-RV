// G5-M3 IME q4_K forward-bridge INTEGRATION UT (k1 - IME harts 0-3). The
// SUPER-BLOCK K-quant sibling of the q4_0 bridge UT (g5m3_bridge_ut.c). Closes the
// integration layer for q4_K: the two data-path bridges that connect our
// fragment-major SUPER-BLOCK IME kernel (real vmadot 0xe210312b, two-level 6-bit
// scale/min fold) to the REAL ggml native q4_K/q8_K formats, validated against the
// REAL ggml quantizers/dequantizer (linked from libggml-base.so), not a
// re-implementation.
//
//   bridge #4 (weight repack): ggml NATIVE q4_K (row-major, N rows x nsb 144B
//     block_q4_K, as produced by quantize_row_q4_K_ref) -> tcrv (nj-tile, sb, nl)
//     block-gathered Bq4k (native 144B blocks preserved verbatim, 4 per col-tile
//     super-block). MIRAGE de-risk: our decode (raw-nibble + 6-bit sc/m + fp16
//     d/dmin) of the repacked weight MUST equal ggml dequantize_row_q4_K of the
//     native bytes, byte-exact (0 ULP: identical nibble/scale/min/fp16 arithmetic).
//
//   bridge #3 (activation quant+pack): ggml f32 activation -> q8_K (ggml
//     quantize_row_q8_K_ref, the format q4_K is paired with) -> tcrv fragment-major
//     Apack (raw int8, no scale) + per-(row,super-block) dA = q8_K.d. MIRAGE
//     de-risk: reconstruct(Apack) MUST equal the native q8_K int8 qs byte-exact and
//     dA MUST equal the native q8_K.d float-exact.
//
//   FULL A==B (correctness gate, single-tensor mul_mat): A = our full bridge
//     (native q4_K + f32 act -> #4 -> #3 -> two-level scale/min-fold GEMM with the
//     REAL vmadot asm leaf on silicon, activation-scale y.d folded per super-block).
//     B = the ZERO-MODEL reference == exactly the ggml q4_K x q8_K dot the stock RVV
//     kernel computes (re-derived from the SAME native bytes in double, a different
//     code path), AND cross-checked against the REAL ggml_vec_dot_q4_K_q8_K from
//     libggml-cpu.so. A==B bounded-ULP (f32 reassociation only). Real vmadot ->
//     silicon seal of the integrated super-block path.
//
// Build (SpacemiT/board toolchain that assembles `vmadot`, link ggml-base + cpu):
//   gcc -O2 -std=c11 -march=rv64gcv_xsmtvdotii1p0 -mabi=lp64d g5m3_q4k_bridge_ut.c \
//       -L <build-off/bin> -lggml-base -lggml-cpu -lm -Wl,-rpath,<build-off/bin> -o ut
//   taskset -c 0-3 ./ut
// objdump golden: the MAC leaf disassembles to `vmadot v2,v0,v1` = 0xe210312b.
// Exit 0 + "Q4K-BRIDGE-UT PASS" iff #4/#3 are exact and full A==B within f32 tol.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- REAL ggml symbols (libggml-base.so / libggml-cpu.so; ggml is C) ---------
extern void  quantize_row_q4_K_ref(const float *x, void *y, int64_t k);
extern void  quantize_row_q8_K_ref(const float *x, void *y, int64_t k);
extern void  dequantize_row_q4_K(const void *x, float *y, int64_t k);
extern float ggml_fp16_to_fp32(uint16_t h);
// real stock RVV kernel dot (independent oracle, libggml-cpu.so):
extern void  ggml_vec_dot_q4_K_q8_K(int n, float *s, size_t bs, const void *vx,
                                     size_t bx, const void *vy, size_t by, int nrc);

// native block byte sizes (ggml-common.h)
#define Q4K_NATIVE_BYTES 144 // fp16 d + fp16 dmin + 12B 6-bit scales + 128B nibbles
#define Q8K_NATIVE_BYTES 292 // float d + 256 int8 qs + 16 int16 bsums
#define QK_K 256

// ---------------------------------------------------------------------------
// (A) Emitter-verbatim kernel leaves (byte-identical to the M2b K1 seal:
//     q4-K-matmul-tile-int32-k1seal.c). The asm leaf is the EMITTER-VERBATIM
//     batched register-resident `vmadot` MAC (0xe210312b), the one justified
//     instruction leaf; all surrounding dataflow is the structured C the IME
//     emitter emits for the q4_K super-block region.
// ---------------------------------------------------------------------------

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

static inline unsigned short tcrv_ime_load_fp16(const uint8_t *p) {
  return (unsigned short)((unsigned)p[0] | ((unsigned)p[1] << 8));
}

// q4_K RAW-nibble DECODE (emitter-verbatim). q4_K nibbles are UNSIGNED [0,15]
// (NO offset-binary centering); the min bias is applied downstream.
static inline void tcrv_ime_q4_K_dequant_fragment(const uint8_t *blk, int b,
                                                  int kf, int8_t *out8) {
  const uint8_t *qs = blk + 16; // past fp16 d/dmin + 12-byte 6-bit scales
  for (int kl = 0; kl < 8; ++kl) {
    int pl = kf * 8 + kl;
    uint8_t byte = qs[(b / 2) * 32 + pl];
    out8[kl] = (int8_t)((b & 1) ? (byte >> 4) : (byte & 0x0F));
  }
}

// q4_K 6-bit scale/min UNPACK (canonical ggml get_scale_min_k4; emitter-verbatim).
static inline void tcrv_ime_q4_K_get_scale_min(int j, const uint8_t *q,
                                               uint8_t *sc, uint8_t *m) {
  if (j < 4) {
    *sc = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// ---------------------------------------------------------------------------
// (B) SESSION-2 INTEGRATION BRIDGES (the new net code): repack the REAL ggml
//     native q4_K/q8_K into the fragment-major super-block kernel layouts.
// ---------------------------------------------------------------------------

// bridge #4: ggml NATIVE q4_K weight (row-major, N rows x nsb 144B blocks) ->
// (nj-tile, sb, nl) block-gathered Bq4k. 4 native 144B blocks per (col-tile,
// super-block); the block bytes are preserved VERBATIM (no nibble reshuffle --
// the emitter decode reads native q4_K layout directly).
static void tcrv_bridge4_repack_q4_K(const uint8_t *wq_native, uint8_t *Bq4k,
                                     long N, long K) {
  const long nsb = K / 256, q4kb = Q4K_NATIVE_BYTES;
  for (long n = 0; n < N; ++n)
    for (long sb = 0; sb < nsb; ++sb) {
      const uint8_t *src = wq_native + (size_t)(n * nsb + sb) * q4kb;
      long nj = n / 4, nl = n % 4;
      uint8_t *dst = Bq4k + (size_t)(((nj * nsb + sb) * 4) + nl) * q4kb;
      memcpy(dst, src, q4kb);
    }
}

// bridge #3: ggml f32 activation (M rows x K) -> q8_K (ggml quantize_row_q8_K_ref)
// -> fragment-major Apack (row-tiles of K/8 32B fragments, raw int8) + per-(row,
// super-block) dA = q8_K.d.
static void tcrv_bridge3_quant_pack_q8_K(const float *X, int8_t *Apack, float *dA,
                                         long M, long K, uint8_t *scratch) {
  const long nsb = K / 256;
  for (long m = 0; m < M; ++m) {
    quantize_row_q8_K_ref(X + m * K, scratch, K); // REAL ggml q8_K quant
    for (long sb = 0; sb < nsb; ++sb) {
      const uint8_t *blk = scratch + (size_t)sb * Q8K_NATIVE_BYTES;
      float d;
      memcpy(&d, blk, sizeof(float));
      dA[m * nsb + sb] = d;
      const int8_t *qs = (const int8_t *)(blk + 4);
      for (long kk = 0; kk < 256; ++kk) {
        long k = sb * 256 + kk;
        long mi = m / 4, ml = m % 4, kf = k / 8, kl = k % 8;
        Apack[mi * 4 * K + kf * 32 + ml * 8 + kl] = qs[kk];
      }
    }
  }
}

// q4_K two-level scale/min-fold f32 GEMM (bridge #1 + #2 + activation-scale fold).
// int32 core (S_scale = Sum_b sc_b*sumi_b, S_min = Sum_b m_b*asum_b) is emitter-
// verbatim (real vmadot); the epilogue folds the per-(row,super-block) activation
// scale dA: C[m,n] += y.d * (d*S_scale - dmin*S_min), matching ggml
// vec_dot_q4_K_q8_K where d=x.d*y.d, dmin=x.dmin*y.d.
static void tcrv_ime_q4_K_vmadot_matmul_f32(const int8_t *Apack, const float *dA,
                                            const uint8_t *Bq4k, float *Cf, long M,
                                            long N, long K) {
  const long mt = M / 4, nt = N / 4, nsb = K / 256, q4kb = Q4K_NATIVE_BYTES;
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Apack + (long)mi * 4 * K;
    for (long nj = 0; nj < nt; ++nj) {
      for (long sb = 0; sb < nsb; ++sb) {
        const uint8_t *blk[4];
        uint8_t sc[8][4], mm[8][4];
        for (int nl = 0; nl < 4; ++nl) {
          blk[nl] = Bq4k + (size_t)((((nj * nsb) + sb) * 4) + nl) * q4kb;
          for (int b = 0; b < 8; ++b)
            tcrv_ime_q4_K_get_scale_min(b, blk[nl] + 4, &sc[b][nl], &mm[b][nl]);
        }
        int32_t Sc[16], Sm[16];
        for (int r = 0; r < 16; ++r) { Sc[r] = 0; Sm[r] = 0; }
        for (int b = 0; b < 8; ++b) {
          int32_t sumi[16];
          int32_t asum[4] = {0, 0, 0, 0};
          int8_t Bdec[128];
          for (int kf = 0; kf < 4; ++kf) {
            long gf = sb * 32 + b * 4 + kf;
            const int8_t *Aframe = Arow + gf * 32;
            for (int nl = 0; nl < 4; ++nl)
              tcrv_ime_q4_K_dequant_fragment(blk[nl], b, kf, Bdec + kf * 32 + nl * 8);
            for (int ml = 0; ml < 4; ++ml)
              for (int kl = 0; kl < 8; ++kl)
                asum[ml] += (int32_t)Aframe[ml * 8 + kl];
          }
          const int8_t *Ablk = Arow + (long)(sb * 32 + b * 4) * 32;
          tcrv_ime_vmadot_mac_kloop(Ablk, Bdec, 4, sumi);
          for (int ml = 0; ml < 4; ++ml)
            for (int nl = 0; nl < 4; ++nl) {
              Sc[ml * 4 + nl] += (int32_t)sc[b][nl] * sumi[ml * 4 + nl];
              Sm[ml * 4 + nl] += (int32_t)mm[b][nl] * asum[ml];
            }
        }
        for (int ml = 0; ml < 4; ++ml)
          for (int nl = 0; nl < 4; ++nl) {
            long mo = mi * 4 + ml, no = nj * 4 + nl;
            float dw = ggml_fp16_to_fp32(tcrv_ime_load_fp16(blk[nl] + 0));
            float dminw = ggml_fp16_to_fp32(tcrv_ime_load_fp16(blk[nl] + 2));
            float ad = dA[mo * nsb + sb];
            Cf[mo * N + no] += (dw * ad) * (float)Sc[ml * 4 + nl] -
                               (dminw * ad) * (float)Sm[ml * 4 + nl];
          }
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

// raw-nibble accessor for the ZERO-MODEL reference (native q4_K block)
static int ref_nibble(const uint8_t *blk, long p) {
  long b = p / 32, pl = p % 32;
  const uint8_t *qs = blk + 16;
  uint8_t byte = qs[(b / 2) * 32 + pl];
  return (b & 1) ? (byte >> 4) : (byte & 0x0F);
}

// --- bridge #4 UT: decode(repacked Bq4k) == ggml dequantize_row_q4_K (0 ULP) -
static int ut_bridge4(long N, long K, unsigned seed) {
  const long nsb = K / 256;
  float *W = malloc(sizeof(float) * N * K);
  frand_fill(W, N * K, seed);
  uint8_t *wq = malloc((size_t)N * nsb * Q4K_NATIVE_BYTES);
  for (long n = 0; n < N; ++n)
    quantize_row_q4_K_ref(W + n * K, wq + (size_t)n * nsb * Q4K_NATIVE_BYTES, K);

  uint8_t *Bq4k = malloc((size_t)N * nsb * Q4K_NATIVE_BYTES);
  tcrv_bridge4_repack_q4_K(wq, Bq4k, N, K);

  // ggml native dequant reference
  float *wdeq = malloc(sizeof(float) * N * K);
  for (long n = 0; n < N; ++n)
    dequantize_row_q4_K(wq + (size_t)n * nsb * Q4K_NATIVE_BYTES, wdeq + n * K, K);

  long mism = 0;
  double maxabs = 0.0;
  for (long n = 0; n < N; ++n) {
    long nj = n / 4, nl = n % 4;
    for (long k = 0; k < K; ++k) {
      long sb = k / 256, p = k % 256, b = p / 32;
      const uint8_t *blk = Bq4k + (size_t)(((nj * nsb + sb) * 4) + nl) * Q4K_NATIVE_BYTES;
      uint8_t sc, m;
      tcrv_ime_q4_K_get_scale_min(b, blk + 4, &sc, &m);
      float dw = ggml_fp16_to_fp32(tcrv_ime_load_fp16(blk + 0));
      float dminw = ggml_fp16_to_fp32(tcrv_ime_load_fp16(blk + 2));
      // reconstruct with the SAME float ops ggml dequant uses: d1=dw*sc; d1*nib - dmin*m
      float got = (dw * (float)sc) * (float)ref_nibble(blk, p) - (dminw * (float)m);
      float ref = wdeq[n * K + k];
      double ae = fabs((double)got - (double)ref);
      if (got != ref) mism++;
      if (ae > maxabs) maxabs = ae;
    }
  }
  printf("  #4 weight-repack N=%ld K=%ld: dequant %ld/%ld exact vs ggml native "
         "(mism=%ld max_abs=%.3e)\n",
         N, K, N * K - mism, N * K, mism, maxabs);
  if (mism) g_fail = 1;
  free(W); free(wq); free(Bq4k); free(wdeq);
  return mism == 0;
}

// --- bridge #3 UT: Apack+dA reconstruct == native q8_K (byte/float exact) ----
static int ut_bridge3(long M, long K, unsigned seed) {
  const long nsb = K / 256;
  float *X = malloc(sizeof(float) * M * K);
  frand_fill(X, M * K, seed);

  int8_t *Apack = malloc((size_t)(M / 4) * (K / 8) * 32);
  float *dA = malloc(sizeof(float) * M * nsb);
  uint8_t *scratch = malloc((size_t)nsb * Q8K_NATIVE_BYTES);
  tcrv_bridge3_quant_pack_q8_K(X, Apack, dA, M, K, scratch);

  long q_mism = 0, d_mism = 0;
  uint8_t *nat = malloc((size_t)nsb * Q8K_NATIVE_BYTES);
  for (long m = 0; m < M; ++m) {
    quantize_row_q8_K_ref(X + m * K, nat, K);
    for (long sb = 0; sb < nsb; ++sb) {
      const uint8_t *blk = nat + (size_t)sb * Q8K_NATIVE_BYTES;
      float d;
      memcpy(&d, blk, sizeof(float));
      if (dA[m * nsb + sb] != d) d_mism++;
      const int8_t *qs = (const int8_t *)(blk + 4);
      for (long kk = 0; kk < 256; ++kk) {
        long k = sb * 256 + kk, mi = m / 4, ml = m % 4, kf = k / 8, kl = k % 8;
        if (Apack[mi * 4 * K + kf * 32 + ml * 8 + kl] != qs[kk]) q_mism++;
      }
    }
  }
  printf("  #3 act-quant+pack M=%ld K=%ld: int8 %ld/%ld exact (mism=%ld) | scale "
         "%ld/%ld exact (mism=%ld)\n",
         M, K, M * K - q_mism, M * K, q_mism, M * nsb - d_mism, M * nsb, d_mism);
  if (q_mism || d_mism) g_fail = 1;
  free(X); free(Apack); free(dA); free(scratch); free(nat);
  return (q_mism == 0) && (d_mism == 0);
}

// --- FULL A==B: our bridge (real vmadot) vs ZERO-MODEL q4_K x q8_K + real ggml -
static int ut_full(long M, long N, long K, unsigned seed) {
  const long nsb = K / 256;
  float *W = malloc(sizeof(float) * N * K);
  float *X = malloc(sizeof(float) * M * K);
  frand_fill(W, N * K, seed);
  frand_fill(X, M * K, seed ^ 0x5a5a5a5au);

  // REAL ggml native quantization (the on-disk / forward-input format)
  uint8_t *wq = malloc((size_t)N * nsb * Q4K_NATIVE_BYTES);
  for (long n = 0; n < N; ++n)
    quantize_row_q4_K_ref(W + n * K, wq + (size_t)n * nsb * Q4K_NATIVE_BYTES, K);
  uint8_t *aq = malloc((size_t)M * nsb * Q8K_NATIVE_BYTES);
  for (long m = 0; m < M; ++m)
    quantize_row_q8_K_ref(X + m * K, aq + (size_t)m * nsb * Q8K_NATIVE_BYTES, K);

  // A: our full integrated bridge (#4 + #3 + two-level fold GEMM, real vmadot)
  uint8_t *Bq4k = malloc((size_t)N * nsb * Q4K_NATIVE_BYTES);
  tcrv_bridge4_repack_q4_K(wq, Bq4k, N, K);
  int8_t *Apack = malloc((size_t)(M / 4) * (K / 8) * 32);
  float *dA = malloc(sizeof(float) * M * nsb);
  uint8_t *scratch = malloc((size_t)nsb * Q8K_NATIVE_BYTES);
  tcrv_bridge3_quant_pack_q8_K(X, Apack, dA, M, K, scratch);
  float *Cf = calloc((size_t)M * N, sizeof(float));
  tcrv_ime_q4_K_vmadot_matmul_f32(Apack, dA, Bq4k, Cf, M, N, K);

  // B: ZERO-MODEL q4_K x q8_K, re-derived from the SAME native bytes in double
  // (ground truth), AND the REAL ggml_vec_dot_q4_K_q8_K stock kernel (independent
  // oracle, different code path). Gate = NORMWISE relative error max_abs/||C||.
  double max_abs = 0.0, ref_max = 0.0, max_abs_vs_ggml = 0.0;
  double max_ggml_vs_ref = 0.0;
  for (long m = 0; m < M; ++m)
    for (long n = 0; n < N; ++n) {
      double ref_d = 0.0;
      for (long sb = 0; sb < nsb; ++sb) {
        const uint8_t *wblk = wq + (size_t)(n * nsb + sb) * Q4K_NATIVE_BYTES;
        const uint8_t *ablk = aq + (size_t)(m * nsb + sb) * Q8K_NATIVE_BYTES;
        float x_d = ggml_fp16_to_fp32(tcrv_ime_load_fp16(wblk + 0));
        float x_dmin = ggml_fp16_to_fp32(tcrv_ime_load_fp16(wblk + 2));
        float y_d;
        memcpy(&y_d, ablk, sizeof(float));
        const int8_t *aqs = (const int8_t *)(ablk + 4);
        int64_t Ss = 0, Sm = 0;
        for (int b = 0; b < 8; ++b) {
          uint8_t sc, mn;
          tcrv_ime_q4_K_get_scale_min(b, wblk + 4, &sc, &mn);
          int64_t sumi = 0, asum = 0;
          for (int pl = 0; pl < 32; ++pl) {
            long p = b * 32 + pl;
            int a = (int)aqs[p];
            sumi += (int64_t)a * (int64_t)ref_nibble(wblk, p);
            asum += (int64_t)a;
          }
          Ss += (int64_t)sc * sumi;
          Sm += (int64_t)mn * asum;
        }
        ref_d += (double)y_d *
                 ((double)x_d * (double)(int32_t)Ss - (double)x_dmin * (double)(int32_t)Sm);
      }
      // real ggml stock kernel (one row.row dot; different code path)
      float ggml_s = 0.0f;
      ggml_vec_dot_q4_K_q8_K((int)K, &ggml_s, 0,
                             wq + (size_t)n * nsb * Q4K_NATIVE_BYTES, 0,
                             aq + (size_t)m * nsb * Q8K_NATIVE_BYTES, 0, 1);
      double got = (double)Cf[m * N + n];
      double ae = fabs(got - ref_d);
      double avg = fabs(got - (double)ggml_s);
      double gvr = fabs((double)ggml_s - ref_d);
      if (ae > max_abs) max_abs = ae;
      if (fabs(ref_d) > ref_max) ref_max = fabs(ref_d);
      if (avg > max_abs_vs_ggml) max_abs_vs_ggml = avg;
      if (gvr > max_ggml_vs_ref) max_ggml_vs_ref = gvr;
    }
  double normwise = max_abs / (ref_max > 1e-9 ? ref_max : 1.0);
  double normwise_ggml = max_abs_vs_ggml / (ref_max > 1e-9 ? ref_max : 1.0);
  int ok = normwise < 1e-5 && normwise_ggml < 1e-5;
  printf("  A==B  M=%ld N=%ld K=%ld: normwise=%.3e (max_abs=%.3e |C|max=%.3f) "
         "vs_real_ggml_normwise=%.3e (ggml_vs_zeromodel=%.3e) %s\n",
         M, N, K, normwise, max_abs, ref_max, normwise_ggml, max_ggml_vs_ref,
         ok ? "OK" : "FAIL");
  if (!ok) g_fail = 1;
  free(W); free(X); free(wq); free(aq); free(Bq4k);
  free(Apack); free(dA); free(scratch); free(Cf);
  return ok;
}

int main(void) {
  printf("G5-M3 IME q4_K forward-bridge INTEGRATION UT "
         "(real ggml native q4_K/q8_K; real vmadot super-block leaf)\n");
  printf("[bridge #4 weight repack: native q4_K -> (nj,sb,nl) block-gather]\n");
  ut_bridge4(8, 256, 111u);
  ut_bridge4(16, 512, 222u);
  ut_bridge4(32, 768, 333u);
  printf("[bridge #3 activation quant+pack: f32 -> q8_K -> fragment-major Apack+dA]\n");
  ut_bridge3(8, 256, 444u);
  ut_bridge3(12, 512, 555u);
  ut_bridge3(8, 768, 666u);
  printf("[FULL single-tensor mul_mat A==B: our bridge (real vmadot) vs ZERO-MODEL "
         "q4_K x q8_K + real ggml_vec_dot_q4_K_q8_K]\n");
  ut_full(8, 8, 256, 1234567u);
  ut_full(4, 8, 256, 2244668u);
  ut_full(12, 16, 512, 9988776u);
  ut_full(8, 32, 512, 5551234u);
  ut_full(16, 64, 768, 4242424u);
  if (!g_fail) {
    printf("Q4K-BRIDGE-UT PASS: #4 repack + #3 quant/pack exact vs ggml native; "
           "full single-tensor mul_mat A==B within f32 tol (real vmadot on K1)\n");
    return 0;
  }
  printf("Q4K-BRIDGE-UT FAIL\n");
  return 1;
}
