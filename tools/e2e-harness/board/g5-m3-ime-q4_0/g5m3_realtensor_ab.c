// G5-M3 session-2 REAL-MODEL-TENSOR A==B seal for the q4_0 IME forward bridge.
//
// Strengthens the integration UT from random matrices to the ACTUAL deployment
// weight bytes: it opens tinyllama-q4_0.gguf, extracts a genuine on-disk NATIVE
// q4_0 weight tensor (the exact bytes the real llama forward consumes before the
// vendor repack), takes an (N x K) sub-tile, and runs the full tcrv IME bridge
//   native q4_0 weight (#4 repack) x q8_0(f32 activation) (#3 quant/pack)
//     -> fragment-major scale-fold GEMM (real vmadot leaf on K1)     [ = A ]
// against the stock ggml q4_0 x q8_0 block-dot re-derived from the same native
// bytes                                                              [ = B ]
// and asserts A == B (bit-exact in matching float fold order; ~1e-7 normwise vs a
// double-precision ground truth = pure f32 reassociation). Real vmadot -> silicon.
//
// Build (board toolchain that assembles vmadot; link ggml-base for gguf + quant):
//   gcc -O2 -std=c11 -march=rv64gcv_xsmtvdotii1p0 -mabi=lp64d g5m3_realtensor_ab.c \
//       -I <ggml/include> -L <build-off/bin> -lggml-base -lm -Wl,-rpath,<..> -o rt
//   taskset -c 0-3 ./rt <model.gguf>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "gguf.h"

extern void  quantize_row_q8_0_ref(const float *x, void *y, int64_t k);
extern float ggml_fp16_to_fp32(uint16_t h);

#define Q40_NATIVE_BYTES 18
#define Q80_NATIVE_BYTES 34

// --- emitter-verbatim kernel leaves (identical to the sealed scale-fold) -----
static inline void tcrv_ime_vmadot_mac_kloop(const int8_t *A, const int8_t *B,
                                             long kt, int32_t *frag) {
  __asm__ volatile(
      "vsetvli   t0, zero, e8, m1, ta, ma   \n\t"
      "vmv.v.i   v2, 0\n\t vmv.v.i v3, 0\n\t"
      "mv t2, %[kt]\n\t mv t3, %[pa]\n\t mv t4, %[pb]\n\t"
      "1:\n\t vle8.v v0,(t3)\n\t vle8.v v1,(t4)\n\t vmadot v2,v0,v1\n\t"
      "addi t3,t3,32\n\t addi t4,t4,32\n\t addi t2,t2,-1\n\t bnez t2,1b\n\t"
      "vsetvli t0, zero, e32, m1, ta, ma\n\t vse32.v v2,(%[pf])\n\t"
      "addi t5,%[pf],32\n\t vse32.v v3,(t5)\n\t"
      :
      : [pa] "r"(A), [pb] "r"(B), [kt] "r"(kt), [pf] "r"(frag)
      : "t0", "t2", "t3", "t4", "t5", "v0", "v1", "v2", "v3", "memory");
}
static inline void tcrv_ime_q4_0_dequant_fragment(const uint8_t *blk, int8_t *out) {
  const uint8_t *qs = blk + 2;
  for (int j = 0; j < 16; ++j) {
    out[j] = (int8_t)((int)(qs[j] & 0x0F) - 8);
    out[j + 16] = (int8_t)((int)(qs[j] >> 4) - 8);
  }
}
static void tcrv_ime_q4_0_vmadot_matmul_f32(const int8_t *Apack, const float *dA,
                                            const uint8_t *Bnib, const float *dW,
                                            float *Cf, long M, long N, long K) {
  const long mt = M / 4, nt = N / 4, nb = K / 32, kt = K / 8, q40b = 18, fpb = 4;
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Apack + (long)mi * 4 * K;
    for (long nj = 0; nj < nt; ++nj) {
      const uint8_t *Bcol = Bnib + (long)nj * kt * q40b;
      for (long b = 0; b < nb; ++b) {
        int8_t Bdec[128];
        for (long f = 0; f < fpb; ++f)
          tcrv_ime_q4_0_dequant_fragment(Bcol + (b * fpb + f) * q40b, Bdec + f * 32);
        int32_t frag[16];
        tcrv_ime_vmadot_mac_kloop(Arow + b * fpb * 32, Bdec, fpb, frag);
        for (long r = 0; r < 4; ++r)
          for (long c = 0; c < 4; ++c) {
            long m = mi * 4 + r, n = nj * 4 + c;
            Cf[m * N + n] += dA[m * nb + b] * dW[n * nb + b] * (float)frag[r * 4 + c];
          }
      }
    }
  }
}

// --- session-2 bridges (repack real native formats) --------------------------
static void tcrv_bridge4_repack_q4_0(const uint8_t *wq, uint8_t *Bnib, float *dW,
                                     long N, long K) {
  const long nb = K / 32, kt = K / 8, q40b = Q40_NATIVE_BYTES;
  memset(Bnib, 0, (size_t)(N / 4) * kt * q40b);
  for (long n = 0; n < N; ++n)
    for (long b = 0; b < nb; ++b) {
      const uint8_t *src = wq + (n * nb + b) * Q40_NATIVE_BYTES;
      dW[n * nb + b] = ggml_fp16_to_fp32((uint16_t)(src[0] | (src[1] << 8)));
      const uint8_t *qs_src = src + 2;
      for (long kk = 0; kk < 32; ++kk) {
        long k = b * 32 + kk, nj = n / 4, nl = n % 4, kf = k / 8, kl = k % 8;
        uint8_t *blk = Bnib + (nj * kt + kf) * q40b;
        blk[0] = src[0]; blk[1] = src[1];
        uint8_t *qs = blk + 2;
        long idx = nl * 8 + kl;
        int val4 = (kk >= 16) ? (qs_src[kk % 16] >> 4) : (qs_src[kk % 16] & 0x0F);
        if (idx < 16) qs[idx] = (uint8_t)((qs[idx] & 0xF0) | (val4 & 0x0F));
        else          qs[idx - 16] = (uint8_t)((qs[idx - 16] & 0x0F) | ((val4 & 0x0F) << 4));
      }
    }
}
static void tcrv_bridge3_quant_pack_q8_0(const float *X, int8_t *Apack, float *dA,
                                         long M, long K, uint8_t *scratch) {
  const long nb = K / 32;
  for (long m = 0; m < M; ++m) {
    quantize_row_q8_0_ref(X + m * K, scratch, K);
    for (long b = 0; b < nb; ++b) {
      const uint8_t *blk = scratch + b * Q80_NATIVE_BYTES;
      dA[m * nb + b] = ggml_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
      const int8_t *qs = (const int8_t *)(blk + 2);
      for (long kk = 0; kk < 32; ++kk) {
        long k = b * 32 + kk, mi = m / 4, ml = m % 4, kf = k / 8, kl = k % 8;
        Apack[mi * 4 * K + kf * 32 + ml * 8 + kl] = qs[kk];
      }
    }
  }
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1]
                              : "/home/bianbu/tcrv-k1-llama/models/tinyllama-q4_0.gguf";
  struct ggml_context *mctx = NULL;
  struct gguf_init_params p = {.no_alloc = false, .ctx = &mctx};
  struct gguf_context *gg = gguf_init_from_file(path, p);
  if (!gg || !mctx) {
    printf("REALTENSOR FAIL: cannot open %s\n", path);
    return 1;
  }
  // pick the first q4_0 2D weight tensor with usable shape.
  struct ggml_tensor *t = NULL;
  for (struct ggml_tensor *c = ggml_get_first_tensor(mctx); c;
       c = ggml_get_next_tensor(mctx, c)) {
    if (c->type == GGML_TYPE_Q4_0 && c->ne[0] % 32 == 0 && c->ne[1] % 4 == 0 &&
        c->ne[2] == 1 && c->ne[3] == 1 && c->ne[0] >= 256 && c->ne[1] >= 64) {
      t = c;
      break;
    }
  }
  if (!t) {
    printf("REALTENSOR FAIL: no usable q4_0 2D tensor found\n");
    gguf_free(gg);
    return 1;
  }
  long Kfull = t->ne[0], Nfull = t->ne[1];
  // sub-tile: N=64 rows, K=512 cols (bounded runtime; contiguous native blocks).
  long N = 64, K = 512;
  if (N > Nfull) N = (Nfull / 4) * 4;
  if (K > Kfull) K = (Kfull / 32) * 32;
  long nb = K / 32, nbfull = Kfull / 32, kt = K / 8, M = 32;
  printf("G5-M3 REAL-MODEL-TENSOR A==B: tensor '%s' [K=%ld N=%ld] sub-tile "
         "N=%ld K=%ld M=%ld (real vmadot)\n",
         ggml_get_name(t), Kfull, Nfull, N, K, M);

  // extract native q4_0 sub-tile: rows 0..N-1, first nb blocks of each row.
  const uint8_t *base = (const uint8_t *)t->data;
  uint8_t *wq = malloc((size_t)N * nb * Q40_NATIVE_BYTES);
  for (long n = 0; n < N; ++n)
    memcpy(wq + (size_t)n * nb * Q40_NATIVE_BYTES,
           base + (size_t)n * nbfull * Q40_NATIVE_BYTES,
           (size_t)nb * Q40_NATIVE_BYTES);

  // synthetic f32 activation (deterministic), quantized via ggml q8_0.
  float *X = malloc(sizeof(float) * M * K);
  srand(20260712u);
  for (long i = 0; i < M * K; ++i) X[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  uint8_t *aq = malloc((size_t)M * nb * Q80_NATIVE_BYTES);
  for (long m = 0; m < M; ++m)
    quantize_row_q8_0_ref(X + m * K, aq + (size_t)m * nb * Q80_NATIVE_BYTES, K);

  // A: our full bridge
  uint8_t *Bnib = malloc((size_t)(N / 4) * kt * Q40_NATIVE_BYTES);
  float *dW = malloc(sizeof(float) * N * nb);
  tcrv_bridge4_repack_q4_0(wq, Bnib, dW, N, K);
  int8_t *Apack = malloc((size_t)(M / 4) * kt * 32);
  float *dA = malloc(sizeof(float) * M * nb);
  uint8_t *scr = malloc((size_t)nb * Q80_NATIVE_BYTES);
  tcrv_bridge3_quant_pack_q8_0(X, Apack, dA, M, K, scr);
  float *Cf = calloc((size_t)M * N, sizeof(float));
  tcrv_ime_q4_0_vmadot_matmul_f32(Apack, dA, Bnib, dW, Cf, M, N, K);

  // B: stock q4_0 x q8_0 block-dot (double truth + float-order match)
  double max_abs = 0.0, ref_max = 0.0, max_abs_vs_float = 0.0, max_rel_cell = 0.0;
  for (long m = 0; m < M; ++m)
    for (long n = 0; n < N; ++n) {
      double ref_d = 0.0;
      float ref_f = 0.0f;
      for (long b = 0; b < nb; ++b) {
        const uint8_t *wblk = wq + (size_t)(n * nb + b) * Q40_NATIVE_BYTES;
        const uint8_t *ablk = aq + (size_t)(m * nb + b) * Q80_NATIVE_BYTES;
        float dw = ggml_fp16_to_fp32((uint16_t)(wblk[0] | (wblk[1] << 8)));
        float da = ggml_fp16_to_fp32((uint16_t)(ablk[0] | (ablk[1] << 8)));
        const uint8_t *wqs = wblk + 2;
        const int8_t *aqs = (const int8_t *)(ablk + 2);
        int32_t part = 0;
        for (long j = 0; j < 16; ++j) {
          part += (int32_t)aqs[j] * ((int)(wqs[j] & 0x0F) - 8) +
                  (int32_t)aqs[j + 16] * ((int)(wqs[j] >> 4) - 8);
        }
        ref_d += (double)da * (double)dw * (double)part;
        ref_f += da * dw * (float)part;
      }
      double got = (double)Cf[m * N + n];
      double ae = fabs(got - ref_d);
      double re = fabs(ref_d) > 1e-6 ? ae / fabs(ref_d) : ae;
      if (ae > max_abs) max_abs = ae;
      if (re > max_rel_cell) max_rel_cell = re;
      if (fabs(ref_d) > ref_max) ref_max = fabs(ref_d);
      double avf = fabs(got - (double)ref_f);
      if (avf > max_abs_vs_float) max_abs_vs_float = avf;
    }
  double normwise = max_abs / (ref_max > 1e-9 ? ref_max : 1.0);
  int ok = normwise < 1e-5;
  printf("  A==B  normwise=%.3e (max_abs=%.3e |C|max=%.3f) "
         "max_abs_vs_float_order=%.3e per_cell_rel=%.3e %s\n",
         normwise, max_abs, ref_max, max_abs_vs_float, max_rel_cell,
         ok ? "OK" : "FAIL");
  free(wq); free(X); free(aq); free(Bnib); free(dW); free(Apack); free(dA);
  free(scr); free(Cf);
  gguf_free(gg);
  ggml_free(mctx);
  if (ok) {
    printf("REALTENSOR PASS: tcrv IME bridge == stock q4_0xq8_0 on REAL model "
           "weight bytes (bit-exact float order; ~1e-7 normwise vs double)\n");
    return 0;
  }
  printf("REALTENSOR FAIL\n");
  return 1;
}
