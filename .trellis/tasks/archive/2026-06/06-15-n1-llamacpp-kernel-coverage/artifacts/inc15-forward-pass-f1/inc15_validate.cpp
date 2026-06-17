// INC-15 F1 — ggml_vec_scale_f32 bit-exact HW validation harness (ssh rvv,
// -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.ggml_vec_scale_f32 op
// (the f32 in-place elementwise scale y[i] *= v: ONE vsetvl_e32m8(n-i) strip
// loop with vle32 / vfmul_vf / vse32) computes the SAME f32 buffer as ggml's
// OWN hand-written ggml_vec_scale_f32 RVV path (vec.h:733-739), BIT-FOR-BIT
// (memcmp of the float bits), over random f32 buffers at many n + named edge
// cases (NaN / +-inf / denormal / +-0 / huge / v=0 / v=1 / v=NaN / v=inf).
// A negative control (a perturbed scale) must FAIL, proving discrimination.
//
// REFERENCE: the REAL ggml RVV path, transcribed intrinsic-for-intrinsic from
// llama.cpp/ggml/src/ggml-cpu/vec.h:733-739:
//   for (int i = 0, avl; i < n; i += avl) {
//     avl = __riscv_vsetvl_e32m8(n - i);
//     vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
//     vfloat32m8_t ny = __riscv_vfmul_vf_f32m8(ay, v, avl);
//     __riscv_vse32_v_f32m8(&y[i], ny, avl);
//   }
// We ALSO cross-check against ggml's SCALAR reference (y[i] *= v). Byte-exactness
// is UNCONDITIONAL here: y[i]*v is a bare per-lane fp32 multiply -- no FMA (so
// -ffp-contract is irrelevant; we still build the whole TU with one setting),
// no cross-lane reduction (so the strip width is correctness-free). The compiler
// kernel anchors at m8 (its bounded lmul knob); ggml's RVV path is m8 too, and
// the scalar ref is order-identical per lane, so all three are bit-equal.
//
// The kernel under test is the UNMODIFIED, compiler-emitted C in
// scale_kernel.cpp -- every line tagged source_op=tcrv_rvv.ggml_vec_scale_f32,
// emitted by tcrv-opt --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp
// from the committed rvv-to-emitc-ggml-vec-scale-f32.mlir.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <riscv_vector.h>

// ---- The kernel our compiler emitted (declared; defined in scale_kernel.cpp) -
// ABI as exported: (size_t n, float *y, float v); y is read AND written in place.
extern "C" void
tcrv_emitc_ggml_vec_scale_f32_kernel_ggml_vec_scale_f32(size_t n, float *y,
                                                        float v);

static void our_kernel(int n, float *y, float v) {
  tcrv_emitc_ggml_vec_scale_f32_kernel_ggml_vec_scale_f32((size_t)n, y, v);
}

// ---- ggml's REAL ggml_vec_scale_f32 RVV path (vec.h:733-739) ----------------
static void ggml_rvv(int n, float *y, float v) {
  for (int i = 0, avl; i < n; i += avl) {
    avl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
    vfloat32m8_t ny = __riscv_vfmul_vf_f32m8(ay, v, avl);
    __riscv_vse32_v_f32m8(&y[i], ny, avl);
  }
}

// ---- ggml's SCALAR reference (vec.h:763) ------------------------------------
static void ggml_scalar(int n, float *y, float v) {
  for (int i = 0; i < n; ++i) {
    y[i] *= v;
  }
}

// ---- bit compare ------------------------------------------------------------
static int bits_equal(const float *a, const float *b, int n) {
  return std::memcmp(a, b, (size_t)n * sizeof(float)) == 0;
}

// A simple xorshift PRNG for reproducible f32 bit patterns.
static uint32_t rng_state = 0x12345678u;
static uint32_t xs32() {
  uint32_t x = rng_state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  rng_state = x;
  return x;
}
static float rand_f32() {
  // Mix normal-range values with the occasional extreme bit pattern.
  uint32_t r = xs32();
  if ((r & 0xF) == 0) {
    // raw bit pattern (covers denormals/inf/nan/huge)
    uint32_t bits = xs32();
    float f; std::memcpy(&f, &bits, 4); return f;
  }
  // value in [-100, 100)
  return ((float)(int32_t)r / (float)0x7fffffff) * 100.0f;
}

int main() {
  const int sizes[] = {8, 16, 17, 32, 33, 64, 128, 255, 256, 1023, 1024, 4096};
  const float scales[] = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 1e-30f, 1e30f, 3.14159f};
  const int NS = (int)(sizeof(sizes) / sizeof(sizes[0]));
  const int NV = (int)(sizeof(scales) / sizeof(scales[0]));

  const int MAXN = 4096;
  float *base = (float *)malloc(sizeof(float) * MAXN);
  float *yo = (float *)malloc(sizeof(float) * MAXN);
  float *yr = (float *)malloc(sizeof(float) * MAXN);
  float *ys = (float *)malloc(sizeof(float) * MAXN);

  int total = 0, pass = 0;

  // 1) random buffers x scales (every size x every scale, 40 seeds each)
  for (int si = 0; si < NS; ++si) {
    int n = sizes[si];
    for (int vi = 0; vi < NV; ++vi) {
      float v = scales[vi];
      for (int seed = 0; seed < 40; ++seed) {
        for (int i = 0; i < n; ++i) base[i] = rand_f32();
        std::memcpy(yo, base, n * 4);
        std::memcpy(yr, base, n * 4);
        std::memcpy(ys, base, n * 4);
        our_kernel(n, yo, v);
        ggml_rvv(n, yr, v);
        ggml_scalar(n, ys, v);
        // NaN bit patterns are unordered under ==; memcmp on the raw bits is
        // the right test (a NaN input * v stays a NaN with our/ggml producing
        // the SAME quiet-NaN bits via the same vfmul). For the scalar ref we
        // ALSO compare bits.
        int ok = bits_equal(yo, yr, n) && bits_equal(yo, ys, n);
        total++; pass += ok;
        if (!ok) {
          printf("MISMATCH n=%d v=%a seed=%d\n", n, (double)v, seed);
          for (int i = 0; i < n && i < 8; ++i) {
            uint32_t bo, br, bs;
            std::memcpy(&bo, &yo[i], 4); std::memcpy(&br, &yr[i], 4);
            std::memcpy(&bs, &ys[i], 4);
            printf("  [%d] in=%a our=%08x ggml=%08x scalar=%08x\n",
                   i, (double)base[i], bo, br, bs);
          }
        }
      }
    }
  }

  // 2) named edge-case buffers (NaN / inf / denormal / +-0 / huge) x scales
  auto bit = [](uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; };
  float edges[] = {
    bit(0x7fc00000), // qNaN
    bit(0xffc00000), // -qNaN
    bit(0x7f800000), // +inf
    bit(0xff800000), // -inf
    bit(0x00000001), // smallest denormal
    bit(0x807fffff), // -largest denormal
    bit(0x00000000), // +0
    bit(0x80000000), // -0
    bit(0x7f7fffff), // FLT_MAX
    bit(0xff7fffff), // -FLT_MAX
    3.14159f, -2.5f, 100.0f, 0.0001f,
  };
  int ne = (int)(sizeof(edges) / sizeof(edges[0]));
  const float edge_scales[] = {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, 1e30f, 1e-30f,
                               bit(0x7fc00000) /*v=NaN*/, bit(0x7f800000) /*v=inf*/};
  int nev = (int)(sizeof(edge_scales) / sizeof(edge_scales[0]));
  for (int vi = 0; vi < nev; ++vi) {
    float v = edge_scales[vi];
    int n = ne;
    std::memcpy(yo, edges, n * 4);
    std::memcpy(yr, edges, n * 4);
    std::memcpy(ys, edges, n * 4);
    our_kernel(n, yo, v);
    ggml_rvv(n, yr, v);
    ggml_scalar(n, ys, v);
    int ok = bits_equal(yo, yr, n) && bits_equal(yo, ys, n);
    total++; pass += ok;
    if (!ok) {
      printf("EDGE MISMATCH v=%a\n", (double)v);
      for (int i = 0; i < n; ++i) {
        uint32_t bo, br, bs;
        std::memcpy(&bo, &yo[i], 4); std::memcpy(&br, &yr[i], 4);
        std::memcpy(&bs, &ys[i], 4);
        if (bo != br || bo != bs)
          printf("  [%d] in=%08x our=%08x ggml=%08x scalar=%08x\n",
                 i, *(uint32_t*)&edges[i], bo, br, bs);
      }
    }
  }

  // 3) NEGATIVE control: a perturbed scale must DIFFER (proves discrimination).
  int neg_caught = 0, neg_total = 0;
  for (int si = 0; si < NS; ++si) {
    int n = sizes[si];
    for (int i = 0; i < n; ++i) base[i] = rand_f32();
    std::memcpy(yo, base, n * 4);
    std::memcpy(yr, base, n * 4);
    our_kernel(n, yo, 3.0f);
    ggml_rvv(n, yr, 3.0f + 1e-6f); // perturbed
    neg_total++;
    if (!bits_equal(yo, yr, n)) neg_caught++;
  }

  printf("INC-15 F1 ggml_vec_scale_f32: %d/%d bit-exact cases PASS\n", pass, total);
  printf("negative control: %d/%d perturbed-scale cases correctly DIFFER\n",
         neg_caught, neg_total);

  free(base); free(yo); free(yr); free(ys);
  if (pass == total && neg_caught == neg_total) {
    printf("RESULT: PASS (byte-exact vs ggml RVV AND scalar; negative control discriminates)\n");
    return 0;
  }
  printf("RESULT: FAIL\n");
  return 1;
}
