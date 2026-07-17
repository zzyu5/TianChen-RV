// INC-17 F5 — ggml silu (the FIRST forward-pass VECTORIZED TRANSCENDENTAL) bit-
// exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.ggml_vec_silu_f32 op
// (y[i] = x[i]*sigmoid(x[i]), sigmoid via ggml's EXACT vectorized minimax exp
// polynomial ggml_v_expf_m2) computes the SAME f32 output as ggml's OWN
// vectorized ggml_vec_silu_f32 / ggml_v_silu_m2 RVV path (vec.cpp:404-410 +
// vec.h:1324-1368), BIT-FOR-BIT (memcmp of the float bits), over many sizes x
// value distributions including the exp range (large +ve -> saturate to x, large
// -ve -> 0, around 0, the polynomial knots, NaN/inf/denormal).
//
// The crux is matching ggml's EXACT vectorized exp polynomial. ggml computes
// e^x with a fully vectorized minimax polynomial built entirely from __riscv_v
// intrinsics (NO libm expf). The ORACLE is ggml's VECTORIZED silu (the one that
// actually runs in inference) -- a verbatim transcription of ggml_v_silu_m2 /
// ggml_v_expf_m2 (vec.h). Bit-exactness vs the libm-scalar sigmoidf is NOT the
// goal (they may differ in the last ULP); the vectorized one is the deployment
// target, and that is what we match.
//
// NEGATIVE CONTROL: a WRONG-METHOD silu that computes sigmoid via libm expf
// (nc_libm_silu). It is mathematically the same function but uses a DIFFERENT
// exp implementation (libm's correctly-rounded-ish expf, not ggml's minimax),
// so it diverges from ggml's vectorized silu on many inputs. If our kernel
// matches ggml's vectorized silu bit-for-bit while nc_libm_silu DIFFERS, we have
// matched ggml's exact POLYNOMIAL (its method), not merely the silu function to
// some tolerance.
//
// The kernel under test is the UNMODIFIED, compiler-emitted C in silu_kernel.cpp
// -- every line tagged source_op=tcrv_rvv.ggml_vec_silu_f32, emitted by
// tcrv-opt --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp from the
// committed rvv-to-emitc-ggml-vec-silu-f32.mlir.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <riscv_vector.h>

// ---- The kernel our compiler emitted (declared; defined in silu_kernel.cpp) --
// ABI as exported: (size_t n, const float *x, float *y).
extern "C" void
tcrv_emitc_ggml_vec_silu_f32_kernel_ggml_vec_silu_f32(size_t n, const float *x,
                                                       float *y);

static void our_kernel(int n, const float *x, float *y) {
  tcrv_emitc_ggml_vec_silu_f32_kernel_ggml_vec_silu_f32((size_t)n, x, y);
}

// ---- ggml's REAL vectorized exp polynomial (vec.h:1324-1360), VERBATIM -------
// This is ggml_v_expf_m2 transcribed node-for-node (the deployment oracle). The
// `if (!vcpop_m(c))` short-circuit is KEPT here (it is what ggml does); our
// emitted kernel takes the unconditional slow path, and the two are bitwise
// equal -- which is exactly what this harness proves.
static inline vfloat32m2_t ggml_v_expf_m2_ref(vfloat32m2_t x, int vl) {
  const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
  const vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
  const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
  const vfloat32m2_t b = __riscv_vfnmsac_vf_f32m2(
      __riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
  const vuint32m2_t e =
      __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
  const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(
      __riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
  const vbool16_t c =
      __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
  const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
  const vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
      __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
      __riscv_vfmacc_vv_f32m2(
          __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl),
                                  0x1.555e66p-3f, b, vl),
          __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl),
                                  0x1.0e4020p-7f, b, vl),
          u, vl),
      u, vl);
  if (!__riscv_vcpop_m_b16(c, vl))
    return __riscv_vfmacc_vv_f32m2(k, j, k, vl);
  const vbool16_t dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
  const vuint32m2_t d =
      __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
  const vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(
      __riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
  const vfloat32m2_t s2 =
      __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
  const vfloat32m2_t r1 = __riscv_vmerge_vvm_f32m2(
      __riscv_vfmacc_vv_f32m2(k, k, j, vl),
      __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl), c,
      vl);
  return __riscv_vmerge_vvm_f32m2(
      r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
      __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl), vl);
}

static inline vfloat32m2_t ggml_v_silu_m2_ref(vfloat32m2_t x, int vl) {
  const vfloat32m2_t neg_x = __riscv_vfneg_v_f32m2(x, vl);
  const vfloat32m2_t exp_neg_x = ggml_v_expf_m2_ref(neg_x, vl);
  const vfloat32m2_t one_plus_exp_neg_x =
      __riscv_vfadd_vf_f32m2(exp_neg_x, 1.0f, vl);
  return __riscv_vfdiv_vv_f32m2(x, one_plus_exp_neg_x, vl);
}

// ggml's REAL ggml_vec_silu_f32 RVV strip (vec.cpp:404-410), VERBATIM.
static void ggml_ref(int n, const float *x, float *y) {
  int i = 0;
  for (int vl; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m2(n - i);
    vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
    vfloat32m2_t vy = ggml_v_silu_m2_ref(vx, vl);
    __riscv_vse32_v_f32m2(&y[i], vy, vl);
  }
}

// ---- NEGATIVE CONTROL: WRONG exp method (scalar libm expf) -------------------
// Same silu function, but sigmoid via libm expf rather than ggml's minimax
// polynomial -> different bits on many inputs. Proves we matched ggml's METHOD.
static void nc_libm_silu(int n, const float *x, float *y) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] / (1.0f + expf(-x[i]));
  }
}

static int bits_equal(const float *a, const float *b, int n) {
  return std::memcmp(a, b, (size_t)n * sizeof(float)) == 0;
}

// xorshift PRNG for reproducible f32 bit patterns.
static uint32_t rng_state = 0x13579bdfu;
static uint32_t xs32() {
  uint32_t x = rng_state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  rng_state = x;
  return x;
}
// dist: 0 around 0 [-2,2] (the polynomial-dense region),
//       1 moderate [-20,20] (the polynomial knots + reduction range),
//       2 saturation tails [-120,120] (large +ve -> x, large -ve -> 0,
//         crosses the 88.38 +inf flush and the -103.97 zero flush),
//       3 near the overflow knots [85,95] and [-100,-110] (the |n|>126/192
//         vmerge fixup boundaries),
//       4 raw bits (denormal/inf/nan/huge occasionally).
static float rand_f32(int dist) {
  uint32_t r = xs32();
  float u = ((float)(int32_t)r / (float)0x7fffffff); // [-1,1)
  switch (dist) {
  case 0: return u * 2.0f;
  case 1: return u * 20.0f;
  case 2: return u * 120.0f;
  case 3: {
    // Straddle the saturation knots where the |n|>126/|n|>192 vmerge fixup
    // kicks in -- the part of ggml's polynomial a naive impl gets wrong.
    float base = (r & 1) ? 90.0f : -105.0f;
    return base + u * 8.0f;
  }
  default: {
    if ((r & 0xF) == 0) {
      uint32_t bits = xs32();
      float f; std::memcpy(&f, &bits, 4); return f;
    }
    return u * 50.0f;
  }
  }
}

int main() {
  // The real llama-2-7b FFN intermediate dims + small/odd/tail sizes. silu runs
  // on the FFN gate (ne = 11008 for llama-2-7b) per token.
  const int sizes[] = {1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 127, 128,
                       256, 255, 257, 1024, 4096, 11008, 11007};
  const int NS = (int)(sizeof(sizes) / sizeof(sizes[0]));

  const int MAXN = 11008;
  float *x = (float *)malloc(sizeof(float) * MAXN);
  float *yo = (float *)malloc(sizeof(float) * MAXN);
  float *yr = (float *)malloc(sizeof(float) * MAXN);
  float *ync = (float *)malloc(sizeof(float) * MAXN);

  int total = 0, pass = 0;

  // 1) random rows x distributions x seeds, bit-exact vs ggml's VECTORIZED silu.
  for (int si = 0; si < NS; ++si) {
    int n = sizes[si];
    for (int dist = 0; dist <= 4; ++dist) {
      for (int seed = 0; seed < 24; ++seed) {
        for (int i = 0; i < n; ++i) x[i] = rand_f32(dist);
        our_kernel(n, x, yo);
        ggml_ref(n, x, yr);
        int ok = bits_equal(yo, yr, n);
        total++; pass += ok;
        if (!ok) {
          printf("MISMATCH n=%d dist=%d seed=%d\n", n, dist, seed);
          for (int i = 0; i < n && i < 8; ++i) {
            uint32_t bo, br;
            std::memcpy(&bo, &yo[i], 4); std::memcpy(&br, &yr[i], 4);
            if (bo != br)
              printf("  [%d] x=%a our=%08x ggml=%08x\n", i, (double)x[i], bo, br);
          }
        }
      }
    }
  }

  // 2) named edge-case rows (inf/nan/denormal/+-0/huge/saturation knots).
  auto bit = [](uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; };
  float edge_row[] = {
    0.0f, bit(0x80000000) /*-0*/, 1.0f, -1.0f, 0.5f, -0.5f,
    88.38f, 88.0f, 89.0f, -103.97f, -104.0f, -110.0f, // saturation knots
    bit(0x7f800000) /*+inf*/, bit(0xff800000) /*-inf*/,
    bit(0x7fc00000) /*nan*/, bit(0x00000001) /*denormal*/,
    bit(0x80000001) /*-denormal*/, 1e20f, -1e20f, 1e-20f,
    126.0f, -126.0f, 192.0f, -192.0f, 3.14159f, -2.71828f,
    bit(0x7f7fffff) /*FLT_MAX*/, bit(0xff7fffff) /*-FLT_MAX*/, 0.0001f, -0.0001f,
    20.0f, -20.0f,
  };
  int ne = (int)(sizeof(edge_row) / sizeof(edge_row[0]));
  our_kernel(ne, edge_row, yo);
  ggml_ref(ne, edge_row, yr);
  // NaN output: memcmp of the bit pattern is exact (ggml + ours both produce the
  // canonical NaN bit pattern on a NaN input, and identical bits on every other).
  {
    int ok = bits_equal(yo, yr, ne);
    total++; pass += ok;
    if (!ok) {
      printf("EDGE MISMATCH:\n");
      for (int i = 0; i < ne; ++i) {
        uint32_t bo, br;
        std::memcpy(&bo, &yo[i], 4); std::memcpy(&br, &yr[i], 4);
        if (bo != br)
          printf("  [%d] x=%a our=%08x ggml=%08x\n", i, (double)edge_row[i], bo, br);
      }
    }
  }

  // 3) NEGATIVE CONTROL: the WRONG-exp-method silu (libm expf) must DIFFER from
  // ggml's vectorized silu on at least the polynomial-active region. If it
  // matched everywhere, the byte-compare would not be discriminating "matches
  // ggml's exact polynomial" from "merely close".
  int nc_caught = 0, nc_total = 0;
  // Instrument the AGREEING NC cases by (dist, n) to show *why* they agree (the
  // saturated tails where exp is no longer the discriminator), rather than guess.
  int nc_agree_by_dist[3] = {0, 0, 0};
  int nc_agree_saturated = 0; // agreeing rows whose every lane is a saturation tail
  for (int si = 0; si < NS; ++si) {
    int n = sizes[si];
    for (int dist = 0; dist <= 2; ++dist) {
      for (int seed = 0; seed < 16; ++seed) {
        for (int i = 0; i < n; ++i) x[i] = rand_f32(dist);
        ggml_ref(n, x, yr);     // ggml's vectorized polynomial
        nc_libm_silu(n, x, ync); // WRONG: libm expf
        nc_total++;
        if (!bits_equal(yr, ync, n)) {
          nc_caught++;
        } else {
          nc_agree_by_dist[dist]++;
          // A row agrees iff NO lane is in the polynomial-active region: classify
          // it "saturated" if every lane is |x| >= 104 (past the +inf/zero flush
          // knots, where silu collapses to x or 0 in BOTH impls identically).
          int all_saturated = 1;
          for (int i = 0; i < n; ++i)
            if (fabsf(x[i]) < 104.0f) { all_saturated = 0; break; }
          if (all_saturated) nc_agree_saturated++;
        }
      }
    }
  }

  printf("INC-17 F5 ggml_vec_silu_f32: %d/%d bit-exact cases PASS (vs ggml's "
         "VECTORIZED silu)\n", pass, total);
  printf("NC wrong-exp-method (libm expf vs ggml minimax): %d/%d correctly "
         "DIFFER\n", nc_caught, nc_total);
  printf("NC agreement breakdown by dist: d0(~0)=%d d1(moderate)=%d "
         "d2(saturation)=%d; of all %d agreeing rows, %d are fully-saturated "
         "(every lane |x|>=104, exp not the discriminator)\n",
         nc_agree_by_dist[0], nc_agree_by_dist[1], nc_agree_by_dist[2],
         nc_total - nc_caught, nc_agree_saturated);

  free(x); free(yo); free(yr); free(ync);
  // The verdict GATES on the NC catching a majority -- the control that proves
  // the byte-compare discriminates ggml's EXACT polynomial from a merely-close
  // libm silu. If our kernel matches ggml's vectorized silu bit-for-bit while
  // the libm-exp silu DIFFERS, we matched ggml's METHOD (the minimax polynomial
  // node-for-node), not just the silu function to some tolerance.
  int nc_ok = (nc_caught * 2 > nc_total);
  if (pass == total && nc_ok) {
    printf("RESULT: PASS (byte-exact vs ggml VECTORIZED silu; the libm-exp NC "
           "discriminates -> ggml's exact minimax exp polynomial is matched "
           "node-for-node, not just its tolerance)\n");
    return 0;
  }
  printf("RESULT: FAIL\n");
  return 1;
}
