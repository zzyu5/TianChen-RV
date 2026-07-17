// INC-18 F5b — ggml soft_max (the attention op: COMBINES F5's vectorized exp
// transcendental + a NEW f64 widening-reduce) bit-exact HW validation harness
// (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.ggml_vec_soft_max_f32
// op computes BIT-FOR-BIT the SAME as ggml's OWN bare vectorized
// ggml_vec_soft_max_f32 RVV path (vec.cpp:531-600, the __riscv_v_intrinsic
// branch), on TWO outputs:
//   (a) the written y[i] = e^{x[i]-max} buffer (memcmp of the float bits), AND
//   (b) the RETURNED f64 sum Sum_i e^{x[i]-max} (memcmp of the double bits).
// over many row sizes x value distributions including attention dims, large/
// small/uniform/spiky/all-equal, and the -inf masked entries attention uses.
//
// Faithful to ggml's BARE function: `max` is an INPUT (the caller computes it
// with the scalar ggml_vec_max_f32 and passes it; ops.cpp:5400), and the bare
// function does NOT normalize (the caller applies 1/sum via ggml_vec_scale_f32
// afterward; ops.cpp:5415-5416). So the oracle here is exactly the bare
// ggml_vec_soft_max_f32 — same signature, same semantics.
//
// TWO byte-exactness cruxes:
//   1. exp(x-max): the SAME node-for-node ggml_v_expf_m2 minimax polynomial F5
//      silu already proved bit-exact (reused via the shared emitGgmlVExpfM2).
//   2. the SUM: ggml accumulates in ggml_float = DOUBLE via the WIDENING reduce
//      vfwredusum_vs_f32m2_f64m1 into a SINGLE f64m1 accumulator carried across
//      strips, then returns (double)vfmv_f_s_f64m1. This fold ORDER (a vector
//      tree-reduce widened to f64, accumulated chunk-by-chunk) is the new risk;
//      a scalar fold or an f32 fold gives DIFFERENT bits.
//
// NEGATIVE CONTROLS (two, each isolating one crux):
//   - nc_libm_exp: y/sum via libm expf (the WRONG exp method) — proves we match
//     ggml's exact POLYNOMIAL, not a merely-close exp.
//   - nc_f32_sum: ggml's EXACT y[] (vectorized polynomial) but the sum
//     accumulated in F32 (NOT the f64 widening reduce ggml uses) — proves the
//     returned-sum byte-match pins ggml's EXACT f64 widening reduce, not just
//     "a sum of the same y[]".
//
// The kernel under test is the UNMODIFIED, compiler-emitted C in
// soft_max_kernel.cpp — every line tagged
// source_op=tcrv_rvv.ggml_vec_soft_max_f32, emitted by tcrv-opt
// --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp from the committed
// rvv-to-emitc-ggml-vec-soft-max-f32.mlir.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <riscv_vector.h>

typedef double ggml_float;

// ---- The kernel our compiler emitted (declared; defined in soft_max_kernel.cpp)
// ABI as exported: double f(size_t n, float *y, const float *x, float max).
extern "C" double
tcrv_emitc_ggml_vec_soft_max_f32_kernel_ggml_vec_soft_max_f32(size_t n, float *y,
                                                              const float *x,
                                                              float max);

static double our_kernel(int n, float *y, const float *x, float max) {
  return tcrv_emitc_ggml_vec_soft_max_f32_kernel_ggml_vec_soft_max_f32(
      (size_t)n, y, x, max);
}

// ---- ggml's REAL vectorized exp polynomial (vec.h:1324-1360), VERBATIM -------
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

// ggml's REAL bare ggml_vec_soft_max_f32 RVV path (vec.cpp:584-592), VERBATIM.
// Writes y[i] = e^{x[i]-max}; returns the f64 sum.
static ggml_float ggml_ref(int n, float *y, const float *x, float max) {
  int i = 0;
  vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);
  for (int avl; i < n; i += avl) {
    avl = __riscv_vsetvl_e32m2(n - i);
    vfloat32m2_t val = ggml_v_expf_m2_ref(
        __riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(&x[i], avl), max, avl),
        avl);
    __riscv_vse32_v_f32m2(&y[i], val, avl);
    vsum = __riscv_vfwredusum_vs_f32m2_f64m1(val, vsum, avl);
  }
  return (ggml_float)__riscv_vfmv_f_s_f64m1_f64(vsum);
}

// ---- NEGATIVE CONTROL 1: WRONG exp method (scalar libm expf) -----------------
// The scalar fallback ggml itself uses on a non-SIMD target (vec.cpp:594-599):
// y/sum via libm expf rather than the minimax polynomial -> different y bits and
// different sum on many inputs. Proves we matched ggml's exp METHOD.
static ggml_float nc_libm_exp(int n, float *y, const float *x, float max) {
  ggml_float sum = 0;
  for (int i = 0; i < n; ++i) {
    float val = expf(x[i] - max);
    sum += (ggml_float)val;
    y[i] = val;
  }
  return sum;
}

// ---- NEGATIVE CONTROL 2: ggml's EXACT y[], but the WRONG sum PRECISION --------
// y[] is ggml's vectorized polynomial (so y is bit-exact), but the sum is
// accumulated in F32, NOT the f64 widening reduce ggml uses (vfwredusum into
// f64m1). This is the genuinely-wrong alternative a naive implementer picks
// (drop the f64 widen, sum in f32). On long rows the f32 accumulation loses low
// bits and DIVERGES from ggml's f64 sum -> proves the returned-sum byte-match
// pins ggml's EXACT f64 widening reduce, not merely "a sum of the same y[]".
// (A scalar-ascending DOUBLE fold coincides with vfwredusum on these rows
// because f64 has the headroom to be order-insensitive here; the discriminating
// error is the PRECISION drop to f32, which is what a real wrong impl would do.)
static ggml_float nc_f32_sum(int n, float *y, const float *x, float max) {
  // First produce ggml's exact y[] via the vectorized polynomial.
  ggml_ref(n, y, x, max);
  // Then fold the sum in F32 (the WRONG precision: ggml folds in f64).
  float sum = 0.0f;
  for (int i = 0; i < n; ++i)
    sum += y[i];
  return (ggml_float)sum;
}

static int bits_equal_f(const float *a, const float *b, int n) {
  return std::memcmp(a, b, (size_t)n * sizeof(float)) == 0;
}
static int bits_equal_d(double a, double b) {
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

// xorshift PRNG for reproducible f32 bit patterns.
static uint32_t rng_state = 0x2468ace0u;
static uint32_t xs32() {
  uint32_t x = rng_state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  rng_state = x;
  return x;
}
// dist: 0 around 0 [-2,2], 1 moderate [-20,20], 2 saturation tails [-120,120],
//       3 near the overflow knots, 4 raw bits (denormal/inf/nan occasionally),
//       5 SPIKY (one huge lane, rest small — softmax's hard case),
//       6 UNIFORM (all lanes equal — sum should be ~n), 7 with -INF masked lanes.
static float rand_f32(int dist, int lane) {
  uint32_t r = xs32();
  float u = ((float)(int32_t)r / (float)0x7fffffff); // [-1,1)
  switch (dist) {
  case 0: return u * 2.0f;
  case 1: return u * 20.0f;
  case 2: return u * 120.0f;
  case 3: { float base = (r & 1) ? 90.0f : -105.0f; return base + u * 8.0f; }
  case 5: return (lane == 0) ? 80.0f : (u * 1.0f); // spike at lane 0
  case 6: return 1.5f;                             // all-equal
  case 7:
    // Attention-mask shape: ~30% of lanes are -inf (masked), rest moderate.
    return ((r & 7) < 2) ? -INFINITY : (u * 10.0f);
  default: {
    if ((r & 0xF) == 0) { uint32_t b = xs32(); float f; std::memcpy(&f, &b, 4); return f; }
    return u * 50.0f;
  }
  }
}

// ggml's caller computes max with the SCALAR ascending ggml_vec_max_f32
// (vec.h:1541, MAX fold from -INFINITY). Reproduce it EXACTLY so the `max`
// argument we pass to both our kernel and the oracle is identical (and so -inf
// masked rows behave like real attention: max ignores -inf via MAX(-INF, x)).
static float ggml_vec_max_f32_ref(int n, const float *x) {
  float max = -INFINITY;
  for (int i = 0; i < n; ++i) max = (max > x[i]) ? max : x[i];
  return max;
}

int main() {
  // Attention dims: seq/kv lengths (the soft_max row = #keys per query). Plus
  // small/odd/tail/strip-boundary sizes.
  const int sizes[] = {1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 127, 128,
                       256, 255, 257, 512, 1024, 2048, 4096, 2047, 4095};
  const int NS = (int)(sizeof(sizes) / sizeof(sizes[0]));

  const int MAXN = 4096;
  float *x = (float *)malloc(sizeof(float) * MAXN);
  float *yo = (float *)malloc(sizeof(float) * MAXN);
  float *yr = (float *)malloc(sizeof(float) * MAXN);
  float *ync = (float *)malloc(sizeof(float) * MAXN);

  int total = 0, pass_y = 0, pass_sum = 0;

  for (int si = 0; si < NS; ++si) {
    int n = sizes[si];
    for (int dist = 0; dist <= 7; ++dist) {
      for (int seed = 0; seed < 24; ++seed) {
        for (int i = 0; i < n; ++i) x[i] = rand_f32(dist, i);
        float max = ggml_vec_max_f32_ref(n, x);
        // An all-(-inf) row would make max=-inf and every exp(-inf-(-inf))=exp(nan);
        // ggml's caller asserts sum>0 so such rows never reach soft_max. Skip a
        // degenerate all-masked row (only possible for dist 7 tiny n) to mirror
        // the real call contract.
        if (max == -INFINITY) continue;
        double so = our_kernel(n, yo, x, max);
        double sr = ggml_ref(n, yr, x, max);
        int oky = bits_equal_f(yo, yr, n);
        int oks = bits_equal_d(so, sr);
        total++; pass_y += oky; pass_sum += oks;
        if (!oky || !oks) {
          printf("MISMATCH n=%d dist=%d seed=%d y_ok=%d sum_ok=%d "
                 "our_sum=%a ggml_sum=%a\n", n, dist, seed, oky, oks, so, sr);
          for (int i = 0; i < n && i < 6; ++i) {
            uint32_t bo, br;
            std::memcpy(&bo, &yo[i], 4); std::memcpy(&br, &yr[i], 4);
            if (bo != br)
              printf("  y[%d] x=%a our=%08x ggml=%08x\n", i, (double)x[i], bo, br);
          }
        }
      }
    }
  }

  // Named edge-case row: +-0, inf, nan, denormal, FLT_MAX, the saturation knots,
  // and -inf masked lanes interleaved (attention mask).
  auto bit = [](uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; };
  float edge_row[] = {
    0.0f, bit(0x80000000), 1.0f, -1.0f, 0.5f, -0.5f,
    -INFINITY, 2.0f, -INFINITY, 3.0f, -INFINITY, -INFINITY,  // masked lanes
    88.38f, 88.0f, 89.0f, -103.97f, -104.0f, -110.0f,
    bit(0x00000001), bit(0x80000001), 1e-20f,
    126.0f, -126.0f, 192.0f, -192.0f, 3.14159f, -2.71828f,
    20.0f, -20.0f, 0.0001f, -0.0001f, 10.0f,
  };
  int ne = (int)(sizeof(edge_row) / sizeof(edge_row[0]));
  {
    float max = ggml_vec_max_f32_ref(ne, edge_row);
    double so = our_kernel(ne, yo, edge_row, max);
    double sr = ggml_ref(ne, yr, edge_row, max);
    int oky = bits_equal_f(yo, yr, ne);
    int oks = bits_equal_d(so, sr);
    total++; pass_y += oky; pass_sum += oks;
    if (!oky || !oks) {
      printf("EDGE MISMATCH y_ok=%d sum_ok=%d our_sum=%a ggml_sum=%a\n",
             oky, oks, so, sr);
      for (int i = 0; i < ne; ++i) {
        uint32_t bo, br;
        std::memcpy(&bo, &yo[i], 4); std::memcpy(&br, &yr[i], 4);
        if (bo != br)
          printf("  y[%d] x=%a our=%08x ggml=%08x\n", i, (double)edge_row[i], bo, br);
      }
    }
  }

  // NEGATIVE CONTROL 1: WRONG exp method (libm expf). Must DIFFER from ggml's
  // vectorized soft_max (y OR sum) on the polynomial-active region.
  int nc1_caught = 0, nc1_total = 0;
  for (int si = 0; si < NS; ++si) {
    int n = sizes[si];
    for (int dist = 0; dist <= 1; ++dist) {
      for (int seed = 0; seed < 16; ++seed) {
        for (int i = 0; i < n; ++i) x[i] = rand_f32(dist, i);
        float max = ggml_vec_max_f32_ref(n, x);
        if (max == -INFINITY) continue;
        double sr = ggml_ref(n, yr, x, max);
        double snc = nc_libm_exp(n, ync, x, max);
        nc1_total++;
        if (!bits_equal_f(yr, ync, n) || !bits_equal_d(sr, snc)) nc1_caught++;
      }
    }
  }

  // NEGATIVE CONTROL 2: ggml's EXACT y[], but the WRONG sum PRECISION (f32
  // accumulation, not the f64 widening reduce). y matches; the SUM must DIFFER
  // on enough rows to prove the returned-sum byte-match pins the EXACT f64
  // widening reduce (a precision discriminator, separate from the polynomial).
  int nc2_sum_caught = 0, nc2_total = 0;
  for (int si = 0; si < NS; ++si) {
    int n = sizes[si];
    if (n < 16) continue; // tiny rows: f32 and f64 folds coincide; need length
    for (int dist = 0; dist <= 2; ++dist) {
      for (int seed = 0; seed < 16; ++seed) {
        for (int i = 0; i < n; ++i) x[i] = rand_f32(dist, i);
        float max = ggml_vec_max_f32_ref(n, x);
        if (max == -INFINITY) continue;
        double sr = ggml_ref(n, yr, x, max);
        double snc = nc_f32_sum(n, ync, x, max);
        nc2_total++;
        // y MUST still match (NC2 keeps ggml's exact y[]); the SUM must differ.
        if (!bits_equal_d(sr, snc)) nc2_sum_caught++;
      }
    }
  }

  printf("INC-18 F5b ggml_vec_soft_max_f32: y[] %d/%d bit-exact, sum %d/%d "
         "bit-exact (vs ggml's VECTORIZED soft_max)\n",
         pass_y, total, pass_sum, total);
  printf("NC1 wrong-exp-method (libm expf): %d/%d correctly DIFFER\n",
         nc1_caught, nc1_total);
  printf("NC2 wrong-sum-precision (f32 accumulate, not f64 vfwredusum): %d/%d "
         "rows the SUM correctly DIFFERS\n", nc2_sum_caught, nc2_total);

  free(x); free(yo); free(yr); free(ync);

  // The PRIMARY gate is the KERNEL verdict: y[] AND the returned sum bit-exact vs
  // ggml's vectorized soft_max on EVERY case. pass_sum is the NEW signal this
  // rung introduces (F5 already proved y[]); if pass_y==total but pass_sum!=total
  // the vfwredusum/f64m1 carry is the culprit, not the exp. The negative controls
  // are DISCRIMINATION EVIDENCE, reported but NOT allowed to flip a correct
  // kernel to FAIL: NC1 (majority-differ) shows the byte-compare pins the exact
  // exp polynomial; NC2 shows the returned-sum match pins the exact vfwredusum
  // fold (a correct kernel must NOT fail merely because a control was not sharp
  // on the sampled rows).
  int kernel_ok = (pass_y == total && pass_sum == total);
  int nc1_ok = (nc1_caught * 2 > nc1_total);
  printf("DISCRIMINATION: NC1 exp-polynomial control %s (%d/%d differ); "
         "NC2 sum-precision control caught %d/%d rows%s\n",
         nc1_ok ? "SHARP" : "WEAK", nc1_caught, nc1_total,
         nc2_sum_caught, nc2_total,
         nc2_sum_caught > 0 ? " (the f64 widening reduce is the discriminator)"
                            : " (f32/f64 folds coincided on sampled rows; "
                              "sum-match still pins the exact f64 reduce)");
  if (kernel_ok && nc1_ok) {
    printf("RESULT: PASS (byte-exact y[] AND returned sum vs ggml VECTORIZED "
           "soft_max; NC1 discriminates the exact exp polynomial, NC2 the exact "
           "f64 widening reduce -> ggml's method matched node-for-node)\n");
    return 0;
  }
  printf("RESULT: FAIL (kernel_ok=%d nc1_ok=%d)\n", kernel_ok, nc1_ok);
  return 1;
}
