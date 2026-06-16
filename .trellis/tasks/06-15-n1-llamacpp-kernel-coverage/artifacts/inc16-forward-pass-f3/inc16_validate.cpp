// INC-16 F3 — ggml rms_norm (first forward-pass REDUCTION op) bit-exact HW
// validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.ggml_rms_norm_f32 op
// (the row rms_norm: a SCALAR-DOUBLE Sum x[i]^2 fold -> scalar 1/sqrtf(mean+eps)
// -> a vectorized vle32/vfmul_vf/vse32 normalize strip) computes the SAME f32
// output row as ggml's OWN ggml_compute_forward_rms_norm_f32 (non-fused, no
// weight; ops.cpp:3758-3817), BIT-FOR-BIT (memcmp of the float bits), over many
// row sizes (incl. the real llama-2-7b dims {32,256,1024,4096,11008}) x value
// distributions (normal/large/small/mixed-sign) x eps values + named edge cases.
//
// The crux is the DOUBLE accumulation. ggml folds Sum x[i]^2 in ggml_float
// (= double), SCALAR, in strict ASCENDING index order; the product x[i]*x[i]
// rounds in f32 FIRST, is then widened to double, then accumulated in double.
// TWO negative controls prove that method is LOAD-BEARING (not just a tolerance):
//   (NC1) a WRONG-PRODUCT-TYPE ref ((double)x*(double)x) -- diverges on a long
//         row with ONE huge spike among unit values (x[0]=5e19f, x[1..]=1.0,
//         n>=64): the f32 product of the spike overflows to +inf -> mean=inf ->
//         scale=0 -> y all 0; the double product keeps the spike FINITE -> mean
//         finite-below-FLT_MAX -> scale finite -> y nonzero -> different bits.
//         (An ALL-large row does NOT discriminate: there the double-product mean
//         itself overflows the float cast to +inf, so both collapse to scale=0
//         identically -- which is also exactly why our kernel matches ggml there.)
//   (NC2) a WRONG-ACCUMULATOR-TYPE ref (f32 running sum) -- diverges on long
//         mixed-magnitude rows (ne00 = 1024 .. 11008). This is the control that
//         most directly proves the CRUX: the double *accumulation*.
// The verdict GATES on NC2 (the crux control); NC1 is bonus rigor on the
// product-type distinction. If our kernel matches ggml bit-for-bit while NC2
// DIFFERS, we have matched ggml's accumulation METHOD, not merely its tolerance.
//
// REFERENCE: a verbatim transcription of ggml's rms_norm row body
// (ops.cpp:3791-3817), since rms_norm has NO RVV path in ggml (the Sum is a
// scalar ggml_float loop) -- a faithful transcription is equivalent and simpler,
// exactly as F1 transcribed ggml's RVV scale path.
//
// The kernel under test is the UNMODIFIED, compiler-emitted C in
// rms_norm_kernel.cpp -- every line tagged source_op=tcrv_rvv.ggml_rms_norm_f32,
// emitted by tcrv-opt --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp
// from the committed rvv-to-emitc-ggml-rms-norm-f32.mlir.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <riscv_vector.h>

typedef double ggml_float; // ggml/src/ggml-cpu/vec.h:15

// ---- The kernel our compiler emitted (declared; defined in rms_norm_kernel.cpp)
// ABI as exported: (size_t ne00, const float *x, float *y, float eps).
extern "C" void
tcrv_emitc_ggml_rms_norm_f32_kernel_ggml_rms_norm_f32(size_t ne00,
                                                      const float *x, float *y,
                                                      float eps);

static void our_kernel(int ne00, const float *x, float *y, float eps) {
  tcrv_emitc_ggml_rms_norm_f32_kernel_ggml_rms_norm_f32((size_t)ne00, x, y, eps);
}

// ---- ggml's REAL rms_norm row body (ops.cpp:3791-3817, non-fused/no-weight) --
// Verbatim: ggml_float(double) scalar ascending Sum x[i]^2 (the f32 product
// widened to double), mean = sum/ne00 cast to float, scale = 1/sqrtf(mean+eps),
// then memcpy + ggml_vec_scale_f32 (= y[i] = x[i]*scale per lane).
static void ggml_ref(int ne00, const float *x, float *y, float eps) {
  ggml_float sum = 0.0;
  for (int i00 = 0; i00 < ne00; i00++) {
    sum += (ggml_float)(x[i00] * x[i00]);
  }
  const float mean = sum / ne00;
  const float scale = 1.0f / sqrtf(mean + eps);
  for (int i = 0; i < ne00; i++) {
    y[i] = x[i] * scale;
  }
}

// ---- NEGATIVE CONTROL 1: WRONG PRODUCT TYPE (double*double, not f32 product) -
static void nc_wrong_product(int ne00, const float *x, float *y, float eps) {
  ggml_float sum = 0.0;
  for (int i00 = 0; i00 < ne00; i00++) {
    sum += (ggml_float)x[i00] * (ggml_float)x[i00]; // WRONG: f64 product, no f32 round
  }
  const float mean = sum / ne00;
  const float scale = 1.0f / sqrtf(mean + eps);
  for (int i = 0; i < ne00; i++)
    y[i] = x[i] * scale;
}

// ---- NEGATIVE CONTROL 2: WRONG ACCUMULATOR TYPE (f32 running sum) -----------
static void nc_wrong_accum(int ne00, const float *x, float *y, float eps) {
  float sum = 0.0f; // WRONG: f32 accumulator instead of double
  for (int i00 = 0; i00 < ne00; i00++)
    sum += x[i00] * x[i00];
  const float mean = sum / ne00;
  const float scale = 1.0f / sqrtf(mean + eps);
  for (int i = 0; i < ne00; i++)
    y[i] = x[i] * scale;
}

static int bits_equal(const float *a, const float *b, int n) {
  return std::memcmp(a, b, (size_t)n * sizeof(float)) == 0;
}

// xorshift PRNG for reproducible f32 bit patterns.
static uint32_t rng_state = 0x2468abcdu;
static uint32_t xs32() {
  uint32_t x = rng_state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  rng_state = x;
  return x;
}
// dist: 0 normal[-2,2], 1 large[-1e20,1e20], 2 small[-1e-20,1e-20],
//       3 mixed-magnitude, 4 raw bits (denormal/inf/nan/huge occasionally).
static float rand_f32(int dist) {
  uint32_t r = xs32();
  float u = ((float)(int32_t)r / (float)0x7fffffff); // [-1,1)
  switch (dist) {
  case 0: return u * 2.0f;
  case 1: return u * 1e20f;
  case 2: return u * 1e-20f;
  case 3: {
    // mixed: a few huge spikes among small values (the worst case for an f32
    // accumulator -- proves NC2 diverges).
    if ((r & 0x1F) == 0) return u * 1e15f;
    return u * 1e-3f;
  }
  default: {
    if ((r & 0xF) == 0) {
      uint32_t bits = xs32();
      float f; std::memcpy(&f, &bits, 4); return f;
    }
    return u * 100.0f;
  }
  }
}

int main() {
  // The real llama-2-7b row dims + small/odd/tail sizes around them.
  const int sizes[] = {1, 7, 31, 32, 33, 63, 256, 255, 257, 1023, 1024, 1025,
                       4096, 4095, 11008, 11007};
  const float epss[] = {1e-5f, 1e-6f, 0.0f, 1e-3f, 1e-1f};
  const int NS = (int)(sizeof(sizes) / sizeof(sizes[0]));
  const int NE = (int)(sizeof(epss) / sizeof(epss[0]));

  const int MAXN = 11008;
  float *x = (float *)malloc(sizeof(float) * MAXN);
  float *yo = (float *)malloc(sizeof(float) * MAXN);
  float *yr = (float *)malloc(sizeof(float) * MAXN);

  int total = 0, pass = 0;

  // 1) random rows x distributions x eps x seeds, bit-exact vs ggml.
  for (int si = 0; si < NS; ++si) {
    int n = sizes[si];
    for (int dist = 0; dist <= 4; ++dist) {
      for (int ei = 0; ei < NE; ++ei) {
        float eps = epss[ei];
        for (int seed = 0; seed < 12; ++seed) {
          for (int i = 0; i < n; ++i) x[i] = rand_f32(dist);
          our_kernel(n, x, yo, eps);
          ggml_ref(n, x, yr, eps);
          int ok = bits_equal(yo, yr, n);
          total++; pass += ok;
          if (!ok) {
            printf("MISMATCH n=%d dist=%d eps=%a seed=%d\n", n, dist,
                   (double)eps, seed);
            for (int i = 0; i < n && i < 6; ++i) {
              uint32_t bo, br;
              std::memcpy(&bo, &yo[i], 4); std::memcpy(&br, &yr[i], 4);
              printf("  [%d] x=%a our=%08x ggml=%08x\n", i, (double)x[i], bo, br);
            }
          }
        }
      }
    }
  }

  // 2) named edge-case rows (inf/nan/denormal/+-0/huge mixed in a row).
  auto bit = [](uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; };
  float edge_row[] = {
    1.0f, -1.0f, 0.0f, bit(0x80000000) /*-0*/, bit(0x00000001) /*denormal*/,
    1e20f, -1e20f, 1e-20f, 3.14159f, -2.5f, 100.0f, 0.001f,
    bit(0x7f7fffff) /*FLT_MAX*/, 0.5f, -0.5f, 2.0f,
  };
  int ne = (int)(sizeof(edge_row) / sizeof(edge_row[0]));
  for (int ei = 0; ei < NE; ++ei) {
    float eps = epss[ei];
    our_kernel(ne, edge_row, yo, eps);
    ggml_ref(ne, edge_row, yr, eps);
    int ok = bits_equal(yo, yr, ne);
    total++; pass += ok;
    if (!ok) printf("EDGE MISMATCH eps=%a\n", (double)eps);
  }

  // 3) NEGATIVE CONTROLS: each WRONG-method ref must DIFFER from ours on at
  // least the cases where the double-accumulation is load-bearing.
  int nc1_caught = 0, nc1_total = 0; // wrong product type (spike-in-long-row)
  int nc2_caught = 0, nc2_total = 0; // wrong accumulator (long mixed magnitude)
  int nc2_det_caught = 0, nc2_det_total = 0; // the DETERMINISTIC swamping subset
  for (int si = 0; si < NS; ++si) {
    int n = sizes[si];
    // NC1: a long row (n>=64) with ONE huge spike among unit values. The f32
    // product of the spike overflows to inf (-> scale 0 -> y all 0); the double
    // product keeps it finite (-> finite scale -> y nonzero) -> different bits.
    if (n >= 64) {
      for (int i = 0; i < n; ++i) x[i] = 1.0f;
      x[0] = 5e19f; // spike: 5e19^2 = 2.5e39 > FLT_MAX in f32, finite in f64
      our_kernel(n, x, yo, 1e-5f);
      nc_wrong_product(n, x, yr, 1e-5f);
      nc1_total++;
      if (!bits_equal(yo, yr, n)) nc1_caught++;
    }
    // NC2: long mixed-magnitude rows where the f32 accumulator loses bits. THIS
    // is the crux control (the double accumulation).
    if (n >= 1024) {
      // (a) a DETERMINISTIC swamping row: one big value then many tiny ones whose
      // f32 partial sum drops bits the f64 sum keeps (x[0]=1e4, x[1..]=1e-3).
      // 1e8 + ~n*1e-6: the 1e-6 increments fall below the f32 ULP of ~1e8
      // (~8 -> the relative ULP near 1e8 is ~8, so a 1e-6 add is swallowed in
      // f32 but accumulates in f64), so the f32 sum != f64 sum -> scale differs.
      {
        // x[0]=1e4 -> square 1e8 (exactly representable; ULP there is 8). The
        // tail x[i]=1.9 -> square 3.61 < ULP/2=4, so in f32 EVERY tail add to a
        // running sum near 1e8 rounds away (the sum stays pinned at 1e8), while
        // in f64 the tail accumulates to ~(n-1)*3.61. The f32-vs-f64 sums diverge
        // by many ULP at n>=1024 -> different mean -> different scale -> different
        // y bits. A CONSTRUCTED case, not a seed-dependent coin flip.
        for (int i = 0; i < n; ++i) x[i] = 1.9f;
        x[0] = 1e4f;
        our_kernel(n, x, yo, 0.0f);
        nc_wrong_accum(n, x, yr, 0.0f);
        nc2_total++;
        nc2_det_total++;
        if (!bits_equal(yo, yr, n)) { nc2_caught++; nc2_det_caught++; }
      }
      // (b) randomized mixed-magnitude rows.
      for (int seed = 0; seed < 8; ++seed) {
        for (int i = 0; i < n; ++i) x[i] = rand_f32(3); // mixed magnitude
        our_kernel(n, x, yo, 1e-5f);
        nc_wrong_accum(n, x, yr, 1e-5f);
        nc2_total++;
        if (!bits_equal(yo, yr, n)) nc2_caught++;
      }
    }
  }

  printf("INC-16 F3 ggml_rms_norm_f32: %d/%d bit-exact cases PASS\n", pass,
         total);
  printf("NC1 wrong-product-type (large-magnitude): %d/%d correctly DIFFER\n",
         nc1_caught, nc1_total);
  printf("NC2 wrong-accumulator-type (long mixed):  %d/%d correctly DIFFER "
         "(of which the DETERMINISTIC swamping subset: %d/%d)\n",
         nc2_caught, nc2_total, nc2_det_caught, nc2_det_total);

  free(x); free(yo); free(yr);
  // The verdict GATES on NC2 (majority) -- the control that directly proves the
  // CRUX (the double ACCUMULATION) and, because the reference is a transcription,
  // ALSO validates the reference: if both kernel and reference secretly used f32
  // accumulation, NC2 (which IS f32 accumulation) would catch 0/N. It catches a
  // majority -> the reference is genuinely double-accum, the kernel matches it
  // everywhere (4805/4805) -> the kernel is genuinely double-accum. The
  // DETERMINISTIC swamping subset (a constructed 1e4-spike + 1.9 tail row) is
  // reported as additional evidence that the discrimination is not seed-luck.
  // NC1 (the product-type distinction) is reported as bonus rigor.
  int nc2_ok = (nc2_caught * 2 > nc2_total);
  if (pass == total && nc2_ok) {
    printf("RESULT: PASS (byte-exact vs ggml rms_norm; NC2 the scalar-double "
           "ACCUMULATION control discriminates -> ggml's METHOD is matched, not "
           "just its tolerance)\n");
    return 0;
  }
  printf("RESULT: FAIL\n");
  return 1;
}
