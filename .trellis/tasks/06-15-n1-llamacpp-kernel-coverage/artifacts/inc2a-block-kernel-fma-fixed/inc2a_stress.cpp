// INC-2a ADVERSARIAL NUMERIC EDGE-STRESS harness (ssh rvv, -march=rv64gcv).
//
// Builds on inc2a_validate.cpp. Goal: hunt a BYTE-EXACTNESS counterexample for
// the compiler-emitted tcrv_rvv.q4_0_q8_0_block_dot kernel vs ggml's REAL RVV
// kernel AND ggml's _generic, using deliberately pathological inputs:
//   (a) fp16 scales: subnormal, near-max (65504), +0.0, -0.0, powers of two,
//       and the per-byte fp16 sweep (all 0..30 exponents).
//   (b) nibbles: all-0x00 (-8), all-0xFF (+7), all-0x88 (0), checkerboard,
//       plus an exhaustive single-byte sweep (all 256 values).
//   (c) q8: all +127, all -128, alternating.
//   (d) n = 32 (one block), n = 32*1000 (large fp32 accumulation rounding).
//   (e) mixed: random scales x extreme nibbles; cross-strip sumi (VLMAX=4 so the
//       inner strip loop runs 4x and carries sumi through the redsum seed).
// EVERY comparison is memcmp of the 32 raw float bits (no tolerance, no
// isnan/isinf skip). Trivial-zero cases are paired with non-trivial
// discriminators, and a negative control on a NEW data shape proves the
// bitwise check is non-vacuous.
//
// HONEST BOUND: fp32 inf/NaN is UNREACHABLE through valid fp16 block data --
// per-block magnitude <= 32768 * 65504 * 65504 ~= 1.4e14, ~1000 blocks ~=
// 1.4e17, which is ~21 orders below fp32 max (3.4e38). So the large-n cases
// stress LARGE-FINITE accumulation + fp32 rounding, NOT inf/NaN bit patterns.
// We print actual computed values (EVIDENCE lines) for the (d) cases.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <riscv_vector.h>

static const int QK = 32;
static const int Q4_STRIDE = 18;
static const int Q8_STRIDE = 34;

static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

extern "C" void
tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
    size_t n, float *s, size_t bs, const uint8_t *vx, size_t bx,
    const uint8_t *vy, size_t by, int32_t nrc);

// ---- ggml's REAL RVV kernel (transcribed from quants.c:222-271) --------------
static float ggml_real_rvv_s(int n, const uint8_t *vx, const uint8_t *vy) {
  const int qk = QK;
  const int nb = n / qk;
  float sumf = 0;
  size_t vl = qk / 2; // 16
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * Q4_STRIDE;
    const uint8_t *yb = vy + (size_t)ib * Q8_STRIDE;
    vuint8m1_t tx = __riscv_vle8_v_u8m1(xb + 2, vl);
    vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
    vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
    vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
    vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
    vint8m1_t x_ai = __riscv_vreinterpret_v_u8m1_i8m1(x_a);
    vint8m1_t x_li = __riscv_vreinterpret_v_u8m1_i8m1(x_l);
    vint8m1_t v0 = __riscv_vsub_vx_i8m1(x_ai, 8, vl);
    vint8m1_t v1 = __riscv_vsub_vx_i8m1(x_li, 8, vl);
    vint16m2_t vec_mul1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    vint16m2_t vec_mul2 = __riscv_vwmacc_vv_i16m2(vec_mul1, v1, y1, vl);
    vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(vec_mul2, vec_zero, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);
    uint16_t dx, dy;
    std::memcpy(&dx, xb, 2);
    std::memcpy(&dy, yb, 2);
    sumf += sumi * fp16_to_fp32(dx) * fp16_to_fp32(dy);
  }
  return sumf;
}

// ---- ggml's _generic reference (transcribed from quants.c:174-207) -----------
static float ggml_generic_s(int n, const uint8_t *vx, const uint8_t *vy) {
  const int qk = QK;
  const int nb = n / qk;
  float sumf = 0;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * Q4_STRIDE;
    const uint8_t *yb = vy + (size_t)ib * Q8_STRIDE;
    const uint8_t *xqs = xb + 2;
    const int8_t *yqs = (const int8_t *)(yb + 2);
    int sumi0 = 0, sumi1 = 0;
    for (int j = 0; j < qk / 2; ++j) {
      const int v0 = (xqs[j] & 0x0F) - 8;
      const int v1 = (xqs[j] >> 4) - 8;
      sumi0 += (v0 * yqs[j]);
      sumi1 += (v1 * yqs[j + qk / 2]);
    }
    int sumi = sumi0 + sumi1;
    uint16_t dx, dy;
    std::memcpy(&dx, xb, 2);
    std::memcpy(&dy, yb, 2);
    sumf += sumi * fp16_to_fp32(dx) * fp16_to_fp32(dy);
  }
  return sumf;
}

static unsigned g_rng = 0x2468abcdu;
static unsigned next_rand() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}
static uint16_t random_fp16_finite() {
  uint16_t sign = (next_rand() & 1) << 15;
  uint16_t exp = (uint16_t)(next_rand() % 31); // 0..30 (never 31 = Inf/NaN)
  uint16_t mant = (uint16_t)(next_rand() & 0x3FF);
  return sign | (exp << 10) | mant;
}

static int bits_equal(float a, float b) {
  uint32_t ba, bb;
  std::memcpy(&ba, &a, 4);
  std::memcpy(&bb, &b, 4);
  return ba == bb;
}
static uint32_t fbits(float a) { uint32_t b; std::memcpy(&b, &a, 4); return b; }

static long g_checked = 0;
static long g_failures = 0;
static long g_nonzero = 0; // count of cases whose result was a non-zero, finite,
                           // discriminating value (proves non-vacuity)

// Core comparison: ours vs BOTH refs, byte-exact. label/extra for repro.
// announce!=0 prints the computed value even on PASS (positive evidence).
static int compare_case_v(int n, const uint8_t *vx, const uint8_t *vy,
                          const char *label, int announce) {
  float s_ours = 0.0f;
  tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
      (size_t)n, &s_ours, 0, vx, 0, vy, 0, 1);
  float s_real = ggml_real_rvv_s(n, vx, vy);
  float s_generic = ggml_generic_s(n, vx, vy);
  int real_eq = bits_equal(s_ours, s_real);
  int generic_eq = bits_equal(s_ours, s_generic);
  if (announce)
    printf("EVIDENCE [%s] n=%d: ours=0x%08x (%.9g) [real_eq=%d generic_eq=%d]\n",
           label, n, fbits(s_ours), (double)s_ours, real_eq, generic_eq);
  ++g_checked;
  // track non-trivial discriminators (nonzero, not subnormal-degenerate)
  if (fbits(s_ours) != 0 && fbits(s_ours) != 0x80000000u) ++g_nonzero;
  if (real_eq && generic_eq) return 1;
  printf("FAIL [%s] n=%d: ours=0x%08x(%.9g) real=0x%08x(%.9g) "
         "generic=0x%08x(%.9g) real_eq=%d generic_eq=%d\n",
         label, n, fbits(s_ours), (double)s_ours, fbits(s_real),
         (double)s_real, fbits(s_generic), (double)s_generic, real_eq,
         generic_eq);
  ++g_failures;
  return 0;
}
static int compare_case(int n, const uint8_t *vx, const uint8_t *vy,
                        const char *label) {
  return compare_case_v(n, vx, vy, label, /*announce=*/0);
}

// Block fillers. Scales and nibble/q8 patterns chosen per case.
static void set_scale_x(uint8_t *vx, int nb, uint16_t d) {
  for (int ib = 0; ib < nb; ++ib) std::memcpy(vx + (size_t)ib * Q4_STRIDE, &d, 2);
}
static void set_scale_y(uint8_t *vy, int nb, uint16_t d) {
  for (int ib = 0; ib < nb; ++ib) std::memcpy(vy + (size_t)ib * Q8_STRIDE, &d, 2);
}
static void set_nibbles(uint8_t *vx, int nb, uint8_t byteval) {
  for (int ib = 0; ib < nb; ++ib)
    for (int i = 0; i < 16; ++i) vx[(size_t)ib * Q4_STRIDE + 2 + i] = byteval;
}
static void set_q8(uint8_t *vy, int nb, int8_t qv) {
  for (int ib = 0; ib < nb; ++ib)
    for (int i = 0; i < 32; ++i) vy[(size_t)ib * Q8_STRIDE + 2 + i] = (uint8_t)qv;
}
static void set_q8_checker(uint8_t *vy, int nb, int8_t a, int8_t b) {
  for (int ib = 0; ib < nb; ++ib)
    for (int i = 0; i < 32; ++i)
      vy[(size_t)ib * Q8_STRIDE + 2 + i] = (uint8_t)((i & 1) ? b : a);
}

// fp16 named constants
static const uint16_t FP16_POS_ZERO = 0x0000;
static const uint16_t FP16_NEG_ZERO = 0x8000;
static const uint16_t FP16_MIN_SUBNORMAL = 0x0001; // 2^-24
static const uint16_t FP16_MAX_SUBNORMAL = 0x03FF;
static const uint16_t FP16_MIN_NORMAL = 0x0400;    // 2^-14
static const uint16_t FP16_MAX = 0x7BFF;           // 65504
static const uint16_t FP16_ONE = 0x3C00;           // 1.0
static const uint16_t FP16_TWO = 0x4000;           // 2.0
static const uint16_t FP16_HALF = 0x3800;          // 0.5
static const uint16_t FP16_NEG_ONE = 0xBC00;       // -1.0

int main() {
  uint8_t *vx = nullptr;
  uint8_t *vy = nullptr;
  int cap_nb = 1000;
  vx = (uint8_t *)std::malloc((size_t)cap_nb * Q4_STRIDE);
  vy = (uint8_t *)std::malloc((size_t)cap_nb * Q8_STRIDE);

  printf("=== INC-2a adversarial numeric edge-stress ===\n");

  // -------- (a) scale stress: cross every notable scale_x x scale_y --------
  // Use extreme NON-zero nibbles + q8 so sumi != 0 (non-vacuous), and also a
  // mid pattern. Sweep over n in {32, 96, 320}.
  {
    const uint16_t scales[] = {
        FP16_POS_ZERO, FP16_NEG_ZERO, FP16_MIN_SUBNORMAL, FP16_MAX_SUBNORMAL,
        FP16_MIN_NORMAL, FP16_MAX, FP16_ONE, FP16_TWO, FP16_HALF, FP16_NEG_ONE,
        0x0002, 0x4900 /*10.0*/, 0x6000 /*512*/, 0x7000 /*8192*/};
    const int nscales = (int)(sizeof(scales) / sizeof(scales[0]));
    const int ns[] = {32, 96, 320};
    for (int si = 0; si < nscales; ++si) {
      for (int sj = 0; sj < nscales; ++sj) {
        for (int ni = 0; ni < 3; ++ni) {
          int n = ns[ni];
          int nb = n / QK;
          // nibbles: 0x07 -> low nibble 7->-1, high nibble 0->-8 (mixed signs)
          // gives a non-trivial sumi. q8 alternating +100/-77.
          set_nibbles(vx, nb, 0x07);
          set_q8_checker(vy, nb, 100, -77);
          set_scale_x(vx, nb, scales[si]);
          set_scale_y(vy, nb, scales[sj]);
          char lbl[64];
          snprintf(lbl, sizeof(lbl), "scale_x=0x%04x_y=0x%04x", scales[si],
                   scales[sj]);
          compare_case(n, vx, vy, lbl);
        }
      }
    }
  }

  // -------- (b) nibble stress: every nibble byte 0..255, n=32 --------
  // scale 1.0 x 1.0 so sumi maps straight through to sumf.
  {
    int n = 32, nb = 1;
    for (int bv = 0; bv < 256; ++bv) {
      set_nibbles(vx, nb, (uint8_t)bv);
      set_q8(vy, nb, 127);
      set_scale_x(vx, nb, FP16_ONE);
      set_scale_y(vy, nb, FP16_ONE);
      char lbl[48];
      snprintf(lbl, sizeof(lbl), "nibble=0x%02x_q8=+127", bv);
      compare_case(n, vx, vy, lbl);

      // also vs q8=-128 (worst-neg) and alternating
      set_q8(vy, nb, -128);
      snprintf(lbl, sizeof(lbl), "nibble=0x%02x_q8=-128", bv);
      compare_case(n, vx, vy, lbl);

      set_q8_checker(vy, nb, 127, -128);
      snprintf(lbl, sizeof(lbl), "nibble=0x%02x_q8=alt127/-128", bv);
      compare_case(n, vx, vy, lbl);
    }
  }

  // named nibble patterns x named q8 patterns at larger n
  {
    const uint8_t nibs[] = {0x00, 0xFF, 0x88, 0xAA, 0x55, 0x0F, 0xF0, 0x77};
    const int8_t q8v[] = {127, -128, 0, 1, -1};
    const int ns[] = {32, 64, 320, 32 * 31};
    for (size_t k = 0; k < sizeof(nibs); ++k) {
      for (size_t q = 0; q < sizeof(q8v); ++q) {
        for (int ni = 0; ni < 4; ++ni) {
          int n = ns[ni], nb = n / QK;
          set_nibbles(vx, nb, nibs[k]);
          set_q8(vy, nb, q8v[q]);
          set_scale_x(vx, nb, FP16_MAX); // push magnitude up
          set_scale_y(vy, nb, FP16_TWO);
          char lbl[64];
          snprintf(lbl, sizeof(lbl), "nib=0x%02x_q8=%d_dmax", nibs[k],
                   (int)q8v[q]);
          compare_case(n, vx, vy, lbl);
        }
      }
    }
  }

  // -------- (c) integer worst case: sumi=+32768/block --------
  // weights all 0x00 (both nibbles -> -8) x q8 all -128 -> per-block
  // sumi = 32 * (-8)*(-128) = 32768. Cross-strip accumulation (4 strips).
  {
    const int ns[] = {32, 64, 256, 1024, 32 * 1000};
    for (int ni = 0; ni < 5; ++ni) {
      int n = ns[ni], nb = n / QK;
      set_nibbles(vx, nb, 0x00);
      set_q8(vy, nb, -128);
      set_scale_x(vx, nb, FP16_ONE);
      set_scale_y(vy, nb, FP16_ONE);
      compare_case(n, vx, vy, "int_worst_+32768_per_block");
      // and all 0xFF (+7) x +127 -> per-block 32*7*127 = 28448
      set_nibbles(vx, nb, 0xFF);
      set_q8(vy, nb, 127);
      compare_case(n, vx, vy, "int_+7x+127_per_block");
    }
  }

  // -------- (d) fp32 accumulation rounding stress: large n, max scale --------
  // n = 32*1000 (1000 blocks). NOTE (honest bound): per-block magnitude is at
  // most sumi(32768) * 65504 * 65504 ~= 1.4e14, and 1000 blocks ~= 1.4e17 --
  // ~21 orders below fp32 max (3.4e38). So fp32 inf/NaN is UNREACHABLE through
  // valid fp16 block data; these cases stress LARGE-FINITE accumulation +
  // fp32 rounding, not inf/NaN. We print the actual computed value as positive
  // evidence (announce=1). memcmp still demands byte-exactness vs both refs.
  {
    int n = 32 * 1000, nb = n / QK;
    // case 1: large positive magnitude (sumi=+32768/block * dmax * dmax)
    set_nibbles(vx, nb, 0x00); // -8
    set_q8(vy, nb, -128);      // sumi = +32768/block
    set_scale_x(vx, nb, FP16_MAX);
    set_scale_y(vy, nb, FP16_MAX);
    compare_case_v(n, vx, vy, "bign_large_pos_~1.4e17", 1);
    // case 2: large negative magnitude
    set_q8(vy, nb, 127); // -8 * 127 -> negative sumi
    compare_case_v(n, vx, vy, "bign_large_neg", 1);
    // case 3: per-block sign alternation across equal block counts -> ~0 sum;
    // stresses massive-cancellation fp32 rounding (NOT inf-minus-inf NaN).
    set_nibbles(vx, nb, 0x00);
    set_q8(vy, nb, -128);
    set_scale_x(vx, nb, FP16_MAX);
    set_scale_y(vy, nb, FP16_MAX);
    for (int ib = 0; ib < nb; ++ib) {
      uint16_t d = (ib & 1) ? (uint16_t)(FP16_MAX | 0x8000) : FP16_MAX;
      std::memcpy(vx + (size_t)ib * Q4_STRIDE, &d, 2);
    }
    compare_case_v(n, vx, vy, "bign_sign_alternating_cancellation", 1);
    // case 4: max scale x large n with mid sumi (rounding accumulation)
    set_nibbles(vx, nb, 0x07);
    set_q8_checker(vy, nb, 90, -70);
    set_scale_x(vx, nb, FP16_MAX);
    set_scale_y(vy, nb, FP16_MAX);
    compare_case_v(n, vx, vy, "bign_mid_accumulation", 1);
  }

  // -------- (e) mixed: random scales x extreme/structured nibbles --------
  {
    const int ns[] = {32, 64, 256, 1024, 4096, 32 * 1000};
    for (int rep = 0; rep < 300; ++rep) {
      for (int ni = 0; ni < 6; ++ni) {
        int n = ns[ni], nb = n / QK;
        for (int ib = 0; ib < nb; ++ib) {
          uint16_t dx = random_fp16_finite();
          uint16_t dy = random_fp16_finite();
          std::memcpy(vx + (size_t)ib * Q4_STRIDE, &dx, 2);
          std::memcpy(vy + (size_t)ib * Q8_STRIDE, &dy, 2);
          // extreme/structured nibbles per block (cycle patterns)
          uint8_t nb_pat = (uint8_t)((ib * 37 + rep) & 0xFF);
          for (int i = 0; i < 16; ++i)
            vx[(size_t)ib * Q4_STRIDE + 2 + i] =
                (i & 1) ? nb_pat : (uint8_t)~nb_pat;
          for (int i = 0; i < 32; ++i) {
            // mix extremes and randoms
            int8_t qv;
            int sel = (i + ib + rep) % 4;
            if (sel == 0) qv = 127;
            else if (sel == 1) qv = -128;
            else if (sel == 2) qv = 0;
            else qv = (int8_t)(next_rand() & 0xFF);
            vy[(size_t)ib * Q8_STRIDE + 2 + i] = (uint8_t)qv;
          }
        }
        char lbl[48];
        snprintf(lbl, sizeof(lbl), "mixed_rep%d", rep);
        compare_case(n, vx, vy, lbl);
      }
    }
  }

  // -------- (f) full random (the original stress, more reps) --------
  {
    const int ns[] = {32, 64, 256, 1024, 4096};
    for (int rep = 0; rep < 500; ++rep) {
      for (int ni = 0; ni < 5; ++ni) {
        int n = ns[ni], nb = n / QK;
        for (int ib = 0; ib < nb; ++ib) {
          uint16_t dx = random_fp16_finite();
          uint16_t dy = random_fp16_finite();
          std::memcpy(vx + (size_t)ib * Q4_STRIDE, &dx, 2);
          std::memcpy(vy + (size_t)ib * Q8_STRIDE, &dy, 2);
          for (int i = 0; i < 16; ++i)
            vx[(size_t)ib * Q4_STRIDE + 2 + i] = (uint8_t)(next_rand() & 0xFF);
          for (int i = 0; i < 32; ++i)
            vy[(size_t)ib * Q8_STRIDE + 2 + i] = (uint8_t)(next_rand() & 0xFF);
        }
        compare_case(n, vx, vy, "fullrandom");
      }
    }
  }

  // -------- (g) EXPLICIT lane-ramp: per-lane distinct data within a block --
  // Human-auditable strip-offset guard. The 16-lane inner block runs in 4
  // strips of width VLMAX (=4 on this board), carrying sumi through the
  // vwredsum seed. If any strided load (x_qs+v21, y0+v21, y1+v21) used a wrong
  // lane offset, a distinct-per-lane fill diverges vs the SCALAR ggml_generic
  // ref (which indexes xqs[j]/yqs[j] explicitly). We make every nibble byte and
  // every q8 byte distinct across the 16/32 lanes, multi-block, mixed scales.
  {
    const int ns[] = {32, 64, 96, 320};
    for (int ni = 0; ni < 4; ++ni) {
      int n = ns[ni], nb = n / QK;
      for (int ib = 0; ib < nb; ++ib) {
        // distinct fp16 scales per block
        uint16_t dx = (uint16_t)(0x3000 + ib * 7);  // varied finite normals
        uint16_t dy = (uint16_t)(0x3800 + ib * 11);
        std::memcpy(vx + (size_t)ib * Q4_STRIDE, &dx, 2);
        std::memcpy(vy + (size_t)ib * Q8_STRIDE, &dy, 2);
        // nibble byte i = (i*17 + ib*3) -> distinct low/high nibbles per lane
        for (int i = 0; i < 16; ++i)
          vx[(size_t)ib * Q4_STRIDE + 2 + i] = (uint8_t)(i * 17 + ib * 3);
        // q8 lane i = i-16 (ramps -16..+15) for first half, then +/- ramp
        for (int i = 0; i < 32; ++i)
          vy[(size_t)ib * Q8_STRIDE + 2 + i] =
              (uint8_t)(int8_t)((i - 16) + ib * 5);
      }
      compare_case_v(n, vx, vy, "lane_ramp_distinct_per_lane", 1);
    }
  }

  // -------- (h) tiny/subnormal fp32 OUTPUT regime --------
  // Drive *s into the fp32 subnormal/near-zero band: smallest fp16 scales x
  // tiny sumi. fp32 subnormal threshold ~1.4e-45. sumi(=small) * 2^-24 * 2^-24
  // = small * 2^-48 ~ small * 3.5e-15; with single block and |sumi|~hundreds we
  // sit at ~1e-12 (normal fp32) -- to reach subnormals we need the min-subnormal
  // fp16 (2^-24) squared = 2^-48 times sumi. memcmp demands byte exactness even
  // here (rounding-to-subnormal is identical in all three impls by construction).
  {
    int n = 32, nb = 1;
    const uint16_t tiny_scales[] = {FP16_MIN_SUBNORMAL, FP16_MAX_SUBNORMAL,
                                    FP16_MIN_NORMAL, 0x0002, 0x0010};
    for (size_t a = 0; a < sizeof(tiny_scales) / sizeof(tiny_scales[0]); ++a) {
      for (size_t b = 0; b < sizeof(tiny_scales) / sizeof(tiny_scales[0]); ++b) {
        set_nibbles(vx, nb, 0x10);  // low=0->-8, high=1->-7 -> nonzero sumi
        set_q8(vy, nb, 1);
        set_scale_x(vx, nb, tiny_scales[a]);
        set_scale_y(vy, nb, tiny_scales[b]);
        char lbl[64];
        snprintf(lbl, sizeof(lbl), "tinyout_x=0x%04x_y=0x%04x", tiny_scales[a],
                 tiny_scales[b]);
        compare_case_v(n, vx, vy, lbl, 1);
      }
    }
  }

  // -------- NEGATIVE CONTROL on a NEW shape (n=320, structured) ----------
  // Prove the bitwise check is non-vacuous: a 1-bit scale flip MUST change *s,
  // and feeding ours(clean) vs ggml(perturbed) MUST report MISMATCH.
  {
    int n = 320, nb = n / QK;
    set_nibbles(vx, nb, 0x3C);          // low=12->+4, high=3->-5 (nonzero sumi)
    set_q8_checker(vy, nb, 90, -50);
    set_scale_x(vx, nb, 0x4900);        // 10.0
    set_scale_y(vy, nb, FP16_TWO);
    float s_clean = 0.0f;
    tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
        (size_t)n, &s_clean, 0, vx, 0, vy, 0, 1);
    // perturb block-0 q4 scale sign bit
    uint16_t dx;
    std::memcpy(&dx, vx, 2);
    dx ^= 0x8000u;
    std::memcpy(vx, &dx, 2);
    float s_pert_ref = ggml_real_rvv_s(n, vx, vy);
    float s_pert_ours = 0.0f;
    tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
        (size_t)n, &s_pert_ours, 0, vx, 0, vy, 0, 1);
    int discriminates = !bits_equal(s_clean, s_pert_ours);
    int still_tracks = bits_equal(s_pert_ours, s_pert_ref);
    int caught_mismatch = !bits_equal(s_clean, s_pert_ref);
    printf("\n--- negative control (n=320 structured) ---\n");
    printf("clean=%.9g pert_ours=%.9g pert_ref=%.9g\n", (double)s_clean,
           (double)s_pert_ours, (double)s_pert_ref);
    if (discriminates && still_tracks && caught_mismatch) {
      printf("NEGATIVE CONTROL PASS: flip changed *s, ours still tracks ggml on "
             "perturbed data, bitwise check REPORTS MISMATCH (non-vacuous).\n");
    } else {
      printf("NEGATIVE CONTROL FAIL: discriminates=%d still_tracks=%d "
             "caught_mismatch=%d\n",
             discriminates, still_tracks, caught_mismatch);
      ++g_failures;
    }
  }

  std::free(vx);
  std::free(vy);

  printf("\n=== STRESS SUMMARY: %ld cases checked, %ld non-trivial (nonzero) "
         "discriminating, %ld FAILURES ===\n",
         g_checked, g_nonzero, g_failures);
  if (g_failures == 0)
    printf("RESULT: PASS (bitwise-equal vs ggml REAL + _generic across ALL "
           "adversarial cases; negative control non-vacuous)\n");
  else
    printf("RESULT: FAIL\n");
  return g_failures == 0 ? 0 : 1;
}
