// q8_0_verify_driver.c -- on-device (ssh rvv) numerical + perf harness for the
// M-FLAT typed q8_0 x q8_0 flat block-dot kernel.
//
// It links against the REAL exported RISC-V relocatable object
//   tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot
// (front-door -> materialize-emission-plans -> tcrv-translate
//  --tcrv-export-target-artifact), calls it on hardware, and:
//   1. VERIFY: quantizes random data to q8_0, compares the kernel result
//      bit-for-bit against a hand-written scalar q8_0 vec_dot that uses ggml's
//      fold order. Two reference fold variants are computed (non-contracted and
//      explicit fmaf) so the kernel's actual float-contraction choice is
//      DISCOVERED empirically, not assumed. Reports exact-match count + worst ULP.
//   2. PERF: times the kernel vs the scalar reference in the same binary. When
//      built -march=rv64gcv the reference is clang-autovectorized (autovec
//      baseline); when built -march=rv64gc it is pure scalar (scalar baseline).
//
// The kernel ABI (from the emission-plan runtime_abi_parameters, 9 roles):
//   (size_t n, float *s, size_t bs, const uint8_t *vx, size_t bx,
//    const uint8_t *vy, size_t by, int32_t nrc, const int32_t *zero_seed)
// Only n, s, vx, vy are consumed by the emitted body; bs/bx/by/nrc are strides
// for the multi-row (nrc>1) case and zero_seed is a reduce-seed pointer the body
// currently overrides with a literal-0 splat. We pass valid harmless values.
//
// REUSE for the next flat格 (q4_0/q4_1/q5_0/q5_1): copy this file, change
//   - KERNEL_SYM (extern symbol),
//   - the block struct + BLOCK_BYTES + quantize_block(),
//   - ref_block_sumi() (the integer decode, e.g. nibble unpack + offset-binary),
// and drive it with run_ondevice_verify.sh (which stays unchanged).
//
// Board identity is printed at runtime (VLEN, __riscv_vlenb). Compiler + march
// provenance is passed in by the orchestration script and echoed.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// riscv_vector.h intrinsics only compile when V is enabled (-march=...v). The
// precompiled kernel .o still runs its V instructions at runtime regardless of
// how this driver TU was compiled, so under a no-V (rv64gc) scalar-baseline
// build we simply skip the on-device VLEN probe.
#if defined(__riscv_vector)
#include <riscv_vector.h>
#define HAVE_RVV_INTRINSICS 1
#else
#define HAVE_RVV_INTRINSICS 0
#endif

// ---- q8_0 format ----------------------------------------------------------
#define QK 32
#define BLOCK_BYTES 34 // 2-byte fp16 scale + 32 int8

typedef struct __attribute__((packed)) {
  uint16_t d;      // fp16 scale (raw bits)
  int8_t qs[QK];   // 32 signed int8 quants
} block_q8_0;

// ---- fp16 <-> fp32 (pure-integer IEEE, exact) -----------------------------
// IMPORTANT: do NOT use the `_Float16` C type here. On this board
// (openEuler/clang-17, -march=rv64gcv WITHOUT the Zfh half-float extension)
// clang-17 miscompiles `_Float16` conversions -- every value comes out wrong
// (verified: 0x3c00 did not convert to 1.0). The kernel .o is fine because its
// `(float)*(const _Float16 *)` idiom lowers to a single correct __extendhfsf2
// libcall. We must match the kernel's read-back exactly, so we do fp16->fp32
// with integer bit manipulation (exact/lossless for every fp16). The
// fp32->fp16 direction only has to yield valid fp16 bytes -- both the kernel
// and this reference then read those SAME stored bytes identically, so any
// correct rounding keeps the comparison bit-exact.
static float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t man = h & 0x3ffu;
  uint32_t out;
  if (exp == 0) {
    if (man == 0) {
      out = sign; // +/-0
    } else {      // subnormal -> normalize
      int e = 127 - 15 + 1;
      while (!(man & 0x400u)) {
        man <<= 1;
        e--;
      }
      man &= 0x3ffu;
      out = sign | ((uint32_t)e << 23) | (man << 13);
    }
  } else if (exp == 0x1f) {
    out = sign | 0x7f800000u | (man << 13); // inf / nan
  } else {
    out = sign | ((exp - 15 + 127) << 23) | (man << 13); // normal
  }
  float f;
  memcpy(&f, &out, sizeof f);
  return f;
}
static uint16_t fp32_to_fp16(float f) {
  uint32_t x;
  memcpy(&x, &f, sizeof x);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t e = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
  uint32_t man = x & 0x7fffffu;
  if (((x >> 23) & 0xffu) == 0xff)
    return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u)); // inf / nan
  if (e >= 0x1f)
    return (uint16_t)(sign | 0x7c00u); // overflow -> inf
  if (e <= 0) {                        // subnormal / zero
    if (e < -10)
      return (uint16_t)sign;
    man |= 0x800000u; // restore implicit 1
    int sh = 14 - e;
    uint32_t h = man >> sh;
    uint32_t rem = man & ((1u << sh) - 1u);
    uint32_t half = 1u << (sh - 1);
    if (rem > half || (rem == half && (h & 1u)))
      h++; // round to nearest even
    return (uint16_t)(sign | h);
  }
  uint16_t h = (uint16_t)(sign | ((uint32_t)e << 10) | (man >> 13)); // normal
  uint32_t rem = man & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u)))
    h++; // round to nearest even (carry into exp is handled by the +1)
  return h;
}

// ---- q8_0 quantizer (matches ggml quantize_row_q8_0 reference math) --------
static void quantize_block(const float *x, block_q8_0 *b) {
  float amax = 0.0f;
  for (int j = 0; j < QK; j++) {
    float a = fabsf(x[j]);
    if (a > amax)
      amax = a;
  }
  float d = amax / 127.0f;
  float id = (d != 0.0f) ? 1.0f / d : 0.0f;
  b->d = fp32_to_fp16(d);
  for (int j = 0; j < QK; j++) {
    b->qs[j] = (int8_t)lroundf(x[j] * id);
  }
}

// Integer core of one block (shared by scalar reference + autovec baseline).
// Reads the int8 quants straight from the stored struct bytes.
static int32_t ref_block_sumi(const block_q8_0 *x, const block_q8_0 *y) {
  int32_t sumi = 0;
  for (int j = 0; j < QK; j++)
    sumi += (int32_t)x->qs[j] * (int32_t)y->qs[j];
  return sumi;
}

// Compiler barrier: force the optimizer to assume memory changed, so the pure
// scalar reference is genuinely recomputed each timed iteration (not hoisted /
// CSE'd out -- which would give a fake ~4ns baseline).
#define CLOBBER() __asm__ __volatile__("" ::: "memory")

// Scalar reference vec_dot. use_fma selects the float fold contraction form.
// The fp16 scales are read back from the SAME stored struct bytes the kernel
// dereferences (never a kept pre-storage float), so both paths round-trip fp16.
__attribute__((noinline)) static float
ref_vec_dot(size_t n, const block_q8_0 *x, const block_q8_0 *y, int use_fma) {
  size_t nb = n / QK;
  float sumf = 0.0f;
  for (size_t i = 0; i < nb; i++) {
    int32_t sumi = ref_block_sumi(&x[i], &y[i]);
    float dx = fp16_to_fp32(x[i].d);
    float dy = fp16_to_fp32(y[i].d);
    float dxdy = dx * dy; // d_x * d_y first, matching the kernel's emitc.expression
    if (use_fma)
      sumf = fmaf((float)sumi, dxdy, sumf);
    else
      sumf = sumf + (float)sumi * dxdy;
  }
  return sumf;
}

// ---- kernel under test (real exported .o) ---------------------------------
extern void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(
    size_t n, float *s, size_t bs, const uint8_t *vx, size_t bx,
    const uint8_t *vy, size_t by, int32_t nrc, const int32_t *zero_seed);

static const int32_t g_zero_seed = 0;

static void call_kernel(size_t n, float *s, const block_q8_0 *x,
                        const block_q8_0 *y) {
  tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(
      n, s, /*bs=*/0, (const uint8_t *)x, /*bx=*/0, (const uint8_t *)y,
      /*by=*/0, /*nrc=*/1, &g_zero_seed);
}

// ---- helpers --------------------------------------------------------------
static uint32_t f2u(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
// Monotonic total-order key: adjacent representable floats differ by 1.
static int64_t f2key(float f) {
  uint32_t u = f2u(f);
  return (u & 0x80000000u) ? -(int64_t)(u & 0x7fffffffu) : (int64_t)u;
}
static uint64_t ulp_dist(float a, float b) {
  int64_t d = f2key(a) - f2key(b);
  return d < 0 ? (uint64_t)(-d) : (uint64_t)d;
}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts); // matches gate4 harness methodology
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_random(float *v, size_t n, unsigned *seed) {
  for (size_t i = 0; i < n; i++) {
    // range ~[-4,4]. (Does NOT produce all-zero blocks, so the quantizer's
    // id==0 guard is present but not exercised by this generator.)
    float r = (float)((int)(rand_r(seed) % 2001) - 1000) / 250.0f;
    v[i] = r;
  }
}

int main(int argc, char **argv) {
  size_t n = 4096;         // elements per dot product (must be multiple of 32)
  int trials = 256;        // independent random dot products for verification
  long perf_iters = 200000; // kernel calls timed for perf
  const char *march = (argc > 1) ? argv[1] : "unknown";
  if (argc > 2) n = (size_t)strtoul(argv[2], NULL, 10);
  if (argc > 3) trials = atoi(argv[3]);
  if (argc > 4) perf_iters = atol(argv[4]);
  if (n % QK)
    n -= (n % QK);

  size_t nb = n / QK;

  // ---- board identity ----
#if HAVE_RVV_INTRINSICS
  size_t vlenb = __riscv_vlenb();          // bytes per vector register
  size_t vlmax_e8m2 = __riscv_vsetvlmax_e8m2();
#else
  size_t vlenb = 0, vlmax_e8m2 = 0; // no-V build: probe skipped, kernel .o still runs V
#endif
  printf("== board identity ==\n");
  printf("VLEN(bits)=%zu  vlenb=%zu  VLMAX_e8m2=%zu  march=%s  driver_has_rvv_intrinsics=%d\n",
         vlenb * 8, vlenb, vlmax_e8m2, march, HAVE_RVV_INTRINSICS);
  printf("n=%zu blocks/dot=%zu QK=%d BLOCK_BYTES=%d sizeof(block_q8_0)=%zu\n", n,
         nb, QK, BLOCK_BYTES, sizeof(block_q8_0));
  if (sizeof(block_q8_0) != BLOCK_BYTES) {
    printf("FATAL: block struct not %d bytes\n", BLOCK_BYTES);
    return 2;
  }
  if (HAVE_RVV_INTRINSICS && vlmax_e8m2 < QK) {
    printf("WARNING: VLMAX_e8m2=%zu < QK=%d -- one vwredsum cannot cover a full "
           "block on this VLEN; bit-exact WILL fail (this is the m2 anchor "
           "assumption being violated, not a kernel logic bug)\n",
           vlmax_e8m2, QK);
  }

  float *fx = malloc(n * sizeof(float));
  float *fy = malloc(n * sizeof(float));
  block_q8_0 *bx = malloc(nb * sizeof(block_q8_0));
  block_q8_0 *by = malloc(nb * sizeof(block_q8_0));
  if (!fx || !fy || !bx || !by) {
    printf("FATAL: alloc\n");
    return 2;
  }

  // ---- VERIFY ----
  printf("\n== numerical verify (%d trials, n=%zu) ==\n", trials, n);
  unsigned seed = 0x1234567u;
  int match_nofma = 0, match_fma = 0;
  uint64_t worst_ulp_nofma = 0, worst_ulp_fma = 0;
  float sample_k = 0, sample_r = 0;
  for (int t = 0; t < trials; t++) {
    fill_random(fx, n, &seed);
    fill_random(fy, n, &seed);
    for (size_t i = 0; i < nb; i++) {
      quantize_block(&fx[i * QK], &bx[i]);
      quantize_block(&fy[i * QK], &by[i]);
    }
    float s_kernel = 12345.0f;
    call_kernel(n, &s_kernel, bx, by);
    float r_nofma = ref_vec_dot(n, bx, by, /*use_fma=*/0);
    float r_fma = ref_vec_dot(n, bx, by, /*use_fma=*/1);
    if (f2u(s_kernel) == f2u(r_nofma))
      match_nofma++;
    if (f2u(s_kernel) == f2u(r_fma))
      match_fma++;
    uint64_t u0 = ulp_dist(s_kernel, r_nofma);
    uint64_t u1 = ulp_dist(s_kernel, r_fma);
    if (u0 > worst_ulp_nofma)
      worst_ulp_nofma = u0;
    if (u1 > worst_ulp_fma)
      worst_ulp_fma = u1;
    if (t == 0) {
      sample_k = s_kernel;
      sample_r = r_nofma;
    }
  }
  printf("sample[0]: kernel=%.9g (0x%08x)  ref_nofma=%.9g (0x%08x)\n", sample_k,
         f2u(sample_k), sample_r, f2u(sample_r));
  printf("exact-match vs ref(no-contract): %d/%d   worst ULP=%llu\n",
         match_nofma, trials, (unsigned long long)worst_ulp_nofma);
  printf("exact-match vs ref(fmaf fold)  : %d/%d   worst ULP=%llu\n", match_fma,
         trials, (unsigned long long)worst_ulp_fma);
  int bitexact = (match_nofma == trials) || (match_fma == trials);
  printf("VERDICT: %s (kernel fold form = %s)\n",
         bitexact ? "BIT-EXACT" : "NOT bit-exact",
         match_fma == trials ? "fused-multiply-add"
                             : (match_nofma == trials ? "separate mul+add"
                                                      : "neither/indeterminate"));

  // ---- PERF ----
  // one fixed random dataset, many timed calls. volatile sink defeats DCE.
  fill_random(fx, n, &seed);
  fill_random(fy, n, &seed);
  for (size_t i = 0; i < nb; i++) {
    quantize_block(&fx[i * QK], &bx[i]);
    quantize_block(&fy[i * QK], &by[i]);
  }
  volatile float sink = 0.0f;
  float s = 0;
  const int REPEATS = 7; // best-of, to reject scheduler noise (gate4 methodology)

  // warm up both paths
  for (int i = 0; i < 2000; i++) {
    call_kernel(n, &s, bx, by);
    sink += s;
    sink += ref_vec_dot(n, bx, by, 0);
  }

  double best_kernel = 1e30, best_ref = 1e30;
  for (int r = 0; r < REPEATS; r++) {
    double t0 = now_s();
    for (long i = 0; i < perf_iters; i++) {
      CLOBBER();
      call_kernel(n, &s, bx, by);
      sink += s;
    }
    double tk = now_s() - t0;
    if (tk < best_kernel)
      best_kernel = tk;

    double t1 = now_s();
    for (long i = 0; i < perf_iters; i++) {
      CLOBBER(); // defeat hoist/CSE so the baseline is really recomputed
      sink += ref_vec_dot(n, bx, by, 0);
    }
    double tr = now_s() - t1;
    if (tr < best_ref)
      best_ref = tr;
  }

  double ns_kernel = best_kernel * 1e9 / (double)perf_iters;
  double ns_ref = best_ref * 1e9 / (double)perf_iters;
  // bytes touched per call: two quantized operands
  double bytes = 2.0 * (double)nb * (double)BLOCK_BYTES;
  double gbps_kernel = bytes / (ns_kernel);         // bytes/ns == GB/s
  double gbps_ref = bytes / (ns_ref);

  printf("\n== perf (%ld iters, n=%zu, march=%s) ==\n", perf_iters, n, march);
  printf("kernel   : %.1f ns/call   %.2f GB/s\n", ns_kernel, gbps_kernel);
  printf("reference: %.1f ns/call   %.2f GB/s   (baseline for this march)\n",
         ns_ref, gbps_ref);
  printf("speedup kernel vs reference: %.2fx\n", ns_ref / ns_kernel);
  printf("(sink=%.6g)\n", (double)sink); // keep sink live
  printf("NOTE: kernel microbench, NOT e2e decode; kernel .o = local cross-clang "
         "-O2 -march=rv64gcv; baseline = on-board clang -O2 -march=%s.\n",
         march);

  free(fx);
  free(fy);
  free(bx);
  free(by);
  return bitexact ? 0 : 1;
}
