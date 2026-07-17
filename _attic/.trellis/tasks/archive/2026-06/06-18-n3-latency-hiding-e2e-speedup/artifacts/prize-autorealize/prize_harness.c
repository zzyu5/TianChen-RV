/* PRIZE harness: prove tcrv-opt AUTO-EMITS the wide rung correctly + faster.
 *
 * Two kernels under test are BOTH auto-emitted by tcrv-opt from the SAME kernel
 * MLIR, differing ONLY by the architectural vector-register-budget resource fact
 * the realizer reads:
 *   - ON  (wide):   budget 32 -> selector picks i32m8 -> deferred-wide body
 *   - OFF (narrow): budget 16 -> selector prunes m8 -> legacy grouped body
 * Both carry the SAME compiler-emitted symbol name; the OFF object is symbol-
 * renamed at compile (-D) so they link side by side.
 *
 *  (1) byte-exact: ON output vs the scalar _generic oracle over many n -> MUST
 *      match bit-for-bit on the integer reduction (I5/I7, non-negotiable).
 *      OFF is also checked (it is the legacy correct path).
 *  (2) same-compiler ON / OFF: best-of-N timing -> the headline speedup that
 *      proves the COMPILER (not a hand kernel) produces the win.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef void (*kfn)(const int8_t *, const int8_t *, const int32_t *, float,
                    float *, size_t);

/* The kernel objects + scalar ref all export C (unmangled) symbols; this TU is
   compiled as C++ (the EmitC emitter wraps bodies in extern "C"), so the
   declarations need C linkage to resolve. */
#ifdef __cplusplus
extern "C" {
#endif
extern void ref_scalar_byte(const int8_t *, const int8_t *, const int32_t *,
                            float, float *, size_t);
/* ON keeps its compiler-emitted name; OFF is renamed via -DTCRV_OFF_NAME=... */
extern void tcrv_emitc_budget_divergence_kernel_budget_divergence_rvv(
    const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
extern void TCRV_OFF_NAME(const int8_t *, const int8_t *, const int32_t *,
                          float, float *, size_t);
#ifdef __cplusplus
}
#endif

static uint32_t rs = 0x12345678u;
static int8_t r8(void) { rs = rs * 1664525u + 1013904223u; return (int8_t)((rs >> 17) & 0xFF); }
static double ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec * 1e9 + t.tv_nsec; }

#define WARM 3
#define REP 9
#define IT 16

int main(void) {
  const size_t SZ[] = {1, 7, 31, 64, 127, 257, 1024, 4096, 16384, 65536};
  const int NSZ = (int)(sizeof(SZ) / sizeof(SZ[0]));
  float scale = 0.0125f;
  double sink = 0;
  int all_exact = 1;

  printf("# PRIZE: tcrv-opt auto-emitted ON(wide i32m8) vs OFF(narrow), byte-exact + ON/OFF speedup\n");
  for (int s = 0; s < NSZ; ++s) {
    size_t n = SZ[s];
    int8_t *l = (int8_t *)malloc(n ? n : 1);
    int8_t *r = (int8_t *)malloc(n ? n : 1);
    int32_t acc[1] = {-7};
    float o_ref = 0, o_on = 0, o_off = 0;
    for (size_t i = 0; i < n; ++i) { l[i] = r8(); r[i] = r8(); }

    ref_scalar_byte(l, r, acc, scale, &o_ref, n);
    tcrv_emitc_budget_divergence_kernel_budget_divergence_rvv(l, r, acc, scale, &o_on, n);
    TCRV_OFF_NAME(l, r, acc, scale, &o_off, n);

    /* byte-exact: the reduction is integer-exact; the *scale is the same f32 op
       in all three, so identical inputs -> bit-identical float. Compare bits. */
    uint32_t b_ref, b_on, b_off;
    __builtin_memcpy(&b_ref, &o_ref, 4);
    __builtin_memcpy(&b_on, &o_on, 4);
    __builtin_memcpy(&b_off, &o_off, 4);
    int on_exact = (b_on == b_ref);
    int off_exact = (b_off == b_ref);
    if (!on_exact || !off_exact) all_exact = 0;
    printf("BYTEEXACT n=%-6zu ref=%.7g ON=%.7g OFF=%.7g  ON_exact=%d OFF_exact=%d\n",
           n, o_ref, o_on, o_off, on_exact, off_exact);

    /* timing: best-of-REP of IT-iter blocks, warmup first. */
    if (n >= 31) {
      kfn fns[2] = {tcrv_emitc_budget_divergence_kernel_budget_divergence_rvv, TCRV_OFF_NAME};
      double best[2] = {1e30, 1e30};
      for (int v = 0; v < 2; ++v) {
        float out = 0;
        for (int w = 0; w < WARM; ++w)
          for (int it = 0; it < IT; ++it) { fns[v](l, r, acc, scale, &out, n); sink += out; }
        for (int rp = 0; rp < REP; ++rp) {
          double t0 = ns();
          for (int it = 0; it < IT; ++it) { fns[v](l, r, acc, scale, &out, n); sink += out; }
          double per = (ns() - t0) / IT;
          if (per < best[v]) best[v] = per;
        }
      }
      printf("SPEEDUP n=%-6zu ON_ns=%.3f OFF_ns=%.3f  ON_over_OFF=%.3fx\n",
             n, best[0], best[1], best[1] / best[0]);
    }
    free(l); free(r);
  }
  printf("SINK=%g\n", sink);
  printf("VERDICT byte_exact_all=%s\n", all_exact ? "PASS" : "FAIL");
  return all_exact ? 0 : 2;
}
