/* P-B8 RVV variants for the i16 dot-reduce e2e 灯 measure. -march=rv64gcv.
 * The "wide-deferred" column here is the VERBATIM AUTOMATIC e2e COMPILER OUTPUT
 * (selector-driven: kernel -> --tcrv-rvv-materialize-gearbox-schedules ->
 * --tcrv-materialize-selected-lowering-boundaries [the resource-aware selector
 * picks the wide i32m8 rung] -> --tcrv-rvv-lower-to-emitc -> C). NOT hand-fed
 * IR (P-B7 lowered a HAND-AUTHORED typed body; P-B8 produces that same body
 * AUTOMATICALLY from a plain dot-reduce kernel via the autotuner). So this
 * measures the actual end-to-end compiler output's performance (I8). The other
 * two columns (per-iter-reduce = the compiler's CURRENT dot-reduce emission;
 * narrow-deferred = the competent narrow naive) are unchanged from P-B7. */
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

/* (1) per-iter-reduce: the algorithm the COMPILER EMITS TODAY for the narrow
 * dot-reduce (a vredsum into a running seed EVERY iteration, latency-bound). */
__attribute__((noinline)) void
dot_per_iter_reduce(const int16_t *lhs, const int16_t *rhs, const int32_t *acc,
                    int32_t *out, size_t n) {
  int32_t sum = acc[0];
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16mf2(n - i);
    vint16mf2_t a = __riscv_vle16_v_i16mf2(lhs + i, vl);
    vint16mf2_t b = __riscv_vle16_v_i16mf2(rhs + i, vl);
    vint32m1_t p = __riscv_vwmul_vv_i32m1(a, b, vl);
    vint32m1_t seed = __riscv_vmv_v_x_i32m1(sum, 1);
    vint32m1_t red = __riscv_vredsum_vs_i32m1_i32m1(p, seed, vl);
    sum = __riscv_vmv_x_s_i32m1_i32(red);
  }
  out[0] = sum;
}

/* (2) narrow-deferred: the competent narrow naive (one persistent i32m1 acc,
 * vadd.vv, one trailing vredsum; i16mf2 source -> under-vectorized). */
__attribute__((noinline)) void
dot_narrow_deferred(const int16_t *lhs, const int16_t *rhs, const int32_t *acc,
                    int32_t *out, size_t n) {
  size_t vlmax = __riscv_vsetvlmax_e32m1();
  vint32m1_t vacc = __riscv_vmv_v_x_i32m1(0, vlmax);
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16mf2(n - i);
    vint16mf2_t a = __riscv_vle16_v_i16mf2(lhs + i, vl);
    vint16mf2_t b = __riscv_vle16_v_i16mf2(rhs + i, vl);
    vint32m1_t p = __riscv_vwmul_vv_i32m1(a, b, vl);
    vacc = __riscv_vadd_vv_i32m1(vacc, p, vl);
  }
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, vlmax);
  vint32m1_t vred = __riscv_vredsum_vs_i32m1_i32m1(vacc, vzero, vlmax);
  out[0] = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
}

/* (3) wide-deferred = the AUTOMATIC e2e COMPILER-EMITTED body, VERBATIM (defined
 * in automatic_emitted_body.cpp). dot_wide_deferred is a thin forwarding wrapper
 * so the timed column is the actual compiler output (the wrapper is noinline + a
 * single tail call; it adds no vector work). */
extern void tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv(
    const int16_t *, const int16_t *, const int32_t *, int32_t *, size_t);

__attribute__((noinline)) void
dot_wide_deferred(const int16_t *lhs, const int16_t *rhs, const int32_t *acc,
                  int32_t *out, size_t n) {
  tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv(
      lhs, rhs, acc, out, n);
}
