/* RVV0.7 (xtheadvector / C920) variant of lamp_automatic_rvv.c.
 *
 * WHY THIS FILE EXISTS (honest-ledger, read before trusting numbers):
 *   The verbatim RVV1.0 harness (lamp_automatic_rvv.c, sha256 4917c4f0...) does
 *   NOT compile under -march=rv64gc_xtheadvector: its two HAND-AUTHORED naive
 *   baselines (dot_per_iter_reduce, dot_narrow_deferred) use FRACTIONAL-LMUL
 *   i16mf2 source operands, and RVV0.7.1 declares ZERO fractional-LMUL types
 *   (rejected: __riscv_vsetvl_e16mf2, vint16mf2_t, __riscv_vle16_v_i16mf2,
 *   __riscv_vwmul_vv_i32m1 — captured verbatim in rvv_verbatim.err). That
 *   rejection is itself an RVV0.7 datum: RVV1.0 narrow-baseline code is
 *   non-portable across the ISA generation.
 *
 *   To complete the ablation, the two naive baselines are LMUL-FLOORED mf2 -> m1
 *   (the competent-naive choice the RVV0.7 m1-floor selection policy mandates on
 *   a fractional-LMUL-free board): source i16m1, widened product/acc i32m2 EMUL,
 *   trailing reduce i32m2 -> i32m1 (unchanged target). At VLEN=128 this is 8
 *   elts/strip — a STRONGER naive baseline than the RVV1.0 mf2 one (4 elts/strip),
 *   i.e. the tune is compared against the best naive THIS ISA can express.
 *
 *   CRITICAL: the wide-deferred column (variant 3) is UNCHANGED — it forwards to
 *   the byte-identical tcrv-opt compiler output in automatic_emitted_body.cpp
 *   (sha256 ff5ba5b1...). Only the two hand-authored reference variants here are
 *   edited. lamp_driver.c (scalar oracle + driver + timing) is also unchanged.
 *   The harness's built-in scalar-oracle correctness gate runs on all 4 variants;
 *   a bug in the m1 rewrite would print MISMATCH and exit non-zero.
 */
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

/* (1) per-iter-reduce, RVV0.7 m1-floor: vredsum into a running seed EVERY
 * iteration (latency-bound). mf2->m1 source: vsetvl_e16m1 / vle16_v_i16m1 /
 * vwmul_vv_i32m2 (m1*2 EMUL) / vredsum_vs_i32m2_i32m1. */
__attribute__((noinline)) void
dot_per_iter_reduce(const int16_t *lhs, const int16_t *rhs, const int32_t *acc,
                    int32_t *out, size_t n) {
  int32_t sum = acc[0];
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m1(n - i);
    vint16m1_t a = __riscv_vle16_v_i16m1(lhs + i, vl);
    vint16m1_t b = __riscv_vle16_v_i16m1(rhs + i, vl);
    vint32m2_t p = __riscv_vwmul_vv_i32m2(a, b, vl);
    vint32m1_t seed = __riscv_vmv_v_x_i32m1(sum, 1);
    vint32m1_t red = __riscv_vredsum_vs_i32m2_i32m1(p, seed, vl);
    sum = __riscv_vmv_x_s_i32m1_i32(red);
  }
  out[0] = sum;
}

/* (2) narrow-deferred, RVV0.7 m1-floor: one persistent i32m2 acc, vadd.vv, one
 * trailing vredsum. i16m1 source -> "naive done right" at the m1 floor (8
 * elts/strip @ VLEN=128). */
__attribute__((noinline)) void
dot_narrow_deferred(const int16_t *lhs, const int16_t *rhs, const int32_t *acc,
                    int32_t *out, size_t n) {
  size_t vlmax = __riscv_vsetvlmax_e32m2();
  vint32m2_t vacc = __riscv_vmv_v_x_i32m2(0, vlmax);
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m1(n - i);
    vint16m1_t a = __riscv_vle16_v_i16m1(lhs + i, vl);
    vint16m1_t b = __riscv_vle16_v_i16m1(rhs + i, vl);
    vint32m2_t p = __riscv_vwmul_vv_i32m2(a, b, vl);
    vacc = __riscv_vadd_vv_i32m2(vacc, p, vl);
  }
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
  vint32m1_t vred = __riscv_vredsum_vs_i32m2_i32m1(vacc, vzero, vlmax);
  out[0] = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
}

/* (3) wide-deferred = the AUTOMATIC e2e COMPILER-EMITTED body, VERBATIM (defined
 * in automatic_emitted_body.cpp, byte-identical to the RVV1.0 run). This column
 * is the actual tcrv-opt output; the thin forwarding wrapper adds no vector
 * work. UNCHANGED from the RVV1.0 harness. */
extern void tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv(
    const int16_t *, const int16_t *, const int32_t *, int32_t *, size_t);

__attribute__((noinline)) void
dot_wide_deferred(const int16_t *lhs, const int16_t *rhs, const int32_t *acc,
                  int32_t *out, size_t n) {
  tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv(
      lhs, rhs, acc, out, n);
}
