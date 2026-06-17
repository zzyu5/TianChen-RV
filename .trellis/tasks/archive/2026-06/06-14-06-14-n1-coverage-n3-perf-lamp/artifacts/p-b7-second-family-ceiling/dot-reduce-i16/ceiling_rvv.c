/* RVV variants for the i16 dot-reduce ceiling probe. -march=rv64gcv. */
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

/* (1) per-iter-reduce: the algorithm the COMPILER EMITS TODAY for this family.
 * Each iteration widens the product (vwmul i16mf2 -> i32m1) and folds it into a
 * scalar running sum via a vredsum over a running seed. A vredsum per iteration
 * is the latency-bound anti-pattern (cross-lane reduction on the critical path).
 */
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

/* (2) narrow-deferred: the "naive done right" at narrow LMUL. Widen the product
 * (vwmul i16mf2 -> i32m1) and accumulate it into a PERSISTENT i32m1 vector
 * accumulator with a NON-widening vadd.vv (product width == acc width), then do
 * ONE trailing vredsum after the loop. Competent, but i16mf2 is under-vectorized
 * on a 128-bit board (it processes only 4 i16 per strip). */
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

/* (3) wide-deferred (the TUNED candidate): widen the product at the widest legal
 * source LMUL (vwmul i16m4 -> i32m8) into a PERSISTENT i32m8 vector accumulator
 * via a NON-widening vadd.vv, then ONE trailing vredsum i32m8 -> i32m1.
 * Resource fact: i16 source m4 (4 vregs) -> i32 product==acc m8 (8 vregs); the
 * next rung (i16m8 -> i32m16) does not exist (LMUL caps at 8), so m4/m8 is the
 * widest legal rung. */
__attribute__((noinline)) void
dot_wide_deferred(const int16_t *lhs, const int16_t *rhs, const int32_t *acc,
                  int32_t *out, size_t n) {
  size_t vlmax8 = __riscv_vsetvlmax_e32m8();
  vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vlmax8);
  size_t off = 0, rem = n;
  while (rem > 0) {
    size_t vl = __riscv_vsetvl_e16m4(rem);
    vint16m4_t a = __riscv_vle16_v_i16m4(lhs + off, vl);
    vint16m4_t b = __riscv_vle16_v_i16m4(rhs + off, vl);
    vint32m8_t p = __riscv_vwmul_vv_i32m8(a, b, vl);
    vacc = __riscv_vadd_vv_i32m8(vacc, p, vl);
    off += vl;
    rem = (n > off) ? (n - off) : 0;
  }
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
  vint32m1_t vred = __riscv_vredsum_vs_i32m8_i32m1(vacc, vzero, vlmax8);
  out[0] = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
}
