
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
/* v_m1_a2: packed-i4 dot, LMUL=m1 (i32 acc m4, 4 vregs/acc), A=2 accumulators. */
__attribute__((noinline)) void
v_m1_a2(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
     float scale, float *out, size_t n) {
  size_t vl0 = __riscv_vsetvlmax_e32m4();
  vint32m4_t vacc0 = __riscv_vmv_v_x_i32m4(0, vl0);
  vint32m4_t vacc1 = __riscv_vmv_v_x_i32m4(0, vl0);
  size_t off = 0;
  size_t rem = n;
  while (rem > 0) {
    { size_t vl = __riscv_vsetvl_e8m1(rem); vint8m1_t lb = __riscv_vle8_v_i8m1(lhs + off, vl); vint8m1_t rb = __riscv_vle8_v_i8m1(rhs + off, vl); vint8m1_t l_lo = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(lb, 4, vl), 4, vl); vint8m1_t r_lo = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(rb, 4, vl), 4, vl); vint8m1_t l_hi = __riscv_vsra_vx_i8m1(lb, 4, vl); vint8m1_t r_hi = __riscv_vsra_vx_i8m1(rb, 4, vl); vint16m2_t p_lo = __riscv_vwmul_vv_i16m2(l_lo, r_lo, vl); vint16m2_t p_hi = __riscv_vwmul_vv_i16m2(l_hi, r_hi, vl); vacc0 = __riscv_vwadd_wv_i32m4(vacc0, p_lo, vl); vacc0 = __riscv_vwadd_wv_i32m4(vacc0, p_hi, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
    { size_t vl = __riscv_vsetvl_e8m1(rem); vint8m1_t lb = __riscv_vle8_v_i8m1(lhs + off, vl); vint8m1_t rb = __riscv_vle8_v_i8m1(rhs + off, vl); vint8m1_t l_lo = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(lb, 4, vl), 4, vl); vint8m1_t r_lo = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(rb, 4, vl), 4, vl); vint8m1_t l_hi = __riscv_vsra_vx_i8m1(lb, 4, vl); vint8m1_t r_hi = __riscv_vsra_vx_i8m1(rb, 4, vl); vint16m2_t p_lo = __riscv_vwmul_vv_i16m2(l_lo, r_lo, vl); vint16m2_t p_hi = __riscv_vwmul_vv_i16m2(l_hi, r_hi, vl); vacc1 = __riscv_vwadd_wv_i32m4(vacc1, p_lo, vl); vacc1 = __riscv_vwadd_wv_i32m4(vacc1, p_hi, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
  }
  vacc0 = __riscv_vadd_vv_i32m4(vacc0, vacc1, vl0);
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
  vint32m1_t vred = __riscv_vredsum_vs_i32m4_i32m1(vacc0, vzero, vl0);
  int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  out[0] = ((float)sum) * scale;
}
