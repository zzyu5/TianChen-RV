
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
/* v_f4_a1: packed-i4 dot, LMUL=mf4 (i32 acc m1, 1 vregs/acc), A=1 accumulators. */
__attribute__((noinline)) void
v_f4_a1(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
     float scale, float *out, size_t n) {
  size_t vl0 = __riscv_vsetvlmax_e32m1();
  vint32m1_t vacc0 = __riscv_vmv_v_x_i32m1(0, vl0);
  size_t off = 0;
  size_t rem = n;
  while (rem > 0) {
    { size_t vl = __riscv_vsetvl_e8mf4(rem); vint8mf4_t lb = __riscv_vle8_v_i8mf4(lhs + off, vl); vint8mf4_t rb = __riscv_vle8_v_i8mf4(rhs + off, vl); vint8mf4_t l_lo = __riscv_vsra_vx_i8mf4(__riscv_vsll_vx_i8mf4(lb, 4, vl), 4, vl); vint8mf4_t r_lo = __riscv_vsra_vx_i8mf4(__riscv_vsll_vx_i8mf4(rb, 4, vl), 4, vl); vint8mf4_t l_hi = __riscv_vsra_vx_i8mf4(lb, 4, vl); vint8mf4_t r_hi = __riscv_vsra_vx_i8mf4(rb, 4, vl); vint16mf2_t p_lo = __riscv_vwmul_vv_i16mf2(l_lo, r_lo, vl); vint16mf2_t p_hi = __riscv_vwmul_vv_i16mf2(l_hi, r_hi, vl); vacc0 = __riscv_vwadd_wv_i32m1(vacc0, p_lo, vl); vacc0 = __riscv_vwadd_wv_i32m1(vacc0, p_hi, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
  }
  
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
  vint32m1_t vred = __riscv_vredsum_vs_i32m1_i32m1(vacc0, vzero, vl0);
  int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  out[0] = ((float)sum) * scale;
}
