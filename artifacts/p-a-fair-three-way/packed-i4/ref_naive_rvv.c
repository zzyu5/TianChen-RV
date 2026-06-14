
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
__attribute__((noinline)) void
naive_rvv_packed_i4(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
                    float scale, float *out, size_t n) {
  size_t vl;
  size_t vlmax = __riscv_vsetvlmax_e32m1();
  vint32m1_t vacc = __riscv_vmv_v_x_i32m1(0, vlmax);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8mf4(n - i);
    vint8mf4_t lb = __riscv_vle8_v_i8mf4(lhs + i, vl);
    vint8mf4_t rb = __riscv_vle8_v_i8mf4(rhs + i, vl);
    /* sign-extend the low nibble: (b << 4) >>(arith) 4 */
    vint8mf4_t l_lo = __riscv_vsra_vx_i8mf4(__riscv_vsll_vx_i8mf4(lb, 4, vl), 4, vl);
    vint8mf4_t r_lo = __riscv_vsra_vx_i8mf4(__riscv_vsll_vx_i8mf4(rb, 4, vl), 4, vl);
    /* sign-extend the high nibble: b >>(arith) 4 */
    vint8mf4_t l_hi = __riscv_vsra_vx_i8mf4(lb, 4, vl);
    vint8mf4_t r_hi = __riscv_vsra_vx_i8mf4(rb, 4, vl);
    vint16mf2_t p_lo = __riscv_vwmul_vv_i16mf2(l_lo, r_lo, vl);
    vint16mf2_t p_hi = __riscv_vwmul_vv_i16mf2(l_hi, r_hi, vl);
    vacc = __riscv_vwadd_wv_i32m1(vacc, p_lo, vl);
    vacc = __riscv_vwadd_wv_i32m1(vacc, p_hi, vl);
  }
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, vlmax);
  vint32m1_t vred = __riscv_vredsum_vs_i32m1_i32m1(vacc, vzero, vlmax);
  int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  out[0] = ((float)sum) * scale;
}
