
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
__attribute__((noinline)) void
naive_anchor_byte(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
                  float scale, float *out, size_t n) {
  size_t vl;
  size_t vlmax = __riscv_vsetvlmax_e32m1();
  vint32m1_t vacc = __riscv_vmv_v_x_i32m1(0, vlmax);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8mf4(n - i);
    vint8mf4_t vl8 = __riscv_vle8_v_i8mf4(lhs + i, vl);
    vint8mf4_t vr8 = __riscv_vle8_v_i8mf4(rhs + i, vl);
    vint16mf2_t vprod = __riscv_vwmul_vv_i16mf2(vl8, vr8, vl);
    vacc = __riscv_vwadd_wv_i32m1(vacc, vprod, vl);
  }
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, vlmax);
  vint32m1_t vred = __riscv_vredsum_vs_i32m1_i32m1(vacc, vzero, vlmax);
  int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  out[0] = ((float)sum) * scale;
}
