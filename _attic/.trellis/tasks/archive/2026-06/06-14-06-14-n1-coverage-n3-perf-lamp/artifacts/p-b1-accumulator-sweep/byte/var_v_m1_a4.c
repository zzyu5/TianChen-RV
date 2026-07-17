
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
/* v_m1_a4: int8 dot, LMUL=m1 (i32 acc m4, 4 vregs/acc), A=4 accumulators. */
__attribute__((noinline)) void
v_m1_a4(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
     float scale, float *out, size_t n) {
  size_t vl0 = __riscv_vsetvlmax_e32m4();
  vint32m4_t vacc0 = __riscv_vmv_v_x_i32m4(0, vl0);
  vint32m4_t vacc1 = __riscv_vmv_v_x_i32m4(0, vl0);
  vint32m4_t vacc2 = __riscv_vmv_v_x_i32m4(0, vl0);
  vint32m4_t vacc3 = __riscv_vmv_v_x_i32m4(0, vl0);
  size_t off = 0;
  size_t rem = n;
  while (rem > 0) {
    { size_t vl = __riscv_vsetvl_e8m1(rem); vint8m1_t a = __riscv_vle8_v_i8m1(lhs + off, vl); vint8m1_t b = __riscv_vle8_v_i8m1(rhs + off, vl); vint16m2_t p = __riscv_vwmul_vv_i16m2(a, b, vl); vacc0 = __riscv_vwadd_wv_i32m4(vacc0, p, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
    { size_t vl = __riscv_vsetvl_e8m1(rem); vint8m1_t a = __riscv_vle8_v_i8m1(lhs + off, vl); vint8m1_t b = __riscv_vle8_v_i8m1(rhs + off, vl); vint16m2_t p = __riscv_vwmul_vv_i16m2(a, b, vl); vacc1 = __riscv_vwadd_wv_i32m4(vacc1, p, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
    { size_t vl = __riscv_vsetvl_e8m1(rem); vint8m1_t a = __riscv_vle8_v_i8m1(lhs + off, vl); vint8m1_t b = __riscv_vle8_v_i8m1(rhs + off, vl); vint16m2_t p = __riscv_vwmul_vv_i16m2(a, b, vl); vacc2 = __riscv_vwadd_wv_i32m4(vacc2, p, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
    { size_t vl = __riscv_vsetvl_e8m1(rem); vint8m1_t a = __riscv_vle8_v_i8m1(lhs + off, vl); vint8m1_t b = __riscv_vle8_v_i8m1(rhs + off, vl); vint16m2_t p = __riscv_vwmul_vv_i16m2(a, b, vl); vacc3 = __riscv_vwadd_wv_i32m4(vacc3, p, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
  }
  vacc0 = __riscv_vadd_vv_i32m4(vacc0, vacc1, vl0);
  vacc0 = __riscv_vadd_vv_i32m4(vacc0, vacc2, vl0);
  vacc0 = __riscv_vadd_vv_i32m4(vacc0, vacc3, vl0);
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
  /* reduce the (possibly wide) i32 accumulator to a scalar */
  vint32m1_t vred = __riscv_vredsum_vs_i32m4_i32m1(vacc0, vzero, vl0);
  int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  out[0] = ((float)sum) * scale;
}
