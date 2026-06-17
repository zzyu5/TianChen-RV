
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
/* v_f2_a2: int8 dot, LMUL=mf2 (i32 acc m2, 2 vregs/acc), A=2 accumulators. */
__attribute__((noinline)) void
v_f2_a2(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
     float scale, float *out, size_t n) {
  size_t vl0 = __riscv_vsetvlmax_e32m2();
  vint32m2_t vacc0 = __riscv_vmv_v_x_i32m2(0, vl0);
  vint32m2_t vacc1 = __riscv_vmv_v_x_i32m2(0, vl0);
  size_t off = 0;
  size_t rem = n;
  while (rem > 0) {
    { size_t vl = __riscv_vsetvl_e8mf2(rem); vint8mf2_t a = __riscv_vle8_v_i8mf2(lhs + off, vl); vint8mf2_t b = __riscv_vle8_v_i8mf2(rhs + off, vl); vint16m1_t p = __riscv_vwmul_vv_i16m1(a, b, vl); vacc0 = __riscv_vwadd_wv_i32m2(vacc0, p, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
    { size_t vl = __riscv_vsetvl_e8mf2(rem); vint8mf2_t a = __riscv_vle8_v_i8mf2(lhs + off, vl); vint8mf2_t b = __riscv_vle8_v_i8mf2(rhs + off, vl); vint16m1_t p = __riscv_vwmul_vv_i16m1(a, b, vl); vacc1 = __riscv_vwadd_wv_i32m2(vacc1, p, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
  }
  vacc0 = __riscv_vadd_vv_i32m2(vacc0, vacc1, vl0);
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
  /* reduce the (possibly wide) i32 accumulator to a scalar */
  vint32m1_t vred = __riscv_vredsum_vs_i32m2_i32m1(vacc0, vzero, vl0);
  int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  out[0] = ((float)sum) * scale;
}
