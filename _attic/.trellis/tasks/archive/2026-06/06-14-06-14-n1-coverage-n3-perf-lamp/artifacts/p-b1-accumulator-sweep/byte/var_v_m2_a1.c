
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
/* v_m2_a1: int8 dot, LMUL=m2 (i32 acc m8, 8 vregs/acc), A=1 accumulators. */
__attribute__((noinline)) void
v_m2_a1(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
     float scale, float *out, size_t n) {
  size_t vl0 = __riscv_vsetvlmax_e32m8();
  vint32m8_t vacc0 = __riscv_vmv_v_x_i32m8(0, vl0);
  size_t off = 0;
  size_t rem = n;
  while (rem > 0) {
    { size_t vl = __riscv_vsetvl_e8m2(rem); vint8m2_t a = __riscv_vle8_v_i8m2(lhs + off, vl); vint8m2_t b = __riscv_vle8_v_i8m2(rhs + off, vl); vint16m4_t p = __riscv_vwmul_vv_i16m4(a, b, vl); vacc0 = __riscv_vwadd_wv_i32m8(vacc0, p, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
  }
  
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
  /* reduce the (possibly wide) i32 accumulator to a scalar */
  vint32m1_t vred = __riscv_vredsum_vs_i32m8_i32m1(vacc0, vzero, vl0);
  int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  out[0] = ((float)sum) * scale;
}
