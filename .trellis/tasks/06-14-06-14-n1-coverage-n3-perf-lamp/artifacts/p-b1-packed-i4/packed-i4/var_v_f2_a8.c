
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
/* v_f2_a8: packed-i4 dot, LMUL=mf2 (i32 acc m2, 2 vregs/acc), A=8 accumulators. */
__attribute__((noinline)) void
v_f2_a8(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
     float scale, float *out, size_t n) {
  size_t vl0 = __riscv_vsetvlmax_e32m2();
  vint32m2_t vacc0 = __riscv_vmv_v_x_i32m2(0, vl0);
  vint32m2_t vacc1 = __riscv_vmv_v_x_i32m2(0, vl0);
  vint32m2_t vacc2 = __riscv_vmv_v_x_i32m2(0, vl0);
  vint32m2_t vacc3 = __riscv_vmv_v_x_i32m2(0, vl0);
  vint32m2_t vacc4 = __riscv_vmv_v_x_i32m2(0, vl0);
  vint32m2_t vacc5 = __riscv_vmv_v_x_i32m2(0, vl0);
  vint32m2_t vacc6 = __riscv_vmv_v_x_i32m2(0, vl0);
  vint32m2_t vacc7 = __riscv_vmv_v_x_i32m2(0, vl0);
  size_t off = 0;
  size_t rem = n;
  while (rem > 0) {
    { size_t vl = __riscv_vsetvl_e8mf2(rem); vint8mf2_t lb = __riscv_vle8_v_i8mf2(lhs + off, vl); vint8mf2_t rb = __riscv_vle8_v_i8mf2(rhs + off, vl); vint8mf2_t l_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(lb, 4, vl), 4, vl); vint8mf2_t r_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(rb, 4, vl), 4, vl); vint8mf2_t l_hi = __riscv_vsra_vx_i8mf2(lb, 4, vl); vint8mf2_t r_hi = __riscv_vsra_vx_i8mf2(rb, 4, vl); vint16m1_t p_lo = __riscv_vwmul_vv_i16m1(l_lo, r_lo, vl); vint16m1_t p_hi = __riscv_vwmul_vv_i16m1(l_hi, r_hi, vl); vacc0 = __riscv_vwadd_wv_i32m2(vacc0, p_lo, vl); vacc0 = __riscv_vwadd_wv_i32m2(vacc0, p_hi, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
    { size_t vl = __riscv_vsetvl_e8mf2(rem); vint8mf2_t lb = __riscv_vle8_v_i8mf2(lhs + off, vl); vint8mf2_t rb = __riscv_vle8_v_i8mf2(rhs + off, vl); vint8mf2_t l_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(lb, 4, vl), 4, vl); vint8mf2_t r_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(rb, 4, vl), 4, vl); vint8mf2_t l_hi = __riscv_vsra_vx_i8mf2(lb, 4, vl); vint8mf2_t r_hi = __riscv_vsra_vx_i8mf2(rb, 4, vl); vint16m1_t p_lo = __riscv_vwmul_vv_i16m1(l_lo, r_lo, vl); vint16m1_t p_hi = __riscv_vwmul_vv_i16m1(l_hi, r_hi, vl); vacc1 = __riscv_vwadd_wv_i32m2(vacc1, p_lo, vl); vacc1 = __riscv_vwadd_wv_i32m2(vacc1, p_hi, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
    { size_t vl = __riscv_vsetvl_e8mf2(rem); vint8mf2_t lb = __riscv_vle8_v_i8mf2(lhs + off, vl); vint8mf2_t rb = __riscv_vle8_v_i8mf2(rhs + off, vl); vint8mf2_t l_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(lb, 4, vl), 4, vl); vint8mf2_t r_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(rb, 4, vl), 4, vl); vint8mf2_t l_hi = __riscv_vsra_vx_i8mf2(lb, 4, vl); vint8mf2_t r_hi = __riscv_vsra_vx_i8mf2(rb, 4, vl); vint16m1_t p_lo = __riscv_vwmul_vv_i16m1(l_lo, r_lo, vl); vint16m1_t p_hi = __riscv_vwmul_vv_i16m1(l_hi, r_hi, vl); vacc2 = __riscv_vwadd_wv_i32m2(vacc2, p_lo, vl); vacc2 = __riscv_vwadd_wv_i32m2(vacc2, p_hi, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
    { size_t vl = __riscv_vsetvl_e8mf2(rem); vint8mf2_t lb = __riscv_vle8_v_i8mf2(lhs + off, vl); vint8mf2_t rb = __riscv_vle8_v_i8mf2(rhs + off, vl); vint8mf2_t l_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(lb, 4, vl), 4, vl); vint8mf2_t r_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(rb, 4, vl), 4, vl); vint8mf2_t l_hi = __riscv_vsra_vx_i8mf2(lb, 4, vl); vint8mf2_t r_hi = __riscv_vsra_vx_i8mf2(rb, 4, vl); vint16m1_t p_lo = __riscv_vwmul_vv_i16m1(l_lo, r_lo, vl); vint16m1_t p_hi = __riscv_vwmul_vv_i16m1(l_hi, r_hi, vl); vacc3 = __riscv_vwadd_wv_i32m2(vacc3, p_lo, vl); vacc3 = __riscv_vwadd_wv_i32m2(vacc3, p_hi, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
    { size_t vl = __riscv_vsetvl_e8mf2(rem); vint8mf2_t lb = __riscv_vle8_v_i8mf2(lhs + off, vl); vint8mf2_t rb = __riscv_vle8_v_i8mf2(rhs + off, vl); vint8mf2_t l_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(lb, 4, vl), 4, vl); vint8mf2_t r_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(rb, 4, vl), 4, vl); vint8mf2_t l_hi = __riscv_vsra_vx_i8mf2(lb, 4, vl); vint8mf2_t r_hi = __riscv_vsra_vx_i8mf2(rb, 4, vl); vint16m1_t p_lo = __riscv_vwmul_vv_i16m1(l_lo, r_lo, vl); vint16m1_t p_hi = __riscv_vwmul_vv_i16m1(l_hi, r_hi, vl); vacc4 = __riscv_vwadd_wv_i32m2(vacc4, p_lo, vl); vacc4 = __riscv_vwadd_wv_i32m2(vacc4, p_hi, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
    { size_t vl = __riscv_vsetvl_e8mf2(rem); vint8mf2_t lb = __riscv_vle8_v_i8mf2(lhs + off, vl); vint8mf2_t rb = __riscv_vle8_v_i8mf2(rhs + off, vl); vint8mf2_t l_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(lb, 4, vl), 4, vl); vint8mf2_t r_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(rb, 4, vl), 4, vl); vint8mf2_t l_hi = __riscv_vsra_vx_i8mf2(lb, 4, vl); vint8mf2_t r_hi = __riscv_vsra_vx_i8mf2(rb, 4, vl); vint16m1_t p_lo = __riscv_vwmul_vv_i16m1(l_lo, r_lo, vl); vint16m1_t p_hi = __riscv_vwmul_vv_i16m1(l_hi, r_hi, vl); vacc5 = __riscv_vwadd_wv_i32m2(vacc5, p_lo, vl); vacc5 = __riscv_vwadd_wv_i32m2(vacc5, p_hi, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
    { size_t vl = __riscv_vsetvl_e8mf2(rem); vint8mf2_t lb = __riscv_vle8_v_i8mf2(lhs + off, vl); vint8mf2_t rb = __riscv_vle8_v_i8mf2(rhs + off, vl); vint8mf2_t l_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(lb, 4, vl), 4, vl); vint8mf2_t r_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(rb, 4, vl), 4, vl); vint8mf2_t l_hi = __riscv_vsra_vx_i8mf2(lb, 4, vl); vint8mf2_t r_hi = __riscv_vsra_vx_i8mf2(rb, 4, vl); vint16m1_t p_lo = __riscv_vwmul_vv_i16m1(l_lo, r_lo, vl); vint16m1_t p_hi = __riscv_vwmul_vv_i16m1(l_hi, r_hi, vl); vacc6 = __riscv_vwadd_wv_i32m2(vacc6, p_lo, vl); vacc6 = __riscv_vwadd_wv_i32m2(vacc6, p_hi, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
    { size_t vl = __riscv_vsetvl_e8mf2(rem); vint8mf2_t lb = __riscv_vle8_v_i8mf2(lhs + off, vl); vint8mf2_t rb = __riscv_vle8_v_i8mf2(rhs + off, vl); vint8mf2_t l_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(lb, 4, vl), 4, vl); vint8mf2_t r_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(rb, 4, vl), 4, vl); vint8mf2_t l_hi = __riscv_vsra_vx_i8mf2(lb, 4, vl); vint8mf2_t r_hi = __riscv_vsra_vx_i8mf2(rb, 4, vl); vint16m1_t p_lo = __riscv_vwmul_vv_i16m1(l_lo, r_lo, vl); vint16m1_t p_hi = __riscv_vwmul_vv_i16m1(l_hi, r_hi, vl); vacc7 = __riscv_vwadd_wv_i32m2(vacc7, p_lo, vl); vacc7 = __riscv_vwadd_wv_i32m2(vacc7, p_hi, vl); off += vl; rem = (n > off) ? (n - off) : 0; }
  }
  vacc0 = __riscv_vadd_vv_i32m2(vacc0, vacc1, vl0);
  vacc0 = __riscv_vadd_vv_i32m2(vacc0, vacc2, vl0);
  vacc0 = __riscv_vadd_vv_i32m2(vacc0, vacc3, vl0);
  vacc0 = __riscv_vadd_vv_i32m2(vacc0, vacc4, vl0);
  vacc0 = __riscv_vadd_vv_i32m2(vacc0, vacc5, vl0);
  vacc0 = __riscv_vadd_vv_i32m2(vacc0, vacc6, vl0);
  vacc0 = __riscv_vadd_vv_i32m2(vacc0, vacc7, vl0);
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
  vint32m1_t vred = __riscv_vredsum_vs_i32m2_i32m1(vacc0, vzero, vl0);
  int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  out[0] = ((float)sum) * scale;
}
