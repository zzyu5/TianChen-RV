/* RVV variants for the u8 product-reduce insurance probe. -march=rv64gcv.
 * u8*u8 -> u16 product -> widen-accumulate into u32. Two widening steps, exactly
 * the proven byte-kernel shape but UNSIGNED (vwmulu / vwaddu.wv). */
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

/* (1) per-iter-reduce: vwmulu u8mf4->u16mf2, vwredsumu into running scalar. */
__attribute__((noinline)) void
u8_per_iter_reduce(const uint8_t *lhs, const uint8_t *rhs, const uint32_t *acc,
                   uint32_t *out, size_t n) {
  uint32_t sum = acc[0];
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8mf4(n - i);
    vuint8mf4_t a = __riscv_vle8_v_u8mf4(lhs + i, vl);
    vuint8mf4_t b = __riscv_vle8_v_u8mf4(rhs + i, vl);
    vuint16mf2_t p = __riscv_vwmulu_vv_u16mf2(a, b, vl);
    vuint32m1_t seed = __riscv_vmv_v_x_u32m1(sum, 1);
    vuint32m1_t red = __riscv_vwredsumu_vs_u16mf2_u32m1(p, seed, vl);
    sum = __riscv_vmv_x_s_u32m1_u32(red);
  }
  out[0] = sum;
}

/* (2) narrow-deferred: vwmulu u8mf4->u16mf2, vwaddu.wv into u32m1 acc, one
 * trailing vredsum. Competent but under-vectorized (mf4 source). */
__attribute__((noinline)) void
u8_narrow_deferred(const uint8_t *lhs, const uint8_t *rhs, const uint32_t *acc,
                   uint32_t *out, size_t n) {
  size_t vlmax = __riscv_vsetvlmax_e32m1();
  vuint32m1_t vacc = __riscv_vmv_v_x_u32m1(0, vlmax);
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8mf4(n - i);
    vuint8mf4_t a = __riscv_vle8_v_u8mf4(lhs + i, vl);
    vuint8mf4_t b = __riscv_vle8_v_u8mf4(rhs + i, vl);
    vuint16mf2_t p = __riscv_vwmulu_vv_u16mf2(a, b, vl);
    vacc = __riscv_vwaddu_wv_u32m1(vacc, p, vl);
  }
  vuint32m1_t vzero = __riscv_vmv_v_x_u32m1(0, vlmax);
  vuint32m1_t vred = __riscv_vredsum_vs_u32m1_u32m1(vacc, vzero, vlmax);
  out[0] = acc[0] + __riscv_vmv_x_s_u32m1_u32(vred);
}

/* (3) wide-deferred (tuned candidate): vwmulu u8m2->u16m4, vwaddu.wv into u32m8
 * acc, ONE trailing vredsum u32m8->u32m1. Same resource shape as the signed byte
 * winner: u8m2 (2 vregs) -> u16m4 product (4) -> u32m8 acc (8); widest legal. */
__attribute__((noinline)) void
u8_wide_deferred(const uint8_t *lhs, const uint8_t *rhs, const uint32_t *acc,
                 uint32_t *out, size_t n) {
  size_t vlmax8 = __riscv_vsetvlmax_e32m8();
  vuint32m8_t vacc = __riscv_vmv_v_x_u32m8(0, vlmax8);
  size_t off = 0, rem = n;
  while (rem > 0) {
    size_t vl = __riscv_vsetvl_e8m2(rem);
    vuint8m2_t a = __riscv_vle8_v_u8m2(lhs + off, vl);
    vuint8m2_t b = __riscv_vle8_v_u8m2(rhs + off, vl);
    vuint16m4_t p = __riscv_vwmulu_vv_u16m4(a, b, vl);
    vacc = __riscv_vwaddu_wv_u32m8(vacc, p, vl);
    off += vl;
    rem = (n > off) ? (n - off) : 0;
  }
  vuint32m1_t vzero = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
  vuint32m1_t vred = __riscv_vredsum_vs_u32m8_u32m1(vacc, vzero, vlmax8);
  out[0] = acc[0] + __riscv_vmv_x_s_u32m1_u32(vred);
}
