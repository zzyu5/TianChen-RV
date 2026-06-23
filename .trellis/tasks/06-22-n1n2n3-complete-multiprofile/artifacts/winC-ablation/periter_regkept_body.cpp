#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
/*
 * periter_regkept — HAND-WRITTEN analysis kernel (NOT compiler-emitted, NOT a
 * Win-C arm).  Same PER-ITERATION vredsum reduction STRUCTURE as periter_body
 * (a vredsum on the loop-carried scalar accumulator EVERY iteration), but the
 * running scalar accumulator is KEPT IN A REGISTER across iterations: it lives
 * in a vint32m1 scalar-element register (the [0] lane), seeded once from
 * acc[0], NEVER read back from / stored to out[0] inside the loop.  Store once
 * after the loop.
 *
 * Same i16mf2 source / i32m1 product, fixed m1 LMUL, same e32m1 strip (vl=4 at
 * VLEN128 — matches periter_body which also uses vsetvl_e32m1), same loads
 * (vle16_v_i16mf2), same vwmul_vv_i32m1 product, same per-iteration vredsum,
 * same loop trip count.  The ONLY difference from periter_body is the removal of
 * the per-iteration out[0] memory round-trip (the splat-load + store).
 *
 * Seeding: periter_body seeds the running scalar from out[0], which the driver
 * has set to acc[0] (==v3[0]) before the call; the loop's vredsum then folds the
 * product into that already-acc-inclusive seed.  We mirror that exactly by
 * seeding the m1 vredsum accumulator's [0] lane from v3[0] (acc[0]) ONCE — so we
 * do NOT add acc[0] again after the loop (that would double-count).
 */
extern "C" void tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv_periter_regkept(const int16_t* v1, const int16_t* v2, const int32_t* v3, int32_t* v4, size_t v5) {
  size_t v6 = __riscv_vsetvl_e32m1(v5);
  // Running scalar accumulator KEPT IN A REGISTER (vint32m1 scalar-element).
  // Seed [0] lane with acc[0] (v3[0]) ONCE; vredsum accumulates into the [0]
  // lane each iteration.  NO out[0] read/splat/store inside the loop.
  const int32_t v7 = v3[0];
  vint32m1_t acc = __riscv_vmv_v_x_i32m1(v7, 1);
  for (size_t v9 = 0; v9 < v5; v9 += v6) {
    size_t v10 = v5 - v9;
    size_t v11 = __riscv_vsetvl_e32m1(v10);
    const int16_t* v12 = v1 + v9;
    vint16mf2_t v13 = __riscv_vle16_v_i16mf2(v12, v11);
    const int16_t* v14 = v2 + v9;
    vint16mf2_t v15 = __riscv_vle16_v_i16mf2(v14, v11);
    vint32m1_t v16 = __riscv_vwmul_vv_i32m1(v13, v15, v11);
    // PER-ITERATION vredsum on the loop-carried scalar (register-kept) acc.
    acc = __riscv_vredsum_vs_i32m1_i32m1(v16, acc, v11);
  }
  // Store the register-kept scalar [0] lane ONCE, after the loop.
  int32_t result = __riscv_vmv_x_s_i32m1_i32(acc);
  vint32m1_t vout = __riscv_vmv_v_x_i32m1(result, 1);
  __riscv_vse32_v_i32m1(v4, vout, 1);
  return;
}
