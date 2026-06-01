// Classroom baseline for the llama.cpp q8_0_q8_0 RVV dot-product shape.
//
// This is intentionally hand-written RVV C. It is not the student solution.
// Student work should make TianChenRV generate an equivalent kernel from MLIR.
// The structure follows the RVV path of llama.cpp ggml_vec_dot_q8_0_q8_0,
// with float scales instead of fp16 scales to keep the classroom ABI focused.

#include "llama_q8_0_q8_0.h"

#include <assert.h>
#include <riscv_vector.h>

void tcrv_llama_q8_0_q8_0_reference_rvv(
    const tcrv_classroom_block_q8_0 *x,
    const tcrv_classroom_block_q8_0 *y,
    float *out,
    size_t n) {
  const size_t qk = TCRV_CLASSROOM_QK8_0;
  assert(n % qk == 0);

  const size_t nb = n / qk;
  float sumf = 0.0f;

  for (size_t ib = 0; ib < nb; ++ib) {
    const size_t vl = qk;

    vint8m2_t vx = __riscv_vle8_v_i8m2(x[ib].qs, vl);
    vint8m2_t vy = __riscv_vle8_v_i8m2(y[ib].qs, vl);

    vint16m4_t prod = __riscv_vwmul_vv_i16m4(vx, vy, vl);
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t reduced =
        __riscv_vwredsum_vs_i16m4_i32m1(prod, zero, vl);

    const int32_t block_sum = __riscv_vmv_x_s_i32m1_i32(reduced);
    sumf += (float)block_sum * x[ib].d * y[ib].d;
  }

  *out = sumf;
}

