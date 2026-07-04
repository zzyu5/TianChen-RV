// GGML FACTORY q8_0 x q8_0 vec_dot -- the REAL contribution baseline (opponent_class=factory).
//
// Body copied VERBATIM from the __riscv_v branch of ggml_vec_dot_q8_0_q8_0 at
//   llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c:452-472
// (vl=qk=32 fixed, i8m2 loads, vwmul i16m4, vmv_v_x_i32m1(0,vl) seed,
//  vwredsum i16m4_i32m1, vmv.x.s, then sumf += sumi*(dx*dy) = premul + fmaf).
//
// FIDELITY FIX (T3 step3): block_q8_0 matches REAL ggml (ggml-common.h:242-245):
//   { ggml_half d; int8_t qs[QK8_0]; }  ggml_half = uint16_t, NATURAL layout, NOT __attribute__((packed)).
//   static_assert(sizeof == 34) holds by natural size (uint16_t align 2 + 32*int8 = 34, even -> no tail pad).
//   The recon TU wrongly declared it packed -> forced a byte-wise unaligned scale read + per-block
//   stack-bounce (lbu,lbu,slli,or,sh->stack,flh<-stack) that PENALIZED the opponent. Removed -> faithful.
//
// GGML_CPU_FP16_TO_FP32 realized with the same `(float)*(const _Float16*)` idiom our emitted kernels use;
// with a zfh-capable -march BOTH sides emit hardware fcvt.s.h -> fp16-conversion cost is identical and
// cancels in the A/B ratio. Compiled -x c++ (like our mlir-to-cpp kernels), same clang/flags on-board.
//
// Exported under the driver's 9-role ABI + shared symbol so the SAME q8_0_verify_driver.c links it.

#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

#define QK8_0 32
typedef uint16_t ggml_half;
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;  // real ggml layout (NOT packed)
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");

static inline float g_f16_to_f32(ggml_half h) { return (float)*(const _Float16 *)&h; }

extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(
    size_t n, float* s, size_t bs, const uint8_t* vx, size_t bx,
    const uint8_t* vy, size_t by, int32_t nrc, const int32_t* zero_seed) {
  (void)bs; (void)bx; (void)by; (void)nrc; (void)zero_seed;
  const int qk = QK8_0;
  const int nb = (int)(n / (size_t)qk);
  const block_q8_0* x = (const block_q8_0*)vx;
  const block_q8_0* y = (const block_q8_0*)vy;
  int ib = 0;
  float sumf = 0;
  size_t vl = qk;
  for (; ib < nb; ++ib) {
    vint8m2_t bx_0 = __riscv_vle8_v_i8m2(x[ib].qs, vl);
    vint8m2_t by_0 = __riscv_vle8_v_i8m2(y[ib].qs, vl);
    vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx_0, by_0, vl);
    vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t v_sum = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(v_sum);
    sumf += sumi * (g_f16_to_f32(x[ib].d) * g_f16_to_f32(y[ib].d));
  }
  *s = sumf;
}
