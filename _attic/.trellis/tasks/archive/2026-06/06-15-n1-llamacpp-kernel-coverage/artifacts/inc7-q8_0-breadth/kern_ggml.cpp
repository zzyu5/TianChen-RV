// ggml's REAL hand-written q8_0 RVV kernel (quants.c:435-481), as a SEPARATE TU
// with the SAME extern "C" + flags as the kern_* shapes under test, so the
// baseline is measured under identical compilation conditions. The per-block
// fp16 scale is read via (float)*(const _Float16*)(ptr) -- the SAME direct-FPR
// load real ggml uses on this board (GGML_CPU_FP16_TO_FP32 -> scalar fcvt.s.h,
// simd-mappings.h:95) -- NOT a memcpy-to-GPR-then-fmv.h.x transcription, so the
// baseline's scalar epilogue is the genuine ggml one (fair).
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>
#define QK 32
#define Q8_STRIDE 34
extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = (int)n / QK;
  float sumf = 0;
  size_t vl = QK;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + ib * Q8_STRIDE;
    const uint8_t *yb = vy + ib * Q8_STRIDE;
    const int8_t *xq = (const int8_t *)(xb + 2);
    const int8_t *yq = (const int8_t *)(yb + 2);
    float dx = (float)*(const _Float16 *)(xb);
    float dy = (float)*(const _Float16 *)(yb);
    vint8m2_t bx = __riscv_vle8_v_i8m2(xq, vl);
    vint8m2_t by = __riscv_vle8_v_i8m2(yq, vl);
    vint16m4_t m = __riscv_vwmul_vv_i16m4(bx, by, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t r = __riscv_vwredsum_vs_i16m4_i32m1(m, z, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(r);
    sumf += sumi * (dx * dy);
  }
  *s = sumf;
}
