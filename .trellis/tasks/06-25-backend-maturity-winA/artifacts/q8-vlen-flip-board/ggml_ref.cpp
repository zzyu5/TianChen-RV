// ggml's REAL hand-written RVV q8_0 x q8_0 block-dot kernel, verbatim recipe
// (the SAME direct-FPR fp16 load + i8m2->i16m4->vwredsum ggml ships). Emitted as
// its OWN translation unit so it is contracted under -ffp-contract=fast exactly
// as the standalone kernel that ships -- the byte-exact reference AND the perf
// baseline. vl=32 -> needs VLEN256 for a single-vsetvl cover; on VLEN128 it folds
// in two strips, but ggml's own recipe handles that identically. We match it.
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>
extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = (int)n / 32; float sumf = 0; size_t vl = 32;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * 34, *yb = vy + (size_t)ib * 34;
    const int8_t *xq = (const int8_t *)(xb + 2), *yq = (const int8_t *)(yb + 2);
    float dx = (float)*(const _Float16 *)(xb), dy = (float)*(const _Float16 *)(yb);
    vint8m2_t bx = __riscv_vle8_v_i8m2(xq, vl), by = __riscv_vle8_v_i8m2(yq, vl);
    vint16m4_t m = __riscv_vwmul_vv_i16m4(bx, by, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(m, z, vl));
    sumf += sumi * (dx * dy);
  }
  *s = sumf;
}
