#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
// R5.1-C grid family flip · iq2_xs vec_dot NARROW-PER-ENTRY + VWMACC-ACCUMULATE variant.
//
// Attack-loop variant vD: keeps the W4 gather-free per-entry loads (scalar sh3add grid
// pointer + unit-stride vle8 vl=8, NO __riscv_vluxei HW gather, NO vslidedown/vget) BUT
// restores the deployed body's 16-reduction floor -- the first narrow variant (vC-style,
// one vwredsum per GROUP = 32 reductions) lost to the deployed pair-batched gather because
// vec_dot's widening was ALREADY narrow, so removing the gather did not pay for the doubled
// reduction count + lost q8-load amortization.
//
// vD folds the two groups of a half into ONE 8-lane i16 accumulator with __riscv_vwmacc
// (acc16[j] += grid_signed[j]*q8[j], widening i8*i8->i16, max |acc16[j]| = 2*43*127 = 10922
// < 32767) then ONE vwredsum per half => 16 reductions/super-block == deployed, but
// gather-free. Byte-exact: sum_j acc16[j] over 8 lanes == sum over the half's 16 products
// (integer add associative/exact). VLEN-agnostic (AVL=8). OWNED intrinsics, no inline-asm.
#include "iq2xs_grid_tables.h"

extern "C" void weft_emitc_ggml_vec_dot_iq2_xs_q8_K_kernel_rvv_iq2_xs_q8_K_block_dot(
    size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  const uint8_t* gridb = (const uint8_t*) weft_iq2xs_grid;
  const int8_t*  signb = (const int8_t*)  weft_iq2xs_signs64;
  size_t nb = v1 / 256;
  float sumf = 0.0f;
  for (size_t ib = 0; ib < nb; ib += 1) {
    const uint8_t* xb = v3 + ib * 74;
    const uint8_t* yb = v4 + ib * 292;
    float dx = (float)*(const _Float16 *)(xb);
    float dy = *(const float *)(yb);
    float d = dx * dy;
    const uint8_t* qs = xb + 2;
    const uint8_t* sc = xb + 66;
    const int8_t*  q8 = (const int8_t*)(yb + 4);
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < 8; ib32 += 1) {
      int scb = (int) sc[ib32];
      int ls1 = 2 * (scb & 15) + 1;
      int ls2 = 2 * (scb >> 4) + 1;
      for (int half = 0; half < 2; half += 1) {
        vint16m1_t acc = __riscv_vmv_v_x_i16m1(0, 8);
        for (int lInHalf = 0; lInHalf < 2; lInHalf += 1) {
          int l = 2 * half + lInHalf;
          const uint8_t* a = qs + (size_t)(4 * ib32 + l) * 2;
          uint32_t w = (uint32_t)a[0] | ((uint32_t)a[1] << 8);
          uint32_t idx = w & 511u;
          uint32_t sel = w >> 9;
          const int8_t* gp = (const int8_t*)(gridb + (size_t)idx * 8);
          vint8mf2_t gv = __riscv_vle8_v_i8mf2(gp, 8);
          const int8_t* sp = signb + (size_t)sel * 8;
          vint8mf2_t sv = __riscv_vle8_v_i8mf2(sp, 8);
          vint8mf2_t gs = __riscv_vmul_vv_i8mf2(gv, sv, 8);
          const int8_t* q8p = q8 + (size_t)(ib32 * 32 + l * 8);
          vint8mf2_t q8v = __riscv_vle8_v_i8mf2(q8p, 8);
          acc = __riscv_vwmacc_vv_i16m1(acc, gs, q8v, 8); // fold both groups into 8-lane i16
        }
        vint32m1_t seed = __riscv_vmv_v_x_i32m1(0, 1);
        seed = __riscv_vwredsum_vs_i16m1_i32m1(acc, seed, 8); // ONE reduction per half
        int32_t sumi = __riscv_vmv_x_s_i32m1_i32(seed);
        bsum += sumi * (half == 0 ? ls1 : ls2);
      }
    }
    sumf += d * (float)bsum;
  }
  *v2 = 0.125f * sumf;
}
