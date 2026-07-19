#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
// R5.1-C grid family flip · iq2_xs vec_dot NARROW-PER-ENTRY research variant.
//
// W4 lever ported from dequant (iq3_xxs 0.36->1.39) to the vec_dot grid family:
//   (a) GATHER-FREE assembly  [necessary] -- the deployed iq2_xs body hits a i64m4
//       __riscv_vluxei16 HARDWARE indexed gather (8 per super-block) + i8m4 sign fold
//       + 6 vslidedown/8 vget per pair (ISSUE-120 VLEN-agnostic recovery machinery).
//   (b) NARROW per-entry cheap widening [dominant] -- each 8-byte grid entry is decoded
//       by ONE narrow OWNED pipeline over its 8 CONTIGUOUS grid bytes: a SCALAR-computed
//       grid pointer (grid_u8 + idx*8, a sh3add) + a unit-stride vle8 (vl=8, NO indexed
//       gather) + i8 sign fold (grid in [8,43] so grid*(+-1) never overflows) + a cheap
//       vwmul_i16m1 (8-lane, mf2->m1) + vwredsum accumulate. NO m4 gather, NO m4 vmul,
//       NO vslidedown, NO vget.
//
// BYTE-EXACT to the deployed body by construction: same idx=w&511 / sel=w>>9 decode,
// same signs64 (+-1) plane, same grid; the per-half integer reduction sum(16 lanes) ==
// vwredsum(group0 8 lanes) + vwredsum(group1 8 lanes) (integer add is associative/exact,
// max half sum 16*43*127 << 2^31). VLEN-agnostic (AVL=8 everywhere; no register-subgroup
// extraction) => VLEN128 == VLEN256, no ISSUE-120 slide needed. OWNED __riscv_ intrinsics,
// NO inline-asm, NO hand-scheduling.
#include "iq2xs_grid_tables.h"   // weft_iq2xs_grid[512] (i64) + weft_iq2xs_signs64[1024] (i8 +-1)

extern "C" void weft_emitc_ggml_vec_dot_iq2_xs_q8_K_kernel_rvv_iq2_xs_q8_K_block_dot(
    size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  const uint8_t* gridb = (const uint8_t*) weft_iq2xs_grid;    // byte view: grid_u8 + idx*8
  const int8_t*  signb = (const int8_t*)  weft_iq2xs_signs64; // signs_u8 + sel*8 (+-1 plane)
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
      // half 0 = groups l=0,1 weighted ls1 ; half 1 = groups l=2,3 weighted ls2.
      for (int half = 0; half < 2; half += 1) {
        vint32m1_t seed = __riscv_vmv_v_x_i32m1(0, 1);
        for (int lInHalf = 0; lInHalf < 2; lInHalf += 1) {
          int l = 2 * half + lInHalf;             // group 0..3
          const uint8_t* a = qs + (size_t)(4 * ib32 + l) * 2;
          uint32_t w = (uint32_t)a[0] | ((uint32_t)a[1] << 8);
          uint32_t idx = w & 511u;
          uint32_t sel = w >> 9;
          // NARROW per-entry: scalar sh3add grid pointer + unit-stride vle8 (NO gather).
          const int8_t* gp = (const int8_t*)(gridb + (size_t)idx * 8);
          vint8mf2_t gv = __riscv_vle8_v_i8mf2(gp, 8);
          const int8_t* sp = signb + (size_t)sel * 8;
          vint8mf2_t sv = __riscv_vle8_v_i8mf2(sp, 8);
          vint8mf2_t gs = __riscv_vmul_vv_i8mf2(gv, sv, 8);   // sign folds onto GRID
          const int8_t* q8p = q8 + (size_t)(ib32 * 32 + l * 8);
          vint8mf2_t q8v = __riscv_vle8_v_i8mf2(q8p, 8);
          vint16m1_t prod = __riscv_vwmul_vv_i16m1(gs, q8v, 8); // cheap m1 widening
          seed = __riscv_vwredsum_vs_i16m1_i32m1(prod, seed, 8);
        }
        int32_t sumi = __riscv_vmv_x_s_i32m1_i32(seed);
        bsum += sumi * (half == 0 ? ls1 : ls2);
      }
    }
    sumf += d * (float)bsum;
  }
  *v2 = 0.125f * sumf;
}
