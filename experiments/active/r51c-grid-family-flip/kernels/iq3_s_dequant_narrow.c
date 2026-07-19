#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
// R5.1-C grid family flip · iq3_s dequantize_row NARROW-PER-ENTRY OWNED body (de-lottery).
//
// iq3_s dequant currently has NO owned vector body -- the deployed emit is a SCALAR AoS
// super-block loop (weft-opt regen: 2 vsetvl, 0 vector arithmetic) left to clang's
// codegen-lottery (ISSUE-001/002). This is the W4 iq3_xxs paradigm ported to its grid-of-4
// EXPLICIT-SIGNS sibling: replace the lottery scalar emit with an OWNED narrow per-entry
// pipeline.
//
// W4 lever (dequant · iq3_xxs 0.36->1.39): each 4-byte grid entry decoded by ONE narrow
// OWNED pipeline over its 4 CONTIGUOUS grid bytes -- SCALAR-computed grid pointer
// (gridb + idx*4, a sh2add) + unit-stride vle8 (vl=4, NO indexed gather) + i8 sign fold
// (grid bytes < 16, grid*(+-1) never overflows) + cheap m1 widening (vsext_vf4->i32m1 +
// vfcvt) + vfmul(db) + vse32. NO wide-LMUL m8 vsext/vfcvt (the W4 cost center), NO gather.
//
// iq3_s deltas from iq3_xxs: (1) 9th index bit merged from qh (idx = q | ((qh<<k)&256));
// (2) EXPLICIT per-lane sign BYTES (signs[32], bit-tested) instead of a ksigns[aux>>7l]
// lookup -- one signByte covers a grid-entry PAIR (entry1 = bits 0-3 = mask{1,2,4,8},
// entry2 = bits 4-7 = mask{16,32,64,128}, the SAME klo/khi fold as iq3_xxs); (3) db =
// d*(1+2*scale), no trailing factor. Byte-exact to dequantize_row_iq3_s by construction.
// VLEN-agnostic (AVL=4). OWNED __riscv_ intrinsics, NO inline-asm, NO hand-scheduling.
#include "iq3s_grid_table.h"   // weft_iq3s_grid[512] (uint32)

extern "C" void weft_emitc_dequant_iq3_s_kernel_dequant_iq3_s(
    size_t v1, const uint8_t* x, float* y) {
  static const uint8_t weft_iq3s_kmask_lo[4] = {1, 2, 4, 8};
  static const uint8_t weft_iq3s_kmask_hi[4] = {16, 32, 64, 128};
  size_t nb = v1 / 256;
  const uint8_t* gridb = (const uint8_t*) weft_iq3s_grid;
  vuint8mf4_t klo = __riscv_vle8_v_u8mf4(weft_iq3s_kmask_lo, 4);
  vuint8mf4_t khi = __riscv_vle8_v_u8mf4(weft_iq3s_kmask_hi, 4);
  for (size_t ib = 0; ib < nb; ib += 1) {
    const uint8_t* xb = x + ib * 110;
    float* yb = y + ib * 256;
    float d = (float)*(const _Float16 *)(xb);
    for (int g = 0; g < 4; g += 1) {
      int scByte = (int) xb[106 + g];
      float db1 = d * (float)(1 + 2 * (scByte & 15));
      float db2 = d * (float)(1 + 2 * (scByte >> 4));
      int qh0 = (int) xb[66 + 2 * g + 0];
      int qh1 = (int) xb[66 + 2 * g + 1];
      // two passes (qh0/db1 then qh1/db2) per outer group g.
      for (int p = 0; p < 2; p += 1) {
        int qh = p ? qh1 : qh0;
        float db = p ? db2 : db1;
        const uint8_t* qs = xb + (2 + 16 * g + 8 * p);
        const uint8_t* sg = xb + (74 + 8 * g + 4 * p);
        float* yg = yb + (g * 64 + 32 * p);
        for (int l = 0; l < 4; l += 1) {
          int q1 = (int) qs[2 * l + 0];
          int q2 = (int) qs[2 * l + 1];
          int idx1 = q1 | ((qh << (8 - 2 * l)) & 256);
          int idx2 = q2 | ((qh << (7 - 2 * l)) & 256);
          uint8_t signByte = sg[l];
          // entry1: 4 contiguous grid bytes, sign bits 0-3 (klo).
          const int8_t* gp1 = (const int8_t*)(gridb + (size_t)idx1 * 4);
          vint8mf4_t gv1 = __riscv_vle8_v_i8mf4(gp1, 4);
          vuint8mf4_t sb1 = __riscv_vand_vx_u8mf4(klo, signByte, 4);
          vbool32_t m1 = __riscv_vmsne_vx_u8mf4_b32(sb1, 0, 4);
          vint8mf4_t gs1 = __riscv_vmerge_vvm_i8mf4(gv1, __riscv_vneg_v_i8mf4(gv1, 4), m1, 4);
          vint32m1_t g321 = __riscv_vsext_vf4_i32m1(gs1, 4);
          vfloat32m1_t r1 = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(g321, 4), db, 4);
          __riscv_vse32_v_f32m1(yg + l * 8 + 0, r1, 4);
          // entry2: 4 contiguous grid bytes, sign bits 4-7 (khi).
          const int8_t* gp2 = (const int8_t*)(gridb + (size_t)idx2 * 4);
          vint8mf4_t gv2 = __riscv_vle8_v_i8mf4(gp2, 4);
          vuint8mf4_t sb2 = __riscv_vand_vx_u8mf4(khi, signByte, 4);
          vbool32_t m2 = __riscv_vmsne_vx_u8mf4_b32(sb2, 0, 4);
          vint8mf4_t gs2 = __riscv_vmerge_vvm_i8mf4(gv2, __riscv_vneg_v_i8mf4(gv2, 4), m2, 4);
          vint32m1_t g322 = __riscv_vsext_vf4_i32m1(gs2, 4);
          vfloat32m1_t r2 = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(g322, 4), db, 4);
          __riscv_vse32_v_f32m1(yg + l * 8 + 4, r2, 4);
        }
      }
    }
  }
}
