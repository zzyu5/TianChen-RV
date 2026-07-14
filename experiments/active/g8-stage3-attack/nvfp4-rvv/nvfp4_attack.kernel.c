/* nvfp4_attack.kernel.c — G8 §六.3 nvfp4 rvv 攻坚 · 一轮等价构造尝试.
 *
 * Lever: register-pressure de-unroll (re-roll the 4-way manually-unrolled sub-block
 * body into a genuine s=0..3 loop) + hoist the redundant per-q8-block fp16->fp32
 * scale decode (baseline recomputes the SAME fcvt.s.h twice per q8_0 block, once for
 * each of the two sub-blocks that share it).
 *
 * Diagnosed bottleneck (objdump ours_nvfp4_baseline.o on rvv, clang-18.1.8,
 * VLEN128, -O3): the manually-unrolled 4x straight-line body forces LLVM's RVV
 * register allocator to SPILL vector state to the stack repeatedly (11x vs1r.v +
 * 11x vl1r.v + 22x `csrr a0,vlenb` address arithmetic out of 371 total loop-body
 * instructions => ~18-20% pure spill overhead, incl. the 16-entry codebook table
 * itself being spilled/reloaded from the stack instead of staying vector-register-
 * resident). Re-rolling collapses 4 duplicated straight-line copies into 1, cutting
 * the live-range / register-pressure footprint that triggers the spills.
 *
 * Byte-exact safety argument: (1) hoisting the fp16->fp32 q8-scale conversion is a
 * pure deterministic function of the SAME input bits read twice in baseline -> same
 * output bits, no rounding/order change. (2) the float accumulation order into sumf
 * is preserved bit-for-bit: s=0,1,2,3 sequential update `sumf = sumf + (dy*scale)*isum`,
 * identical order/grouping to the original 4x-unrolled body. (3) the integer
 * decode/dot-product path (vand/vsrl/vrgather/vwmul/vwmacc/vwredsum) is untouched
 * per-sub-block arithmetic, only the C-level loop structure changed, not vl width
 * or op sequence per sub-block.
 */
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
#include <math.h>

extern "C" void weft_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_rvv_nvfp4_q8_0_block_dot_ATTACK(
    size_t n, float* out, const uint8_t* wq_base, const uint8_t* aq_base) {
  static const int8_t weft_nvfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  vint8m1_t codebook = __riscv_vle8_v_i8m1(weft_nvfp4_kvalues, 16);

  float sumf = 0.0f;
  size_t block_count = n / 64;

  for (size_t sb = 0; sb < block_count; sb++) {
    const uint8_t* blk = wq_base + sb * 36;
    size_t q8_base_idx = sb * 2;
    const uint8_t* q8blk0 = aq_base + q8_base_idx * 34;
    const uint8_t* q8blk1 = aq_base + (q8_base_idx + 1) * 34;
    /* hoisted: baseline recomputes these TWICE per q8 block (once per sub-block
     * sharing it) -- same bits in, same bits out, computed once here instead. */
    float dy0 = (float)*(const _Float16*)(q8blk0);
    float dy1 = (float)*(const _Float16*)(q8blk1);
    const int8_t* qs0 = (const int8_t*)(q8blk0 + 2);
    const int8_t* qs1 = (const int8_t*)(q8blk1 + 2);

    for (int s = 0; s < 4; s++) {
      const uint8_t sbyte = blk[s];
      const uint32_t si = (uint32_t)sbyte;
      const uint32_t exp_bits = (si >> 3) & 0xF;
      const uint32_t man_bits = si & 0x7;
      const int exp_i = (int)exp_bits;
      const float man_f = (float)(int)man_bits;
      const float v_denorm = ldexpf(man_f, -9);
      const float v_norm = ldexpf(1.0f + man_f / 8.0f, exp_i - 7);
      float scale = (exp_bits == 0) ? v_denorm : v_norm;
      scale *= 0.5f;
      if (si == 0 || si == 0x7F) scale = 0.0f;

      const uint8_t* wnib = blk + 4 + (size_t)s * 8;
      const int q8_blk = s / 2;
      const int q8_off = (s % 2) * 16;
      const int8_t* act = (q8_blk == 0 ? qs0 : qs1) + q8_off;
      const float dy = (q8_blk == 0 ? dy0 : dy1);

      size_t vl = __riscv_vsetvl_e8m1(8);
      vuint8m1_t raw = __riscv_vle8_v_u8m1(wnib, vl);
      vuint8m1_t lo = __riscv_vand_vx_u8m1(raw, 0x0F, vl);
      vuint8m1_t hi = __riscv_vsrl_vx_u8m1(raw, 0x04, vl);
      vint8m1_t wlo = __riscv_vrgather_vv_i8m1(codebook, lo, vl);
      vint8m1_t whi = __riscv_vrgather_vv_i8m1(codebook, hi, vl);
      vint8m1_t a_lo = __riscv_vle8_v_i8m1(act, vl);
      vint8m1_t a_hi = __riscv_vle8_v_i8m1(act + 8, vl);
      vint16m2_t p0 = __riscv_vwmul_vv_i16m2(wlo, a_lo, vl);
      vint16m2_t p1 = __riscv_vwmacc_vv_i16m2(p0, whi, a_hi, vl);
      vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
      vint32m1_t red = __riscv_vwredsum_vs_i16m2_i32m1(p1, zero, vl);
      int32_t isum = __riscv_vmv_x_s_i32m1_i32(red);

      sumf = sumf + (dy * scale) * (float)isum;
    }
  }
  out[0] = sumf;
  return;
}
