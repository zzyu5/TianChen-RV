// GGML FACTORY q5_K x q8_K vec_dot -- the REAL contribution baseline (opponent_class=factory).
//
// Body copied VERBATIM from the `#if defined __riscv_v` branch of
//   ggml_vec_dot_q5_K_q8_K  @ llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c:2081-2185
// This is the softest K-quant opponent (see kquant-factory-opponent-probe.md):
//   - PLAIN intrinsics, NO inline asm (the other 4 K-quant gears all have vl128 asm)
//   - NO vlNNN VLEN dispatch, NO xtheadvector variant (single monolithic body)
//   - fixed vl=8 (mins) / vl=32 (main), fixed LMUL m2/m4/m8 (VLEN-blind)
//   - register-resident decode (no aux8 scratch)  <-- factory's structural edge
//   - vredsum-in-loop: TWO __riscv_vredsum per j-iteration (8 per super-block)
//        + 1 vredsum for the q8sums*mins term                <-- factory's stall
//
// Exported under OUR kernel's 4-role ABI + shared symbol so the SAME
// q5k_verify_driver.c links either this factory or our emitted kernel by name.
//   (size_t n, float *s, const uint8_t *vx, const uint8_t *vy)
// GGML_CPU_FP16_TO_FP32 is realized with the SAME `(float)*(const _Float16*)`
// idiom our emitted kernel uses; with a zfh -march BOTH sides emit hardware
// fcvt.s.h so the fp16-conversion cost is identical and cancels in the A/B ratio.
// Compiled -x c++ (like our mlir-to-cpp kernel), same clang/flags on-board.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

#define QK_K 256
#define K_SCALE_SIZE 12
typedef uint16_t ggml_half;

typedef struct {
  ggml_half d;                  // super-block scale for quantized scales
  ggml_half dmin;               // super-block scale for quantized mins
  uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
  uint8_t qh[QK_K / 8];         // quants, high bit
  uint8_t qs[QK_K / 2];         // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 176, "wrong q5_K block size/padding");

typedef struct {
  float d;                  // delta
  int8_t qs[QK_K];          // quants
  int16_t bsums[QK_K / 16]; // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == 292, "wrong q8_K block size/padding");

static inline float g_f16_to_f32(ggml_half h) { return (float)*(const _Float16 *)&h; }

extern "C" void tcrv_emitc_ggml_vec_dot_q5_K_q8_K_kernel_rvv_q5_K_q8_K_block_dot(
    size_t nsz, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int n = (int)nsz;
  assert(n % QK_K == 0);

  const block_q5_K *__restrict x = (const block_q5_K *)vx;
  const block_q8_K *__restrict y = (const block_q8_K *)vy;

  const int nb = n / QK_K;

  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  uint32_t utmp[4];

  const uint8_t *scales = (const uint8_t *)&utmp[0];
  const uint8_t *mins = (const uint8_t *)&utmp[2];

  float sumf = 0;
  float sums = 0.0;

  size_t vl;

  for (int i = 0; i < nb; ++i) {

    vl = 8;

    const uint8_t *__restrict q5 = x[i].qs;
    const uint8_t *__restrict hm = x[i].qh;
    const int8_t *__restrict q8 = y[i].qs;

    const float d = g_f16_to_f32(x[i].d) * y[i].d;
    const float dmin = g_f16_to_f32(x[i].dmin) * y[i].d;

    vint16m1_t q8sums_0 = __riscv_vlse16_v_i16m1(y[i].bsums, 4, vl);
    vint16m1_t q8sums_1 = __riscv_vlse16_v_i16m1(y[i].bsums + 1, 4, vl);
    vint16m1_t q8sums = __riscv_vadd_vv_i16m1(q8sums_0, q8sums_1, vl);

    memcpy(utmp, x[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    vuint8mf2_t mins8 = __riscv_vle8_v_u8mf2(mins, vl);
    vint16m1_t v_mins = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(mins8, vl));
    vint32m2_t prod = __riscv_vwmul_vv_i32m2(q8sums, v_mins, vl);

    vint32m1_t sumi = __riscv_vredsum_vs_i32m2_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);
    sumf -= dmin * __riscv_vmv_x_s_i32m1_i32(sumi);

    vl = 32;
    int32_t aux32 = 0;
    int is = 0;

    uint8_t m = 1;
    vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
    vuint8m2_t vqh = __riscv_vle8_v_u8m2(hm, vl);

    for (int j = 0; j < QK_K / 64; ++j) {
      // load Q5 and Q8
      vuint8m2_t q5_x = __riscv_vle8_v_u8m2(q5, vl);
      vint8m2_t q8_y1 = __riscv_vle8_v_i8m2(q8, vl);
      vint8m2_t q8_y2 = __riscv_vle8_v_i8m2(q8 + 32, vl);

      // compute mask for addition
      vint8m2_t q5_a = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vand_vx_u8m2(q5_x, 0x0F, vl));
      vuint8m2_t qh_m1 = __riscv_vand_vx_u8m2(vqh, m, vl);
      vbool4_t vmask_1 = __riscv_vmsne_vx_u8m2_b4(qh_m1, 0, vl);
      vint8m2_t q5_m1 = __riscv_vadd_vx_i8m2_mu(vmask_1, q5_a, q5_a, 16, vl);
      m <<= 1;

      vint8m2_t q5_l = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vsrl_vx_u8m2(q5_x, 0x04, vl));
      vuint8m2_t qh_m2 = __riscv_vand_vx_u8m2(vqh, m, vl);
      vbool4_t vmask_2 = __riscv_vmsne_vx_u8m2_b4(qh_m2, 0, vl);
      vint8m2_t q5_m2 = __riscv_vadd_vx_i8m2_mu(vmask_2, q5_l, q5_l, 16, vl);
      m <<= 1;

      vint16m4_t v0 = __riscv_vwmul_vv_i16m4(q5_m1, q8_y1, vl);
      vint16m4_t v1 = __riscv_vwmul_vv_i16m4(q5_m2, q8_y2, vl);

      vint32m8_t vs1 = __riscv_vwmul_vx_i32m8(v0, scales[is++], vl);
      vint32m8_t vs2 = __riscv_vwmul_vx_i32m8(v1, scales[is++], vl);

      vint32m1_t vacc1 = __riscv_vredsum_vs_i32m8_i32m1(vs1, vzero, vl);
      vint32m1_t vacc2 = __riscv_vredsum_vs_i32m8_i32m1(vs2, vacc1, vl);

      aux32 += __riscv_vmv_x_s_i32m1_i32(vacc2);
      q5 += 32;
      q8 += 64;
    }

    sums += aux32 * d;
  }

  *s = sumf + sums;
}
