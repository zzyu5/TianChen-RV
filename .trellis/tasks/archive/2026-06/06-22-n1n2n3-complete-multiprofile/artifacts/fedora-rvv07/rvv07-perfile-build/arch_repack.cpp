#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "traits.h"

#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdlib> // for qsort
#include <cstdio>  // for GGML_ASSERT

#define GGML_CPU_CLANG_WORKAROUND
#include "../../repack.h"

// TianChen-RV: COMPILER-EMITTED q4_0 repack GEMM (tcrv-opt -> mlir-translate).
#include "tcrv_emitted_repack_gemm.inc"
#include "tcrv_emitted_repack_gemv.inc"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

void ggml_quantize_mat_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

#if defined(__riscv_v_intrinsic)
    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;
    const size_t vl_calc = __riscv_vsetvl_e32m8(QK8_0);
    const size_t vl_save = __riscv_vsetvl_e64m2(4);
    vfloat32m1_t v_scalar_zero = __riscv_vfmv_s_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));

    for (int i = 0; i < nb; i++) {
        const float *x_block_base = x + i * QK8_0;
        vint8m2_t q_r0, q_r1, q_r2, q_r3;
        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 0 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            float amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            float d = amax / 127.0f;
            y[i].d[0] = GGML_CPU_FP32_TO_FP16(d);

            float id = d ? 1.0f / d : 0.0f;
            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r0 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        asm volatile ("" ::: "memory");

        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 1 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            float amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            float d = amax / 127.0f;
            y[i].d[1] = GGML_CPU_FP32_TO_FP16(d);
            float id = d ? 1.0f / d : 0.0f;

            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r1 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        asm volatile ("" ::: "memory");
        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 2 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            float amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            float d = amax / 127.0f;
            y[i].d[2] = GGML_CPU_FP32_TO_FP16(d);
            float id = d ? 1.0f / d : 0.0f;

            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r2 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        asm volatile ("" ::: "memory");
        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 3 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            float amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            float d = amax / 127.0f;
            y[i].d[3] = GGML_CPU_FP32_TO_FP16(d);
            float id = d ? 1.0f / d : 0.0f;

            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r3 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        vint64m2_t v_q64_r0 = __riscv_vreinterpret_v_i8m2_i64m2(q_r0);
        vint64m2_t v_q64_r1 = __riscv_vreinterpret_v_i8m2_i64m2(q_r1);
        vint64m2_t v_q64_r2 = __riscv_vreinterpret_v_i8m2_i64m2(q_r2);
        vint64m2_t v_q64_r3 = __riscv_vreinterpret_v_i8m2_i64m2(q_r3);
        vint64m2x4_t v_quant_tuple = __riscv_vcreate_v_i64m2x4(v_q64_r0, v_q64_r1, v_q64_r2, v_q64_r3);
        __riscv_vsseg4e64_v_i64m2x4((int64_t*)y[i].qs, v_quant_tuple, vl_save);
    }
#else
    UNUSED(nb);
    ggml_quantize_mat_q8_0_4x8_generic(x, vy, k);
#endif
}

void ggml_gemv_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

#if defined __riscv_zvfh
void ggml_gemv_q4_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    // TianChen-RV RVV0.7/C920 mitigation: the COMPILER-EMITTED RVV0.7 q4_0 GEMV
    // kernel (fraction-free i8m1->i16m2->i32m4->f32m4, from tcrv-opt) engages on
    // the VLEN=128 C920; the fractional-LMUL hand fallback is removed (RVV0.7 has
    // no fractional LMUL). Non-VLEN128 falls through to the scalar _generic reference.
#if defined __riscv_v_intrinsic
    if (__riscv_vlenb() * 8 == 128) {
        { static volatile int announced_egemv = 0; if (!announced_egemv) { announced_egemv = 1;
            fprintf(stderr, "TCRV EMITTED GEMV(q4_0_16x1 VLEN128 compiler-emitted RVV0.7) ENGAGED nc=%d nb=%d\n", nc, n / QK8_0); } }
        tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_ggml_repack_gemv_q4_0_q8_0(
            (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);
        return;
    }
#endif
    ggml_gemv_q4_0_16x1_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}
void ggml_gemv_q4_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q4_K_16x1_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_iq4_nl_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_iq4_nl_16x1_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q8_0_16x1_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q2_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q2_K_16x1_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}
#endif

void ggml_gemm_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

#if defined __riscv_zvfh
void ggml_gemm_q4_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    // TianChen-RV RVV0.7/C920 mitigation: COMPILER-EMITTED RVV0.7 q4_0 GEMM kernel.
#if defined __riscv_v_intrinsic
    if (__riscv_vlenb() * 8 == 128) {
        { static volatile int announced_egemm = 0; if (!announced_egemm) { announced_egemm = 1;
            fprintf(stderr, "TCRV EMITTED GEMM(q4_0_16x1 VLEN128 compiler-emitted RVV0.7) ENGAGED nr=%d nc=%d nb=%d\n", nr, nc, n / QK8_0); } }
        tcrv_emitc_ggml_repack_gemm_q4_0_q8_0_kernel_ggml_repack_gemm_q4_0_q8_0(
            (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nr, (size_t)nc, bs);
        return;
    }
#endif
    ggml_gemm_q4_0_16x1_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}
void ggml_gemm_q4_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q4_K_16x1_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_iq4_nl_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_iq4_nl_16x1_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q8_0_16x1_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q2_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q2_K_16x1_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}
#endif
