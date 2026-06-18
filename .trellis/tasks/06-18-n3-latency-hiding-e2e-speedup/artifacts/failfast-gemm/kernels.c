// kernels.c -- q4_0_q8_0 GEMM (prefill / M output columns) latency-hiding probe.
//
// Hypothesis: GEMV (decode, batch=1) caps at ~1.22x (reduction-bound). Does the
// q4_0 GEMM clear ~1.5x over ggml via DECODE AMORTIZATION across M activation
// columns? One decoded weight block's nibbles are REUSED across M columns; the
// per-(block,column) reduction is unchanged (V4 structure).
//
// Compiled as a SEPARATE TU from the harness so -O3 cannot hoist the
// loop-invariant work out of the best-of-N timing loop.
//
// Data layout (matches ggml exactly for one weight row and one activation col):
//   block_q4_0 = { fp16 d; uint8_t qs[16]; }   stride XS=18   (weight row)
//   block_q8_0 = { fp16 d; int8_t  qs[32]; }    stride YS=34   (activation col)
//   QK = 32 elements per block.
//
// ACTIVATION PACKING (block-major). For the GEMM inner loop over M columns to be
// cache-friendly, the M columns' data for a given weight block ib must be
// CONTIGUOUS. We pack activations block-major: vyp[ ib*M*YS + j*YS ... ] holds
// column j's block-ib q8_0 record. This is the natural prefill repack (amortized
// across the whole weight matrix); we treat activations as pre-packed and state
// that assumption in RESULTS.md. The weight row stays standard q4_0 (XS stride).
//
// ABI: GEMV kernels:  void k(size_t n, float *s, const uint8_t *vx, const uint8_t *vy)
//      GEMM kernels:  void k(size_t n, int M, float *out, const uint8_t *vx, const uint8_t *vyp)
//        out[j] == ggml_vec_dot_q4_0_q8_0(weight_row, column_j) for j in [0,M).

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <riscv_vector.h>

#define XS 18  // block_q4_0 stride (weight)
#define YS 34  // block_q8_0 stride (activation)

// ---------------------------------------------------------------------------
// V0: ggml's real RVV q4_0_q8_0 kernel (baseline + bit-faithful reference).
// One output (one weight row dot one activation column). GEMV/per-output.
// ---------------------------------------------------------------------------
void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32;
    float sumf = 0;
    size_t vl = 16;
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = vx + (size_t)ib * XS, *yb = vy + (size_t)ib * YS;
        vuint8m1_t tx = __riscv_vle8_v_u8m1(xb + 2, vl);
        vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
        vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
        vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
        vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
        vint8m1_t v0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_a), 8, vl);
        vint8m1_t v1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_l), 8, vl);
        vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
        p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
        vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
        int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(p, z, vl));
        float dx = (float)*(const _Float16 *)(xb), dy = (float)*(const _Float16 *)(yb);
        sumf += sumi * dx * dy;
    }
    *s = sumf;
}

// Decode one weight block's 16 qs bytes -> v0,v1 (low/high nibble int8).
#define DECODE_W(xb, v0, v1, vl)                                                     \
    vuint8m1_t tx_##v0 = __riscv_vle8_v_u8m1((xb) + 2, (vl));                        \
    vint8m1_t v0 = __riscv_vsub_vx_i8m1(                                             \
        __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(tx_##v0, 0x0F, (vl))), 8, (vl)); \
    vint8m1_t v1 = __riscv_vsub_vx_i8m1(                                             \
        __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(tx_##v0, 0x04, (vl))), 8, (vl));

// ===========================================================================
// V4-GEMV: the best accuracy-preserving per-output kernel from the prior probe.
// Carried into THIS binary so M=1 has a same-clock reference and so we can
// re-derive the 1.5x threshold from this run (not the stale 2.6 GHz numbers).
// 8 independent vwredsums -> contiguous vector store -> vfcvt/vfmacc fold,
// 2 independent fp32 accumulators, one vfredusum at end.
// ===========================================================================
void kern_v4_gemv(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32;
    const int P = 8;
    size_t vl = 16, fvl = 8;
    vfloat32m2_t facc0 = __riscv_vfmv_v_f_f32m2(0.0f, fvl);
    vfloat32m2_t facc1 = __riscv_vfmv_v_f_f32m2(0.0f, fvl);
    int32_t buf0[8], buf1[8];
    float sc0[8], sc1[8];
    int ib = 0;
    for (; ib + 2 * P <= nb; ib += 2 * P) {
        for (int j = 0; j < P; ++j) {
            const uint8_t *xa = vx + (size_t)(ib + j) * XS, *ya = vy + (size_t)(ib + j) * YS;
            const uint8_t *xb = vx + (size_t)(ib + P + j) * XS, *yb = vy + (size_t)(ib + P + j) * YS;
            { DECODE_W(xa, v0, v1, vl);
              vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(ya + 2), vl);
              vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(ya + 2 + 16), vl);
              vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
              p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
              vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
              __riscv_vse32_v_i32m1(buf0 + j, __riscv_vwredsum_vs_i16m2_i32m1(p, z, vl), 1); }
            { DECODE_W(xb, v0, v1, vl);
              vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
              vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
              vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
              p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
              vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
              __riscv_vse32_v_i32m1(buf1 + j, __riscv_vwredsum_vs_i16m2_i32m1(p, z, vl), 1); }
            sc0[j] = (float)*(const _Float16 *)(xa) * (float)*(const _Float16 *)(ya);
            sc1[j] = (float)*(const _Float16 *)(xb) * (float)*(const _Float16 *)(yb);
        }
        vfloat32m2_t f0 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2(buf0, fvl), fvl);
        vfloat32m2_t f1 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2(buf1, fvl), fvl);
        facc0 = __riscv_vfmacc_vv_f32m2(facc0, f0, __riscv_vle32_v_f32m2(sc0, fvl), fvl);
        facc1 = __riscv_vfmacc_vv_f32m2(facc1, f1, __riscv_vle32_v_f32m2(sc1, fvl), fvl);
    }
    for (; ib + P <= nb; ib += P) {
        for (int j = 0; j < P; ++j) {
            const uint8_t *xa = vx + (size_t)(ib + j) * XS, *ya = vy + (size_t)(ib + j) * YS;
            DECODE_W(xa, v0, v1, vl);
            vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(ya + 2), vl);
            vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(ya + 2 + 16), vl);
            vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
            p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
            vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
            __riscv_vse32_v_i32m1(buf0 + j, __riscv_vwredsum_vs_i16m2_i32m1(p, z, vl), 1);
            sc0[j] = (float)*(const _Float16 *)(xa) * (float)*(const _Float16 *)(ya);
        }
        vfloat32m2_t f0 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2(buf0, fvl), fvl);
        facc0 = __riscv_vfmacc_vv_f32m2(facc0, f0, __riscv_vle32_v_f32m2(sc0, fvl), fvl);
    }
    facc0 = __riscv_vfadd_vv_f32m2(facc0, facc1, fvl);
    vfloat32m1_t zf = __riscv_vfmv_v_f_f32m1(0.0f, 4);
    float sumf = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(facc0, zf, fvl));
    for (; ib < nb; ++ib) {
        const uint8_t *xb = vx + (size_t)ib * XS, *yb = vy + (size_t)ib * YS;
        DECODE_W(xb, v0, v1, vl);
        vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
        vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
        vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
        p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
        vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
        int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(p, z, vl));
        sumf += sumi * (float)*(const _Float16 *)(xb) * (float)*(const _Float16 *)(yb);
    }
    *s = sumf;
}

// ===========================================================================
// GEMM: V4 reduction structure with WEIGHT-DECODE REUSE across M columns.
//
//   outer loop over weight blocks ib:
//     decode weight block ib's nibbles ONCE -> v0,v1
//     inner loop over M activation columns j (block-major packed, contiguous):
//       reuse v0,v1; load column-j block-ib activations y0,y1
//       vwmul/vwmacc -> per-(block,col) products -> vwredsum -> sumi[j]
//       fold sumi[j]*scale into the per-column fp32 accumulator
//
// The M reductions for a given block are NATURALLY INDEPENDENT (same v0/v1,
// different activations) -- this is exactly V4's 8-independent-redsum trick, but
// the independence comes for free from the M columns. We carry M scalar fp32
// accumulators across the outer block loop (the fold is per-column, deferred,
// no serial cross-block reduction until the very end).
//
// vyp layout (block-major): column j's block-ib record at
//   vyp + ((size_t)ib * M + j) * YS.
// ===========================================================================
void kern_gemm(size_t n, int M, float *out, const uint8_t *vx, const uint8_t *vyp) {
    const int nb = (int)n / 32;
    size_t vl = 16;
    // Per-column fp32 accumulators (deferred fold). M small (<=16).
    float acc[16];
    for (int j = 0; j < M; ++j) acc[j] = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = vx + (size_t)ib * XS;
        DECODE_W(xb, v0, v1, vl);                 // decode weight block ONCE
        float dx = (float)*(const _Float16 *)(xb);
        const uint8_t *yblk = vyp + (size_t)ib * M * YS;  // M columns' block-ib data, contiguous
        for (int j = 0; j < M; ++j) {
            const uint8_t *yb = yblk + (size_t)j * YS;
            vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
            vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
            vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);   // reuse decoded v0/v1
            p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
            vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
            int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(p, z, vl));
            float dy = (float)*(const _Float16 *)(yb);
            acc[j] += sumi * dx * dy;
        }
    }
    for (int j = 0; j < M; ++j) out[j] = acc[j];
}

// ===========================================================================
// GEMM-V4FOLD: the SPEC kernel. V4 reduction structure with weight-decode reuse
// across M columns AND a VECTORISED fp32 fold across columns (no per-(block,col)
// scalar extract, no per-column scalar fadd).
//
//   outer loop over weight blocks ib:
//     decode weight block ib ONCE -> v0,v1; dx = weight block scale (scalar)
//     inner loop over M columns j:
//       reuse v0,v1; load col-j acts y0,y1; vwmul/vwmacc -> vwredsum ->
//       vse32(buf+j, ...,1)  [lane-0 store, NO vmv.x.s GPR sync]
//       dyf[j] = (float)dy_j  (col-j block-ib activation scale)
//     ONE vector fold for the whole block:
//       sumiv = vfcvt(vle32(buf, M))           (M int sumis -> fp32, M lanes)
//       scalev = vfmul_vf(vle32(dyf,M), dx)    (per-col dy * shared dx)
//       facc = vfmacc(facc, sumiv, scalev)     (M-lane fp32 accumulator)
//   end: vse32(out, facc, M).
//
// The fold is now ONE vfmacc per block for ALL M columns -> its per-output cost
// -> 0 as M grows. Uses f32m4 (16 lanes @ VLEN=128) so M<=16 fits one register.
// vyp block-major: col j's block-ib record at vyp + (ib*M + j)*YS.
// ===========================================================================
void kern_gemm_v4fold(size_t n, int M, float *out, const uint8_t *vx, const uint8_t *vyp) {
    const int nb = (int)n / 32;
    size_t vl = 16;
    size_t fvl = (size_t)M;                       // M fp32 lanes (M<=16 -> m4)
    vfloat32m4_t facc = __riscv_vfmv_v_f_f32m4(0.0f, fvl);
    int32_t buf[16];
    float dyf[16];
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = vx + (size_t)ib * XS;
        DECODE_W(xb, v0, v1, vl);                  // decode weight block ONCE
        float dx = (float)*(const _Float16 *)(xb);
        const uint8_t *yblk = vyp + (size_t)ib * M * YS;
        for (int j = 0; j < M; ++j) {
            const uint8_t *yb = yblk + (size_t)j * YS;
            vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
            vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
            vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);   // reuse decoded weights
            p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
            vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
            // lane-0 vector store of the per-(block,col) reduction; NO vmv.x.s.
            __riscv_vse32_v_i32m1(buf + j, __riscv_vwredsum_vs_i16m2_i32m1(p, z, vl), 1);
            dyf[j] = (float)*(const _Float16 *)(yb);
        }
        // ONE vectorised fold over all M columns for this block.
        vfloat32m4_t sumiv = __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(buf, fvl), fvl);
        vfloat32m4_t scalev = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(dyf, fvl), dx, fvl);
        facc = __riscv_vfmacc_vv_f32m4(facc, sumiv, scalev, fvl);
    }
    __riscv_vse32_v_f32m4(out, facc, fvl);
}

// ===========================================================================
// COST-ONLY PROBES (correctness irrelevant; isolate the cost axes).
// ===========================================================================

// decode-ONLY: weight nibble unpack of ONE block's 16 bytes -> v0,v1 int8.
// NO activation load, NO product, NO reduce. ONLY the amortizable work:
// vle8(tx), vand, vsrl, 2x vsub. Accumulate v0^v1 lanes into a sink so -O3
// cannot delete it; one redsum at the very end (outside the per-block cost).
void kern_decode_only(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    (void)vy;
    const int nb = (int)n / 32;
    size_t vl = 16;
    vint8m1_t sink = __riscv_vmv_v_x_i8m1(0, vl);
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = vx + (size_t)ib * XS;
        DECODE_W(xb, v0, v1, vl);
        sink = __riscv_vadd_vv_i8m1(sink, __riscv_vxor_vv_i8m1(v0, v1, vl), vl);
    }
    vint16m1_t z = __riscv_vmv_v_x_i16m1(0, vl);
    *s = (float)__riscv_vmv_x_s_i16m1_i16(__riscv_vwredsum_vs_i8m1_i16m1(sink, z, vl));
}

// vwmacc-ONLY: the widening product+accumulate, pre-decoded weight inputs, no
// reduce. Decode is hoisted OUT of the loop (one decode reused every block) so
// the per-block cost is purely the activation loads + vwmul/vwmacc into ONE i16
// accumulator (deliberately wrong; isolates the product/accumulate axis).
void kern_vwmacc_only(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32;
    size_t vl = 16;
    // pre-decode block 0's weights once, reuse every iteration.
    const uint8_t *x0 = vx;
    DECODE_W(x0, v0, v1, vl);
    vint16m2_t acc = __riscv_vmv_v_x_i16m2(0, vl);
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t *yb = vy + (size_t)ib * YS;
        vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
        vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
        acc = __riscv_vwmacc_vv_i16m2(acc, v0, y0, vl);
        acc = __riscv_vwmacc_vv_i16m2(acc, v1, y1, vl);
    }
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    *s = (float)__riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(acc, z, vl));
}
