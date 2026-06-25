// Thin ABI adapter: maps ggml's positional GEMM ABI onto OUR RVV0.7-emitted
// kernel symbol (tcrv-opt --tcrv-rvv-materialize-repack-strip-width=
// march=rv64gc_xtheadvector --tcrv-rvv-lower-to-emitc | mlir-translate).
//
// Emitted symbol signature (from gemm_rvv07.cpp):
//   tcrv_emitc_..._kernel_..._gemm_q4_0_q8_0(
//       size_t n, float *s, const uint8_t *vx, const uint8_t *vy,
//       size_t nr, size_t nc, size_t bs)
// i.e. bs is LAST and vx/vy/nr/nc shift up by one slot, vs ggml's
//   ggml_gemm_q4_0_16x1_q8_0(int n, float *s, size_t bs,
//                            const void *vx, const void *vy, int nr, int nc)
// (cross-checked against the emitted index math: arg3 @stride 288 -> weight vx,
//  arg4 @stride 136 -> block_q8_0x4 activation vy, arg5/4 -> nr-groups,
//  arg6/16 -> nc-groups, arg7 -> output row stride bs). Identical symbol name
// and ABI to the RVV1.0 board-confirm emission -- only the body's LMUL differs.
//
// A rename/#define would mis-wire bs<->vx and silently produce garbage, so this
// is a real wrapper, not an alias. We pull in the emitted .cpp directly so the
// only translation-unit is this one.

#include "gemm_rvv07.cpp"

extern "C" void ggml_gemm_q4_0_16x1_q8_0(int n, float *s, size_t bs,
                                         const void *vx, const void *vy,
                                         int nr, int nc) {
    tcrv_emitc_ggml_repack_gemm_q4_0_q8_0_kernel_ggml_repack_gemm_q4_0_q8_0(
        (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy,
        (size_t)nr, (size_t)nc, bs);
}
