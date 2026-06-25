// Thin ABI adapter: maps a ggml-style positional GEMV ABI onto the
// COMPILER-EMITTED q5_0 repack GEMV symbol tcrv-opt produced (mlir-translate
// --mlir-to-cpp). The emitted symbol's signature is:
//   tcrv_emitc_..._ggml_repack_gemv_q5_0_q8_0(
//       size_t n, float *s, const uint8_t *vx, const uint8_t *vy, size_t nc)
#include "emitted-repack-gemv-q5_0-m1.cpp"

extern "C" void ggml_gemv_q5_0_16x1_q8_0(int n, float *s, size_t bs,
                                         const void *vx, const void *vy,
                                         int nr, int nc) {
    (void)bs;
    (void)nr;
    tcrv_emitc_ggml_repack_gemv_q5_0_q8_0_kernel_ggml_repack_gemv_q5_0_q8_0(
        (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);
}
