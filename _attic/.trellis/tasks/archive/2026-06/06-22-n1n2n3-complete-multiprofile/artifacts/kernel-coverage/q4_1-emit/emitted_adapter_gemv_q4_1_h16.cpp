// Thin ABI adapter: maps a ggml-style positional GEMV ABI onto the
// COMPILER-EMITTED q4_1 repack GEMV symbol tcrv-opt produced (mlir-translate
// --mlir-to-cpp). The emitted symbol's signature is:
//   tcrv_emitc_..._ggml_repack_gemv_q4_1_q8_1(
//       size_t n, float *s, const uint8_t *vx, const uint8_t *vy, size_t nc)
// (GEMV has only 5 args: NO bs, NO nr). We pull in the emitted .cpp directly so
// the only translation unit is this one.
#include "emitted-repack-gemv-q4_1-h16.cpp"

extern "C" void ggml_gemv_q4_1_16x1_q8_1(int n, float *s, size_t bs,
                                         const void *vx, const void *vy,
                                         int nr, int nc) {
    (void)bs;
    (void)nr;
    tcrv_emitc_ggml_repack_gemv_q4_1_q8_1_kernel_ggml_repack_gemv_q4_1_q8_1(
        (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);
}
