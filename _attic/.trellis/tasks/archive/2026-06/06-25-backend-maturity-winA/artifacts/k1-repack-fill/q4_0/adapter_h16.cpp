#include "k_gemm_q4_0_h16.cpp"
extern "C" void OURS_gemm_q4_0_16x1_q8_0(int n, float *s, size_t bs,
    const void *vx, const void *vy, int nr, int nc) {
  tcrv_emitc_ggml_repack_gemm_q4_0_q8_0_kernel_ggml_repack_gemm_q4_0_q8_0(
      (size_t)n, s, (const uint8_t*)vx, (const uint8_t*)vy, (size_t)nr, (size_t)nc, bs);
}
