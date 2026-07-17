// 4-arg ABI adapter for our q4_0 block-dot emit, which uses the 8-arg ggml-style
// signature (n, s, bs, vx, bx, vy, by, nrc). The harness calls a 4-arg OURS
// (n, s, vx, vy); this wrapper forwards with bs=0, bx=0, by=0, nrc=1 (the
// scalar-row vec_dot contract; all unused by the kernel body).
#include <stdint.h>
#include <stddef.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
    size_t, float *, size_t, const uint8_t *, size_t, const uint8_t *, size_t, int32_t);
extern "C" void ours_q4_0_adapter(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(n, s, 0, vx, 0, vy, 0, 1);
}
