/* format_micro_kernel_stub.c -- HOST-ONLY self-test stub for the OURS side.
 *
 * Provides trivial host definitions of the 6 exported constructed-kernel symbols so
 * format_micro_driver.c LINKS + RUNS on the x86 host (structure self-test). On the
 * BOARD these are NOT used: the real constructed kernels come from the pinned-HEAD
 * exported objects (experiments/active/format-micro-rvv-vlen128/exported_objects/*.o).
 *
 * NOT numerically faithful -- self-test checks STRUCTURE, not values.
 */
#include <stdint.h>
#include <stddef.h>

#define STUB_KERNEL(SYM) \
  void SYM(size_t n, float* s, const uint8_t* vx, const uint8_t* vy){ \
    long acc=0; for(size_t i=0;i<n && i<64;i++) acc += (long)vx[i]*(long)vy[i]; \
    *s = (float)acc * 1e-6f; \
  }

STUB_KERNEL(tcrv_emitc_ggml_vec_dot_iq3_s_q8_K_kernel_rvv_iq3_s_q8_K_block_dot)
STUB_KERNEL(tcrv_emitc_ggml_vec_dot_iq2_s_q8_K_kernel_rvv_iq2_s_q8_K_block_dot)
STUB_KERNEL(tcrv_emitc_ggml_vec_dot_iq2_xs_q8_K_kernel_rvv_iq2_xs_q8_K_block_dot)
STUB_KERNEL(tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_rvv_iq2_xxs_q8_K_block_dot)
STUB_KERNEL(tcrv_emitc_ggml_vec_dot_iq3_xxs_q8_K_kernel_rvv_iq3_xxs_q8_K_block_dot)
STUB_KERNEL(tcrv_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_rvv_iq4_xs_q8_K_block_dot)
STUB_KERNEL(tcrv_emitc_ggml_vec_dot_tq2_0_q8_K_kernel_rvv_tq2_0_q8_K_block_dot)
STUB_KERNEL(tcrv_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot)
