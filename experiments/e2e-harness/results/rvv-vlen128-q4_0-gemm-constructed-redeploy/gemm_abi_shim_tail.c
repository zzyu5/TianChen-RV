
/* ------------------------------------------------------------------------ *
 * ABI shim (reversible redeploy) -- keeps the DEPLOYED symbol + ABI that
 * repack.cpp already calls, so repack.cpp is UNCHANGED. Forwards to the
 * front-door-CONSTRUCTED kernel above (be66c917 typed_repack_gemm_loop_body).
 *
 *   deployed ABI : (n, s, vx, vy, nr, nc, bs)   [retired monolith emit order]
 *   constructed  : (nr, bs, n, s, nc, vx, vy)   [7-role GEMM ABI, exported]
 *
 * vx=q4 weight (block_q4_0x16, 288B stride), vy=q8 act (block_q8_0x4, 136B).
 * A one-shot banner proves at RUNTIME that the A-tree prefill path executes the
 * CONSTRUCTED kernel (not the retired hand op, not vendor).
 * ------------------------------------------------------------------------ */
#include <stdio.h>
extern "C" void tcrv_emitc_ggml_repack_gemm_q4_0_q8_0_kernel_ggml_repack_gemm_q4_0_q8_0(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy,
    size_t nr, size_t nc, size_t bs) {
  { static volatile int announced_cgemm = 0; if (!announced_cgemm) { announced_cgemm = 1;
      fprintf(stderr,
        "TCRV CONSTRUCTED GEMM(front-door be66c917 typed_repack_gemm_loop_body) "
        "ENGAGED n=%zu nr=%zu nc=%zu bs=%zu\n", n, nr, nc, bs); } }
  tcrv_emitc_ggml_gemm_q4_0_q8_0_repack_gemm_kernel_ggml_gemm_q4_0_q8_0_repack_gemm(
      /*nr=*/nr, /*bs=*/bs, /*n=*/n, /*s=*/s, /*nc=*/nc, /*vx=*/vx, /*vy=*/vy);
}
