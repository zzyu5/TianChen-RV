/* tcrv_q6_k_q8_k.cpp — INC-12 (q6_K K2) ggml-ABI drop-in DEFINITION.
 *
 * Defines a REAL external `ggml_vec_dot_q6_K_q8_K` symbol (the exact 8-arg ggml
 * vec_dot signature) that link-overrides ggml's reference and delegates to our
 * compiler-emitted kernel. Compiled as a separate TU and linked alongside
 * tcrv_emitted_kernel.cpp; the linker resolves callers' `ggml_vec_dot_q6_K_q8_K`
 * to THIS definition (a genuine drop-in, not a header-local inline copy).
 *
 * No kernel math here -- only the ABI bridge (drop unused bs/bx/by/nrc, cast the
 * const void* block pointers, widen int n -> size_t).
 */
#include "tcrv_q6_k_q8_k.h"

extern "C" void ggml_vec_dot_q6_K_q8_K(int n, float *s, size_t bs,
                                       const void *vx, size_t bx,
                                       const void *vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    tcrv_emitc_ggml_vec_dot_q6_K_q8_K_kernel_ggml_vec_dot_q6_K_q8_K(
        (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);
}
