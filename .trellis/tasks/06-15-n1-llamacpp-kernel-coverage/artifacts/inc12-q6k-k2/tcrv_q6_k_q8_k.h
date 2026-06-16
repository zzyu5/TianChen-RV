/* tcrv_q6_k_q8_k.h — INC-12 (q6_K K2) deployable ggml-ABI bridge (declarations).
 *
 * Declares the ggml symbol `ggml_vec_dot_q6_K_q8_K` (the EXACT 8-arg ggml
 * vec_dot signature) and the compiler-emitted 4-arg kernel it bridges to. The
 * bridge DEFINITION is a non-inline extern "C" function in tcrv_q6_k_q8_k.cpp
 * (a separate TU), so it emits a REAL external `ggml_vec_dot_q6_K_q8_K` symbol
 * that link-overrides ggml's reference -- a genuine drop-in, not a header-local
 * inline copy.
 *
 * We do NOT hand-write any kernel math. The real kernel body lives in
 * tcrv_emitted_kernel.cpp, emitted verbatim by the TianChen-RV compiler from the
 * committed MLIR (tcrv_rvv.q6_k_q8_k_block_dot). The wrapper is an ABI bridge
 * only -- exactly the role inc3's tcrv_q4_integ.h delegation plays, here
 * specialized to the q6_K arity (our op carries the 4 live ABI operands vx/vy/
 * s/n; ggml's trailing bs/bx/by/nrc are unused by the _generic q6_K math).
 */
#ifndef TCRV_Q6_K_Q8_K_H
#define TCRV_Q6_K_Q8_K_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The compiler-emitted kernel symbol (defined in tcrv_emitted_kernel.cpp). */
void tcrv_emitc_ggml_vec_dot_q6_K_q8_K_kernel_ggml_vec_dot_q6_K_q8_K(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

/* The ggml-ABI drop-in entry: the EXACT 8-arg ggml vec_dot signature. The real
 * external definition is in tcrv_q6_k_q8_k.cpp (link-overrides ggml's). */
void ggml_vec_dot_q6_K_q8_K(int n, float *s, size_t bs,
                            const void *vx, size_t bx,
                            const void *vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif

#endif /* TCRV_Q6_K_Q8_K_H */
