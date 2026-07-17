/* tcrv_q4_integ.h — INC-3 llama.cpp integration shim.
 *
 * This header is #included INSIDE ggml's ggml_vec_dot_q4_0_q8_0() (arch/riscv/quants.c)
 * at the top of the function body, guarded by build-time -D macros, so the canary and
 * the real override both ride the EXACT same delegation site (same compile flags, same TU).
 *
 *   -DTCRV_Q4_0_CANARY    -> delegate to a deliberately WRONG kernel (writes *s = 0).
 *                           If inference visibly breaks, vec_dot is the live hot path.
 *   -DTCRV_Q4_0_OVERRIDE  -> delegate to our COMPILER-EMITTED kernel
 *                           (tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0).
 *
 * A global call counter is bumped on every delegation and printed once at process exit,
 * giving direct "our kernel actually ran N times" evidence (not a per-call flood).
 *
 * NOTE: we do NOT hand-write any kernel math here. The real kernel body lives in
 * tcrv_q4_kernel.cpp, emitted verbatim by the TianChen-RV compiler from the committed MLIR.
 */
#ifndef TCRV_Q4_INTEG_H
#define TCRV_Q4_INTEG_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#if defined(TCRV_Q4_0_CANARY) || defined(TCRV_Q4_0_OVERRIDE)

#ifdef __cplusplus
extern "C" {
#endif

/* The compiler-emitted kernel symbol (defined in tcrv_q4_kernel.cpp). */
void tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
    size_t n, float *s, size_t bs,
    const uint8_t *vx, size_t bx,
    const uint8_t *vy, size_t by, int32_t nrc);

/* The deliberately-wrong canary kernel (defined in tcrv_q4_shim.cpp). */
void tcrv_q4_canary_block_dot(
    size_t n, float *s, size_t bs,
    const uint8_t *vx, size_t bx,
    const uint8_t *vy, size_t by, int32_t nrc);

/* Bump the global delegation counter (defined atomically in tcrv_q4_shim.cpp).
 * Called from quants.c (C) — kept a plain C function so C never sees std::atomic. */
void tcrv_q4_bump_count(void);

#ifdef __cplusplus
}
#endif

#endif /* CANARY || OVERRIDE */

/* The delegation site, expanded inside ggml_vec_dot_q4_0_q8_0 at the top of the body.
 * Honors ggml's ABI exactly (int n -> size_t, const void* -> const uint8_t*) and returns.
 * When neither macro is set this is a no-op (stock / control builds). */
#if defined(TCRV_Q4_0_CANARY)
#define TCRV_Q4_0_DELEGATE()                                                       \
    do {                                                                           \
        tcrv_q4_bump_count();                                                       \
        tcrv_q4_canary_block_dot((size_t)(n), (s), (bs),                            \
                                 (const uint8_t *)(vx), (bx),                       \
                                 (const uint8_t *)(vy), (by), (int32_t)(nrc));      \
        return;                                                                     \
    } while (0)
#elif defined(TCRV_Q4_0_OVERRIDE)
#define TCRV_Q4_0_DELEGATE()                                                       \
    do {                                                                           \
        tcrv_q4_bump_count();                                                       \
        tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(            \
            (size_t)(n), (s), (bs),                                                 \
            (const uint8_t *)(vx), (bx),                                            \
            (const uint8_t *)(vy), (by), (int32_t)(nrc));                           \
        return;                                                                     \
    } while (0)
#else
#define TCRV_Q4_0_DELEGATE() do { } while (0)
#endif

#endif /* TCRV_Q4_INTEG_H */
