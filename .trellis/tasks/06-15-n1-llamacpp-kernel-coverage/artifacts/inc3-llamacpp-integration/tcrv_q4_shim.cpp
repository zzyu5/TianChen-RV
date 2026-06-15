/* tcrv_q4_shim.cpp — INC-3 llama.cpp integration: counter + canary + exit reporter.
 *
 * Compiled as a separate TU and linked into ggml-cpu. Provides:
 *   - tcrv_q4_call_count : global delegation counter (proof our path ran, printed at exit).
 *   - tcrv_q4_canary_block_dot : the deliberately WRONG kernel (writes *s = 0) used by
 *     -DTCRV_Q4_0_CANARY to prove ggml_vec_dot_q4_0_q8_0 is the live hot path.
 *
 * No kernel math is hand-written here. The REAL kernel is tcrv_q4_kernel.cpp (compiler-emitted).
 *
 * The counter is std::atomic (incremented from up to 16 OpenMP threads). Reported qualitatively.
 * The at-exit report fires from BOTH a C++ static destructor AND a C destructor attribute, so a
 * canary garbage-run that is killed mid-stream still has a chance to print the count.
 */
#include <stddef.h>
#include <stdint.h>
#include <cstdio>
#include <atomic>

extern "C" {

std::atomic<unsigned long long> tcrv_q4_call_count{0};

void tcrv_q4_bump_count(void) {
    tcrv_q4_call_count.fetch_add(1, std::memory_order_relaxed);
}

/* Deliberately-wrong canary kernel: ignores all inputs, writes *s = 0.
 * If inference visibly breaks when this is on the Q4_0 path, vec_dot is hot. */
void tcrv_q4_canary_block_dot(size_t n, float *s, size_t bs,
                              const uint8_t *vx, size_t bx,
                              const uint8_t *vy, size_t by, int32_t nrc) {
    (void)n; (void)bs; (void)vx; (void)bx; (void)vy; (void)by; (void)nrc;
    s[0] = 0.0f;
}

void tcrv_q4_report(void) {
    std::fprintf(stderr,
        "[TCRV_Q4_INTEG] ggml_vec_dot_q4_0_q8_0 delegated to our kernel %llu times\n",
        tcrv_q4_call_count.load());
}

} /* extern "C" */

/* C destructor-attribute fallback (fires even on some abnormal exits / before C++ dtors). */
__attribute__((destructor))
static void tcrv_q4_report_dtor(void) { tcrv_q4_report(); }
