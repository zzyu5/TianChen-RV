/* format_micro_factory_stub.c -- HOST-ONLY self-test stub for the ggml factory side.
 *
 * Provides trivial host definitions of the 6 `ggml_vec_dot_<fmt>_q8_K` symbols so
 * that (a) format_micro_driver.c LINKS + RUNS on the x86 host (structure self-test)
 * and (b) format_micro_opponent.sh's nm-based `locate` step has a REAL object with
 * the 6 factory `T` symbols to find. On the BOARD these stubs are NOT used: the real
 * dispatched factory is compiled from pinned ggml source by format_micro_opponent.sh.
 *
 * These are NOT numerically faithful (self-test checks STRUCTURE, not values). Any
 * board perf/correctness use of a stub is a bug the paired preflight rejects.
 */
#include <stdint.h>
#include <stddef.h>

#define STUB_FACTORY(FMT) \
  void ggml_vec_dot_##FMT##_q8_K(int n, float* s, size_t bs, const void* vx, \
                                 size_t bx, const void* vy, size_t by, int nrc){ \
    (void)bs;(void)bx;(void)by;(void)nrc; \
    const uint8_t* a=(const uint8_t*)vx; const uint8_t* b=(const uint8_t*)vy; \
    long acc=0; for(int i=0;i<n && i<64;i++) acc += (long)a[i]*(long)b[i]; \
    *s = (float)acc * 1e-6f; \
  }

STUB_FACTORY(iq3_s)
STUB_FACTORY(iq2_s)
STUB_FACTORY(iq2_xs)
STUB_FACTORY(iq2_xxs)
STUB_FACTORY(iq3_xxs)
STUB_FACTORY(iq4_xs)
