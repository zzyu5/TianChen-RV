// Host syntax-check TU for B2: validates the emitted body + ABI shim compile
// together for riscv64 + V + zvfh (the board ISA). Freestanding stdio stub
// stands in for the board TU's real <stdio.h>; the SHIM TEXT is identical.
#include <stddef.h>
#include <stdint.h>
extern "C" int fprintf(void *, const char *, ...);   // board: real <stdio.h>
static void *tcrv_host_stderr;
#define stderr tcrv_host_stderr
#include "../tcrv_q4k_gemv_vlen128.inc"   // emitted body (defines the symbol)
#include "../tcrv_q4k_gemv_shim.inc"      // ABI shim (fwd-decl + static inline)
// Force the shim to be checked:
extern "C" void (*tcrv_force_gemv)(int, float *, size_t, const void *,
                                   const void *, int, int) = &tcrv_q4k_gemv_shim;
