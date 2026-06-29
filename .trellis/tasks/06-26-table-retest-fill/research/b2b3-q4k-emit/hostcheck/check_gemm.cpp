// Host syntax-check TU for B3: emitted body + ABI shim, riscv64 + V + zvfh.
#include <stddef.h>
#include <stdint.h>
extern "C" int fprintf(void *, const char *, ...);   // board: real <stdio.h>
static void *tcrv_host_stderr;
#define stderr tcrv_host_stderr
#include "../tcrv_q4k_gemm_vlen128.inc"   // emitted body (defines the symbol)
#include "../tcrv_q4k_gemm_shim.inc"      // ABI shim (fwd-decl + static inline)
extern "C" void (*tcrv_force_gemm)(int, float *, size_t, const void *,
                                   const void *, int, int) = &tcrv_q4k_gemm_shim;
