// ===========================================================================
// TCRV q4_K REPACK GEMM (prefill) ABI shim  --  cell B3 (board-pending)
// ---------------------------------------------------------------------------
// Maps ggml's positional repack-GEMM ABI onto the COMPILER-EMITTED q4_K repack
// GEMM symbol that tcrv-opt + mlir-translate produced (see EMIT-COMMAND.md;
// emitted body lives in tcrv_q4k_gemm_vlen128.inc).
//
// The emitted symbol's signature (7 args, bs LAST -- vx/vy/nr/nc shift up one
// slot vs ggml) is:
//   tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
//       size_t n, float *s, const uint8_t *vx, const uint8_t *vy,
//       size_t nr, size_t nc, size_t bs)
// vs ggml's repack ABI (bs is 3rd):
//   ggml_gemm_q4_K_16x1_q8_K(int n, float *s, size_t bs,
//                            const void *vx, const void *vy, int nr, int nc)
// The shim REORDERS bs to the end (it is NOT a drop -- a rename/#define would
// mis-wire bs<->vx and silently produce garbage), mirroring the proven q4_0
// GEMM adapter (archive emit-repack-gemm/board-confirm/emitted_adapter.cpp).
//
// INTEGRATION (see RUNBOOK.md): *static inline* helper -- include it (plus
// tcrv_q4k_gemm_vlen128.inc) at FILE SCOPE in arch/riscv/repack.cpp and call it
// from a VLEN128-gated branch at the TOP of ggml_gemm_q4_K_16x1_q8_K, leaving
// the existing VLEN256 body verbatim below:
//     if (__riscv_vlenb()*8 == 128) { tcrv_q4k_gemm_shim(n,s,bs,vx,vy,nr,nc); return; }
//
// REQUIRES: <stdio.h> (fprintf/stderr) in the including TU -- true in
// arch/riscv/repack.cpp. This file deliberately does NOT include <stdio.h> so
// it stays freestanding-clean for the host syntax-check.
// ===========================================================================
#include <stddef.h>
#include <stdint.h>

// Forward declaration of the emitted body (defined in tcrv_q4k_gemm_vlen128.inc).
extern "C" void
tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy,
    size_t nr, size_t nc, size_t bs);

static inline void tcrv_q4k_gemm_shim(int n, float *s, size_t bs,
                                      const void *vx, const void *vy,
                                      int nr, int nc) {
    static bool tcrv_q4k_gemm_engaged = false;
    if (!tcrv_q4k_gemm_engaged) {
        tcrv_q4k_gemm_engaged = true;
        fprintf(stderr,
                "TCRV REPACK GEMM(q4_K_16x1 VLEN128 compiler-emitted) "
                "ENGAGED n=%d nr=%d nc=%d bs=%zu\n", n, nr, nc, bs);
    }
    tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
        (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy,
        (size_t)nr, (size_t)nc, bs);
}
