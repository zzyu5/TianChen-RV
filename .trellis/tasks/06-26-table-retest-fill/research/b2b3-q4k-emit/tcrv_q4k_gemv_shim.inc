// ===========================================================================
// TCRV q4_K REPACK GEVM (decode) ABI shim  --  cell B2 (board-pending)
// ---------------------------------------------------------------------------
// Maps ggml's positional repack-GEVM ABI onto the COMPILER-EMITTED q4_K repack
// GEVM symbol that tcrv-opt + mlir-translate produced (see EMIT-COMMAND.md;
// emitted body lives in tcrv_q4k_gemv_vlen128.inc).
//
// The emitted symbol's signature (5 args -- GEVM has NO bs, NO nr) is:
//   tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K(
//       size_t n, float *s, const uint8_t *vx, const uint8_t *vy, size_t nc)
// vs ggml's 7-arg repack ABI:
//   ggml_gemv_q4_K_16x1_q8_K(int n, float *s, size_t bs,
//                            const void *vx, const void *vy, int nr, int nc)
// The shim drops bs + nr (GEVM: nr == 1, bs unused), mirroring the proven
// q8_0 / q4_1 7->5-arg shims (archive q8-repack-e2e-FINDING.md;
// kernel-coverage/q4_1-emit/emitted_adapter_gemv_q4_1.cpp).
//
// INTEGRATION (see RUNBOOK.md): this is a *static inline* helper -- include it
// (plus tcrv_q4k_gemv_vlen128.inc) at FILE SCOPE in arch/riscv/repack.cpp and
// call it from a VLEN128-gated branch at the TOP of ggml_gemv_q4_K_16x1_q8_K,
// leaving the existing VLEN256 body verbatim below:
//     if (__riscv_vlenb()*8 == 128) { tcrv_q4k_gemv_shim(n,s,bs,vx,vy,nr,nc); return; }
//
// REQUIRES: <stdio.h> (fprintf/stderr) in the including TU -- true in
// arch/riscv/repack.cpp (the live q4_0 ENGAGED print uses the same). This file
// deliberately does NOT include <stdio.h> so it stays freestanding-clean for
// the host syntax-check (the board TU is hosted; see RUNBOOK host-check note).
// ===========================================================================
#include <stddef.h>
#include <stdint.h>

// Forward declaration of the emitted body (defined in tcrv_q4k_gemv_vlen128.inc;
// byte-identical signature -- a name typo here would fail to link on the board,
// so the host check greps this name == the emitted symbol).
extern "C" void
tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy, size_t nc);

static inline void tcrv_q4k_gemv_shim(int n, float *s, size_t bs,
                                      const void *vx, const void *vy,
                                      int nr, int nc) {
    static bool tcrv_q4k_gemv_engaged = false;
    if (!tcrv_q4k_gemv_engaged) {
        tcrv_q4k_gemv_engaged = true;
        fprintf(stderr,
                "TCRV REPACK GEVM(q4_K_16x1 VLEN128 compiler-emitted) "
                "ENGAGED n=%d nc=%d\n", n, nc);
    }
    (void)bs;  // GEVM: output is one column-group, bs (row stride) unused
    (void)nr;  // GEVM: nr == 1
    tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K(
        (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);
}
