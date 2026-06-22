// Thin ABI adapter: maps ggml's positional GEMV ABI onto the COMPILER-EMITTED
// GEMV kernel symbol that tcrv-opt produced (mlir-translate --mlir-to-cpp).
//
// The emitted symbol's signature (from emitted-repack-gemv.cpp) is:
//   tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_ggml_repack_gemv_q4_0_q8_0(
//       size_t n, float *s, const uint8_t *vx, const uint8_t *vy, size_t nc)
// i.e. GEMV has only 5 args: NO bs, NO nr. The body's index math proves the
// mapping (cross-checked against emitted-repack-gemv.cpp):
//   arg1 (n)  -> vsetvl + n/32 block_count                 -> n
//   arg2 (s)  -> output store (s + col_group*16)           -> s
//   arg3 (vx) -> weight base @ stride 288 (block_q4_0x16)  -> vx
//   arg4 (vy) -> activation base @ stride 34 (block_q8_0)  -> vy
//   arg5 (nc) -> nc/16 col_group_count                     -> nc
//
// vs ggml's hand GEMV positional ABI (from repack.cpp / verify_kernel.cpp):
//   ggml_gemv_q4_0_16x1_q8_0(int n, float *s, size_t bs,
//                            const void *vx, const void *vy, int nr, int nc)
// so the adapter DROPS bs and nr and forwards (n, s, vx, vy, nc).
//
// A rename/#define to ggml's name would mis-wire the positional args (e.g.
// bs<->vx) and silently produce garbage, so this is a real wrapper. We pull in
// the emitted .cpp directly so the only translation unit is this one.

#include "emitted-repack-gemv.cpp"

extern "C" void ggml_gemv_q4_0_16x1_q8_0(int n, float *s, size_t bs,
                                         const void *vx, const void *vy,
                                         int nr, int nc) {
    (void)bs;
    (void)nr;
    tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_ggml_repack_gemv_q4_0_q8_0(
        (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);
}
