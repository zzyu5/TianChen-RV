/* abi_shim_weft_to_tcrv.c -- W5 board byte-correct ABI shim.
 *
 * The validated q3_K repack verifier (tools/e2e-harness/board/kquant_repack_verify_q3K.c,
 * HEAD 924dc31f) calls the OLD tcrv_ symbols with the OLD arg order:
 *   GEMM: tcrv_...gemm(size_t n, float* s, const uint8_t* vx, const uint8_t* vy,
 *                      size_t nr, size_t nc, size_t bs)          // bs == nc in harness
 *   GEVM: tcrv_...gemv(size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc)
 *
 * The current weft-opt emit (HEAD d55f9ab4e, weft_ prefix) has:
 *   GEMM: weft_...gemm(size_t v1, size_t v2, size_t v3, float* v4,
 *                      const uint8_t* v5, const uint8_t* v6, size_t v7)
 *         where (verified against the emit body, lines 8-20):
 *           v1 = arg0 : v1/4  = row_group_count   => v1 = nr
 *           v2 = arg1 : output row stride         => v2 = bs (== nc in harness)
 *           v3 = arg2 : vsetvl(v3), v3/256 = block_count => v3 = n (K)
 *           v7 = arg6 : v7/16 = col_group_count   => v7 = nc
 *   GEVM: weft_...gemv(size_t v1, float* v2, const uint8_t* v3,
 *                      const uint8_t* v4, size_t v5)   // == (n, s, vx, vy, nc), IDENTICAL order
 *
 * So GEVM is a pure rename; GEMM is an arg re-order. No numeric change -- the emit is
 * byte-exact to the retired direct emitter (per the fixture comment). ZERO-MODEL: the shim
 * only permutes the ABI, it computes nothing.
 */
#include <cstddef>
#include <cstdint>

extern "C" void weft_emitc_ggml_repack_gemm_q3_K_q8_K_kernel_ggml_repack_gemm_q3_K_q8_K(
    size_t v1, size_t v2, size_t v3, float* v4,
    const uint8_t* v5, const uint8_t* v6, size_t v7);
extern "C" void weft_emitc_ggml_repack_gemv_q3_K_q8_K_kernel_ggml_repack_gemv_q3_K_q8_K(
    size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5);

extern "C" void tcrv_emitc_ggml_repack_gemm_q3_K_q8_K_kernel_ggml_repack_gemm_q3_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy,
    size_t nr, size_t nc, size_t bs) {
    /* weft order: (nr, bs, n, s, vx, vy, nc) */
    weft_emitc_ggml_repack_gemm_q3_K_q8_K_kernel_ggml_repack_gemm_q3_K_q8_K(
        nr, bs, n, s, vx, vy, nc);
}

extern "C" void tcrv_emitc_ggml_repack_gemv_q3_K_q8_K_kernel_ggml_repack_gemv_q3_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc) {
    weft_emitc_ggml_repack_gemv_q3_K_q8_K_kernel_ggml_repack_gemv_q3_K_q8_K(
        n, s, vx, vy, nc);
}
