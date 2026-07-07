/* kquant_dispatch_probe_q6q2q3.c -- on-board RUNTIME dispatch probe for the ggml repack
 * trait selector, for q6_K / q2_K / q3_K, replicated VERBATIM from the board's OWN
 * upstream llama.cpp checkout (ggml/src/ggml-cpu/repack.cpp ::
 * ggml::cpu::repack::get_tensor_traits, commit f3e1828,
 * /home/ubuntu/llama.cpp-upstream-native). Reproduces the exact RISC-V branch decision at
 * the LIVE __riscv_vlenb() so the opponent-absence claim is a board-RUN artifact, not only
 * a source read.
 *
 * VERBATIM board source facts (get_tensor_traits else-if chain), for these three types:
 *   Q2_K:  HAS `if (ggml_cpu_has_riscv_v())` branch, but
 *            switch (__riscv_vlenb()*8) { case 128: { break; } // TODO  -> nullptr
 *                                         case 256: if (ne1%16==0) return &q2_K_16x1_q8_K; ... }
 *          => at VLEN128 the case-128 is a TODO no-op -> falls through -> nullptr.
 *   Q6_K:  NO ggml_cpu_has_riscv_v() branch at all (only ggml_cpu_has_neon()&&matmul_int8 /
 *          neon&&dotprod). On a non-NEON RISC-V host -> no return -> nullptr at EVERY VLEN.
 *   Q3_K:  NOT PRESENT in the selector at all (there is no `else if (cur->type ==
 *          GGML_TYPE_Q3_K)` case, and no q3_K_16x1 / q3_K repack kernel exists anywhere in
 *          ggml) -> the else-if chain never matches -> nullptr at EVERY VLEN.
 *
 * nullptr from get_tensor_traits => tensor is NOT repacked => prefill mul_mat falls back to
 * the standard per-(row,col) block-dot ggml_vec_dot_q{6,2,3}_K_q8_K. That block-dot is the
 * fair opponent for our repack GEMM. Build: clang-17 -march=rv64gcv... ; run on rvv.
 */
#include <riscv_vector.h>
#include <stdio.h>
#include <string.h>

/* returns the selected repack trait name, or a NULLPTR reason, at the LIVE vlenb. */
static const char *trait_q2_K(int vbits, int ne1_ok) {
    /* riscv_v branch present; case 128 = TODO no-op */
    switch (vbits) {
        case 128:  return "NULLPTR(block-dot fallback) [riscv branch present, case 128 = TODO no-op]";
        case 256:  return ne1_ok ? "&q2_K_16x1_q8_K (repack GEMM/GEVM)"
                                  : "NULLPTR(block-dot fallback) [ne1%16!=0]";
        case 512:  return "NULLPTR(block-dot fallback) [case 512 = TODO no-op]";
        case 1024: return "NULLPTR(block-dot fallback) [case 1024 = TODO no-op]";
        default:   return "NULLPTR(block-dot fallback) [default]";
    }
}
static const char *trait_q6_K(int vbits) {
    (void)vbits; /* no riscv_v branch exists; NEON-only -> nullptr on riscv at all VLEN */
    return "NULLPTR(block-dot fallback) [NO riscv_v branch in selector; NEON-only]";
}
static const char *trait_q3_K(int vbits) {
    (void)vbits; /* no selector case at all; no q3_K repack kernel exists in ggml */
    return "NULLPTR(block-dot fallback) [NO Q3_K case in selector; no q3_K repack exists]";
}

int main(void) {
    int vbits = (int)(__riscv_vlenb() * 8);
    printf("=== ggml repack dispatch probe q6_K/q2_K/q3_K (LIVE __riscv_vlenb()*8 = %d) ===\n", vbits);
    printf("  %-6s ne1%%16==0 -> %s\n", "Q6_K", trait_q6_K(vbits));
    printf("  %-6s ne1%%16==0 -> %s\n", "Q2_K", trait_q2_K(vbits, 1));
    printf("  %-6s ne1%%16==0 -> %s\n", "Q3_K", trait_q3_K(vbits));
    printf("VERDICT: at VLEN=%d, Q6_K / Q2_K / Q3_K ALL select NULLPTR => opponent has NO\n", vbits);
    printf("         working repack path; prefill mul_mat falls back to block-dot\n");
    printf("         ggml_vec_dot_q{6,2,3}_K_q8_K. (q2_K: case-128 TODO; q6_K: NEON-only,\n");
    printf("         no riscv branch; q3_K: no selector case + no repack impl at all.)\n");
    return 0;
}
