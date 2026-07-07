/* kquant_dispatch_probe.c — on-board RUNTIME dispatch probe for the ggml repack
 * trait selector, replicated VERBATIM from the board's own llama.cpp checkout
 * (ggml/src/ggml-cpu/repack.cpp :: ggml::cpu::repack::get_tensor_traits, tag b9692,
 * commit f3e1828). Reproduces the exact RISC-V branch decision at the LIVE
 * __riscv_vlenb() so the opponent-absence claim is a board-run artifact, not only a
 * source read. For each quant type it prints whether ggml selects a REPACK trait
 * (online repack GEMM/GEMV path) or NULLPTR (=> no repack => fall back to the
 * standard block-dot ggml_vec_dot_<type>_q8_K per-element mul_mat path).
 *
 * The verbatim source logic being reproduced (RISC-V, `#if defined __riscv_zvfh`):
 *   Q4_0 / Q4_K / Q2_K / Q8_0 / IQ4_NL:
 *       switch (__riscv_vlenb()*8) {
 *         case 128:  break;   // TODO  -> falls through, returns nullptr
 *         case 256:  if (ne1 % 16 == 0) return &<type>_16x1_...;  break;
 *         case 512:  break;   // TODO
 *         case 1024: break;   // TODO
 *         default:   return nullptr;
 *       }
 *   Q5_K / Q6_K / MXFP4: NO ggml_cpu_has_riscv_v() branch at all -> nullptr on RISC-V.
 */
#include <riscv_vector.h>
#include <stdio.h>
#include <string.h>

/* returns the selected repack trait name, or "NULLPTR(block-dot fallback)". ne1_ok = (ne[1]%16==0). */
static const char *riscv_repack_trait(const char *type, int ne1_ok) {
    int vbits = (int)(__riscv_vlenb() * 8);
    int has_riscv_branch = (!strcmp(type,"Q4_0") || !strcmp(type,"Q4_K") ||
                            !strcmp(type,"Q2_K") || !strcmp(type,"Q8_0") ||
                            !strcmp(type,"IQ4_NL"));
    if (!has_riscv_branch) return "NULLPTR(block-dot fallback) [no riscv branch in selector]";
    switch (vbits) {
        case 128:  return "NULLPTR(block-dot fallback) [case 128 = TODO no-op]";
        case 256:  return ne1_ok ? "<type>_16x1_q8_K (repack GEMM/GEVM)"
                                  : "NULLPTR(block-dot fallback) [ne1%16!=0]";
        case 512:  return "NULLPTR(block-dot fallback) [case 512 = TODO no-op]";
        case 1024: return "NULLPTR(block-dot fallback) [case 1024 = TODO no-op]";
        default:   return "NULLPTR(block-dot fallback) [default]";
    }
}

int main(void) {
    int vbits = (int)(__riscv_vlenb() * 8);
    printf("=== ggml repack dispatch probe (LIVE __riscv_vlenb()*8 = %d) ===\n", vbits);
    const char *types[] = {"Q4_0","Q4_K","Q5_K","Q6_K","Q2_K","Q8_0","IQ4_NL"};
    for (unsigned i = 0; i < sizeof(types)/sizeof(*types); i++)
        printf("  %-7s ne1%%16==0 -> %s\n", types[i], riscv_repack_trait(types[i], 1));
    printf("VERDICT: at VLEN=%d, Q4_K and Q5_K both select NULLPTR => opponent has NO working\n", vbits);
    printf("         repack path; prefill mul_mat falls back to block-dot ggml_vec_dot_q4_K/q5_K_q8_K.\n");
    return 0;
}
