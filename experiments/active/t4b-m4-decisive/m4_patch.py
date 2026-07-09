#!/usr/bin/env python3
# [G3-cert-hardening M4] MINIMAL, NON-DESTRUCTIVE reversible patch of the ggml A-tree (board).
# Edits ONLY arch/riscv/repack.cpp, and ONLY by APPENDING (after the existing tcrv emitted
# includes): (1) our instrumented q4_K kernel .inc, (2) our clean q5_K kernel .inc, (3) two
# visibility-default extern"C" wrappers so the driver can call our kernels directly.
# It does NOT touch the dispatch selector and does NOT modify ggml_gemm_q4_K_16x1_q8_K or its
# _generic -- ggml's OWN repack GEMM stays intact and callable as the head-to-head opponent.
# 1 tracked file edited. Backup under /tmp/m4_decisive. NO git stash/rm/mv/add/commit.
import hashlib
ARCH = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/arch/riscv/repack.cpp"
def md5(p): return hashlib.md5(open(p,"rb").read()).hexdigest()

a = open(ARCH).read()
anchor = ('#include "tcrv_emitted_repack_gemm.inc"\n'
          '#include "tcrv_emitted_repack_gemv.inc"\n')
assert a.count(anchor) == 1, f"anchor count = {a.count(anchor)}"
addition = (
    '#include "tcrv_emitted_repack_gemm.inc"\n'
    '#include "tcrv_emitted_repack_gemv.inc"\n'
    '// TianChen-RV [G3-cert-hardening M4]: board-scratch includes (reversible, NON-destructive).\n'
    '// ggml_gemm_q4_K_16x1_q8_K and its _generic are UNTOUCHED and remain the head-to-head opponent.\n'
    '#include "/tmp/m4_decisive/instr_q4K.inc"   // our q4_K repack-GEMM (base md5 90d454da + pure-observation captures)\n'
    '#include "/tmp/m4_decisive/fresh_q5K.inc"   // our q5_K repack-GEMM (md5 c209226b, clean)\n'
    'extern "C" __attribute__((visibility("default")))\n'
    'void tcrv_m4_call_q4K(size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs){\n'
    '    tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(n, s, vx, vy, nr, nc, bs);\n'
    '}\n'
    'extern "C" __attribute__((visibility("default")))\n'
    'void tcrv_m4_call_q5K(size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs){\n'
    '    tcrv_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K(n, s, vx, vy, nr, nc, bs);\n'
    '}\n'
)
a = a.replace(anchor, addition, 1)
open(ARCH,"w").write(a)
print("PATCH OK (M4: non-destructive include+wrappers)")
print("ARCH md5:", md5(ARCH))
print("q4K include present:", '/tmp/m4_decisive/instr_q4K.inc' in a)
print("q5K include present:", '/tmp/m4_decisive/fresh_q5K.inc' in a)
print("wrappers present:", ('tcrv_m4_call_q4K' in a) and ('tcrv_m4_call_q5K' in a))
