#!/usr/bin/env python3
# [G3 T4b seal-fix] Reversible in-place patcher for the ggml A-tree (board rvv, VLEN128).
# Deploys the CLEAN compiler-emitted q4_K repack-GEMM (md5 90d454da) + repack-GEVM VLEN128
# vl=8 kernels through the SAME reversible dispatch mechanism as the q4_0 WinB (and the M1
# instrumented bisect), but for BOTH gemm AND gevm and with the clean (non-instrumented)
# kernels. Edits exactly 2 tracked files (GEN repack.cpp + ARCH arch/riscv/repack.cpp).
# NO git stash/rm/mv/add/commit. NO emitter source change. Backups taken by the run wrapper.
import hashlib, sys
GEN  = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/repack.cpp"
ARCH = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/arch/riscv/repack.cpp"
GEMM_INC = "/tmp/t4b_seal_fix/gemm_q4_K.inc"   # md5 90d454da, S6-tiled VLEN128 vl=8
GEVM_INC = "/tmp/t4b_seal_fix/gemv_q4_K.inc"   # sibling repack-GEVM VLEN128 vl=8
BASE_GEN  = "deb61a29dd079440ffdc8996b5bd2fa1"
BASE_ARCH = "99131cf791e30348b588423b2388e0b8"
def md5(p): return hashlib.md5(open(p,"rb").read()).hexdigest()

# ---- baseline assertion (must be at m1 baseline before patching) ----
g0, a0 = md5(GEN), md5(ARCH)
if g0 != BASE_GEN or a0 != BASE_ARCH:
    print(f"*** A-tree NOT at baseline: GEN={g0} ARCH={a0} -- ABORT"); sys.exit(10)

# ---- GEN edit1: flip q4_K@VLEN128 traits selection ON (routing + repack gate) ----
g = open(GEN).read()
old1 = ("            switch (__riscv_vlenb() * 8) {\n"
        "                case 128:  { break; } // TODO\n"
        "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; }\n")
new1 = ("            switch (__riscv_vlenb() * 8) {\n"
        "                case 128:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; } /* TCRV-SEALFIX q4_K@VLEN128 */\n"
        "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; }\n")
assert g.count(old1) == 1, f"Edit1 anchor count = {g.count(old1)}"
g = g.replace(old1, new1); open(GEN,"w").write(g)

a = open(ARCH).read()

# ---- ARCH edit2a: include the clean q4_K gemm + gevm emitted kernels ----
old2a = ('#include "tcrv_emitted_repack_gemm.inc"\n'
         '#include "tcrv_emitted_repack_gemv.inc"\n')
new2a = ('#include "tcrv_emitted_repack_gemm.inc"\n'
         '#include "tcrv_emitted_repack_gemv.inc"\n'
         '// TianChen-RV [G3 T4b seal-fix]: clean compiler-emitted q4_K repack GEMM (md5 90d454da)\n'
         '// + repack GEVM, VLEN128 vl=8 two-8-lane-strip. Included from board scratch (reversible).\n'
         f'#include "{GEMM_INC}"\n'
         f'#include "{GEVM_INC}"\n')
assert a.count(old2a) == 1, f"Edit2a anchor count = {a.count(old2a)}"
a = a.replace(old2a, new2a)

def insert_branch_after_unused(a, func_sig, branch):
    si = a.find(func_sig)
    assert si != -1 and a.count(func_sig) == 1, f"sig {func_sig!r} count={a.count(func_sig)}"
    marker = "    UNUSED(blocklen);\n"
    mi = a.find(marker, si); assert mi != -1, f"marker not found after {func_sig!r}"
    at = mi + len(marker)
    return a[:at] + branch + a[at:]

# ---- ARCH edit2b: VLEN128 branch in q4_K GEMM -> clean emitted kernel ----
gemm_branch = (
    "\n"
    "#if defined __riscv_v_intrinsic\n"
    "    if (__riscv_vlenb() * 8 == 128) {\n"
    "        { static volatile int announced_gemm_q4k = 0; if (!announced_gemm_q4k) { announced_gemm_q4k = 1;\n"
    "            fprintf(stderr, \"TCRV EMITTED GEMM(q4_K_16x1 VLEN128 compiler-emitted) ENGAGED nr=%d nc=%d nb=%d\\n\", nr, nc, nb); } }\n"
    "        tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(\n"
    "            (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy,\n"
    "            (size_t)nr, (size_t)nc, bs);\n"
    "        return;\n"
    "    }\n"
    "#endif\n"
)
a = insert_branch_after_unused(a, "void ggml_gemm_q4_K_16x1_q8_K(int n", gemm_branch)

# ---- ARCH edit2c: VLEN128 branch in q4_K GEVM -> clean emitted kernel ----
gevm_branch = (
    "\n"
    "#if defined __riscv_v_intrinsic\n"
    "    if (__riscv_vlenb() * 8 == 128) {\n"
    "        { static volatile int announced_gemv_q4k = 0; if (!announced_gemv_q4k) { announced_gemv_q4k = 1;\n"
    "            fprintf(stderr, \"TCRV EMITTED GEMV(q4_K_16x1 VLEN128 compiler-emitted) ENGAGED nc=%d nb=%d\\n\", nc, nb); } }\n"
    "        tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K(\n"
    "            (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);\n"
    "        return;\n"
    "    }\n"
    "#endif\n"
)
a = insert_branch_after_unused(a, "void ggml_gemv_q4_K_16x1_q8_K(int n", gevm_branch)

open(ARCH,"w").write(a)
print("PATCH OK (seal-fix: clean q4_K gemm md5 90d454da + gevm, VLEN128 vl=8)")
print("GEN  md5:", md5(GEN))
print("ARCH md5:", md5(ARCH))
print("gemm include present:", GEMM_INC in a)
print("gevm include present:", GEVM_INC in a)
print("gemm banner present:", "TCRV EMITTED GEMM(q4_K_16x1 VLEN128" in a)
print("gevm banner present:", "TCRV EMITTED GEMV(q4_K_16x1 VLEN128" in a)
print("GEN case128 flipped:", "TCRV-SEALFIX q4_K@VLEN128" in g)
