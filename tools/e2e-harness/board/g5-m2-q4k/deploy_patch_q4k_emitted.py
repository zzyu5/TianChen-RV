#!/usr/bin/env python3
# [G5 M2 q4_K] Reversible in-place q4_K wiring patch that DEPLOYS OUR CURRENT-HEAD
# compiler-EMITTED vl=8 q4_K repack kernels (correctness-carrier) into the ggml
# A-tree (board rvv, openEuler VLEN128).
#
# WHY correctness-carrier (not routing): the upstream arch/riscv q4_K body is a
# VLEN256 variant -- every load/acc uses hardcoded AVL=16 (f32m2/f16m1/u8mf2 VLMAX=8
# on VLEN128) and AVL=64 (u8m2 VLMAX=32 on VLEN128) => on VLEN128 everything clamps
# to half => garbage (== the vl16 PPL 822057 / q8_0-MIRAGE class). So flipping the
# case128 gate ALONE routes to the broken body. This patch INTERCEPTS the VLEN128
# gemv/gemm with our EMITTED vl=8 kernels BEFORE the broken upstream body (same
# mechanism as the deployed q4_0 WinB emitted kernel and the q8_0 M1b carrier).
#
# The two current-HEAD .inc files (tcrv_emitted_gemm_q4_K.inc md5 6cbd9c19,
# tcrv_emitted_gevm_q4_K.inc md5 e909a9bd) must already be copied into ARCHDIR by
# the run wrapper. Edits exactly 2 tracked files. Baseline = WinB-q4_0-ON tree. NO git.
#
# CURRENT-HEAD GEMM signature (params REORDERED vs the 07-10 seal-proven 90d454da):
#   (size_t v1=nr, size_t v2=bs, size_t v3=n, float* v4=s,
#    const uint8_t* v5=vx, const uint8_t* v6=vy, size_t v7=nc)
#   decoded: v3/256=nb, v1/4=row_groups, v7/16=col_groups, v5 stride 2304 (q4_Kx16),
#            v6 stride 1168 (q8_Kx4), store base v4 + row*v2 (=bs) + colgrp*16.
# CURRENT-HEAD GEVM signature (byte-identical to seal-proven e909a9bd):
#   (size_t v1=n, float* v2=s, const uint8_t* v3=vx, const uint8_t* v4=vy, size_t v5=nc)
import hashlib, sys
GEN  = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/repack.cpp"
ARCH = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/arch/riscv/repack.cpp"
BASE_GEN  = "deb61a29dd079440ffdc8996b5bd2fa1"
BASE_ARCH = "99131cf791e30348b588423b2388e0b8"
def md5(p): return hashlib.md5(open(p,"rb").read()).hexdigest()

# ---- baseline assertion (must be at WinB-q4_0-ON baseline before patching) ----
g0, a0 = md5(GEN), md5(ARCH)
if g0 != BASE_GEN or a0 != BASE_ARCH:
    print(f"*** A-tree NOT at baseline: GEN={g0} ARCH={a0} -- ABORT"); sys.exit(10)

# ---- GEN edit1: flip q4_K@VLEN128 traits gate ON (route to repack trait) ----
g = open(GEN).read()
old1 = ("            switch (__riscv_vlenb() * 8) {\n"
        "                case 128:  { break; } // TODO\n"
        "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; }\n")
new1 = ("            switch (__riscv_vlenb() * 8) {\n"
        "                case 128:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; } /* TCRV-G5-M2 q4_K@VLEN128 emitted-kernel route */\n"
        "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; }\n")
assert g.count(old1) == 1, f"GEN edit1 anchor count = {g.count(old1)}"
g = g.replace(old1, new1); open(GEN,"w").write(g)

# ---- ARCH edit2: include our current-HEAD emitted q4_K kernels ----
a = open(ARCH).read()
inc_anchor = '#include "tcrv_emitted_repack_gemv.inc"\n'
assert a.count(inc_anchor) == 1, f"inc anchor count={a.count(inc_anchor)}"
a = a.replace(inc_anchor,
              inc_anchor +
              '// TianChen-RV [G5 M2]: current-HEAD compiler-emitted q4_K repack GEMM (md5 6cbd9c19)\n'
              '// + GEVM (md5 e909a9bd, byte-identical to 07-10 seal-proven). VLEN128 vl=8. Reversible.\n'
              '#include "tcrv_emitted_gemm_q4_K.inc"\n'
              '#include "tcrv_emitted_gevm_q4_K.inc"\n')

def insert_branch_after_unused(a, func_sig, branch):
    si = a.find(func_sig)
    assert si != -1 and a.count(func_sig) == 1, f"sig {func_sig!r} count={a.count(func_sig)}"
    marker = "    UNUSED(blocklen);\n"
    mi = a.find(marker, si); assert mi != -1, f"marker not found after {func_sig!r}"
    at = mi + len(marker)
    return a[:at] + branch + a[at:]

# GEVM (decode): call current-HEAD emitted GEVM (n, s, vx, vy, nc)  [byte-identical map]
gevm_branch = (
    "\n"
    "#if defined __riscv_v_intrinsic\n"
    "    // [TCRV-G5-M2] VLEN128 correctness-carrier: intercept with our EMITTED vl=8\n"
    "    // repack-GEVM BEFORE the broken upstream AVL=16 (VLEN256) body.\n"
    "    if (__riscv_vlenb() * 8 == 128) {\n"
    "        static volatile int announced_egevm_q4kb = 0; if (!announced_egevm_q4kb) { announced_egevm_q4kb = 1;\n"
    "            fprintf(stderr, \"TCRV G5-M2 EMITTED GEVM(q4_K_16x1 VLEN128 compiler-emitted vl=8) ENGAGED n=%d nc=%d nb=%d\\n\", n, nc, nb); }\n"
    "        tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K(\n"
    "            (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);\n"
    "        return;\n"
    "    }\n"
    "#endif\n"
)
a = insert_branch_after_unused(a, "void ggml_gemv_q4_K_16x1_q8_K(int n", gevm_branch)

# GEMM (prefill): call current-HEAD emitted GEMM (nr, bs, n, s, vx, vy, nc)  [reordered map]
gemm_branch = (
    "\n"
    "#if defined __riscv_v_intrinsic\n"
    "    // [TCRV-G5-M2] VLEN128 correctness-carrier: intercept with our EMITTED vl=8\n"
    "    // repack-GEMM BEFORE the broken upstream AVL=16 (VLEN256) body.\n"
    "    // current-HEAD sig reorder: (nr, bs, n, s, vx, vy, nc).\n"
    "    if (__riscv_vlenb() * 8 == 128) {\n"
    "        static volatile int announced_egemm_q4kb = 0; if (!announced_egemm_q4kb) { announced_egemm_q4kb = 1;\n"
    "            fprintf(stderr, \"TCRV G5-M2 EMITTED GEMM(q4_K_16x1 VLEN128 compiler-emitted vl=8) ENGAGED n=%d nr=%d nc=%d\\n\", n, nr, nc); }\n"
    "        tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(\n"
    "            (size_t)nr, bs, (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);\n"
    "        return;\n"
    "    }\n"
    "#endif\n"
)
a = insert_branch_after_unused(a, "void ggml_gemm_q4_K_16x1_q8_K(int n", gemm_branch)

open(ARCH,"w").write(a)
print("PATCH OK (G5-M2 q4_K current-HEAD EMITTED vl=8 kernel deploy: gate flipped + gevm/gemm emitted-call intercept)")
print("GEN  md5:", md5(GEN))
print("ARCH md5:", md5(ARCH))
print("GEN gate128 flipped:", "TCRV-G5-M2 q4_K@VLEN128" in g)
print("ARCH gemm include present:", 'tcrv_emitted_gemm_q4_K.inc' in a)
print("ARCH gevm include present:", 'tcrv_emitted_gevm_q4_K.inc' in a)
print("ARCH gevm emitted-call present:", "tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel" in a)
print("ARCH gemm emitted-call present:", "tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel" in a)
print("ARCH gevm banner present:", "TCRV G5-M2 EMITTED GEVM(q4_K_16x1 VLEN128" in a)
print("ARCH gemm banner present:", "TCRV G5-M2 EMITTED GEMM(q4_K_16x1 VLEN128" in a)
