#!/usr/bin/env python3
# [G5 M1b] Reversible in-place q8_0 wiring patch that DEPLOYS OUR EMITTED vl=8
# q8_0 repack kernels (correctness-carrier) into the ggml A-tree (board rvv,
# openEuler VLEN128). Unlike M1 (banner-ONLY -> fell through to the BROKEN upstream
# vl=16 body -> e2e garbage, [GAP-Q8_0-VLEN128-KERNEL]), this patch INTERCEPTS the
# VLEN128 gemv/gemm with our front-door-constructed vl=8 kernels BEFORE the broken
# upstream body -- exactly like the deployed q4_0 emitted kernel.
# Edits exactly 2 tracked files (GEN repack.cpp gate + ARCH arch/riscv/repack.cpp
# include + two emitted-call branches). Baseline = WinB-q4_0-ON tree. NO git.
# The combined .inc (tcrv_emitted_q8_0.inc) must already be copied next to the
# q4_0 .inc files in arch/riscv/ by the run wrapper.
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

# ---- GEN edit1: flip q8_0@VLEN128 traits gate ON (route to repack trait) ----
g = open(GEN).read()
old1 = ("            switch (__riscv_vlenb() * 8) {\n"
        "                case 128:  { break; } // TODO\n"
        "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q8_0_16x1_q8_0; } break; }\n")
new1 = ("            switch (__riscv_vlenb() * 8) {\n"
        "                case 128:  { if (cur->ne[1] % 16 == 0) { return &q8_0_16x1_q8_0; } break; } /* TCRV-G5-M1b q8_0@VLEN128 emitted-kernel route */\n"
        "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q8_0_16x1_q8_0; } break; }\n")
assert g.count(old1) == 1, f"GEN edit1 anchor count = {g.count(old1)}"
g = g.replace(old1, new1); open(GEN,"w").write(g)

# ---- ARCH edit2: include our emitted q8_0 kernels next to the q4_0 ones ----
a = open(ARCH).read()
inc_anchor = '#include "tcrv_emitted_repack_gemv.inc"\n'
assert a.count(inc_anchor) == 1, f"inc anchor count={a.count(inc_anchor)}"
a = a.replace(inc_anchor,
              inc_anchor + '#include "tcrv_emitted_q8_0.inc"  /* TCRV-G5-M1b emitted vl=8 q8_0 GEVM+GEMM */\n')

def insert_branch_after_unused(a, func_sig, branch):
    si = a.find(func_sig)
    assert si != -1 and a.count(func_sig) == 1, f"sig {func_sig!r} count={a.count(func_sig)}"
    marker = "    UNUSED(blocklen);\n"
    mi = a.find(marker, si); assert mi != -1, f"marker not found after {func_sig!r}"
    at = mi + len(marker)
    return a[:at] + branch + a[at:]

# GEVM (decode): call emitted GEVM (n, s, nc, vx, -, vy, -, -); v3 slot = nc (col count)
gevm_branch = (
    "\n"
    "#if defined __riscv_v_intrinsic\n"
    "    // [TCRV-G5-M1b] VLEN128 correctness-carrier: intercept with our EMITTED\n"
    "    // vl=8 two-8-lane-halves repack-GEVM BEFORE the broken upstream vl=16 body.\n"
    "    if (__riscv_vlenb() * 8 == 128) {\n"
    "        static volatile int announced_egevm_q80b = 0; if (!announced_egevm_q80b) { announced_egevm_q80b = 1;\n"
    "            fprintf(stderr, \"TCRV G5-M1b EMITTED GEVM(q8_0_16x1 VLEN128 compiler-emitted vl=8) ENGAGED n=%d nc=%d\\n\", n, nc); }\n"
    "        tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0(\n"
    "            (size_t)n, s, (size_t)nc, (const uint8_t *)vx, (size_t)0, (const uint8_t *)vy, (size_t)0, (int32_t)0);\n"
    "        return;\n"
    "    }\n"
    "#endif\n"
)
a = insert_branch_after_unused(a, "void ggml_gemv_q8_0_16x1_q8_0(int n", gevm_branch)

# GEMM (prefill): call emitted GEMM (nr, bs, n, s, nc, vx, vy)
gemm_branch = (
    "\n"
    "#if defined __riscv_v_intrinsic\n"
    "    // [TCRV-G5-M1b] VLEN128 correctness-carrier: intercept with our EMITTED\n"
    "    // vl=8 4x8 repack-GEMM BEFORE the broken upstream vl=16 body.\n"
    "    if (__riscv_vlenb() * 8 == 128) {\n"
    "        static volatile int announced_egemm_q80b = 0; if (!announced_egemm_q80b) { announced_egemm_q80b = 1;\n"
    "            fprintf(stderr, \"TCRV G5-M1b EMITTED GEMM(q8_0_16x1 VLEN128 compiler-emitted vl=8) ENGAGED n=%d nr=%d nc=%d\\n\", n, nr, nc); }\n"
    "        tcrv_emitc_ggml_gemm_q8_0_q8_0_kernel_ggml_gemm_q8_0_q8_0(\n"
    "            (size_t)nr, bs, (size_t)n, s, (size_t)nc, (const uint8_t *)vx, (const uint8_t *)vy);\n"
    "        return;\n"
    "    }\n"
    "#endif\n"
)
a = insert_branch_after_unused(a, "void ggml_gemm_q8_0_16x1_q8_0(int n", gemm_branch)

open(ARCH,"w").write(a)
print("PATCH OK (G5-M1b q8_0 EMITTED vl=8 kernel deploy: gate flipped + gevm/gemm emitted-call intercept)")
print("GEN  md5:", md5(GEN))
print("ARCH md5:", md5(ARCH))
print("GEN gate128 flipped:", "TCRV-G5-M1b q8_0@VLEN128" in g)
print("ARCH include present:", 'tcrv_emitted_q8_0.inc' in a)
print("ARCH gevm emitted-call present:", "tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel" in a)
print("ARCH gemm emitted-call present:", "tcrv_emitc_ggml_gemm_q8_0_q8_0_kernel" in a)
print("ARCH gevm banner present:", "TCRV G5-M1b EMITTED GEVM(q8_0_16x1 VLEN128" in a)
print("ARCH gemm banner present:", "TCRV G5-M1b EMITTED GEMM(q8_0_16x1 VLEN128" in a)
