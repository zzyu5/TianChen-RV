#!/usr/bin/env python3
# [G5 M1 tracer bullet] Reversible in-place q8_0 wiring patch for the ggml A-tree
# (board rvv, openEuler VLEN128). ROUTING-FREEBIE probe: flips the q8_0@VLEN128
# repack gate ON (route q8_0 mul_mat -> the SAME upstream q8_0_16x1_q8_0 repack
# GEMM/GEVM path that is already live at VLEN256) + adds one-shot engage banners
# in the arch/riscv q8_0 gemv/gemm bodies. NO kernel swap (we deploy NO
# compiler-emitted q8_0 kernel here: no tcrv-opt on board, no q8_0 .inc; the
# upstream repack kernel is the freebie, exactly like the q4_0 routing win).
# Edits exactly 2 tracked files (GEN repack.cpp + ARCH arch/riscv/repack.cpp).
# Baseline = current WinB-q4_0-ON tree (q4_0 gate ON is inert for a q8_0 model).
# NO git stash/rm/mv/add/commit. Backups taken by the run wrapper.
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

# ---- GEN edit1: flip q8_0@VLEN128 traits gate ON (route to upstream repack) ----
# Anchor is unique to the GGML_TYPE_Q8_0 block (case256 returns q8_0_16x1_q8_0).
g = open(GEN).read()
old1 = ("            switch (__riscv_vlenb() * 8) {\n"
        "                case 128:  { break; } // TODO\n"
        "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q8_0_16x1_q8_0; } break; }\n")
new1 = ("            switch (__riscv_vlenb() * 8) {\n"
        "                case 128:  { if (cur->ne[1] % 16 == 0) { return &q8_0_16x1_q8_0; } break; } /* TCRV-G5-M1 q8_0@VLEN128 routing-freebie */\n"
        "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q8_0_16x1_q8_0; } break; }\n")
assert g.count(old1) == 1, f"GEN edit1 anchor count = {g.count(old1)}"
g = g.replace(old1, new1); open(GEN,"w").write(g)

# ---- ARCH edit2: one-shot VLEN128 engage banners in q8_0 gemv + gemm bodies ----
# Banner-ONLY (no return, no kernel swap): announce then fall through to the
# existing upstream repack kernel. Proves BOTH repack kernels engage at VLEN128.
a = open(ARCH).read()

def insert_branch_after_unused(a, func_sig, branch):
    si = a.find(func_sig)
    assert si != -1 and a.count(func_sig) == 1, f"sig {func_sig!r} count={a.count(func_sig)}"
    marker = "    UNUSED(blocklen);\n"
    mi = a.find(marker, si); assert mi != -1, f"marker not found after {func_sig!r}"
    at = mi + len(marker)
    return a[:at] + branch + a[at:]

gevm_branch = (
    "\n"
    "#if defined __riscv_v_intrinsic\n"
    "    if (__riscv_vlenb() * 8 == 128) {\n"
    "        static volatile int announced_gevm_q80 = 0; if (!announced_gevm_q80) { announced_gevm_q80 = 1;\n"
    "            fprintf(stderr, \"TCRV G5-M1 GEVM(q8_0_16x1 VLEN128 upstream-repack routing-freebie) ENGAGED n=%d nr=%d nc=%d\\n\", n, nr, nc); }\n"
    "    }\n"
    "#endif\n"
)
a = insert_branch_after_unused(a, "void ggml_gemv_q8_0_16x1_q8_0(int n", gevm_branch)

gemm_branch = (
    "\n"
    "#if defined __riscv_v_intrinsic\n"
    "    if (__riscv_vlenb() * 8 == 128) {\n"
    "        static volatile int announced_gemm_q80 = 0; if (!announced_gemm_q80) { announced_gemm_q80 = 1;\n"
    "            fprintf(stderr, \"TCRV G5-M1 GEMM(q8_0_16x1 VLEN128 upstream-repack routing-freebie) ENGAGED n=%d nr=%d nc=%d\\n\", n, nr, nc); }\n"
    "    }\n"
    "#endif\n"
)
a = insert_branch_after_unused(a, "void ggml_gemm_q8_0_16x1_q8_0(int n", gemm_branch)

open(ARCH,"w").write(a)
print("PATCH OK (G5-M1 q8_0 routing-freebie: gate flipped + gevm/gemm engage banners)")
print("GEN  md5:", md5(GEN))
print("ARCH md5:", md5(ARCH))
print("GEN gate128 flipped:", "TCRV-G5-M1 q8_0@VLEN128" in g)
print("ARCH gevm banner present:", "TCRV G5-M1 GEVM(q8_0_16x1 VLEN128" in a)
print("ARCH gemm banner present:", "TCRV G5-M1 GEMM(q8_0_16x1 VLEN128" in a)
