#!/usr/bin/env python3
# [G5-M2 iq4_nl] Reversible in-place iq4_nl wiring patch that DEPLOYS OUR EMITTED vl=8
# iq4_nl repack kernels (correctness-carrier) into the ggml A-tree (board rvv, VLEN128).
#
# Upstream state (recon, board baseline WinB-q4_0-ON):
#   iq4_nl has a FULL upstream riscv 16x1 repack scaffold ALREADY PRESENT:
#     * struct block_iq4_nlx16 (repack.h:118, stride 288 = 16 fp16 d + QK4_NL*8 nibbles)
#     * make_block_iq4_nlx16 + repack_iq4_nl_to_iq4_nl_16_bl (repack.cpp:3691/3715)
#     * repack<block_iq4_nl,1,16> + gemv/gemm<block_iq4_nl,1,16,Q8_0> templates
#     * trait iq4_nl_16x1_q8_0 registered (repack.cpp:4568)
#     * ARCH riscv bodies ggml_gemv/gemm_iq4_nl_16x1_q8_0 (arch/riscv/repack.cpp:463/1342)
#   BUT the ARCH bodies are HARDCODED AVL=16 (every intrinsic uses literal 16 +
#   vrgather_vv register gather) -> on VLEN128 (mf2/SEW8 VLMAX=8) they process only 8 of
#   16 interleaved columns = HALF GARBAGE ([GAP-Q8_0-VLEN128-KERNEL] pattern), AND the
#   dispatch gate is `case 128: break; // TODO` -> get_tensor_traits returns nullptr ->
#   stock falls to block-dot. q8_0 activation is fully present upstream (no sub-scaffold).
#
# So deploy = (1) flip iq4_nl@VLEN128 dispatch gate ON (route to the existing trait) +
# (2) INTERCEPT the two ARCH bodies with our EMITTED vl=8 kernels (VLEN128-safe, memory
# gather vluxei16 not vrgather) BEFORE the broken upstream vl=16 code -- exactly the q4_0/
# q8_0-M1b correctness-carrier pattern. Edits EXACTLY 2 tracked files (GEN + ARCH); the
# upstream scaffold (struct/make/repack/trait) is REUSED unchanged. Reversible. NO git.
# The two .inc (tcrv_emitted_{gemm,gevm}_iq4_nl.inc) must already be copied next to the
# q4_0 .inc in arch/riscv/ by the run wrapper.
import hashlib, sys
GEN  = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/repack.cpp"
ARCH = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/arch/riscv/repack.cpp"
BASE_GEN  = "deb61a29dd079440ffdc8996b5bd2fa1"
BASE_ARCH = "99131cf791e30348b588423b2388e0b8"
def md5(p): return hashlib.md5(open(p,"rb").read()).hexdigest()

g0, a0 = md5(GEN), md5(ARCH)
if g0 != BASE_GEN or a0 != BASE_ARCH:
    print(f"*** A-tree NOT at baseline: GEN={g0} ARCH={a0} -- ABORT"); sys.exit(10)

# ---- GEN edit1: flip iq4_nl@VLEN128 traits gate ON (route to existing repack trait) ----
g = open(GEN).read()
old1 = ("                case 128:  { break; } // TODO\n"
        "                case 256:  { if (cur->ne[1] % 16 == 0) { return &iq4_nl_16x1_q8_0; } break; }\n")
new1 = ("                case 128:  { if (cur->ne[1] % 16 == 0) { return &iq4_nl_16x1_q8_0; } break; } /* TCRV-G5-M2 iq4_nl@VLEN128 emitted-kernel route */\n"
        "                case 256:  { if (cur->ne[1] % 16 == 0) { return &iq4_nl_16x1_q8_0; } break; }\n")
assert g.count(old1) == 1, f"GEN edit1 anchor count = {g.count(old1)}"
g = g.replace(old1, new1); open(GEN,"w").write(g)

# ---- ARCH edit2: include our emitted iq4_nl kernels next to the q4_0 ones ----
a = open(ARCH).read()
inc_anchor = '#include "tcrv_emitted_repack_gemv.inc"\n'
assert a.count(inc_anchor) == 1, f"inc anchor count={a.count(inc_anchor)}"
a = a.replace(inc_anchor,
              inc_anchor +
              '// TianChen-RV [G5-M2] iq4_nl: compiler-emitted vl=8 GEMM + GEVM (VLEN128 correctness-carrier).\n'
              '#include "tcrv_emitted_gemm_iq4_nl.inc"\n'
              '#include "tcrv_emitted_gevm_iq4_nl.inc"\n')

def insert_branch_after_unused(a, func_sig, branch):
    si = a.find(func_sig)
    assert si != -1 and a.count(func_sig) == 1, f"sig {func_sig!r} count={a.count(func_sig)}"
    marker = "    UNUSED(blocklen);\n"
    mi = a.find(marker, si); assert mi != -1, f"marker not found after {func_sig!r}"
    at = mi + len(marker)
    return a[:at] + branch + a[at:]

# GEVM (decode): emitted ABI (n, s, vx, vy, nc)
gevm_branch = (
    "\n"
    "#if defined __riscv_v_intrinsic\n"
    "    // [TCRV-G5-M2] VLEN128 correctness-carrier: intercept with our EMITTED vl=8 memory-gather\n"
    "    // repack-GEVM BEFORE the broken upstream vl=16 vrgather body (proven bit-exact vs oracle UT).\n"
    "    if (__riscv_vlenb() * 8 == 128) {\n"
    "        static volatile int announced_egevm_iq4nl = 0; if (!announced_egevm_iq4nl) { announced_egevm_iq4nl = 1;\n"
    "            fprintf(stderr, \"TCRV G5-M2 EMITTED GEVM(iq4_nl_16x1 VLEN128 compiler-emitted vl=8) ENGAGED n=%d nc=%d\\n\", n, nc); }\n"
    "        tcrv_emitc_ggml_repack_gemv_iq4_nl_q8_0_kernel_ggml_repack_gemv_iq4_nl_q8_0(\n"
    "            (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);\n"
    "        return;\n"
    "    }\n"
    "#endif\n"
)
a = insert_branch_after_unused(a, "void ggml_gemv_iq4_nl_16x1_q8_0(int n", gevm_branch)

# GEMM (prefill): emitted ABI (nr, bs, n, s, vx, vy, nc)
gemm_branch = (
    "\n"
    "#if defined __riscv_v_intrinsic\n"
    "    // [TCRV-G5-M2] VLEN128 correctness-carrier: intercept with our EMITTED vl=8 memory-gather\n"
    "    // repack-GEMM BEFORE the broken upstream vl=16 vrgather body (proven bit-exact vs oracle UT).\n"
    "    if (__riscv_vlenb() * 8 == 128) {\n"
    "        static volatile int announced_egemm_iq4nl = 0; if (!announced_egemm_iq4nl) { announced_egemm_iq4nl = 1;\n"
    "            fprintf(stderr, \"TCRV G5-M2 EMITTED GEMM(iq4_nl_16x1 VLEN128 compiler-emitted vl=8) ENGAGED n=%d nr=%d nc=%d\\n\", n, nr, nc); }\n"
    "        tcrv_emitc_ggml_repack_gemm_iq4_nl_q8_0_kernel_ggml_repack_gemm_iq4_nl_q8_0(\n"
    "            (size_t)nr, bs, (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);\n"
    "        return;\n"
    "    }\n"
    "#endif\n"
)
a = insert_branch_after_unused(a, "void ggml_gemm_iq4_nl_16x1_q8_0(int n", gemm_branch)

open(ARCH,"w").write(a)
print("PATCH OK (G5-M2 iq4_nl EMITTED vl=8 kernel deploy: gate flipped + gevm/gemm emitted-call intercept; upstream scaffold reused)")
print("GEN  md5:", md5(GEN))
print("ARCH md5:", md5(ARCH))
print("GEN gate128 flipped:", "TCRV-G5-M2 iq4_nl@VLEN128" in g)
print("ARCH include gemm:", 'tcrv_emitted_gemm_iq4_nl.inc' in a, " gevm:", 'tcrv_emitted_gevm_iq4_nl.inc' in a)
print("ARCH gevm emitted-call:", "tcrv_emitc_ggml_repack_gemv_iq4_nl_q8_0_kernel" in a,
      " gemm emitted-call:", "tcrv_emitc_ggml_repack_gemm_iq4_nl_q8_0_kernel" in a)
print("ARCH gevm banner:", "TCRV G5-M2 EMITTED GEVM(iq4_nl_16x1 VLEN128" in a,
      " gemm banner:", "TCRV G5-M2 EMITTED GEMM(iq4_nl_16x1 VLEN128" in a)
