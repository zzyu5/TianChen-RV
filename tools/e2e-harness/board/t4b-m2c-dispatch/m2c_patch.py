#!/usr/bin/env python3
# [SEL-1 T4b / M2c] Reversible in-place patcher for the ggml A-tree (board).
# IDENTICAL to M2's m2_patch.py EXCEPT the golden include path points to the FRESH
# re-emitted S6-tiled q4_K repack-GEMM kernel (md5 90d454da), NOT the old full-unroll
# golden (md5 b0b5beac). Everything else (case128 selector, VLEN128 branch, banner,
# ABI adapter) is byte-identical -> the ONLY variable vs M2 is the kernel body.
# Edits exactly 2 tracked files; backups captured separately under /tmp/m2c_q4k_dispatch.
import hashlib

GEN = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/repack.cpp"
ARCH = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/arch/riscv/repack.cpp"

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

# ---- Edit 1: generic repack.cpp -- Q4_K selector case 128 (NULL/TODO -> return 16x1 trait) ----
g = open(GEN).read()
old1 = ("            switch (__riscv_vlenb() * 8) {\n"
        "                case 128:  { break; } // TODO\n"
        "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; }\n")
new1 = ("            switch (__riscv_vlenb() * 8) {\n"
        "                case 128:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; } /* TCRV-M2c q4_K@VLEN128 */\n"
        "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; }\n")
n1 = g.count(old1)
assert n1 == 1, f"Edit1 anchor count = {n1} (expected 1)"
g = g.replace(old1, new1)
open(GEN, "w").write(g)

# ---- Edit 2a: arch/riscv/repack.cpp -- include FRESH q4_K kernel after emitted incs ----
a = open(ARCH).read()
old2a = ('#include "tcrv_emitted_repack_gemm.inc"\n'
         '#include "tcrv_emitted_repack_gemv.inc"\n')
new2a = ('#include "tcrv_emitted_repack_gemm.inc"\n'
         '#include "tcrv_emitted_repack_gemv.inc"\n'
         '// TianChen-RV [SEL-1 T4b/M2c]: COMPILER-EMITTED q4_K repack-GEMM FRESH kernel (md5 90d454da,\n'
         '// S6-tiled, re-emitted from the CURRENT committed emitter). Included from board scratch to keep\n'
         '// the A-tree reversible (no new tree file). Differs from M2 ONLY in this kernel body (was b0b5beac).\n'
         '#include "/tmp/m2c_q4k_dispatch/fresh_q4K.inc"\n')
n2a = a.count(old2a)
assert n2a == 1, f"Edit2a anchor count = {n2a} (expected 1)"
a = a.replace(old2a, new2a)

# ---- Edit 2b: arch/riscv/repack.cpp -- VLEN128 branch in ggml_gemm_q4_K_16x1_q8_K ----
sig = "void ggml_gemm_q4_K_16x1_q8_K(int n"
si = a.find(sig)
assert si != -1, "gemm q4_K sig not found"
assert a.count(sig) == 1, "unexpected multiple gemm q4_K sigs"
marker = "    UNUSED(blocklen);\n"
mi = a.find(marker, si)
assert mi != -1, "UNUSED(blocklen) not found after sig"
insert_at = mi + len(marker)
branch = (
    "\n"
    "#if defined __riscv_v_intrinsic\n"
    "    // TianChen-RV [SEL-1 T4b/M2c] VLEN=128: the intrinsic body below is the VLEN256 16-lane\n"
    "    // form (vfloat32m2_t at vl=16); at VLEN128 an e32m2 vector holds only 8 f32 lanes, so\n"
    "    // route to the COMPILER-EMITTED VLEN128 FRESH kernel (md5 90d454da) through the\n"
    "    // ggml->emitted ABI adapter (ggml (n,s,bs,vx,vy,nr,nc) -> emitted (n,s,vx,vy,nr,nc,bs)).\n"
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
a = a[:insert_at] + branch + a[insert_at:]
open(ARCH, "w").write(a)

print("PATCH OK (M2c: FRESH S6 kernel md5 90d454da)")
print("GEN  md5:", md5(GEN))
print("ARCH md5:", md5(ARCH))
print("Edit1 applied:", new1.strip().splitlines()[1].strip())
print("Edit2a fresh include present:", '#include "/tmp/m2c_q4k_dispatch/fresh_q4K.inc"' in a)
print("Edit2b banner present:", "TCRV EMITTED GEMM(q4_K_16x1 VLEN128" in a)
