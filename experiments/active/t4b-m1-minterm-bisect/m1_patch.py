#!/usr/bin/env python3
# [G3-minterm-fix M1] Reversible in-place patcher for the ggml A-tree (board).
# IDENTICAL to M2c's m2c_patch.py EXCEPT the emitted-kernel include path points to the
# INSTRUMENTED S6 kernel (/tmp/m1_minterm_board/instr_q4K.inc, base md5 90d454da + pure-
# observation captures). Same case128 selector edit, same VLEN128 branch, same banner,
# same ABI adapter. Edits exactly the SAME 2 tracked files. Backups under /tmp/m1_minterm_board.
import hashlib
GEN  = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/repack.cpp"
ARCH = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/arch/riscv/repack.cpp"
INC  = "/tmp/m1_minterm_board/instr_q4K.inc"
def md5(p): return hashlib.md5(open(p,"rb").read()).hexdigest()

g = open(GEN).read()
old1 = ("            switch (__riscv_vlenb() * 8) {\n"
        "                case 128:  { break; } // TODO\n"
        "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; }\n")
new1 = ("            switch (__riscv_vlenb() * 8) {\n"
        "                case 128:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; } /* TCRV-M1 q4_K@VLEN128 */\n"
        "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; }\n")
assert g.count(old1) == 1, f"Edit1 anchor count = {g.count(old1)}"
g = g.replace(old1, new1); open(GEN,"w").write(g)

a = open(ARCH).read()
old2a = ('#include "tcrv_emitted_repack_gemm.inc"\n'
         '#include "tcrv_emitted_repack_gemv.inc"\n')
new2a = ('#include "tcrv_emitted_repack_gemm.inc"\n'
         '#include "tcrv_emitted_repack_gemv.inc"\n'
         '// TianChen-RV [G3-minterm-fix M1]: INSTRUMENTED q4_K repack-GEMM kernel (base md5 90d454da,\n'
         '// S6-tiled) with pure-observation min-term captures. Included from board scratch (reversible).\n'
         '#include "/tmp/m1_minterm_board/instr_q4K.inc"\n')
assert a.count(old2a) == 1, f"Edit2a anchor count = {a.count(old2a)}"
a = a.replace(old2a, new2a)

sig = "void ggml_gemm_q4_K_16x1_q8_K(int n"
si = a.find(sig); assert si != -1 and a.count(sig) == 1
marker = "    UNUSED(blocklen);\n"
mi = a.find(marker, si); assert mi != -1
insert_at = mi + len(marker)
branch = (
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
a = a[:insert_at] + branch + a[insert_at:]
open(ARCH,"w").write(a)
print("PATCH OK (M1: INSTRUMENTED S6 kernel)")
print("GEN  md5:", md5(GEN)); print("ARCH md5:", md5(ARCH))
print("Edit2a instr include present:", '#include "/tmp/m1_minterm_board/instr_q4K.inc"' in a)
print("Edit2b banner present:", "TCRV EMITTED GEMM(q4_K_16x1 VLEN128" in a)
