#!/usr/bin/env python3
# [G7 §3] Reversible in-place splice of OUR weft-emitted q5_K q8_K block-dot vec_dot
# into the tcrv-llamacpp A-tree (rvv, gcc-15.2, VLEN128).
#
# Feasibility (recon, this役 Step 1):
#   q5_K@rvv at M>1 routes through the STANDARD vec_dot block-dot path
#   (ggml_compute_forward_mul_mat_one_chunk, 16x16 block-tiled), because:
#     * llamafile_sgemm has NO q5_K case (only F32/BF16/F16/Q8_0/Q4_0/Q5_0/IQ4_NL) -> falls through
#     * get_optimal_repack_type() for Q5_K has ONLY neon branches, NO ggml_cpu_has_riscv_v()
#       branch (unlike q4_K/q2_K/iq4_nl) -> q5_K NEVER repacked on RISC-V -> stays block-dot
#   => splice point = ggml_vec_dot_q5_K_q8_K (arch/riscv/quants.c). nrc always 1 for q5_K.
#
# Edits EXACTLY 1 tracked file (arch/riscv/quants.c). The weft kernel .inc is untracked and
# copied next to quants.c by the run wrapper. Reversible (file backup + md5 double-proof). NO git.
import hashlib, sys
Q = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/arch/riscv/quants.c"
BASE = "58827c2e5be62a305ba26fea22d2dc83"   # stock upstream (commit f3e1828), unmodified
def md5(p): return hashlib.md5(open(p,"rb").read()).hexdigest()

q0 = md5(Q)
if q0 != BASE:
    print(f"*** quants.c NOT at baseline: {q0} != {BASE} -- ABORT"); sys.exit(10)

q = open(Q).read()
SIG = "void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc) {"
assert q.count(SIG) == 1, f"function signature anchor count = {q.count(SIG)}"

# (1) include the weft static kernel immediately before the function definition
INC = '// TianChen-RV [G7 §3] weft-emitted q5_K block-dot vec_dot (deploy for batched-e2e A/B).\n#include "weft_q5k_blockdot.inc"\n'
q = q.replace(SIG, INC + SIG, 1)

# (2) intercept: at function top (after the asserts/UNUSED) call the weft kernel and return.
#     Guarded by __riscv_v so non-vector builds keep the generic fallback. One-shot banner
#     proves the weft path actually engages in e2e (MIRAGE exclusion).
si = q.find(SIG)
marker = "    UNUSED(bs);\n"
mi = q.find(marker, si); assert mi != -1, "UNUSED(bs) marker not found after q5_K sig"
at = mi + len(marker)
BRANCH = (
    "\n"
    "#if defined __riscv_v\n"
    "    {\n"
    "        // [TCRV-G7-§3] deploy OUR weft-emitted q5_K block-dot (wider LMUL m2 core) in place of\n"
    "        // the stock upstream RVV vl=8 vec_dot. Byte-exact vs stock (M-sweep VERIFY_FAIL=0).\n"
    "        static volatile int weft_q5k_announced = 0;\n"
    "        if (!weft_q5k_announced) { weft_q5k_announced = 1;\n"
    "            fprintf(stderr, \"TCRV G7-S3 WEFT q5_K block-dot ENGAGED (n=%d nrc=%d)\\n\", n, nrc); }\n"
    "        weft_q5k_block_dot((size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);\n"
    "        return;\n"
    "    }\n"
    "#endif\n"
)
q = q[:at] + BRANCH + q[at:]

open(Q,"w").write(q)
print("PATCH OK (G7 §3 q5_K weft block-dot deploy: include + intercept-call + banner)")
print("quants.c md5:", md5(Q))
print("include present:", 'weft_q5k_blockdot.inc' in q)
print("intercept-call present:", "weft_q5k_block_dot((size_t)n" in q)
print("banner present:", "TCRV G7-S3 WEFT q5_K block-dot ENGAGED" in q)
