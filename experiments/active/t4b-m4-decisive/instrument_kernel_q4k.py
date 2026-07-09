#!/usr/bin/env python3
# [G3-cert-hardening M4] CONFIRM-ONLY instrumenter (VERBATIM logic from M1's instrument_kernel.py;
# only the scratch paths changed). Takes the FRESH re-emitted S6 q4_K repack-GEMM kernel
# (md5 90d454da) and produces an INSTRUMENTED copy that DUMPS the per-block/row/col integer
# MAIN accumulator (v47/v49/v51/v53), integer MIN accumulator (v46), fp16 d/dmin and a_d for the
# FIRST row-group (v12==0) FIRST weight-col-group (v16==0) tile0 (weight cols 0..7). ADDS ONLY
# pure-observation memory captures (ZERO change to the vector math). NOT emitter source.
import hashlib
SRC = "/tmp/m4_decisive/fresh_q4K.inc"
DST = "/tmp/m4_decisive/instr_q4K.inc"
def md5(b): return hashlib.md5(b).hexdigest()

s = open(SRC).read()
assert md5(s.encode()) == "90d454da655f2fc1f88435d2d5826942", "fresh kernel md5 mismatch -- refuse to instrument"

inc_anchor = "#include <riscv_vector.h>\n"
assert s.count(inc_anchor) == 1
globals_blk = (
    "#include <riscv_vector.h>\n"
    "/* ==== [G3 M4 instrumentation: pure-observation min-term capture] ==== */\n"
    "#define TCRV_VIS __attribute__((visibility(\"default\")))\n"
    "extern \"C\" {\n"
    "  TCRV_VIS int      tcrv_cap_want   = 0;\n"
    "  TCRV_VIS int      tcrv_cap_nblk   = 0;\n"
    "  TCRV_VIS int      tcrv_cap_hits   = 0;\n"
    "  TCRV_VIS int      tcrv_main_i[8][4][8];\n"
    "  TCRV_VIS int      tcrv_min_i [8][4][8];\n"
    "  TCRV_VIS float    tcrv_d     [8][8];\n"
    "  TCRV_VIS float    tcrv_dmin  [8][8];\n"
    "  TCRV_VIS float    tcrv_ad    [8][4];\n"
    "  TCRV_VIS short    tcrv_bsums [8][64];\n"
    "}\n"
)
s = s.replace(inc_anchor, globals_blk, 1)

cap_anchor = "        v26 = v5805;\n"
assert s.count(cap_anchor) == 1, f"cap anchor count={s.count(cap_anchor)}"
cap_blk = (
    "        v26 = v5805;\n"
    "        /* ==== [G3 M4] tile0 (weight cols 0..7) min-term capture, group0 row0..3, this block ==== */\n"
    "        if (tcrv_cap_want && v12 == 0 && v16 == 0 && (int)v28 < 8) {\n"
    "          __riscv_vse32_v_i32m2(tcrv_main_i[v28][0], v47, 8);\n"
    "          __riscv_vse32_v_i32m2(tcrv_main_i[v28][1], v49, 8);\n"
    "          __riscv_vse32_v_i32m2(tcrv_main_i[v28][2], v51, 8);\n"
    "          __riscv_vse32_v_i32m2(tcrv_main_i[v28][3], v53, 8);\n"
    "          for (int _c = 0; _c < 8; ++_c) {\n"
    "            tcrv_min_i[v28][0][_c] = v46[_c];\n"
    "            tcrv_min_i[v28][1][_c] = v46[8 + _c];\n"
    "            tcrv_min_i[v28][2][_c] = v46[16 + _c];\n"
    "            tcrv_min_i[v28][3][_c] = v46[24 + _c];\n"
    "          }\n"
    "          __riscv_vse32_v_f32m2(tcrv_d[v28],    v5765, 8);\n"
    "          __riscv_vse32_v_f32m2(tcrv_dmin[v28], v5762, 8);\n"
    "          tcrv_ad[v28][0] = v34; tcrv_ad[v28][1] = v37; tcrv_ad[v28][2] = v40; tcrv_ad[v28][3] = v43;\n"
    "          for (int _i = 0; _i < 64; ++_i) tcrv_bsums[v28][_i] = *(const short *)(v32 + 1040 + 2 * _i);\n"
    "          if ((int)v28 + 1 > tcrv_cap_nblk) tcrv_cap_nblk = (int)v28 + 1;\n"
    "          ++tcrv_cap_hits;\n"
    "        }\n"
)
s = s.replace(cap_anchor, cap_blk, 1)

open(DST, "w").write(s)
print("INSTRUMENT OK ->", DST)
print("dst md5:", md5(s.encode()))
print("globals present:", "tcrv_min_i" in s, "capture present:", "G3 M4] tile0" in s)
