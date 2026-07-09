#!/usr/bin/env python3
# [G3-minterm-fix M1] CONFIRM-ONLY instrumenter. Takes the FRESH re-emitted S6 q4_K
# repack-GEMM kernel (md5 90d454da) and produces an INSTRUMENTED copy that additionally
# DUMPS the per-block / per-row / per-column min-term intermediate values for the FIRST
# row-group (v12==0) FIRST weight-col-group (v16==0) tile0 (weight cols 0-7):
#   - integer MAIN accumulator  v47/v49/v51/v53  (rows 0..3)         -> tcrv_main_i[blk][r][c]
#   - integer MIN accumulator   v46[r*8+c]        (rows 0..3, cols 0..7) -> tcrv_min_i[blk][r][c]
#   - float d    per col   v5765  -> tcrv_d[blk][c]
#   - float dmin per col   v5762  -> tcrv_dmin[blk][c]
#   - float a_d  per row   v34/v37/v40/v43 -> tcrv_ad[blk][r]
# ADDS ONLY fprintf-free memory captures (pure observation, ZERO change to the vector math).
# The captured arrays are DEFINED (extern "C", default visibility) in the kernel TU so the
# driver reads them from libggml-cpu.so directly. NOT emitter source. Kernel body math UNCHANGED.
import sys, hashlib
SRC = "/tmp/m1_minterm/fresh_q4K.inc"
DST = "/tmp/m1_minterm/instr_q4K.inc"

def md5(b): return hashlib.md5(b).hexdigest()

s = open(SRC).read()
assert md5(s.encode()) == "90d454da655f2fc1f88435d2d5826942", "fresh kernel md5 mismatch -- refuse to instrument"

# ---- 1. prepend capture globals (extern "C" so no C++ mangling; defined once, .inc is included once) ----
inc_anchor = "#include <riscv_vector.h>\n"
assert s.count(inc_anchor) == 1
globals_blk = (
    "#include <riscv_vector.h>\n"
    "/* ==== [G3 M1 instrumentation: pure-observation min-term capture] ==== */\n"
    "#define TCRV_VIS __attribute__((visibility(\"default\")))\n"
    "extern \"C\" {\n"
    "  TCRV_VIS int      tcrv_cap_want   = 0;   /* driver sets 1 only around the case it wants */\n"
    "  TCRV_VIS int      tcrv_cap_nblk   = 0;   /* number of blocks captured (K/256) */\n"
    "  TCRV_VIS int      tcrv_cap_hits   = 0;   /* diagnostic: how many capture events fired */\n"
    "  TCRV_VIS int      tcrv_main_i[8][4][8];  /* [blk][row][col] integer main accumulator (v47..v53) */\n"
    "  TCRV_VIS int      tcrv_min_i [8][4][8];  /* [blk][row][col] integer min  accumulator (v46)      */\n"
    "  TCRV_VIS float    tcrv_d     [8][8];     /* [blk][col] d    fp16->f32 (v5765) */\n"
    "  TCRV_VIS float    tcrv_dmin  [8][8];     /* [blk][col] dmin fp16->f32 (v5762) */\n"
    "  TCRV_VIS float    tcrv_ad    [8][4];     /* [blk][row] activation scale a_d   */\n"
    "  TCRV_VIS short    tcrv_bsums [8][64];    /* [blk][g16*4+row] raw q8 bsums region the kernel reads (v32+1040) */\n"
    "}\n"
)
s = s.replace(inc_anchor, globals_blk, 1)

# ---- 2. inject tile0 capture right after row3 fold result `v26 = v5805;` (unique anchor) ----
cap_anchor = "        v26 = v5805;\n"
assert s.count(cap_anchor) == 1, f"cap anchor count={s.count(cap_anchor)}"
cap_blk = (
    "        v26 = v5805;\n"
    "        /* ==== [G3 M1] tile0 (weight cols 0..7) min-term capture, group0 row0..3, this block ==== */\n"
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
    "          /* raw stderr backup dump (robust vs symbol-export) */\n"
    "          for (int _r = 0; _r < 4; ++_r) {\n"
    "            fprintf(stderr, \"KCAP blk=%d row=%d ad=%.6f\", (int)v28, _r, tcrv_ad[v28][_r]);\n"
    "            fprintf(stderr, \" MAIN\"); for (int _c=0;_c<8;++_c) fprintf(stderr, \" %d\", tcrv_main_i[v28][_r][_c]);\n"
    "            fprintf(stderr, \" MIN\");  for (int _c=0;_c<8;++_c) fprintf(stderr, \" %d\", tcrv_min_i[v28][_r][_c]);\n"
    "            if (_r==0){ fprintf(stderr, \" D\"); for(int _c=0;_c<8;++_c) fprintf(stderr, \" %.6f\", tcrv_d[v28][_c]);\n"
    "                        fprintf(stderr, \" DMIN\"); for(int _c=0;_c<8;++_c) fprintf(stderr, \" %.6f\", tcrv_dmin[v28][_c]); }\n"
    "            fprintf(stderr, \"\\n\");\n"
    "          }\n"
    "        }\n"
)
s = s.replace(cap_anchor, cap_blk, 1)

open(DST, "w").write(s)
print("INSTRUMENT OK ->", DST)
print("dst md5:", md5(s.encode()))
print("dst lines:", s.count("\n"))
print("globals present:", "tcrv_min_i" in s, "capture present:", "G3 M1] tile0" in s)
