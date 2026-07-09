#!/usr/bin/env python3
# t4b_q4k_gate.py -- surgically toggle ONLY the q4_K @VLEN128 repack selector gate
# in tcrv-llamacpp ggml/src/ggml-cpu/repack.cpp. Idempotent, verifies context.
# usage: t4b_q4k_gate.py {on|off}
import sys, re
REPACK = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/repack.cpp"
mode = sys.argv[1]
OFF = "                case 128:  { break; } // TODO"
ON  = "                case 128:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; } /* TCRV-Q4K-SEAL */"
src = open(REPACK).read().splitlines()
# find the q4_K block: the "GGML_TYPE_Q4_K" branch, then its riscv switch's case 128 line
qk = None
for i,l in enumerate(src):
    if "cur->type == GGML_TYPE_Q4_K" in l:
        qk = i; break
assert qk is not None, "Q4_K branch not found"
# within next ~40 lines find the case 128 line that precedes 'return &q4_K_16x1_q8_K'
target = None
for i in range(qk, min(qk+50, len(src))):
    if "case 128:" in src[i] and ("q4_K_16x1_q8_K" in src[i+1] or "q4_K_16x1_q8_K" in "".join(src[i:i+3])):
        target = i; break
assert target is not None, "q4_K case 128 gate not found"
cur = src[target]
want = ON if mode == "on" else OFF
src[target] = want
open(REPACK,"w").write("\n".join(src)+"\n")
print(f"line {target+1}: set q4_K gate {mode.upper()}")
print(f"  was : {cur.strip()}")
print(f"  now : {want.strip()}")
