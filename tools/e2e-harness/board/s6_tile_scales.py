#!/usr/bin/env python3
# S6 stack-panel tile for the plain q3_K repack GEMM export: stage every decoded
# signed 6-bit scale vector (vsext_vf2_i16m1 -> vint16m1_t) through a stack panel
# via vse16/vle16, deferring the reload to JUST BEFORE the scale's first use
# (the vwmacc_vv_i32m2 scale x weight-partial accumulation). This is the q3_K
# analogue of the q6_K S6 no-min emitter (decode-scale int16 stack panel; NO min
# fold -- q3_K has no min). Byte-exact by construction (reloaded value == stored
# value == original decode); it only cuts the scale live range so the register
# allocator need not keep the 32 scale vectors resident across the ~4400-line
# fully-unrolled weight-reconstruction body. Deterministic, idempotent-safe.
import re, sys

src = sys.argv[1]; dst = sys.argv[2]
with open(src) as f:
    lines = f.readlines()

decl_re = re.compile(r'^\s*vint16m1_t (v\d+) = __riscv_vsext_vf2_i16m1\(')
# collect scale decls in order
scales = []  # (decl_line_idx, varname)
for i, ln in enumerate(lines):
    m = decl_re.match(ln)
    if m:
        scales.append((i, m.group(1)))

assert scales, "no scale decls found"
# assign a unique panel slot per scale var (dedupe by name; names are unique here)
slot = {}
for _, v in scales:
    if v not in slot:
        slot[v] = len(slot)
NSLOT = len(slot)

# For each scale var: find first USE line strictly after its decl (non-comment,
# real code reference of the whole word), where it is consumed (vwmacc_vv etc.).
word = {v: re.compile(r'\b' + re.escape(v) + r'\b') for v in slot}
first_use = {}
for decl_i, v in scales:
    for j in range(decl_i + 1, len(lines)):
        s = lines[j]
        if s.lstrip().startswith('//'):
            continue
        if word[v].search(s):
            first_use[v] = j
            break
    assert v in first_use, f"no use found for {v}"

# Build insertion lists keyed by original line index.
# store goes AFTER decl line; reload goes BEFORE first-use line.
after = {}   # idx -> list of lines to append after original line idx
before = {}  # idx -> list of lines to insert before original line idx
for decl_i, v in scales:
    sl = slot[v]
    store = f"        __riscv_vse16_v_i16m1(&__q3k_scale_panel[{sl}*8], {v}, 8); // S6-tile: stage scale to stack panel\n"
    reload_ = f"        {v} = __riscv_vle16_v_i16m1(&__q3k_scale_panel[{sl}*8], 8); // S6-tile: reload scale from stack panel\n"
    after.setdefault(decl_i, []).append(store)
    before.setdefault(first_use[v], []).append(reload_)

# Find function opening brace line to declare the panel array (function-scope,
# outside the loops, reused each K-block iteration).
func_open = None
for i, ln in enumerate(lines):
    if ln.startswith('extern "C" void tcrv_emitc_ggml_repack_gemm_q3_K_q8_K_kernel'):
        func_open = i
        break
assert func_open is not None
panel_decl = f"  int16_t __q3k_scale_panel[{NSLOT}*8]; // S6-tile stack panel ({NSLOT} signed-scale strips)\n"

out = []
for i, ln in enumerate(lines):
    if i in before:
        out.extend(before[i])
    out.append(ln)
    if i == func_open:
        out.append(panel_decl)
    if i in after:
        out.extend(after[i])

with open(dst, 'w') as f:
    f.writelines(out)

print(f"scales staged={len(scales)} slots={NSLOT} lines {len(lines)}->{len(out)}")
