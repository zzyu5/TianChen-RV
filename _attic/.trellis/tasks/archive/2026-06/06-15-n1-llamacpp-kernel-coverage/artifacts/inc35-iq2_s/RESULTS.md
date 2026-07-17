# INC-35 — iq2_s × q8_K byte-exact: the GRID-codebook qh-plane sibling (16th ggml dot kernel)

## Summary

16th ggml dot kernel: **`ggml_vec_dot_iq2_s_q8_K`** — the THIRD member of the deep-IQ
GRID-codebook class, the SIBLING of iq2_xs (INC-34). New op
`tcrv_rvv.iq2_s_q8_k_block_dot`, fully STRUCTURED (raw()=0; I5). It REUSES the GRID + sign
mechanism — the packed-uint64 GRID codebook (indexed pointer lookup `grid_i8 + idx*8` +
`vle8(8)`, NOT vrgather), the `kmask` sign-bit plane (broadcast / `vand` / `vmsne` /
`vmerge(grid, vneg(grid))`), the signed widening product + chained `vwredsum`, the
integer-domain `bsum` fold, the explicit per-sub-block `scales[]` two-half split, and the
trailing `0.125f` factor — with THREE structural deltas.

Byte-exact `*s` vs ggml's OWN `_generic` (`quants.c:947-997`) on real `ssh rvv`
(`-march=rv64gcv`, Zvl128b ⇒ VLEN≥128): **2503/2503 cases, 0 failures** at `-ffp-contract=off`
AND `fast`; the named march case drives the 10-bit grid index to **max idx = 1023 (hit YES)**.
FOUR negative controls + a compiler-knob control prove every mechanism is live.

## The three structural deltas vs iq2_xs

iq2_s is **NOT** a grid-size swap of iq2_xs. The three deltas (read from the exact
`_generic`, `quants.c:970-988`):

- **(a) 1024-entry grid, 10-bit index.** `iq2s_grid[1024]` (vs iq2_xs's 512). Index range
  goes to 1023 — emitted as `static const int64_t tcrv_iq2s_grid[1024]` (ggml's exact hex
  literals, byte-viewed via `(const int8_t *)`).
- **(b) 10-bit index assembled from a single qs byte + a qh-bit plane.** The low 8 bits are a
  single byte `qs[l]` (a `uint8_t`, NOT a uint16 word); the high 2 bits are injected from a
  per-sub-block `qh[ib32]` byte: `idx = qs[l] | ((qh[ib32] << (8-2*l)) & 0x300)`. The shift
  is `8,6,4,2` for `l = 0,1,2,3` (the q5_K-style qh-plane pattern — each of the 4 groups
  takes its 2 high bits from a different field of the SAME qh byte). There is **NO** uint16
  reassembly, **NO** `& 511` / `>> 9` sign-selector read.
- **(c) EXPLICIT signs read directly from a sign region inside qs[64].** The sign byte for
  group l is `signs[l]` where `signs = qs + QK_K/8 = qs + 32` — i.e. the **upper half of the
  64-byte qs array**, NOT a `ksigns_iq2xs[128]` lookup. iq2_s carries **NO ksigns plane**
  (the op/verifier/lowering drop it entirely). bit j (via the SAME `kmask {1<<j}`) selects
  the per-lane sign.

The explicit per-sub-block `scales[]` two-half split is **REUSED** from iq2_xs: `ls1 =
2*(sc&0xf)+1` for groups 0,1; `ls2 = 2*(sc>>4)+1` for groups 2,3; two `sumi` resets, two
`bsum += sumi*ls` contributions; `q8` advances 8 per group CONTINUOUSLY across the half
boundary.

Block format: `block_iq2_s = { ggml_half d; uint8_t qs[64]; uint8_t qh[8]; uint8_t
scales[8] }`, stride **82** (= 2 + 64 + 8 + 8; verified by ggml's `static_assert` and a
native layout probe: d@0, qs@2, **signs@34**, qh@66, scales@74). Activation `block_q8_K`
unchanged (stride 292).

## (1) The op + lowering + reuse + files

### New op `tcrv_rvv.iq2_s_q8_k_block_dot`
- ODS: `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (`GgmlBlockDotIQ2SQ8KOp`). Four runtime
  ABI operands (vx/vy/s/n) + the vl token; the super-block-format structural attrs PLUS the
  TWO new offset attrs `weight_signs_byte_offset` (= 34, the explicit-sign-region delta) and
  `weight_qh_byte_offset` (= 66, the qh-plane delta); the `grid` (1024 int64) table as a
  typed array attr (I4 mirror). **NO `ksigns` attr** (deleted — iq2_s has no ksigns plane).
- Verifier: `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (`GgmlBlockDotIQ2SQ8KOp::verify`).
  Fail-closed (I7): pins `kind`, `scale_model`
  (`per-half-int4-explicit-scales-grid-codebook-qh-plane-explicit-signs-int-domain`), qk=256,
  sub_block=32, stride=**82**, qs@2, **signs@34**, **qh@66**, **scales@74**, act stride=292,
  act qs@4, grid size==**1024**, the operand C types, the i32m1 result, the policy on the
  enclosing with_vl. Rejects any forbidden/unknown attr; **no ksigns size check** (no ksigns).
- Lowering: `lib/Conversion/RVV/RVVToEmitC.cpp` — `isIQ2SQ8KBlockDotBody` recognizer, a
  dispatch branch (inserted right after the iq2_xs branch, before mxfp4), and the
  `emitIQ2SQ8KBlockDot` emitter (a self-contained parallel of `emitIQ2XSQ8KBlockDot`, NOT a
  shared-helper refactor — preserves "siblings byte-identical / additive").

### Reuse of the GRID + sign mechanism (identical)
The per-group decode is the iq2_xs path: indexed grid lookup (`grid_i8 + idx*8` + `vle8`),
the sign apply (`vmv` broadcast / `vand` kmask / `vmsne 0` / `vmerge(grid, vneg(grid), m)`),
the signed widening product (`vwmul_i16m2`, each lane ≤ 43·127 = 5461 < 32767), the chained
`vwredsum` i32 reduce, the explicit two-half `scales[]` split, the per-super-block fp32 fold
(ONE `emitc.expression` `sumf + d*(float)bsum`), the `*s = 0.125f*sumf` store. Only the index
source (single qs byte + qh injection), grid size, and the sign source (explicit memory vs
ksigns) differ.

### Files
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (+ `GgmlBlockDotIQ2SQ8KOp`)
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (+ `GgmlBlockDotIQ2SQ8KOp::verify`)
- `lib/Conversion/RVV/RVVToEmitC.cpp` (+ recognizer/branch/`emitIQ2SQ8KBlockDot`)
- `test/Conversion/RVV/rvv-to-emitc-iq2-s-q8-k-block-dot.mlir` (NEW conversion FileCheck)
- `test/Dialect/RVV/iq2-s-q8-k-block-dot-dataflow.mlir` (NEW dialect dataflow + 6 negatives)

Diff is purely **additive**: 976 insertions across the 3 source files + 2 new test files,
**0 deletions**.

## (2) Byte-exact vs ggml iq2_s (cases, 0 failures)

`ssh rvv`, `-march=rv64gcv`, clang 18.1.3, both `-ffp-contract=off` and `=fast`:

```
iq2_s byte-exact: 2503/2503 passed, 0 failed (generic delta 0/2503)
grid-index coverage (march case): max idx = 1023, hit 1023 = YES
```

Cases: 2400 random (n ∈ {256,512,768,1024,2048,4096,8192,16384}, full-random qs index bytes +
explicit sign bytes + qh bytes + scale bytes) + 100 q8 edges (+127, ±127) + 3 named edges:
- `grid1023-march+{scale0,scale31,ls1!=ls2}` — the 10-bit index **marches the FULL [0,1023]
  range via the qh-bit plane** (stride-4 covers all four `>>8` high-bit combos; the last group
  forced to exactly **1023**), with scale extremes (ls=1, ls=31) AND distinct `ls1 != ls2`
  nibbles. A coverage assertion (`max idx >= 1023 && hit 1023`) gates the RESULT so the qh
  injection is provably exercised, not silently masked.
- random cases additionally drive idx 0..1023 across ~600k groups (full-random qh), so
  byte-exact over thousands of words independently proves the qh injection + explicit signs.

**Negative controls (all DIVERGE = all mechanisms load-bearing):**
```
NEG-CTRL wrong-grid:                          ours=-289.795197 wrong=-287.928741 -> DIVERGES
NEG-CTRL wrong-signs (explicit):              ours=-289.795197 wrong=-2494.8186  -> DIVERGES
NEG-CTRL wrong-scale (drop +1):               ours=-289.795197 wrong=-135.4823   -> DIVERGES
NEG-CTRL wrong-qh (drop high-index plane):    ours=-289.795197 wrong= 463.998993 -> DIVERGES
```
The **wrong-qh** control (drops the `(qh<<(8-2*l))&0x300` injection so idx = `qs[l]` only) is
the iq2_s-specific discriminator: with qh forced live (0xff in every group) the large
divergence (464 vs −290) proves the qh 2-bit high-index plane (delta b) is genuinely
implemented. The **wrong-signs** control clears bit 7 of every EXPLICIT sign byte (the
sign-from-memory delta c, distinct from iq2_xs's wrong-ksigns).

**Compiler-knob control (attr → emitted const → HW), `-ffp-contract=off`:**
```
REAL-attr kernel      = 3053.45361   (real-table ref = 3053.45361) -> MATCH
PERTURBED-attr kernel = 3047.76123   (pert-table ref = 3047.76123) -> MATCH
real-attr vs perturbed-attr -> DIVERGES (attr is the live knob)
RESULT: KNOB LIVE (attr -> const -> HW)
```
The perturbed kernel (grid[5] byte0 +1 edited in the **typed body attr**, regenerated by the
compiler — `0x...0819` → `0x...081a` in the emitted const) computes a different `*s`, each
matching its own table — proving the 1024-entry grid DenseArray attr flows `attr → emitted
const → HW` (not a dead hardcoded table).

## (3) raw() = 0 + structured (I5)

`grep -c "raw("` of the compiler-emitted kernel = **0**. Every executable value is an emitc
node: 13 distinct `__riscv_*` intrinsics (all structured `CallOpaque`), the grid + kmask as
`static const` decls (NO ksigns table), the 1024-grid byte-view + indexed lookup as
structured pointer arith, the qh 2-bit injection `(qhb << {8,6,4,2}) & 768 | idx` as
structured int ops, the explicit sign region read at `xb + 34` (NO ksigns lookup), the
explicit scales `+ 74` with `& 15` / `>> 4` two-scale split. See `structured_proof.txt`. The
kernel is emitted ONLY by the compiler (`tcrv-opt --tcrv-rvv-lower-to-emitc | mlir-translate-20
--mlir-to-cpp`), byte-identical to a fresh regen.

## (4) lit + reds

- 2 NEW lit tests PASS (conversion FileCheck + dialect dataflow with 6 fail-closed negatives:
  unknown kind, wrong scale_model, wrong stride, wrong signs offset, wrong qh offset, wrong
  grid size).
- `ninja check-tianchenrv`: **660 tests, 657 pass, exactly 3 reds** (was 658/655/3 at the
  iq2_xs baseline + 2 new = 660/657/3). The 3 reds are the PRE-EXISTING
  `rvv-generated-bundle-abi-e2e` strided computed-mask widening dot-reduce dry-runs + e2e
  self-test — unrelated to iq2/grid/codebook (my diff touches only 3 source files, none of
  which relate to those tests).
- **Full CLEAN rebuild: green** (all 240 targets link, incl. tcrv-opt + tcrv-translate + the
  plugin/dialect tests; from `ninja clean`).
- Siblings byte-identical (additive): the iq2_xs emitted kernel regenerates byte-IDENTICAL
  post-change; the iq2_xs dialect+conversion lit tests still pass; mxfp4 still lowers cleanly
  (recognizer unaffected).

## (5) The qh 9th/10th-bit plane + explicit signs vs iq2_xs's ksigns (the deltas, restated)

| | iq2_xs (INC-34) | iq2_s (INC-35) |
|---|---|---|
| grid size | 512 | **1024** |
| index source | uint16 word `q2[l] & 511` (9-bit) | **single byte `qs[l]` + 2 qh bits → 10-bit** |
| qh plane | none | **`(qh[ib32] << (8-2*l)) & 0x300`** (q5_K-style, shifts 8/6/4/2) |
| signs | `ksigns_iq2xs[q2[l] >> 9]` (table lookup) | **explicit byte `signs[l]` from qs+32 (memory)** |
| ksigns table | `uint8_t[128]` attr + decl | **NONE (deleted)** |
| scale | explicit `scales[8]`: ls1/ls2 (two-half) | **explicit `scales[8]`: ls1/ls2 (REUSED)** |
| weight stride | 74 | **82** |

The "9th-bit" shorthand in the task is loose: iq2_s injects **2** high bits (bits 8 and 9) per
group from the qh plane, giving the 10-bit index into the 1024-grid — exactly ggml's
`qs[l] | ((qh[ib32] << (8-2*l)) & 0x300)`.

## (6) Remaining IQ tail

- **iq3_xxs** — 256-entry grid of **4 int8** (not 8) + the iq2_xxs-style packed aux scale/sign.
  The "grid of 4" changes the per-group lane count (4 not 8) — a new vwmul width, NOT just a
  grid-size swap.
- **iq3_s** — the CLOSEST successor to iq2_s: a 512-entry grid of **4 int8** + a `qh` high-bit
  plane for the index (same q5_K-style mechanism INC-35 just built) + an EXPLICIT `signs` byte
  stream (the sign-from-memory mechanism INC-35 just built) + explicit `scales`. iq2_s's qh
  injection + explicit-sign decode transfer almost directly; the new fact is the grid-of-4
  lane width.
- **iq1_s / iq1_m** — 2048-entry ternary grid (different value domain: {-1,0,1}-ish) + a
  per-block delta/shift; the ternary class, structurally furthest.
