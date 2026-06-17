# INC-34 — iq2_xs × q8_K byte-exact: the GRID-codebook sibling (15th ggml dot kernel)

## Summary

15th ggml dot kernel: **`ggml_vec_dot_iq2_xs_q8_K`** — the SECOND member of the deep-IQ
GRID-codebook class, the SIBLING of iq2_xxs (INC-33). New op
`tcrv_rvv.iq2_xs_q8_k_block_dot`, fully STRUCTURED (raw()=0; I5). It REUSES iq2_xxs's
grid + sign mechanism — the packed-uint64 GRID codebook (indexed pointer lookup
`grid_i8 + idx*8` + `vle8(8)`, NOT vrgather), the `ksigns_iq2xs[128]`/`kmask` SIGN plane
(broadcast / `vand` / `vmsne` / `vmerge(grid, vneg(grid))`), the integer-domain `bsum`
fold, and the trailing `0.125f` factor — with THREE structural deltas.

Byte-exact `*s` vs ggml's OWN `_generic` (`quants.c:897-945`) on real `ssh rvv`
(`-march=rv64gcv`, VLEN=128): **2503/2503 cases, 0 failures** at `-ffp-contract=off` AND
`fast`. FOUR negative controls + a compiler-knob control prove every mechanism is live.

## The three structural deltas vs iq2_xxs

iq2_xs is **NOT** a grid-size swap of iq2_xxs. The three deltas (read from the exact
`_generic`):

- **(a) 512-entry grid, 9-bit index.** `iq2xs_grid[512]` (vs 256). Index range goes to
  511 — emitted as `static const int64_t tcrv_iq2xs_grid[512]` (ggml's exact hex
  literals, byte-viewed via `(const int8_t *)`).
- **(b) 9-bit index read DIRECTLY from each uint16 `qs` word.** Each `qs[]` is a `q2[l]`
  uint16: low 9 bits = grid index `q2[l] & 511`, high 7 bits = sign selector
  `q2[l] >> 9`. There is **NO** aux1 word — no `2*(aux1>>28)+1` packed scale, no
  `(aux1>>7*l)&127` packed selector. The uint16 is reassembled from 2 little-endian byte
  loads in the UNSIGNED domain (`& 511u` / `>> 9u` are logical, alignment-safe, no
  `*(uint16_t*)`).
- **(c) explicit per-sub-block `scales[]` array with a two-scale split.** A new byte
  stream `scales[QK_K/32]` at offset 66: `ls1 = 2*(sc & 0xf)+1` applied to groups l=0,1;
  `ls2 = 2*(sc >> 4)+1` applied to groups l=2,3. So each sub-block has **TWO** `sumi`
  resets and **TWO** `bsum += sumi*ls` contributions (NOT one chained 4-group
  reduction). `q8` advances 8 per group CONTINUOUSLY across the half boundary — only the
  sumi/bsum bookkeeping splits.

Block format: `block_iq2_xs = { ggml_half d; uint16_t qs[32]; uint8_t scales[8] }`,
stride **74** (= 2 + 64 + 8; verified by ggml's `static_assert`). Activation `block_q8_K`
unchanged (stride 292). `ksigns_iq2xs[128]` is the **same** table as iq2_xxs.

## (1) The op + lowering + reuse

### New op `tcrv_rvv.iq2_xs_q8_k_block_dot`
- ODS: `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (`GgmlBlockDotIQ2XSQ8KOp`). Four
  runtime ABI operands (vx/vy/s/n) + the vl token; the super-block-format structural
  attrs PLUS a NEW `weight_scales_byte_offset` attr (= 66, the explicit-scales delta);
  the `grid` (512 int64) + `ksigns` (128 int32) tables as typed array attrs (I4 mirror).
- Verifier: `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (`::verify()`). Fail-closed
  (I7): pins `kind`, `scale_model` (`per-half-int4-explicit-scales-grid-codebook-int-domain`),
  qk=256, sub_block=32, stride=74, qs@2, **scales@66**, act stride=292, act qs@4,
  grid size==**512**, ksigns size==128, the operand C types, the i32m1 result, the
  policy on the enclosing with_vl. Rejects any forbidden/unknown attr.
- Lowering: `lib/Conversion/RVV/RVVToEmitC.cpp` — `isIQ2XSQ8KBlockDotBody` recognizer,
  a dispatch branch (inserted right after the iq2_xxs branch, before mxfp4), and the
  `emitIQ2XSQ8KBlockDot` emitter (a self-contained parallel of `emitIQ2XXSQ8KBlockDot`,
  NOT a shared-helper refactor — preserves "siblings byte-identical / additive").

### Reuse of iq2_xxs (the grid + sign mechanism, identical)
The per-group decode is byte-for-byte the iq2_xxs path: indexed grid lookup
(`grid_i8 + idx*8` + `vle8`), the sign plane apply (`vmv` broadcast / `vand` kmask /
`vmsne 0` / `vmerge(grid, vneg(grid), m)`), the signed widening product
(`vwmul_i16m2`, each lane ≤ 43·127 = 5461 < 32767), the chained `vwredsum` i32 reduce,
the per-super-block fp32 fold (ONE `emitc.expression` `sumf + d*(float)bsum`), the
`*s = 0.125f*sumf` store. Only the index source, grid size, and scale structure differ.

### Files
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (+ `GgmlBlockDotIQ2XSQ8KOp`)
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (+ `GgmlBlockDotIQ2XSQ8KOp::verify`)
- `lib/Conversion/RVV/RVVToEmitC.cpp` (+ recognizer/branch/`emitIQ2XSQ8KBlockDot`)
- `test/Conversion/RVV/rvv-to-emitc-iq2-xs-q8-k-block-dot.mlir` (NEW conversion FileCheck)
- `test/Dialect/RVV/iq2-xs-q8-k-block-dot-dataflow.mlir` (NEW dialect dataflow + 7 negatives)

## (2) Byte-exact vs ggml iq2_xs (cases, 0 failures)

`ssh rvv`, `-march=rv64gcv`, clang 18.1.3, both `-ffp-contract=off` and `=fast`:

```
iq2_xs byte-exact: 2503/2503 passed, 0 failed (generic delta 0/2503)
```

Cases: 2400 random (n ∈ {256,512,768,1024,2048,4096,8192,16384}, full-random 16-bit qs +
random scale bytes) + 100 q8 edges (+127, ±127) + 3 named edges:
- `grid511-march` — **9-bit indices stride-2 span the FULL [0,511] range** (the key
  iq2_xs delta; a low-8-bit-mask bug would diverge on every index ≥ 256), scale extremes
  (ls=1, ls=31) AND distinct `ls1 != ls2` nibbles.
- random cases additionally cover ≥256 indices on ~half of all qs words, so byte-exact
  over thousands of words independently proves the 9-bit index is honored.

**Negative controls (all DIVERGE = all mechanisms load-bearing):**
```
NEG-CTRL wrong-grid:               ours=-1398.55298 wrong=-1590.43457 -> DIVERGES
NEG-CTRL wrong-signs:              ours=-1398.55298 wrong= 1249.69092 -> DIVERGES
NEG-CTRL wrong-scale (drop +1):    ours=-1398.55298 wrong=-1346.65125 -> DIVERGES
NEG-CTRL collapse-split (ls2=ls1): ours=-1398.55298 wrong=  115.06172 -> DIVERGES
```
The **collapse-split** control (115.06 vs −1398.55) is the iq2_xs-specific discriminator:
it forces `ls2 = ls1` (the iq2_xxs-style single scale), and the large divergence proves
the two-scale half split (delta c) is genuinely implemented, not copied.

**Compiler-knob control (attr → emitted const → HW):**
```
REAL-attr kernel      = 49.970417   (real-table ref = 49.970417) -> MATCH
PERTURBED-attr kernel =  7.96230602 (pert-table ref =  7.96230602) -> MATCH
real-attr vs perturbed-attr -> DIVERGES (attr is the live knob)
RESULT: KNOB LIVE (attr -> const -> HW)
```
The perturbed kernel (grid[5] byte0 +1, ksigns[5] 5→200 edited in the **typed body attr**,
regenerated by the compiler) computes a different `*s`, each matching its own table —
proving the grid/ksigns DenseArray attrs flow `attr → emitted const → HW` (not a dead
hardcoded table).

## (3) raw() = 0 + structured (I5)

`grep -c "raw("` of the compiler-emitted kernel = **0**. Every executable value is an
emitc node: 13 distinct `__riscv_*` intrinsics (all structured `CallOpaque`), the three
tables as `static const` decls, the 512-grid byte-view + indexed lookup as structured
pointer arith, the 9-bit index `& 511u` / sign selector `>> 9u` as logical ops, the
explicit scales `+ 66` with `& 15` / `>> 4` two-scale split. See `structured_proof.txt`.
The kernel is emitted ONLY by the compiler (`tcrv-opt --tcrv-rvv-lower-to-emitc |
mlir-translate-20 --mlir-to-cpp`), byte-identical to a fresh regen.

## (4) lit + reds

- 2 NEW lit tests PASS (conversion FileCheck + dialect dataflow with 7 fail-closed
  negatives: unknown kind, wrong scale_model, wrong stride, wrong scales offset, wrong
  grid size, wrong ksigns size, wrong output ctype).
- `ninja check-tianchenrv`: **658 tests, 655 pass, exactly 3 reds** (was 656/653/3 at the
  iq2_xxs baseline + 2 new = 658/655/3). The 3 reds are the PRE-EXISTING
  `rvv-generated-bundle-abi-e2e` strided computed-mask widening dot-reduce dry-runs +
  e2e self-test — unrelated to iq2/grid/codebook (my diff touches only 3 source files,
  none of which relate to those tests).
- Full clean build: green (all 131 targets link, incl. tcrv-translate + plugin tests).
- Siblings byte-identical (additive): iq2_xxs AND iq4_xs emitted kernels regenerate
  byte-IDENTICAL post-change; mxfp4 still lowers cleanly (recognizer unaffected).

## (5) The 9-bit index + explicit scales vs iq2_xxs (the deltas, restated)

| | iq2_xxs (INC-33) | iq2_xs (INC-34) |
|---|---|---|
| grid size | 256 | **512** |
| index source | aux0 byte (`aux8[l]`, 8-bit) | **uint16 word `q2[l] & 511` (9-bit)** |
| sign selector | `(aux1 >> 7*l) & 127` | **`q2[l] >> 9`** |
| scale | one packed `ls = 2*(aux1>>28)+1` per sub-block | **explicit `scales[8]`: ls1=2*(sc&0xf)+1, ls2=2*(sc>>4)+1** |
| reduction shape | 1 chained 4-group reduce | **2 half-reduces (sumi reset per half)** |
| weight stride | 66 | **74** |

## (6) Remaining IQ tail

- **iq2_s** — 1024-entry grid + a `qh` 9th-bit plane for the index + a sign-from-qs
  scheme + explicit scales (closest to iq2_xs, adds the qh high-bit plane like q5_K).
- **iq3_xxs** — 256-entry grid of 4 int8 (not 8) + the iq2_xxs-style packed aux scale/sign.
- **iq3_s** — 512-entry grid of 4 int8 + a `qh` high-bit + an explicit `signs` byte stream
  + `scales`.
- **iq1_s / iq1_m** — 2048-entry ternary grid (different value domain: {-1,0,1}-ish) +
  a per-block delta/shift; the ternary class, structurally furthest.
