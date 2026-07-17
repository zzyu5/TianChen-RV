# INC-37 — iq3_s × q8_K byte-exact; the iq3 GRID re-composition (18th dot kernel)

18th ggml dot kernel: `ggml_vec_dot_iq3_s_q8_K` — the iq3 GRID variant that RE-COMPOSES
three already-built mechanisms (it introduces NO new mechanism). It combines:
- **iq3_xxs's grid-of-4** (512 entries now): each grid entry = 4 int8, two 4-lane passes
  per sign group (grid1 lanes 0..3 kmask{1,2,4,8}, grid2 lanes 4..7 kmask{16,32,64,128}).
- **iq2_s's qh 9th-bit plane**: ONE high bit (mask 256 = bit 8) injected per index from a
  per-sub-block qh byte; the two passes take DIFFERENT shifts (8-2l grid1, 7-2l grid2).
- **iq2_s's EXPLICIT signs from memory**: the sign byte is read DIRECTLY from a dedicated
  32-byte `signs[]` region (NOT ksigns).
- **iq2_s's explicit per-sub-block 4-bit scales**: the two-nibble `scales[]` split.
- **NO trailing factor** (`*s = sumf` — unlike iq3_xxs's 0.25f / iq2_s's 0.125f).

New op `tcrv_rvv.iq3_s_q8_k_block_dot`, fully STRUCTURED (raw()=0; I5). Byte-exact `*s`
vs ggml's OWN `_generic` (`quants.c:1043-1097`) on real `ssh rvv` (`-march=rv64gcv`,
Zvl128b ⇒ VLEN≥128): **2464/2464 cases, 0 failures** at `-ffp-contract=off` AND `fast`;
the 9-bit grid index marches to **max idx = 511 (hit YES)**. FOUR negative controls +
a compiler-knob control prove every re-composed mechanism is live.

## (1) The op + lowering (512-grid-of-4 + qh + explicit signs/scales) + reuse + files

### New op `tcrv_rvv.iq3_s_q8_k_block_dot`
- ODS: `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (`GgmlBlockDotIQ3SQ8KOp`). Four
  runtime ABI operands (vx/vy/s/n) + the vl token; the super-block-format structural
  attrs (`weight_qs_byte_offset`=2, `weight_qh_byte_offset`=66, `weight_signs_byte_offset`
  =74, `weight_scales_byte_offset`=106 — the iq3_s SEPARATE-region layout) PLUS the `grid`
  (512 int32) table as a typed array attr (I4 mirror). **NO `ksigns` attr** (iq3_s has no
  ksigns plane), **NO `gas` offset** (iq3_s is not the iq3_xxs packed-aux layout).
- Verifier: `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (`GgmlBlockDotIQ3SQ8KOp::verify`).
  Fail-closed (I7): pins `kind` (`ggml_iq3_s_q8_k_block_dot`), `scale_model`
  (`per-sub-block-int4-explicit-scales-grid-of-4-codebook-qh-plane-explicit-signs-int-domain`),
  qk=256, sub_block=32, **stride=110** (sizeof block_iq3_s = 2+64+8+32+4), qs@2, **qh@66**,
  **signs@74**, **scales@106**, act stride=292, act qs@4, **grid size==512**, the operand C
  types, the i32m1 result, the policy on the enclosing with_vl. Rejects any forbidden/
  unknown attr; **no ksigns size check** (no ksigns).
- Lowering: `lib/Conversion/RVV/RVVToEmitC.cpp` — `isIQ3SQ8KBlockDotBody` recognizer, a
  dispatch branch (inserted right after the iq3_xxs branch, before mxfp4), and the
  `emitIQ3SQ8KBlockDot` emitter (a self-contained parallel of `emitIQ3XXSQ8KBlockDot` +
  `emitIQ2SQ8KBlockDot`, NOT a shared-helper refactor — preserves "siblings byte-identical
  / additive").

### Reuse (the task mandate "REUSE the built pieces")
- **From iq3_xxs**: the entire grid-of-4 `gridOf4Pass` body — the `vle8(4)` grid load over
  `grid_i8 + idx*4`, the sign apply (`vmv` broadcast / `vand` kmask / `vmsne 0` /
  `vmerge(grid, vneg(grid), m)`), the signed `vwmul_i16m2` product (each lane ≤ 15·127 =
  1905 < 32767), the chained `vwredsum` i32 reduce; the kmaskLo/kmaskHi split; the two-pass
  pairing of an 8-element sign group; the structured `uint32_t[512]` grid decl read via a
  `(const int8_t *)` byte cast; the per-super-block fp32 fold (ONE `emitc.expression`).
- **From iq2_s**: the qh-plane injection into the index, the EXPLICIT sign-from-memory read,
  the explicit two-nibble `scales[]` split, the q6_K-style super-block structure (q8_K
  activation, integer `bsum` fold).
- The ONLY iq3_s-specific deltas in the emitter: the index is assembled OUTSIDE the
  grid-of-4 helper (because the two passes take DIFFERENT qh shifts — 8-2l vs 7-2l, both
  masked with 256), the sign byte source (a dedicated signs region, not inside qs), and
  `*s = sumf` (no trailing factor).

### Files
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (+ `GgmlBlockDotIQ3SQ8KOp`)
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (+ `GgmlBlockDotIQ3SQ8KOp::verify`)
- `lib/Conversion/RVV/RVVToEmitC.cpp` (+ recognizer/branch/`emitIQ3SQ8KBlockDot`)
- `test/Conversion/RVV/rvv-to-emitc-iq3-s-q8-k-block-dot.mlir` (NEW conversion FileCheck)
- `test/Dialect/RVV/iq3-s-q8-k-block-dot-dataflow.mlir` (NEW dialect dataflow + 7 negatives)

Diff is purely **additive** (siblings byte-untouched).

## (2) Byte-exact vs ggml iq3_s — 0 failures

`ssh rvv`, riscv64, `-march=rv64gcv`, VLEN=128, clang 18.1.3. The compiler-emitted
kernel's `*s` is BIT-FOR-BIT equal (memcmp of the float bits) to ggml's **VERBATIM
`_generic`** (transcription cross-checked byte-identical to ggml's REAL `_generic` over
2000 host cases before any board run):

```
-ffp-contract=off : iq3_s byte-exact: 2464/2464 passed, 0 failed (generic delta 0/2464)
-ffp-contract=fast: iq3_s byte-exact: 2464/2464 passed, 0 failed (generic delta 0/2464)
grid-index coverage (edge): max idx = 511, hit 511 = YES
```

- n ∈ {256, 512, 768, 1024, 2048, 4096, 8192, 16384} (multiples of 256), 300 random reps
  (full-random qs index bytes + qh bytes + explicit sign bytes + scale bytes) + q8 edges
  (+127, ±127) + 64 edge blocks.
- The edge block forces qs=0xff + qh=0xff so the 9-bit index marches the FULL [0,511] range
  via the qh plane (a coverage assertion gates the RESULT on `max idx ≥ 511 && hit 511`, so
  the qh 9th-bit injection is PROVABLY exercised, not silently masked), with scale extremes
  (nibble 0 → ls=1, nibble 15 → ls=31) and distinct ls1≠ls2.

Byte-exact on the FIRST hardware run at both fp-contract modes.

**Negative controls — every re-composed mechanism is LIVE, load-bearing** (all DIVERGE, at
both fp-contract modes):

```
NEG-CTRL wrong-grid  (every grid byte +1)         ours=-482942.81 wrong=-523587.81 -> DIVERGES
NEG-CTRL wrong-signs (clear explicit sign bit)    ours=-482942.81 wrong=-131964.22 -> DIVERGES
NEG-CTRL wrong-scale (drop the +1)                ours=-482942.81 wrong=-461682.94 -> DIVERGES
NEG-CTRL wrong-qh    (drop the 9th-bit plane)     ours=-482942.81 wrong=-100410.02 -> DIVERGES
```

- **wrong-grid** swaps the 512-grid (every byte +1) → the grid-of-4 codebook is live.
- **wrong-signs** clears bit 7 of every EXPLICIT sign byte → the sign-from-memory read is live.
- **wrong-scale** drops the `+1` in `2*nibble+1` → the explicit two-nibble scale is live.
- **wrong-qh** drops the `(qh<<(8/7-2l))&256` injection (idx = qs only) → the large
  divergence (−100410 vs −482942) proves the qh 9th-bit high-index plane is genuinely
  implemented (the iq3_s-specific discriminator, the part NOT inherited wholesale).

**COMPILER-KNOB control — the attr is the live knob → const → HW** (the stronger N1/I4
evidence), `-ffp-contract=off`:

```
REAL-attr kernel      = 87244.55469   (real-table ref = 87244.55469) -> MATCH
PERTURBED-attr kernel = 101755.50000  (pert-table ref = 101755.50000) -> MATCH
real-attr vs perturbed-attr -> DIVERGES (attr is the live knob)
RESULT: KNOB LIVE (attr -> const -> HW)
```

A PERTURBED-attr typed body (grid[5]'s low byte +1, `0x01010301` → `0x01010302`, edited in
the `$grid` DenseArray attr) was REGENERATED by the compiler into a DISTINCT kernel symbol
(`…_PERT_…`). On the board, with inputs that drive grid index 5: the REAL-attr kernel =
87244.55 (= real-table ref) and the PERTURBED-attr kernel = 101755.50 (= perturbed-table
ref) — each matches the reference built with ITS OWN table, and the two HW results DIVERGE.
A hardcoded-table emitter (where the attr is dead) could not diverge. So the 512-entry grid
DenseArray attr genuinely flows attr → emitted const → hardware result.

**Reference faithfulness:** the board's real `ggml_vec_dot_iq3_s_q8_K` has hand-written RVV
helpers (vl128/vl256/vl512, runtime-VLEN-dispatched) but they are numerically equivalent to
`_generic` (the externally-defined numerically-exact contract); validating against the
transcribed `_generic` — itself proven byte-identical to ggml's source `_generic` on the
host — validates exactly the contract the board's kernel must satisfy.

## (3) raw()=0 + STRUCTURED

`raw(` actual-call count in the compiler-emitted kernel: **0**. Every executable value is
an emitc node: 13 distinct `__riscv_*` intrinsics (all structured `CallOpaque`), the grid
+ kmask as `static const` decls (NO ksigns table), the 512-grid byte-view + indexed lookup
as structured pointer arith, the qh 9th-bit injection `(qhb << {8/7-2l}) & 256 | idx` as
structured int ops (64 `& 256` occurrences = 8 sub-blocks × 4 groups × 2 passes), the
explicit sign region read at `xb + 74` (NO ksigns lookup), the explicit scales at `xb +
106` with `& 15` / `>> 4` two-nibble split, `*s = sumf` (NO 0.25f/0.125f). See
`structured_proof.txt`. `nm` on the board object shows ONE exported symbol
(`tcrv_emitc_ggml_vec_dot_iq3_s_q8_K_kernel_…`) and ONE external ref (`__extendhfsf2`, the
compiler-rt fp16→fp32 builtin) — no kernel fallback, the RVV intrinsics inline. The kernel
is emitted ONLY by the compiler (`tcrv-opt --tcrv-rvv-lower-to-emitc | mlir-translate-20
--mlir-to-cpp`), byte-identical to a fresh regen.

## (4) lit + reds

- 2 NEW lit tests PASS (conversion FileCheck + dialect dataflow with 7 fail-closed
  negatives: unknown kind, wrong scale_model, wrong stride, wrong qh offset, wrong signs
  offset, wrong scales offset, wrong grid size).
- `ninja check-tianchenrv`: **664 discovered, 661 pass, exactly the 3 documented
  pre-existing reds** (`Scripts/rvv-generated-bundle-abi-e2e-*` — explicit/pre-realized
  computed-masked-strided dot-reduce dry-runs ×2 + the abi-e2e-self-test, all unrelated to
  the dialect/conversion). +2 over inc36's 662 = my two new tests.
- Sibling byte-identity (regenerated through `tcrv-opt | mlir-translate-20` and diffed
  byte-for-byte): iq3_xxs (inc36) AND iq2_s (inc35) emitted bodies **BYTE-IDENTICAL** →
  the change is purely additive.
- Full CLEAN rebuild green (`ninja clean` → all 240 targets link).

## (5) The re-composition (which pieces from which kernel)

| | iq3_xxs (inc36) | iq2_s (inc35) | **iq3_s (inc37)** |
|---|---|---|---|
| grid table | 256 × uint32 (4 int8) | 1024 × uint64 (8 int8) | **512 × uint32 (4 int8)** |
| grid lookup | grid-of-4, two 4-lane passes | grid-of-8, one 8-lane pass | **grid-of-4, two 4-lane passes** ← iq3_xxs |
| index source | q3 byte (8-bit) | qs byte + qh 2-bit plane (10-bit) | **qs byte + qh 1-bit plane (9-bit)** ← iq2_s |
| qh mask / shifts | none | `& 0x300`, shifts 8/6/4/2 | **`& 256`, grid1 8-2l / grid2 7-2l** |
| signs | `ksigns_iq2xs[(aux>>7l)&127]` | EXPLICIT `signs[l]` from qs+32 | **EXPLICIT `signs[l]` from xb+74** ← iq2_s |
| ksigns plane | present | NONE | **NONE** ← iq2_s |
| scale | packed-aux `2*(aux>>28)+1` | explicit `scales[8]` two-nibble | **explicit `scales[4]` two-nibble** ← iq2_s |
| weight stride | 98 | 82 | **110** |
| trailing factor | 0.25f | 0.125f | **none (`*s = sumf`)** |

So: the **grid-of-4 lane mechanism comes from iq3_xxs**, the **qh 9th-bit plane + the
explicit-sign-from-memory + the explicit two-nibble scales come from iq2_s**. iq3_s is the
re-composition the inc35/inc36 RESULTS predicted — "iq2_s's qh injection + explicit-sign
decode transfer almost directly; the new fact is the grid-of-4 lane width" — confirmed: the
emitter is iq3_xxs's grid-of-4 body with iq2_s's index/sign/scale bookkeeping, no new
mechanism. The TWO iq3_s-specific facts (the single-bit 256 mask vs iq2_s's 0x300, and the
per-pass shift difference 8-2l vs 7-2l) are byte-format deltas, not new mechanisms.

## (6) Remaining IQ tail (the last)

iq3_s is the 5th GRID-codebook kernel (iq2_xxs / iq2_xs / iq2_s / iq3_xxs / iq3_s). The IQ
tail now has only the **ternary** class left:
- **iq1_s** / **iq1_m** — a 2048-entry ternary `iq1_grid` (each entry packs 1.5-bit ternary
  {−1,0,1}-ish values, NOT signed int8), a per-block `delta` (`qh & 0x8000 ? −1 : 1`
  scaled by `IQ1S_DELTA`, folded via the q8 `bsums`), a per-sub-block scale from
  `(qh>>12)&7`, and the 8-bit index assembled from `qs[l] | (((qh>>3l)&7)<<8)` (a 3-bit qh
  field, NOT a sign plane). iq1_m additionally carries per-sub-block scale nibbles. The
  ternary grid + the 1.5-bit packing + the bsums-based delta is the genuinely-new mechanism
  — the structurally furthest, and the last common ggml dot class.

The 18 covered kernels now span every COMMON ggml dot class (legacy block-quant q4_0..q5_1,
K-quant super-block q2_K..q6_K, the 16-entry codebook iq4_nl/iq4_xs, fp4 mxfp4) plus the
full iq2 GRID-codebook trio + the iq3 grid-of-4 pair (iq3_xxs/iq3_s) — only the iq1 ternary
class remains.

## Reproduce
```
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-iq3-s-q8-k-block-dot.mlir \
  --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp > tcrv_emitted_kernel.cpp
scp tcrv_emitted_kernel.cpp inc37_validate.cpp rvv:~/inc37_iq3s/
ssh rvv 'cd ~/inc37_iq3s && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off  -c tcrv_emitted_kernel.cpp -o k_off.o  && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off  inc37_validate.cpp k_off.o  -o val_off  && ./val_off && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=fast -c tcrv_emitted_kernel.cpp -o k_fast.o && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=fast inc37_validate.cpp k_fast.o -o val_fast && ./val_fast'
```

## Files in this artifact dir
- `iq3_s_input.mlir` — the typed-body input (the conversion test module).
- `iq3_s_emitted.cpp` / `tcrv_emitted_kernel.cpp` — the compiler-emitted kernel C.
- `inc37_validate.cpp` — the byte-exact HW harness (verbatim `_generic` reference +
  4 negative controls: wrong grid + wrong signs + wrong scale + wrong qh).
- `iq3_s_input_perturbed.mlir` — the PERTURBED `$grid` typed-body (grid[5] low byte +1), a
  distinct kernel symbol.
- `iq3_s_perturbed_kernel.cpp` — the compiler-regenerated perturbed kernel C.
- `inc37_knob.cpp` + `inc37_grids.h` — the COMPILER-KNOB control (real-attr vs perturbed-
  attr kernel diverge on HW, each matching its own table → the attr is the live attr →
  const → HW knob).
- `ssh_rvv_byte_exact_stdout.txt` — the raw board stdout (off + fast + nm + knob).
- `structured_proof.txt` — the raw()=0 / structured-emission proof.
