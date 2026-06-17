# INC-38 — iq1_s × q8_K byte-exact; the TERNARY class (19th dot kernel)

19th ggml dot kernel: `ggml_vec_dot_iq1_s_q8_K` — the **TERNARY class**, the last common
ggml dot kernel (only `iq1_m`, its sibling, remains). It REUSES the grid-codebook
indexed-lookup integer core from the iq2/iq3 siblings but adds a **genuinely new
mechanism — the per-block DELTA term**. New op `tcrv_rvv.iq1_s_q8_k_block_dot`, fully
STRUCTURED (raw()=0; I5).

Byte-exact `*s` vs ggml's OWN `_generic` (`quants.c:1099-1140`) on real `ssh rvv`
(`-march=rv64gcv`, Zvl128b ⇒ VLEN≥128): **2464/2464 cases, 0 failures** at
`-ffp-contract=off` AND `fast`; the 11-bit grid index marches to **max idx = 2047 (hit
YES)**. THREE negative controls (wrong grid + wrong scale + **drop-the-DELTA**) + a
compiler-knob control all diverge.

## The new mechanism (the ternary grid + the DELTA term)

### (a) the 2048-entry TERNARY grid — NO sign plane
`iq1s_grid[2048]` is a uint64 table where each entry packs **8 ternary values in
{0x00, 0x01, 0xff} = {0, +1, −1}**. Byte-viewed as a `const int8_t *`, the `0xff` bytes
read as **−1** — the grid **IS** the signed value. So unlike every iq2/iq3 sibling there
is **NO sign-bit plane, NO `kmask`, NO `vmsne`/`vand`/`vneg`/`vmerge` sign apply**: the
inner loop is just `idx → vle8 grid → vle8 q8 → vwmul_i16m2 → vwredsum`. This is the only
grid family whose stored bytes are themselves negative. The 11-bit index is
`idx = qs[ib*4 + l] | (((qh[ib] >> 3*l) & 7) << 8)` (range to 2047): a single qs index
byte plus a 3-bit field of the per-sub-block **uint16** `qh[ib]`, a different field per
group l=0..3 (shifts 0,3,6,9).

### (b) the qh-encoded scale
`ls = 2*((qh[ib] >> 12) & 7) + 1` (qh bits 12..14) — ONE scale per sub-block applied to
all 4 groups (no explicit two-nibble `scales[]` split; iq1_s carries no scales[] array).

### (c) the per-block DELTA term — the NEW fold piece
`delta = (qh[ib] & 0x8000) ? -1 : 1` (qh bit 15), and the `IQ1S_DELTA = 0.125` constant
biases each value by `+delta·IQ1S_DELTA`. ggml folds the delta contribution through the
**q8 sub-block sums (`bsums`)**: per sub-block `sumi1 += ls·delta·(bsums[2ib]+bsums[2ib+1])`.
This is the **first** time the activation `int16 bsums[16]` region (offset 260) is read in
the entire IQ tail. The two integer accumulators are kept **SEPARATE** across all 8
sub-blocks; `sumi` (the ternary dot) and `sumi1` (the delta-bsum term). The 0.125 is
applied **EXACTLY ONCE** per super-block, inside the fold, to `(float)sumi1` only:
`sumf += d · ((float)sumi + 0.125f · (float)sumi1)`, and **`*s = sumf`** (NO global
trailing factor — the iq1_s store is the bare accumulator, unlike iq2_s's `0.125f*sumf`).

## (1) The op + lowering + files

### New op `tcrv_rvv.iq1_s_q8_k_block_dot`
- ODS: `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (`GgmlBlockDotIQ1SQ8KOp`). Four
  runtime ABI operands (vx/vy/s/n) + the vl token; the super-block-format structural attrs
  (`weight_block_stride`=50, `weight_qs_byte_offset`=2, `weight_qh_byte_offset`=34 — the
  uint16 qh array, `activation_bsums_byte_offset`=260 — the NEW bsums region) PLUS the
  `grid` (2048 int64) table as a typed array attr (I4 mirror). **NO `weight_signs_byte_offset`,
  NO `weight_scales_byte_offset`** (no sign plane, no scales[] array — the scale lives in qh).
- Verifier: `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (`GgmlBlockDotIQ1SQ8KOp::verify`).
  Fail-closed (I7): pins `kind` (`ggml_iq1_s_q8_k_block_dot`), `scale_model`
  (`per-sub-block-qh-scale-ternary-grid-codebook-qh-plane-delta-bsum-int-domain`), qk=256,
  sub_block=32, **stride=50** (sizeof block_iq1_s = 2+32+16), qs@2, **qh@34**, act
  stride=292, act qs@4, **bsums@260**, **grid size==2048**, the operand C types, the i32m1
  result, the policy on the enclosing with_vl. Rejects any forbidden/unknown attr.
- Lowering: `lib/Conversion/RVV/RVVToEmitC.cpp` — `isIQ1SQ8KBlockDotBody` recognizer, a
  dispatch branch (inserted right after the iq3_s branch, before mxfp4), and the
  `emitIQ1SQ8KBlockDot` emitter (a self-contained parallel of `emitIQ2SQ8KBlockDot`, NOT a
  shared-helper refactor — preserves "siblings byte-identical / additive").

### Reuse (the task mandate "REUSE the grid lookup mechanism")
- **From the iq2/iq3 grid siblings**: the indexed packed-uint64 GRID lookup
  (`grid_i8 + idx*8` then `vle8(8)`), the signed `vwmul_i16m2` widening product, the chained
  `vwredsum` i32 reduce, the scalar `vmv_x_s` extract, the structured `static const`
  grid-table decl read via a `(const int8_t *)` byte cast, the q6_K-style super-block
  structure (q8_K activation, integer accumulation, the per-super-block fp32 fold as ONE
  `emitc.expression`).
- The iq1_s-specific deltas: the ternary grid carries signed values so the **entire
  sign-apply block is deleted** (no kmask/vmsne/vand/vneg/vmerge); the scale is read from the
  **uint16 qh word**; the **DELTA term** reads the int16 bsums and folds
  `ls·delta·(bsums[2ib]+bsums[2ib+1])` into a second integer accumulator; the fold is
  `d·((float)sumi + 0.125f·(float)sumi1)` (the inner add is genuinely new vs iq2_s) and
  `*s = sumf` (no trailing factor).
- The grid is emitted as `static const uint64_t` (not int64_t): the ternary entries have the
  high bit set (e.g. `0xffffffffffffffff` = all −1), which would not narrow into a signed
  int64_t under C++11; the `(const int8_t *)` byte cast still yields the signed ternary value.

### Files (additive)
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (+ `GgmlBlockDotIQ1SQ8KOp`)
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (+ `GgmlBlockDotIQ1SQ8KOp::verify`)
- `lib/Conversion/RVV/RVVToEmitC.cpp` (+ recognizer/branch/`emitIQ1SQ8KBlockDot`)
- `test/Conversion/RVV/rvv-to-emitc-iq1-s-q8-k-block-dot.mlir` (NEW lit, full grid + CHECK)
- `test/Dialect/RVV/iq1-s-q8-k-block-dot-dataflow.mlir` (NEW lit, positive + 6 fail-closed
  rejections incl. the bsums-offset and grid-size pins)

## (2) Byte-exact vs ggml iq1_s `_generic` (cases, incl. the DELTA control, 0 failures)

`ssh rvv`, `-march=rv64gcv -mabi=lp64d`, both `-ffp-contract=off` AND `fast`:
```
iq1_s byte-exact: 2464/2464 passed, 0 failed (generic delta 0/2464)
grid-index coverage (edge): max idx = 2047, hit 2047 = YES
NEG-CTRL wrong-grid  (every grid byte +1)         -> DIVERGES
NEG-CTRL wrong-scale (drop the +1)                -> DIVERGES
NEG-CTRL drop-DELTA  (omit ls*delta*bsum term)    -> DIVERGES   <- the new mechanism is load-bearing
RESULT: PASS (byte-exact vs ggml iq1_s _generic + 3 negative controls live)
```
- **2464 cases** = 300 reps × 8 sizes {256,512,768,1024,2048,4096,8192,16384} + 64 edge
  super-blocks; idxModes exercise q8=+127 and q8=±127. Edge blocks drive qs=0xff + qh group
  fields=7 → index 2047, scale field 0 (ls=1) and 7 (ls=15), the delta sign toggled.
- The **drop-DELTA** control proves the new piece is live: it omits the
  `ls·delta·(bsums[2ib]+bsums[2ib+1])` contribution → diverges for generic data (sumi1≠0
  regardless of the sign bit). 0 failures = our kernel keeps it.
- **NOT a tautology**: the harness reference is a *differential* re-implementation, not the
  same code path. The reference computes `lsum` via a scalar nested `l=0..3 / j=0..7` loop;
  our emitter chains `vwredsum` across the 4 groups (a structurally different reduction) in a
  flat 8-sub-block structure. The 2048-entry grid is extracted **programmatically** from
  `ggml-common.h`'s `GGML_TABLE_BEGIN(uint64_t, iq1s_grid, 2048)` block (not hand-retyped),
  so the harness grid and the compiler-attr grid share a single source of truth.

### Compiler-knob control (attr → const → HW flow is LIVE)
`ssh_rvv_knob_stdout.txt`: the REAL-grid kernel and a PERTURBED-grid kernel (grid entry 5
low byte +1, emitted by the SAME compiler from a perturbed-attr input — byte-identical to
the real kernel except the grid const) produce DIFFERENT `*s` on identical data:
```
KNOB real-grid=104.623032 perturbed-grid=107.377823 -> DIVERGES (attr->const->HW flow live)
```
This rules out a hardcoded table — the `$grid` typed attr is the live source of the emitted
const and the hardware result.

## (3) raw()=0 + structured (`structured_proof.txt`)
- **raw() actual-call count: 0**; the only non-comment verbatim op in the lowered IR is the
  structured `static const uint64_t tcrv_iq1s_grid[2048]` table decl. No raw escapes, no asm.
- The distinct `__riscv_*` intrinsics are exactly the ternary integer core: `vsetvl_e8m1`,
  `vle8_v_i8m1`, `vmv_v_x_i32m1` (seed), `vwmul_vv_i16m2`, `vwredsum_vs_i16m2_i32m1`,
  `vmv_x_s_i32m1_i32` (extract) — and **NO sign-apply intrinsics** (`vand`/`vmsne`/`vneg`/
  `vmerge` all 0, `kmask`/`ksigns` 0), because the ternary grid is itself signed.
- The uint16 qh read drives the scale (`>>12`, 8×) and the delta sign (`>>15`, 8×); the int16
  bsums read drives the delta term; the 0.125f appears exactly once, inside the fold; `*s = sumf`.

## (4) lit + reds
- Two new lit tests PASS (Conversion FileCheck + Dialect `--verify-diagnostics`).
- **Full clean rebuild** (`ninja -t clean && ninja check-tianchenrv`, tablegen `.inc`
  regenerated from scratch): **666 tests, 663 pass, 3 fail** — the 3 are the SAME pre-existing
  DOCUMENTED reds, unrelated to iq1_s:
  `rvv-generated-bundle-abi-e2e-{explicit,pre-realized}-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test`
  and `rvv-generated-bundle-abi-e2e-self-test.test`. The count went 664→666 (the 2 new tests
  both pass; 661→663 passing).
- **Siblings byte-identical (additive)**: iq3_s / iq2_s / iq3_xxs (and all other families)
  still lower byte-identically — their FileCheck conversion tests pass unchanged; the new code
  is a self-contained branch that touches no shared codepath.

## (5) the ternary grid + the delta term (the new mechanism) + how byte-exact
The ternary grid is the genuinely-new codebook (signed-in-place {−1,0,+1}, no sign plane);
the DELTA term is the genuinely-new fold (a per-sub-block ±ls constant × the q8 16-element
sums, via the bsums, with the 0.125 IQ1S_DELTA applied once). Byte-exactness holds because
(i) the grid is read identically to ggml (`(const int8_t *)(iq1s_grid + idx)`), (ii) the
integer dot/accumulation is order-free (so chaining the vwredsum across groups is exact),
(iii) the per-super-block fp32 fold is invoked in STRICT ascending order, and (iv) the fold
`d·((float)sumi + 0.125f·(float)sumi1)` is a single `emitc.expression` whose left-assoc
grouping and FMA fusion match ggml's `d·(sumi + IQ1S_DELTA·sumi1)` exactly — confirmed by
2464/2464 under `-ffp-contract=fast` (the only new contraction opportunity, the inner add,
is inside the expression tree).

## (6) iq1_m is the LAST — its sibling
`iq1_m` is the only remaining common ggml dot kernel (the iq1_s sibling). Block format
`block_iq1_m = { uint8_t qs[QK_K/8]; uint8_t qh[QK_K/16]; uint8_t scales[QK_K/32] }` (NO
fp16 d field — the scale is reconstructed from a `iq1m_scale_t` union packed across the
`scales[]` bytes; per-group 3-bit grid-high + a grid-shift bit; the IQ1M_DELTA = 0.125
delta is per **half-sub-block** of 16 not per-32). It reuses this same 2048 ternary grid +
the delta mechanism, with a different qh layout (uint8 qh[16], 3 high bits + a grid-shift
bit per group of 8) and the packed-scale reconstruction — the natural next (final) rung.

## Reproduce
```
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-iq1-s-q8-k-block-dot.mlir \
  --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp > tcrv_emitted_kernel.cpp
scp tcrv_emitted_kernel.cpp inc38_validate.cpp rvv:~/inc38_iq1s/
ssh rvv 'cd ~/inc38_iq1s && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off  -c tcrv_emitted_kernel.cpp -o k_off.o  && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off  inc38_validate.cpp k_off.o  -o val_off  && ./val_off && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=fast -c tcrv_emitted_kernel.cpp -o k_fast.o && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=fast inc38_validate.cpp k_fast.o -o val_fast && ./val_fast'
```

## Files in this artifact dir
- `iq1_s_input.mlir` — the typed-body input (= the conversion lit test, full grid + CHECK).
- `tcrv_emitted_kernel.cpp` / `iq1_s_emitted.cpp` — the compiler-emitted kernel C.
- `inc38_validate.cpp` — the byte-exact HW harness (differential `_generic` reference +
  3 negative controls: wrong grid + wrong scale + drop-the-DELTA).
- `iq1_s_input_perturbed.mlir` — the PERTURBED `$grid` typed-body (grid[5] low byte +1).
- `iq1_s_perturbed_kernel.cpp` / `iq1_s_perturbed_kernel_PERT.cpp` — the perturbed-grid
  kernel (the `_PERT` copy renames the symbol so both link into the knob harness).
- `inc38_knob.cpp` — the compiler-knob control harness.
- `ssh_rvv_byte_exact_stdout.txt` — the board output (off + fast, 2464/2464, controls).
- `ssh_rvv_knob_stdout.txt` — the board knob output (real vs perturbed grid divergence).
- `structured_proof.txt` — the raw()=0 + intrinsic/mechanism census.
