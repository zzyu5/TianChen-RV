# INC-9 — q4_1 x q8_1 breadth: a structurally NEW (scale+MIN, Family-B) ggml kernel

The template + autotuner generalize to `ggml_vec_dot_q4_1_q8_1` — a DIFFERENT decode
AND dequant from the q4_0/q8_0 kernels already done. New op
`tcrv_rvv.q4_1_q8_1_block_dot`, structurally lowered (raw()=0), the SAME autotuner
selecting its shape from a DERIVED latency depth, byte-exact on `ssh rvv`.

## What's structurally new (vs q4_0/q8_0) — verified against quants.c + ggml-common.h
- `block_q4_1 = {fp16 d; fp16 m; uint8 qs[16]}` stride **20**; `block_q8_1 = {fp16 d;
  fp16 s; int8 qs[32]}` stride **36**. Quants at **+4** (after the TWO inline fp16
  scales); q8 high half at +16; the 2nd scales (m / s) at +2.
- **UNSIGNED nibbles [0,15]** — NO offset-binary `-8`, NO XOR-0x88. Decode = `vand
  0x0F` / `vsrl 0x04` (LOGICAL) on the u8 weight lane, value-identity reinterpret to
  i8, then the SAME signed vwmul/vwmacc against the plain q8 halves.
- The fold carries a **SECOND scale on each operand**: ggml's exact statement
  `sumf += (d_x*d_y)*sumi + m_x*s_y` (quants.c:319) — the two products are SUMMED
  FIRST (the `+` binds before `+=`), then added to sumf: `sumf + (A + B)`. This fp
  grouping is significant and was a byte-exactness trap (see (2) below).

## (1) The op + lowering + autotuner inheritance
- Op `tcrv_rvv.q4_1_q8_1_block_dot` mirrors the block-dot ops: same 5 ABI operands +
  block-format mirror attrs (incl. the new `weight_min_byte_offset`/
  `activation_sum_byte_offset` = 2) + the SAME shape knobs. Fail-closed verifier (I7)
  pins every q4_1 fact (stride 20/36, quant +4, min/sum +2, anchors mf4/m1, elided⇒m1).
- Lowering `emitQ4_1Q8_1BlockDot` REUSES the q4_0 block-loop / unroll / tail /
  strip-elision scaffolding; the kernel-specific parts are a NEW
  `emitUnsignedNibbleDecodeProductValue` (added cleanly alongside the offset-binary
  one, which is byte-untouched) and the 4-scale fold. raw()=0 — every value an emitc
  node (verified: zero `raw(`, zero non-comment `emitc.verbatim`).
- **Autotuner inherited, depth DERIVED**: a q4_1 materialize pass reuses the SAME
  enumerate/prune/select + `deriveHasZvl128b` + cost model. q4_1's `coreLatencyDepth`
  is **derived = 4** (base product→reduce 2 + a NEW `"nibble-unsigned"` decode prefix
  **2** = vand+vsrl, the reinterprets free) — SHORTER than q4_0's 7 (offset-binary
  prefix 5). Both still exceed the unroll range {1,2,4}, so the picks match q4_0's,
  but the depth is a computed structural fact off the format, not a per-kernel
  constant (`getRVVBlockDotDecodePrefixLength`).

## (2) BYTE-EXACT vs ggml's REAL q4_1 kernel + _generic (ssh rvv) — ssh_rvv_byte_exact_stdout.txt
Pipeline: `tcrv-opt … --tcrv-rvv-materialize-q4-1-schedule=march=… --tcrv-rvv-lower-to-emitc
| mlir-translate-20 --mlir-to-cpp`. 3300 random cases/config (n∈{32..8192}) + a
negative control (perturbed MIN scale must diverge — proves the m_x*s_y term is consumed).

| autotuner pick           | -ffp-contract=off (vs ggml-rvv AND _generic) | -ffp-contract=fast (vs ggml-rvv) |
|--------------------------|----------------------------------------------|----------------------------------|
| full-V (m1,mb4,elided)   | 3300/3300 PASS, _generic delta 0             | 3300/3300 PASS, _generic delta 0 |
| zve32x (m1,mb2,robust)   | 3300/3300 PASS, _generic delta 0             | 3300/3300 PASS, _generic delta 0 |

The _generic cross-check delta is **0 even at =fast** — the emitted fold tree mirrors
ggml's statement so faithfully that the compiler forms the SAME FMA (no INC-7-style
reference-vs-reference =fast artifact on this kernel; honestly notable). Negative
control passed every config.

THE TRAP FIXED: the first board run failed ~1865/3300 at large n (tiny ~1-ULP deltas vs
BOTH refs, only n≥64). Diagnosis (advisor-confirmed): the fold add-grouping. ggml's `+`
binds before `+=` → `sumf + (A + B)`; the first emission did `(sumf + A) + B`. Both are
the same at n=32 (sumf=0) but diverge once sumf is nonzero. Fixed the emitFold tree to
`sumf + (A + B)` → 0 failures.

## (3) DIVERGENCE — autotuner-divergence test + q4_1_{fullv,zve32x}_autotuned.cpp
The SAME attr-less q4_1 op, stamped by the autotuner purely by --march:

| --march          | derived Zvl128b | SELECTED shape          | emitted C |
|------------------|-----------------|-------------------------|-----------|
| rv64gcv          | true            | (m1, factor=4, elided)  | 343 lines (4 elided cores, nb%4 robust tail) |
| rv64gc_zve32x    | false           | (m1, factor=2, robust)  | 229 lines (2 robust cores) |

One capability fact (Zvl128b) → N1 legality divergence (elided pruned without it), on a
structurally NEW kernel. Lit-checked (`rvv-q4-1-q8-1-block-dot-autotuner-divergence.mlir`)
at both the stamp boundary AND the emission.

## (4) PERF vs ggml (HONEST: parity at best, NO win) — board_ladder_stdout.txt
ssh rvv, rv64gcv, taskset -c 3, -O3 -ffp-contract=fast, n=4096, best-of-N=200 interleaved,
correctness-gated. Full m1 shape ladder (the discriminating measurement):

| shape                    | best-ns | ggml/best (>1 ⇒ faster) |
|--------------------------|---------|-------------------------|
| ggml(real)               | 9909.8  | 1.000x                  |
| mb1_robust               | 9989.5  | **0.992x** (fastest of ours) |
| mb2_robust               | 10090.2 | 0.982x                  |
| mb4_robust               | 10287.2 | 0.963x                  |
| mb1_elided               | 10279.7 | 0.964x                  |
| mb2_elided               | 15458.3 | 0.641x                  |
| mb4_elided (FULL-V PICK) | 15642.0 | 0.634x                  |

**HONEST VERDICT: the q4_0 win does NOT generalize to the scale+MIN family.** No q4_1
shape beats ggml; the best (mb1_robust) is parity (0.992x). The autotuner's full-V pick
(mb4_elided) is the SLOWEST (0.634x) — the INVERSE of q4_0, where mb4_elided won.

WHY (mechanism, confirmed by `-S`): q4_1's fold reads FOUR fp16 scales/block
(d_x,m_x,d_y,s_y) vs q4_0's two. Under the 4-block unroll the emitter holds all four
blocks' cores + 16 fp32 scalars live before the folds, crossing the scalar-spill
threshold; the elided form (one wide vwredsum/half-block, more vector state live)
tips it over. Spill stores in the hot loop: **mb1_robust = 19, mb4_elided = 49**
(~2.6×). The shared cost model (`computeBlockDotShapeCostCore`) models reductions /
strip penalty / unroll-overlap but NOT the 4-scale fold's scalar-register pressure —
precisely the term that differs between q4_0 and q4_1.

This is a cost-model BLIND SPOT, not a correctness or template defect. Extending the
cost model with a fold-operand-pressure term (so the autotuner picks mb1_robust for
q4_1) is flagged as FUTURE WORK — it is out of scope here because recalibrating the
shared constants would force re-validating the committed q4_0/q8_0 picks.

**DEPLOYMENT GUIDANCE (the honest corollary):** for q4_1, the shape to SHIP is
**mb1_robust** (0.992x, parity with ggml), NOT the autotuner's full-V pick
(mb4_elided, 0.634x). A user who runs `--tcrv-rvv-materialize-q4-1-schedule`
expecting the q4_0-style win gets a ~1.6x SLOWDOWN. Until the cost model gains a
fold-operand-pressure term, prefer hand-pinning `integer_core_lmul="m1",
multi_block_factor=1, strip_elision="robust"` for q4_1 (or just keep ggml's kernel,
which mb1_robust merely matches).

## (5) raw()=0 + structured proof
`tcrv-opt … --tcrv-rvv-lower-to-emitc` on the q4_1 op emits ZERO `raw(` and ZERO
non-comment `emitc.verbatim` — every value is an emitc node (load/mul/add/cast/for/
call_opaque). The only opaque pieces are the sanctioned fp16→fp32 reads + the intrinsic
call_opaques, exactly as q4_0/q8_0.

## (6) lit + the 3 documented reds
`check-tianchenrv`: 613/616. The 3 failures are the pre-existing documented Scripts
e2e dry-run/self-test reds (computed-masked-strided-input), untouched by this work. The
5 NEW q4_1 tests all PASS:
- `Dialect/RVV/q4-1-q8-1-block-dot-dataflow.mlir` (verifier, fail-closed I7)
- `Conversion/RVV/rvv-to-emitc-q4-1-q8-1-block-dot.mlir` (structured lowering)
- `Conversion/RVV/rvv-q4-1-q8-1-block-dot-autotuner-divergence.mlir` (selection+emission divergence)
- `Plugin/rvv-q4-0-q8-0-shape-selection.test` (extended: q4_1 depth=4, picks, derived-sum)
q4_0/q8_0/deferred-wide lit unchanged (no regression). Clean full rebuild green.

## (7) deviations + dishonesty risk
- DEVIATION: perf is parity/loss, not a win. The task asked to "report match/beat/lose
  TRUTHFULLY" — this is the truthful loss. Breadth (the goal) is met: byte-exact on a
  structurally new family, divergence live, autotuner inherited with a genuinely derived
  depth. The committed op/pass comments claim *selection*, never *win* — kept honest.
- DISHONESTY RISK AVOIDED: NOT writing "the autotuner selects the winning shape" for
  q4_1 (the selected mb4_elided is the slowest). The autotuner SELECTS a byte-exact,
  capability-correct shape; on THIS kernel that shape does not win — stated plainly.
- The materialize pass is opt-in (`--tcrv-rvv-materialize-q4-1-schedule`),
  march-selection-live (NOT ssh-rvv-probe-live, I6) — same posture as the q4_0/q8_0
  siblings.

## Reproduce
```
# byte-exact (both profiles × {off,fast}):
tcrv-opt q4_1_input.mlir --tcrv-rvv-materialize-q4-1-schedule=march=rv64gcv \
  --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp > q4_1_fullv_autotuned.cpp
ssh rvv: clang++-18 -O2 -march=rv64gcv -ffp-contract=off -DCONTRACT_REFS=1 \
  inc9_validate.cpp q4_1_fullv_autotuned.cpp -o v && taskset -c 3 ./v
# ladder:
ssh rvv: clang++-18 -O3 -march=rv64gcv -ffp-contract=fast ladder.cpp kern_ggml_q4_1.cpp \
  lad_mb{1,2,4}{r,e}.cpp -o ladder && taskset -c 3 ./ladder
```
