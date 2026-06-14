# Stage 3 换心 — retire the contraction owner's dequant body path by making the Gearbox dequant convertible

Author: Fable (senior MLIR architect). Date: 2026-06-13.
Task: `.trellis/tasks/06-12-stage3-replace-string-machine/`. Parent ADR: 06-12-mlir-audit-refactor.

This plan decides HOW to move the last two Gearbox dequant families off the shared legacy
string-plan body emission so the real `RVV->emitc` DialectConversion reproduces their C without
reading any mirror metadata (no I4/I5 violation). It is the deepest, last rung of the Section-4
strangler-fig recipe in `heart-replacement-plan.md`.

---

## Verdict: **realization-rewrite**

Make the Gearbox selected-body realization emit the dequant compute STRUCTURE as typed ops, then
convert by walking that structure. NOT `direct-emitc-semantic`, NOT `blocked-needs-bigger`.

### Why not `direct-emitc-semantic`
The two surviving families (`WideningProductReduceDequantizeF32`,
`WideningProductReduceDequantClampF32`) are each gated by a packed-i4 candidate, and the contraction
owner is only retired once ALL their candidates are off the legacy path. For the packed-i4 candidate
the realized typed body is `load i8mf4, load i8mf4, widening_product {kind=signed_widening_product,
product_relation="signed-i8mf4xi8mf4-to-i16mf2"}, standalone_reduce, gearbox_cross_region_handoff,
with_vl{dequantize, store}` — the TYPES say "multiply two i8 bytes." The fact that each i8 packs two
i4 nibbles that must be sign-extended/unpacked before the product lives ONLY in the
`operand_form="packed-i4-nibbles"` / `unpack_intent="sign-extend-i4-nibbles-before-widening-product"`
strings on the handoff op (verified `RVVContractionSelectedBodyRealizationOwner.cpp:1996-2099`,
candidate-ID branch at `RVVEmitCStatementPlanOwners.cpp:997-1004`). A literal conversion walk of
today's IR therefore emits a single plain i8 `vwmul` = numerically WRONG; the only way to emit the
correct 7-op nibble unpack is to read those candidate-mirror strings inside the conversion = exactly
the I5 violation Stage 3 exists to remove. So a direct-emitc walk cannot retire the owner.
(`direct-emitc-semantic` IS the right technique for the grouped sub-structure once it is typed — see
conversion_change — but it is not the verdict, because packed-i4 forces realization to emit new ops.)

### Why not `blocked-needs-bigger`
The reduction-owner retirement (`reduction-owner-conversion-plan.md`, 6th owner, 1950-line owner
deleted) is a comparable-sized, already-shipped precedent: detect typed body structure, emit emitc
directly, byte-identical/HW-validated, delete the statement-plan owner. This is the same shape plus
two new ODS ops and a register-accumulator conversion. It is large but bounded and precedented, not
blocked.

---

## Scope decision (the thesis call the analysis under-stated): structuralize the grouped unroll for EMISSION PARITY — do NOT decline it

g3's "lowest-cost" path proposed emitting a PLAIN single loop for the grouped candidate ("no body-C
lit gate, breaks nothing"). I verified this is NOT thesis-safe and reject that part of g3. The
PRIMARY justification is emission parity (not "it's a measured win" — see below):

- **PARITY (primary).** A behavior-preserving retirement must reach emission parity with the legacy
  path BEFORE deleting it. Legacy grouped emission produces an unroll-by-2 main loop (step=2·vlmax)
  + a scalar tail loop (`RVVEmitCStatementPlanOwners.cpp:1638-1686` second slice; 1688-1743 tail).
  Emitting an unroll=1 single loop where legacy emits unroll-2 is an UN-VALIDATED structural change
  to a shipped artifact — regardless of whether unroll-2 is faster. The retirement's contract
  (Section 4 of `heart-replacement-plan.md`) is equivalence to the legacy path, then deletion; a
  silently different loop shape breaks that contract. So parity alone forces structuralizing the
  grouped main+tail loop.
- **TUNE OWNERSHIP (secondary, do NOT overclaim).** Grouped (unroll=2) IS the SELECTED candidate:
  selection picks the highest-unroll LEGAL candidate
  (`selectRVVLowPrecisionProductReductionResourceCandidate`, `RVVGearboxSchedule.h:1540-1552`:
  `candidate.unrollFactor > best->unrollFactor`; constants 733,737,741: static=1, grouped=2,
  packed-i4=1). Project memory is explicit: `tune 是用户拥有的 novelty（别再提砍）`, so silently
  collapsing a selected candidate's shape is the kind of tune erosion the user has pushed back on.
  BUT honesty caveat: grouped's unroll-2 is SELECTED, not MEASURED — the `measured-win` admission
  (`kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionDecision`, `RVVGearboxSchedule.h
  :703`) is packed-i4-only, and the HW lamp is correctness-only (`tolerance=1e-05`, no cycles). So
  do NOT claim "unroll-2 is a measured win"; the load-bearing argument is parity, with tune-ownership
  as the reason a parity gap here is not a harmless simplification.

Consequence: the realization rewrite must structuralize BOTH (a) the packed-i4 nibble unpack and
(b) the grouped main(unroll-by-2)+scalar-tail multi-region loop carrying the mutable accumulator.
This is the g1/g2 "model the grouped/tail loop as typed multi-region control flow" scope, NOT the
g3-minimal "decline the unroll" scope. It is larger, but it is what keeps the换心 thesis-honest.

**Scope GATE (do this first, cheaply):** the broad scope rests on legacy grouped emission actually
producing the two-loop unroll-2+tail shape. This is cited from source lines, not yet confirmed from
generated output (the grouped fixtures have ZERO body-C gates, so no oracle exists). Step 1 GENERATES
the grouped C and confirms the main(step=2·vlmax)+tail shape BEFORE any ODS work. If that second
slice / tail is dead or guarded off in practice, scope collapses toward g3-minimal and the largest
piece of work (the new control op) is saved.

---

## What the realization emits (realization_change)

In `RVVContractionSelectedBodyRealizationOwner.cpp`, replace the metadata-only
`createRealizedVSetVLRegionMarker` placeholders (lines 1996-2008) + single-slice body
(2039-2099) with TYPED structure. Two parts:

### A. packed-i4 nibble unpack (typed op chain replacing the candidate-ID string branch)
For the packed-i4 candidate, before the widening product, emit ONE dedicated typed nibble-unpack op
(see new_ops_needed) that consumes the loaded packed i8mf4 vector and produces the unpacked operands
for the low/high i4 products. The realization is ALLOWED to read the selected candidate (that is the
realization layer's job — it owns lowering the selected Gearbox candidate to typed IR); the win is
that the unpack STRUCTURE now lives in a typed op, so the downstream conversion never reads
`operand_form`/`unpack_intent`. Delete the conversion's need for those strings; they remain
mirror-only (evidence), not on the compute decision path.

### B. grouped multi-region loop + mutable accumulator (typed control op replacing empty markers)
For the grouped candidate (and as the carrier for ALL dequant, since the accumulator crosses i32->f32),
emit a typed multi-region grouped-loop op (see new_ops_needed) whose regions ACTUALLY CONTAIN the
per-region product/reduce slices:
- main region: `unroll_factor` (=2) distinct product/reduce slices (the second slice is the
  currently-synthesized `grouped_loop_vl_u1` second load+product+reduce, `RVVEmitCStatementPlanOwners
  .cpp:1638-1686`), stepping by `vlmax * unroll_factor`;
- tail region: one scalar product/reduce slice (the synthesized `tailLoop`, lines 1688-1743);
- a loop-carried `dot_acc_vec` (i32m1) accumulator value (replacing the unconditionally-injected
  mutable local at lines 1078-1107 / 1504-1510).

`gearbox_cross_region_handoff` is retained but DOWNGRADED to a pure accumulator pass-through SSA
value; its `operand_form`/`unpack_intent`/`realized_unroll_factor`/`realized_vsetvl_region_count`
attrs stay as mirror evidence but are removed from any compute-shaping decision path (the shape is
now in the typed unpack op + grouped-loop op + their regions).

The dequant epilogue (vfcvt/vfmul + scalar extract + scale + store, and the clamp via
Splat/Compare/Select) stays as-is: those ops are already typed (DequantizeOp, SplatOp, CompareOp,
SelectOp) and already converted (`RVVToEmitC.cpp:1086,1000,1065,1091`). No realization change there
beyond placing them in the grouped-loop's post-region epilogue / consumer with_vl.

---

## new_ops_needed (ODS in RVVOps.td)

TWO new typed ops. (NOT shift-kinds on BinaryOp — a dedicated unpack op is lowest-cost: the 7-stmt
nibble sequence is one semantic operation, and granular shift kinds would force the conversion to
re-derive the shift amounts.)

1. **`tcrv_rvv.packed_i4_nibble_unpack`** (or a tight pair: low-nibble-extract / high-nibble-extract
   + widening-product). Consumes the loaded packed i8mf4 vector(s) + vl; carries the i4 sign-extend
   semantics structurally (shift-left-low / arith-shift-right-high to sign-extend each nibble, then
   low-product i16 rescale, then high-nibble vwmacc). Its lowering expands to the fixed intrinsic
   chain `__riscv_vsll_vx_i8mf4` / `__riscv_vsra_vx_i8mf4` / `__riscv_vwmacc_vv_i16mf2` /
   `__riscv_vsra_vx_i16mf2` (the strings currently at `RVVEmitCStatementPlanOwners.cpp:432-448`).
   I5-honesty argument: the op's FIXED intrinsic expansion via `emitc.call_opaque` with string
   callees is explicitly blessed by the master plan ("callee string = legitimate name"); the win is
   that candidate SELECTION moved from conversion-reads-candidate-string to realization-emits-typed-op.

2. **`tcrv_rvv.grouped_product_reduce_loop`** (a typed multi-region / unroll control op). Replaces
   the empty `vsetvl_region_marker` placeholders. Regions: `main` (holds `unroll_factor` slices,
   carries the accumulator as a loop-carried value, step = vlmax * unroll_factor) and `tail` (one
   scalar slice over the remainder). Carries `unroll_factor` as a STRUCTURAL I64 op attr (op-intrinsic,
   like `BinaryOp.kind` / `ReduceOp.result_layout` — NOT a candidate-ID mirror), so the conversion
   reads op semantics, not Gearbox mirror state.

   Lower-risk ALTERNATIVE to evaluate first (the only region-bearing op today is `WithVLOp`,
   single-region, so a multi-region op is new territory): model the grouped shape as TWO `with_vl`
   scopes — a main scope carrying a structural `unroll_factor` attr (holding the unroll_factor slices)
   plus a tail scope — instead of one brand-new multi-region op. This reuses the already-converted
   `with_vl -> emitc.for` machinery and lowers `ConversionTarget` legality risk; pick it if it reaches
   emission parity, falling back to the dedicated multi-region op only if two scopes cannot thread the
   loop-carried accumulator cleanly.

NOT needed: a new op for the mutable accumulator (modeled as the grouped-loop op's loop-carried value;
the conversion lowers it to `emitc.variable`+`emitc.assign`) or for the clamp epilogue (SplatOp/
CompareOp/SelectOp already typed+converted).

---

## conversion_change (lib/Conversion/RVV/RVVToEmitC.cpp)

The op-walk dispatch (`RVVToEmitC.cpp:1086-1198`) currently falls back to legacy for any unhandled
op (`else -> notifyMatchFailure`, line 1195) — so today the dequant body cleanly falls through to the
legacy owner. Add handlers so the dequant body legalizes through patterns instead:

1. **TypeConverter**: add i8/mf4 (`i8mf4 -> vint8mf4_t`) and i16/mf2 (already added for reduction
   owner) and f32/m1 (already present). Extend `vectorDType`/`vectorScalarCType` for i8.
2. **Register accumulator (NEW conversion machinery, needed by BOTH families).** The conversion uses
   NO `emitc::VariableOp`/`AssignOp` today (verified grep = 0) and carries reductions through MEMORY
   (`out[0]`/`acc[0]`). The dequant output is f32 but the accumulator is i32, so the `out[0]` memory
   carry cannot be reused. Introduce the i32 `dot_acc_vec` as `emitc.variable` + `emitc.assign`:
   seed from `acc[0]` before the loop, reassign inside, deriving the i32->f32 crossing from the
   TYPED `standalone_reduce -> gearbox_cross_region_handoff -> dequantize` chain. Allowed-read ledger:
   `standalone_reduce`'s `accumulator_layout`/`result_layout` are op-intrinsic structural attrs (ODS:
   "structural in the selected RVV body ... Route ids ... must not infer ... standalone reduction
   semantics" — `RVVOps.td:3398-3418`), so reading them is thesis-legal (same class as the already-read
   `BinaryOp.kind` / `ReduceOp.result_layout` at `RVVToEmitC.cpp:1185`). Reading `operand_form`/
   `unpack_intent`/`realized_unroll_factor` (candidate mirrors) is FORBIDDEN — replaced by the new ops.
3. **`tcrv_rvv.packed_i4_nibble_unpack` handler**: emit the fixed `vsll/vsra/vwmacc/vsra` intrinsic
   chain via `emitc.call_opaque` with string callees from the op's structure (mirrors `emitWideningProduct`
   at `RVVToEmitC.cpp:1978`).
4. **`tcrv_rvv.grouped_product_reduce_loop` handler**: emit the grouped main `emitc.for`
   (step = vlmax*unroll, walking the main region's `unroll_factor` slices in IR order) + the scalar
   tail `emitc.for` (the tail region), threading the `dot_acc_vec` `emitc.variable` as the loop-carried
   accumulator. This is the structural replacement for the single `emitc.for` at `RVVToEmitC.cpp:831`.
5. **`gearbox_cross_region_handoff` handler**: lower as the accumulator pass-through (value identity),
   ignoring its mirror attrs.
6. **Epilogue**: dequant extract + (float) + *scale + store reuses existing `emitDequantize`
   (`RVVToEmitC.cpp:2534`); clamp reuses SplatOp/CompareOp/SelectOp handlers.
7. **Guards**: malformed bodies -> `notifyMatchFailure` -> legacy fallback (re-guard each negative
   probe), preserving strangler-fig until green on hardware.

---

## byte_identity_tradeoff

**Re-baseline the 2 packed-i4 body-C fixtures to the converted output; gate numerics on a re-run HW
lamp for BOTH families.** Only the 2 packed-i4 fixtures carry body-C gates (verified:
`pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir` = 17
`emitc-to-cpp` RUN lines; `...-dequant-clamp-f32-packed-i4.mlir` = 20; the two grouped/unpacked
fixtures = 0). Byte-identity for packed-i4 is a red herring: the old golden C was produced by the
string machine being retired, so it is NOT an independent authority — the I5-honest authority is the
structurally-derived emission. Re-baseline those CPP CHECK lines to the converted output.

Critically, byte-identity is negotiable but CORRECTNESS + the TUNE are not: because BOTH families'
emission changes (packed-i4 nibble unpack now typed; grouped now a real unroll-by-2 main + tail loop),
re-validate on `ssh rvv` against the existing `widening-product-reduce-dequantize-f32` (and
`dequant-clamp`) HW PASS for the grouped AND packed-i4 candidates — the dequant numerics, especially
the packed-i4 nibble unpack, are currently only golden via the legacy emission.

---

## ordered_steps

1. Pin baseline + SCOPE GATE: GENERATE the legacy grouped C (via `--tcrv-materialize-emission-plans |
   tcrv-translate --tcrv-rvv-emitc-to-cpp`) and CONFIRM the main(step=2·vlmax)+scalar-tail unroll-2
   shape from output, not just source-line cites — this gates the broad scope (if dead/guarded off,
   collapse to g3-minimal and skip the new control op). Also capture legacy golden C for all 4
   dequant fixtures + the grouped/packed-i4 HW PASS as the equivalence + correctness oracle; confirm
   build+lit green baseline.
2. ODS: add `tcrv_rvv.packed_i4_nibble_unpack` and `tcrv_rvv.grouped_product_reduce_loop` (with
   structural `unroll_factor`) to `RVVOps.td`; build the dialect.
3. Realization (`RVVContractionSelectedBodyRealizationOwner.cpp`): emit the typed nibble-unpack op
   for packed-i4, and the typed grouped-loop op (main `unroll_factor` slices + scalar tail) carrying
   the loop-carried accumulator, replacing the empty region markers and the single-slice body. Add a
   structural lit test asserting the realized IR carries the new ops + regions (not strings).
4. Conversion (`RVVToEmitC.cpp`): TypeConverter i8mf4; register-accumulator (`emitc.variable`/
   `assign`); handlers for the unpack op, the grouped-loop op, and the handoff pass-through; reuse
   `emitDequantize` + Splat/Compare/Select for the epilogue/clamp. Guard malformed bodies to legacy.
5. Structural conversion lit test: FileCheck the emitted emitc (`emitc.for` main step=vlmax*2,
   tail `emitc.for`, `emitc.variable dot_acc_vec`, the nibble `emitc.call_opaque` chain).
6. Re-baseline the 2 packed-i4 body-C fixtures' CPP CHECK lines to the converted output (documented
   as I5-honest authority, old golden = retired string machine).
7. ssh-rvv hardware lamp: re-run the bundle-ABI e2e for the grouped AND packed-i4 dequant +
   dequant-clamp ops WITHOUT `--dry-run`; require `PASS op=... tolerance=1e-05` +
   `ssh_evidence=true`. Record evidence.json + remote stdout.
8. Flip default + delete the dead path: remove the shared `RVVEmitCStatementPlanOwners.cpp`
   DirectContraction DEQUANT body-emission path (the `usesGroupedLowPrecisionProductReduction` /
   `usesPackedI4LowPrecisionProductReduction` branches and the dot_acc_vec/grouped_tail_start/nibble
   string synthesis). KEEP `RVVEmitCContractionRouteFamilyPlanOwners.cpp` (0 statement-plan steps;
   6/6 prior owner retirements kept the *RouteFamilyPlanOwners.cpp files — precedent). Re-green
   build + lit (honest-green, ≤3 environmental reds).

### Honesty caveat on "retire the contraction owner"
"Retire the owner" = retire the shared `RVVEmitCStatementPlanOwners.cpp` DirectContraction dequant
BODY path, NOT delete the 11,745-line `RVVEmitCContractionRouteFamilyPlanOwners.cpp` (it has 0
statement-plan steps and is the route-family/provider-facts/diagnostic layer; no prior owner
retirement deleted any *RouteFamilyPlanOwners.cpp). The real dead-code win is the shared body path.

---

## risks

1. **PARITY-GAP / TUNE-CUT TRAP (top risk).** Declining the grouped unroll-by-2 (g3's tempting "no
   lit gate" shortcut) emits an unroll=1 loop where legacy emits unroll-2 — an un-validated structural
   change to a shipped artifact (parity break), and a silent collapse of a SELECTED grouped candidate's
   shape (tune erosion the user has pushed back on). Note (anti-overclaim): grouped's unroll-2 is
   selected, not HW-MEASURED, so the argument is parity-first, tune-ownership-second — not "unroll-2 is
   a proven win." Mitigation: structuralize the unroll-by-2 main+tail loop as typed regions (Part B);
   the converted C must still be unroll-by-2; re-validate the grouped HW lamp.
2. **Allowed-read ledger drift.** The conversion must read ONLY op-intrinsic attrs
   (`standalone_reduce.accumulator_layout/result_layout`, the new op's `unroll_factor`) and NEVER the
   candidate mirrors (`operand_form`/`unpack_intent`/`realized_unroll_factor`/`resource_selected_candidate`).
   Mitigation: assert in the structural lit test that the converted IR derives from the new ops, and
   keep the mirror attrs evidence-only on the handoff.
3. **packed-i4 numeric correctness.** The nibble unpack is correctness-load-bearing and currently
   golden only via legacy emission; a wrong shift amount / sign-extend = wrong result. Mitigation:
   re-run the packed-i4 HW lamp (not dry-run) before deleting the legacy path.
4. **New control op + partial conversion legality.** A multi-region grouped-loop op is new territory
   (today the only region-bearing op is `WithVLOp`, single region); mixed converted/unconverted
   families could trip `ConversionTarget` legality. Mitigation: gate the target by op-kind; keep the
   grouped-loop/with_vl conversion atomic with the dequant family; guard malformed -> legacy.
5. **Accumulator i32->f32 crossing.** First use of `emitc.variable`/`assign` in the conversion +
   first i32-accumulator-into-f32-dequant; the seed/reassign placement must match the legacy carry
   semantics. Mitigation: derive purely from the typed `standalone_reduce->handoff->dequantize` chain;
   diff converted C against the legacy carry before re-baselining.
6. **HW access flakiness.** `ssh rvv` (ProxyJump) can be down; an I8 claim must not be asserted from
   a dry-run. Mitigation: gate the legacy-path deletion on real `ssh_evidence=true`.
7. **Scope creep.** The grouped-loop op is the deepest piece; resist generalizing it into a full
   scf-style loop dialect. Mitigation: scope it to the two dequant families' main+tail shape only.

---

## effort

Large but bounded and precedented (comparable to the reduction-owner retirement that already shipped,
plus two new ODS ops and the first register-accumulator + first multi-region control op in the
conversion). Sequencing: ODS+realization (the deepest, ~the bulk) then conversion handlers then
re-baseline+HW lamp then deletion. This is the genuine last/deepest Stage-3 rung — materially more
than a per-family conversion, but it is the only thesis-honest way to retire the contraction owner's
dequant body path.
