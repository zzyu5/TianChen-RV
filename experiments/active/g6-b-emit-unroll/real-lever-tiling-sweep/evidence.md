# G6-B P3 · q2_K real-lever (register-resident rolled) · FEASIBILITY assessment

- **campaign**: G6-B emit-quality lever — P3 pursues the *true* q2_K lever named in P2:
  a **rolled-WITHOUT-accumulator-spill** tiling (compact ∧ register-resident *simultaneously*, like the
  STOCK hand-brick's vwmacc=16 spill=4), which NEITHER current emit form produces (unrolled = register-resident
  but 2304-vwmacc code volume; rolled = 384 vwmacc but spill=53 stack-panel round-trip).
- **date**: 2026-07-13 · **status**: P3 **FEASIBILITY GATE** · source + toolchain analysis (NO board contact this task)
- **scope**: judge whether the emitter can express *variable output-tile-width ("nr") + rolled with
  register-resident accumulators*. Board tiling sweep is GATED on this gate + emitter-change coordination.
- **board/tree**: NO rvv/k1 contact (feasibility answered from source + LLVM-20 EmitC.td + existing existence
  proof). Emitter READ-ONLY, untouched: `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` md5 `2a3a234f...`
  (= P1 state). No A-tree action, no build, no md5 double-cert needed. No git add/commit.

## VERDICT — FEASIBILITY **POSITIVE (expressible)** but **REQUIRES A BOUNDED EMITTER CHANGE** → coordination-gated
The framework CAN express register-resident rolled accumulators. It is **NOT** a framework-level impossibility.
BUT it is **not reachable by any existing parameter** — the current rolled path hard-codes a stack panel.
Realizing it needs TWO coupled edits to the SHARED emitter → reported for coordination before touching it
(hard constraint "若改发射器先报我协调"; memory `parallel-lines-need-disjoint-files` — P1 committed the
rolled path into this same file, shared-file edits must serialize).

## Evidence 1 — `emitc.for` is **iter-arg-less** (LLVM-20, the toolchain this project links)
`/usr/lib/llvm-20/include/mlir/Dialect/EmitC/IR/EmitC.td` `def EmitC_ForOp`: `let results = (outs);`
"This operation has no result." ⇒ NO loop-carried SSA iter_args. The ONLY mechanism for a value that
survives across an `emitc.for` iteration is an `emitc.variable` **lvalue** (load/assign inside the loop) —
whose C-level residence (register vs stack) is decided by whether its **address is taken**.

## Evidence 2 — register-resident lvalue across a RUNTIME emitc.for is ALREADY PROVEN *in this function*
`emitRepackKQuantGemmBodyQ2K` `sumfVar[c]` (line ~11140): an `emitc::VariableOp` of `LValueType(f32m2)`,
seeded ABOVE the runtime contraction-block loop (`blockLoop = emitc::ForOp` over `nb`, ~11150), then
load(11477)/assign(11497) INSIDE it. Its address is never taken ⇒ mem2reg keeps it in a vreg. The code
comment (11130-11132) states it verbatim: *"columnsPerPass accumulators seeded ABOVE this strip's block loop
and carried across it as SSA-register VariableOps (NEVER rolled into the iter-arg-less emitc.for)."*
The STOCK hand-brick (vwmacc=16, spill=4, rolled) is the toolchain existence proof that a rolled loop CAN
keep its partials register-resident. ⇒ the mechanism the true lever needs already exists and is proven here.

## Evidence 3 — the CURRENT rolled path DELIBERATELY spills the HOT partials (why spill=53)
Rolled branch (line ~11330, `if (rolledMainTerm)`): the per-column-per-shift i16 partials (`sPartial`) are
staged to `sPartialPanel = emitc::ArrayType(...)` (11342-11348), addressed via `panelPtr` →
`emitc::SubscriptOp` + `emitc::ApplyOp "&"` (11225-11234). **The taken address forces stack residence**
(the code comment 11194-11196 states this is exactly how the COLD bands are deliberately paneled) — but the
rolled path applies it to the HOT partials too ⇒ load-accumulate-store every mm iteration
(324 vle16 + 373 vse16, spill 6→53, P2 objdump). The UNROLLED path keeps `sPartial` as SSA registers
(11414-11418, `SmallVector<Value>` not an array) — register-resident but fully unrolled (2304 vwmacc).
Neither form does compact ∧ register-resident. That gap is the lever.

## The change the true lever needs (two coupled edits — NOT a pure param sweep)
1. **Register-resident rolled**: in the rolled branch, carry `sPartial` as `columnsPerPass*4` `emitc::VariableOp`
   **lvalues** (i16m1, seed/load/assign inside the runtime mm `emitc.for`) INSTEAD of `sPartialPanel` —
   mirroring the proven `sumfVar` mechanism. Byte-exact preserved (same vwmacc16 order; only residence changes).
2. **Output-tile-width ("nr") knob**: `columnsPerPass` is currently `(coreLmul=="m1")?1:activationInterleave`
   (=4 for the mf2 RVV1.0 core) — tied to core LMUL, no independent knob. Add one so the main-term
   partial-carry tile can be narrowed for the register-residence path.

## Live-set math (mf2 core: l16=m1 ⇒ i16m1=1vreg; l32=m2 ⇒ i32m2/f32m2=2vreg) — quantifies the hypothesis
rolled mm-loop peak-live with lvalue partials at tile-width C ≈ **4C** (partials) + **2C** (sumiVar must
survive the loop) + wLane/act/temps overhead ≈ **6C + ~6**:
- **C=4** (current default) → ~30+ vreg → **spills** (matches observed rolled spill=53).
- **C=2** → ~18 vreg → **likely register-resident**.
- **C=1** → ~12 vreg → **comfortably register-resident** (closest to hand-brick 16x1).
⇒ the P2 "smaller nr → accumulators stay in registers" hypothesis is quantitatively sound. Sweep design:
C∈{1,2} = register-resident candidates, C=4 = spill control; × {rolled-lvalue, rolled-panel(current), unrolled}.

## Pre-registered exit (P2 dual exit, carried forward)
- Gate outcome = **(b')-conditional**: NOT "framework-level impossible" (exit b) and NOT yet "promising config
  found" (exit a). It is **expressible via a bounded, byte-exact SHARED-emitter change** → main session must
  (i) authorize the emitter edit and (ii) ensure no parallel line is mid-edit on this file, THEN the board
  tiling sweep (nr∈{1,2,4} × {rolled-lvalue, panel, unrolled}, rvv gcc-15+clang symmetric, byte-exact hard
  gate, objdump FORM: target spill~4 ∧ vwmacc low, kernel-axis vs stock 16x1) can run. Candidate that reaches
  register-resident ∧ compact ∧ kernel-axis ≥ hand-brick → flag k1 e2e成色-质变 probe (exit a).

## Touch list (this task)
READ-ONLY. Read: P1/P2 evidence.md, RVVToEmitCBlockQuantLinear.cpp (ranges), LLVM-20 EmitC.td, CMake config.
NO writes to tracked source. Only new file = this casefile (experiments/, untracked). Emitter md5 unchanged.
