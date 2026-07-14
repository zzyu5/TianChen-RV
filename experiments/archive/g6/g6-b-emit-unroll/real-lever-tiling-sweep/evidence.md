# G6-B P3 · q2_K real-lever (register-resident rolled) · FEASIBILITY + tiling sweep

- **campaign**: G6-B emit-quality lever — P3 pursues the *true* q2_K lever named in P2:
  a **rolled-WITHOUT-accumulator-spill** tiling (compact ∧ register-resident *simultaneously*, like the
  STOCK hand-brick's vwmacc=16 spill=4), which NEITHER prior emit form produces (unrolled = register-resident
  but 2304-vwmacc code volume; rolled-panel = 384 vwmacc but the hot i16 partials stack-panel round-trip).
- **date**: 2026-07-13 · **status**: P3 COMPLETE · feasibility POSITIVE, emitter change built+byte-exact,
  rvv clang-17 tiling sweep MEASURED · **VERDICT = feasibility boundary (exit b), NO k1 e2e probe warranted**.
- **board**: rvv / openEuler / riscv64 / VLEN128 / 64c · clang-17.0.6 (gcc-15 absent on board ⇒ clang-17
  SYMMETRIC A/B, matches P1 rvv seal; valid per [CASE-COMPILER-ASYMMETRY] — all configs one compiler).
- **emitter change (AUTHORIZED by coordinator, emit-quality campaign)**: `lib/Conversion/RVV/
  RVVToEmitCBlockQuantLinear.cpp` +112 lines (3 edits: `#include <cstdlib>`; two env MEASUREMENT knobs after
  `columnsPerPass`; a rolled-LVALUE-partials branch in `emitRepackKQuantGemmBodyQ2K`). **Byte-exact-neutral
  default**: env-unset reproduces P1 provenance md5 EXACTLY (unrolled `2dd964fe`, rolled-panel `4e1f8089`) ⇒
  ZERO shipped-behavior change. Coordinator confirmed disjoint from the M3/IME line. Post-edit md5 `53f8f8c4`.
- **A-tree**: NONE touched (standalone .o microbench; NO deployed lib, NO ggml tree, NO seal). No git add.

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

## BYTE-EXACT hard gate — GREEN all configs (rvv clang-17, verifier oracle, seed 0xC0FFEE)
Independent scalar ZERO-MODEL oracle (`kquant_repack_verify_q2K`, 8 shapes): rolled_lvalue t1/t2/t4 AND
rolled_panel_t1 ALL **INT_mismatch_total=0**, NORM worst_ulp=**3402** — IDENTICAL to unrolled_default to the
digit. ⇒ the lvalue rewrite + the nr knob are byte-exact (columns accumulate independently; vwmacc16 order
+ fold unchanged). The env-unset default md5-matches P1 (above) = shipped path provably untouched.

## FORM seal — clang-17 -O2 SYMMETRIC (full q2_K GEMM kernel objdump)
| config | textB | vsetvli | vwmacc | vse16/vle16 (panel RT) | vsNr/vlNr (VEC spill) | sp_sd/ld (SCALAR spill) |
|---|---:|---:|---:|---:|---:|---:|
| unrolled_default (shipped) | 26968 | 86 | 2304 | 64 / 68 (cold floor) | **7** / 12 | 36 / 42 |
| rolled_panel_t4 (P1)       | 12522 | 82 | 384  | **320 / 324** (HOT RT) | 63 / 73 | 100 / 116 |
| rolled_lvalue_t4           | 10530 | 86 | 384  | **64 / 68** (RT GONE)  | 65 / 88 | 71 / 84 |
| rolled_lvalue_t2           | 10392 | 172| 384  | 128 / 136              | **10** / 27 | 75 / 109 |
| rolled_lvalue_t1           | 15380 | 328| 384  | 256 / 272              | **9** / 35  | 92 / **248** |
- **The lvalue rewrite WORKS at FORM**: it ELIMINATES the P1 hot stack-panel round-trip (lvalue_t4 vse16
  320→64, back to the unrolled cold floor). **The nr knob WORKS at FORM**: narrowing tile collapses VECTOR
  spill (vsNr 63→10→9) — the P2 "smaller nr → register-resident" hypothesis is CONFIRMED at the FORM level.
- BUT no config reaches the hand-brick's **vwmacc=16 ∧ spill=4**: every rolled form is vwmacc=**384** (we roll
  only the innermost 16-iter mm loop, NOT the k/mh/sub-block nest) and all carry heavier scalar spill than the
  unrolled. Narrowing to win vector-residence PAYS with redundant weight re-decode (t1 = 4 cLo passes, textB
  15380, sp_ld 248).

## KERNEL-AXIS throughput — load-gated, single-core (taskset -c 4), clang-17 symmetric, relIQR ≤0.51%
GMAC/s (higher=faster), ×unrolled_default. Two shapes, vLLM idle (1.7% CPU), loadavg1 2.3–2.4 (bg floor).
| config | shape A (nr4·nc512·n4096, ×3) | shape B (nr16·nc256·n2560, ×2) | ×unrolled | vs P1 panel |
|---|---:|---:|---:|---:|
| **unrolled_default** | **6.93** | **7.05** | **1.000×** | 2.45× |
| rolled_lvalue_t4 | 4.39 | 4.41 | **0.63×** (best rolled) | **1.55×** |
| rolled_lvalue_t2 | 4.22 | 4.35 | 0.61× | 1.52× |
| rolled_lvalue_t1 | 2.86 | 2.92 | 0.41× | 1.01× |
| rolled_panel_t4 (P1) | 2.83 | 2.86 | 0.41× | 1.00× |
- **The lvalue fix beats the P1 rolled-panel 1.55×** (4.39 vs 2.83) — confirms the P2 spill diagnosis: the hot
  stack-panel round-trip WAS the P1 rolled regressor, and eliminating it recovers a big chunk.
- **BUT the best rolled (lvalue_t4) is still only 0.63× the shipped unrolled.** No rolled config reaches parity.
- **The nr knob FALSIFIES on THROUGHPUT (opposite of hypothesis)**: smaller tile → more register-resident
  (vsNr↓) but SLOWER (lvalue_t1 0.41× ≈ P1 panel). Throughput tracks **redundant-weight-decode passes +
  scalar spill**, NOT vector-register-residence. lvalue_t4 wins among rolled only because 1 cLo pass = no
  redundant decode (despite vsNr=65). Ordering + ratios identical across both shapes ⇒ robust, shape-independent.

## VERDICT — exit (b): FEASIBILITY BOUNDARY (measured, honest). NO k1 e2e probe.
The register-resident-rolled lever P2 named IS expressible, byte-exact, and BUILT (delivered), and it measurably
beats the P1 rolled-panel form (1.55×) — but it does **NOT** reach the shipped unrolled (best 0.63× kernel-axis),
and the tile-width knob that achieves FORM vector-residence **actively hurts** throughput (redundant decode
dominates). Since the deployed unrolled already trails the stock hand-brick e2e (P2: 0.873×), a rolled form at
0.63× kernel-axis would be **worse** e2e ⇒ **no k1 e2e成色质变 probe warranted** (exit a NOT reached).
**Named boundary**: the hand-brick's compact(16 vwmacc) ∧ resident(spill 4) form is NOT reachable by rolling the
inner mm loop + narrowing tile — it requires rolling the **entire K-super/sub-block nest** into runtime loops with
a single wide resident accumulator (a different S6 loop structure). That deeper restructuring, not this schedule/
tile knob, is the remaining q2_K emitter-maturity target — and per the compute-bound kernel-axis + P2 e2e evidence
it is a **mechanism/method target, not a warranted perf battle** (q2_K prefill is fastest as the shipped unroll).

## Touch list (this task)
- **MODIFIED (authorized)**: `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` (+112 lines, post-edit md5
  `53f8f8c4`; was P1 `2a3a234f`). Byte-exact-neutral default (env-unset reproduces P1 md5s). Coordinator commits.
- **NEW casefile**: this `evidence.md`. Board scratch `/tmp/g6b-p3-sweep` (rvv, standalone microbench artifacts;
  no deployed lib / ggml tree touched). NO ODS/verifier/schema/T8/ROADMAP touched. NO git add/commit by me.
