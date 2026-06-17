# INC-6 — Compiler-driven Q4_0×Q8_0 autotuner: the COMPILER SELECTS the kernel shape

The deliverable: make the COMPILER select the ggml `ggml_vec_dot_q4_0_q8_0`
kernel shape via a capability/resource-aware autotuner — enumerate candidates →
prune by facts → rank by a principled cost → SELECT → stamp onto the op. The
shape is DERIVED from capability + resource facts, NOT a lookup table. The win
(beat ggml) is a *consequence* of the selection, not the goal.

## The three pieces (DERIVED, not a lookup)

1. **Capability fact** — `deriveHasZvl128b(march, isaVectorHints)` in
   `lib/Plugin/RVV/RVVCapabilityProfile.{cpp,h}` (mirrors `deriveSupportedSEW
   AllowList`). Real ISA reasoning: the ratified RISC-V "V" extension MANDATES
   Zvl128b, so full-V (rv64gcv) ⇒ VLEN≥128 ⇒ true; plain zve32x/zve64x mandate
   only Zvl32b/Zvl64b ⇒ false (unless an explicit `zvl{N≥128}b` token is named).

2. **Schedule engine** (enumerate→prune→rank→select) in
   `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h` (mirrors the i8 LMUL rung
   engine ~1879–2085). The Q4_0 candidate space is `{integer_core_lmul ∈ mf4,m1}
   × {multi_block_factor ∈ 1,2,4} × {strip_elision ∈ robust,elided}` = 12
   candidates. PRUNE by (a) **legality**: `elided` requires `m1` AND the Zvl128b
   fact (the ONLY place capability enters); (b) **vreg budget**: peak-live
   acc+product+load+reserve ≤ 32 (genuine — a shrunk budget binds; never binds
   on this light ≤6-vreg kernel). RANK by a **CAPABILITY-BLIND** structural cost
   (`computeRVVQ40ShapeCost(lmul, factor, elision)` — no capability argument):
       cost = kReductionUnit·reductions(lmul)        // mf4=4×, m1=1× → mf4 loses
            + kOuterLoopOverhead / factor             // amortized by unroll
            + kStripPenalty(elision)·factor           // robust>0 (U-curve), elided≈0
            + kBaseConstant
   Constants `{600, 500, 170(robust)/40(elided), 120}` are MEASUREMENT-CALIBRATED
   to this board's design-space sweep (see §win): robust reproduces the measured
   U-curve 1390/1310/1525 (min@factor=2), elided reproduces 1005 @factor=4.
   SELECT min-cost legal candidate (`selectRVVQ40Q80MinCostShape`).
   **Calibration scope (honest):** the constants are fit to the measured
   anchors — the robust F1/F2/F4 sweep (1388/1306/1525) and the elided F4 point
   (1021); the model's intermediate elided costs (F1=1260, F2=1050) are NOT
   pinned to the S1 `body_f{1,2}_elided.inc` numbers. This does not affect the
   argmin (elided is monotone-decreasing in F, so F4 is the elided minimum
   regardless), and the discriminating ordering (mf4 ≫ m1; robust min@2; elided
   min@4; elided-F4 < every robust) is what the selection uses.

3. **Stamping** — `lib/Plugin/RVV/RVVQ40ScheduleMaterialization.cpp`, the
   `--tcrv-rvv-materialize-q4-0-schedule=march=…` pass (mirrors
   `RVVProbedCapabilityAxesMaterialization.cpp`). Walks each
   `tcrv_rvv.q4_0_q8_0_block_dot`, derives Zvl128b, runs the engine, and
   `setAttr`s the chosen knobs (integer_core_lmul / multi_block_factor /
   strip_elision) + a `tcrv_rvv.q4_0_schedule.*` resource-provenance audit trail.
   No-clobber: an op already carrying a hand-set knob is left untouched (additive;
   existing fixtures byte-identical).

## Why it is DERIVED, not a lookup

Capability enters ONLY through the legality prune (`enumerateRVVQ40Q80Shape
Candidates`: `elision=="elided"` legal iff `coreLMUL=="m1" && hasZvl128b`). The
cost model is a pure structural function with NO capability argument. So the SAME
argmin yields different winners purely because the admitted candidate set
differs:
  * Zvl128b=true  → elided admitted → min-cost legal = (m1, 4, elided), cost 1005
  * Zvl128b=false → elided pruned   → min-cost legal = (m1, 2, robust), cost 1310
The unit test flips ONLY the Zvl128b boolean (not the march, not the cost) and
the shape flips elided↔robust — the derivation, not a constant.

## Divergence (N1, the headline) — `test/Conversion/RVV/rvv-q4-0-q8-0-block-dot-autotuner-divergence.mlir`

The SAME attr-less kernel (no shape knobs), two RUN lines:
  * `--march=rv64gcv`      → compiler stamps + emits **mb4-elided** (4 adjacent
    vwredsums, NO inner strip for-loop, by-4 outer loop) — the compiler SELECTING.
  * `--march=rv64gc_zve32x` → compiler stamps + emits **mb2-robust** (per-block
    inner strip for-loop kept, by-2 outer loop). The elided shape was PRUNED.
Two different emissions FROM THE SAME INPUT, purely by capability. PASS.

## Byte-exact preserved (ssh rvv)

The autotuner-selected (rv64gcv → mb4-elided) kernel is **byte-identical** to the
committed S1 `kern_mb4_elided.cpp` (mlir-translate `--mlir-to-cpp` diff = 0,
modulo the function name) and to the committed `mb4-elided.mlir` lit fixture's
lowered emitc (diff exit 0). The zve32x→mb2-robust selection is byte-identical to
`kern_mb2_robust.cpp`. Since S1 (8a7c5e36) ssh-rvv-validated mb4-elided as bitwise
== ggml real + `_generic` over 3300 checks/mode, the autotuner output inherits
byte-exactness by construction (it IS that body, stamped by the compiler).

## WIN (N3, ssh rvv) — re-confirmed on the AUTOTUNER-SELECTED kernels

Board `ssh rvv` (VLEN=128, rv64imafdcv+zvfh), `taskset -c 3`, best-of-40 min
ns/call, `-O3 -march=rv64gcv_zfh_zvfh -mabi=lp64d -ffp-contract=fast`, n=4096,
all shapes bitwise == ggml (ref −5.22822143e+12). The mb4-elided / mb2-robust
kernels here were GENERATED BY THE AUTOTUNER PASS (`board_microbench_run{1,2}.txt`):

| shape (autotuner-selected) | ns/call | vs ggml | note |
|---|---|---|---|
| ggml (i8m1, serial)            | 1169–1293 | 1.00x | reference (hand-written RVV) |
| mb1-robust (OLD hand-set anchor)| 1388 | 1.07–1.19x | the pre-autotuner anchor |
| mb2-robust (autotuner zve32x)  | 1306 | 1.01–1.12x | robust optimum |
| **mb4-elided (autotuner rv64gcv)** | **1021** | **0.79–0.87x** | **~13% FASTER than ggml** |

The autotuner-SELECTED rv64gcv kernel (mb4-elided, 1021 ns) beats BOTH ggml
(1021 < 1169) AND the old hand-set mb1 anchor (1021 < 1388, ~26% faster), rock-
stable across both runs (1021.28 ns both times). The ggml reference is warmup-
sensitive (1169→1293) but the SELECTED kernel beats it in every run.

## Validation summary
- `raw(` count in emitted code = 0 (structured emitc, I5).
- Selector unit test (`test/Plugin/RVVQ40Q80ShapeSelectionTest.cpp`): Zvl128b⇒mb4
  -elided, ¬Zvl128b⇒mb2-robust, fact-flip flips shape, budget prune binds,
  Zvl128b derivation real — ALL PASS.
- `check-tianchenrv`: 610 tests, 607 pass, exactly the 3 documented environmental
  reds (the `Scripts/rvv-generated-bundle-abi-e2e-*` board-toolchain tests).
- Existing q4_0/mb2-robust/mb4-elided/deferred-wide lit + byte-identical preserved
  (additive; the no-clobber guard leaves hand-set fixtures untouched).
