# Research: Gearbox current state — real autotuner vs MVP placeholder (N3 性能灯)

- **Query**: Is the TianChen-RV RVV Gearbox a REAL capability/resource-aware autotuner, or an MVP placeholder? (spec bar: enumerate + prune-by-resource-facts; and 实测赢 scalar 且赢 naive RVV)
- **Scope**: internal (code + committed measurement artifacts)
- **Date**: 2026-06-14
- **Spec bar**: `.trellis/spec/variant-pipeline/generation-selection-tuning.md` lines 51–55 (Resource model + "怎么判断 tune 是否真的 resource-aware")

## VERDICT

**MVP placeholder with elaborate scaffolding — NOT a resource-aware autotuner, and NOT a measured win.**

It *does* enumerate >1 candidate and *does* have a formal prune step, so it is one notch past the "固定单候选" floor on the product-reduction path. But:
- the candidate facts are **hardcoded constants**, not derived from capability/body facts (fails the spec's "必须由编译器可见的 capability + body facts 推导，而非硬编码常量");
- selection is a **max-unroll tiebreak**, not a cost model over the facts;
- the resource prune (budget) is **inert** (never binds);
- and on real `ssh rvv` hardware **no candidate wins**: the auto-selected one is a measured **regression** (~2x slower than scalar C), the other is **no-win** (parity).

There are **two distinct Gearbox surfaces** and they differ:
- **Dequantize i32→f32 schedule** (`materializeGearboxAttrs`, RVVGearboxSchedules.cpp:182–250) = the literal MVP "固定单候选": one labelled candidate, all-constant attrs (CandidateSet/SelectedCandidate/SelectionReason/Unroll=2 are fixed string/int literals, header lines 442–458).
- **Low-precision product-reduction schedule** (`buildRVVLowPrecisionProductReductionResourceCandidates`, header 1257–1499) = the 3-variant enumerate+prune path analyzed below. This is the most "autotuner-like" surface and is still constants-driven.

---

## Findings

### Files Found

| File Path | Role |
|---|---|
| `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h` | **Heart of it.** Candidate constants (733–746), candidate enumeration `build…ResourceCandidates` (1257–1499), pruning/legality (1344–1497), selection `select…ResourceCandidate` (1540–1551). All inline. |
| `lib/Plugin/RVV/RVVGearboxSchedules.cpp` | MLIR pass. `materializeGearboxAttrs` (182–250) = single-labelled dequant schedule. `materializeLowPrecisionResourceAttrs` (515–) calls build+select and stamps facts as attrs. |
| `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp` | Re-runs build+select at realization (340–909) and cross-checks the pass-stamped facts; emits typed body ops. |
| `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyLowPrecisionResource.cpp` | Route-family plan owner; reads/mirrors the resource facts into the EmitC route (callers at 745, 2584, 2688). |
| `lib/Plugin/RVV/EmitC/RVVLowPrecisionPerformancePolicy.cpp` | Low-precision performance policy surface (267 KB; admission/no-win plumbing). |
| `scripts/rvv_generated_bundle_same_target_measure.py` | **The gate4 measurement tool.** Builds generated artifact + scalar-C baseline, runs both on `ssh rvv`, correctness-before-timing, emits speedup + `result_classification`. (Task referred to this as `same_target_measure.py`; that exact path does not exist.) |
| `artifacts/gate4-candidate-feedback-ssh/.../{grouped-u2,packed-i4}/.../same_target_measurement_evidence.json` + `remote_measure_run_stdout.txt` | **Committed real-hardware runs** (commit 28d9e4ac "measure labelled gearbox candidates"). Contain actual per-iter ns + speedup + classification. |

---

### Q1 — Enumeration: YES, 3 candidates (product-reduction path), but from CONSTANTS

`buildRVVLowPrecisionProductReductionResourceCandidates` (header 1257) always builds exactly **3** candidates per operation:

| Candidate | unrollFactor | accumulatorCount | vsetvlRegionCount | peakLiveVectorGroups | header line |
|---|---|---|---|---|---|
| base (byte) | 1 | 1 | 2 | 4 | 1299–1304 |
| grouped | 2 | 2 | 3 | 7 | 1374–1380 |
| packed-i4 | 1 | 1 | 2 | 5 | 1421–1427 |

Every one of these values is a hardcoded literal, declared at header **733–746**:
```cpp
constexpr std::int64_t kRVVLowPrecisionResourceStaticUnroll = 1;          // 733
constexpr std::int64_t kRVVLowPrecisionResourceAccumulatorCount = 1;      // 734
constexpr std::int64_t kRVVLowPrecisionResourceVSetVLRegions = 2;         // 735
constexpr std::int64_t kRVVLowPrecisionResourcePeakLiveVectorGroups = 4;  // 736
constexpr std::int64_t kRVVLowPrecisionResourceGroupedUnroll = 2;         // 737
... GroupedAccumulatorCount=2, GroupedVSetVLRegions=3, GroupedPeakLiveVectorGroups=7
constexpr std::int64_t kRVVLowPrecisionResourceVectorRegisterBudget = 32; // 746
```
The "resource facts" the spec lists (VLEN/ELEN, SEW/LMUL/EMUL, peak-live groups, vsetvl region count, accumulator count, unroll, mask/tail policy) are all **carried as fields on the candidate**, but they are assigned from these constants — not *computed from* target capability or body facts. SEW/LMUL come in as function args but are immediately asserted to be exactly the one supported shape (i8mf4→i16mf2→i32m1→f32m1, header 1344–1347 / RVVGearboxSchedules.cpp 299–309), so there is no real shape/LMUL space to search.

The dequant-i32→f32 surface (`materializeGearboxAttrs`) is even thinner: **one** labelled candidate, no enumeration at all — `kRVVGearboxDequantizeI32ToF32CandidateSet` / `SelectedCandidate` / `SelectionReason` / `Unroll=2` are fixed (header 442–458). This is the literal "固定单候选、固定 unroll" MVP the spec names.

### Q2 — Pruning: PRESENT but INERT; selection is a heuristic, not a cost model

**Prune step (header 1344–1483):** each candidate gets
```cpp
isLegal = hasSupportedShape && hasSupportedPolicy && isWithinRegisterBudget;
```
- `hasSupportedShape` — fixed equality against the one i8mf4… shape (1344–1347).
- `hasSupportedPolicy` — `tailPolicy=="agnostic" && maskPolicy=="agnostic"` (1348–1349).
- `isWithinRegisterBudget` — `peakLiveVectorGroups <= vectorRegisterBudget`, i.e. **max 7 ≤ 32** (1350–1351). This **can never fail** for any built candidate. The one genuinely fact-shaped prune dimension is dead weight.

So pruning only ever rejects on the two fixed equality checks (shape/policy); it never prunes by resource pressure. "按 resource facts 剪枝候选" is formally instantiated but does no resource reasoning.

**Selection (header 1540–1551):**
```cpp
for (candidate : candidates)
  if (candidate.isLegal)
    if (!best || candidate.unrollFactor > best->unrollFactor)
      best = candidate;            // pick max unrollFactor
```
This is a **single-key max-unroll tiebreak**. It always returns the grouped candidate (unroll=2). It does **not** reason over the cost/resource facts it carries — there is no cost model, no comparison of vsetvl regions / peak-live / accumulator layout, no profile feedback driving the choice. The packed-i4 candidate carries a `resourceCostModel`/`resourceCostContract` string and a long `performanceAdmission*`/`remediation*` apparatus (header 603–732, populated at 1428–1484), but those are **descriptive string constants**, not inputs to selection — and they mark packed-i4 as `PerformanceSelectionEligible="false"` / `PerformanceMaturityOutcome="no-win"` (header 727–729) anyway.

### Q3 — Measured win: NO. Real hardware runs exist, and every candidate is no-win/regression

Real `ssh rvv` measurement IS wired and HAS been run and committed (commit 28d9e4ac). Target is genuine riscv64 hardware (`remote_arch=riscv64`, `clang 18.1.3`, 64 CPUs; `remote_target_profile_stdout.txt`). Correctness-before-timing guard passes. The committed `result_classification` in `same_target_measurement_evidence.json`:

| Candidate (selected = grouped-u2) | best_speedup range vs scalar-C | classification | outcome_family |
|---|---|---|---|
| **grouped-u2 (auto-selected)** | **0.501820 .. 0.610294** | **regression** | no-win |
| packed-i4 | 0.901869 .. 1.021921 | no-win | no-win |

`classification_rule`: *win iff every SUMMARY best_speedup > 1.0; regression iff every < 1.0; otherwise no-win.*

- The candidate the selector actually picks (grouped-u2) is **~1.6–2.0x slower** than scalar C on every size (n=257/4096/65536): generated ~340/4810/76485 ns vs baseline ~205/2420/41300 ns (`grouped-u2/.../remote_measure_run_stdout.txt`).
- packed-i4 is at best parity (≈1.02 at large n, 0.90 at small n) → **no-win**.
- Provider perf facts are hardcoded to match this honesty: `PackedI4PerformanceBestSpeedupRange = "0.895307..1.027027"`, `PerformanceMaturityOutcome="no-win"`, admission `"deny-performance-preferred...no-win-blocker"` (header 605–729).

So: cycle/time data exists and is real; the answer to "has any candidate won?" is **no**. Evidence is correctness-PASS + measured-no-win/regression. Selection is provably **not** performance-driven: the selection key is `unrollFactor`, and the measurements show unroll is not a performance proxy — the selected unpacked-byte schedule (grouped-u2, unroll=2) measures as a **regression** vs scalar, while the only candidate reaching parity (packed-i4) is a different kernel on a different input encoding (packed nibbles, separate scalar baseline) and is marked `SelectionEligible="false"` anyway. There is no fact-driven ranking that would pick a winner, because none exists.

### Q4 — The gap (ranked by effort, low→high)

What's missing to make this a real resource-aware autotuner that WINS:

1. **(d) naive-RVV baseline — LOWEST effort.** The harness has only `scalar-c-reference/*` baselines (`BASELINE_IDENTITIES`, script 41–56). The N3 bar is "赢 scalar **且赢 naive RVV**." Even the measurement infra currently cannot prove the full claim — a naive/hand-RVV baseline must be added. (Infra to run it already exists.)
2. **(b) a discriminating cost/resource model — MEDIUM.** Replace the max-unroll tiebreak (header 1540–1551) and the inert budget check with a model that actually reasons over the carried facts (vsetvl regions, peak-live vs real VLEN-derived budget, accumulator/reduction layout, unroll) and can rank candidates differently per target.
3. **(a) facts-derived enumeration — MEDIUM/HIGH.** Derive candidate facts (unroll, peak-live, vsetvl regions, LMUL/EMUL space) from compiler-visible capability + body facts instead of the literals at header 733–746; widen beyond the single hardcoded i8mf4… shape so there is a real space to prune.
4. **(c) hardware perf measurement — ALREADY EXISTS** (`rvv_generated_bundle_same_target_measure.py` + committed gate4 runs). Not a gap; it is what *proves* today's no-win.
5. **(e) the WIN itself — HIGHEST effort, gated on real codegen.** Current measured reality is regression (grouped) / no-win (packed-i4). Closing this needs a generated body that is actually faster than scalar (and naive RVV) on hardware — not attribute plumbing. *Where the win is closest:* large-n packed-i4 (n=4096/65536) crosses just above parity (1.015–1.022, i.e. within ~2% — noise-adjacent) and only loses at n=257; that is the nearest-to-a-win path, but it is still no-win by the rule and still has no naive-RVV baseline. The packed-i4 apparatus already records this as `no-win-repair-required-before-performance-claim` with a remediation campaign closed at `no-further-repair` (header 615–701).

---

## Caveats / Not Found

- Task named `scripts/same_target_measure.py`; the real file is `scripts/rvv_generated_bundle_same_target_measure.py`. Same tool.
- `RVVLowPrecisionPerformancePolicy.cpp` (267 KB) was not read in full; its role (admission/no-win policy plumbing) is confirmed via the header constants it consumes and the field names in the measurement record. The verdict does not depend on its internals — selection logic and measured speedups are decisive on their own.
- "Capability/resource facts" today reduce to: a single fixed VLEN-agnostic shape + a fixed 32-register budget. No VLEN/ELEN value is actually read into the candidate space; LMUL is asserted, not searched. So "resource-aware" is not yet true in the spec's sense.
