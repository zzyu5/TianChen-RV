# N1 coverage increment + N3 performance lamp

## Goal

Push TianChen-RV from "通但慢的玩具" (compiles + runs correctly, but no measured speed win) to a
novel, fully-covering compiler by landing the two evidence-bearing novelty claims:

- **N1 覆盖增量** — the same kernel, on *multiple real capability profiles*, is routed to different
  legality / selection / dispatch results by capability queries; and the set of RVV features the
  compiler actually covers via that path is broadened toward "full RVV coverage" (spec N1 evidence
  bar: capability-driven divergence across real profiles).
- **N3 性能灯** — the Gearbox becomes a *real* capability/resource-aware autotuner: it **enumerates +
  prunes a candidate space derived from resource facts** (not fixed unroll/LMUL/single-candidate
  placeholders), and on several kernels it **MEASURABLY beats both scalar and naive RVV on real
  `ssh rvv` hardware**. The "灯" = reproducible hardware speedup evidence. (Spec: "没有胜出的 tuning
  没有论文故事" — generation-selection-tuning.md.)

## What I already know (from this session + spec)

- The codegen is now real MLIR `DialectConversion` (the string machine is gone), every covered RVV
  family is `ssh rvv` correctness-validated, modular backend base exists. So correctness ("通") holds.
- The ssh-rvv e2e harness (`scripts/rvv_generated_bundle_abi_e2e.py`) is **correctness-focused**
  (generate bundle → remote clang → run → numeric check vs reference, PASS/tolerance). Whether it can
  **measure time/cycles** (needed for the perf lamp) is an open question to verify.
- A Gearbox layer exists (RVVGearboxSchedules.cpp + the contraction realization owner + the typed
  `tcrv_rvv.gearbox.*` IR-attr channel). Commit `28d9e4ac` "measure labelled gearbox candidates"
  suggests *some* candidate-measurement scaffolding. Whether it genuinely enumerates+prunes by
  resource facts and whether any candidate has WON on hardware is the N3 gap to measure.
- `low_precision_resource.*` description metadata is N3 tune-measurement evidence (memory:
  low-precision-resource-is-n3-evidence) — the gate4 measurement gate consumes it. KEEP.
- N1 today: ~16 RVV families convert (elementwise, compare-select, all memory, reduce, macc,
  widening product/macc/dot-reduce, dequant/gearbox). The capability-driven multi-profile *divergence*
  evidence (same kernel → different route on different profiles) needs auditing — is it demonstrated?

## Assumptions (to validate via research)

- The Gearbox is currently closer to MVP (fixed/labelled candidates, no measured win) than a real
  resource-aware autotuner — i.e. N3 is the bigger lift.
- The harness measures correctness but not performance; a **timing/cycle measurement seam on `ssh rvv`**
  + scalar & naive-RVV **baselines** must be built for the perf lamp.
- N1 coverage is partial; "full coverage" + the multi-profile divergence demo is incremental but needs
  a concrete target set.

## Open Questions (blocking/preference — keep short)

- Sequencing: N3 性能灯 first (the headline novelty + the "慢→快" story) vs N1 覆盖 first vs interleave?
- Perf-lamp bar: which kernels are the "win" demo set, and what counts as a win (cycles? wall-time?
  vs scalar AND naive RVV, what margin)?
- N1 "full coverage" scope: which RVV features/profiles are the target, and is the headline the
  *divergence across profiles* (the spec's N1 evidence) or raw op-count coverage?

## Requirements (evolving)

- [ ] (N3) Gearbox enumerates a resource-fact-derived candidate space + prunes it (not fixed).
- [ ] (N3) A `ssh rvv` **performance** measurement seam (cycles/time) + scalar & naive-RVV baselines.
- [ ] (N3) On ≥ a few kernels, measured win vs scalar AND naive RVV, reproducible (the 灯).
- [ ] (N1) Same kernel → divergent legality/selection/dispatch across ≥2 real profiles (capability-driven).
- [ ] (N1) Broadened RVV feature coverage toward the agreed target set.

## Acceptance Criteria (evolving)

- [ ] N3: a documented, reproducible `ssh rvv` speedup table (kernel × {scalar, naive-RVV, tuned}) where
      the tuned Gearbox output wins, with the candidate enumeration+pruning being resource-derived.
- [ ] N1: a lit/test demonstrating capability-profile-driven route divergence + the expanded coverage set,
      with real-profile evidence.

## Out of Scope (provisional)

- New high-level tensor/tile IR; non-RVV families for the *perf* lamp (RVV is the hardware family).
- Data-dependent JIT autotuning beyond what the win demo needs.

## Definition of Done

- Build/lit honest-green; hardware claims backed by real `ssh rvv` evidence (I8); no I1–I9 violation;
  metadata stays mirror (I4) and tune decisions realize into typed body before route construction (I5).

## Research References (DONE — the honest current state)

- [`research/gearbox-current-state.md`](research/gearbox-current-state.md) — **Gearbox is an MVP placeholder, not a
  resource-aware autotuner**: enumerates 3 candidates but facts are HARDCODED constants (not derived); the
  budget prune is inert (max 7 ≤ 32, never binds); selection is a max-unroll tiebreak (no cost model). On real
  `ssh rvv`: NO candidate wins — auto-selected grouped-u2 is a **regression ~0.50–0.61× (≈2× slower than scalar
  C)**; packed-i4 no-win (0.90–1.02×). Candidate build/select lives in RVVGearboxSchedule.h:733-1551.
- [`research/perf-measurement-infra.md`](research/perf-measurement-infra.md) — **a real `ssh rvv` wall-time seam
  EXISTS** (gate4 `rvv_generated_bundle_same_target_measure.py`: warmup + best-of-N `clock_gettime` vs scalar-C,
  real ratio + win/regression classification) and has RUN — but recorded a **regression** (tuned RVV ~0.76–0.80×,
  20–24% slower; `performance_win_claim_allowed=false`). Gaps: no `rdcycle` (wall-time only), no naive-RVV
  baseline, and the scalar baseline (`-O2 -march=rv64gcv`) likely **autovectorizes** (so "scalar" is secretly
  clang's RVV → unfair).
- [`research/n1-coverage-gap.md`](research/n1-coverage-gap.md) — ~16 families convert but config space is NARROW
  (int8–64 + **f32 only**, LMUL mostly **m1**). The capability-DIVERGENCE mechanism is real (4 query axes
  sew/lmul/tail/mask) **but has ZERO writers** — `buildRVVTargetCapabilitiesFromProbeFacts` (RVVCapabilityProfile.cpp
  :276-341) sets none of them, so a real *probed* profile cannot reach the gate; divergence is reachable only from
  synthetic test IR, and no single kernel is shown diverging across two profiles. `proposeVariants` returns empty.

## The honest reframing (what "通但慢" actually means)

The hand-emitted RVV is currently ~2× SLOWER than the scalar-C baseline on real hardware — partly because the
"scalar" baseline is itself autovectorized by clang (`-march=rv64gcv`), so we're losing to clang's autovectorizer,
and partly because the emission/tune is naive (inert prune, max-unroll pick, no cost model). The N3 灯 is honestly
OFF. The credible win story is on kernels clang CANNOT autovectorize (the Gearbox low-precision/packed-i4
contraction + dequant) measured against a GENUINE scalar baseline + a naive-RVV baseline.

## Converged plan (phased — pending sequencing decision)

- **P-A (N3 measurement honesty)**: fix the baselines — a genuine non-autovectorized scalar + a naive-RVV emitter →
  a fair 3-way table (scalar / naive-RVV / tuned), optionally `rdcycle`. Honest current-state measurement.
- **P-B (N3 autotuner)**: make Gearbox enumerate a REAL resource-fact-derived candidate space (unroll/LMUL/policy)
  + a resource/cost prune that binds + emit the tuned variants → find a WINNING config on the no-autovec kernels →
  the 灯 (reproducible win table). This is the headline novelty + the hardest/uncertain part.
- **P-C (N1 unlock + coverage)**: wire the probe→capability path to actually carry sew/lmul/tail/mask (the true
  unlock) → a one-kernel-two-profile divergence lit demo (the N1 evidence) → a coverage increment (f64, then a
  capability-gated extension like zvfh/f16 so divergence also carries coverage).

## Decision (ADR-lite)

**Context**: N3 灯 honestly off (RVV ~2× slower than autovectorized-scalar); autotuner is a placeholder; N1
divergence unreachable from real profiles. User delegated the sequencing ("你判断").
**Decision**: P-A → P-B → P-C. P-A first because it is both the foundation (fair baselines are required for ANY
win claim, I8) and the diagnostic (it reveals whether the current tuned emission already beats a *genuine* scalar
+ naive-RVV, i.e. how far the 灯 is). Then P-B (make the autotuner real + lock the win) and P-C (N1 unlock + coverage).
**Consequences**: P-B is uncertain (the win may need real emission/tune improvement, not just measurement). The 灯
demo targets the no-autovectorizable Gearbox low-precision/packed-i4 kernels (where the capability-aware tune
legitimately matters), not simple kernels clang already vectorizes.

## Technical Notes

- Spec bars: variant-pipeline/generation-selection-tuning.md (N3 resource model + win requirement),
  core-invariants I4/I5/I8, validation/experiment-reference.md (evidence口径).
- Key files: lib/Plugin/RVV/RVVGearboxSchedules.cpp, RVVContractionSelectedBodyRealizationOwner.cpp,
  lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyLowPrecisionResource.cpp (the resource selection
  authority), scripts/rvv_generated_bundle_abi_e2e.py + same_target_measure.py (gate4), the capability
  model (lib/Support/CapabilityModel.cpp, the profiles).
