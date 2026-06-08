# RVV production-kernel packed low-precision performance repair campaign

## Goal

Start a macro RVV production-kernel performance repair campaign for the packed
low-precision contraction path. The campaign begins from the measured
packed-i4 same-target no-win/regression evidence and feeds it back into
RVV-owned selected-body/resource/statement/artifact planning, so the compiler
path either carries a more resource-aware packed low-precision structure, fails
closed for stale or unsupported resource facts, or truthfully records that the
current generated RVV artifact has no same-target performance win.

The production path under repair is:

```text
selected/pre-realized packed low-precision product-reduction-dequant body
  -> RVV Gearbox/resource-aware realization facts
  -> RVV route/statement/artifact validation
  -> generated RVV artifact
  -> same-target measurement against scalar-c-reference/product-reduction-dequant-packed-i4-v1
```

## Campaign Gates

- [x] Gate 1: no-win diagnosis tied to production owners, with at least one
  machine-checkable resource/schedule/statement/artifact fact or validation
  seam identified from the measurement evidence.
- [x] Gate 2: production compiler/source change in RVV plugin-local
  Gearbox/resource-aware selected-body realization, route planning, statement
  planning, or artifact validation that addresses diagnosed packed
  low-precision overhead or fails closed for stale resource/performance facts.
- [ ] Gate 3: generated artifact/header/C/correctness evidence showing
  provider-owned packed resource and performance-feedback facts are preserved
  after the production change.
- [ ] Gate 4: same-target measurement against the scalar baseline with honest
  win, no-win, or regression reporting.

## Current Round Slice

This round completes Gate 1 and the first coherent production-surface action
under Gate 2. It maps the previous same-target no-win result to the packed-i4
resource selection boundary, then adds a provider-owned performance feedback
guard for the packed-i4 resource selection. The guard must be consumed by
route/statement/artifact validation as a mirror and fail-closed seam; it must
not be treated as route authority or as a performance win claim.

If this slice cannot safely improve performance, it must still leave a
machine-checkable production improvement: packed-i4 resource facts carry
truthful same-target no-win feedback and reject stale performance feedback
mirrors before artifact acceptance. Later slices can then regenerate artifact
evidence and rerun same-target timing only after this production path changes.

## Repository Findings

- The archived resource-realization campaign completed Gate 4 and measured the
  packed-i4 generated RVV artifact on `ssh rvv` against
  `scalar-c-reference/product-reduction-dequant-packed-i4-v1`.
- The per-op evidence at
  `artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`
  records `packed_i4_resource_metadata_selected = true` and the packed scalar
  reference oracle derived from provider-owned `packed-i4-nibbles` metadata.
- Raw timing stdout records all 12 parsed `SUMMARY` lines below 1.0
  best-speedup, from `0.761006` to `0.807006`, so the current production
  artifact is no-win/regression versus the scalar C packed-i4 baseline.
- The current packed fixture explicitly selects the packed-i4 candidate through
  `tcrv_rvv.low_precision_resource.selected_candidate`; the candidate carries
  packed nibble form, sign-extension intent, unroll 1, two realized `vsetvl`
  regions, peak live vector groups 6, product region 1, and dequant region 2.
- `RVVLowPrecisionContractionResourceSelection` is already consumed by route
  family planning, direct contraction statement planning, and target artifact
  validation. That is the correct production owner seam for the first no-win
  feedback loop.

## Requirements

- Keep the macro task active after this slice unless Gates 1 through 4 are all
  genuinely complete with committed evidence.
- Do not introduce q4/q8/llama.cpp, benchmark-name, route-id, artifact-name,
  descriptor, source-front-door, Common EmitC, or exact intrinsic spelling as
  route authority.
- Map the measured no-win/regression to RVV-owned packed low-precision resource
  facts, not to the scalar baseline implementation or measurement harness.
- Add or harden one production owner under RVV Gearbox/resource realization,
  route planning, statement planning, or target validation.
- If adding performance feedback facts, they must be provider-owned mirrors
  consumed by statement and target validation, must truthfully encode no-win
  repair-required state, and must fail closed on stale/missing packed-i4
  mirrors.
- Preserve existing correctness/artifact evidence routes; do not claim a
  performance win without new same-target measurement after the production
  change.

## Acceptance Criteria

- [x] PRD and task context identify this as a macro campaign and this round as
  Gate 1 plus first Gate 2 action.
- [x] The no-win evidence is summarized with the scalar baseline identity,
  speedup range, selected generated artifact, and packed provider metadata
  source.
- [x] A production source change adds or hardens a machine-checkable
  RVV-owned packed low-precision performance/resource guard.
- [x] Direct contraction route/statement planning or target validation consumes
  the new guard and rejects stale packed-i4 performance feedback mirrors.
- [x] Focused tests cover positive packed-i4 metadata and one stale/missing
  feedback failure path.
- [x] Relevant C++ test binaries for touched production owners pass when
  available: `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test`.
- [x] If emitted facts change, run a focused generated artifact dry-run or
  direct pipeline check for the packed-i4 fixture; do not run same-target timing
  unless claiming performance impact.
- [x] Run `git diff --check`, `git diff --cached --check`, and a bounded
  old-authority scan over touched files/added diff lines.
- [x] Update the task PRD/journal with completed and remaining campaign gates.
- [x] Create one coherent commit for the slice while keeping the macro task
  active.

## Definition of Done

- Production compiler/source change is implemented in C++/MLIR/LLVM stack.
- Tests or focused checks exercise the changed owner and fail-closed path.
- Trellis task files and journal record the campaign state and continuation
  point.
- Worktree is clean after commit.

## Out of Scope

- No q4/q8-named route authority.
- No llama.cpp wrapper or benchmark-name owner.
- No adjacent generated-bundle evidence closeout without production owner
  change.
- No new standalone MAcc, mask, broadcast, dtype/LMUL clone, or source-front
  door positive route.
- No broad autotuning database or dashboard/report-only milestone.
- No Common EmitC invention of RVV semantics.
- No scalar baseline optimization masquerading as RVV compiler progress.
- No performance-win claim without same-target measurement after the production
  change.

## Technical Approach

The current slice will add a small provider-owned performance feedback contract
for the packed-i4 low-precision resource selection. The initial contract value
records the prior same-target result as no-win/regression against
`scalar-c-reference/product-reduction-dequant-packed-i4-v1` with the measured
speedup range. Route planning and statement planning compare the feedback field
inside `RVVLowPrecisionContractionResourceSelection`; target artifact
validation compares the candidate mirror exactly before accepting the header or
bundle.

This is intentionally a fail-closed/truthful-recording production action, not a
performance fix. It makes the feedback loop machine-checkable so the next
slice can regenerate artifact evidence and then decide whether to improve the
packed execution structure or continue recording no-win honestly.

## Current Round Result

Completed:

- Gate 1: the measured same-target no-win/regression is tied to the packed-i4
  resource-selection boundary, not to a benchmark name or artifact label. The
  production seam is `RVVLowPrecisionContractionResourceSelection` flowing
  through Gearbox attrs, selected-body realization, route-family planning,
  statement planning, route metadata, target artifact validation, and generated
  bundle metadata mirrors.
- Gate 2: the production path now carries provider-owned packed-i4 performance
  feedback facts:
  `same-target-packed-i4-no-win.v1`,
  `scalar-c-reference/product-reduction-dequant-packed-i4-v1`,
  `0.761006..0.807006`, and
  `no-win-repair-required-before-performance-claim`. Stale packed-i4 feedback
  fails closed before target artifact acceptance.

Remaining:

- Gate 3: regenerate and commit full generated artifact/header/C/correctness
  evidence for the changed production surface. This round ran a focused local
  dry-run to prove field preservation, but did not claim full remote
  correctness evidence.
- Gate 4: rerun same-target packed-i4 timing against
  `scalar-c-reference/product-reduction-dequant-packed-i4-v1` only after a
  later production compiler path change attempts an actual performance repair.

Continuation point:

Start from the packed-i4 resource/performance feedback fields now preserved in
selected-body realization, provider planning, statement planning, target
artifact validation, and generated-bundle dry-run metadata. The next slice
should either improve the packed execution structure under RVV-owned
Gearbox/resource/statement planning or keep the fail-closed no-win feedback
truthful while collecting Gate 3 evidence.

## Technical Notes

- Direction source: Hermes Direction Brief in the user prompt for this session.
- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Archived campaign read:
  `.trellis/tasks/archive/2026-06/06-09-rvv-production-kernel-packed-resource-realization-campaign/prd.md`.
- Evidence read:
  `artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`
  and `remote_measure_run_stdout.txt`.
- First code owners inspected:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, and
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir`.

## Open Questions

No blocking user question is needed. The Direction Brief provides the owner,
gates, boundaries, non-goals, and expected first slice.
