# Stage2 RVV packed-i4 performance-remediation campaign

## Goal

Create a macro production-capability owner for the packed-i4
`widening_product_reduce_dequantize_f32` low-precision contraction path. The
campaign turns the final Gate 4 same-target regression evidence from the
previous selected-dispatch campaign into RVV-owned, resource-aware Gearbox
selected-body realization and scheduling contracts that production compiler code
can consume, validate, and fail closed before any future performance-preferred
dispatch claim.

## What I Already Know

- Session start had no `.trellis/.current-task`; a new macro task is required
  from the Hermes Direction Brief before source edits.
- The previous selected-dispatch campaign is archived at
  `.trellis/tasks/archive/2026-06/06-10-06-10-rvv-low-precision-production-kernel-selected-dispatch-campaign/`.
- Commit `a2180eb9` closed that campaign with final real `ssh rvv` same-target
  evidence for packed-i4:
  `classification=regression`,
  `best_speedup_range=0.684318..0.708057`,
  and policy outcome `correctness-fallback`.
- Existing production code already carries many provider-owned packed-i4 facts:
  primitive/resource fields, performance feedback, remediation plan, maturity
  fields, selected-dispatch policy input, target artifact mirrors, and a
  provider statement plan that emits low/high nibble sign-extension, two
  widening products, pair-sum, and a single `vwredsum`.
- The current bottleneck is not another generated-bundle evidence seam. The
  next production improvement is to make the packed-i4 remediation schedule and
  resource plan explicitly actionable and fail-closed in RVV-owned compiler
  code.

## Requirements

- Keep this as one macro Trellis task until all campaign gates are complete or
  human steering redirects it.
- Complete one coherent milestone slice per worker round and update this PRD
  with completed and remaining gates.
- Preserve the previous `ssh rvv` regression evidence as a policy input. Do not
  claim a performance win or performance-preferred dispatch from the accepted
  no-win/regression input.
- Production-source progress is required for each implemented gate unless a
  precise production blocker is proven first.
- RVV plugin code owns packed-i4 resource/schedule facts, selected-body
  realization, route planning, and fail-closed diagnostics. Common EmitC/export
  may carry provider-built facts only.
- The packed-i4 path must continue to start from typed `tcrv_rvv` body/resource
  facts, not q4/q8 labels, llama.cpp names, route ids, artifact names,
  descriptor residue, ABI strings, or test names.
- Missing, stale, disconnected, metadata-only, or measurement-only resource,
  schedule, remediation, artifact, ABI, or selected-dispatch facts must fail
  closed before route support, target artifact acceptance, or performance
  preference.

## Macro Campaign Gates

- [x] Gate 1: establish production-owned packed-i4 resource/schedule facts and
  a baseline/remediation contract tied to the final Gate 4 regression evidence.
- [x] Gate 2: implement one coherent resource-aware selected-body/Gearbox
  schedule improvement or fail-closed capability guard in production code.
- [ ] Gate 3: prove generated artifact correctness and same-target measurement
  for the changed packed-i4 path when runtime/correctness/performance behavior
  is claimed.
- [ ] Gate 4: consume measurement in selected-dispatch policy, choosing
  `performance-preferred` only for a real same-target win and
  `correctness-fallback` otherwise.

## Current Slice: Gate 1 Plus First Gate 2 Production Guard

- [x] Inspect the previous selected-dispatch campaign PRD and final Gate 4
  evidence outcome.
- [x] Inspect relevant specs for RVV plugin authority, Common EmitC neutrality,
  variant tuning/resource-aware realization, and testing/evidence requirements.
- [x] Inspect current packed-i4 Gearbox/resource/policy/provider/target code to
  find the smallest production seam.
- [x] Add or repair an RVV-owned packed-i4 remediation schedule planning
  contract that is more actionable than the existing broad remediation text.
- [x] Tie the contract to selected resource facts: packed-i4 operand form,
  low/high signed nibble unpack, pair-sum product accumulation, single
  `vwredsum`, region/VL placement, vector group budget, and final Gate 4
  no-win/remediation evidence.
- [x] Make provider/target validation fail closed for missing or stale
  schedule-remediation facts before packed-i4 route/artifact acceptance.
- [x] Preserve correctness support and conservative fallback; do not change the
  accepted no-win policy result.
- [x] Add focused C++ tests for accepted schedule-remediation facts and at least
  one stale/missing rejection.
- [x] Run focused RVV plugin and target artifact tests; run relevant export/dry
  checks if route/header/script surfaces change.
- [x] Update this PRD and journal with completed slice and precise continuation
  point.
- [x] Commit one coherent slice while keeping this macro task active.

## Acceptance Criteria For This Round

- [x] The diff includes production compiler code in the RVV Gearbox/resource
  schedule, selected-body realization, provider validation, low-precision
  performance policy, or target artifact boundary.
- [x] The new or repaired contract is provider/RVV-owned and consumed by
  production validation, not only documented in tests or reports.
- [x] The accepted packed-i4 route still validates the known 13-step low/high
  nibble unpack, pair-sum, single-`vwredsum` statement path.
- [x] Missing or stale packed-i4 remediation schedule/resource facts fail closed
  with targeted diagnostics.
- [x] The no-win/regression evidence keeps `correctness-fallback` and
  `not-performance-preferred`; no new performance result is claimed.
- [x] No Common EmitC code infers RVV semantics, dtype/config, resource schedule,
  or dispatch preference.
- [x] No q4/q8/llama label, artifact name, route id, helper name, descriptor,
  status field, or test name becomes authority.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes if RVV plugin code
  or tests are touched.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes if target
  artifact validation/export code or tests are touched.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] A coherent commit records the slice and `.trellis/.current-task` remains
  active unless all macro gates are genuinely complete.

## Completed Slice: 2026-06-10

- Added provider-owned packed-i4 remediation schedule facts:
  `remediation_schedule_contract`, `remediation_unpack_plan`,
  `remediation_product_plan`, `remediation_reduction_plan`, and
  `remediation_vl_plan`.
- Threaded those facts through the packed-i4 resource candidate, selected-body
  Gearbox handoff, RVV dialect verifier, route-family provider validation,
  statement-plan consistency checks, route metadata, target artifact candidate
  mirror validation, target header support bundle, and dry-run evidence scripts.
- Added stale product-plan rejection at the Gearbox handoff and target artifact
  mirror boundaries. The accepted no-win/regression evidence remains the policy
  input, so packed-i4 stays correctness-supported but not
  performance-preferred.
- Repaired C++ packed-i4 fixtures to carry the new schedule facts instead of
  relaxing validation.
- The macro task remains active because Gate 3 real generated artifact
  correctness/same-target measurement and Gate 4 measured-policy consumption are
  not complete for a changed runtime/performance path.

## Out Of Scope

- New q8/q4/llama-named route ids, artifact authority, or benchmark authority.
- High-level Linalg/StableHLO frontend work.
- Per-Linalg route authority or broad dtype/LMUL clone batches.
- Source-front-door positive routes or descriptor-driven computation.
- Common EmitC invention of RVV semantics.
- Standalone generated-bundle evidence closeout without a production compiler
  change.
- A performance-win claim without new real same-target evidence and provider
  policy update.
- Unrelated MAcc, mask, segment, broadcast, or general reduction expansion
  outside the packed-i4 remediation campaign.

## Technical Notes

- Required specs read for this task:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/index.md`.
- Relevant previous campaign:
  `.trellis/tasks/archive/2026-06/06-10-06-10-rvv-low-precision-production-kernel-selected-dispatch-campaign/prd.md`.
- Current production seams inspected:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h`,
  `lib/Plugin/RVV/EmitC/RVVLowPrecisionPerformancePolicy.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`.

## Continuation Point

Continue from Gate 3 only after the next Gate 2 production change modifies the
real generated runtime path or scheduling behavior. The next likely milestone is
to make Gearbox selected-body realization consume the schedule contract into a
resource-aware scheduling decision beyond the current static packed-i4
candidate, then run generated artifact correctness and real same-target `ssh rvv`
measurement before any performance-preferred policy update is considered.
