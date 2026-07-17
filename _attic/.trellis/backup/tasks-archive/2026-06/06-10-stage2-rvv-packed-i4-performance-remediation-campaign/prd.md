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
- [x] Gate 2a: implement the first production fail-closed contract guard for
  provider-owned packed-i4 resource/remediation schedule facts.
- [x] Gate 2b: make the packed-i4 remediation schedule contract actionable in
  Gearbox selected-body realization by choosing or rejecting a resource-aware
  packed-i4 schedule from the provider-owned remediation facts before route
  planning.
- [x] Gate 3: prove generated artifact correctness and same-target measurement
  for the changed packed-i4 path when runtime/correctness/performance behavior
  is claimed.
- [x] Gate 4: consume measurement in selected-dispatch policy, choosing
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

## Current Slice: Gate 2b Resource-Aware Gearbox Scheduling Decision

- [x] Repair the resource candidate model so packed-i4 remediation schedule
  facts are not only mirrored metadata but the accepted input to an
  RVV-owned scheduling decision.
- [x] Define the accepted resource-aware packed-i4 decision from the five
  remediation schedule facts: low/high signed nibble unpack, two widening
  products plus pair-sum, single `vwredsum`, two-region runtime-AVL VL
  placement, and 7-of-32 vector group budget.
- [x] Fail closed when a packed-i4 resource candidate is selected without the
  accepted schedule decision, with stale remediation facts, or with missing
  decision facts before route planning/provider acceptance.
- [x] Thread the selected scheduling decision through Gearbox selected-body
  realization, route-family provider planning, statement planning, and target
  artifact validation as provider-owned facts.
- [x] Preserve correctness support and conservative `correctness-fallback`;
  do not claim a performance win or run Gate 3/4 unless this slice produces
  corresponding real same-target evidence.
- [x] Add focused C++ coverage for accepted packed-i4 schedule decision facts
  and at least one missing/stale/unsupported decision rejection.
- [x] Run the focused RVV plugin and target artifact tests plus relevant build
  and diff checks, then commit this Gate 2b slice while keeping the macro task
  active.

## Current Slice: Gate 3 Generated Artifact Correctness And Same-Target Measurement

- [x] Keep the active macro task as the owner and continue from the Gate 2b
  schedule-decision path instead of creating a neighboring generated-bundle
  evidence task.
- [x] Harden generated-bundle evidence so the packed-i4 remediation evidence
  block explicitly carries the remediation schedule facts and the Gate 2b
  `schedule_decision_contract`, `schedule_decision`, and
  `schedule_decision_reason`.
- [x] Harden same-target measurement evidence so the measurement record carries
  both `schedule_decision` and `provider_schedule_decision` fields sourced from
  validated provider/target artifact metadata.
- [x] Prove generated artifact correctness for the packed-i4
  `widening_product_reduce_dequantize_f32` path with the Gate 2b schedule
  decision visible in generated evidence, bundle index, and header mirrors.
- [x] Run real same-target `ssh rvv` measurement for the updated packed-i4 path
  and record parsed timing, correctness, target profile, and schedule-decision
  tie-back evidence.
- [x] Preserve conservative policy state after measurement: the new result is a
  regression, so `performance_win_claim_allowed = false`,
  `provider_performance_selection_eligible = false`, and
  `dispatch_preference = not-performance-preferred`.
- [x] Leave Gate 4 open for selected-dispatch policy consumption of the new
  measurement evidence.

## Current Slice: Gate 4 Measured Selected-Dispatch Policy Consumption

- [x] Keep this macro task as the owner and consume the Gate 3
  `gate3-packed-i4-schedule-decision-ssh` same-target measurement in the
  selected-dispatch policy boundary.
- [x] Treat the Gate 2b provider-owned schedule decision facts
  `schedule_decision_contract`, `schedule_decision`, and
  `schedule_decision_reason` as required policy-input tie-backs, not as
  mirror-only report metadata.
- [x] Update the accepted packed-i4 measurement identity and speedup range to
  the current Gate 3 evidence:
  `gate3-packed-i4-schedule-decision-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`
  and `0.689815..0.705331`.
- [x] Preserve the current conservative outcome:
  `correctness-fallback`, `not-performance-preferred`,
  `performance_selection_allowed = false`, and
  `performance_win_claim_allowed = false`.
- [x] Keep measured-win promotion bounded so `performance-preferred` is
  accepted only when same-target win evidence and provider maturity,
  eligibility, dispatch preference, remediation, target, measurement, and
  schedule-decision facts all match.
- [x] Add focused C++ rejection coverage for stale or missing measurement,
  target, provider tie-back, primitive, and schedule-decision facts.
- [x] Update generated-bundle / same-target script mirrors and dry-run
  expectations only as mirrors of the provider-owned policy facts.
- [x] Run the focused RVV plugin, target artifact, script, dry-run, diff, and
  old-authority checks before closing Gate 4.

## Gate 4 Acceptance Criteria

- [x] Production selected-dispatch policy consumes provider-owned Gate 2b
  schedule-decision facts and Gate 3 same-target measurement facts in one
  strict handoff.
- [x] Missing, stale, disconnected, metadata-only, or measurement-only
  schedule-decision / measurement / provider facts fail strict verification;
  the safe resolver preserves correctness fallback when the route remains
  legal.
- [x] The accepted Gate 3 regression maps to correctness fallback and
  `not-performance-preferred`; no performance-preferred dispatch or win claim
  is made from the regression evidence.
- [x] A measured-win fixture can select `performance-preferred` only when the
  provider contract, selected-dispatch boundary, schedule decision, target
  profile, measurement counts, and win-claim allowance all agree.
- [x] Common EmitC/export remains neutral and carries provider-built facts only.
- [x] No q4/q8/llama label, route id, artifact name, ABI string, helper name,
  descriptor residue, status field, source-front-door marker, or test name
  becomes dispatch, route, dtype/config, or evidence authority.
- [x] Focused C++ tests, script self-tests, Gate 4 same-target dry-run, diff
  whitespace checks, and bounded old-authority scans pass.
- [x] If all Gate 4 criteria are complete, update spec/journal, archive the
  macro task, and commit one coherent Gate 4 closeout slice.

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

## Completed Slice: 2026-06-10 Gate 2b

- Added a provider-owned packed-i4 schedule decision contract:
  `schedule_decision_contract`, `schedule_decision`, and
  `schedule_decision_reason`.
- The accepted decision is derived from the five remediation schedule facts:
  low/high signed nibble unpack, two widening products plus pair-sum, single
  `vwredsum`, two-region runtime-AVL VL placement, and the 7-of-32 vector group
  budget.
- Threaded the selected decision through the RVV resource candidate,
  Gearbox selected-body realization, cross-region handoff, route-family plan
  selection, statement/route planning consistency checks, target artifact
  validation, target header bundle, and dry-run evidence scripts.
- Added fail-closed checks for stale or missing schedule-decision facts at the
  provider and target artifact boundaries. The accepted no-win/regression input
  still keeps the path correctness-supported but not performance-preferred.
- No real `ssh rvv` measurement was run in this slice because Gate 2b changes
  schedule decision authority and validation visibility, while Gate 3 remains
  the next owner for generated artifact correctness and real same-target
  measurement.

## Completed Slice: 2026-06-10 Gate 3

- Generated-bundle ABI evidence now exposes the packed-i4 remediation schedule
  fields and Gate 2b schedule-decision fields inside
  `packed_i4_resource_remediation_evidence`, separating the concise evidence
  summary from the lower-level target artifact mirror keys.
- Same-target measurement evidence now records
  `measurement_schedule_decision_evidence` and
  `same_target_schedule_decision_evidence` with the provider-owned schedule
  decision:
  `select-packed-i4-pair-sum-single-reduce-u1-two-region-budget-7of32.v1`.
- Real same-target `ssh rvv` measurement completed at:
  `artifacts/tmp/rvv_generated_bundle_same_target_measure/gate3-packed-i4-schedule-decision-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`.
- Measurement result: `summaries=12`, `measurements=60`, `correctness=12`,
  `classification=regression`, `best_speedup_range=0.689815..0.705331`,
  `provider_performance_selection_eligible=false`, and
  `performance_win_claim_allowed=false`.
- The generated artifact and same-target records both carry
  `schedule_decision` / `provider_schedule_decision`; stale or missing schedule
  facts remain fail-closed through the provider and target validation paths from
  Gate 2b.
- No selected-dispatch policy promotion was made in Gate 3. Gate 4 must consume
  this measurement explicitly and preserve correctness fallback unless future
  same-target evidence and provider facts prove a real win.

## Completed Slice: 2026-06-10 Gate 4

- Updated the accepted packed-i4 policy evidence id and best-speedup range to
  the Gate 3 schedule-decision measurement:
  `gate3-packed-i4-schedule-decision-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`
  and `0.689815..0.705331`.
- Extended the production RVV low-precision dispatch/performance policy
  handoff so both `RVVLowPrecisionPerformanceMeasurementOutcome` and
  `RVVLowPrecisionSameTargetMeasurementPolicyInput` carry
  `providerScheduleDecisionContract`, `providerScheduleDecision`, and
  `providerScheduleDecisionReason` and verify those fields against the
  provider-owned Gate 2b resource selection.
- The accepted regression/no-win input selects `correctness-fallback`, keeps
  `dispatchPreference = not-performance-preferred`, preserves route/correctness
  support, and denies performance selection and performance-win claims.
- Measured-win promotion remains bounded by provider maturity, eligibility,
  remediation, dispatch preference, target, measurement, and schedule-decision
  tie-backs; measurement-only win promotion still fails strict verification.
- Updated generated-bundle and same-target dry-run mirrors plus RVV plugin spec
  text to reflect the Gate 3 evidence id/range and the schedule-decision policy
  input requirement.
- No new real `ssh rvv` run was made in this slice because the policy consumes
  the existing Gate 3 real measurement and does not claim a new runtime or
  performance result.
- Focused checks passed: rebuild of `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`; both C++ test binaries; script
  `py_compile`; both script self-tests; same-target dry-run; generated-bundle
  ABI dry-run; direct header export; bounded added-line old-authority scan;
  `git diff --check`; and `git diff --cached --check`.

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

All macro gates are complete. The task can be archived after final status
checks and the coherent Gate 4 closeout commit.
