# Stage2 RVV low-precision production-kernel selected-dispatch campaign

## Goal

Create a macro Trellis owner for the RVV Stage 2 low-precision
production-kernel selected-dispatch campaign. The campaign connects the already
proven low-precision primitive surface, Gearbox/resource facts, target artifact
evidence, and same-target measurement policy to a bounded selected
`tcrv.exec` RVV production-kernel dispatch workflow. Gates 1-4 are complete.
Gate 4 refreshed the final same-target `ssh rvv` correctness/performance
evidence, synchronized the accepted provider-owned evidence contract, and kept
the selected-dispatch policy on the conservative correctness fallback path
because the measured packed-i4 result is still a regression/no-win.

## What I Already Know

- There is no `.trellis/.current-task` at session start, and the worktree is
  clean.
- The previous low-precision contraction primitive-surface campaign is
  archived at
  `.trellis/tasks/archive/2026-06/06-10-rvv-low-precision-contraction-primitive-surface-campaign/`.
- Commit `f07722c3` completed the previous campaign Gate 4 by making the
  low-precision measurement policy consume same-target measurement/resource
  facts for the packed-i4 widening product-reduce-dequantize path.
- The next bottleneck is not another generated-bundle evidence closeout. It is
  a production selected-dispatch workflow that uses the completed primitive,
  resource, artifact, and measurement facts as structural policy input.
- Durable authority remains:
  selected `tcrv.exec` RVV boundary -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned realization and route/provider planning -> provider-built
  `TCRVEmitCLowerableRoute` -> common EmitC materialization -> target artifact
  -> `ssh rvv` evidence when runtime, correctness, or performance is claimed.
- A dispatch or preference decision may mirror route/artifact/evidence facts,
  but it must not derive authority from route ids, q8/q4 labels, artifact
  names, test names, helper names, ABI strings, Common EmitC inference, or
  descriptor residue.

## Requirements

- Keep this as one macro Trellis task until all campaign gates are complete or
  human steering redirects it.
- Complete one coherent milestone slice per round and update this PRD with
  completed and remaining gates.
- Gate 1 must introduce or repair the production selected-dispatch policy
  boundary so it consumes low-precision primitive/resource/measurement facts
  structurally.
- Gate 1 must consume, at minimum for the bounded current production path:
  primitive source signedness/type facts, Gearbox selected resource and
  remediation facts, target artifact evidence tie-back, same-target
  measurement identity/range/outcome, and fallback eligibility.
- Gate 1 must fail closed with targeted diagnostics for stale or missing
  primitive, Gearbox/resource, artifact, measurement, ABI, or policy facts.
- Conservative fallback must remain available when route correctness is
  structurally supported but performance preference is not justified.
- Common EmitC/export must remain neutral mechanics. The selected-dispatch
  policy may consume provider-owned route/export/measurement payloads, but
  Common EmitC must not infer RVV semantics or policy state.
- Runtime, correctness, or performance claims require `ssh rvv` evidence only
  when the changed production path makes such a claim in the current gate.
  Gate 1 may use dry-run or focused unit evidence if it validates the changed
  policy boundary without claiming new runtime/performance behavior.

## Macro Campaign Gates

- [x] Gate 1: selected-dispatch policy boundary consumes low-precision
  primitive/resource/measurement facts structurally and fails closed for stale
  or missing facts.
- [x] Gate 2: selected/pre-realized low-precision production-kernel body reaches
  route/export/artifact validation with fallback semantics preserved.
- [x] Gate 3: same-target measurement evidence updates only the bounded policy
  input and demonstrates preferred-route or conservative-fallback behavior.
- [x] Gate 4: final `ssh rvv` correctness/performance evidence and spec/task
  archive after production gates are satisfied.

## Current Slice: Gate 1 Selected-Dispatch Policy Boundary

- [x] Inspect selected-dispatch, low-precision performance policy,
  Gearbox/resource handoff, route validation, target artifact validation, and
  same-target measurement consumption code to locate the smallest production
  boundary.
- [x] Add or repair a production compiler surface that represents the bounded
  selected low-precision RVV dispatch decision as provider-owned facts rather
  than route/test/artifact names.
- [x] Consume primitive signedness/type, Gearbox resource/remediation,
  artifact-evidence, same-target measurement, and fallback facts in the policy
  boundary.
- [x] Reject missing, stale, disconnected, cross-target, measurement-only, or
  metadata-only facts before a performance-preferred dispatch decision is
  accepted.
- [x] Preserve conservative fallback when structural correctness support is
  present but performance preference is denied.
- [x] Add focused positive tests for structural policy consumption.
- [x] Add focused negative tests for stale or missing primitive/resource/
  measurement/policy facts.
- [x] Run affected RVV plugin, target artifact, route/export, and script checks.
- [x] Run bounded old-authority scans over touched files and added diff lines.
- [x] Keep this macro task active after Gate 1 unless all gates are truly
  complete.

## Gate 1 Acceptance Criteria

- [x] Production source diff is in the selected-dispatch, policy, or
  route-validation seam that affects the bounded low-precision production path.
- [x] The selected-dispatch policy consumes provider-owned low-precision
  primitive signedness/type facts, Gearbox resource/remediation facts,
  target-artifact evidence tie-back, and same-target measurement facts.
- [x] Missing or stale primitive, resource, artifact, measurement, ABI, or
  policy facts fail closed with targeted diagnostics.
- [x] Conservative fallback remains structurally available when performance
  preference is denied.
- [x] No q8/q4/llama label, route id, artifact name, helper name, ABI string,
  test name, descriptor residue, status field, or Common EmitC inference
  becomes dispatch, route, dtype/config, or evidence authority.
- [x] Focused tests exercise accepted structural policy facts and stale/missing
  fact rejection for the changed seam.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes if RVV plugin code
  or tests are touched.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes if target
  artifact validation/export code or tests are touched.
- [x] Relevant `tcrv-opt`, `tcrv-translate`, or script checks pass if
  route/export/generated-bundle paths change.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] One coherent commit records the Gate 1 slice.
- [x] The macro task remains active with Gates 2-4 unchecked and a precise
  continuation point.

## Gate 1 Completion Notes

Gate 1 is complete. The production seam now carries an explicit
`RVVLowPrecisionSelectedDispatchPolicyBoundary` from selected
`tcrv.exec.dispatch` case/fallback collection into the RVV low-precision
performance policy and contraction route-family provider validation. The
boundary requires provider-owned dispatch case origin/policy, conservative
fallback role/origin/policy, selected-dispatch mirrors, accepted primitive/
resource/measurement handoff, and conservative correctness fallback when the
same-target measurement is a no-win/regression. Missing fallback facts and
metadata-only fallback origins fail closed in focused tests.

No new runtime or performance claim is made in this gate; the accepted Gate 4
same-target measurement remains a no-win/regression policy input, so the Gate 1
selected-dispatch decision keeps the conservative fallback path.

## Current Slice: Gate 2 Route/Export/Artifact Validation

- [x] Inspect the low-precision product-reduction-dequant route planning,
  Common EmitC metadata materialization, and RVV target artifact validation
  path to determine whether the selected dispatch case/fallback boundary is
  consumed as part of target acceptance.
- [x] Ensure the selected/pre-realized packed-i4 low-precision
  production-kernel body reaches route/export/artifact validation with
  selected case, conservative fallback, primitive signedness/type, Gearbox
  resource/remediation, artifact-evidence, same-target measurement,
  runtime AVL/VL, ABI/header, and provider mirror facts preserved.
- [x] Fail closed when selected case or fallback mirrors are missing, stale, or
  disconnected from the provider-owned low-precision resource/performance
  policy handoff.
- [x] Keep conservative fallback available for the current no-win/regression
  measurement input; Gate 2 must not claim a new performance-preferred result.
- [x] Add focused positive route/export/artifact evidence for the
  selected/pre-realized packed-i4 body.
- [x] Add focused negative target artifact evidence for stale or missing
  selected dispatch case/fallback facts.
- [x] Run affected RVV plugin, target artifact, `tcrv-opt`, and
  `tcrv-translate` checks.
- [x] Keep this macro task active after Gate 2 unless Gates 3-4 are also
  genuinely complete.

## Gate 2 Acceptance Criteria

- [x] Production source diff, or exact no-source-change proof, demonstrates
  that selected/pre-realized low-precision production-kernel bodies reach
  provider route validation, Common EmitC export mirrors, and target artifact
  validation with selected dispatch case/fallback facts intact.
- [x] Target artifact validation consumes selected dispatch case/fallback
  mirrors together with provider-owned primitive/resource/measurement facts
  for the bounded packed-i4 product-reduction-dequant path.
- [x] Missing or stale selected case, fallback, primitive/resource handoff,
  artifact tie-back, measurement identity, runtime AVL/VL, ABI/header, or
  provider validation facts fail closed with targeted diagnostics.
- [x] Conservative fallback remains structurally available when the selected
  RVV route is correctness-supported but performance preference is denied by
  the accepted no-win/regression measurement.
- [x] No q8/q4/llama label, route id, artifact name, helper name, ABI string,
  test name, descriptor residue, status field, or Common EmitC inference
  becomes dispatch, route, dtype/config, or evidence authority.
- [x] Focused tests exercise accepted structural route/export/artifact facts
  and stale/missing selected dispatch fact rejection for the changed seam.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes if RVV plugin
  code or tests are touched.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes if target
  artifact validation/export code or tests are touched.
- [x] Relevant `tcrv-opt` and `tcrv-translate` checks pass for the selected
  pre-realized packed-i4 artifact fixture.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] One coherent commit records the Gate 2 slice.
- [x] The macro task remains active with Gates 3-4 unchecked and a precise
  continuation point.

## Gate 2 Completion Notes

Gate 2 is complete. The widening product-reduction/dequantization route
validation contract now carries the provider-owned
`RVVLowPrecisionSelectedDispatchPolicyBoundary` from route planning into target
artifact validation. The target artifact consumer validates
`tcrv_rvv.selected_dispatch_case_mirror` and
`tcrv_rvv.selected_dispatch_fallback_mirror` against that provider boundary
alongside the existing primitive signedness/type, Gearbox resource/remediation,
artifact-evidence, same-target measurement, runtime AVL/VL, ABI/header, and
provider support mirrors.

Focused target artifact tests cover the accepted selected-dispatch packed-i4
product-reduction/dequantization candidate, stale case mirror rejection,
missing fallback mirror rejection, and provider boundary missing fallback facts.
The selected pre-realized fixture also has route/export evidence showing the
case/fallback mirrors in the target header artifact and stale mirror rejection
at `tcrv-translate --tcrv-export-target-header-artifact`.

No new runtime, correctness, or performance claim is made in this gate. The
accepted same-target measurement remains a no-win/regression input, so the
selected-dispatch policy keeps the conservative fallback path and denies a new
performance-preferred result.

## Current Slice: Gate 3 Same-Target Measurement Policy Input

- [x] Inspect the low-precision performance policy, selected-dispatch
  boundary, target artifact validation, and same-target measurement script
  evidence-input path to locate the smallest production seam.
- [x] Represent bounded same-target measurement evidence as explicit policy
  input rather than ad hoc measurement/result strings.
- [x] Cross-check that input against provider-owned selected candidate,
  route-family plan, provider support mirror, runtime ABI order, primitive
  chain facts, remediation handoff, target capability mirrors, and selected
  dispatch case/fallback boundary facts before policy acceptance.
- [x] Demonstrate conservative fallback for the accepted no-win/regression
  input and performance-preferred selection only when a matched provider
  measured-win contract also agrees.
- [x] Reject stale, missing, cross-target, disconnected, metadata-only,
  route-id-only, or measurement-only policy claims before they can update
  selected-dispatch preference.
- [x] Keep this macro task active after Gate 3 unless Gate 4 is also genuinely
  complete.

## Gate 3 Acceptance Criteria

- [x] Production source diff, or exact no-source-change proof, demonstrates
  that same-target measurement evidence is consumed through a bounded policy
  input contract, not through artifact names, route ids, q8/q4 labels, test
  names, descriptors, or Common EmitC inference.
- [x] The bounded input matches provider-owned primitive/resource/artifact
  facts before it can drive selected-dispatch preference.
- [x] Missing or stale measurement identity, summary/count evidence,
  same-target/ssh target provenance, provider candidate, route-family plan,
  provider-supported mirror, runtime ABI, primitive chain, remediation handoff,
  target capability mirrors, or selected dispatch case/fallback facts fail
  closed with targeted diagnostics.
- [x] Accepted no-win/regression input keeps route/correctness support but
  selects the conservative fallback path and denies performance claims.
- [x] Matched measured-win input can select a performance-preferred path only
  when provider resource, maturity, remediation, target, and dispatch facts
  already agree.
- [x] Focused C++ and script tests cover the positive and negative cases for
  the changed seam.
- [x] Relevant RVV plugin, target artifact, and same-target measurement script
  checks pass.
- [x] Bounded old-authority scans over touched files and added diff lines show
  no new q8/q4/llama route authority, dtype-prefixed helper route, descriptor
  authority, source-front-door authority, or Common EmitC RVV semantic
  inference.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] One coherent commit records the Gate 3 slice.
- [x] The macro task remains active with Gate 4 unchecked and a precise
  continuation point.

## Gate 3 Completion Notes

Gate 3 is complete. The RVV low-precision performance policy now has an
explicit `RVVLowPrecisionSameTargetMeasurementPolicyInput` contract. Provider
verification builds this bounded input from the accepted same-target
measurement outcome and the provider-owned low-precision resource selection
before selected-dispatch policy acceptance. The input is checked against the
selected candidate, route-family plan, provider-supported mirror, runtime ABI
order, primitive chain facts, remediation handoff, target capability mirrors,
and selected dispatch case/fallback boundary facts before it can update
dispatch preference.

Focused tests prove the accepted no-win/regression input preserves route and
correctness support while selecting the conservative fallback path. Matched
measured-win input can select the performance-preferred path only when provider
resource, maturity, remediation, target, and dispatch facts agree. Missing
measurement identity, cross-target input, metadata-only provider support,
route-family-only claims, disconnected candidates, measurement-only primitive
claims, stale runtime ABI, and stale script maturity input fields fail closed.

The same-target measurement script now emits and self-validates the policy
input fields used by the C++ contract, including authority, correctness count,
same-target/ssh evidence, target profile, provider route/resource/primitive
tie-backs, remediation handoff, target capability mirrors, and performance
preference flags. No new runtime, correctness, or performance claim is made in
this gate; Gate 4 remains the final `ssh rvv` evidence and closeout owner.

## Current Slice: Gate 4 Final ssh rvv Evidence And Closeout

- [x] Ran final same-target `ssh rvv` measurement for the bounded packed-i4
  `widening_product_reduce_dequantize_f32` selected-dispatch path.
- [x] Synchronized the accepted Gate 4 evidence id and best-speedup range
  across provider resource facts, policy constants, script mirrors, target
  header/artifact fixtures, and RVV plugin spec text.
- [x] Preserved route/correctness support while denying performance selection
  and performance-win claims for the accepted regression/no-win input.
- [x] Preserved the measured-win positive path only for a matched provider
  maturity/eligibility/remediation/measurement fixture; measurement-only wins
  still fail closed.
- [x] Verified generated header/artifact mirrors carry selected-dispatch
  case/fallback facts, provider-owned low-precision resource facts, and the
  final Gate 4 evidence id/range.
- [x] Ran focused C++ tests, script self-tests, generated-bundle dry-runs,
  direct header export, bounded old-authority scans, and whitespace checks.

## Gate 4 Acceptance Criteria

- [x] Real `ssh rvv` evidence exists for the final selected-dispatch
  low-precision path:
  `artifacts/tmp/rvv_generated_bundle_same_target_measure/gate4-selected-dispatch-final-ssh/`.
- [x] The measurement reports `summaries=12`, `measurements=60`,
  `correctness=12`, `classification=regression`, and
  `best_speedup_range=0.684318..0.708057`.
- [x] The policy input reports `same_target_measurement=true`,
  `ssh_evidence=true`, `target_profile=ssh rvv`,
  `provider_performance_selection_eligible=false`, and
  `performance_win_claim_allowed=false`.
- [x] The production accepted evidence id is
  `gate4-selected-dispatch-final-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`.
- [x] Provider/header/artifact mirrors use the final evidence id and range,
  while Common EmitC/export remains a neutral carrier of provider-built facts.
- [x] Conservative fallback is selected for the no-win/regression input:
  route support and correctness execution are allowed, performance selection is
  denied, `dispatchPolicyPath=correctness-fallback`, and
  `dispatchPreference=not-performance-preferred`.
- [x] Measured-win selection remains bounded: the policy selects
  `performance-preferred` only when same-target win evidence and provider
  maturity, eligibility, dispatch preference, remediation, target, and
  measurement tie-backs all agree.
- [x] Stale measurement identity, stale speedup range, missing `ssh rvv`
  evidence, stale provider tie-back, stale primitive facts, stale selected
  dispatch mirrors, sibling measurements, and measurement-only win promotion
  fail closed.
- [x] No q8/q4/llama label, route id, artifact name, helper name, ABI string,
  test name, descriptor residue, status field, or Common EmitC inference
  becomes dispatch, route, dtype/config, or evidence authority.
- [x] `.trellis/spec/extension-plugins/rvv-plugin.md` records the final Gate 4
  selected-dispatch policy contract evidence id and range.
- [x] One coherent commit records the Gate 4 closeout slice.

## Gate 4 Completion Notes

Gate 4 is complete. The final same-target `ssh rvv` run for the bounded
packed-i4 `widening_product_reduce_dequantize_f32` path completed successfully
under `gate4-selected-dispatch-final-ssh` with 12 summary records, 60 timing
measurements, 12 correctness guards, classification `regression`, and
best-speedup range `0.684318..0.708057`. The selected-dispatch policy continues
to preserve route/correctness support while denying performance selection and
performance-win claims.

The accepted evidence contract now points at
`gate4-selected-dispatch-final-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`
and the provider/header/artifact mirrors carry the final range
`0.684318..0.708057`. The previous accepted range
`0.689938..0.705891` is retained only as a stale-evidence negative case.

Focused C++ tests prove the conservative fallback outcome for the accepted
no-win/regression input and the positive measured-win outcome only when
provider maturity, selection eligibility, dispatch preference, remediation,
target, and measurement facts all agree. Script self-tests and dry-runs prove
the evidence-input bridge is mirror-only and cannot promote measurement-only
claims into dispatch authority.

Spec closeout is complete for this campaign: the RVV plugin spec records the
Gate 4 packed-i4 dispatch/performance policy surface and the final selected
evidence id/range. No additional durable rule is needed beyond that spec
update. All campaign gates are complete and the task is ready to archive after
the final commit.

## Out Of Scope

- New q8/q4 route ids or llama.cpp-named route authority.
- Another standalone generated-bundle ABI evidence closeout.
- One-op fixture work that does not affect the production selected-dispatch
  policy boundary.
- Broad smoke matrices unrelated to the changed Gate 1 seam.
- Global autotuning databases or readiness state machines.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority or dtype/LMUL clone batches.
- Common EmitC invention of RVV semantics.
- Unrelated MAcc, mask, broadcast, or reduction expansion outside this selected
  low-precision production-kernel dispatch campaign.

## Technical Notes

- Required specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous completed campaign:
  `.trellis/tasks/archive/2026-06/06-10-rvv-low-precision-contraction-primitive-surface-campaign/prd.md`.
- Likely production entry points from the Direction Brief:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/EmitC/RVVLowPrecisionPerformancePolicy.cpp`,
  RVV Gearbox/selected-body realization and route-provider files under
  `lib/Plugin/RVV`, target artifact validation under `lib/Target/RVV`,
  and `scripts/rvv_generated_bundle_abi_e2e.py`.
- The current slice should prefer focused dry-run/unit evidence unless the
  policy change itself makes a new runtime/correctness/performance claim.

## Continuation Point

All macro gates are complete. The task can be archived after final checks and
the coherent Gate 4 closeout commit.
