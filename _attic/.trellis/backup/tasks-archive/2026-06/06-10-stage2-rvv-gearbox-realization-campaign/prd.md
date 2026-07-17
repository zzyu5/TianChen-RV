# Stage2 RVV Gearbox Resource-Aware Selected-Body Realization Campaign

## Goal

Create one active macro owner for RVV plugin-local resource-aware
selected-body realization for low-precision production kernels. The campaign
makes source-backed production pressure/profile facts, primitive/resource
facts, target capability facts, runtime ABI facts, and selected-dispatch policy
results affect RVV selected-body realization and later Gearbox schedule
admission.

Gates 1-4 are complete. The current round completed Gate 4 closeout: selected
dispatch policy, RVV plugin-local realization admission, target artifact
evidence identity, and same-target measurement provenance now compose into one
fail-closed production-kernel capability workflow for the packed-i4
dequant-clamp path. Downstream proof fails closed when admission proof,
dispatch policy, artifact evidence identity, measurement provenance, provider
support, or performance mirrors are missing, stale, mismatched, metadata-only,
sibling-route-derived, or not tied to source-backed same-target records. The
boundary is not route authority, artifact-name authority, helper-name
authority, or Common EmitC semantic inference.

## What I Already Know

- Commit `4ab050fa rvv: close production capability campaign` archived the
  previous Stage 2 RVV production-kernel capability campaign.
- The archived campaign already added
  `RVVLowPrecisionProductionPressureProfile` and source-backed
  same-target measurement / selected-dispatch policy validation.
- That pressure-profile boundary currently lives in the performance-policy /
  measurement-record path. The next bottleneck is selected-body realization
  admission, not another generated artifact or ssh evidence closeout.
- Specs require the authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body ->
  RVV plugin-owned realization/provider -> common EmitC materialization.
- Common EmitC/export may carry provider payloads and mirrors only. It must not
  infer RVV dtype, SEW/LMUL, schedule, runtime ABI semantics, primitive facts,
  measurement identity, or dispatch policy.

## Requirements

- Keep this Trellis task active across rounds until all campaign gates are
  complete or human steering redirects the campaign.
- Complete one coherent slice per round and update this PRD with completed and
  remaining gates.
- Gate 3 must add production source changes in the realized-body,
  route/provider, artifact, or measurement proof consumption path, not only
  tests or report metadata.
- Admission must consume the existing source-backed production pressure profile
  plus provider-owned primitive/resource facts, target capability mirrors,
  runtime ABI facts, and selected-dispatch policy results.
- Realized route/artifact/measurement proof must consume explicit admitted
  schedule facts from Gate 2 and source-backed same-target record identity,
  with targeted diagnostics when required provenance is missing, stale,
  mismatched, sibling-route, label-only, metadata-only, or otherwise untrusted.
- Admission must not change computation semantics. It only gates whether the
  selected-body realization owner may consume the resource/profile/schedule
  facts.
- The selected-body realization owner must continue to consume the admission
  decision and schedule choice at the low-precision contraction boundary, and
  the RVV provider/target/measurement consumers must prove those admitted facts
  agree with the selected resource schedule and source-backed same-target
  record.
- Route/artifact/report metadata remain mirrors. The admission API must not use
  artifact names, route ids, helper names, q8/q4 labels, or llama.cpp labels as
  authority.

## Macro Campaign Gates

- [x] Gate 1: pressure-profile admission boundary is modeled and consumed by
  RVV selected-body realization with fail-closed provenance checks.
- [x] Gate 2: Gearbox/resource-aware schedule choice is consumed by the
  realization owner for at least one low-precision contraction class without
  changing compute semantics.
- [x] Gate 3: realized body plus route/artifact/measurement path proves the
  selected schedule against source-backed same-target records.
- [x] Gate 4: selected-dispatch policy, realization admission, artifact
  evidence, and measurement provenance compose into one closeout test; archive
  only after this gate.

## Current Round Slice: Gate 4

Implement a focused production closeout seam:

- Feed the source-backed dequant-clamp `same_target_measurement_record` parsed
  from the generated artifact evidence JSON through the selected-dispatch
  record overload, not only through a helper-built representative record.
- Compose the parsed measurement record with the provider-selected packed-i4
  resource facts, admitted Gearbox schedule proof, selected dispatch boundary,
  target artifact identity, production pressure profile, and selected-body
  realization admission decision.
- Preserve existing computation, dtype, parameter roles, selected variant
  origin, dispatch/fallback behavior, and runtime AVL/VL values.
- Fail closed when schedule proof, correctness allowance, artifact identity,
  provider-supported facts, or dispatch/performance mirrors are stale,
  metadata-only, or sibling-route-derived.
- Keep Common EmitC/export neutral; route/artifact/report metadata remain
  mirrors only.

Acceptance criteria:

- [x] Production C++ policy/admission APIs compose parsed dequant-clamp
  evidence, selected dispatch, pressure profile, and selected-body realization
  admission in one Gate 4 closeout path.
- [x] Focused C++ tests prove the parsed dequant-clamp artifact evidence
  selects correctness fallback, denies performance preference, and admits
  selected-body realization through the source-backed pressure profile.
- [x] Focused C++ tests prove stale schedule proof, correctness-disabled
  records, stale artifact identity, and metadata-only provider support fail
  closed before Gate 4 closeout acceptance.
- [x] Existing Gate 1-3 pressure-profile, selected-dispatch policy, target
  artifact, and generated evidence tests still pass.
- [x] Bounded check confirms Common EmitC/export remains neutral and metadata
  fields are mirrors only.
- [x] Relevant focused checks pass.
- [x] All gates are satisfied; the macro task is ready to archive after this
  slice is committed.

## Completed Slice: Gate 1

Implement a focused production admission seam:

- Add an RVV low-precision realization admission data model/API that consumes
  `RVVLowPrecisionProductionPressureProfile`.
- Add validation that the admission profile matches the selected resource
  candidate, runtime ABI order, primitive facts, target capability mirrors,
  selected-dispatch fallback/case facts, and policy outcome.
- Wire the selected-body contraction realization owner to consume admission
  before materializing resource-aware low-precision realization attributes.
- Preserve existing realization behavior when a trusted source-backed profile
  admits realization.
- Fail closed with targeted diagnostics for:
  missing profile input, missing selected resource, stale runtime ABI,
  stale schedule decision, stale primitive/profile fields, sibling-route
  measurement/profile mismatch, metadata-only provider support, label-only
  q8/q4 pressure markers, and selected-dispatch mismatch.

Acceptance criteria:

- [x] Production C++ API models realization admission with explicit
  realize/defer/deny state.
- [x] Low-precision contraction selected-body realization consumes a successful
  admission before emitting resource-aware realization facts.
- [x] Focused C++ tests prove a source-backed production pressure profile
  admits packed-i4 low-precision contraction realization.
- [x] Focused C++ tests prove missing/stale/mismatched/untrusted admission
  inputs fail closed with targeted diagnostics.
- [x] Existing pressure-profile and selected-dispatch policy tests still pass.
- [x] Bounded check confirms common EmitC/export remains neutral and metadata
  fields are mirrors only.
- [x] Relevant focused checks pass.
- [x] Task remains active with Gate 1 marked complete and Gates 2-4 remaining.

## Non-Goals

- No adjacent generated-bundle or `ssh rvv` evidence closeout as this round's
  main owner.
- No one-op route-family seam, high-level Linalg/Vector/StableHLO frontend,
  global autotuning database, dashboard, source-front-door positive route,
  q8/q4 named route authority, llama.cpp wrapper authority, dtype/LMUL clone
  batch, or Common EmitC RVV semantic inference.
- No campaign archive before all four gates are complete.

## Technical Notes

- Read `.trellis/spec/index.md`.
- Read `.trellis/spec/extension-plugins/rvv-plugin.md`.
- Read `.trellis/spec/lowering-runtime/emitc-route.md`.
- Read archived campaign PRD:
  `.trellis/tasks/archive/2026-06/06-10-stage2-rvv-production-kernel-capability-campaign/prd.md`.
- Relevant production files inspected:
  `include/TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h`,
  `lib/Plugin/RVV/EmitC/RVVLowPrecisionPerformancePolicy.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  and focused low-precision route/provider test regions in
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Verification Plan

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- `build/bin/tianchenrv-target-artifact-export-test`
- Focused packed-i4 target/artifact lit check when lit is available.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-gearbox-realization-campaign`
- `git diff --check`
- `git diff --cached --check`
- Bounded added-diff scan for legacy RVV route-authority markers.

## Verification Results

- Passed `cmake --build build --target tianchenrv-rvv-extension-plugin-test`.
- Passed `build/bin/tianchenrv-rvv-extension-plugin-test`.
- Passed `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`.
- Passed `build/bin/tianchenrv-target-artifact-export-test`.
- Passed focused packed-i4 target/artifact lit check:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir`
  from `build/test`.
- Passed `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-gearbox-realization-campaign`.
- Passed `git diff --check`.
- Passed `git diff --cached --check`.
- Passed bounded added-diff scan over source/test changes for legacy RVV
  route-authority markers; no matches.
- Confirmed changed paths are limited to this task PRD and
  `test/Plugin/RVVExtensionPluginTest.cpp`; Common EmitC/Conversion remained
  untouched.

## Spec Update Decision

[NO SPEC UPDATE] Gate 4 implements the existing
`Packed-I4 Dispatch/Performance Policy Consumption` and same-target evidence
contracts already recorded in `.trellis/spec/extension-plugins/rvv-plugin.md`
and `.trellis/spec/testing/mlir-testing-contract.md`. This slice adds focused
closeout coverage and PRD status repair only; it does not introduce a new
signature, payload field, validation matrix, or long-lived convention.

## Completed Slices

- Gate 1 pressure-profile admission boundary:
  `RVVLowPrecisionSelectedBodyRealizationAdmission` now models
  realize/defer/deny admission state for production low-precision realization.
  Admission consumes `RVVLowPrecisionProductionPressureProfile` plus
  provider-owned resource, primitive, runtime ABI, target mirror, and
  selected-dispatch facts. The contraction selected-body realization owner has
  a pressure-profile overload that admits a source-backed profile before
  materializing low-precision resource-aware realization mirrors. Focused C++
  tests cover successful packed-i4 admission, owner consumption, and
  fail-closed missing/stale/metadata-only/label-only/selected-dispatch
  mismatches.
- Gate 2 schedule-choice consumption:
  `RVVLowPrecisionSelectedBodyRealizationAdmission` now carries the trusted
  packed-i4 schedule decision contract, decision, and reason after successful
  source-backed pressure-profile admission. The contraction selected-body
  realization owner materializes those admitted schedule facts as explicit
  realization mirrors on realized `with_vl` operations and the Gearbox
  cross-region handoff. Focused C++ tests cover successful admitted schedule
  consumption plus missing/stale schedule/admission fail-closed diagnostics.

## Completed Slice: Gate 3

Implemented the production route/artifact/measurement proof-consumption seam:

- Extended `RVVLowPrecisionContractionResourceSelection` and the packed-i4
  same-target record/policy/profile payloads with explicit provider-owned
  realization-admission proof fields.
- Imported Gate 2 admission mirrors from realized `with_vl` / Gearbox handoff
  facts into route/provider resource selection and validated them against the
  selected packed-i4 schedule contract, decision, reason, dispatch policy, and
  source-backed same-target measurement identity.
- Made target artifact validation require and compare realization-admission
  proof mirrors before accepting packed-i4 artifacts.
- Kept Common EmitC/export neutral: route planning and target support only
  transport explicit provider mirrors after RVV route/provider validation.
- Added focused C++ and lit coverage for successful proof flow and
  missing/stale/mismatched/sibling evidence fail-closed cases.

## Completed Slice: Gate 4

Implemented the production closeout proof:

- Repaired the current round PRD from stale Gate 3 wording to Gate 4 closeout
  acceptance criteria.
- Extended focused RVV plugin coverage so the dequant-clamp
  `same_target_measurement_record` parsed from generated artifact evidence JSON
  flows through selected-dispatch record policy, production pressure profile,
  and selected-body realization admission.
- Added fail-closed dequant-clamp coverage for stale schedule proof,
  correctness-disabled evidence, stale artifact object identity, and
  metadata-only provider support before closeout acceptance.
- Re-ran focused plugin, target artifact, tool, lit, Trellis validation, and
  bounded neutrality/legacy-authority checks.

## Remaining Slices

- None.

## Next Continuation Point

Archive the macro task after the Gate 4 closeout commit.
