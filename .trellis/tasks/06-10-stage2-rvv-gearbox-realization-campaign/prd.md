# Stage2 RVV Gearbox Resource-Aware Selected-Body Realization Campaign

## Goal

Create one active macro owner for RVV plugin-local resource-aware
selected-body realization for low-precision production kernels. The campaign
makes source-backed production pressure/profile facts, primitive/resource
facts, target capability facts, runtime ABI facts, and selected-dispatch policy
results affect RVV selected-body realization and later Gearbox schedule
admission.

Gates 1-3 are complete. The current round completed Gate 3: after a trusted
Gate 1 pressure-profile admission and Gate 2 admitted Gearbox/resource-aware
schedule choice, the realized packed-i4 low-precision contraction body plus
RVV-owned route/artifact/measurement proof path must consume the admitted
schedule facts. Downstream proof must fail closed when schedule evidence is
missing, stale, mismatched, label-only, metadata-only, sibling-route-derived,
or not tied to source-backed same-target records. The boundary is not route
authority, artifact-name authority, helper-name authority, or Common EmitC
semantic inference.

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
- [ ] Gate 4: selected-dispatch policy, realization admission, artifact
  evidence, and measurement provenance compose into one closeout test; archive
  only after this gate.

## Current Round Slice: Gate 3

Implement a focused production proof-consumption seam:

- Carry admitted packed-i4 schedule proof from realized `with_vl` / Gearbox
  handoff structure into the RVV route/provider selected-resource fact surface.
- Make target artifact validation and source-backed same-target measurement
  record/policy input validation consume that admitted schedule proof as exact
  mirrors of provider-owned facts.
- Preserve existing computation, dtype, parameter roles, selected variant
  origin, dispatch/fallback behavior, and runtime AVL/VL values.
- Fail closed when admitted schedule proof or same-target proof is missing,
  stale, mismatched, metadata-only, label-only, sibling-route-derived, or not
  source-backed.
- Keep Common EmitC/export neutral; route/artifact/report metadata remain
  mirrors only.

Acceptance criteria:

- [x] Production C++ route/provider fact surface carries admitted schedule
  proof only after the realized body materialized Gate 2 admission mirrors.
- [x] Target artifact validation consumes admitted schedule proof mirrors and
  rejects missing/stale/mismatched proof before accepting packed-i4 artifacts.
- [x] Source-backed same-target record and policy input validation tie schedule
  proof to the provider-selected resource candidate and measurement identity.
- [x] Focused C++ tests prove successful Gate 3 proof from realized admission
  through route/artifact/measurement consumers.
- [x] Focused C++ tests prove missing/stale/mismatched admission schedule proof
  fails closed before route/artifact/measurement proof can be accepted.
- [x] Existing Gate 1 pressure-profile and selected-dispatch policy tests still
  pass.
- [x] Bounded check confirms Common EmitC/export remains neutral and metadata
  fields are mirrors only.
- [x] Relevant focused checks pass.
- [x] Task remains active with Gates 1-3 marked complete and Gate 4
  remaining.

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
- No campaign archive after Gate 2.

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
- Any focused target artifact/export check touched by the implementation.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-gearbox-realization-campaign`
- `git diff --check`
- `git diff --cached --check`
- Bounded added-diff scan for legacy RVV route-authority markers.

## Verification Results

- Passed `cmake --build build --target tianchenrv-rvv-extension-plugin-test`.
- Passed `build/bin/tianchenrv-rvv-extension-plugin-test`.
- Passed `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-gearbox-realization-campaign`.
- Passed `git diff --check`.
- Passed `git diff --cached --check`.
- Passed bounded added-diff scan over source/test changes for legacy RVV
  route-authority markers. The PRD's own forbidden-marker vocabulary is
  non-authority reporting text only.
- Confirmed changed paths stay in RVV plugin/admission/owner/test/spec/task
  documentation and do not modify common EmitC/export materializer code.
- No focused target artifact/export check was run because this Gate 2 slice did
  not touch route/artifact/export boundaries.
- Initial `task.py validate` failed because command strings were placed in
  `check.jsonl.file`; repaired `check.jsonl` to reference real files.

## Spec Update Decision

Updated `.trellis/spec/extension-plugins/rvv-plugin.md` because Gate 2 changed
the executable realization admission payload. The new
Low-Precision Realization Admission Schedule Handoff contract records the
schedule decision fields, validation matrix, mirror-only realized structure
contract, tests required, and Common EmitC/export neutrality boundary.

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

## Remaining Slices

- Gate 4 composed closeout and archive.

## Next Continuation Point

Continue with Gate 4: compose selected-dispatch policy, realization admission,
target artifact evidence, and measurement provenance into one closeout test,
then archive only if the composed Gate 4 acceptance criteria are actually met.
