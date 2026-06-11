> Continuation from `journal-29.md` (archived at more than 2000 lines)

## Session 597: Stage2 RVV Gate 4 admission boundary cleanup

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the first Gate 4 admission
boundary cleanup slice for packed-i4 low-precision resource/schedule and target
validation ownership. The cleanup separates stable compiler/resource facts from
generic selected-dispatch policy-output facts and campaign evidence/admission
facts without admitting a measured-win or changing fallback behavior.

### Main Changes

- Classified stable facts as candidate selection, planning contract,
  dtype/SEW/LMUL, packed-i4 load/unpack, primitive product/reduction,
  resource cost, schedule decision, realization/region/live-vector facts,
  target capability mirrors, legality, and rejection reason.
- Classified generic policy-output facts as selected-dispatch case/fallback
  mirrors plus `RVVLowPrecisionPerformancePolicyDecision` route, correctness,
  performance, win-claim, dispatch-path, and path-selection outputs.
- Classified campaign evidence/admission facts as performance
  feedback/baseline/speedup/action, remediation narrative/evidence,
  performance admission closure/reopen, beyond-local repair admission,
  performance maturity, and same-target measurement records.
- Renamed the packed-i4 schedule-decision predicate to a stable Gearbox
  schedule helper and removed Gate 4 admission/evidence fields from its
  acceptance predicate.
- Split target artifact validation into stable resource mirror validation and
  Gate 4 evidence/admission mirror validation. Stale Gate 4 mirrors still fail
  closed, but diagnostics now label them as evidence/admission ownership rather
  than stable resource authority.
- Added focused plugin coverage proving stale or missing Gate 4 admission
  fields do not invalidate an otherwise stable packed-i4 schedule decision.
- Updated the RVV plugin code-spec, macro PRD, task notes, and Trellis JSONL
  logs with the completed cleanup slice and remaining macro continuation.

### Testing

- [OK] `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-same-target-measure-gate4-dry-run|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4'`
- [OK] `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-production-gearbox-low-precision-contraction`
- [OK] `rtk git diff --check`
- [OK] bounded old-authority scan over added diff lines returned only
  spec/PRD negative boundary text for artifact-name authority.

### Self-Repair

- The first focused lit run failed because `tcrv-translate` had not been
  rebuilt after the target diagnostic-label split and still emitted the prior
  stable-resource diagnostic label. Rebuilding `tcrv-translate` and rerunning
  the same lit filter passed 3/3.

### Status

[OPEN MACRO TASK] Gate 4 admission-boundary cleanup is complete for the first
packed-i4 production slice. Accepted dequant and dequant-clamp records remain
no-win/regression, preserving route support and correctness execution while
denying performance-preferred selection and performance-win claims. The macro
task remains active for measured-win admission and final provider,
selected-dispatch, target mirror, and dispatch/fallback closure.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 599: Stage2 RVV Gate 2 signed product-reduction realization closure

**Date**: 2026-06-11
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and closed the interrupted dirty Gate 2 signed
selected-body realization slice for the plain low-precision product-reduction
representative. The slice adds a typed pre-realized signed i8 product-reduction
body, validates it in the RVV dialect and provider boundary, and realizes it
through the RVV contraction selected-body owner before existing route/provider,
emission-plan, and target artifact validation consume the facts.

### Main Changes

- Added `tcrv_rvv.typed_widening_product_reduce_pre_realized_body` for the
  bounded signed i8 -> i16 product -> i32 reduction body.
- Added verifier checks for direct selected-variant nesting, runtime ABI roles
  and C types, signed source/product/accumulator/result SEW/LMUL, policy,
  memory form, product relation, and product-reduction chain relation.
- Added provider-side pre-realized validation and wired the RVV contraction
  selected-body realization owner to materialize `setvl`, `with_vl`, two
  loads, `widening_product`, `standalone_reduce`, and store before route
  construction.
- Added focused lit coverage for the pre-realized signed path, emission-plan
  mirrors, header export, stale source SEW rejection, and preservation of the
  existing explicit signed and unsigned product-reduction fixtures.
- Updated the RVV plugin spec and active macro PRD/task notes with the Gate 2
  signed realization contract and continuation boundary.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-add|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8'`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-low-precision-contraction-surface`
- [OK] `git diff --check`
- [OK] bounded old-authority scan over added diff lines returned only negative
  q8/q4/source-front-door/descriptor boundary text.

### Self-Repair

- Fixed a formatting issue left in the dirty realization branch after the
  first successful build/lit pass.
- Corrected primitive fact mismatch diagnostics so provider-owned stale facts
  are reported as the found value against the selected typed body contract.
- Corrected the new RVV plugin spec signature example to use the actual
  verifier/provider attribute values for accumulator role/layout and result
  layout.

### Status

[OPEN MACRO TASK] Gate 2 is complete for the signed i8
product-reduction selected-body realization representative. The macro task
remains active. The exact continuation point is unsigned u8 selected-body
realization for the same product-reduction primitive surface, or a bounded
decision to keep unsigned u8 as explicit-body-only until the next gate. Gate 3
route/provider/artifact carry-through and Gate 4 measured same-target evidence
remain future work only when those claims are made.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 592: Stage2 RVV low-precision primitive Gate 1 mirror hardening

**Date**: 2026-06-11
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Created the new macro campaign task and completed the bounded Gate 1 foundation
slice. This round hardened the existing typed low-precision product-reduction
primitive contract by carrying typed config and runtime control facts in the
`tcrv_rvv.low_precision_primitive.*` mirror namespace instead of leaving them
only as generic route-family metadata.

### Main Changes

- Added provider-emitted low-precision primitive mirrors for
  source/product/accumulator/result SEW/LMUL, tail policy, mask policy,
  runtime control plan, and runtime AVL source.
- Added support-bundle mappings so target header/export artifacts can carry
  the new primitive mirrors.
- Required target artifact validation to compare the new primitive mirrors
  against provider-owned widening-product and product-reduction contracts
  before artifact acceptance.
- Extended signed i8 and unsigned u8 product-reduction lit fixtures with
  positive mirror checks and stale runtime AVL source / stale tail policy
  fail-closed coverage.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v . --filter 'explicit-selected-body-artifact-widening-product-reduce-add'` from `build/test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] Added-diff old-authority scan for `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, exact `__riscv_*_i32m1`, q8/q4,
  source-front-door, and artifact-name authority patterns returned no matches.

### Self-Repair

- The first focused lit run passed the signed fixture but failed the unsigned
  header check because FileCheck expected new primitive mirrors after
  `low_precision_resource.*` mirrors. The actual support-bundle order is
  primitive mirrors first, resource mirrors second, so the unsigned header
  checks were reordered without changing production code.

### Status

[OPEN MACRO TASK] Gate 1 foundation is complete for the current
product-reduction primitive mirror hardening slice. The task remains active for
Gate 2 selected-body realization from this primitive surface, Gate 3 broader
route/artifact carry-through, and later Gate 4 source-backed same-target
evidence/admission only if fresh measured evidence is available.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 598: Stage2 RVV Gate 4 final no-win admission consumer closure

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Completed the final Gate 4 admission/consumer closure slice for the current
packed-i4 low-precision Gearbox campaign. Current source-backed same-target
evidence remains no-win/regression, so this round did not admit
performance-preferred dispatch. The closure proves the conservative path at the
target consumer boundary and rejects synthetic measured-win promotion without a
fresh source-backed measured-win target admission chain.

### Main Changes

- Added target-artifact consumer coverage showing parsed dequant evidence-root
  records populate selected-dispatch policy-output mirrors through the explicit
  record overload before target validation consumes no-win mirrors.
- Added target validation coverage for synthetic measured-win promotion:
  provider maturity/admission fields, selected-dispatch policy-output mirrors,
  and target metadata rewritten together still fail closed because the current
  target admission boundary remains source-backed no-win.
- Added stale no-win provider policy-output cases for
  `dispatch_policy_path = performance-preferred` and
  `performance_win_claim_allowed = true` where candidate metadata mirrors the
  stale provider value.
- Recorded the durable spec rule that target validation using the default
  packed-i4 helper is a current no-win/fallback consumer, not a measured-win
  admission seam.
- Updated PRD/task notes to mark Gate 4 and the current macro campaign scope
  complete.

### Testing

- [OK] `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-same-target-measure-gate4-dry-run|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4'`
- [OK] `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-production-gearbox-low-precision-contraction`
- [OK] `rtk git diff --check`
- [OK] bounded old-authority scan returned only archived PRD
  negative/non-authority boundary text for q8/q4 labels, artifact names, route
  IDs, helper names, and helper-op authority; production/test/spec added lines
  returned no matches.

### Self-Repair

- The first target test run failed because a measured-win fixture reused a
  boundary already populated with no-win policy-output facts. The test now
  clears provider-owned policy-output before record-derived population.
- The next target test run showed production target validation still consumes
  the default current no-win record at the target boundary. The test was
  corrected from a synthetic measured-win target-positive to an explicit
  fail-closed measured-win-promotion rejection, matching the no-fresh-win PRD.

### Status

[READY TO FINISH] Gate 1 resource/primitive fact spine, Gate 2 packed-i4
primitive/resource surface, Gate 3 resource-aware selected-body realization,
and Gate 4 same-target no-win admission/consumer closure are complete for the
current macro campaign scope. Future measured-win work requires new
source-backed same-target evidence and synchronized provider-owned admission
facts before dispatch preference may change.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 596: Stage2 RVV Gearbox Gate 4 dispatch/fallback consumption audit

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed a bounded Gate 4 selected-dispatch
/ fallback consumption consistency slice. The production gap was that target
artifact validation could prove candidate metadata exactly matched provider
policy-output mirrors, but still needed to reject a stale provider
policy-output value that the candidate metadata had mirrored exactly. This
round anchored target-artifact consumption back to provider-owned resource
selection and same-target no-win/regression policy facts.

### Main Changes

- Extended the active macro PRD from policy-output mirror admission into the
  current dispatch/fallback consumption slice.
- Required target artifact validation to consume selected-dispatch
  policy-output mirrors only with complete provider-owned selected-dispatch
  case and fallback facts.
- Added packed-i4 no-win/regression consumption checks that preserve
  `correctness-fallback`, `not-performance-preferred`, route support,
  correctness execution, no performance-selection allowance, no performance-win
  claim, correctness fallback path selection, and no performance-preferred path
  selection.
- Rejected stale provider `selected_dispatch_preference` and
  `correctness_fallback_path_selected` values even when candidate metadata was
  rewritten to match those stale provider values exactly.
- Added C++ target validation coverage for the stale-provider exact-mirror
  cases.
- Added dequant-clamp lit coverage for positive policy-output plan/header
  mirrors and stale target-export rejection.
- Updated the RVV plugin code-spec with the target-artifact consumption
  contract that selected-dispatch policy-output mirrors must be anchored in
  resource-selection and same-target measurement facts, not metadata equality.

### Testing

- [OK] `git diff --check`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-same-target-measure-gate4-dry-run|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4'`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-production-gearbox-low-precision-contraction`
- [OK] bounded old-authority scan over production/test/spec/PRD diff lines
  returned only PRD negative boundary text forbidding artifact metadata, route
  id, and helper-name authority.

### Self-Repair

- The first focused lit run failed because the new dequant-clamp `PLAN-DAG`
  policy-output checks were placed after stable low-precision metadata that
  appears later in the emission plan. The checks were moved to the actual plan
  order and the same focused lit filter passed 3/3.

### Status

[OPEN MACRO TASK] Gate 4 dispatch/fallback consumption consistency is complete
for the accepted packed-i4 dequant and dequant-clamp no-win evidence paths.
The macro task remains active for the remaining Gate 4 measured-win admission
and final end-to-end provider, selected-dispatch, target mirror, and
dispatch/fallback closure. No performance-preferred dispatch is admitted by
this slice.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 595: Stage2 RVV Gearbox Gate 4 policy-output mirror admission

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed a bounded Gate 4 selected-dispatch
/ target artifact policy-output mirror admission slice. The production gap was
that `selected_dispatch_preference` was specified as a provider-owned
policy-output field, but was not yet emitted and target-validated as its own
mirror alongside the existing low-precision `dispatch_preference` mirror.

### Main Changes

- Extended the active macro PRD from the completed Gate 4 root-admission slice
  into the current policy-output mirror admission slice.
- Emitted `tcrv_rvv.low_precision_resource.selected_dispatch_preference` from
  RVV provider-owned route metadata only when
  `hasSelectedDispatchPolicyOutput` is true.
- Added the same key to the RVV target support-bundle metadata mapping.
- Required target artifact validation to reject metadata-only or stale
  `selected_dispatch_preference` mirrors before artifact acceptance.
- Added focused C++ target validation coverage for positive source-backed
  no-win policy output, stale `performance-preferred` preference promotion, and
  metadata-only selected-dispatch preference mirrors.
- Added lit/FileCheck plan/header checks and stale target-export rejection for
  the packed-i4 dequant representative.
- Updated the RVV plugin code-spec so future Gate 4 target tests must cover
  stale `selected_dispatch_preference` policy-output mirrors explicitly.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-same-target-measure-gate4-dry-run|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4'`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-production-gearbox-low-precision-contraction`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded old-authority scan over production/test diff lines returned no
  matches; docs-inclusive scan returned only negative PRD/spec boundary text.

### Status

[OPEN MACRO TASK] Gate 4 selected-dispatch policy-output mirror admission is
complete for the `selected_dispatch_preference` target mirror slice. The
accepted packed-i4 evidence remains no-win, so dispatch stays
`correctness-fallback` / `not-performance-preferred`. The macro task remains
active for the rest of Gate 4: full same-target measured-win/no-win provider,
selected-dispatch, target mirror, and fallback/dispatch behavior audit.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 594: Stage2 RVV Gearbox Gate 4 evidence-root admission closure

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed a bounded Gate 4 slice at the
source-backed same-target evidence-root ingestion boundary. The representative
path is the packed-i4 `widening_product_reduce_dequantize_f32` evidence root,
with packed-i4 `widening_product_reduce_dequant_clamp_f32` preserved as the
sibling audit path.

### Main Changes

- Repaired the active macro PRD from Gate 3C completion state into the current
  Gate 4 root-admission slice.
- Made `verifyPackedI4SameTargetEvidenceRoot` compare root-level
  `measurement_harness`, `measurement_schedule_decision_evidence`, and
  `packed_i4_reference_oracle` performance-admission closure/reopen mirrors
  against the parsed source-backed measurement record.
- Added C++ fail-closed coverage for stale root
  `provider_performance_admission_closure` and
  `provider_performance_admission_reopen_requirement` fields, proving the
  policy verifier does not rely only on the nested
  `same_target_measurement_record`.
- Updated the RVV plugin code-spec to record the Gate 4 root-ingestion
  closure/reopen mirror contract and required stale-root tests.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-same-target-measure-gate4-dry-run|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4'`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-production-gearbox-low-precision-contraction`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded old-authority scan over added diff lines returned only PRD
  negative boundary text for artifact-name logic.

### Status

[OPEN MACRO TASK] Gate 4 root-admission closure/reopen validation is complete
for the current packed-i4 source-backed evidence-root slice. The accepted
measurement result remains no-win, so dispatch stays
`correctness-fallback` / `not-performance-preferred`. The macro task remains
active for remaining Gate 4 campaign-level measurement/admission audit and any
future measured-win admission only after fresh source-backed same-target
evidence plus matching provider admission facts exist.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 590: Stage2 RVV Gearbox Gate 2 candidate facts

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the Gate 2
widening-product/reduction resource-candidate fact slice. The previous slice
made packed-i4 load/unpack resource facts explicit. This round made the
selected widening-product and widening-reduction candidate facts explicit in
the RVV-owned low-precision resource selection, consumed them during
selected-body realization and route planning, mirrored them into target
artifacts, and rejected stale provider-side and target-side facts.

### Main Changes

- Added explicit widening-product and widening-reduction candidate fact fields
  to RVV low-precision resource candidate/selection payloads.
- Derived signed i8 and unsigned u8 candidate facts from provider-owned
  primitive chain facts instead of from route ids, artifact names, helper names,
  or emitted C spelling.
- Required selected-body realization to consume pre-realized candidate facts and
  emit them on `tcrv_rvv.gearbox_cross_region_handoff`.
- Required selected-body route planning and provider family validation to
  reject stale Gearbox handoff candidate facts before constructing a
  `TCRVEmitCLowerableRoute`.
- Mirrored the candidate facts through route metadata, target support bundle
  keys, and target artifact validation with stale target-mirror rejection.
- Updated the RVV plugin code-spec to record the new resource candidate facts,
  handoff fields, target stale-mirror behavior, and focused test requirements.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-gearbox-widening-product-reduce-dequantize-f32|explicit-selected-body-artifact-widening-product-reduce-dequantize-f32|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8'`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8'`
- [OK] Manual FileCheck stale-provider/stale-target cases for
  `STALE-WIDENING-CANDIDATE`, `STALE-REDUCTION-CANDIDATE`,
  `STALE-HANDOFF-CANDIDATE`, `STALE-TARGET-CANDIDATE`, and
  `STALE-RESOURCE-REDUCTION-CANDIDATE`.

### Self-Repair

- Seeded C++ selected-body and target artifact fixtures with the new explicit
  candidate facts after stricter selected-body realization and target
  validation exposed missing fixture attrs.
- Updated explicit selected-body dequant and pre-realized dequant expectations
  to match the current fail-closed diagnostics and stable metadata/header order.
- Added the code-spec update after recognizing that the new candidate fields are
  a cross-layer compiler contract, not only an implementation detail.

### Status

[OPEN MACRO TASK] Gate 2 is complete for the current low-precision resource
fact spine: packed-i4 load/unpack facts plus widening-product/reduction
candidate facts are explicit, produced by RVV-owned resource paths, consumed by
selected-body realization and route planning, mirrored by target artifacts, and
stale-rejected in focused signed/unsigned and packed-i4-adjacent coverage.

Gates 3 and 4 remain open. The next continuation point is Gate 3 selected-body
realization consumption using this completed fact spine to build
resource-aware executable bodies without changing computation semantics, ABI
roles, dispatch, fallback, or runtime AVL. Do not move to Gate 4
performance-preferred dispatch without fresh source-backed same-target measured
win evidence.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 593: Stage2 RVV Gearbox Gate 3C family completion audit

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the Gate 3C family
completion/audit slice for the current low-precision resource-aware
selected-body realization campaign. The audit covered signed/unsigned
plain product-reduction resource facts as adjacent Gate 2 route-supported
representatives, plus product-reduction/dequantize-f32 and
product-reduction/dequant-clamp-f32 grouped/unpacked-byte and packed-i4
Gearbox representatives.

### Main Changes

- Rewired Gearbox product-dequant handoff validation to use the canonical
  provider-owned realization-decision helpers for supported decision,
  expected `with_vl` region count, and product phase.
- Added non-clamp product-dequant stale-handoff coverage proving a
  dequant-clamp-only `clamp_phase` fact is rejected before Common EmitC
  materialization.
- Updated the macro PRD with an explicit Gate 3C inventory, acceptance criteria,
  Gate 3 completion status, and Gate 4 continuation boundary.
- Updated the RVV plugin code-spec with the Gate 3C completion/audit evidence
  contract so future slices keep Gearbox schedule validation, selected-body
  realization, dialect handoff verification, route planning, and target
  validation on the same realization-decision mapping.

### Testing

- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32'`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4|rvv-gearbox-widening-product-reduce-dequantize-f32|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8'`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-production-gearbox-low-precision-contraction`
- [OK] `git diff --check`

### Self-Repair

- The Gate 3C audit found one production-source consistency gap: Gearbox
  product-dequant handoff validation locally rederived the supported decisions,
  expected region count, and product phase instead of consuming the same
  provider-owned realization-decision mapping used by realization, route
  planning, and target validation. The fix replaced the local derivation with
  the canonical helpers.

### Status

[OPEN MACRO TASK] Gate 3 is complete for the current campaign body-family
audit. Gate 4 remains blocked until fresh source-backed same-target measured-win
evidence exists for performance-preferred dispatch admission. The Trellis task
stays active.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 592: Stage2 RVV Gearbox Gate 3B dequant-clamp realization fact consumption

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the Gate 3B slice for
`widening_product_reduce_dequant_clamp_f32`. This round extended the Gate 3A
realized-body consumption pattern from product/dequant region and phase facts
to provider-derived dequant-clamp clamp/compare-select facts, while preserving
semantics, ABI roles, dispatch/fallback behavior, and runtime AVL.

### Main Changes

- Added dequant-clamp low-precision resource fields:
  `clamp_region_index`, `clamp_phase`, `clamp_compare_select_phase`, and
  `clamp_select_layout`.
- Materialized those facts on realized producer and consumer `with_vl` scopes
  and on `tcrv_rvv.gearbox_cross_region_handoff`.
- Made Gearbox schedule materialization, dialect handoff verification, route
  planning, and target artifact validation consume and stale-reject those facts
  before route or artifact acceptance.
- Mirrored the provider-selected clamp facts into route-plan metadata and target
  header metadata only after provider/route consumers have validated them.
- Added focused lit coverage for positive dequant-clamp fact materialization
  and stale realized-body, handoff, and target mirror rejection.
- Preserved the packed-i4 dequant-clamp sibling fixture with positive clamp
  fact/mirror checks.
- Updated the RVV plugin code-spec and the active macro PRD with the Gate 3B
  field contract and acceptance criteria.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4'`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-production-gearbox-low-precision-contraction`
- [OK] bounded old-authority scan over added diff lines returned no matches.

### Self-Repair

- `tianchenrv-target-artifact-export-test` initially failed because a synthetic
  packed-i4 sibling-candidate negative test changed `selectedCandidateID` to
  dequant-clamp without completing the new clamp facts. The test now fills the
  clamp facts from provider helpers so it still exercises the intended stale
  performance-baseline diagnostic.
- A header helper declaration was split to match the surrounding formatting
  style before the final build/check pass.

### Status

[OPEN MACRO TASK] Gate 3B is complete for
`widening_product_reduce_dequant_clamp_f32` realization-fact consumption. Gate
2 and Gate 3A facts remain preserved. The macro task stays active for remaining
Gate 3 resource-aware body-family completion/audit and later Gate 4
source-backed same-target measured-win dispatch admission.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 591: Stage2 RVV Gearbox Gate 3A realization fact consumption

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the Gate 3A selected-body
realization-consumption slice for the current low-precision
product-reduction/dequantize-f32 spine. This round made the route-planning path
consume realization-produced product/dequant region and phase facts from
realized `with_vl` scopes rather than rebuilding those accepted facts locally
from candidate names or route metadata.

### Main Changes

- Added explicit `product_region_index`, `dequant_region_index`,
  `product_phase`, and `dequant_phase` low-precision resource attr constants.
- Required RVV selected-body realization to write product/dequant region and
  phase facts on realized producer and consumer `with_vl` scopes.
- Required the Gearbox schedule materialization path to materialize or
  fail-closed-validate the same realization producer/decision, realized
  unroll/region/peak-live, and product/dequant region/phase facts for explicit
  selected-body fixtures.
- Changed low-precision route-family planning to read realization-produced
  resource facts before route construction and to reject stale realized-body
  product/dequant facts.
- Added focused stale `product_phase` and `dequant_region_index` FileCheck
  coverage for the pre-realized product-reduction/dequantize-f32 artifact path.
- Updated the RVV plugin code-spec to preserve the Gate 3A realization fact
  consumption contract.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'explicit-selected-body-artifact-widening-product-reduce-dequantize-f32|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|rvv-gearbox-widening-product-reduce-dequantize-f32'`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4|rvv-gearbox-widening-product-reduce-dequantize-f32'`

### Self-Repair

- The first explicit selected-body lit run exposed that the Gearbox schedule
  materialization path wrote provider resource facts but not the new
  realization region/phase facts. The fix moved the fact materialization into
  `materializeLowPrecisionResourceAttrs`, so both explicit and pre-realized
  paths share the same producer-owned fact surface.
- The first pre-realized FileCheck update used an over-broad unordered match
  and could bind to the consumer `with_vl`. The check was tightened to an
  ordered selected-variant anchor and `SAME` region/phase assertions.

### Status

[OPEN MACRO TASK] Gate 3A is complete for the current
product-reduction/dequantize-f32 realization-consumption slice. Gate 3 remains
open for extending the same resource-aware realized-body consumption pattern to
the next low-precision body-family representative. Gate 4 remains later work
and still requires fresh source-backed same-target measured-win evidence before
performance-preferred dispatch admission.

### Git Commits

Final coherent commit is created after this journal entry.
