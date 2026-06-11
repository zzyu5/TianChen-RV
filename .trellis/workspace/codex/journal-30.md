> Continuation from `journal-29.md` (archived at more than 2000 lines)

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
