> Continuation from `journal-28.md` (archived at ~1980 lines)

## Session 577: Stage2 RVV low-precision contraction primitive Gate 1

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Created the new macro Trellis task for the RVV Stage 2 low-precision
contraction primitive-surface campaign and completed Gate 1 only. Source
inspection showed the signed i8 product-reduction primitive facts, unsigned u8
widening-product path, selected-body resource facts, and target mirrors already
exist in production. The remaining Gate 1 blocker was the target artifact
provider-facts preflight: it did not directly compare the selected
low-precision resource primitive surface fields against
`RVVLowPrecisionWideningReductionPrimitiveFacts` before artifact export.

### Main Changes

- Hardened target artifact provider-facts validation so
  `lowPrecisionResourceSelection` source/product/accumulator/reduction/final
  primitive facts must match provider-owned widening-reduction primitive facts
  before artifact export.
- Added focused target artifact C++ coverage mutating a packed-i4
  product-reduction dequant resource selection's primitive product SEW to prove
  stale resource primitive facts fail closed.
- Created and left open the macro Trellis task with Gate 1 complete and Gates
  2-4 remaining.

### Testing

- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-campaign`
- [OK] `git diff --check`

### Spec Update Decision

[NO SPEC UPDATE] The RVV plugin spec already requires low-precision primitive
facts to be provider-owned and target artifact validation to reject stale
primitive/resource mirrors. This slice implements that existing contract at one
missing target provider-facts preflight.

### Status

[OPEN MACRO TASK] Gate 1 is complete. Gates 2-4 remain open. The next
continuation point is Gate 2 selected-body realization consumption of these
typed primitive facts for a representative low-precision contraction/dequant
path without changing compute semantics, ABI roles, runtime AVL/VL, dispatch,
or fallback behavior.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 579: Stage2 RVV production-kernel capability Gate 2

**Date**: 2026-06-10
**Task**: Stage2 RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed Gate 2 only. The previous slice
made the production pressure-profile boundary real; this round made generated
artifact and same-target measurement workflow records source-backed enough for
that boundary to consume without treating q8/q4 labels, artifact names, route
ids, or handwritten metadata as authority.

### Main Changes

- Extended `RVVLowPrecisionSameTargetMeasurementRecord`,
  `RVVLowPrecisionSameTargetMeasurementPolicyInput`, and
  `RVVLowPrecisionProductionPressureProfile` with selected-boundary,
  generated artifact identity, measurement target, runtime-count, and
  non-authoritative pressure-label provenance fields.
- Added C++ fail-closed validation for missing/stale source record contract,
  selected boundary, generated object/header identity, measurement target,
  runtime-count provenance, and q8/q4 label-only pressure.
- Updated `rvv_generated_bundle_same_target_measure.py` to emit the
  source-backed record into per-op/root evidence and provider feedback inputs.
- Refreshed existing packed-i4 dequant-clamp `ssh rvv` evidence JSONs with the
  new record fields derived from already validated artifact identity and
  measurement configuration, without claiming a new runtime/performance run.
- Added focused C++ tests, script self-test cases, FileCheck coverage, and an
  RVV plugin code-spec update for the new cross-layer record/API contract.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] packed-i4 dequant dry-run plus manual `FileCheck --check-prefix=PACKED-WPRD`
- [OK] packed-i4 dequant-clamp dry-run plus manual `FileCheck --check-prefix=PACKED-CLAMP-WPRDC`
- [OK] `python3 -m json.tool` on refreshed packed-i4 evidence JSON files
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-production-kernel-capability-campaign`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded diff scan found no new legacy RVV route-authority path; q8/q4
  and metadata-only occurrences are negative guards, pressure labels, spec
  text, or evidence mirrors.

### Spec Update Decision

[SPEC UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records the
Gate 2 source-backed measurement-record/API contract, including field names,
validation matrix, bad cases, and tests required for selected-boundary,
generated artifact, measurement target, runtime-count, and non-authoritative
pressure-label provenance.

### Status

[OPEN MACRO TASK] Gates 1-2 are complete. Gates 3-4 remain open. The next
continuation point is Gate 3: selected-dispatch/performance policy consumes the
source-backed pressure-profile records for preference, denial, fallback, and
stale-provenance rejection.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 581: Stage2 RVV production-kernel capability campaign Gate 1

**Date**: 2026-06-10
**Task**: Stage2 RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Created the new active macro task for the RVV production-kernel capability
campaign and completed Gate 1 only. The completed slice adds a production
pressure-profile boundary in the RVV low-precision performance-policy layer so
provider-owned resource/Gearbox facts, typed low-precision primitive facts,
runtime ABI facts, target capability mirrors, source-backed same-target
measurement inputs, and selected-dispatch policy facts are validated together
before selected-dispatch policy-input or measurement-record decisions return.

### Main Changes

- Added `RVVLowPrecisionProductionPressureProfile` and build/verify APIs to
  the low-precision performance policy contract.
- Added label-only q8/q4 and metadata-only pressure preflight rejection while
  reusing the existing provider tie-back, sibling-route, stale measurement, and
  selected-dispatch boundary validators.
- Wired selected-dispatch policy-input and source-backed record overloads
  through the pressure-profile materializer.
- Added focused RVV plugin positive coverage and fail-closed coverage for
  label-only authority, stale runtime ABI, stale primitive intrinsic, stale
  schedule decision, metadata-only provider support, sibling-route measurement,
  and stale selected-dispatch origin.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-production-kernel-capability-campaign`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded current-diff scan for legacy RVV route-authority markers returned
  no matches.

### Spec Update Decision

[NO SPEC UPDATE] The existing RVV plugin, EmitC route, variant-pipeline, and
testing specs already require provider-owned typed facts, same-target
measurement records, selected-dispatch mirrors, and fail-closed stale or
metadata-only rejection. This slice implements that existing contract as a
production pressure-profile boundary; no new durable rule was discovered.

### Status

[OPEN MACRO TASK] Gate 1 is complete. Gates 2-4 remain open. The next
continuation point is Gate 2: generated artifact and measurement workflow emits
source-backed same-target comparison records for the representative pressure
path on `ssh rvv`, using the production pressure-profile boundary as the
policy/measurement input contract.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 580: Stage2 RVV low-precision contraction primitive Gate 4

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the final Gate 4 reconciliation.
Live source inspection showed the selected-dispatch/performance policy consumer
already exists in production source: `RVVLowPrecisionPerformancePolicy` exposes
strict evaluate/verify entry points and safe dispatch resolvers for same-target
measurement outcomes, policy inputs, measurement records, and selected-dispatch
boundary facts; provider route planning and target artifact validation consume
that policy before accepting packed-i4 performance preference or selected
dispatch mirrors.

### Main Changes

- Repaired the macro PRD from stale Gate 3 wording to the current Gate 4 final
  state and marked Gates 1-4 complete.
- Updated `task.json` so Trellis task metadata no longer reports Gate 4 open.
- Refreshed `check.jsonl` with the Gate 4 inspection and verification evidence.
- Did not change production C++ because the exact Gate 4 consumer requested by
  the direction brief was already present and covered: plugin tests parse the
  checked-in Gate 3 dequant-clamp `same_target_measurement_record` JSON, feed
  the selected-dispatch record overload, preserve correctness fallback, deny
  regression/no-win performance preference, reject stale schedule and
  correctness-disabled records, and cover the measured-win preference path.

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-campaign`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `python3 -m json.tool artifacts/gate3-packed-i4-dequant-clamp-ssh/widening_product_reduce_dequant_clamp_f32/same_target_measurement_evidence.json`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded current-diff scan for legacy RVV route-authority markers returned
  no positive legacy-authority additions. The only `metadata-only` /
  `descriptor residue` matches are negative PRD guardrails.

### Spec Update Decision

[NO SPEC UPDATE] The RVV plugin spec already contains the Gate 4 packed-i4
selected-dispatch/performance policy consumption contract, including the
source-backed record overload, selected-dispatch boundary checks, resolver
fallback behavior, stale provenance rejection, and measured-win acceptance
requirements. This session reconciled the active Trellis task state with that
implemented contract.

### Status

[READY TO ARCHIVE] Gates 1-4 are complete. After final status checks, archive
the macro task and clear `.trellis/.current-task`.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 579: Stage2 RVV low-precision contraction primitive Gate 3

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed Gate 3 only. Source inspection
showed route/provider planning and target artifact mirrors already carried the
low-precision widening-reduction primitive facts, but source-backed same-target
measurement records only tied back a narrow primitive-chain subset. This round
expanded the measurement/evidence record boundary so later selected-dispatch
policy cannot consume a measurement unless it matches the provider-owned
primitive/resource/config/runtime provenance already validated by export.

### Main Changes

- Extended `RVVLowPrecisionSameTargetMeasurementRecord` and
  `RVVLowPrecisionSameTargetMeasurementPolicyInput` with provider primitive
  contract/kind, source/product/accumulator/result dtype and SEW/LMUL facts,
  primitive widening/reduction intrinsics, scalar seed splat,
  accumulator/result layouts, and reduction store-VL.
- Hardened record parsing and policy-input construction so missing or stale
  primitive measurement fields fail closed before selected-dispatch or
  performance policy consumption.
- Updated the same-target measurement script and checked-in dequant-clamp
  evidence JSON so generated records mirror validated low-precision
  resource/artifact metadata rather than carrying only primitive-chain names.
- Added plugin and target artifact C++ coverage for positive record propagation
  plus missing/stale primitive intrinsic fail-closed cases.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `python3 -m json.tool artifacts/gate3-packed-i4-dequant-clamp-ssh/widening_product_reduce_dequant_clamp_f32/same_target_measurement_evidence.json`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py`

### Spec Update Decision

[NO SPEC UPDATE] The existing RVV plugin and EmitC route specs already require
Gate 3 measurement records to be evidence tie-backs for provider-owned
resource/primitive facts and to fail closed on stale provider provenance. This
slice implements that contract at the measurement-record schema and C++ policy
input boundary.

### Status

[OPEN MACRO TASK] Gates 1-3 are complete. Gate 4 remains open. The next
continuation point is Gate 4: selected-dispatch/performance policy consumes the
expanded source-backed same-target measurement records fail-closed, preserving
correctness fallback and denying stale/no-win performance claims.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 578: Stage2 RVV low-precision contraction primitive Gate 2

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed Gate 2 only. The production
selected-body realization path already consumed many low-precision
product-reduction/dequant resource facts, but the selected resource candidate's
primitive-chain fields were not directly compared against provider-owned
`RVVLowPrecisionWideningReductionPrimitiveFacts` at the realization boundary.
This round tightened that RVV plugin-local boundary before route/provider or
target artifact authority can accept stale primitive-chain facts.

### Main Changes

- Added `validateLowPrecisionResourceCandidatePrimitiveFacts` in the RVV
  contraction selected-body realization owner so primitive contract/kind,
  chain contract/kind, product/reduction relations, intrinsic spellings,
  accumulator/result layouts, and reduction store-VL must match provider-owned
  primitive facts before realized `with_vl`, region-marker, handoff, or
  provider-visible resource facts are accepted.
- Added C++ regression coverage for missing primitive resource facts and a
  stale primitive reduction intrinsic, both failing closed at selected-body
  realization before route construction or artifact validation.
- Updated the macro PRD/task description to mark Gate 2 complete and leave
  Gates 3-4 open.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-campaign`
- [OK] `git diff --check`
- [OK] bounded diff scan for legacy RVV route-authority markers returned no
  matches.

### Spec Update Decision

[NO SPEC UPDATE] The RVV plugin spec already states that low-precision
product-reduction selected-body realization must consume provider-owned
widening-reduction primitive facts and pass-produced resource facts before
route construction. This slice implements that existing contract at the
candidate primitive-chain comparison point.

### Status

[OPEN MACRO TASK] Gates 1-2 are complete. Gates 3-4 remain open. The next
continuation point is Gate 3: carry the selected-body-realization-consumed
primitive/resource facts through route/provider/artifact export into generated
artifacts and source-backed same-target measurement records with fail-closed
stale mirror diagnostics.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 579: Stage2 RVV production-kernel capability Gate 3

**Date**: 2026-06-10
**Task**: Stage2 RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed Gate 3 only. Gate 2 already
emitted source-backed pressure-profile records from selected-boundary,
provider-owned resource/primitive, generated artifact, and same-target
measurement facts. This round hardened the production selected-dispatch safe
resolver so accepted source-backed measurement records must also pass through
the production pressure-profile boundary before the resolver returns a dispatch
/ performance decision.

### Main Changes

- Updated the macro PRD from the stale Gate 2 current-slice description to Gate
  3 selected-dispatch/performance policy consumption, then marked Gate 3
  complete after implementation and checks.
- Changed `resolveRVVLowPrecisionDispatchPerformancePolicy(selection, record,
  dispatchBoundary, context)` so an accepted record/resource/measurement
  handoff also materializes `RVVLowPrecisionProductionPressureProfile`. If that
  boundary rejects stale or marker-only pressure facts, the resolver denies
  performance preference and win claims, preserves correctness fallback for the
  legal route, and carries the precise failure in `fallbackReason`.
- Added focused C++ coverage for a metadata-only selected-dispatch marker in
  the record resolver path, proving the resolver consumes the pressure-profile
  boundary instead of treating marker-only dispatch facts as policy authority.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Bounded added-diff scan for legacy RVV route-authority markers returned
  no `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  `__riscv_*_i32m1`, source-front-door, or descriptor-driven route-authority
  matches. The broader scan only found PRD non-authority wording and the new
  metadata-only negative test marker.

### Spec Update Decision

[NO SPEC UPDATE] The RVV plugin spec already requires selected-dispatch policy
consumption to prefer source-backed measurement records, validate the
record/resource/measurement handoff through the pressure-profile boundary, deny
performance preference for stale or marker-only facts, and preserve correctness
fallback where the selected route remains legal. This slice implements that
existing contract in production policy code and tests.

### Status

[OPEN MACRO TASK] Gates 1-3 are complete. Gate 4 remains open. The next
continuation point is Gate 4: final campaign closeout must prove the full
selected boundary -> plugin-owned realization/provider -> artifact/runtime
measurement -> selected-dispatch policy path, including parsed generated
evidence through the selected-dispatch record overload, before this macro task
can be finished or archived.

### Git Commits

Final coherent commit is created after this journal entry.
