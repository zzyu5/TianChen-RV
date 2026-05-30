# Journal - codex (Part 19)

> Continuation from `journal-18.md` (archived at ~2000 lines)
> Started: 2026-05-30

---

## Session 351: Stage2 RVV reduction selected-body owner cleanup

**Date**: 2026-05-31
**Task**: Stage2 RVV reduction selected-body owner cleanup
**Branch**: `main`

### Summary

Moved `TypedReducePreRealizedBodyOp` selected-body realization ownership out of
central `RVVSelectedBodyRealization.cpp` into a dedicated RVV reduction owner
module; preserved selected-boundary realization, realized typed
load/load/reduce/store facts, provider route consumption, common EmitC
neutrality, and target artifact behavior.

### Main Changes

- Added
  `RVVReductionSelectedBodyRealizationOwner.{h,cpp}` and CMake wiring.
- Central `RVVSelectedBodyRealization.cpp` now keeps only the reduction owner
  registry entry and dispatch hook for this family; reduction predicate,
  validation, and materialization live in the new owner module.
- Added focused C++ coverage for reduction owner classification, null/wrong
  family rejection, invalid typed fact diagnostics, selected-boundary
  materialization, and provider route consumption.
- Verified plain reduction artifact/materialization lit flows, neighboring
  standalone reduction/runtime-scalar memory owner paths via the RVV plugin
  smoke test, authority scans, and full `check-tianchenrv`.
- Spec update review completed: no `.trellis/spec/` edits were needed because
  this task implemented existing selected-body owner registry rules rather
  than introducing a new durable contract.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | `rvv: move reduction selected-body owner-side` |

### Testing

- [OK] `cmake --build build --target TianChenRVRVVPlugin -j2`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused reduction artifact/materialization checks for
  `pre-realized-selected-body-artifact-reduce-add.mlir`,
  `rvv-generic-stage2-reduction-materialization.mlir`, and
  `rvv-generic-stage2-reduction-negative.mlir`.
- [OK] Bounded authority scans over touched production files.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 464/464.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 350: Stage2 RVV elementwise/compare-select selected-body owner interface split

**Date**: 2026-05-31
**Task**: Stage2 RVV elementwise/compare-select selected-body owner interface split
**Branch**: `main`

### Summary

Split elementwise/compare-select selected-body owner APIs out of the EmitC
route-family planning header into a dedicated RVV owner header, kept the
central selected-body registry dependent on the owner interface, and preserved
route-family planning as route-plan/mirror/operand-binding API only.

### Main Changes

- Added `RVVElementwiseSelectedBodyRealizationOwner.h` for the
  elementwise/compare-select owner result type, predicate, variant predicate,
  owner hook, and selected-body helper.
- Removed owner declarations from
  `RVVEmitCElementwiseRouteFamilyPlanOwners.h`; it now keeps route-family
  planning declarations only.
- Updated `RVVSelectedBodyRealization.cpp`,
  `RVVElementwiseSelectedBodyRealizationOwner.cpp`, and RVV plugin tests to
  include the dedicated owner header explicitly.
- Recorded the owner-header locality contract in
  `.trellis/spec/extension-plugins/rvv-plugin.md`.

### Git Commits

| Hash | Message |
|------|---------|
| this commit | rvv: split elementwise selected-body owner interface |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tcrv-opt -j2`
- [OK] `cmake --build build --target tcrv-translate -j2`
- [OK] `git diff --check`
- [OK] Bounded owner/API and authority scans
- [OK] `cmake --build build --target check-tianchenrv -j2`: 464/464 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 350: Stage2 RVV widening-conversion selected-body owner cleanup

**Date**: 2026-05-31
**Task**: Stage2 RVV widening-conversion selected-body owner cleanup
**Branch**: `main`

### Summary

Moved widening-conversion selected-body realization ownership out of the
central RVV selected-body file into a dedicated owner module; preserved
typed widening-conversion realization facts, provider route construction, and
artifact/materialization behavior.

### Main Changes

- Added `RVVWideningConversionSelectedBodyRealizationOwner.{h,cpp}` and CMake
  wiring.
- Central `RVVSelectedBodyRealization.cpp` now keeps only the
  widening-conversion owner registry entry and generic dispatch.
- Moved widening-conversion op-kind, memory-form, relation/signature,
  runtime ABI, mixed-body, requires, and materialization checks into the
  dedicated owner.
- Added C++ coverage for dedicated owner predicate selection, null/wrong-family
  diagnostics, invalid typed facts, realized load/widening_convert/store facts,
  and provider route consumption.
- Checks passed: `tianchenrv-rvv-extension-plugin-test`, focused
  widening-conversion lit tests, authority scans, `git diff --check`, and
  `cmake --build build --target check-tianchenrv -j2` (464/464).

### Git Commits

| Hash | Message |
|------|---------|
| this commit | rvv: move widening conversion selected-body owner-side |

### Testing

- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused lit filter for widening-conversion selected-body artifacts and
  negative lowering-boundary coverage
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` (464/464)

### Status

[OK] **Completed**

### Next Steps

- None - task complete



## Session 339: Stage2 RVV contraction selected-body handoff closure

**Date**: 2026-05-30
**Task**: Stage2 RVV contraction selected-body handoff closure
**Branch**: `main`

### Summary

Moved pre-realized widening contraction validation into the contraction route-family owner, kept selected-body realization as dispatcher/construction only, archived the task, and validated with RVV focused tests, generated-bundle dry-runs, MAcc non-regression, and check-tianchenrv 464/464.

### Main Changes

- Added owner-local computed-mask segment2 pre-realized selected-body
  validation APIs to `RVVEmitCSegment2RouteFamilyPlanOwners`.
- Moved computed-mask segment2 legality, ABI-role, mask-policy, segment-field,
  memory-form, dtype/config/policy, selected-variant `requires`, mixed-body,
  and update arithmetic checks out of `RVVSelectedBodyRealization.cpp`.
- Kept central selected-body realization on dispatch and neutral setvl/with_vl,
  compare, mask, load/store, and update materialization.
- Added focused C++ coverage that directly exercises the segment2 owner-local
  validation APIs for the computed-mask segment2 load, store, and update
  selected-boundary routes.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | `rvv: close segment2 computed-mask selected-body handoff` |

### Testing

- [OK] `git diff --check`
- [OK] `ninja -C build tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Generated-bundle dry-run for
  `computed_masked_segment2_load_unit_store`,
  `computed_masked_segment2_store_unit_load`, and
  `computed_masked_segment2_update_unit_load`.
- [OK] Non-segment computed-mask generated-bundle dry-run non-regression for
  `computed_masked_unit_load_store`.
- [OK] Bounded production authority scan and central selected-body validation
  authority scan.
- [OK] `ninja -C build check-tianchenrv` passed 464/464 tests.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 350: Stage2 RVV runtime-scalar memory selected-body owner cleanup

**Date**: 2026-05-31
**Task**: Stage2 RVV runtime-scalar memory selected-body owner cleanup
**Branch**: `main`

### Summary

Moved runtime-scalar splat-store, runtime-scalar computed-mask store, and
runtime-scalar computed-mask load-store selected-body realization ownership out
of central `RVVSelectedBodyRealization.cpp` into a dedicated RVV owner module;
preserved selected-boundary realization, provider facts, common EmitC
neutrality, and target artifact ABI behavior.

### Main Changes

- Added `RVVRuntimeScalarMemorySelectedBodyRealizationOwner.{h,cpp}` and CMake
  wiring.
- Central `RVVSelectedBodyRealization.cpp` now keeps only runtime-scalar memory
  owner registry entries plus shared registry/dispatch mechanics for this
  family.
- Moved splat-store, computed-mask store, and computed-mask load-store
  realization branches into the dedicated owner.
- Added focused C++ coverage for the new owner predicates, owner mismatch/null
  rejection, and selected-boundary materialization for computed-mask
  load-store.
- Checks passed: `TianChenRVRVVPlugin`,
  `tianchenrv-rvv-extension-plugin-test`, focused runtime-scalar lit/generated
  bundle dry-runs, authority scans, `git diff --check`, and
  `cmake --build build --target check-tianchenrv -j2`.

### Git Commits

| Hash | Message |
|------|---------|
| this commit | rvv: move runtime-scalar memory selected-body owner-side |

### Testing

- [OK] Focused C++ and lit/generated-bundle checks passed.
- [OK] `check-tianchenrv` passed 464/464.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 349: Stage2 RVV base memory-movement selected-body owner cleanup

**Date**: 2026-05-31
**Task**: Stage2 RVV base memory-movement selected-body owner cleanup
**Branch**: `main`

### Summary

Moved base-memory selected-body realization into a dedicated
`RVVBaseMemoryMovementSelectedBodyRealizationOwner` module, kept
`RVVSelectedBodyRealization.cpp` on registry and neutral dispatch duty, and
preserved the existing base-memory route-path and fail-closed behavior for the
selected strided, indexed, and masked memory routes.

### Main Changes

- Added `include/TianChenRV/Plugin/RVV/RVVBaseMemoryMovementSelectedBodyRealizationOwner.h`
  and `lib/Plugin/RVV/RVVBaseMemoryMovementSelectedBodyRealizationOwner.cpp`.
- Rewired the selected-body owner registry in
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` to point base-memory
  selection at the new owner module.
- Removed the base-memory realization helpers and branch body from the central
  selected-body file so the remaining code stays on routing/glue and the other
  non-base owner families.
- Updated `lib/Plugin/RVV/CMakeLists.txt` to compile the new owner translation
  unit.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/base-memory-owner-cleanup --run-id strided-load-unit-store --overwrite --op-kind strided_load_unit_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --stride-bytes 4 --stride-bytes 8 --stride-bytes 12 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/base-memory-owner-cleanup --run-id indexed-gather-unit-store --overwrite --op-kind indexed_gather_unit_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/base-memory-owner-cleanup --run-id indexed-scatter-unit-load --overwrite --op-kind indexed_scatter_unit_load --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/base-memory-owner-cleanup --run-id masked-unit-load-store --overwrite --op-kind masked_unit_load_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [OK] direct-pre-realized strided-load/unit-store fail-closed script rejected `--direct-pre-realized-route-entry` as expected
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 464/464 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete

## Session 346: Stage2 RVV computed-mask memory selected-body owner cleanup

**Date**: 2026-05-30
**Task**: Stage2 RVV computed-mask memory selected-body owner cleanup
**Branch**: `main`

### Summary

Moved non-segment computed-mask memory selected-body realization into an
RVV owner-local component while preserving typed `tcrv_rvv` authority and the
existing provider route path. The central selected-body realization file now
keeps registry/neutral dispatch residue for this family, not the family
validation/materialization branches.

### Main Changes

- Added `RVVComputedMaskMemorySelectedBodyRealizationOwner` for unit
  load/store, strided store, strided load, indexed gather, and indexed scatter
  computed-mask memory pre-realized bodies.
- Rewired the computed-mask memory selected-body registry entry to the
  owner-local predicate/hook and fail-closed the central branch helper for this
  family.
- Added C++ production route-path coverage for computed-mask memory owner
  selection, out-of-family rejection, strided/indexed realized typed facts, and
  provider consumption.
- Updated the RVV plugin spec with the durable owner-extraction rule that
  owner-local realization must preserve provider construction role sequence.

### Evidence

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j 8`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- Focused generated-bundle dry-runs: pre-realized computed-mask unit
  load/store, strided store, and indexed gather passed.
- Focused lit materialization checks: pre-realized computed-mask strided load
  and indexed scatter passed.
- Focused non-regression dry-runs: pre-realized standalone reduction add and
  scalar-broadcast elementwise add passed.
- Bounded central scan leaves no non-segment computed-mask memory typed body or
  realization branch in `RVVSelectedBodyRealization.cpp`.
- Added-line authority scan found no new legacy-i32, descriptor,
  source-front-door, artifact-name, or route-id authority.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j 8`: 464/464 passed.

### Self-Repair

- Initial owner-local extraction generated computed-mask passthrough/source
  loads before compare input loads, which broke provider construction-role
  conformance. The owner now preserves canonical role order before provider
  route analysis.

### Status

[OK] **Completed**

### Next Steps

- None - task complete

---

## Session 340: Stage2 RVV base-memory selected-body handoff closure

**Date**: 2026-05-30
**Task**: Stage2 RVV base-memory selected-body provider handoff closure
**Branch**: `main`

### Summary

Moved base-memory pre-realized selected-body validation authority for strided
load/store, indexed gather/scatter, and static-mask load/store into
`RVVEmitCBaseMemoryRouteFamilyPlanOwners`. `RVVSelectedBodyRealization.cpp`
now delegates validation through owner-local APIs and retains selected-body
dispatch plus realized IR materialization.

### Main Changes

- Added owner-local selected-body validation declarations and definitions for
  the five base-memory pre-realized body ops.
- Removed central base-memory validation helper ownership from
  `RVVSelectedBodyRealization.cpp`.
- Added focused C++ coverage proving selected-boundary production code can call
  owner-local base-memory validation.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | `rvv: close base memory selected-body handoff` |

### Testing

- [OK] `git diff --check`
- [OK] `ninja -C build tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Base-memory generated-bundle dry-run for all six affected variants.
- [OK] Direct route-entry negative dry-run failed closed with the expected
  retired-shortcut diagnostic.
- [OK] Contraction generated-bundle dry-run non-regression.
- [OK] `ninja -C build check-tianchenrv` passed 464/464 tests.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 341: Stage2 RVV computed-mask memory handoff

**Date**: 2026-05-30
**Task**: Stage2 RVV computed-mask memory handoff
**Branch**: `main`

### Summary

Moved non-segment computed-mask memory selected-body validation into RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners; central realization now calls owner-local APIs; focused plugin/generated-bundle/check-tianchenrv validation passed.

### Main Changes

- Added `RVVContractionSelectedBodyRealizationOwner` as the owner-local
  selected-body realization module for widening MAcc, widening dot-reduce,
  strided-input widening dot-reduce, computed-mask widening dot-reduce, and
  computed-mask strided-input widening dot-reduce bodies.
- Rewired the central selected-body owner registry to use the owner-local
  contraction predicate/hook and removed contraction validators,
  materialization plans, realized compute builders, and dispatch logic from
  `RVVSelectedBodyRealization.cpp`.
- Preserved typed realized `tcrv_rvv` facts consumed by provider planning:
  `setvl`, `with_vl`, source/stride loads, compare mask, accumulator/seed,
  widening contraction compute, store, ABI order, SEW/LMUL/policy,
  accumulator/result layout, mask/stride facts, and direct contraction provider
  facts.
- Spec-update judgment: no `.trellis/spec/**` edit was needed because this
  implements the existing Stage2 selected-body owner-local contraction contract
  without adding a new durable rule.
- No new runtime, correctness, performance, route, artifact, or executable
  claim was made; no `ssh rvv` run was required for this refactor.

### Git Commits

(No commits - planning session)

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused generated-bundle/direct-entry lit filter: 10/10 passed for
  representative pre-realized widening MAcc, widening dot, strided widening
  dot, computed-mask widening dot, computed-mask strided widening dot, and
  matching direct-route-entry fail-closed probes.
- [OK] bounded central and production authority scans.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 464/464 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 342: Stage2 RVV segment2 computed-mask selected-body provider handoff

**Date**: 2026-05-30
**Task**: Stage2 RVV segment2 computed-mask selected-body provider handoff
**Branch**: `main`

### Summary

Moved computed-mask segment2 selected-body validation into the segment2 route-family owner surface; central realization now delegates validation and retains neutral materialization; focused plugin, generated-bundle, non-regression, authority-scan, and full check-tianchenrv evidence passed.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused segment2/generated-bundle lit filters: 14/14 passed, including
  computed-mask segment2 load/store/update, plain deinterleave/interleave,
  direct pre-realized computed-mask segment2 dry-runs, direct plain segment2
  fail-closed checks, and one base-memory non-regression.
- [OK] bounded authority scans and central segment2 ownership scan.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` (464/464)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 343: Stage2 RVV plain segment2 selected-body provider handoff

**Date**: 2026-05-30
**Task**: Stage2 RVV plain segment2 selected-body provider handoff
**Branch**: `main`

### Summary

Moved plain segment2 deinterleave/interleave selected-body validation into the
segment2 route-family owner surface. Central selected-body realization now
delegates plain segment2 validation and retains owner dispatch plus neutral
realized IR materialization. Focused C++ coverage, generated-bundle dry-runs,
computed-mask segment2 non-regression, authority scans, and full
check-tianchenrv evidence passed.

### Main Changes

- Added owner-local validation APIs for
  `TypedSegment2DeinterleaveMemoryPreRealizedBodyOp` and
  `TypedSegment2InterleaveMemoryPreRealizedBodyOp`.
- Moved plain segment2 op-kind, memory-form, segment-count, field-role,
  source/destination memory-form, runtime ABI role, SEW/LMUL/policy,
  selected-variant `requires`, and mixed-body checks into
  `RVVEmitCSegment2RouteFamilyPlanOwners`.
- Removed central plain segment2 legality predicates and validators from
  `RVVSelectedBodyRealization.cpp`; central code now delegates validation and
  materializes `setvl`/`with_vl`, segment load/store, field load/store, move,
  tuple/store, and erase mechanics only.
- Added focused C++ positive and fail-closed owner-local coverage for plain
  segment2 selected-body validation.

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | `rvv: close plain segment2 selected-body handoff` |

### Testing

- [OK] `git diff --check`
- [OK] `ninja -C build tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Generated-bundle dry-run for `segment2_deinterleave_unit_store`,
  `segment2_interleave_unit_load`,
  `computed_masked_segment2_load_unit_store`,
  `computed_masked_segment2_store_unit_load`, and
  `computed_masked_segment2_update_unit_load`.
- [OK] Bounded central selected-body and touched production-file authority
  scans.
- [OK] `ninja -C build check-tianchenrv` passed 464/464 tests.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 343: Stage2 RVV computed-mask segment2 selected-body migration

**Date**: 2026-05-30
**Task**: Stage2 RVV computed-mask segment2 selected-body migration
**Branch**: `main`

### Summary

Moved computed-mask segment2 load/store/update selected-body realization to the RVV owner boundary, retained route/provider fact flow, and added a direct pre-realized route-entry fail-closed probe.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `515fcc5d` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 344: Stage2 RVV elementwise compare-select selected-body owner cleanup

**Date**: 2026-05-30
**Task**: Stage2 RVV elementwise compare-select selected-body owner cleanup
**Branch**: `main`

### Summary

Moved elementwise and compare/select pre-realized selected-body validation and realization behind an RVV owner-local component while keeping central selected-body realization thin and neutral.

### Main Changes

- Added `RVVElementwiseSelectedBodyRealizationOwner.cpp` as the owner-local realization component for typed elementwise arithmetic, masked arithmetic, scalar-broadcast elementwise, compare/select, computed-mask select, runtime-scalar compare/select, and runtime-scalar dual-compare mask-and-select bodies.
- Kept `RVVSelectedBodyRealization.cpp` as registry/neutral dispatch plus shared mechanics; removed the elementwise/compare-select family-specific validation and realization branch bodies from central code.
- Exported owner-local selected-body APIs through `RVVEmitCElementwiseRouteFamilyPlanOwners.h` so realization ownership sits next to elementwise route-family planning rather than in the central materialization branch.
- Preserved typed selected-boundary facts: op kind, predicate kind, mask source, result layout, scalar/runtime operands, unit/strided/scalar-broadcast memory form, strides, SEW/LMUL/policy, runtime n/AVL/VL, required capabilities, and runtime ABI roles.
- Preserved generated bundle evidence and route facts through dry-run artifacts under `artifacts/tmp/05-30-stage2-rvv-elementwise-compare-select-selected-body-owner-cleanup/owner-elementwise-compare-select` and ssh rvv artifacts under `artifacts/tmp/05-30-stage2-rvv-elementwise-compare-select-selected-body-owner-cleanup/owner-elementwise-compare-select-ssh`.

Testing:
- [OK] `ninja -C build TianChenRVRVVPlugin`
- [OK] focused lit filter for RVV plugin, selected-boundary materialization, generated-bundle elementwise/compare-select dry-runs, direct pre-realized fail-closed tests, and computed-mask segment2 non-regression: 27/27 passed.
- [OK] generated-bundle dry-run for `add`, `masked_add`, `scalar_broadcast_add`, `cmp_select`, `computed_mask_select`, and `runtime_scalar_cmp_select`.
- [OK] `ssh rvv` generated-bundle subset for `add`, `cmp_select`, and `runtime_scalar_cmp_select`, counts 5/17/65, rhs scalars 3 and -4 where applicable.
- [OK] central elementwise/compare-select authority scan and touched-file forbidden-authority scan.
- [OK] `git diff --check`
- [OK] `ninja -C build check-tianchenrv` passed 464/464 tests.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 345: Stage2 RVV standalone-reduction selected-body owner cleanup

**Date**: 2026-05-30
**Task**: Stage2 RVV standalone-reduction selected-body owner cleanup
**Branch**: `main`

### Summary

Moved standalone reduction selected-body validation/realization into an RVV owner-local component, shrank the central selected-body file to registry/neutral dispatch, and validated plain, computed-mask, and runtime-scalar standalone reduction paths with focused C++/lit/generated-bundle/ssh rvv/check-tianchenrv evidence.

### Main Changes

- Added `RVVStandaloneReductionSelectedBodyRealizationOwner` as the reduction-family owner-local selected-body realization component for plain, computed-mask, and runtime-scalar computed-mask standalone reduction bodies.
- Rewired the standalone reduction selected-body registry entry to the owner-local predicate/hook and removed central standalone reduction validators, materializers, and unreachable central family branches from `RVVSelectedBodyRealization.cpp`.
- Added C++ owner-local coverage for standalone reduction owner discovery, retired direct route-entry diagnostics, invalid typed facts, stale runtime `n` role, LMUL config rejection, and outside-family fail-closed behavior.
- Preserved realized typed `tcrv_rvv` facts: `setvl`, `with_vl`, source/compare loads, runtime scalar splat, compare-produced masks, `standalone_reduce` / `masked_standalone_reduce`, scalar-result store, runtime AVL/VL, and required-capability metadata.
- Spec-update judgment: no `.trellis/spec/**` edit was needed because this round implemented the existing Stage2 selected-body owner-local contract without adding a new durable rule.

### Evidence

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- Focused standalone reduction lit filter: 7/7 passed for pre-realized plain, LMUL m2, computed-mask, runtime-scalar computed-mask, and direct-entry fail-closed paths.
- Focused elementwise/compare-select non-regression lit filter: 4/4 passed.
- Generated dry-run bundle:
  `artifacts/tmp/stage2-rvv-standalone-reduction-owner-cleanup/pre-realized-standalone-reduction-owner-dry-run`
- `ssh rvv` generated bundle:
  `artifacts/tmp/stage2-rvv-standalone-reduction-owner-cleanup/pre-realized-standalone-reduction-owner-ssh-rvv`
  with `standalone_reduce_add`, `computed_mask_standalone_reduce_add_lmul_m2`, and `runtime_scalar_cmp_masked_standalone_reduce_add` passing counts `0,1,16,23,257`.
- Bounded central scan leaves only the standalone reduction registry entry and neutral dispatch guard in `RVVSelectedBodyRealization.cpp`.
- Added-line authority scan found no new legacy-i32, source-front-door, descriptor, route-id, direct-route-entry-only, or exact-intrinsic authority.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`: 464/464 passed.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused standalone reduction lit filter: 7/7 passed.
- [OK] focused elementwise/compare-select non-regression lit filter: 4/4 passed.
- [OK] representative dry-run generated bundle in `artifacts/tmp/stage2-rvv-standalone-reduction-owner-cleanup/pre-realized-standalone-reduction-owner-dry-run`.
- [OK] representative `ssh rvv` generated bundle in `artifacts/tmp/stage2-rvv-standalone-reduction-owner-cleanup/pre-realized-standalone-reduction-owner-ssh-rvv`.
- [OK] bounded central and added-line authority scans.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 464/464 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 346: Stage2 RVV computed-mask MAcc selected-body owner cleanup

**Date**: 2026-05-30
**Task**: Stage2 RVV computed-mask MAcc selected-body owner cleanup
**Branch**: `main`

### Summary

Moved computed-mask MAcc selected-body realization into an RVV owner-local component, rewired the central registry to neutral dispatch, preserved typed MAcc route facts, and validated with focused plugin tests, generated-bundle dry-runs, authority scans, and check-tianchenrv 464/464.

### Main Changes

- Added `RVVComputedMaskMAccSelectedBodyRealizationOwner` for vector computed-mask MAcc and runtime-scalar computed-mask MAcc pre-realized bodies.
- Rewired the computed-mask MAcc selected-body registry entry to the owner-local predicate/hook and fail-closed central branch-helper handling for this family.
- Kept `RVVSelectedBodyRealization.cpp` to neutral include/registry/guard residue for computed-mask MAcc; the owner owns validation, runtime AVL/VL construction, typed compare/mask/MAcc materialization, and pre-realized body erasure.
- Preserved mask role/source/memory form, predicate kind, compare inputs, scalar RHS binding, lhs/rhs/acc/out roles, accumulator/result layouts, SEW, LMUL, policy, runtime `n`/AVL/VL, selected requires, and MAcc provider-consumed route facts.
- Updated focused C++ coverage for owner predicate selection and retained existing fail-closed, realized-fact, runtime-scalar, direct-entry rejection, and provider-plan consumption coverage.
- No `.trellis/spec/**` update was needed because this implements the existing RVV selected-body owner-local and neutral EmitC contracts.
- No new runtime, correctness, or performance claim was made; no `ssh rvv` run was required for this refactor.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused generated-bundle dry-run lit filter: 7/7 passed for computed-mask MAcc, runtime-scalar computed-mask MAcc, and computed-mask memory non-regression cases.
- [OK] focused dialect/target lit filter: 6/6 passed for computed-mask MAcc negative/dataflow/materialization and computed-mask memory non-regression cases.
- [OK] central selected-body scan found no computed-mask MAcc typed-body/helper residue in `RVVSelectedBodyRealization.cpp`.
- [OK] production and added-line authority scans found no new legacy-i32, descriptor, source-front-door, route-id, direct-route-entry-only, common-EmitC semantic, or exact-intrinsic authority.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 464/464 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 347: Stage2 RVV non-computed MAcc selected-body owner cleanup

**Date**: 2026-05-31
**Task**: Stage2 RVV non-computed MAcc selected-body owner cleanup
**Branch**: `main`

### Summary

Moved non-computed MAcc selected-body realization into an RVV owner-local component, kept central selected-body dispatch neutral, preserved MAcc provider facts, and validated focused MAcc/generated-bundle checks plus check-tianchenrv 464/464.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 348: Stage2 RVV contraction selected-body owner cleanup

**Date**: 2026-05-31
**Task**: Stage2 RVV contraction selected-body owner cleanup
**Branch**: `main`

### Summary

Moved contraction selected-body realization into RVVContractionSelectedBodyRealizationOwner, kept central selected-body dispatch neutral, preserved contraction provider facts, and validated focused C++/generated-bundle checks plus check-tianchenrv 464/464.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 349: Stage2 RVV segment2 memory selected-body owner cleanup

**Date**: 2026-05-31
**Task**: Stage2 RVV segment2 memory selected-body owner cleanup
**Branch**: `main`

### Summary

Moved segment2 selected-body realization ownership out of the central RVV selected-body file into a dedicated owner module; preserved segment2 provider route facts and generated-bundle behavior.

### Main Changes

- Added `RVVSegment2MemorySelectedBodyRealizationOwner.{h,cpp}` and CMake wiring.
- Central `RVVSelectedBodyRealization.cpp` now keeps only segment2 owner registration/dispatch glue.
- Plain segment2 deinterleave/interleave realization moved to the dedicated owner; computed-mask segment2 load/store/update route through the owner while preserving existing segment2 validation/realization helpers.
- Added C++ coverage that the dedicated owner predicate recognizes plain and computed-mask segment2 pre-realized bodies.
- Checks passed: `tianchenrv-rvv-extension-plugin-test`, focused segment2 generated-bundle lit tests, direct pre-realized computed-mask segment2 dry-runs, authority scans, `git diff --check`, and `cmake --build build --target check-tianchenrv -j2`.


### Git Commits

| Hash | Message |
|------|---------|
| this commit | rvv: move segment2 memory selected-body owner-side |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
