# Stage2 RVV computed-mask MAcc selected-body owner cleanup

## Goal

Move active computed-mask MAcc selected-body realization out of the central
`RVVSelectedBodyRealization.cpp` owner into a dedicated RVV plugin-local owner
component, preserving the current typed selected-boundary and provider route
facts for both `TypedComputedMaskMAccPreRealizedBodyOp` and
`TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp`.

## What I already know

* The repository is clean on `main`; recent commits extracted computed-mask
  memory, standalone reduction, elementwise, and segment2 selected-body owners.
* The brief identifies `RVVSelectedBodyRealization.cpp` as still directly
  owning computed-mask MAcc predicate and realization branches.
* Existing owner-local extraction pattern should be taken from
  `RVVComputedMaskMemorySelectedBodyRealizationOwner`.
* The target architecture remains:
  selected `tcrv.exec` RVV variant -> typed pre-realized `tcrv_rvv` body ->
  realized typed mask/MAcc facts -> MAcc provider route facts ->
  `TCRVEmitCLowerableRoute` -> neutral EmitC materializer.

## Requirements

* Add a dedicated owner-local computed-mask MAcc selected-body realization
  component with a clear predicate/hook boundary.
* The owner must recognize, validate, and realize both
  `TypedComputedMaskMAccPreRealizedBodyOp` and
  `TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp`.
* Preserve typed mask role/source/memory form, predicate kind, compare inputs,
  scalar RHS binding where present, lhs/rhs/acc/out roles, accumulator/result
  layouts, SEW, LMUL, policy, runtime `n`/AVL/VL, selected requires, and
  provider-consumed MAcc route facts.
* Leave `RVVSelectedBodyRealization.cpp` with only neutral registry, dispatch,
  and glue for computed-mask MAcc realization.
* Keep common EmitC/export neutral; no RVV semantic choice moves into common
  lowering.
* Fail closed for wrong owner body, missing kernel/variant, stale runtime ABI
  binding, wrong mask binding, wrong scalar role, layout mismatch,
  dtype/config/policy mismatch, stale route id/metadata,
  direct-route-entry-only authority, exact-intrinsic-as-authority, or
  common-EmitC semantic choice.

## Acceptance Criteria

* [x] New header/source for computed-mask MAcc selected-body owner exists and is
  wired into CMake/registry.
* [x] Central selected-body realization no longer owns computed-mask MAcc helper
  logic beyond neutral dispatch/glue.
* [x] Focused C++ tests prove unique owner selection, wrong-owner fail-closed
  behavior, realized typed `tcrv_rvv` facts, runtime-scalar mask/MAcc facts,
  and MAcc provider-plan consumption.
* [x] Focused generated-bundle dry-runs cover existing computed-mask MAcc add
  and runtime-scalar computed-mask MAcc cases.
* [x] Computed-mask memory owner extraction does not regress.
* [x] Authority scans show no route/executable claim depends on central ad hoc,
  name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
  script-derived, artifact-name-derived, common-EmitC-derived,
  source-front-door-derived, route-id-derived, exact-intrinsic-derived,
  direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
  authority.
* [x] `git diff --check` passes.
* [x] `check-tianchenrv` passes, or an exact blocker is recorded.

## Out of Scope

* No route semantic changes or new MAcc operation coverage.
* No re-opening completed computed-mask memory, elementwise, reduction, segment,
  or contraction work.
* No source-front-door positive route, dtype/LMUL matrix expansion, wrapper-only
  abstraction, selected-body registry rewrite, broad smoke matrix, or new
  runtime/correctness/performance claim.
* No RVV semantic migration into common EmitC/export code.

## Completion Evidence

* Added `RVVComputedMaskMAccSelectedBodyRealizationOwner` as the owner-local
  computed-mask MAcc selected-body realization component.
* Rewired the computed-mask MAcc selected-body registry entry to the
  owner-local predicate/hook and left `RVVSelectedBodyRealization.cpp` with
  neutral include/registry/guard residue for this family.
* Preserved typed mask, predicate, compare-input, runtime-scalar, accumulator,
  result, SEW, LMUL, policy, runtime AVL/VL, selected-requires, and
  provider-consumed route facts through owner-local realization.
* Verified focused C++ plugin coverage, computed-mask MAcc generated-bundle
  dry-runs, runtime-scalar computed-mask MAcc dry-runs, computed-mask memory
  non-regression, authority scans, `git diff --check`, and full
  `check-tianchenrv` 464/464.
* No `.trellis/spec/**` update was required: the existing RVV plugin,
  `tcrv.exec`, and EmitC-route specs already define the owner-local boundary
  and neutral common EmitC contract implemented here.
* No new runtime, correctness, or performance claim was made; no `ssh rvv`
  evidence was required for this owner-local refactor.

## Technical Notes

* Read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
* Relevant prior task:
  `.trellis/tasks/archive/05-30-stage2-rvv-computed-mask-memory-selected-body-owner-cleanup`
  or current archive equivalent, if present.
* Relevant implementation files:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  focused generated-bundle tests under `test/Scripts`.
