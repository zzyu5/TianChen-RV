# Stage2 RVV elementwise/compare-select selected-body owner interface split

## Goal

Split the elementwise/compare-select selected-body realization owner API out
of `RVVEmitCElementwiseRouteFamilyPlanOwners.h` into a dedicated RVV owner
header, while preserving existing selected-body realization behavior and
keeping EmitC route-family planning focused on route-plan derivation,
application, validation, mirror checks, and operand binding.

The production boundary must remain:

```text
selected tcrv.exec RVV variant
  -> elementwise/compare-select selected-body owner predicate and realization hook
  -> realized typed tcrv_rvv body
  -> separate EmitC route-family planning/provider interface
  -> TCRVEmitCLowerableRoute
  -> neutral EmitC/materialization
```

## Direction Source

- Direction title: `Switch: Stage2 RVV elementwise/compare-select selected-body owner interface split`.
- Module owner: RVV elementwise/compare-select selected-body realization
  owner interface.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- `.trellis/.current-task` was absent at session start, so this task was
  created from the supplied Hermes Direction Brief.

## Current Repository Facts

- Session start HEAD was
  `46e32fc3 rvv: move widening conversion selected-body owner-side`, with a
  clean worktree.
- The archived widening-conversion cleanup moved its owner predicate and
  realization hook into `RVVWideningConversionSelectedBodyRealizationOwner.h`
  / `.cpp`; the central selected-body realization file now includes that owner
  header and keeps only registry/dispatch residue for that family.
- `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp` already owns
  elementwise/compare-select predicate, validation, materialization, and
  fail-closed diagnostics, but it includes
  `RVVEmitCElementwiseRouteFamilyPlanOwners.h`.
- `include/TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h`
  currently mixes selected-body owner declarations with route-family planning
  declarations:
  `RVVElementwiseCompareSelectRealizationResult`,
  `isPreRealizedRVVElementwiseCompareSelectClusterOp`,
  `realizePreRealizedRVVElementwiseCompareSelectCluster`,
  `realizePreRealizedRVVElementwiseCompareSelectOwner`, and
  `realizePreRealizedRVVElementwiseCompareSelectSelectedBody` are selected-body
  owner APIs, not EmitC route-planning APIs.
- `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h` pulls the
  route-family planning header only to expose
  `variantContainsPreRealizedRVVElementwiseCompareSelectSelectedBody`, creating
  an avoidable selected-body registry dependency on a route-planning header.
- `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp` contains
  route-plan derivation, plan validation, mirror validation, intrinsic/header
  payload planning, and operand-binding declarations. Those declarations should
  stay in the route-family planning interface.

## Requirements

1. Add or repair a dedicated
   `RVVElementwiseSelectedBodyRealizationOwner.h` interface for
   elementwise/compare-select selected-body realization owner APIs.
2. Move selected-body owner declarations and owner-local result types out of
   `RVVEmitCElementwiseRouteFamilyPlanOwners.h` unless a declaration is
   genuinely route-planning API.
3. Make `RVVElementwiseSelectedBodyRealizationOwner.cpp` include the dedicated
   owner header directly.
4. Make `RVVSelectedBodyRealization.cpp` include the dedicated owner header for
   the elementwise/compare-select owner registry entry.
5. Make public selected-body realization callers include the owner header when
   they use elementwise/compare-select owner APIs, rather than relying on the
   route-planning header as a carrier.
6. Keep route-family planning declarations in
   `RVVEmitCElementwiseRouteFamilyPlanOwners.h`: route operation predicates,
   route-family consumer predicates, route-plan derivation, route-plan
   application, route-plan validation, route description mirror validation,
   and operand-binding helpers.
7. Preserve selected-body owner registry ordering where semantically required,
   especially the first `elementwise/compare-select` owner entry.
8. Preserve existing elementwise arithmetic, masked elementwise,
   compare-select, computed-mask select, runtime-scalar compare/select, and
   dual runtime-scalar compare-mask-and-select selected-body behavior and
   fail-closed diagnostics.
9. Do not change route/executable semantics, provider route facts, common
   EmitC behavior, target artifacts, runtime ABI roles, or performance claims.
10. If a declaration is shared by both selected-body realization and route
    planning, make the ownership boundary explicit with includes rather than
    leaving selected-body ownership implicit in the route-planning header.

## Acceptance Criteria

- [x] `RVVElementwiseSelectedBodyRealizationOwner.h` exists and exposes the
      elementwise/compare-select selected-body owner predicate, owner hook,
      selected-body helper, variant predicate, and realization result type.
- [x] `RVVEmitCElementwiseRouteFamilyPlanOwners.h` no longer declares
      selected-body owner predicates, owner hooks, owner-local realization
      results, or selected-body realization helpers; it keeps only route-family
      planning and mirror/operand-binding APIs.
- [x] `RVVSelectedBodyRealization.cpp` includes the dedicated elementwise owner
      header directly and still registers exactly one explicit
      `elementwise/compare-select` owner entry with the existing predicate and
      realization hook.
- [x] `RVVSelectedBodyRealization.h` no longer depends on the elementwise
      route-family planning header for selected-body owner declarations.
- [x] Public callers/tests that use elementwise selected-body realization APIs
      include the dedicated owner header explicitly.
- [x] Existing selected-body realization behavior remains unchanged for
      elementwise arithmetic, masked elementwise, compare-select,
      computed-mask select, runtime-scalar compare/select, and dual
      runtime-scalar compare-mask-and-select cases.
- [x] Existing route-family planning behavior and mirror validation remain
      unchanged; route planning still owns plan derivation/application/
      validation and operand binding only.
- [x] C++ plugin tests covering elementwise arithmetic, masked elementwise,
      compare-select, computed-mask select, runtime-scalar compare/select,
      selected-body owner registry selection, and fail-closed wrong-family or
      invalid-body cases still pass.
- [x] Focused build targets `tianchenrv-rvv-extension-plugin-test`,
      `tcrv-opt`, and `tcrv-translate` pass if configured in this build.
- [x] `git diff --check` passes.
- [x] A bounded touched-file authority scan shows no new descriptor-,
      source-front-door-, artifact-name-, route-id-, exact-intrinsic-,
      legacy-i32-, ABI-string-, metadata-, script-, common-EmitC-, or
      direct-route-entry-derived selected-body realization authority.
- [x] `check-tianchenrv` passes, or the exact blocker is recorded.
- [x] Task status/context/journal are truthful; task is finished/archived and
      a coherent commit is created if acceptance passes.

## Completion Evidence

- Added `RVVElementwiseSelectedBodyRealizationOwner.h` as the selected-body
  owner interface for elementwise/compare-select predicates, realization hooks,
  result type, selected-body helper, and variant predicate.
- Cleaned `RVVEmitCElementwiseRouteFamilyPlanOwners.h` so it no longer exports
  selected-body owner APIs and continues to expose route-family planning,
  mirror validation, and operand-binding APIs only.
- Updated `RVVSelectedBodyRealization.cpp`, the elementwise owner source, and
  plugin tests to include the dedicated owner header directly.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the durable
  owner-header locality contract.
- Checks passed:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`,
  `./build/bin/tianchenrv-rvv-extension-plugin-test`,
  `cmake --build build --target tcrv-opt -j2`,
  `cmake --build build --target tcrv-translate -j2`,
  `git diff --check`, bounded owner/API and authority scans, and
  `cmake --build build --target check-tianchenrv -j2` with 464/464 tests.

## Out of Scope

- No new RVV op coverage, dtype/LMUL clone batches, compare/select variants,
  broadcast variants, reduction/conversion/memory/contraction work, or
  route table growth.
- No source/front-end lowering, descriptor-driven computation, direct-route
  compatibility path, common EmitC semantic branch, target artifact authority,
  performance tuning, dashboard/report work, or broad evidence-only task.
- No rewrite of the selected-body owner registry framework.
- No movement of already-clean owner families unless a compile-time dependency
  proves they are part of this elementwise/compare-select boundary.
- No hardware runtime/correctness/performance claim; this is an interface
  boundary refactor and should not require `ssh rvv` evidence.

## Validation Plan

1. Run focused include/API scans to verify selected-body declarations moved
   out of the route-family planning header and public callers include the
   dedicated owner header.
2. Build/run `tianchenrv-rvv-extension-plugin-test` and ensure focused
   elementwise/compare-select selected-body and owner-registry tests pass.
3. Build `tcrv-opt` and `tcrv-translate` if available in the configured build.
4. Run `git diff --check`.
5. Run a bounded authority scan over touched production files and the central
   selected-body registry.
6. Run `check-tianchenrv` when focused checks are green, or record the exact
   blocker.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Immediate precedent read:
  `.trellis/tasks/archive/2026-05/05-31-05-31-stage2-rvv-widening-conversion-selected-body-owner-cleanup/prd.md`.
- Relevant production files inspected:
  `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.
- No blocking user question remains; the supplied direction brief is specific
  enough to implement.
