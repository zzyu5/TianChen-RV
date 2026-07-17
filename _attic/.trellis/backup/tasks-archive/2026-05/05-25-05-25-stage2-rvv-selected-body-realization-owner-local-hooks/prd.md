# Stage2 RVV Selected-Body Realization Owner-Local Hooks

## Goal

Replace the shared non-elementwise pre-realized RVV selected-body realization helper for the route-entry-capable non-elementwise owner families with owner-local validation and materialization hooks. The migrated hooks must consume typed pre-realized body facts into realized `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` structure before route-control planning and route provider construction.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created from the Hermes Direction Brief.
* Current HEAD is `0de0ee90 rvv: add selected-body realization owner registry`, and `git status --short` was clean before this task was created.
* The archived predecessor task `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-selected-body-realization-owner-registry/` added `RVVSelectedBodyRealizationOwner`, registry-backed selected-body discovery, route-entry discovery, and negative route-entry rejection.
* The predecessor task explicitly left this continuation point: non-elementwise owners still dispatch through `realizePreRealizedRVVExistingFamilyOwner(...)` into `realizePreRealizedRVVSelectedBodyWithExistingFamilyBranches(...)`.
* Current code confirms that `elementwise/compare-select` already uses `realizePreRealizedRVVElementwiseCompareSelectOwner(...)`, while all non-elementwise registry entries still point at the shared existing-family hook.
* The route-entry-capable non-elementwise registry entries are `standalone reduction`, `MAcc` (route-entry only for scalar-broadcast MAcc), and `base memory movement`.
* Relevant specs require selected-body realization owner selection to happen before route-family analysis, route-control provider plans, and `TCRVEmitCLowerableRoute` construction. Common EmitC and target artifacts must remain mirror/materialization consumers, not RVV semantic owners.

## Scope For This Round

This round migrates the route-entry-capable non-elementwise selected-body realization owners:

* `standalone reduction`
* `MAcc`
* `base memory movement`

Each migrated owner gets an owner-local realization hook that validates and materializes only its owned body classes after registry owner selection. The shared existing-family helper may remain only for intentionally deferred non-route-entry owners and must not remain a hidden realization path for migrated owners.

## Deferred Continuation Point

The following non-elementwise owners are intentionally deferred if this round completes the route-entry-capable submodule:

* `runtime scalar splat-store`
* `runtime scalar computed-mask store`
* `runtime scalar computed-mask load-store`
* `reduction`
* `computed-mask MAcc`
* `contraction`
* `widening conversion`
* `computed-mask memory`
* `segment2 memory`

The next continuation owner should migrate these remaining registry families away from `realizePreRealizedRVVExistingFamilyOwner(...)`, preferably by family cluster, without adding new route coverage.

## Requirements

* Add owner-local realization hooks for standalone reduction, MAcc, and base memory movement.
* Each migrated hook must reject a body outside its owner family with a targeted diagnostic.
* Each migrated hook must preserve the existing family validators before materialization, including checks for op kind, memory form, runtime AVL/VL source, ABI roles/order, typed config, SEW/LMUL, policy, mask/passthrough facts, accumulator/result layout, selected capability/`requires`, and mixed realized/pre-realized body rejection where applicable.
* Each migrated hook must materialize the same realized `tcrv_rvv` operations and preserve runtime ABI, dispatch/fallback behavior, emitted statement order, and target artifact semantics.
* `realizePreRealizedRVVSelectedBody(...)` and `realizePreRealizedRVVRouteEntrySelectedBody(...)` must dispatch migrated bodies directly through the migrated owner hook.
* The shared existing-family helper must no longer positively realize standalone reduction, MAcc, or base memory movement bodies.
* Explicit already-realized selected bodies must remain unaffected.
* Realized bodies from migrated route-entry families must still feed route-control owner registry and route/provider evidence.
* No new RVV operation families, dtype/LMUL clone batches, frontend lowering, source-front-door positive routes, common EmitC semantics, or target artifact semantics may be added.

## Acceptance Criteria

* [x] The selected-body realization owner registry points standalone reduction, MAcc, and base memory movement at owner-local hooks rather than `realizePreRealizedRVVExistingFamilyOwner(...)`.
* [x] Each migrated owner hook handles all body op classes covered by that owner predicate and fails closed for non-owned body classes.
* [x] The shared existing-family helper rejects the migrated owner families before shared-branch realization instead of retaining them as reachable positive realization authority.
* [x] Focused C++ tests prove migrated owner names use distinct hook pointers and that the deferred owners still point at the shared continuation hook.
* [x] Negative tests cover wrong/missing family facts through migrated owner-local hooks, including memory_form/op_kind pairing, runtime AVL/ABI role mismatch, config mismatch, and a non-owned body passed to a migrated hook.
* [x] Representative route-entry evidence passes for migrated base memory, standalone reduction, and scalar-broadcast MAcc route-entry families.
* [x] A representative explicit selected-body fixture still passes unchanged.
* [x] Bounded scans over touched files find no new name-, metadata-, descriptor-, ABI-string-, script-, artifact-, common-EmitC-, source-front-door-, or legacy-i32-derived realization/route authority.
* [x] `git diff --check` passes.
* [x] Focused RVV plugin/lit checks and full `check-tianchenrv` pass.

## Out Of Scope

* Migrating all remaining non-route-entry non-elementwise owners in this round.
* Adding new RVV route coverage, new operation families, new dtypes, LMUL clone batches, or frontend lowering.
* Changing computation semantics, runtime ABI, dispatch/fallback behavior, emitted statement order, common EmitC semantics, target artifact semantics, or performance claims.
* Treating route ids, artifact metadata, test names, ABI strings, descriptor residue, source-front-door metadata, common EmitC code, scripts, or legacy i32 names as realization or route authority.

## Technical Notes

* Specs read:
  * `.trellis/spec/index.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
  * `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  * `.trellis/spec/testing/mlir-testing-contract.md`
* Predecessor task read:
  * `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-selected-body-realization-owner-registry/prd.md`
  * `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-selected-body-realization-owner-registry/implement.jsonl`
  * `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-selected-body-realization-owner-registry/check.jsonl`
* Primary production files inspected:
  * `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`
  * `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  * `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  * `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* Primary test surfaces inspected:
  * `test/Plugin/RVVExtensionPluginTest.cpp`
  * `test/Target/RVV/pre-realized-selected-body-artifact-*.mlir`
  * `test/Target/RVV/explicit-selected-body-artifact-*.mlir`

## Definition Of Done

* Trellis task context remains truthful.
* Migrated route-entry-capable non-elementwise selected-body realization owners use owner-local hooks.
* Realized body to route-control/provider evidence remains intact.
* Focused quality checks pass.
* Workspace journal records the result.
* One coherent commit is created if complete.

## Completion Notes

* Migrated owner-local hooks: `standalone reduction`, `MAcc`, and `base memory movement`.
* Intentionally deferred shared-helper continuation owners: `runtime scalar splat-store`, `runtime scalar computed-mask store`, `runtime scalar computed-mask load-store`, `reduction`, `computed-mask MAcc`, `contraction`, `widening conversion`, `computed-mask memory`, and `segment2 memory`.
* The migrated hooks reuse the existing typed family validators before materialization and then materialize the same realized `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` operations, preserving route-control registry and route provider evidence.
* The shared existing-family helper now fails closed if asked to own the migrated route-entry-capable families; deferred owners remain the explicit continuation point.
* No `.trellis/spec` update was needed because `.trellis/spec/extension-plugins/rvv-plugin.md` already defines the selected-body owner registry, owner-local realization, route-entry bridge, and route-control registry contracts implemented here.
