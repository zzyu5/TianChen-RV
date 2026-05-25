# Stage2 RVV Deferred Selected-Body Realization Owner-Local Hooks

## Goal

Retire the shared existing-family selected-body realization helper as active realization authority for the remaining deferred non-route-entry RVV selected-body realization owners. Each migrated owner must be selected by the registry, validate its own typed pre-realized body facts, and materialize realized `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` / body operations before route-control planning and route provider construction.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created from the Hermes Direction Brief.
* Current HEAD is `9bc4f03a rvv: add owner-local selected-body realization hooks`; `git status --short` was clean before this task was created.
* The archived predecessor task `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-selected-body-realization-owner-local-hooks/` migrated `standalone reduction`, `MAcc`, and `base memory movement` to owner-local realization hooks.
* Current `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` confirms the remaining deferred owners still point at `realizePreRealizedRVVExistingFamilyOwner`, which calls `realizePreRealizedRVVSelectedBodyWithExistingFamilyBranches`.
* The selected-body owner registry is upstream of route-family analysis, route-control provider plans, and `TCRVEmitCLowerableRoute` construction.
* Specs require RVV plugin-owned typed body/config/runtime/capability validation and forbid route ids, artifact names, ABI strings, descriptors, common EmitC, source-front-door markers, scripts, metadata, or legacy i32 names as realization authority.

## Scope For This Round

Migrate the remaining deferred selected-body realization owners away from the shared existing-family hook:

* `runtime scalar splat-store`
* `runtime scalar computed-mask store`
* `runtime scalar computed-mask load-store`
* `reduction`
* `computed-mask MAcc`
* `contraction`
* `widening conversion`
* `computed-mask memory`
* `segment2 memory`

If repository evidence shows all nine cannot be safely completed in one round, finish one coherent deferred cluster and explicitly list the owners intentionally left on the shared helper.

## Requirements

* Registry entries for migrated owners must point to owner-local realization hooks, not `realizePreRealizedRVVExistingFamilyOwner`.
* Each owner-local hook must fail closed when called with a body outside its owner family.
* Each migrated hook must preserve existing validators before materialization, including checks for op kind, memory form, runtime AVL/VL source, ABI role/order, mem_window/imported value role, typed config, SEW/LMUL, policy, mask/passthrough facts, accumulator/result layout, selected capability/`requires`, and mixed realized/pre-realized body rejection where applicable.
* Each migrated hook must materialize the same realized `tcrv_rvv` operations and preserve runtime ABI, dispatch/fallback behavior, emitted statement order, and target artifact semantics.
* Explicit already-realized selected bodies must remain unaffected.
* Realized bodies must continue to feed existing route-control owner registry and route/provider paths.
* Unsupported or mismatched pre-realized bodies must fail closed before route/artifact authority.
* The shared existing-family helper must not remain active realization authority for migrated owners.

## Acceptance Criteria

* [x] All migrated registry owners use owner-local hook pointers.
* [x] No registry owner points at `realizePreRealizedRVVExistingFamilyOwner`.
* [x] Owner-local negative tests cover non-owned body dispatch and representative invalid typed facts for a math/accumulation owner and a runtime/memory/mask owner.
* [x] Focused registry tests prove route-entry eligibility remains narrower than general realization support.
* [x] Representative tcrv-opt/tcrv-translate dry-run evidence passes for one math/accumulation family and one runtime/memory/mask family.
* [x] A representative explicit selected-body artifact still passes unchanged.
* [x] Bounded scans over touched files find no new name-, metadata-, descriptor-, ABI-string-, script-, artifact-, common-EmitC-, source-front-door-, or legacy-i32-derived realization or route authority.
* [x] `git diff --check` passes.
* [x] Focused RVV plugin/lit checks pass.
* [x] `check-tianchenrv` passes.

## Out Of Scope

* Adding new RVV route coverage, operation families, dtype/LMUL clone batches, frontend lowering, source-front-door positive routes, common EmitC semantics, target artifact semantics, dashboards, or broad smoke matrices.
* Changing computation semantics, runtime ABI, dispatch/fallback behavior, emitted statement order, or performance claims.
* Treating route ids, artifact metadata, test names, ABI strings, descriptor residue, source-front-door metadata, common EmitC code, scripts, or legacy i32 names as realization or route authority.
* Running `ssh rvv` unless executable semantics change; this task should preserve existing realized-body semantics.

## Technical Notes

* Specs read:
  * `.trellis/spec/index.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
  * `.trellis/spec/extension-plugins/index.md`
* Predecessor task read:
  * `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-selected-body-realization-owner-local-hooks/prd.md`
  * `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-selected-body-realization-owner-local-hooks/task.json`
* Primary production files inspected:
  * `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  * `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`
* Primary test surfaces inspected:
  * `test/Plugin/RVVExtensionPluginTest.cpp`
  * `test/Target/RVV/explicit-selected-body-artifact-*.mlir`
  * `test/Target/RVV/pre-realized-selected-body-artifact-*.mlir`

## Definition Of Done

* Trellis task context remains truthful.
* Remaining deferred selected-body realization owners have owner-local registry hooks.
* Realized body to route-control/provider evidence remains intact.
* Focused checks pass and failures are self-repaired.
* Workspace journal records the result.
* One coherent commit is created if complete.

## Completion Notes

* Migrated owner-local hooks: `runtime scalar splat-store`, `runtime scalar computed-mask store`, `runtime scalar computed-mask load-store`, `reduction`, `computed-mask MAcc`, `contraction`, `widening conversion`, `computed-mask memory`, and `segment2 memory`.
* Previously migrated route-entry-capable hooks remain owner-local: `standalone reduction`, `MAcc`, and `base memory movement`.
* `realizePreRealizedRVVExistingFamilyOwner` and `realizePreRealizedRVVSelectedBodyWithExistingFamilyBranches` are no longer present as active registry realization authority.
* The remaining shared implementation branch is private owner-local materialization plumbing: it takes the already selected `bodyOp` from a registry-selected hook and does not rediscover or classify the owner/body. The registry-selected hook performs the owner-family gate before materialization.
* C++ registry tests now require distinct owner-local hook pointers for all thirteen selected-body realization owners and preserve route-entry eligibility as an owner-scoped subset.
* Added runtime scalar splat-store owner-local negative coverage for null body dispatch, non-owned body dispatch, and wrong `op_kind`, plus successful owner-local realization to `setvl`/`with_vl`.
* Continuation retry self-repair: owner-local materialization wrappers now fail closed on null body before calling `llvm::isa`-based owner predicates, avoiding debug-build predicate assertions and keeping the diagnostic in the owner-local boundary.
* Representative dry-runs passed for computed-mask MAcc, runtime scalar splat-store, and an explicit runtime scalar splat-store selected body.
* No `ssh rvv` rerun was needed because this task changed selected-body realization ownership only; emitted body semantics, runtime ABI, dispatch/fallback behavior, and target artifact semantics were preserved.
