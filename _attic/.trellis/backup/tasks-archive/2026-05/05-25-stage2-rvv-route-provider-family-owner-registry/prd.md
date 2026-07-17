# Stage2 RVV route-provider family owner registry

## Goal

Make existing RVV EmitC route construction family-owned at the production route-provider boundary. The RVV provider should select exactly one owner for each route-supported realized typed `tcrv_rvv` body and construct `TCRVEmitCLowerableRoute` only from typed body/config/capability/runtime facts plus the RVV route-control plan.

## What I already know

* Current HEAD is `6b1073c3 rvv: add deferred selected-body owner-local hooks`; the worktree was clean before this task was created.
* There was no `.trellis/.current-task`, so this task was created from the Hermes direction brief.
* The previous owner-local selected-body realization work moved the upstream selected-body realization boundary to owner-selected hooks.
* The next bottleneck is downstream RVV EmitC route construction. The brief says `RVVEmitCRouteProvider.cpp` and `RVVEmitCRoutePlanning.cpp` still centralize operation-family predicates, switches, and route construction branches.
* Stage2 must keep the authority chain: selected `tcrv.exec` RVV variant -> realized typed `tcrv_rvv` body -> RVV plugin-owned route provider -> `TCRVEmitCLowerableRoute` -> common EmitC materializer.

## Context Read

* `.trellis/spec/index.md` confirms the current RVV path is selected `tcrv.exec` variant -> typed low-level `tcrv_rvv` body -> RVV plugin-owned legality/realization/route provider -> `TCRVEmitCLowerableRoute` -> common EmitC.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires route-family ownership to be explicit and the top-level provider to consume owner registries instead of manually sequencing family verifiers or statement-plan getters.
* `.trellis/spec/lowering-runtime/emitc-route.md` keeps common EmitC neutral; RVV type/header/intrinsic/statement choices stay plugin-owned.
* Archived route-control registry work completed `RVVSelectedBodyRouteControlProviderOwner` and exact-once route-control ownership for thirteen adopted families.
* Archived selected-body realization work completed owner-local realization hooks for all thirteen selected-body realization owners.
* Current `RVVEmitCRouteProvider.cpp` already calls `getRVVSelectedBodyMigratedRouteStatementPlan(...)`, but current `RVVEmitCRoutePlanning.cpp` still builds that aggregate by manually calling each migrated family statement-plan getter in sequence.
* Current direct-provider contraction construction still lives in the central provider body. Moving all contraction statement construction out is larger than this one coherent registry slice.

## Assumptions

* This round should not add new RVV operation coverage or change computation semantics.
* This round will migrate the production statement-plan provider consumption cluster to a registry-selected owner boundary. Direct-provider contraction construction remains a named continuation unless it can be extracted safely without broad rewrites.
* The task does not claim new runtime/correctness/performance behavior, so `ssh rvv` is not required unless emitted target sequence, ABI, or semantics change.

## Requirements

* Add a production RVV route-provider owner registry or equivalent owner table for the migrated statement-plan provider-consumption boundary.
* Ensure each migrated statement-plan route family is represented by exactly one owner, and unrelated/unmigrated realized bodies receive an empty plan or fail closed only when they are marked migrated consumers.
* Owner selection and route construction must be based on realized typed `tcrv_rvv` body/config/materialization facts, selected capability, runtime AVL/VL and operand bindings, plus route-control facts already validated by the owning family plan.
* Do not use route ids, operation-name strings, artifact names, descriptor residue, ABI strings, test names, scripts, common EmitC semantics, metadata, source-front-door state, or legacy i32 helpers as route authority.
* Preserve existing emitted statement order and selected-body artifact behavior unless a change is required to preserve owner semantics.
* Name any remaining central route-provider authority explicitly in the task notes/final report.

## Acceptance Criteria

* [x] Relevant specs and archived predecessor tasks are read and reflected in this PRD before implementation.
* [x] Production C++ route planning/provider code has an owner-registry boundary used by the default RVV route construction path for migrated statement-plan provider consumption.
* [x] The migrated statement-plan cluster covers the existing migrated families exactly once: elementwise arithmetic, compare/select, widening conversion, standalone reduction, plain MAcc, base memory movement, computed-mask memory, segment2 memory, and computed-mask accumulation.
* [x] Tests cover unique owner selection, empty-plan non-consumer behavior, and fail-closed missing/stale dependencies through the migrated owner boundary.
* [x] Tests cover bad runtime AVL/VL, bad operand binding, bad memory form/op kind/config/policy/capability where existing migrated family plan tests already represent those cases.
* [x] Representative `tcrv-opt`/`tcrv-translate` or generated-bundle dry-runs pass for at least one route-entry/direct-provider family and one migrated statement-plan family.
* [x] Explicit selected-body artifact behavior remains unchanged.
* [x] A bounded scan over touched files shows no authority leaks from legacy i32/source-front-door/descriptor/ABI/artifact/script/common-EmitC/metadata/route-id sources.
* [x] `git diff --check` passes.
* [x] `check-tianchenrv` passes, or an exact blocker is recorded.

## Continuation If This Slice Completes

* Extract direct-provider contraction statement construction from the central `RVVEmitCRouteProvider.cpp` body into a route-provider owner, while preserving its route-control provider plan and math operand-binding checks.
* If residual un-migrated fallback routes still require central provider-local statement assembly, name them explicitly and do not count them as migrated owner-registry coverage.

## Definition of Done

* Task context files are real and no longer contain placeholder JSONL entries.
* Implementation and checks are complete for the chosen coherent owner-registry slice.
* Trellis task status and journal are truthful.
* Completed work is committed as one coherent commit, or the unfinished continuation point is explicit.

## Out of Scope

* New RVV operation coverage, dtype/LMUL clone expansion, source-front-door positive routes, frontend lowering, high-level Linalg/Vector route work, common EmitC semantics, target artifact semantics, dashboards, broad smoke matrices, wrapper-only renames, runtime ABI changes, dispatch/fallback changes, emitted statement order changes, performance claims, and `ssh rvv` runtime claims.

## Completion Notes

* Added `RVVSelectedBodyMigratedRouteStatementPlanOwner`, `getRVVSelectedBodyMigratedRouteStatementPlanOwners()`, and `isRVVSelectedBodyMigratedRouteStatementPlanConsumer(...)`.
* Replaced the manual central sequence inside `getRVVSelectedBodyMigratedRouteStatementPlan(...)` with exact-one owner selection and owner-local statement-plan builders.
* Migrated owner registry entries cover elementwise arithmetic, compare/select, widening conversion, standalone reduction, plain MAcc, base memory movement, computed-mask memory, segment2 memory, and computed-mask accumulation.
* Added focused C++ registry coverage for owner count/order/family tags, hook presence, exact-once classification, empty non-consumer behavior, and missing dependency failure through the owner boundary.
* Updated `.trellis/spec/extension-plugins/rvv-plugin.md` because the migrated statement-plan owner registry is a new executable API signature.
* Representative dry-runs passed for direct pre-realized route-entry `cmp_select` and explicit selected-body `add`.
* Remaining central route-provider authority: direct-provider contraction statement construction still lives in `RVVEmitCRouteProvider.cpp`; it is the next continuation point and was not counted as migrated owner-registry coverage.

## Technical Notes

* Read: `.trellis/spec/index.md`, RVV plugin route-provider ownership sections, EmitC route spec, plugin protocol, variant pipeline, MLIR testing contract, predecessor route-control registry and selected-body realization owner-local PRDs, RVV route planning/provider headers and implementations, selected-body realization owner registry as upstream producer, and focused RVV plugin tests.
* Memory-derived guardrail: Stage1 i32 authority cleanup remains active context; do not preserve hidden legacy i32 or source-front-door authority behind owner wrappers.
