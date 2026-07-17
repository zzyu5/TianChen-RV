# Stage2 RVV Route-Family Route-Control Registry Consolidation

## Goal

Consolidate the completed RVV route-control adoption work behind one plugin-local route-control owner registry. Existing route-supported RVV families should no longer rely on a central ad hoc route-control consumer OR chain or per-family branch wiring for route-control eligibility. The registry must classify consumers, require the owning family/materialization facts from the same selected-route analysis, set the correct control flag, and feed both statement-plan and direct-provider paths without changing emitted semantics.

## What I Already Know

* The repository has no `.trellis/.current-task` for this brief; this task was created from the Hermes direction brief.
* Current HEAD is `8fba5f93 rvv: consume route control plan in contraction provider`; recent commits completed per-family route-control consumption through contraction.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires route-family provider ownership and route-control provider-plan boundaries to remain RVV plugin-local and not become common EmitC, artifact, metadata, ABI-string, source-front-door, or legacy i32 authority.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires RVV provider-built `TCRVEmitCLowerableRoute`; common EmitC may materialize provider output but must not infer RVV dtype/config/schedule/intrinsics.
* Current production code already has route-family owner registries for memory, elementwise/select, reduction/accumulation/contraction, and a top-level provider owner registry in `RVVEmitCRoutePlanning`.
* Current route-control code still uses `isRVVSelectedBodyRouteControlProviderPlanConsumer(...)` as a long manual OR chain and `getRVVSelectedBodyRouteControlProviderPlan(...)` as a central sequence of family-specific branches.
* Current tests already cover many family owner registries and per-family route-control behavior; this task should extend that coverage to the route-control boundary itself.

## Assumptions

* The intended output is a production C++ refactor and focused tests, not new route coverage.
* The full registry extraction is feasible if it keeps the existing family-specific validation logic as registry-owned handlers instead of rewriting every validator into a generic abstraction.
* Runtime `ssh rvv` evidence is not required unless emitted target sequence, runtime ABI, or generated semantics change.

## Requirements

* Add a single RVV plugin-local route-control owner registry or equivalent typed owner table for all currently adopted route-control consumers:
  * ordinary elementwise arithmetic
  * scalar-broadcast elementwise
  * plain compare/select
  * computed-mask select
  * widening conversion
  * non-segment computed-mask memory
  * segment2 memory
  * base memory movement
  * standalone reduction
  * scalar-broadcast MAcc
  * runtime scalar splat-store
  * computed-mask accumulation MAcc
  * contraction
* Replace route-control consumer classification with registry-backed classification.
* Make `getRVVSelectedBodyRouteControlProviderPlan(...)` dispatch through the registry and require exactly one active owner for each route-control consumer.
* Preserve existing family-specific fail-closed checks for missing verified family plans, stale/different-analysis materialization facts, typed config facts, selected capability facts, runtime AVL/VL facts, policy mirrors, and family classification facts.
* Ensure unsupported/non-consumer routes still return an empty route-control plan cleanly.
* Ensure direct-provider paths such as contraction and statement-plan paths consume the same registry-owned route-control boundary.
* Do not add operation coverage, change emitted statement order, change runtime ABI, or move RVV semantics into common EmitC/export/scripts/artifacts/metadata.

## Acceptance Criteria

* [x] Production C++ exposes a route-control owner registry with explicit owner names and non-null consumer/control hooks.
* [x] Every current route-control family listed in the RVV plugin spec is represented exactly once in the registry.
* [x] `isRVVSelectedBodyRouteControlProviderPlanConsumer(...)` or its public equivalent is registry-backed rather than a manual OR chain.
* [x] `getRVVSelectedBodyRouteControlProviderPlan(...)` uses the registry to pick the owning route-control family and fails closed if zero/multiple owners are selected for a consumer.
* [x] Missing family plan and stale/different-analysis materialization facts still produce targeted diagnostics through the registry-owned boundary.
* [x] Non-consumer routes continue to receive an empty route-control plan.
* [x] Existing representative explicit and pre-realized RVV route construction remains unchanged for at least one direct-provider family and one statement-plan family.
* [x] Focused C++ tests cover registry membership, exact coverage count/names, classification, empty-plan non-consumer behavior, missing/stale same-analysis diagnostics, and representative provider route construction.
* [x] Bounded scan over touched files finds no new route authority from metadata, route ids, descriptors, ABI strings, scripts, artifacts, common EmitC, source-front-door, or legacy i32 names.
* [x] `git diff --check` passes.
* [x] Focused RVV plugin test target passes; run `check-tianchenrv` if feasible, otherwise report the exact blocker.

## Completion Notes

* Added `RVVSelectedBodyRouteControlProviderOwner` and registry-backed `getRVVSelectedBodyRouteControlProviderOwners()`.
* Replaced the route-control consumer OR chain with registry-backed consumer classification and owner dispatch.
* Moved route-control family plan/materialization same-analysis checks into owner builders while preserving the shared typed config, runtime AVL/VL, policy, and selected capability finalization.
* Added C++ registry tests for thirteen adopted route-control families, exact owner classification, non-consumer empty-plan behavior, missing plan diagnostics, and stale materialization diagnostics.
* Representative artifact dry-runs passed for explicit direct-provider contraction and pre-realized statement-plan compare/select routes.

## Out of Scope

* New RVV route coverage, dtype/LMUL clone batches, Linalg/Vector frontend lowering, source-front-door positive routes, dashboards, reports, broad smoke matrices, or one-intrinsic wrappers.
* Changing emitted statement order, runtime ABI, dispatch/fallback behavior, computation semantics, or target artifact packaging.
* Moving RVV route-control semantics into common EmitC, target export, scripts, metadata, route ids, descriptors, ABI strings, artifact names, or legacy i32 spellings.

## Technical Notes

* Relevant specs read:
  * `.trellis/spec/index.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
* Relevant production surfaces inspected:
  * `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  * `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  * `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* Relevant test surface inspected:
  * `test/Plugin/RVVExtensionPluginTest.cpp`
* Recent workspace journal context read:
  * `.trellis/workspace/codex/index.md`
  * tail of `.trellis/workspace/codex/journal-15.md`

## Definition of Done

* Trellis task context is curated and validates.
* Production route-control consumer/owner logic is registry-owned.
* Focused tests and relevant lit/artifact dry-runs demonstrate unchanged route construction.
* Quality checks pass or exact blockers are recorded.
* Task status and journal are updated truthfully; completed work is committed as one coherent commit.
