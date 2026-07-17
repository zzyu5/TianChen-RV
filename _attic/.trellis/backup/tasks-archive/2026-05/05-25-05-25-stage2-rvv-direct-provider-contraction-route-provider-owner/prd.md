# Stage2 RVV direct-provider contraction route-provider owner

## Goal

Move existing direct-provider contraction statement construction out of the
central `RVVEmitCRouteProvider.cpp` branch and behind an RVV plugin-owned
route-provider owner boundary. The owner must consume the realized typed
`tcrv_rvv` contraction body facts, route-control provider-plan facts,
materialization facts, math operand-binding facts, runtime AVL/VL facts,
selected capability/config facts, and ABI/mem_window role bindings before
building the contraction statements that are attached to
`TCRVEmitCLowerableRoute`.

## What I Already Know

* Repository state before task creation was clean on `main`, with HEAD
  `cefd7cc0 rvv: add route-provider statement-plan owner registry`.
* `.trellis/.current-task` did not exist, so this task was created from the
  Hermes direction brief.
* The predecessor task moved migrated statement-plan provider consumption
  behind `RVVSelectedBodyMigratedRouteStatementPlanOwner` and exact-one owner
  selection.
* The predecessor task intentionally left direct-provider contraction statement
  construction in `RVVEmitCRouteProvider.cpp` as the next continuation point.
* Current direct-provider contraction construction validates route-control and
  math operand-binding facts, but the actual statement sequence still lives in
  the central provider body.
* Existing active contraction routes are `widening_macc_add`,
  `widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`.

## Context Read

* `.trellis/spec/index.md` defines the RVV-first chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires route-provider
  ownership, route-control provider-plan consumption, typed body/config
  authority, fail-closed diagnostics, and mirror-only metadata.
* `.trellis/spec/lowering-runtime/emitc-route.md` keeps common EmitC neutral;
  RVV type/header/intrinsic/statement choices remain RVV-plugin-owned.
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md` requires route
  construction to stay inside the origin plugin and forbids route/dtype
  recovery from metadata, route ids, source front doors, descriptors, or
  artifact names.
* `.trellis/spec/testing/mlir-testing-contract.md` requires focused C++/lit
  coverage for registry/provider behavior and `ssh rvv` only for new runtime,
  correctness, or performance claims.
* Archived predecessor:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-route-provider-family-owner-registry/prd.md`.
* Relevant code:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Requirements

* Add an RVV plugin-owned direct-provider contraction route statement owner
  boundary, or equivalent owner surface, in the RVV planning/provider layer.
* The owner must classify each active direct-provider contraction route exactly
  once and leave unrelated route families outside the owner.
* The owner must validate and consume:
  * operation kind and contraction sub-family;
  * memory form, including strided-input and computed-mask forms;
  * typed config facts: SEW, LMUL, vector/C types, tail policy, mask policy;
  * selected capability facts and route-control provider plan from the same
    analysis;
  * runtime AVL/VL control and runtime element-count source;
  * math operand-binding facts for compare operands, dot operands,
    accumulator/result, strides, and output;
  * required ABI/mem_window role bindings and source-operation provenance;
  * accumulator/result layout and required materialized leaves.
* The production route provider must call this owner boundary before falling
  back to the older central provider-local statement assembly surface.
* The provider may still instantiate `TCRVEmitCLowerableRoute`, attach neutral
  headers/type/ABI mappings, and attach provider-ready statements returned by
  the owner.
* Existing migrated statement-plan owner registry, selected-body realization,
  route-control registry, emitted statement order, runtime ABI,
  dispatch/fallback behavior, and computation semantics must remain unchanged.
* Route ids, artifact names, ABI strings, script names, descriptors, common
  EmitC logic, emission-plan/status metadata, source-front-door markers, test
  names, and legacy i32 helper names must not become owner or route authority.

## Acceptance Criteria

* [x] Task context files contain real spec/task context and no placeholder JSONL.
* [x] Production C++ exposes a plugin-owned direct-provider contraction route
  owner boundary used by the default RVV route construction path.
* [x] Existing active direct-provider contraction route families are classified
  exactly once by the owner boundary.
* [x] Unrelated routes receive an empty/non-consumer result and preserve existing
  provider behavior.
* [x] Missing route-control provider plan, stale materialization/control facts,
  invalid math operand binding, invalid runtime AVL/VL source, invalid
  config/policy/capability, missing ABI role, or missing contraction leaf fails
  closed before route statement construction.
* [x] Existing migrated statement-plan owner registry tests still pass and prove
  non-regression.
* [x] Focused C++ tests cover owner membership, exact-one classification,
  non-consumer behavior, positive contraction owner statement construction, and
  at least one fail-closed owner dependency.
* [x] Representative generated-bundle or `tcrv-translate` dry-run passes for
  one direct-provider contraction route and one existing migrated
  statement-plan route.
* [x] Bounded touched-file scan shows no new legacy i32/source-front-door/
  descriptor/ABI-string/artifact/script/common-EmitC/metadata/route-id
  authority leaks.
* [x] `git diff --check` passes.
* [x] `check-tianchenrv` passes, or an exact blocker is recorded.

## Out Of Scope

* New RVV families, new contraction variants, dtype/LMUL clone batches,
  source-front-door positive routes, frontend/Linalg routes, common EmitC
  semantics, target artifact semantics, reports, dashboards, broad smoke
  matrices, runtime ABI changes, dispatch/fallback changes, emitted statement
  order changes, performance claims, and `ssh rvv` runtime claims.

## Definition Of Done

* PRD, `implement.jsonl`, and `check.jsonl` are truthful.
* The implementation rewires the production/default contraction path, not just
  a wrapper beside the old path.
* Focused and final checks pass or blockers are exact.
* Task status and journal are truthful.
* Completed work is archived and committed as one coherent commit, or the
  unfinished continuation point is explicit.

## Completion Evidence

* Added `RVVSelectedBodyDirectContractionRouteProviderOwner`,
  `RVVSelectedBodyDirectContractionRouteStatementPlan`, owner registry lookup,
  consumer predicate, and aggregate direct contraction statement-plan getter.
* Moved direct contraction statement construction into the RVV planning owner
  boundary and made the default provider attach returned pre-loop and loop
  statements before the old generic provider-local path.
* Removed active central provider branches for direct contraction binding,
  computed-mask dot-reduction loops, widening MAcc loops, dot-reduction loads,
  products, reductions, and stores.
* Added focused C++ coverage for exact-one direct owner classification across
  all five active direct contraction routes, non-consumer empty-plan behavior,
  positive computed-mask strided dot and widening MAcc statement plans,
  provider consumption, migrated owner registry non-regression, and fail-closed
  missing materialization/math/leaf/family-policy dependencies.
* Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the durable
  direct contraction route-provider owner boundary.
* Verified direct-provider contraction dry-run:
  `direct-contraction-computed-mask-strided`.
* Verified migrated statement-plan dry-run:
  `migrated-add-statement-plan`.
* `ssh rvv` was not rerun because this task changed route-provider ownership
  and validation boundaries only; it did not change emitted RVV computation
  semantics, runtime ABI, dispatch/fallback behavior, target artifact semantics,
  or performance claims.
