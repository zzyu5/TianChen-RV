# Stage2 RVV selected-body realization route-path materialization

## Goal

Make the RVV plugin route/emission production entry points invoke the existing
elementwise/compare-select selected-body realization boundary before collecting
route facts. A selected `tcrv.exec` RVV variant that still contains a
pre-realized typed `tcrv_rvv` elementwise or compare/select body should be
realized into `setvl` / `with_vl` / typed dataflow structure by RVV plugin code,
then feed the existing route analysis, materialization facts, operand-binding
facts, statement-plan, provider-built `TCRVEmitCLowerableRoute`, and common
EmitC materializer path.

This is a production-path rewire for an already supported cluster, not new RVV
coverage.

## Direction Source

- Direction title: `Stage2 RVV selected-body realization route-path
  materialization`.
- Module owner: RVV plugin-owned realization invocation and consumption
  boundary from selected pre-realized `tcrv_rvv` bodies into the existing route
  facts, statement-plan, provider, and artifact path for one
  production-active elementwise/compare-select cluster.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `74c6384f rvv: gate pre-realized route planning`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- `.trellis/spec/index.md` requires the RVV-first chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / selected-body realization / route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC materializer -> target artifact.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires
  elementwise/compare-select pre-realized bodies to be realized by a named RVV
  plugin-local boundary before route facts, operand-binding facts,
  statement plans, and provider-built routes are collected.
- The archived route-path integration PRD ended with a route-analysis guard:
  if route analysis sees a pre-realized body, it fails closed with a
  selected-body realization diagnostic. That is necessary, but by itself it is
  still mostly a gate.
- Current code evidence:
  - `RVVExtensionPlugin::materializeSelectedLoweringBoundary(...)` can realize
    pre-realized bodies when the separate boundary pass is run.
  - `buildVariantEmissionPlan(...)` and
    `buildVariantEmitCLowerableRoute(...)` currently look for an already
    realized `tcrv_rvv.with_vl` boundary before route facts are collected.
  - Existing pre-realized target fixtures pass because their RUN lines first
    execute `--tcrv-materialize-selected-lowering-boundaries`.
  - The direct EmitC lowerable materialization pass asks the plugin for a
    provider route and does not itself run selected-body realization.
- The existing C++ tests prove manual boundary materialization followed by
  route/provider consumption for pre-realized add and compare/select, but do
  not prove that the RVV provider/emission route entry itself invokes
  realization before collecting route facts.

## Requirements

1. Add a bounded RVV plugin-local route-entry helper for the
   elementwise/compare-select cluster that finds an existing realized
   `setvl`/`with_vl` boundary or realizes a unique pre-realized
   elementwise/compare-select body before route facts are collected.
2. Wire RVV `buildVariantEmissionPlan(...)` and
   `buildVariantEmitCLowerableRoute(...)` through that helper so the production
   route/emission entries consume realized typed body structure for this
   cluster instead of only failing at the route-analysis guard.
3. Preserve the route-analysis guard. If unsupported or unrelated pre-realized
   bodies reach route analysis, the provider must still fail closed with the
   selected-body realization diagnostic.
4. Keep realization semantics in RVV plugin code. Common EmitC/export may
   consume provider-built routes but must not infer RVV dtype, op kind, policy,
   schedule, ABI roles, body shape, or intrinsic names.
5. Preserve explicit already-realized selected-body behavior and existing
   pre-realized behavior through the explicit
   `--tcrv-materialize-selected-lowering-boundaries` path.
6. Do not add route coverage, new operation families, memory/reduction/
   contraction expansion, dtype/LMUL clone batches, frontend lowering,
   source-front-door positive routes, legacy i32 authority, descriptor/direct-C
   paths, runtime ABI changes, dispatch/fallback behavior changes, or runtime
   correctness/performance claims.

## Acceptance Criteria

- [x] PRD, `implement.jsonl`, and `check.jsonl` reference the relevant RVV
      plugin, EmitC route, plugin-protocol, testing specs, and previous
      route-path integration PRD.
- [x] A bounded RVV helper or equivalent public API can identify/realize only
      elementwise/compare-select pre-realized bodies for route entry use.
- [x] RVV `buildVariantEmissionPlan(...)` and
      `buildVariantEmitCLowerableRoute(...)` invoke that helper before
      `describeRVVSelectedBodyEmitCRoute(...)` or provider route construction.
- [x] Focused C++ coverage proves a pre-realized elementwise add and
      compare/select body can call the production emission/provider route
      entries without a prior manual boundary materialization call, and that
      the realized body feeds provider route construction.
- [x] Focused negative coverage proves an unrelated pre-realized body remains
      fail-closed unless its owning realization path runs.
- [x] Representative lit/FileCheck coverage proves a pre-realized
      elementwise/compare-select fixture can reach EmitC route materialization
      without explicitly running the selected-boundary materialization pass,
      while explicit selected-body fixtures remain unchanged.
- [x] Bounded scan over touched realization/planning/provider/common files
      shows common EmitC/export does not perform semantic realization.
- [x] Active-authority scan over touched files finds no newly introduced
      legacy `RVVI32M1`, `rvv-i32m1`, positive finite `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/source-front-door/direct-C/source-export,
      or mirror-only route authority.
- [x] `git diff --check` passes.
- [x] Focused build/tests and `check-tianchenrv` pass, or an exact blocker is
      documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Out Of Scope

- No new RVV operation coverage, new route families, memory/reduction/
  contraction work, dtype/LMUL matrix expansion, or Stage 2 coverage expansion.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive routes, descriptor-driven computation,
  direct-C/source-export route, or legacy i32 route authority.
- No runtime ABI, dispatch/fallback, emitted target sequence, correctness,
  runtime, or performance claim changes.
- No movement of RVV semantics into common EmitC/export.
- No `ssh rvv` evidence unless emitted target sequence, runtime correctness,
  ABI behavior, or performance is changed or claimed.

## Technical Approach

Add a small route-entry bridge in RVV plugin code rather than changing common
EmitC. The bridge should:

- reuse the existing `realizePreRealizedRVVElementwiseCompareSelectCluster`
  boundary;
- expose a cluster predicate / route-entry realization API from
  `RVVSelectedBodyRealization`;
- preserve mixed pre-realized + already-realized fail-closed behavior;
- return the existing `with_vl` when the body is already realized;
- create a local `mlir::OpBuilder` at RVV plugin entry only when a
  pre-realized elementwise/compare-select body must be consumed;
- leave unrelated pre-realized families to their existing paths and the
  route-analysis guard.

The provider should still consume only realized route analysis,
materialization facts, operand-binding facts, and migrated statement plans.

## Validation Plan

1. Validate and start this Trellis task.
2. Build the focused C++ plugin test target.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build `tcrv-opt` and `tcrv-translate` if needed.
5. Run focused lit/FileCheck coverage for:
   - new pre-realized elementwise/compare-select direct EmitC materialization;
   - existing pre-realized add and compare/select target artifact fixtures;
   - existing explicit selected-body add and compare/select fixtures.
6. Run bounded semantic-realization scan over touched RVV and common files.
7. Run active-authority scan over touched files.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`.

## Implementation Result

- Added RVV plugin public route-entry realization helpers:
  `variantContainsPreRealizedRVVElementwiseCompareSelectSelectedBody(...)` and
  `realizePreRealizedRVVElementwiseCompareSelectSelectedBody(...)`.
- Added an RVV plugin-local helper in `RVVExtensionPlugin.cpp` that either
  returns the existing realized `setvl` / `with_vl` boundary or realizes a
  unique elementwise/compare-select pre-realized body before route/emission
  facts are collected.
- Rewired `RVVExtensionPlugin::buildVariantEmissionPlan(...)` and
  `RVVExtensionPlugin::buildVariantEmitCLowerableRoute(...)` through that
  route-entry helper.
- Preserved the provider-side route-analysis guard: direct route analysis over
  a pre-realized body still fails closed until the RVV plugin realization
  boundary has run.
- Added C++ plugin coverage proving:
  - provider route construction can realize a pre-realized add body and produce
    a statement-plan route before emission planning;
  - emission planning can realize a pre-realized compare/select body before
    collecting route facts;
  - unrelated reduction pre-realized bodies remain fail-closed through the
    bounded elementwise/compare-select route entry.
- Added lit/FileCheck coverage showing
  `--tcrv-materialize-emitc-lowerable-routes` can materialize representative
  pre-realized add and compare/select bodies without first running
  `--tcrv-materialize-selected-lowering-boundaries`.
- Common EmitC/export, target export, and RVV EmitC route provider/planning
  files were not changed in this round.

## Validation Result

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-stage2-rvv-selected-body-realization-route-materialization`
  passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed with
  `RVV extension plugin smoke test passed`.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- Focused lit/FileCheck coverage passed 5/5:
  `Conversion/EmitC/rvv-pre-realized-elementwise-route-materialization.mlir`,
  `Target/RVV/pre-realized-selected-body-artifact-add.mlir`,
  `Target/RVV/pre-realized-selected-body-artifact-cmp-select.mlir`,
  `Target/RVV/explicit-selected-body-artifact-add.mlir`, and
  `Target/RVV/explicit-selected-body-artifact-cmp-select.mlir`.
- First full `check-tianchenrv` run exposed a regression: direct
  `buildVariantEmitCLowerableRoute(...)` route construction was enforcing full
  selected-lowering-boundary conformance metadata on existing explicit
  selected-body EmitC fixtures. The repair removed that direct-route
  conformance validation while keeping route-entry realization and provider
  structural route analysis.
- Added-line active-authority scan over touched implementation and test files
  found no newly introduced legacy `RVVI32M1`, `rvv-i32m1`, positive finite
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/source-front-door/direct-C,
  source-export, or mirror-only authority drift.
- Bounded common/provider diff scan found no changes in common EmitC/export,
  target export, RVV EmitC route provider, or RVV EmitC route planning files.
- `git diff --check` passed.
- Final `cmake --build build --target check-tianchenrv -j2` passed with
  364/364 lit tests.
- No `ssh rvv` evidence was collected because this task did not change or
  claim emitted target sequence, runtime correctness, ABI behavior, or
  performance.

## Definition Of Done

The RVV plugin production route/emission entries materialize the existing
elementwise/compare-select selected-body realization boundary before collecting
route facts for pre-realized bodies; realized bodies feed the existing
statement-plan/provider/common EmitC path; unsupported or unrelated cases fail
closed; focused and full checks pass; Trellis task metadata is truthful; and
the completed work is committed as one coherent change.
