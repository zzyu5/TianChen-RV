# Stage2 RVV selected-body realization route-path integration

## Goal

Integrate the RVV plugin-owned selected-body realization boundary into one
production-active route path so a selected `tcrv.exec` RVV variant with a
pre-realized typed `tcrv_rvv` body is realized by the RVV plugin before route
facts, statement plans, provider route construction, and common EmitC artifact
materialization consume that body.

## What I already know

* `fb60c5fd` added the selected-body realization boundary in
  `RVVSelectedBodyRealization.{h,cpp}`, updated the RVV plugin spec, added
  focused plugin tests, archived its Trellis task, and left a clean worktree.
* The prior changed-file mix did not integrate realization into route provider,
  planning, or target artifact paths.
* The desired path is:
  selected `tcrv.exec` RVV variant -> pre-realized typed `tcrv_rvv` body ->
  RVV plugin-local realization -> realized `tcrv_rvv` body -> RVV facts /
  statement plans / provider-built `TCRVEmitCLowerableRoute` -> common EmitC
  artifact.
* The integration should use one already statement-plan-backed production
  cluster, preferably elementwise or compare-select unless live code evidence
  points to a smaller safer cluster.

## Assumptions

* Realization should happen inside RVV plugin-owned route preparation, not in
  common EmitC/export code.
* Explicit already-realized selected bodies must continue to follow the current
  path unchanged.
* Unsupported or incomplete pre-realized bodies must fail closed with targeted
  diagnostics instead of falling back to descriptor/source/front-door or legacy
  route authority.

## Requirements

* Realize a selected pre-realized RVV body before route facts are collected for
  the chosen production-active cluster.
* Feed the realized body into the existing RVV legality/materialization,
  operand-binding, statement-plan, route-provider, and common EmitC path.
* Preserve computation semantics, dtype/config authority, parameter roles,
  dispatch/fallback behavior, and runtime `n`/AVL/VL values.
* Keep common EmitC/export neutral: common code may materialize a provider-built
  route but must not choose or synthesize RVV realization semantics.
* Fail closed for unsupported, missing, or incomplete realization with
  diagnostic text that identifies the RVV selected-body realization boundary.

## Acceptance Criteria

* [ ] A focused plugin/provider test proves a pre-realized selected body is
      realized before RVV route facts/statement planning/provider construction.
* [ ] A focused negative test proves missing or unsupported realization fails
      closed with a targeted diagnostic.
* [ ] A representative lit/FileCheck fixture proves the affected pre-realized
      selected-body artifact reaches the route/emission path.
* [ ] Existing explicit selected-body behavior remains covered and unchanged.
* [ ] Scans over touched realization/planning/provider/export files show common
      EmitC/export does not invent RVV realization semantics or reintroduce
      legacy route authority.
* [ ] `git diff --check` passes.
* [ ] `check-tianchenrv` passes, or any blocker is documented exactly.

## Out of Scope

* New route coverage, new operation families, memory/reduction/contraction
  expansion, dtype/LMUL clone batches, or Stage 2 coverage expansion.
* High-level frontend lowering, source-front-door positive routes, descriptor
  or direct-C/source-export computation paths, artifact dashboards, broad smoke
  matrices, or common EmitC semantic logic.
* Runtime ABI, dispatch/fallback behavior, runtime `n`/AVL values, emitted
  semantics, correctness/runtime/performance claims, or `ssh rvv` evidence
  unless the implementation changes or claims them.
* Legacy `i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`, `rvv-i32m1`, artifact-name,
  route-id, descriptor, or exact intrinsic spelling as route authority.

## Technical Notes

* Read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-selected-body-realization-boundary/prd.md`,
  `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and representative pre-realized
  and explicit selected-body fixtures under `test/Target/RVV`.
* Live code evidence:
  `RVVExtensionPlugin::materializeSelectedLoweringBoundary(...)` already calls
  `realizePreRealizedRVVSelectedBody(...)` when no `with_vl` boundary exists
  and a pre-realized body is present.
* Route facts are collected in `analyzeRVVSelectedBodyRoute(...)` after
  `collectRVVSelectedBodyRouteSlice(...)` has found a realized `setvl` /
  `with_vl` structure. That route-analysis entry currently has no explicit
  pre-realized-body guard.
* This round will make route analysis fail closed with a selected-body
  realization diagnostic if a pre-realized body reaches route facts, and will
  add a production registry test proving `materializeSelectedLoweringBoundary`
  realizes pre-realized add/cmp-select before `buildVariantEmissionPlan` and
  provider route construction consume the body.

## Current Phase

Finish.

## Implementation Result

* Added a route-analysis guard in `analyzeRVVSelectedBodyRoute(...)` that
  fails closed when a pre-realized RVV selected body reaches route facts before
  selected-body realization has run. The diagnostic explicitly names the RVV
  selected-body realization boundary and rejects pre-realized `tcrv_rvv` input
  before route facts, operand-binding facts, statement plans, or provider
  routes are collected.
* Added focused C++ production-path coverage proving a pre-realized
  elementwise add body and a pre-realized compare/select body first fail before
  realization, then are realized through
  `registry.materializeSelectedLoweringBoundary(...)`, and finally feed
  `buildVariantEmissionPlan(...)`,
  `describeRVVSelectedBodyEmitCRoute(...)`, and
  `buildRVVSelectedBodyEmitCLowerableRoute(...)`.
* Updated the RVV plugin spec with the route-path guard contract: planning and
  provider consume realized `setvl` / `with_vl` structure only, and pre-realized
  body residue fails closed before facts are collected.
* Common EmitC/export code was not changed. The provider does not synthesize
  realization semantics; it receives realized typed body structure or rejects
  the input.

## Validation Result

* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-stage2-rvv-selected-body-realization-route-path-integration`
  passed.
* `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
* `build/bin/tianchenrv-rvv-extension-plugin-test` passed with
  `RVV extension plugin smoke test passed`.
* Focused lit/FileCheck filter passed 4/4 representative fixtures:
  `pre-realized-selected-body-artifact-add.mlir`,
  `pre-realized-selected-body-artifact-cmp-select.mlir`,
  `explicit-selected-body-artifact-add.mlir`, and
  `explicit-selected-body-artifact-cmp-select.mlir`.
* Provider/common semantic-realization scan found only the existing runtime
  AVL/VL helper and the new planning guard; no common EmitC/export semantic
  realization logic was added.
* Added-line active-authority scan found no legacy `RVVI32M1`, `rvv-i32m1`,
  positive finite `tcrv_rvv.i32_*`, descriptor, source-front-door, direct-C,
  source-export, or mirror-only authority drift.
* `git diff --check` passed.
* `cmake --build build --target check-tianchenrv -j2` passed with 363/363 lit
  tests.
* No `ssh rvv` evidence was collected because this task did not change or
  claim emitted target sequence, runtime correctness, ABI behavior, or
  performance.
