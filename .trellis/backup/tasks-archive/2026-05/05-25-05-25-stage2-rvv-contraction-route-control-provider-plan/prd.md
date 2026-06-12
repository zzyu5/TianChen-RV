# Stage2 RVV contraction route-control provider-plan integration

## Goal

Make the existing RVV contraction route family consume the shared
`RVVSelectedBodyRouteControlProviderPlan` before direct provider route
construction. Existing route-supported contraction routes must keep their
current computation, runtime ABI, statement order, and artifact shape, but
their AVL/VL, SEW/LMUL, policy, selected capability, typed config, runtime ABI,
materialization, operand binding, widening MAcc or widening dot classification,
accumulator/result layout, strided-input facts, and computed-mask facts must be
joined through the same RVV-owned route-control boundary now used by other
mature Stage2 families.

This is a production-path control-boundary migration for already-supported
contraction routes. It is not a new contraction coverage, dtype/LMUL,
frontend, dashboard, or evidence-copying task.

## Direction Source

- Direction title: `Stage2 RVV contraction route-control provider-plan
  integration`.
- Module owner: RVV plugin-local route-control provider-plan consumption by
  the existing contraction route-family/provider path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `99fe103b rvv: consume route control plan in computed-mask
  accumulation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes/User brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflow.

## Current Repository Facts

- Specs require the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed or realized `tcrv_rvv` body ->
  RVV plugin-owned legality/realization/route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact mirrors.
- `RVVSelectedBodyRouteControlProviderPlan` currently carries typed config,
  selected target capability, runtime AVL/VL control, runtime ABI order,
  SEW/LMUL, tail policy, mask policy, and explicit consumer flags for migrated
  families through computed-mask accumulation MAcc.
- Current code has `RVVSelectedBodyContractionRouteFamilyPlan`,
  `contractionRouteFamilyPlan`, contraction family consumer predicates,
  `verifyRVVSelectedBodyContractionRouteFamilyProviderPlans(...)`, contraction
  materialization facts, math operand-binding facts, and direct provider
  emission branches.
- The direct provider path still emits widening MAcc and widening dot-reduction
  contraction sequences after family/materialization/binding checks without a
  contraction-specific route-control consumer flag.
- The current active contraction consumers are:
  - `WideningMAccAdd`;
  - `WideningDotReduceAdd`;
  - `StridedInputWideningDotReduceAdd`;
  - `ComputedMaskWideningDotReduceAdd`;
  - `ComputedMaskStridedInputWideningDotReduceAdd`.
- The current contraction path is direct-provider rather than migrated
  statement-plan based. This task should route-control-gate direct provider
  construction and must not invent a wrapper-only statement-plan layer.

## Requirements

1. Add a structural contraction route-control consumer marker to
   `RVVSelectedBodyRouteControlProviderPlan` or an equivalent shared control
   marker.
2. Make `getRVVSelectedBodyRouteControlProviderPlan(...)` recognize only the
   existing active contraction consumers listed above.
3. For contraction consumers, route-control construction must require:
   - same-analysis `RVVSelectedBodyContractionRouteFamilyPlan`;
   - same-analysis `RVVSelectedBodyRouteMaterializationFacts`;
   - same-analysis typed config facts;
   - same-analysis selected target capability facts;
   - contraction-owned `RVVRuntimeAVLVLControlPlan`;
   - runtime ABI order and runtime `n`/AVL binding;
   - SEW, LMUL, tail policy, and mask policy;
   - widening MAcc versus widening dot-reduction classification;
   - accumulator and result layout facts;
   - unit-stride versus strided-input facts where applicable;
   - computed-mask producer and inactive-lane facts where applicable;
   - materialization leaves for setvl, source loads, strided loads, widening
     product, masked widening product, MAcc or reduction compute, compare,
     masked merge, seed splat, and store where applicable.
4. Direct provider route construction must consume the route-control provider
   plan before building contraction setvl/load/stride/compare/product/MAcc or
   dot-reduction/store statements.
5. Direct provider route construction must also require matching math
   operand-binding facts before emitting contraction statements.
6. Stale analysis, missing contraction family plan, stale materialization
   facts, missing route-control plan, wrong runtime AVL role, policy mismatch,
   unsupported capability/config, stale widening MAcc/dot classification, stale
   accumulator/result layout, stale strided or computed-mask facts, stale
   operand binding, and runtime ABI mirror mismatch must fail closed before
   route/artifact authority.
7. Common EmitC, target artifact export, route ids, generated headers,
   metadata fields, scripts, ABI strings, descriptors, and artifact names may
   mirror provider-built contraction facts only after provider route
   construction. They must not infer contraction semantics, AVL/VL, policy,
   dtype/config, accumulator/result layout, striding, mask producer, or
   intrinsic choice.
8. No emitted executable behavior, computation semantics, dispatch/fallback
   behavior, runtime ABI, or statement order may change unless required by the
   route-control boundary and explicitly evidenced.

## Acceptance Criteria

- [x] `RVVSelectedBodyRouteControlProviderPlan` includes a contraction
      consumer flag or equivalent structural marker.
- [x] Route-control provider-plan consumer detection includes the five active
      contraction routes and does not include unrelated MAcc, standalone
      reduction, computed-mask accumulation, memory, compare/select,
      conversion, scalar broadcast, runtime splat-store, source-front-door, or
      future routes.
- [x] `getRVVSelectedBodyRouteControlProviderPlan(...)` requires the verified
      contraction family plan and same-analysis materialization facts before
      exposing runtime AVL/VL, typed config, selected capability, policy,
      runtime ABI, classification, layout, strided-input, computed-mask, and
      mirror facts for contraction consumers.
- [x] `RVVEmitCRouteProvider` consumes the route-control provider plan before
      direct contraction provider statement construction.
- [x] Focused C++ tests prove positive route-control consumption for widening
      MAcc, plain widening dot-reduction, strided widening dot-reduction,
      computed-mask widening dot-reduction, and computed-mask strided widening
      dot-reduction.
- [x] Focused negative C++ tests fail closed for representative stale or
      missing dependencies: missing contraction plan, stale same-analysis
      materialization, missing or wrong runtime AVL role, policy mismatch,
      unsupported selected capability/config, runtime ABI mirror mismatch,
      stale contraction classification, stale accumulator/result layout, stale
      strided facts, stale computed-mask facts, stale materialization leaf, and
      stale math operand binding.
- [x] FileCheck or generated-header evidence remains explicit mirror-only if
      route metadata/header mirrors change.
- [x] Generated-bundle dry-run and `ssh rvv` correctness are rerun only if
      emitted executable behavior, ABI, or artifact mirrors change. If emitted
      behavior stays unchanged, the final report states why historical runtime
      evidence remains sufficient.
- [x] Bounded authority scan over touched planning/provider/test/spec/target or
      script files finds no name-, route-id-, metadata-, descriptor-,
      ABI-string-, script-, artifact-, common-EmitC-, source-front-door-, or
      legacy-i32-derived contraction AVL/VL, policy, dtype, accumulator,
      result-layout, striding, computed-mask, memory-form, or compute
      authority.
- [x] `git diff --check` passes.
- [x] Focused RVV plugin tests and `check-tianchenrv` pass, or an exact blocker
      is recorded.

## Out Of Scope

- New contraction operation coverage, dot-product variants, widening-MAcc
  variants, dtype/LMUL clone batches, unsigned variants, high-level
  Linalg/Vector/StableHLO frontend lowering, source-front-door positive routes,
  dashboards, broad smoke matrices, or evidence-only fixture copying.
- Wrapper-only statement-plan refactors that do not gate the production direct
  provider path.
- Treating runtime counts, route ids, metadata fields, manifests, artifact
  names, ABI strings, descriptors, scripts, tests, common EmitC, target
  artifact code, or legacy i32 spellings as AVL/VL, accumulator, result layout,
  striding, mask producer, policy, dtype, or compute authority.
- Changing computation semantics, dispatch/fallback behavior, runtime ABI, or
  emitted statement order unless required by the control boundary and
  explicitly evidenced.

## Technical Approach

1. Validate and start this Trellis task.
2. Inspect RVV route-control, contraction family/provider verification,
   materialization facts, math operand-binding facts, direct provider emission,
   selected-body realization, and focused C++ tests.
3. Add a bounded contraction route-control consumer predicate and consumer flag.
4. Extend route-control plan construction with a contraction branch that joins
   the verified contraction family plan with materialization facts, typed config
   facts, selected capability facts, runtime AVL/VL control, policy, runtime
   ABI mirrors, contraction classification, accumulator/result layout,
   strided-input facts, computed-mask facts, and materialization leaves.
5. Gate direct contraction provider statement construction on that
   route-control plan and existing math operand-binding facts.
6. Extend focused C++ tests rather than copying broad generated-bundle
   fixtures.
7. Update `.trellis/spec/extension-plugins/rvv-plugin.md` only for durable
   route-control consumer boundary wording established by production code.
8. Run focused build/tests, bounded authority scans, `git diff --check`, and
   `check-tianchenrv`.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-05-25-stage2-rvv-contraction-route-control-provider-plan`
2. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
3. `./build/bin/tianchenrv-rvv-extension-plugin-test`
4. Focused lit/FileCheck or generated-bundle dry-run only if route metadata,
   header mirrors, ABI, artifact schema, or generated output changes.
5. Bounded authority scan over touched RVV planning/provider/test/spec files.
6. `git diff --check`
7. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Relevant previous tasks read:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-runtime-vl-policy-provider-plan/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-scalar-macc-route-control-provider-plan/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-computed-mask-accum-route-control/prd.md`,
  `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-contraction-route-family-planning/prd.md`,
  `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-runtime-binding-closure/prd.md`,
  `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-contraction-dot-reduction-operand-binding/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-widening-dot-reduce-route-family-ownership/prd.md`.
- Workspace journal read: `.trellis/workspace/codex/journal-15.md` sessions
  for route-control provider-plan adoption through computed-mask accumulation.
- Initial code surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Definition Of Done

- Existing contraction routes consume the shared RVV route-control provider
  plan before direct provider construction.
- Focused positive and fail-closed tests pass for the changed boundary.
- No legacy/source/descriptor/common-export/mirror-only authority is
  reintroduced.
- The task is finished/archived only if production code/tests/evidence are
  present in this round; otherwise it remains open with an exact continuation
  point.

## Implementation Result

- Added `controlsContraction` to `RVVSelectedBodyRouteControlProviderPlan`.
- Added bounded contraction route-control consumer detection for:
  `WideningMAccAdd`, `WideningDotReduceAdd`,
  `StridedInputWideningDotReduceAdd`,
  `ComputedMaskWideningDotReduceAdd`, and
  `ComputedMaskStridedInputWideningDotReduceAdd`.
- Extended `getRVVSelectedBodyRouteControlProviderPlan(...)` so contraction
  consumers require same-analysis contraction family/materialization facts,
  typed config facts, selected target capability facts, the contraction runtime
  AVL/VL control plan, runtime ABI mirrors, widening MAcc versus widening dot
  classification, accumulator/result layout, optional strided-input facts,
  optional computed-mask facts, and contraction materialization leaves.
- Gated the existing direct contraction provider path in
  `RVVEmitCRouteProvider` on the shared route-control plan before statement
  construction. The path also requires the existing RVV-owned math
  operand-binding facts before emitting contraction statements.
- Updated focused C++ coverage in `RVVExtensionPluginTest.cpp` for positive
  contraction route-control consumption and fail-closed missing/stale control
  dependencies.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to record contraction
  as a route-control consumer while preserving the current direct-provider
  path and explicitly avoiding a wrapper-only statement-plan requirement.
- No emitted statement order, runtime ABI, generated headers, artifact schema,
  computation semantics, dispatch/fallback behavior, or target executable
  behavior changed. No new `ssh rvv` runtime/correctness claim was made;
  historical runtime evidence remains sufficient for unchanged emitted
  behavior.

## Validation Result

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-05-25-stage2-rvv-contraction-route-control-provider-plan`
- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [x] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] Bounded authority scan over touched planning/provider/test/spec files.
      Matches were spec prohibition text, existing negative route-id/source
      tests, and route-id mirror validation, not new route authority.
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2`: 379/379 passed.
