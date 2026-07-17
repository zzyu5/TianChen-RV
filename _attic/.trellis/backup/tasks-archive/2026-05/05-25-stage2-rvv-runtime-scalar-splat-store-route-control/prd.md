# Stage2 RVV runtime scalar splat-store route-control provider-plan integration

## Goal

Make the existing `runtime_i32_splat_store` RVV route family consume the shared
`RVVSelectedBodyRouteControlProviderPlan` before provider/route statement
construction. This task migrates the already-supported runtime scalar
splat-store path onto the same RVV-owned control boundary used by elementwise,
scalar-broadcast, compare/select, memory, reduction, MAcc, segment2, computed
mask, and widening conversion routes.

The result should preserve emitted semantics while ensuring AVL/VL, SEW/LMUL,
tail policy, mask policy, runtime ABI order, scalar runtime input binding,
typed vector result facts, selected capability facts, materialization facts,
and residual operand binding are validated through provider-owned facts before
any route/artifact mirror is trusted.

## Direction Source

- Direction title: `Stage2 RVV runtime scalar splat-store route-control
  provider-plan integration`.
- Module owner: RVV plugin-local route-control provider-plan consumption by the
  existing runtime scalar splat-store route-family/provider path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `853832d5 rvv: consume route control plan in widening conversion`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflow.

## Current Repository Facts

- Specs require selected `tcrv.exec` RVV variant -> typed or realized
  `tcrv_rvv` body -> RVV plugin-owned planning/provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC, with target artifacts and
  metadata as mirrors only.
- `RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan`,
  `runtimeScalarSplatStoreRouteFamilyPlan`, runtime-control facts, scalar ABI,
  typed vector result facts, splat/store materialization leaves, and residual
  operand-binding facts already exist for `runtime_i32_splat_store`.
- `RVVSelectedBodyRouteControlProviderPlan` currently exposes adopted consumer
  flags for ordinary elementwise, scalar-broadcast elementwise, base memory,
  standalone reduction, scalar-broadcast MAcc, plain compare/select,
  computed-mask select, widening conversion, computed-mask memory, and segment2
  memory.
- The current missing piece is that runtime scalar splat-store is not an
  explicit route-control consumer. The provider path already requires residual
  operand-binding facts, but the splat-store path still does not require a
  route-control plan before splat/store route statement construction.

## Scope

1. Add a runtime scalar splat-store route-control consumer marker or equivalent
   structural flag to `RVVSelectedBodyRouteControlProviderPlan`.
2. Extend route-control consumer detection to include only the existing
   `RuntimeI32SplatStore` route with `RuntimeScalarSplatStore` memory form.
3. Extend `getRVVSelectedBodyRouteControlProviderPlan(...)` so runtime scalar
   splat-store requires same-analysis runtime scalar splat-store family plan
   and materialization facts before exposing route-control facts.
4. Require the provider/route statement construction path for runtime scalar
   splat-store to consume that route-control plan before splat/store material
   is attached to `TCRVEmitCLowerableRoute`.
5. Add focused C++ coverage for positive route-control consumption and
   fail-closed diagnostics around missing/stale runtime splat-store
   family/materialization/control facts.
6. Update `.trellis/spec/extension-plugins/rvv-plugin.md` if the production
   code establishes runtime scalar splat-store as a durable route-control
   consumer.

## Requirements

1. Runtime scalar splat-store route-control construction must require the
   verified `runtimeScalarSplatStoreRouteFamilyPlan` and route materialization
   facts from the same selected route analysis.
2. The route-control plan must point at the runtime-control plan owned by the
   verified runtime scalar splat-store family plan.
3. Runtime scalar splat-store route-control construction must validate typed
   config facts, selected target capability facts, runtime AVL/VL source,
   SEW/LMUL, tail policy, mask policy, runtime ABI order, config/runtime VL
   contract mirrors, selected provider/legality mirrors, scalar splat
   materialization, store materialization, and route-family memory form before
   provider route construction.
4. Runtime scalar splat-store statement/provider construction must still fail
   closed for missing residual operand-binding facts, stale scalar/vector/type
   markers, stale runtime ABI mirrors, missing `rhs_scalar`, `out`, or `n`
   binding, missing splat/store materialized uses, and stale binding summary.
5. Route ids, metadata fields, manifests, artifact names, ABI strings,
   descriptors, scripts, tests, common EmitC, source-front-door markers, and
   legacy i32 helper spellings must not become AVL/VL, policy, scalar input,
   vector result, splat/store, dtype, or compute authority.
6. No new splat-store operation kinds, dtype/LMUL cases, high-level frontend
   lowering, runtime ABI change, dispatch/fallback change, or emitted statement
   order change is in scope.

## Acceptance Criteria

- [x] Production C++ route planning/provider code makes
      `runtime_i32_splat_store` consume `RVVSelectedBodyRouteControlProviderPlan`
      before runtime scalar splat-store route statement/provider construction.
- [x] `RVVSelectedBodyRouteControlProviderPlan` exposes a runtime scalar
      splat-store consumer marker or equivalent structural fact.
- [x] Positive C++ tests prove runtime scalar splat-store route-control
      consumption joins typed config facts, selected capability facts, runtime
      AVL/VL control, route materialization facts, runtime scalar splat-store
      family facts, and residual operand-binding facts.
- [x] Negative diagnostics cover missing runtime scalar splat-store
      family/materialization facts, stale same-analysis ownership, missing
      route-control plan, wrong runtime AVL source or role, policy mismatch,
      unsupported selected capability/config, stale scalar/vector/type or
      splat/store materialization facts, stale residual operand binding, and
      runtime ABI mirror mismatch before route/artifact authority.
- [x] Existing explicit/pre-realized runtime scalar splat-store FileCheck or
      generated artifact coverage still passes, with emitted metadata remaining
      explicit mirror labels only.
- [x] If emitted statement order, target ABI, target mirrors, generated-bundle
      behavior, or executable semantics remain unchanged, final report explains
      why historical `ssh rvv` runtime evidence remains sufficient.
- [x] Bounded scans over touched RVV planning/provider/test/spec/target/script
      files find no new name-, route-id-, metadata-, descriptor-, ABI-string-,
      script-, artifact-, common-EmitC-, source-front-door-, or legacy-i32-
      derived AVL/VL, scalar input, vector result, splat/store, policy, dtype,
      or compute authority.
- [x] `git diff --check` passes.
- [x] Focused RVV plugin tests and `check-tianchenrv` pass, or an exact
      blocker is recorded.
- [x] Task status, journal/archive, and one coherent commit complete if this
      task finishes.

## Out Of Scope

- New RVV operation coverage, scalar splat variants, conversion, contraction,
  accumulation, reduction, compare/select, computed-mask, segment2, memory, or
  dtype/LMUL clone work.
- High-level Linalg/Vector/StableHLO frontend lowering, source-front-door
  positive routes, dashboards, broad smoke matrices, helper-only/report-only
  work, or evidence-only fixture copying.
- Moving runtime scalar splat-store semantics into `tcrv.exec`, common EmitC,
  target export, scripts, metadata, manifests, descriptors, route ids, ABI
  strings, artifact names, or test names.
- Runtime/correctness/performance claims unless emitted behavior or ABI changes
  and real `ssh rvv` evidence is collected.

## Technical Approach

1. Validate and start this Trellis task.
2. Inspect current route-control, runtime scalar splat-store family,
   materialization facts, residual operand-binding facts, provider statement
   construction, selected-body realization, and focused C++ tests.
3. Add runtime scalar splat-store to the route-control consumer set.
4. Extend `getRVVSelectedBodyRouteControlProviderPlan(...)` with a bounded
   runtime scalar splat-store branch that validates same-analysis family and
   materialization facts, memory form, splat/store leaves, typed config,
   selected capability, runtime AVL/VL, and policy mirrors.
5. Require the provider splat-store branch to obtain and check the
   route-control plan before binding splat/store statement operands.
6. Extend focused C++ tests rather than broad fixture copies.
7. Update the RVV plugin spec only for the durable route-control consumer
   boundary.
8. Run focused build/tests, bounded authority scans, `git diff --check`, and
   `check-tianchenrv`.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-runtime-scalar-splat-store-route-control`
2. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
3. `./build/bin/tianchenrv-rvv-extension-plugin-test`
4. Focused lit/FileCheck or generated-bundle dry-run for runtime scalar
   splat-store only if emitted target mirrors, ABI, artifact schema, or
   generated output changes.
5. Bounded authority scan over touched RVV planning/provider/test/spec files.
6. `git diff --check`
7. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and the
  shared guide index/checklists.
- Relevant previous tasks read:
  `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-scalar-broadcast-splat-store-route-family-ownership/prd.md`,
  `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-runtime-scalar-splat-store-runtime-binding-closure/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-runtime-vl-policy-provider-plan/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-scalar-broadcast-route-control/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-widening-conversion-route-control/prd.md`.
- Initial code surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Implementation Result

- Added `controlsRuntimeScalarSplatStore` to
  `RVVSelectedBodyRouteControlProviderPlan`.
- Added runtime scalar splat-store consumer detection for the existing
  `RuntimeI32SplatStore` operation with `RuntimeScalarSplatStore` memory form.
- Extended `getRVVSelectedBodyRouteControlProviderPlan(...)` so
  `runtime_i32_splat_store` requires same-analysis runtime scalar splat-store
  family/materialization facts before route-control facts are exposed.
- The route-control owner now validates runtime scalar splat-store operation
  and memory form, typed config, selected target capability, runtime AVL/VL
  control, tail/mask policy, runtime ABI order, vector type/C type, setvl leaf,
  scalar splat leaf, store leaf, result vector facts, and result name before
  route statement construction.
- `RVVEmitCRouteProvider` now requires the runtime scalar splat-store
  route-control provider plan before its fallback splat/store statement
  construction path binds statements to `TCRVEmitCLowerableRoute`.
- Focused C++ tests now prove positive runtime splat-store route-control
  consumption and fail closed for missing family facts, stale same-analysis
  materialization, stale policy, stale selected target capability, stale
  splat/store materialization facts, stale runtime ABI mirrors, and stale
  residual operand binding.
- Updated the RVV plugin spec so runtime scalar splat-store is listed as a
  durable route-control provider-plan consumer.

## Validation Result

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-runtime-scalar-splat-store-route-control`
- [x] `git diff --check`
- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [x] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [x] Focused lit/FileCheck from `build/test`:
      `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='(runtime-i32-splat-store|runtime-scalar-splat-store)'`
      ran 5 selected tests and passed.
- [x] Generated-bundle dry-run for explicit `runtime_i32_splat_store` with
      counts `7,16,23` and RHS scalars `11,-5`:
      `artifacts/tmp/stage2_runtime_scalar_splat_store_route_control/explicit-dry-run`.
- [x] Generated-bundle dry-run for pre-realized
      `runtime_i32_splat_store` with counts `7,16,23` and RHS scalars `11,-5`:
      `artifacts/tmp/stage2_runtime_scalar_splat_store_route_control/pre-realized-dry-run`.
- [x] Bounded added-line authority scan over touched planning/provider/test/spec
      files. The only match was a stale `__riscv_vle32_v_i32m1` string in a C++
      negative materialization-mirror test; it is not production route
      authority.
- [x] `cmake --build build --target check-tianchenrv -j2`: 379/379 tests
      passed.

No real `ssh rvv` rerun was required because this round did not change emitted
statement order, target ABI, target mirror schema, generated-bundle script
behavior, runtime counts, or executable semantics. The change adds fail-closed
provider-plan ownership checks and focused tests around an existing executable
runtime scalar splat-store route; historical runtime evidence for the unchanged
route remains sufficient.
