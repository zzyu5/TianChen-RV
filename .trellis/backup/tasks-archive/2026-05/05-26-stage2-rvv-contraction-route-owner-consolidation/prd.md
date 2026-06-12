# Stage2 RVV contraction route-family production owner consolidation

## Goal

Make the active widening-dot/reduction contraction provider boundary explicit
in production C++ by separating provider-owned direct contraction planning and
validation from statement assembly. The RVV provider must consume a structured
direct contraction provider plan before constructing `TCRVEmitCLowerableRoute`,
then pass that plan into the direct contraction statement owner.

This round is a production owner consolidation for already-supported
contraction routes. It is not a new contraction variant, new evidence-only
closure, frontend route, source-front-door route, or common EmitC semantic
change.

## Direction Source

- Direction title: `Stage2 RVV contraction route-family production owner consolidation`.
- Module owner: RVV plugin production C++ planning/provider boundary for
  widening-dot/reduction contraction route planning and provider validation.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `7b19c06b rvv: validate computed-mask strided dot-reduce boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes/User brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflow.

## Current Repository Facts

- The previous task completed executable evidence for
  `computed_masked_strided_input_widening_dot_reduce_add` at counts
  `0,7,16,23`, but did not change production C++ because current provider code
  already carried the boundary.
- Current production code has:
  - `RVVSelectedBodyContractionRouteFamilyPlan`;
  - top-level route-family provider owner registry;
  - route-control provider owner registry with contraction as a consumer;
  - math operand-binding facts for widening MAcc and widening dot-reduction;
  - a direct contraction route-provider owner that assembles provider-ready
    statements.
- The remaining maturity gap is interface shape: direct contraction statement
  construction still combines provider-fact aggregation/validation with loop
  statement assembly inside `buildDirectContractionRouteStatementPlan(...)`.
- `RVVEmitCRouteProvider.cpp` currently creates the generic
  `TCRVEmitCLowerableRoute` before calling the direct contraction statement
  owner. For this task, the direct contraction provider plan should be consumed
  first, while common EmitC remains only a neutral materialization consumer.

## Requirements

1. Add a named RVV plugin-owned direct contraction provider-plan surface, or
   equivalent structured interface, in the include/lib RVV planning/provider
   boundary.
2. The provider plan must be consumed by the production provider before
   `TCRVEmitCLowerableRoute` construction.
3. The provider plan must carry or validate, for the active direct contraction
   routes:
   - operation kind and family classification;
   - computed-mask presence/source/predicate facts;
   - unit-stride versus strided-input payload memory form;
   - lhs/rhs stride ABI bindings when present;
   - source dtype/SEW/LMUL and result dtype/SEW/LMUL facts;
   - widening MAcc versus widening dot-reduction relation;
   - accumulator seed/layout and scalar/vector result layout;
   - runtime `n`/AVL/VL policy and route-control facts;
   - runtime ABI order and bound logical operands;
   - materialization leaves for setvl, source loads, compare, masked product,
     merge, widening product, seed splat, reduction/MAcc compute, and store;
   - provider-owned intrinsic/type/header facts and mirror fields only after
     route construction.
4. The direct contraction statement owner must consume the provider plan rather
   than locally rediscovering route-control, binding, mask/stride, dtype,
   accumulator/result, and materialization dependencies.
5. Route construction must fail closed before statement assembly for missing or
   stale contraction family plan, route-control plan, materialization facts,
   math operand-binding facts, mask facts, stride facts, dtype/relation facts,
   accumulator/result facts, runtime ABI facts, selected capability facts, or
   unsupported contraction shape.
6. Existing explicit and pre-realized contraction routes must keep the same
   emitted sequence, runtime ABI, dispatch/fallback behavior, route ids, target
   artifacts, and correctness semantics.
7. Common EmitC, target artifact export, scripts, ABI strings, route ids,
   artifact names, descriptors, source-front-door markers, status fields, and
   mirror metadata must remain non-authoritative.

## Acceptance Criteria

- [x] `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` exposes a named
      direct contraction provider-plan surface.
- [x] `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` builds and validates that
      plan from same-analysis contraction family, materialization,
      route-control, math operand-binding, typed config, selected capability,
      runtime ABI, mask, stride, dtype/relation, accumulator/result, and leaf
      facts.
- [x] `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` obtains the direct
      contraction provider plan before `TCRVEmitCLowerableRoute` construction
      and passes it to the direct contraction statement owner.
- [x] The direct contraction statement owner assembles statements from the
      provider plan and no longer owns the provider fact aggregation itself.
- [x] Focused C++ tests cover positive provider-plan consumption for
      `widening_dot_reduce_add`,
      `computed_masked_widening_dot_reduce_add`,
      `strided_input_widening_dot_reduce_add`, and
      `computed_masked_strided_input_widening_dot_reduce_add`, with
      `widening_macc_add` preserved as the existing direct contraction sibling.
- [x] Focused C++ fail-closed tests cover at least missing family plan, stale
      same-analysis/materialization facts, missing math binding facts, missing
      materialization leaf, stale policy/capability facts, stale mask facts,
      stale stride facts, stale dtype/relation facts, and accumulator/result
      mismatch through the provider-plan boundary.
- [x] Representative generated-bundle or lit non-regression for existing direct
      contraction paths still passes. Real `ssh rvv` is rerun only if emitted
      code, ABI, or artifact facts change.
- [x] Bounded authority scan over touched planning/provider/test files finds no
      newly introduced legacy i32, source-front-door, descriptor, ABI-string,
      script, artifact-name, common-EmitC, metadata-only, status-only, or
      route-id-derived authority.
- [x] `git diff --check` passes.
- [x] Focused RVV plugin test and `check-tianchenrv` pass, or an exact blocker
      is recorded.

## Out Of Scope

- No new contraction operation coverage, dtype/LMUL clone batches, unsigned or
  floating variants, high-level Linalg/Vector/StableHLO frontend lowering,
  source-front-door positive route, dashboard, report, route coverage matrix,
  broad smoke expansion, or evidence-only fixture copying.
- No wrapper-only plan that is not used by the production provider path.
- No migration of RVV semantics into common EmitC/export, target metadata,
  route ids, ABI strings, artifact names, scripts, descriptors, source-front
  doors, test names, or legacy i32 helper names.
- No emitted target sequence, runtime ABI, dispatch/fallback, or computation
  semantic change unless a real provider bug forces it and the change is
  evidenced.

## Technical Approach

1. Validate and start the Trellis task.
2. Add `RVVSelectedBodyDirectContractionRouteProviderPlan` to the RVV planning
   API.
3. Extract the provider-fact validation currently embedded in direct
   contraction statement assembly into
   `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)`.
4. Change direct contraction statement assembly to consume the provider plan.
5. Change `RVVEmitCRouteProvider` to obtain the plan before creating
   `TCRVEmitCLowerableRoute`, then attach the resulting direct contraction
   statements as before.
6. Extend focused C++ tests in `test/Plugin/RVVExtensionPluginTest.cpp`.
7. Run focused build/test, authority scan, `git diff --check`, and
   `check-tianchenrv`.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-stage2-rvv-contraction-route-owner-consolidation`
2. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
3. `./build/bin/tianchenrv-rvv-extension-plugin-test`
4. Focused generated-bundle/lit non-regression for direct contraction only if
   emitted artifact facts change.
5. Bounded authority scan over touched RVV planning/provider/test files.
6. `git diff --check`
7. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Archived tasks read:
  `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-widening-dot-reduce-route-family-ownership/prd.md`,
  `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-math-operand-binding-surface-ownership/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-contraction-route-control-provider-plan/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-direct-provider-contraction-route-provider-owner/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-computed-mask-strided-widening-dot-reduce-boundary/prd.md`.
- Workspace journal read:
  `.trellis/workspace/codex/journal-15.md`, especially sessions 224, 228,
  235, 236, and 237.
- Initial implementation surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Definition Of Done

The production RVV provider consumes a structured direct contraction
provider-plan boundary before route construction; direct contraction statement
assembly consumes that plan without rediscovering provider facts; focused
positive and fail-closed coverage passes; no non-authoritative route source is
introduced; Trellis task state is truthful; final checks pass; and one coherent
commit records the work if all acceptance criteria are met.

## Implementation Summary

- Added `RVVSelectedBodyDirectContractionRouteProviderPlan` to the RVV EmitC
  route-planning API.
- Added
  `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)` as the
  provider-owned validation surface for direct contraction routes.
- Moved direct contraction provider-fact aggregation ahead of
  `TCRVEmitCLowerableRoute` construction in `RVVEmitCRouteProvider.cpp`.
- Changed the direct contraction statement-plan API to consume the provider
  plan rather than rediscovering route-control, math binding, mask, stride,
  dtype/relation, accumulator/result, runtime ABI, and materialization facts.
- Extended focused C++ RVV extension plugin checks for positive plan
  consumption and fail-closed diagnostics.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to document the
  direct contraction provider-plan API as the durable production boundary.

## Verification Results

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-stage2-rvv-contraction-route-owner-consolidation`
  passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 7 --runtime-count 16 --runtime-count 23 --artifact-root artifacts/tmp/stage2_rvv_contraction_route_owner_consolidation --run-id direct-contraction-provider-plan-dry --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
  passed with `rvv_generated_bundle_abi_e2e: dry_run_success`.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 7 --runtime-count 16 --runtime-count 23 --artifact-root artifacts/tmp/stage2_rvv_contraction_route_owner_consolidation --run-id direct-contraction-provider-plan-ssh --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
  passed on `ssh rvv` with `PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,7,16,23 lhs_stride=2 rhs_stride=3`.
- Bounded authority scans over touched production/test/spec files found no new
  positive route authority from legacy i32, source-front-door, descriptor,
  ABI-string, script, artifact-name, common-EmitC, metadata-only, status-only,
  or route-id-derived sources. Matches were spec prohibitions, existing
  fail-closed tests, or existing fail-closed production diagnostics.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed with `381/381`
  tests.
- `clang-format` was attempted but no `clang-format` binary was available in
  this environment; the C++ targets and full project check passed.
