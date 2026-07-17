# Stage2 RVV selected-body route materialization facts ownership

## Goal

Introduce one explicit RVV-owned materialization-facts boundary between the
verified selected-body route-family plans and provider-built
`TCRVEmitCLowerableRoute`. The production `RVVEmitCRouteProvider` should
consume this boundary after aggregate family-plan verification instead of
choosing the same type, header, intrinsic leaf, mask/VL, and route-shape facts
through a central chain of family-specific ternaries.

## Direction Source

- Direction title: `Stage2 RVV selected-body route materialization facts
  ownership`.
- Module owner: RVV plugin-local selected-body EmitC route materialization fact
  selection between verified route-family plans and provider-built
  `TCRVEmitCLowerableRoute`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `b8f0c93d rvv: close top-level route-family provider
  verifier`.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  Direction Brief before source edits.

## What I Already Know

- Specs require the active RVV route to flow from selected `tcrv.exec` variant
  through typed low-level `tcrv_rvv` body/config/runtime facts, RVV-owned
  legality/realization/provider, provider-built `TCRVEmitCLowerableRoute`,
  common EmitC materialization, and target artifact/evidence.
- Common EmitC/export must remain neutral. RVV type strings, headers,
  intrinsic leaves, ABI mapping, route payload, legality, selected-body
  realization, and fail-closed diagnostics remain RVV-plugin-owned.
- Commit `b8f0c93d` added `RVVSelectedBodyRouteFamilyProviderOwner` and rewired
  route construction to call one aggregate family-plan verifier.
- Current provider construction still extracts each optional family plan and
  locally selects materialization facts such as `vlCType`, result/source vector
  types, masks, `setVLLeaf`, load/store leaves, compare/merge leaves, required
  headers, and route-shape booleans.
- This task should regularize a coherent materialization-facts surface. It
  should not add route coverage or preserve old authority through compatibility
  wrappers.

## Requirements

1. Add a planning/provider-owned materialization-facts struct or equivalent
   explicit boundary, public to `RVVEmitCRouteProvider` through the RVV
   planning API.
2. The boundary must be built only after
   `verifyRVVSelectedBodyRouteFamilyProviderPlans()` accepts the analysis.
3. The boundary must preserve plan ownership and expose enough facts for the
   production provider to consume at least:
   - family plan pointers or equivalent presence facts;
   - route-shape booleans for contraction dot reduction, widening MAcc,
     computed-mask contraction, strided-input contraction, standalone
     reduction, computed-mask standalone reduction, runtime-scalar computed
     mask standalone reduction, widening conversion, plain standalone
     reduction, and computed-mask accumulation;
   - required headers;
   - VL/result/source/mask type names and C types;
   - selected `setvl`, load, store, compute, compare, merge, splat, and
     widening intrinsic leaves currently chosen by the provider prelude.
4. Rewire production `RVVEmitCRouteProvider.cpp` so these materialization facts
   are consumed by the route builder, reducing local manual family-specific
   fact selection in the central provider prelude.
5. Preserve route semantics, ABI order, route ids, target leaf/header facts,
   mirror validation, operand-binding closure, generated artifact behavior, and
   existing fail-closed diagnostics.
6. Keep the boundary RVV-local. Do not move RVV semantics into common
   EmitC/export or target metadata.
7. Preserve existing route-family verifier APIs and tests.

## Acceptance Criteria

- [x] PRD and context files reference the RVV plugin and EmitC route specs plus
      the previous top-level provider verifier task.
- [x] A named RVV materialization-facts boundary exists in the RVV planning API.
- [x] The production provider calls the aggregate top-level verifier and then
      obtains materialization facts from that explicit boundary.
- [x] The production provider consumes the new boundary for the prelude's
      required headers, type mappings, intrinsic leaves, and route-shape
      booleans.
- [x] The new boundary fails closed if a computed-mask accumulation route is
      classified as active but its shared family plan is missing.
- [x] Focused C++ tests cover representative materialization facts for at
      least memory, elementwise/select, math, runtime scalar splat-store, and
      widening conversion operations, plus the computed-mask accumulation
      fail-closed diagnostic.
- [x] Representative existing FileCheck/lit routes still pass for at least one
      memory route, one elementwise/select route, one math route, runtime
      scalar splat-store, widening conversion, and one negative diagnostic.
- [x] Active-authority scan over touched RVV planning/provider/test files finds
      no newly introduced legacy `RVVI32M1`, `rvv-i32m1`, positive finite
      `tcrv_rvv.i32_*`, descriptor/source-front-door/direct-C, or mirror-only
      route authority.
- [x] Focused build/tests, `git diff --check`, and `check-tianchenrv` pass, or
      an exact blocker is documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Non-Goals

- No route coverage expansion, new operations, dtype/LMUL clone batches, new
  verifier registries, source-front-door routes, high-level frontend lowering,
  descriptor/direct-C/source-export paths, artifact dashboards, broad smoke
  matrices, or helper-only changes without production provider consumption.
- No emitted target sequence, runtime ABI, operand order, correctness,
  runtime, or performance claim changes.
- No migration of RVV semantics into common EmitC/export.
- No rewrite of unrelated statement emission.

## Validation Plan

1. Validate and start this Trellis task.
2. Build focused C++ plugin test target.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build `tcrv-opt` and `tcrv-translate`.
5. Run focused lit/FileCheck coverage for representative selected-body routes:
   memory, elementwise/select, math, runtime splat-store, widening conversion,
   and a selected-boundary negative diagnostic.
6. Run active-authority scan over touched RVV planning/provider/test files.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant previous task read:

- `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-top-level-route-family-provider-verifier/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- representative fixtures under `test/Target/RVV`

## Definition Of Done

The selected-body RVV provider consumes a named RVV-owned
materialization-facts boundary after aggregate provider-plan verification; the
central provider prelude no longer owns the corresponding family-specific fact
selection; representative route families and fail-closed diagnostics remain
covered; focused and full checks pass; the task is finished/archived using repo
convention; and one coherent commit records the work.

## Completion Evidence

### Production Changes

- Added `RVVSelectedBodyRouteMaterializationFacts` as the explicit RVV-owned
  materialization-facts boundary in the planning API.
- Added `getRVVSelectedBodyRouteMaterializationFacts()` to gather verified
  family plan pointers, route-shape booleans, required headers, type mappings,
  mask/VL facts, and intrinsic leaves.
- Rewired production `RVVEmitCRouteProvider.cpp` so selected-body route
  construction first consumes the top-level provider-plan verifier and then
  obtains materialization facts from the new boundary.
- Preserved common EmitC neutrality: `TCRVEmitCLowerableRoute` is still
  provider-built; common EmitC/export does not infer RVV type/header/intrinsic
  semantics.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the durable
  materialization-facts boundary contract, signature, error matrix, and tests.

### Test Coverage

- Added `runRouteMaterializationFactsBoundaryTest()` to
  `test/Plugin/RVVExtensionPluginTest.cpp`.
- The new C++ coverage proves representative materialization facts for:
  - memory route-family plans;
  - elementwise/select route-family plans;
  - math/contraction route-family plans;
  - runtime scalar splat-store route-family plans;
  - widening conversion route-family plans;
  - computed-mask accumulation missing-plan fail-closed diagnostics.

### Checks Run

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-stage2-rvv-selected-body-route-materialization-facts-ownership`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] Focused lit from `build/test` with filter:
  `explicit-selected-body-artifact-strided-load-unit-store|explicit-selected-body-artifact-masked-add|explicit-selected-body-artifact-widening-macc-add|explicit-selected-body-artifact-runtime-i32-splat-store|explicit-selected-body-artifact-widen-i32-to-i64|emitc-to-cpp-selected-boundary-negative`
  passed 6/6 selected tests.
- [OK] Added-line active-authority scan over touched RVV
  planning/provider/test files found no new legacy `RVVI32M1`, `rvv-i32m1`,
  finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  descriptor/source-front-door/direct-C/source-export, or mirror-only route
  authority.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 363/363.

No `ssh rvv` evidence was run because this refactor changed provider-local
fact ownership only. It did not change emitted target sequence, runtime ABI,
materialized operands, correctness, runtime, or performance claims.
