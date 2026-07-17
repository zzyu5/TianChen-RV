# Stage2 RVV top-level route-family provider verifier closure

## Goal

Close the selected-body RVV EmitC provider verifier at one top-level,
planning-owned route-family owner boundary. The production provider should
consume a single aggregate verifier that dispatches to the already extracted
memory, elementwise/select, reduction/accumulation/contraction, runtime scalar
splat-store, and widening conversion family verifiers.

## Direction Source

- Direction title: `Stage2 RVV top-level route-family provider verifier
  closure`.
- Module owner: RVV plugin-local selected-body EmitC route-family provider
  verification boundary.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: one untracked planning task directory:
  `.trellis/tasks/05-24-stage2-rvv-top-level-route-family-provider-verifier/`.
- Initial HEAD: `646e1e71 rvv: extract math route-family owner registry`.
- `.trellis/.current-task` was absent, so this existing planning task was
  repaired before source edits.

## What I Already Know

- Specs require the active RVV route to flow from selected `tcrv.exec`
  variant through typed low-level `tcrv_rvv` body/config/runtime facts,
  RVV-owned legality/realization/provider, provider-built
  `TCRVEmitCLowerableRoute`, common EmitC materialization, and target
  artifact/evidence.
- Common EmitC/export must stay neutral and must not infer RVV semantics,
  dtype/config, intrinsic spelling, ABI roles, route support, or acceptance
  state from route ids, metadata mirrors, artifact names, helper strings, or
  common exporter branches.
- Recent commits extracted aggregate owner registries for:
  - memory route families: `2bd2fc6d`;
  - elementwise/select route families: `2386dabc`;
  - reduction/accumulation/contraction route families: `646e1e71`.
- Current production provider still manually sequences five top-level verifier
  calls before route construction:
  - `verifyRVVSelectedBodyMemoryRouteFamilyProviderPlans`;
  - `verifyRVVSelectedBodyReductionAccumulationContractionRouteFamilyProviderPlans`;
  - `verifyRVVSelectedBodyElementwiseSelectRouteFamilyProviderPlans`;
  - `verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyProviderPlans`;
  - `verifyRVVSelectedBodyWideningConversionRouteFamilyProviderPlans`.
- Runtime scalar splat-store and widening conversion are mature single-family
  verifier boundaries, but they are not included in a top-level owner list yet.
- This task is closure of verifier dispatch shape only. It should not add route
  coverage, change emitted code, change ABI order, or require new `ssh rvv`
  evidence.

## Requirements

1. Add a planning-owned top-level selected-body route-family provider owner
   registry or equivalent explicit owner table.
2. The registry must include exactly the active top-level provider verifier
   boundaries:
   - memory;
   - reduction/accumulation/contraction;
   - elementwise/select;
   - runtime scalar splat-store;
   - widening conversion.
3. Each top-level owner entry must expose a family name, consumer predicate over
   `RVVSelectedBodyOperationKind`, and provider-plan verifier over
   `RVVSelectedBodyRouteAnalysis`.
4. Add a top-level aggregate consumer predicate and provider-plan verifier that
   dispatch through the registry.
5. Rewire production selected-body EmitC route construction to call only the
   top-level aggregate verifier instead of manually sequencing the five
   verifier boundaries.
6. Preserve all existing cluster and per-family verifier APIs because focused
   tests and adjacent route-family tests use them directly.
7. Preserve route semantics, route ids, ABI order, typed-body authority,
   target leaf/header facts, mirror validation, `RouteOperandBindingPlan`
   closure, generated artifact behavior, and fail-closed diagnostics.
8. Keep family semantics distinct. The top-level registry is dispatch/locality
   structure only; it must not merge memory, mask, compare/select,
   scalar-splat, conversion, reduction, accumulation, contraction, dtype/config,
   intrinsic, ABI, or inactive-lane semantics.
9. Do not move RVV semantics into common EmitC/export or target metadata.

## Acceptance Criteria

- [x] PRD and task context reference the RVV plugin, EmitC route,
      plugin-interface/locality, testing specs, and directly relevant archived
      owner/family tasks.
- [x] A planning-owned top-level route-family provider owner registry or
      equivalent explicit owner table exists with exactly the five active
      provider verifier boundaries.
- [x] The top-level aggregate consumer predicate is registry-backed and covers
      representative memory, elementwise/select, math, runtime splat-store, and
      widening conversion operations.
- [x] The top-level aggregate provider verifier dispatches through the
      registered owners and preserves fail-closed missing-plan and stale-plan
      behavior.
- [x] Production selected-body EmitC route construction consumes the top-level
      aggregate verifier instead of manually sequencing the five verifier
      boundaries.
- [x] Existing per-cluster and per-family verifier tests remain meaningful and
      continue to catch stale mirrors and binding-closure mismatches.
- [x] C++ provider tests prove registry membership, owner names, non-null
      hooks, aggregate consumer classification, missing-plan dispatch, and at
      least one stale-plan dispatch through the top-level verifier.
- [x] Representative FileCheck/lit coverage for existing selected-body
      artifact paths still passes if provider route construction behavior is
      touched.
- [x] Focused build/tests, `git diff --check`, and `check-tianchenrv` pass, or
      an exact blocker is documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Non-Goals

- No new route families, operations, dtype/LMUL clone batches, or coverage
  expansion.
- No modification to emitted target sequence, runtime ABI, materialized
  operands, generated-bundle semantics, correctness claims, runtime claims, or
  performance claims.
- No source-front-door, descriptor/direct-C/source-export, dashboard, broad
  smoke matrix, helper-only, or report-only work.
- No Linalg/Vector frontend lowering or high-level kernel op work.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, exact intrinsic spelling, route id, artifact name, or
  mirror-only route authority.
- No semantic changes to mask, memory form, scalar splat-store, conversion,
  reduction, accumulation, contraction, ABI, runtime `n`/AVL,
  dispatch/fallback, or common EmitC/export behavior.

## Validation Plan

1. Validate and start this Trellis task.
2. Build focused targets that cover `tianchenrv-rvv-extension-plugin-test`,
   `tcrv-opt`, and `tcrv-translate` as needed.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run focused FileCheck/lit tests for representative routes from each
   top-level verifier boundary if provider route construction is rewired.
5. Run at least one existing selected-body negative diagnostic fixture to
   confirm fail-closed behavior before materialization.
6. Run an active-authority scan over touched RVV planning/provider/test files,
   `git diff --check`, and `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/plugin-protocol/locality-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant archived tasks read:

- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-memory-route-family-owner-registry/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-elementwise-select-route-family-owner-registry/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-reduction-accumulation-contraction-owner-registry/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-scalar-broadcast-splat-store-route-family-ownership/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-widening-conversion-route-family-ownership/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- representative fixtures under `test/Target/RVV`

## Definition Of Done

The selected-body RVV provider has one top-level, planning-owned
route-family verifier boundary; production route construction consumes that
aggregate verifier; current route semantics and diagnostics are preserved;
focused tests and `check-tianchenrv` pass; the task is finished/archived using
repo convention; and one coherent commit records the work.

## Completion Evidence

### Production Changes

- Added `RVVSelectedBodyRouteFamilyProviderOwner` with consumer and
  provider-plan verifier hooks.
- Added `getRVVSelectedBodyRouteFamilyProviderOwners()` with exactly five
  active top-level verifier owners:
  - memory;
  - reduction/accumulation/contraction;
  - elementwise/select;
  - runtime scalar splat-store;
  - widening conversion.
- Added aggregate
  `isRVVSelectedBodyRouteFamilyProviderConsumer()` and
  `verifyRVVSelectedBodyRouteFamilyProviderPlans()`.
- Rewired production `RVVEmitCRouteProvider.cpp` so selected-body EmitC route
  construction consumes the top-level aggregate verifier instead of manually
  calling the five verifier boundaries.
- Preserved existing cluster and per-family verifier APIs, typed-body route
  facts, ABI order, route ids, target leaf/header mirrors, binding closure,
  and common EmitC neutrality.

### Test Coverage

- Added `runTopLevelRouteFamilyProviderOwnerRegistryTest()` to
  `test/Plugin/RVVExtensionPluginTest.cpp`.
- The new C++ coverage proves:
  - registry cardinality and explicit owner names;
  - non-null consumer and verifier hooks;
  - isolated owner classification for memory, math, elementwise/select,
    runtime scalar splat-store, and widening conversion boundaries;
  - aggregate consumer classification across representative active owner
    boundaries;
  - missing-plan dispatch for all five top-level owners;
  - stale-plan dispatch through the top-level verifier.

### Spec Update

- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to record the
  durable top-level provider owner registry contract, API shape, and required
  tests.

### Checks Run

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-stage2-rvv-top-level-route-family-provider-verifier`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] Focused FileCheck/lit from `build/test` with filter:
  `explicit-selected-body-artifact-strided-load-unit-store|explicit-selected-body-artifact-masked-add|explicit-selected-body-artifact-widening-macc-add|explicit-selected-body-artifact-runtime-i32-splat-store|explicit-selected-body-artifact-widen-i32-to-i64|emitc-to-cpp-selected-boundary-negative`
  passed 6/6 selected tests.
- [OK] Added-line active-authority scan over touched RVV
  planning/provider/test files found no new legacy `RVVI32M1`, `rvv-i32m1`,
  finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/source-front-door/
  direct-C, or mirror-only route authority.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 363/363.

No new `ssh rvv` runtime evidence was run because this refactor changed only
provider-plan verifier dispatch and did not change emitted target sequence,
runtime ABI, materialized operands, correctness, runtime, or performance
claims.
