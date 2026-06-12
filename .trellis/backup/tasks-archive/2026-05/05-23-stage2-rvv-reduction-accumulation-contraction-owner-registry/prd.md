# Stage2 RVV reduction/accumulation/contraction route-family owner registry extraction

## Goal

Extract one explicit RVV plugin-local route-family owner registry for the
mature math cluster in selected-body EmitC planning/provider code. The registry
must cover contraction, standalone reduction, and computed-mask accumulation
family verifiers, and production route construction must consume the aggregate
owner verifier instead of manually sequencing those per-family verifiers.

## Direction Source

- Direction title: `Stage2 RVV reduction/accumulation/contraction route-family owner registry extraction`.
- Module owner: RVV plugin-local selected-body EmitC route-family ownership
  boundary for standalone reduction, computed-mask accumulation, and
  contraction.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `2386dabc rvv: extract elementwise select route-family owner registry`.
- No `.trellis/.current-task` existed, so this task was created from the
  supplied Direction Brief before source edits.

## What I Already Know

- Specs require the active RVV route to flow from selected `tcrv.exec` variant
  through typed low-level `tcrv_rvv` body/config/runtime facts, RVV-owned
  legality/realization/provider, provider-built `TCRVEmitCLowerableRoute`,
  common EmitC materialization, and target artifact/evidence.
- Common EmitC/export must stay neutral and must not infer RVV semantics,
  dtype/config, intrinsic spelling, ABI roles, route support, or acceptance
  state from route ids, metadata mirrors, artifact names, helper strings, or
  common exporter branches.
- Commit `2bd2fc6d` extracted the memory-family owner registry; commit
  `2386dabc` extracted the elementwise/select owner registry.
- Current planning code already defines per-family plans and verifiers for:
  - `RVVSelectedBodyContractionRouteFamilyPlan`;
  - `RVVSelectedBodyStandaloneReductionRouteFamilyPlan`;
  - `RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan`.
- Current production provider still manually calls the three mature math
  verifiers in central route construction:
  - `verifyRVVSelectedBodyContractionRouteFamilyProviderPlans`;
  - `verifyRVVSelectedBodyStandaloneReductionRouteFamilyProviderPlans`;
  - `verifyRVVSelectedBodyComputedMaskAccumulationRouteFamilyProviderPlans`.
- Existing C++ provider tests already exercise the individual verifier
  surfaces for contraction, standalone reduction, and computed-mask
  accumulation. This task should preserve those per-family tests and add a
  focused aggregate owner-registry test.

## Requirements

1. Add a planning-owned math route-family owner registry or equivalent
   explicit owner table for exactly these active families:
   - contraction;
   - standalone reduction;
   - computed-mask accumulation.
2. Each owner entry must expose its family name, consumer predicate, and
   provider-plan verifier through planning-owned APIs.
3. Add an aggregate math-cluster consumer predicate and provider-plan verifier
   that dispatch through the owner registry.
4. The central provider route construction path must consume the aggregate
   math-cluster verifier instead of manually calling the three per-family
   verifiers.
5. Existing per-family verifier APIs may remain public because focused tests
   and adjacent route-family tests use them, but they must be registered
   through the aggregate owner boundary.
6. Preserve current route semantics, route ids, ABI order, typed-body
   authority, target leaf/header facts, `RouteOperandBindingPlan` closure,
   generated artifact behavior, and fail-closed diagnostics.
7. Keep contraction, standalone reduction, and computed-mask accumulation
   ownership distinct. The registry is owner dispatch only; it must not merge
   dot/macc relation facts, scalar-reduction seed/result facts, mask
   producer/source facts, inactive-lane contracts, runtime ABI order,
   dtype/config facts, or intrinsic mapping across families.
8. Do not move RVV semantics into common EmitC/export or target metadata.

## Acceptance Criteria

- [x] PRD and task context reference the RVV plugin, EmitC route,
      plugin-interface/locality, testing specs, and directly relevant archived
      owner/family tasks.
- [x] A planning-owned math route-family owner registry or equivalent explicit
      owner table exists for contraction, standalone reduction, and
      computed-mask accumulation.
- [x] `verifyRVVSelectedBodyReductionAccumulationContractionRouteFamilyProviderPlans`
      or an equivalent aggregate verifier dispatches through the registered
      owners and fails closed for missing/stale plans across all three
      families.
- [x] The production provider uses the aggregate math-cluster verifier instead
      of manual per-family verifier calls for this cluster.
- [x] Existing per-family plan validation still catches stale mirrors and
      binding-closure mismatches for representative contraction, standalone
      reduction, and computed-mask accumulation routes.
- [x] C++ provider tests prove registry membership, owner names, aggregate
      consumer classification, aggregate missing-plan rejection, and aggregate
      stale-plan rejection for representative family members.
- [x] Representative FileCheck/lit coverage for existing explicit or
      pre-realized standalone reduction, computed-mask accumulation, and
      contraction artifact paths still passes.
- [x] At least one fail-closed diagnostic check remains covered for an
      unsupported or mismatched typed body before materialization.
- [x] Focused build/tests, `git diff --check`, and `check-tianchenrv` pass, or
      an exact blocker is documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Non-Goals

- No new route families or route coverage.
- No new reductions, contractions, dtype/LMUL clone batches, or intrinsic case
  growth.
- No memory or elementwise/select registry churn beyond direct integration
  needs.
- No source-front-door, descriptor/direct-C/source-export, dashboard, broad
  smoke matrix, or helper-only abstraction without production consumers.
- No Linalg/Vector frontend lowering or high-level kernel op work.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, exact intrinsic spelling, route id, artifact name, or
  mirror-only route authority.
- No semantic changes to mask, reduction, contraction, accumulation, ABI,
  runtime `n`/AVL, dispatch/fallback, or common EmitC/export behavior.

## Validation Plan

1. Validate this Trellis task and task context.
2. Build focused targets that cover `tianchenrv-rvv-extension-plugin-test`,
   `tcrv-opt`, and `tcrv-translate` as needed.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run focused FileCheck/lit tests for representative explicit/pre-realized:
   - standalone reduction;
   - computed-mask standalone reduction/accumulation;
   - computed-mask MAcc accumulation;
   - widening MAcc / widening dot-reduce contraction.
5. Run at least one existing negative selected-body diagnostic fixture for
   fail-closed typed-body behavior before materialization.
6. Run an active-authority scan over touched RVV planning/provider/test files,
   `git diff --check`, and `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/index.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/plugin-protocol/locality-contract.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/index.md`
- `.trellis/spec/guides/index.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant archived tasks read:

- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-memory-route-family-owner-registry/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-elementwise-select-route-family-owner-registry/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-widening-dot-reduce-route-family-ownership/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- representative fixtures under `test/Target/RVV`

## Definition Of Done

The active reduction/accumulation/contraction route-family owners are explicit
and registry-driven at the planning/provider boundary, the production provider
consumes the aggregate owner verifier, current route semantics and diagnostics
are preserved, focused tests and `check-tianchenrv` pass, the task is
finished/archived using repo convention, and one coherent commit records the
work.

## Completion Evidence

### Production Changes

- Added
  `RVVSelectedBodyReductionAccumulationContractionRouteFamilyOwner` with
  consumer and provider-plan verifier hooks.
- Added
  `getRVVSelectedBodyReductionAccumulationContractionRouteFamilyOwners()` with
  exactly three active planning-owned entries:
  - contraction;
  - standalone reduction;
  - computed-mask accumulation.
- Added aggregate
  `isRVVSelectedBodyReductionAccumulationContractionRouteFamilyConsumer()` and
  `verifyRVVSelectedBodyReductionAccumulationContractionRouteFamilyProviderPlans()`.
- Rewired production `RVVEmitCRouteProvider.cpp` so selected-body EmitC route
  construction consumes the aggregate math-cluster verifier instead of
  manually calling the three per-family verifiers.
- Preserved existing per-family verifiers, typed-body route facts, ABI order,
  route ids, target leaf/header mirrors, binding closure, and common EmitC
  neutrality.

### Test Coverage

- Added `runReductionAccumulationContractionRouteFamilyOwnerRegistryTest()` to
  `test/Plugin/RVVExtensionPluginTest.cpp`.
- The new C++ coverage proves:
  - registry cardinality and explicit owner names;
  - non-null consumer and verifier hooks;
  - owner classification for contraction, standalone reduction, and
    computed-mask accumulation;
  - aggregate consumer classification across all three families;
  - aggregate missing-plan rejection for representative contraction,
    standalone reduction, and computed-mask accumulation routes;
  - aggregate stale-plan rejection for non-consumer routes carrying any of the
    three math-family plans.

### Checks Run

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-23-stage2-rvv-reduction-accumulation-contraction-owner-registry`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] Focused FileCheck/lit from `build/test` with filter:
  `explicit-selected-body-artifact-standalone-reduce-add|pre-realized-selected-body-artifact-standalone-reduce-add|explicit-selected-body-artifact-computed-mask-standalone-reduce-add|pre-realized-selected-body-artifact-computed-mask-standalone-reduce-add|explicit-selected-body-artifact-computed-masked-macc-add|pre-realized-selected-body-artifact-computed-masked-macc-add|explicit-selected-body-artifact-widening-macc-add|pre-realized-selected-body-artifact-widening-macc-add|explicit-selected-body-artifact-widening-dot-reduce-add|pre-realized-selected-body-artifact-widening-dot-reduce-add|emitc-to-cpp-selected-boundary-negative`
  passed 11/11 selected tests.
- [OK] Added-line active-authority scan over touched RVV
  planning/provider/test files found no new legacy `RVVI32M1`, `rvv-i32m1`,
  finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/source-front-door/
  direct-C, or mirror-only route authority.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 363/363.

No new `ssh rvv` runtime evidence was run because this refactor changed only
provider-plan ownership dispatch and did not change emitted target sequence,
runtime ABI, materialized operands, correctness, runtime, or performance
claims.

### Spec Update Judgment

No `.trellis/spec/` update was required in this round. The existing
`.trellis/spec/extension-plugins/rvv-plugin.md` Route-Family Owner Boundaries
section already defines the durable owner registry contract, aggregate
verifier behavior, common EmitC neutrality, and required tests; this task is
the reduction/accumulation/contraction application of that existing contract.
