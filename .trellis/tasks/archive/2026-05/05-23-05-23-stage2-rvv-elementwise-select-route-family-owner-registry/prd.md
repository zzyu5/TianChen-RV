# Stage2 RVV elementwise/select route-family owner registry extraction

## Goal

Extract one explicit RVV plugin-local elementwise/select route-family owner
registry for the mature selected-body EmitC planning/provider path. The
registry covers the already closed non-memory compute/select cluster:
elementwise arithmetic, scalar-broadcast elementwise, plain compare-select, and
computed-mask select. The production provider should consume the aggregate
owner verifier instead of manually sequencing those per-family verifiers.

## Direction Source

- Direction title: `Stage2 RVV elementwise/select route-family owner registry
  extraction`.
- Module owner: RVV plugin-local selected-body EmitC route-family ownership
  boundary for mature non-memory elementwise/select families.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `2bd2fc6d rvv: extract memory route-family owner registry`.
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
- Commit `2bd2fc6d` extracted the memory-family owner registry and made the
  provider consume `verifyRVVSelectedBodyMemoryRouteFamilyProviderPlans`.
- Current planning code already defines per-family plans and verifiers for:
  - `RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan`;
  - `RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan`;
  - `RVVSelectedBodyPlainCompareSelectRouteFamilyPlan`;
  - `RVVSelectedBodyComputedMaskSelectRouteFamilyPlan`.
- Current provider still manually calls the four mature compute/select
  verifiers in central route construction after the aggregate memory verifier.
  That manual list is the production coupling this task removes.

## Requirements

1. Add a small planning-owned elementwise/select route-family owner registry or
   equivalent explicit owner table for exactly these active families:
   - elementwise arithmetic;
   - scalar-broadcast elementwise;
   - plain compare-select;
   - computed-mask select.
2. Each owner entry must expose its family name, consumer predicate, and
   provider-plan verifier through planning-owned APIs.
3. Add an aggregate elementwise/select consumer predicate and provider-plan
   verifier that dispatch through the owner registry.
4. The central provider route construction path must consume the aggregate
   elementwise/select verifier instead of repeating one verifier call per
   family.
5. Existing per-family verifiers may remain public because focused tests and
   adjacent route-family tests use them, but they must be registered through
   the aggregate owner boundary.
6. Preserve current route semantics, route ids, ABI order, typed-body
   authority, target leaf/header facts, `RouteOperandBindingPlan` closure,
   generated artifact behavior, and fail-closed diagnostics.
7. Keep elementwise arithmetic, scalar-broadcast, plain compare-select, and
   computed-mask select ownership distinct. The registry is owner dispatch
   only; it must not collapse mask producer/source facts, scalar broadcast
   facts, stride/masked arithmetic facts, compare/select facts, dtype/config
   facts, runtime ABI order, or intrinsic mapping across families.
8. Do not move RVV semantics into common EmitC/export or target metadata.

## Acceptance Criteria

- [x] PRD and task context reference the RVV plugin, EmitC route,
      plugin-interface, testing specs, and the directly relevant archived
      memory-family owner task.
- [x] A planning-owned elementwise/select route-family owner registry or
      equivalent explicit owner table exists for elementwise arithmetic,
      scalar-broadcast elementwise, plain compare-select, and computed-mask
      select.
- [x] `verifyRVVSelectedBodyElementwiseSelectRouteFamilyProviderPlans`
      dispatches through the registered owners and fails closed for
      missing/stale plans across all four families.
- [x] The production provider uses the aggregate elementwise/select verifier
      instead of manual per-family verifier calls for this cluster.
- [x] Existing per-family plan validation still catches stale mirrors and
      binding-closure mismatches for representative elementwise arithmetic,
      scalar-broadcast, plain compare-select, and computed-mask select routes.
- [x] C++ provider tests prove registry membership, aggregate consumer
      classification, aggregate missing-plan rejection, and aggregate
      stale-plan rejection for representative family members.
- [x] Representative FileCheck/lit coverage for existing explicit or
      pre-realized elementwise, scalar-broadcast, plain compare-select, and
      computed-mask select artifact paths still passes.
- [x] At least one fail-closed diagnostic check remains covered for an
      unsupported or mismatched typed body before materialization.
- [x] Focused build/tests, `git diff --check`, and `check-tianchenrv` pass, or
      an exact blocker is documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Non-Goals

- No new route families or route coverage.
- No dtype/LMUL clone batches.
- No reduction/contraction registry work.
- No memory registry churn beyond direct integration needs.
- No source-front-door, descriptor/direct-C/source-export, dashboard, broad
  smoke matrix, or helper-only abstraction without production consumers.
- No Linalg/Vector frontend lowering or high-level kernel op work.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, exact intrinsic spelling, route id, artifact name, or
  mirror-only route authority.
- No semantic changes to mask, scalar broadcast, compare/select, stride, ABI,
  runtime `n`/AVL, dispatch/fallback, or common EmitC/export behavior.

## Validation Plan

1. Validate and start this Trellis task.
2. Build focused targets that cover `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test` as needed.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run focused lit/FileCheck tests for representative explicit/pre-realized
   elementwise, scalar-broadcast, plain compare-select, and computed-mask
   select artifact paths plus at least one negative selected-body diagnostic.
5. Run generated-bundle dry-runs for representative affected routes only if
   provider materialization behavior changes.
6. Run an active-authority scan over touched RVV planning/provider/test files,
   `git diff --check`, and `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant archived task read:

- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-memory-route-family-owner-registry/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- representative fixtures under `test/Target/RVV`

## Definition Of Done

The active elementwise/select route-family owners are explicit and
registry-driven at the planning/provider boundary, the production provider
consumes the aggregate owner verifier, current route semantics and diagnostics
are preserved, focused tests and `check-tianchenrv` pass, the task is
finished/archived using repo convention, and one coherent commit records the
work.

## Completion Evidence

### Production Changes

- Added `RVVSelectedBodyElementwiseSelectRouteFamilyOwner` with consumer and
  provider-plan verifier hooks.
- Added `getRVVSelectedBodyElementwiseSelectRouteFamilyOwners()` with exactly
  four active planning-owned entries:
  - elementwise arithmetic;
  - scalar-broadcast elementwise;
  - plain compare-select;
  - computed-mask select.
- Added aggregate
  `isRVVSelectedBodyElementwiseSelectRouteFamilyConsumer()` and
  `verifyRVVSelectedBodyElementwiseSelectRouteFamilyProviderPlans()`.
- Rewired production `RVVEmitCRouteProvider.cpp` so selected-body EmitC route
  construction consumes the aggregate elementwise/select verifier instead of
  manually calling the four per-family verifiers.
- Preserved existing per-family verifiers, typed-body route facts, ABI order,
  route ids, target leaf/header mirrors, binding closure, and common EmitC
  neutrality.

### Test Coverage

- Added `runElementwiseSelectRouteFamilyOwnerRegistryTest()` to
  `test/Plugin/RVVExtensionPluginTest.cpp`.
- The new C++ coverage proves:
  - registry cardinality and explicit owner names;
  - non-null consumer and verifier hooks;
  - isolated owner classification for elementwise arithmetic,
    scalar-broadcast elementwise, plain compare-select, and computed-mask
    select;
  - aggregate consumer classification across all four families;
  - aggregate missing-plan rejection for representative elementwise,
    scalar-broadcast, plain compare-select, and computed-mask select routes;
  - aggregate stale-plan rejection for non-consumer routes carrying any of the
    four elementwise/select family plans.

### Checks Run

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused FileCheck for explicit and pre-realized selected-body artifact
  paths:
  - `explicit-selected-body-artifact-masked-add.mlir`
  - `pre-realized-selected-body-artifact-masked-add.mlir`
  - `explicit-selected-body-artifact-scalar-broadcast-add.mlir`
  - `pre-realized-selected-body-artifact-scalar-broadcast-add.mlir`
  - `explicit-selected-body-artifact-cmp-select.mlir`
  - `pre-realized-selected-body-artifact-cmp-select.mlir`
  - `explicit-selected-body-artifact-computed-mask-select-sle.mlir`
  - `pre-realized-selected-body-artifact-computed-mask-select.mlir`
- [OK] Focused fail-closed FileCheck:
  `emitc-to-cpp-selected-boundary-negative.mlir`
- [OK] Added-line active-authority scan over touched RVV
  planning/provider/test files found no new legacy `RVVI32M1`, `rvv-i32m1`,
  finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/source-front-door/
  direct-C, or artifact-authority terms.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 363/363.
- No new `ssh rvv` runtime evidence was run because this refactor changed only
  provider-plan ownership dispatch and did not change emitted target sequence,
  runtime ABI, materialized operands, correctness, runtime, or performance
  claims.

### Spec Update Judgment

No `.trellis/spec/` update was required in this round. The existing
`.trellis/spec/extension-plugins/rvv-plugin.md` Route-Family Owner Boundaries
section already defines the durable owner registry contract, aggregate
verifier behavior, common EmitC neutrality, and required tests; this task is
the elementwise/select application of that existing contract.
