# Stage2 RVV memory route-family owner registry extraction

## Goal

Extract one explicit RVV plugin-local memory route-family owner registry for
the active memory-family selected-body EmitC planning/provider path. The
registry must cover the already closed base memory movement, computed-mask
memory, and plain segment2 memory families, and the production provider should
consume the registry/aggregate verifier instead of manually wiring each memory
family verifier in the central provider path.

## Direction Source

- Direction title: `Stage2 RVV route-family owner registry extraction for
  selected-body EmitC planning`.
- Module owner: RVV plugin-local selected-body EmitC route-family ownership
  boundary for base memory movement, computed-mask memory, and plain segment2
  memory.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `9d566f91 rvv: close explicit plain segment2 selected bodies`.
- No `.trellis/.current-task` existed, so this task was created from the
  supplied Direction Brief before source edits.

## What I Already Know

- Specs require the active RVV path to flow from selected `tcrv.exec` variant
  through typed low-level `tcrv_rvv` body/config/runtime facts, RVV-owned
  legality/realization/provider, provider-built `TCRVEmitCLowerableRoute`,
  common EmitC materialization, and target artifact/evidence.
- Common EmitC/export must stay neutral and must not infer RVV semantics,
  dtype/config, intrinsic spelling, ABI roles, route support, or acceptance
  state from route ids, metadata mirrors, artifact names, helper strings, or
  common exporter branches.
- Archived tasks show that base memory movement, computed-mask memory, and
  plain segment2 memory already have family plans and provider validation:
  - `0846a4d0 rvv: close base memory provider binding`
  - `f0002e21 rvv: close computed-mask memory provider binding`
  - `5e06dd2c rvv: close plain segment2 memory provider binding`
  - `9d566f91 rvv: close explicit plain segment2 selected bodies`
- Current source already defines:
  - `RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan`
  - `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan`
  - `RVVSelectedBodySegment2MemoryRouteFamilyPlan`
  - consumer predicates and per-family provider-plan verifiers.
- Current provider still calls memory-family verification through a manually
  coupled sequence: an aggregate memory verifier plus separate base,
  computed-mask, and segment2 verifier calls. The aggregate verifier currently
  checks computed-mask and plain segment2 plans, but not base memory movement,
  even though `isRVVSelectedBodyMemoryRouteFamilyConsumer` includes base
  memory movement consumers. That is the bounded production coupling this
  round will repair.

## Requirements

1. Add a small planning-owned memory route-family owner registry or equivalent
   explicit owner table for exactly these active memory families:
   - base memory movement;
   - computed-mask memory;
   - plain segment2 memory.
2. Each owner entry must expose its family name, consumer predicate, and
   provider-plan verifier through planning-owned APIs.
3. The aggregate memory provider verifier must dispatch through this registry
   and must cover all three memory families, including base memory movement.
4. The central provider route construction path must consume the aggregate
   memory verifier rather than repeating one call per memory family.
5. Existing per-family verifiers may remain public because focused tests and
   adjacent route-family tests use them, but they must be registered through
   the aggregate owner boundary.
6. Preserve current route semantics, route ids, ABI order, typed-body
   authority, target leaf/header facts, `RouteOperandBindingPlan` closure,
   generated artifact behavior, and fail-closed diagnostics.
7. Keep base memory movement, computed-mask memory, and plain segment2 memory
   ownership distinct. The registry is neutral owner dispatch only; it must not
   collapse mask, segment, index, stride, ABI, dtype/config, or inactive-lane
   semantics across families.
8. Do not move RVV semantics into common EmitC/export or target metadata.

## Acceptance Criteria

- [x] PRD and task context reference the RVV plugin, EmitC route,
      plugin-interface, testing specs, and the directly relevant archived
      memory-family tasks.
- [x] A planning-owned memory route-family owner registry or equivalent
      explicit owner table exists for base memory movement, computed-mask
      memory, and plain segment2 memory.
- [x] `verifyRVVSelectedBodyMemoryRouteFamilyProviderPlans` dispatches through
      the registered owners and fails closed for missing/stale plans across all
      three memory families.
- [x] The production provider uses the aggregate memory verifier instead of
      manual per-memory-family verifier calls.
- [x] Existing per-family plan validation still catches stale mirrors and
      binding-closure mismatches for representative base memory,
      computed-mask memory, and plain segment2 routes.
- [x] C++ provider tests prove registry membership, aggregate consumer
      classification, aggregate missing-plan rejection for all three memory
      families, and aggregate stale-plan rejection for non-consumers.
- [x] Representative FileCheck/lit coverage for base memory, computed-mask
      memory, and plain segment2 explicit/pre-realized artifact paths still
      passes.
- [x] At least one fail-closed diagnostic check remains covered for an
      unsupported or mismatched typed memory body before materialization.
- [x] Focused build/tests, `git diff --check`, and `check-tianchenrv` pass, or
      an exact blocker is documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Non-Goals

- No new route families or route coverage.
- No dtype/LMUL clone batches.
- No source-front-door, descriptor/direct-C/source-export, dashboard, broad
  smoke matrix, or helper-only abstraction without production consumers.
- No Linalg/Vector frontend lowering or high-level kernel op work.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, exact intrinsic spelling, route id, artifact name, or
  mirror-only route authority.
- No semantic changes to mask, segment, stride, index, ABI, runtime `n`/AVL,
  dispatch/fallback, or common EmitC/export behavior.

## Validation Plan

1. Validate and start this Trellis task.
2. Build focused targets that cover `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test` as needed.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run focused lit/FileCheck tests for representative base memory,
   computed-mask memory, plain segment2, and negative memory fixtures.
5. Run generated-bundle dry-runs for representative affected memory routes if
   provider route materialization behavior changes.
6. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant archived tasks read:

- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-base-memory-movement-route-family-ownership/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-computed-mask-memory-route-family-interface/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-memory-route-family-ownership-extraction/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-plain-segment2-explicit-selected-body-artifact/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- focused fixtures under `test/Target/RVV` and `test/Scripts`.

## Definition Of Done

The active memory route-family owners are explicit and registry-driven at the
planning/provider boundary, the production provider consumes the aggregate
owner verifier, current memory route semantics and diagnostics are preserved,
focused tests and `check-tianchenrv` pass, the task is finished/archived using
repo convention, and one coherent commit records the work.

## Completion Evidence

### Production Changes

- Added `RVVSelectedBodyMemoryRouteFamilyOwner` with consumer and provider-plan
  verifier hooks.
- Added `getRVVSelectedBodyMemoryRouteFamilyOwners()` with exactly three active
  planning-owned entries:
  - base memory movement;
  - computed-mask memory;
  - plain segment2 memory.
- Rewired `isRVVSelectedBodyMemoryRouteFamilyConsumer` to derive aggregate
  memory-family membership from the owner registry.
- Rewired `verifyRVVSelectedBodyMemoryRouteFamilyProviderPlans` to dispatch
  through the owner registry, so base memory movement, computed-mask memory,
  and plain segment2 memory all fail closed through one aggregate provider
  boundary.
- Removed manual per-memory-family verifier calls from production
  `RVVEmitCRouteProvider.cpp`; selected-body EmitC route construction now
  consumes the aggregate memory owner verifier.
- Preserved existing per-family verifiers, typed-body route facts, ABI order,
  route ids, target leaf/header mirrors, binding closure, and common EmitC
  neutrality.
- Added a concise code-spec section in
  `.trellis/spec/extension-plugins/rvv-plugin.md` documenting the durable
  route-family owner registry contract and required tests.

### Test Coverage

- Added `runMemoryRouteFamilyOwnerRegistryTest()` to
  `test/Plugin/RVVExtensionPluginTest.cpp`.
- The new C++ coverage proves:
  - registry cardinality and explicit owner names;
  - non-null consumer and verifier hooks;
  - isolated owner classification for base memory, computed-mask memory, and
    plain segment2 memory;
  - aggregate consumer classification across all three families;
  - aggregate missing-plan rejection for representative base, computed-mask,
    and plain segment2 memory routes;
  - aggregate stale-plan rejection for non-consumer routes carrying any of the
    three memory-family plans.

### Checks Run

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused lit/FileCheck from `build/test` with filter:
  `rvv-extension-plugin|explicit-selected-body-artifact-strided-load-unit-store|pre-realized-selected-body-artifact-strided-load-unit-store|explicit-selected-body-artifact-computed-masked-indexed-gather-load|pre-realized-selected-body-artifact-computed-masked-segment2-load|explicit-selected-body-artifact-segment2-deinterleave-unit-store|pre-realized-selected-body-artifact-segment2-interleave-unit-load|explicit-selected-body-computed-mask-memory-negative|explicit-selected-body-segment2-interleave-negative`
  passed 9/9 selected tests.
- [OK] Focused generated-bundle dry-run lit filter:
  `rvv-generated-bundle-abi-e2e-pre-realized-strided-load-unit-store-dry-run|rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-indexed-gather-load-dry-run|rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-load-dry-run|rvv-generated-bundle-abi-e2e-pre-realized-segment2-interleave-unit-load-dry-run|rvv-generated-bundle-abi-e2e-pre-realized-segment2-deinterleave-unit-store-dry-run`
  passed 5/5 selected tests.
- [OK] Added-line active-authority scan over touched RVV planning/provider/test
  files found no new legacy `RVVI32M1`, `rvv-i32m1`, finite
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/source-front-door/direct-C,
  or mirror-only route authority.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 363/363.
- No new `ssh rvv` runtime evidence was run because this refactor changed only
  provider-plan ownership dispatch and did not change emitted target sequence,
  runtime ABI, or materialized operands.
