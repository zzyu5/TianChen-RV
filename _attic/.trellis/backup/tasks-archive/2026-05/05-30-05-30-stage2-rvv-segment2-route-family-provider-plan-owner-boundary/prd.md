# Stage2 RVV segment2 route-family provider-plan owner boundary

## Goal

Move production-active segment2 route-family provider-plan construction for
computed-mask segment2 load/store/update and plain segment2
deinterleave/interleave out of central `RVVEmitCRoutePlanning.cpp` into an
explicit RVV-owned segment2 owner boundary. Central route planning should
remain a neutral consumer and producer of shared route analysis,
materialization facts, operand-binding facts, route descriptions, and mirror
metadata; it must not own segment2 sub-family provider-plan construction
authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV segment2 route-family provider-plan owner boundary`.
- Module owner: RVV plugin-owned segment2 memory route-family provider-plan
  owner boundary.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `ed19c459 rvv: move control policy plans to owners`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The immediate predecessor
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-route-control-mask-tail-owner-boundary/prd.md`
  moved route-control and mask/tail policy provider-plan construction into
  `RVVEmitCControlPolicyPlanOwners.h/cpp`; central route planning now consumes
  that owner boundary and `check-tianchenrv` passed 464/464.
- `.trellis/spec/extension-plugins/rvv-plugin.md` now has an explicit
  `Segment2 Route-Family Planning Owner Boundary` section requiring active
  computed-mask segment2 load, computed-mask segment2 store, computed-mask
  segment2 update, plain segment2 deinterleave, and plain segment2 interleave
  to select exactly one RVV planning owner.
- The same spec requires
  `getRVVSelectedBodySegment2RouteFamilyProviderPlan(...)` to return an empty
  plan for non-consumer routes, fail closed for no/multiple/incomplete owners,
  validate typed config/runtime/memory/mask/policy/capability/operand facts,
  and build one owner-derived provider plan for the segment2 statement-plan
  boundary.
- `.trellis/spec/extension-plugins/rvv-plugin.md` also requires
  `getRVVSelectedBodySegment2MemoryRouteStatementPlan(...)` to consume the
  owner-built provider plan rather than rediscovering segment2 sub-family
  dispatch through a local operation/memory-form predicate cluster.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires
  `TCRVEmitCLowerableRoute` to be provider-built and common EmitC to remain
  neutral; common EmitC must not infer dtype, SEW, LMUL, policy, operation,
  memory form, ABI order, intrinsic spelling, or RVV schedule.
- `include/TianChenRV/Plugin/RVV/RVVEmitCControlPolicyPlanOwners.h` and
  `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp` provide the
  current owner-boundary model: an explicit header, owner structs, registry
  getter, consumer predicate, and provider-plan getter implemented outside
  central route planning.
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` currently still
  exposes `RVVSelectedBodySegment2RouteFamilyPlanningOwner`,
  `getRVVSelectedBodySegment2RouteFamilyPlanningOwners(...)`,
  `isRVVSelectedBodySegment2RouteFamilyPlanningConsumer(...)`, and
  `getRVVSelectedBodySegment2RouteFamilyProviderPlan(...)` as planning-owned
  APIs.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` currently defines the
  segment2 planning owner predicates, registry, exact-one selection, and
  family-specific provider-plan builders near the segment2 route-family block.
- `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp` already consumes
  the segment2 provider-plan getter when building segment2 memory statement
  plans, so the production consumer wiring can be preserved through a new
  owner-boundary header.
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` currently checks the
  segment2 planning consumer/getter before route construction; after this
  task, that check must go through the explicit segment2 owner boundary.
- `test/Plugin/RVVExtensionPluginTest.cpp` already has focused segment2
  registry, empty-plan, fail-closed, computed-mask segment2, and plain segment2
  provider/statement-plan tests that can be adjusted to include the new owner
  boundary and extended where needed.

## Requirements

1. Add an explicit RVV-owned segment2 route-family provider-plan owner header
   and implementation module, or an equivalent owner-local component matching
   the established control/policy owner pattern.
2. Move `RVVSelectedBodySegment2RouteFamilyPlanningOwner`, segment2 owner
   registry declarations, segment2 registry definitions, exact-once owner
   selection, and segment2 family-specific provider-plan builders out of
   `RVVEmitCRoutePlanning.h/cpp`.
3. Keep shared plan data structures in `RVVEmitCRoutePlanning.h` when they are
   genuinely neutral provider/statement-plan data shared by route provider,
   memory statement owners, tests, and target artifact validation.
4. Keep central `RVVEmitCRoutePlanning.cpp` responsible for neutral route
   analysis, typed config/capability fact collection, route-family provider
   plan verification, materialization facts, operand-binding facts, route
   descriptions, metadata mirrors, and common diagnostics.
5. Preserve exact-one segment2 owner selection for computed-mask segment2 load,
   computed-mask segment2 store, computed-mask segment2 update, plain segment2
   deinterleave, and plain segment2 interleave.
6. Preserve empty/default plan behavior for non-segment2 route descriptions.
7. Preserve fail-closed diagnostics for incomplete owners, multiple owners,
   stale or missing route-family plans, mismatched operation kind/memory form,
   segment count, mask source, arithmetic kind, runtime ABI order, route
   control plan, same-analysis operand-binding facts, required ABI roles,
   required leaves, typed config/policy/capability mirrors, and stale mirror
   metadata.
8. Preserve segment2 facts through the owner boundary: operation kind, memory
   form, segment field roles, mask and passthrough facts, runtime `n`/AVL/VL,
   ABI operand binding, typed config, capability requirements, provider route
   facts, artifact ABI facts, and mirror metadata as mirrors only.
9. Wire `RVVEmitCRouteProvider.cpp`,
   `RVVEmitCMemoryStatementPlanOwners.cpp`, tests, and any other consumer
   through the segment2 owner-boundary header rather than the central route
   planning header for owner registry/getter APIs.
10. Do not add new segment2 operations, new direct-route-entry positive paths,
    source-front-door routes, high-level Linalg/frontend lowering,
    one-intrinsic wrappers, dtype/LMUL clone batches, dashboards, broad smoke
    matrices, report-only work, or common EmitC semantic choices.

## Acceptance Criteria

- [x] New RVV-owned segment2 owner module/header owns segment2
      route-family provider-plan construction.
- [x] `RVVEmitCRoutePlanning.cpp` no longer defines
      `getRVVSelectedBodySegment2RouteFamilyProviderPlan`,
      `getRVVSelectedBodySegment2RouteFamilyPlanningOwners`,
      `isRVVSelectedBodySegment2RouteFamilyPlanningConsumer`, or the moved
      segment2 family-specific provider-plan builder/predicate functions.
- [x] `RVVEmitCRoutePlanning.h` no longer exposes segment2 planning owner
      registry structs/functions as central planning-owned APIs; consumers use
      the explicit segment2 owner-boundary header.
- [x] Shared segment2 provider-plan and statement-plan data structs remain
      available in the correct shared layer without moving unrelated neutral
      route-analysis or materialization code.
- [x] Central RoutePlanning remains a neutral analysis/materialization/
      operand-binding consumer/provider and does not rebuild segment2
      family-specific provider-plan authority.
- [x] Focused C++ tests cover exact-one owner selection, owner order/names,
      non-null predicate/builder hooks, empty/no-owner behavior,
      multiple-owner or incomplete-owner diagnostics where reachable, stale or
      inconsistent fact diagnostics, and representative computed-mask and
      plain segment2 consumer paths.
- [x] Representative generated-bundle dry-runs pass for the segment2
      selected-body paths touched: computed-mask segment2 load/store/update
      and plain segment2 deinterleave/interleave.
- [x] Bounded central scan proves no moved segment2 owner structs, registries,
      exact-one selectors, or family-specific builders remain in
      `RVVEmitCRoutePlanning.cpp` except neutral consumer/getter calls if any.
- [x] Bounded owner-module scan proves segment2 consumers/builders/getters are
      concentrated in the new owner boundary.
- [x] Bounded authority scan over touched production/test files finds no new
      legacy-i32, source-front-door/source-artifact, descriptor-derived,
      direct-route-entry-only, route-id, artifact-name, exact-intrinsic,
      bare status/supported, script-derived, ABI-string, common-EmitC, or
      mirror-only authority drift.
- [x] `git diff --check` passes.
- [x] `tianchenrv-rvv-extension-plugin-test`, `tcrv-opt`, and
      `tcrv-translate` build.
- [x] The RVV extension plugin C++ test passes.
- [x] `check-tianchenrv` passes, or the exact blocker is recorded.
- [x] `ssh rvv` is not required unless this round changes executable
      behavior or makes a new runtime, correctness, or performance claim.

## Technical Approach

The intended production flow after this task is:

```text
selected RVV segment2 variant
  -> typed/realized tcrv_rvv segment2 body/config/runtime facts
  -> route-family provider plan verification
  -> route materialization facts
  -> memory operand-binding facts
  -> RVV-owned route-control provider plan
  -> RVV-owned segment2 route-family provider-plan owner boundary
  -> RVV-owned segment2 statement plan
  -> aggregate statement owner attach
  -> RVVEmitCRouteProvider builds TCRVEmitCLowerableRoute
  -> common EmitC materialization
```

Implementation will add a segment2 owner header under
`include/TianChenRV/Plugin/RVV/` and a matching source file under
`lib/Plugin/RVV/EmitC/`. The new header will expose the segment2 planning
owner struct, registry getter, consumer predicate, and provider-plan getter.
The source file will contain the moved predicates, builders, exact-one owner
selection, and validation currently resident in central route planning. The
provider and memory statement owner modules will include this boundary header
for segment2 owner APIs. CMake will compile the new source into
`TianChenRVRVVEmitCRouteProvider`.

## Validation Plan

1. Validate task context:
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-05-30-stage2-rvv-segment2-route-family-provider-plan-owner-boundary`.
2. Build focused plugin test:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`.
3. Run focused C++ plugin coverage:
   `./build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build touched production tools:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
5. Run representative generated-bundle dry-runs for computed-mask segment2
   load/store/update and plain segment2 deinterleave/interleave selected-body
   paths.
6. Run bounded scans for central residual owner authority and new owner-module
   concentration.
7. Run bounded authority drift scans over touched RVV production/test files.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Out of Scope

- New segment2 route coverage, dtype/SEW/LMUL/policy expansion, runtime
  behavior changes, or high-level frontend integration.
- Moving all memory route-family verification, route materialization facts, or
  operand-binding facts out of route planning.
- Reworking completed route-control/mask-tail owner code except necessary
  include/consumer integration.
- Rewriting common EmitC materialization, target artifact export semantics,
  selected-body realization semantics, or runtime executable behavior.
- IME, Offload, TensorExt, Toy, Template, future plugin work, dashboards,
  artifact indexes, broad smoke matrices, or runtime performance claims.

## Completion Evidence

- Added `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`
  and `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp` as the
  explicit RVV-owned segment2 route-family provider-plan owner boundary.
- Moved segment2 planning owner structs, owner registry declarations,
  registry definitions, exact-one owner selection, consumer predicates, and
  provider-plan builders for computed-mask segment2 load/store/update plus
  plain segment2 deinterleave/interleave out of central
  `RVVEmitCRoutePlanning.h/cpp`.
- Kept shared segment2 provider-plan and statement-plan data structs in
  `RVVEmitCRoutePlanning.h`; route analysis, typed config/capability facts,
  materialization facts, operand-binding facts, route descriptions, and mirror
  metadata remain central neutral planning responsibilities.
- Updated `RVVEmitCRouteProvider.cpp`,
  `RVVEmitCMemoryStatementPlanOwners.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp` to consume the segment2 owner API
  through the explicit owner-boundary header. CMake now compiles the new owner
  source into `TianChenRVRVVEmitCRouteProvider`.
- Reused the existing focused C++ segment2 owner/consumer tests, which cover
  owner registry size/order/names, non-null predicate and builder hooks,
  exact-one classification for all five active segment2 families, empty
  non-consumer plans, missing computed-mask update provider-plan diagnostics,
  computed-mask segment2 provider/statement-plan consumers, and plain segment2
  provider/statement-plan consumers.
- Spec update review found no `.trellis/spec/**` edit was needed: the existing
  RVV plugin spec already defines the segment2 route-family planning owner
  boundary with signature, contracts, validation/error matrix, good/base/bad
  cases, and tests.
- Checks run:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-05-30-stage2-rvv-segment2-route-family-provider-plan-owner-boundary`;
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`;
  `./build/bin/tianchenrv-rvv-extension-plugin-test`;
  `cmake --build build --target tcrv-opt tcrv-translate -j2`;
  focused lit filter for five representative segment2 generated-bundle
  dry-runs; bounded central/owner/authority scans; `git diff --check`; and
  `cmake --build build --target check-tianchenrv -j2`.
- Representative generated-bundle dry-run result: 5/5 passed for
  computed-mask segment2 load, computed-mask segment2 store, explicit
  computed-mask segment2 update, pre-realized plain segment2 deinterleave, and
  pre-realized plain segment2 interleave.
- Final `check-tianchenrv` result: 464/464 passed.
- `clang-format` was not available in this environment, but `git diff --check`
  passed.
- `ssh rvv` was not run because this task changed owner placement only and did
  not make a new runtime, correctness, or performance claim.
