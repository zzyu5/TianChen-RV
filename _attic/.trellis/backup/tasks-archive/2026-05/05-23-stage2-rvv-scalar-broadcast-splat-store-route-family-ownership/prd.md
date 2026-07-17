# Stage2 RVV scalar-broadcast and runtime splat-store route-family ownership

## Goal

Own the active RVV scalar-broadcast elementwise and runtime scalar splat-store
route families through RVV plugin-local planning and provider validation. The
production provider must materialize these routes only from validated family
plans, and stale or missing plans must fail closed instead of allowing mirror
metadata, route ids, helper names, or common EmitC/export code to decide
broadcast or splat-store semantics.

## Direction Source

Hermes direction brief supplied on 2026-05-23:

- module owner: RVV plugin-local scalar-broadcast elementwise and runtime scalar
  splat-store family boundary for active routes
- active route focus: `scalar_broadcast_add`, `scalar_broadcast_sub`,
  `scalar_broadcast_mul`, and `runtime_i32_splat_store`
- previous baseline: computed-mask compare/select family ownership at
  `78f3adcd rvv: own computed-mask select route family`

## Current Facts To Verify

- The worktree starts clean at `78f3adcd`.
- Recent completed route-family ownership patterns exist for memory,
  contraction, reduction, accumulation, and computed-mask select families.
- Scalar-broadcast and runtime splat-store explicit or pre-realized fixtures are
  expected to exist already.
- This task must inventory only active scalar-broadcast and runtime splat-store
  production routes plus directly adjacent provider predicates before editing.

## Requirements

- Inventory active scalar-broadcast and runtime scalar splat-store production
  routes and the directly adjacent provider predicates.
- Add or repair RVV planning-owned consumer predicates for the included route
  families.
- Add or repair family plan verification so provider materialization depends on
  validated scalar-broadcast or splat-store plans.
- Reject missing family plans for route-family consumers.
- Reject stale scalar-broadcast or splat-store family plans attached to
  non-consumers.
- Preserve operation kind, scalar RHS or scalar store parameter role, runtime ABI
  order, AVL/VL behavior, dtype/config, intrinsic leaf/type/header facts,
  result/store layout, target leaf/header mirrors, and
  `RouteOperandBindingPlan` closure.
- Keep RVV semantics in RVV plugin planning/provider/realization/target support;
  common EmitC/export must remain neutral.
- If runtime splat-store proves separable from scalar-broadcast within this
  round, finish scalar-broadcast first and leave a precise continuation point
  for splat-store.

## Non-goals

- No new dtype or LMUL clone expansion.
- No conversion, memory movement, compare/select, contraction, reduction, MAcc,
  frontend/Linalg, source-front-door, dashboard, broad smoke-matrix, helper-only,
  or report-only work.
- No route id, ABI order, computation semantics, scalar value semantics,
  runtime `n`/AVL behavior, dispatch/fallback behavior, target artifact
  contract, or common EmitC/export neutrality changes.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, descriptor
  residue, helper string, artifact name, or mirror-only acceptance authority.

## Acceptance Criteria

- [x] Active scalar-broadcast and runtime splat-store routes are inventoried
      from current source and tests.
- [x] Provider materialization for included consumers requires a validated
      family plan.
- [x] Missing family plans on consumers fail closed with targeted diagnostics.
- [x] Stale scalar-broadcast or splat-store plans on non-consumers fail closed.
- [x] Focused C++ or lit/FileCheck coverage proves consumer classification,
      plan ids, scalar RHS/store roles, runtime ABI order, intrinsic leaf/type
      and header mirror stability, binding closure, and missing/stale-plan
      rejection.
- [x] Generated-bundle dry-runs cover counts 7, 16, and 23 for representative
      explicit and pre-realized scalar-broadcast routes, and
      `runtime_i32_splat_store` if included.
- [x] Real `ssh rvv` evidence covers at least add and mul scalar-broadcast
      representatives, and `runtime_i32_splat_store` if included.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      shows no legacy route authority regression.
- [x] `check-tianchenrv`, `git diff --check`, and clean git status pass.

## Definition Of Done

- Implementation is bounded to the PRD module behavior.
- Focused tests and runtime evidence are recorded in task notes or journal.
- The Trellis task is finished and archived when complete.
- One coherent git commit records the completed task.

## Completion Evidence

### Inventory

- Active scalar-broadcast elementwise consumers:
  `scalar_broadcast_add`, `scalar_broadcast_sub`, and `scalar_broadcast_mul`.
- Active runtime scalar splat-store consumer: `runtime_i32_splat_store`.
- Directly adjacent provider predicates are now public planning-owned consumer
  predicates plus provider-plan verifiers for those two route families.
- Runtime splat-store stayed in this round because it already shared the same
  low-risk route-family ownership shape and had explicit/pre-realized fixtures.

### Implementation

- Added scalar-broadcast and runtime scalar splat-store family plan ids to
  planning structs, route descriptions, generated artifact metadata, and
  generated-bundle expected metadata.
- Provider materialization now calls the two new RVV planning-owned verifiers
  before building `TCRVEmitCLowerableRoute`.
- Verifiers require validated family plans for consumers, reject stale plans on
  non-consumers, and preserve operation kind, memory form, runtime ABI order,
  AVL/VL control plan, target leaf/header mirrors, scalar RHS/store roles,
  intrinsic leaves, result/store layout, runtime ABI parameters, and
  `RouteOperandBindingPlan` closure.
- Common EmitC/export remains neutral; no route ids, ABI order, computation
  semantics, scalar value semantics, dispatch/fallback behavior, or target
  artifact contracts were changed.

### Evidence

- Focused C++ build and unit smoke:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2` and
  `build/bin/tianchenrv-rvv-extension-plugin-test`.
- Focused tools build:
  `cmake --build build --target tcrv-opt tcrv-translate -j2`.
- Focused lit/FileCheck from `build/test`: 10/10 selected scalar-broadcast,
  runtime splat-store, and generated-bundle script tests passed.
- Generated-bundle dry-runs passed for explicit and pre-realized
  `scalar_broadcast_add`, `scalar_broadcast_sub`, `scalar_broadcast_mul`, and
  `runtime_i32_splat_store` with counts `7,16,23` and RHS scalars `-37,91`.
- Real `ssh rvv` evidence passed for explicit `scalar_broadcast_add`,
  `scalar_broadcast_mul`, and `runtime_i32_splat_store` with counts
  `7,16,23` and RHS scalars `-37,91`, including runtime `n` variation and
  tail preservation for splat-store.
- Active-authority scan over touched RVV/plugin/script/test paths found only
  existing negative guardrails and fail-closed checks for descriptor,
  source-front-door, direct-C, `RVVI32M1`, `rvv-i32m1`, and
  `tcrv_rvv.i32_*` strings; no new positive legacy authority was introduced.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`,
  `git diff --check`, and
  `cmake --build build --target check-tianchenrv -j2` passed; full project lit
  result was 361/361.

### Self-Repair

- Moved provider-facing verifier definitions out of the anonymous namespace
  after the focused C++ build exposed ambiguous declarations.
- Rebuilt `tcrv-opt` and `tcrv-translate` before lit after the first focused
  lit attempt observed stale tool output.
- Removed brittle target-header FileCheck additions and kept the checks on
  stable route-plan metadata mirrors.

## Technical Starting Points

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-compare-select-route-family-ownership/`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- focused scalar-broadcast and runtime splat-store fixtures under
  `test/Target/RVV` and `test/Scripts`
