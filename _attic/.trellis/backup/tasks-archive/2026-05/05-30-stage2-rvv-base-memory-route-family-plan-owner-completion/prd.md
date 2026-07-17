# Stage2 RVV base memory movement route-family plan owner completion

## Goal

Move production route-family plan ownership for selected-body base RVV memory
movement out of central `RVVEmitCRoutePlanning.cpp` into a dedicated RVV
plugin-local owner module. The owner covers the active unit/strided/indexed
and static-mask base memory movement family. Central planning may keep shared
typed/config/runtime fact collection, owner dispatch, generic closure checks,
and provider handoff, but it must not retain base-memory-specific plan
constants, ABI order authority, memory-form authority, stride/index/mask layout
authority, intrinsic/header/type mirror authority, target leaf/profile
authority, provider-support mirror authority, or route-description mirror
semantics.

## What I Already Know

- The worktree started clean at `1699434b rvv: move elementwise route plans to owner`.
- No current Trellis task existed, so this task was created from the Hermes
  Direction Brief before source changes.
- The archived elementwise arithmetic owner task added
  `RVVEmitCElementwiseRouteFamilyPlanOwners` and reduced central planning to
  shared facts, neutral owner dispatch, closure checks, and provider handoff
  for that family.
- Current central `RVVEmitCRoutePlanning.cpp` still owns base memory movement
  route-family plan constants, derive/apply/validate functions, route
  description mirror checks, operand binding plan IDs and role tables for base
  memory routes, runtime ABI order, target leaf/profile, required headers, C
  type mapping summaries, intrinsic leaves, memory-form/layout strings, and
  provider-plan compatibility checks.
- `RVVEmitCMemoryStatementPlanOwners.cpp` already consumes a verified base
  memory family plan for statement construction; it does not own the upstream
  route-family plan derivation requested here.

## Requirements

- Add a dedicated RVV plugin-local base memory movement route-family plan owner
  surface for the active routes:
  - `strided_load_unit_store`;
  - `unit_load_strided_store`;
  - `indexed_gather_unit_store`;
  - `indexed_scatter_unit_load`;
  - `masked_unit_load_store`;
  - `masked_unit_store`.
- The owner must derive and validate operation classification, memory-form
  classification, runtime ABI order, ABI roles, operand/mem_window bindings,
  stride/index/mask facts, dtype/config relation, target leaf/profile, required
  headers, C type mappings, intrinsic mirrors, route-description mirrors,
  provider-supported mirrors, and provider-plan compatibility from typed
  `tcrv_rvv` body/config/capability/runtime facts.
- Central route planning may call owner APIs, collect shared typed/config and
  runtime facts, run generic closure checks, and hand validated analysis to the
  provider.
- Unsupported or inconsistent inputs must fail closed with targeted
  diagnostics for stale family IDs, wrong ABI order, wrong mem_window roles,
  missing stride/index/mask bindings, wrong memory form, wrong dtype/config
  relation, wrong target leaf/profile, stale header/type/intrinsic mirrors,
  provider-plan mismatch, and route-description mirror mismatch.
- The implementation must rewire the production/default selected-body provider
  path, not only add tests or unused helper code.

## Acceptance Criteria

- [x] New owner header/source are added for base memory movement route-family
      plan ownership and are built through RVV EmitC CMake.
- [x] `RVVEmitCRoutePlanning.cpp` no longer defines base memory movement
      route-family plan constants, derive/apply/validate bodies, provider-plan
      verification bodies, base memory operand-binding plan IDs/role tables,
      exact intrinsic choices, ABI order constants, memory-form/layout
      authority, target leaf/profile mirrors, provider-supported mirrors, or
      base memory route-description mirror semantics beyond neutral dispatch.
- [x] Central route planning dispatches to the owner for plan construction,
      plan application, base memory operand bindings, route-description mirror
      validation, and provider-plan verification.
- [x] Focused C++ tests prove owner-derived strided load/unit store, unit
      load/strided store, indexed gather/unit store, indexed scatter/unit load,
      static-mask unit load/store, and static-mask unit store plans where
      already supported.
- [x] Focused C++ fail-closed coverage includes stale family ID, wrong ABI
      order, wrong mem_window/ABI role, missing stride/index/mask binding,
      wrong memory form, wrong dtype/config relation, wrong target
      leaf/profile, stale header/type/intrinsic mirrors, provider-plan
      mismatch, and route-description mirror mismatch.
- [x] Existing memory statement owners and completed route-family owners remain
      non-regressed.
- [x] Bounded authority scan shows central no longer owns base memory movement
      plan IDs/constants/ABI/binding/intrinsic choice or mirror-only route
      authority.
- [x] `git diff --check` and focused build/test pass; run `check-tianchenrv`
      unless an exact blocker is found.
- [x] Task status, journal, archive, and final commit are truthful.

## Completion Evidence

- Added `RVVEmitCBaseMemoryRouteFamilyPlanOwners` header/source and wired it
  into `TianChenRVRVVEmitCRouteProvider`.
- Central `RVVEmitCRoutePlanning.cpp` now delegates base-memory route family
  construction, apply, operand binding, provider-plan verification, and
  route-description mirror validation to the owner; central retains shared
  typed/config/runtime fact collection, owner dispatch, materialization-fact
  consumption, and generic closure checks.
- `RVVEmitCMemoryStatementPlanOwners.cpp` and control-policy planning include
  the owner surface instead of relying on base-memory declarations from the
  central route-planning header.
- `RVVExtensionPluginTest.cpp` covers owner-derived unit/strided/indexed/static
  mask base-memory plans plus stale family ID, runtime ABI order, memory form,
  target leaf/profile, binding role, binding summary, typed config, intrinsic,
  provider-plan, and route-description mirror failures.
- Bounded scan confirmed central route planning no longer defines base-memory
  route-family plan IDs/constants, base operand-binding plan IDs, base ABI
  order constants, base provider/profile/type/intrinsic mirror constants, or
  base plan derive/apply/validate/provider bodies.
- Checks passed: `cmake --build build --target TianChenRVRVVEmitCRouteProvider
  -j2`; `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  -j2`; `./build/bin/tianchenrv-rvv-extension-plugin-test`; `git diff
  --check`; `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-30-stage2-rvv-base-memory-route-family-plan-owner-completion`;
  `cmake --build build --target check-tianchenrv -j2` with 464/464 passed.

## Out Of Scope

- No new memory operations, source-front-door routes, artifact/runtime claims,
  performance work, high-level Linalg/Vector frontend lowering,
  one-intrinsic wrappers, dtype/LMUL clone batches, segment2 route-plan work,
  computed-mask memory route-plan work, compare/select work, reduction work,
  MAcc/contraction work, elementwise rework, dashboard/report work, broad smoke
  matrices, or evidence-only tasks.
- Do not move common EmitC materialization into the RVV owner.
- Do not let central/common code choose RVV memory semantics.
- Do not require `ssh rvv` unless this round makes a new runtime,
  correctness, or performance claim.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous owner pattern read:
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-elementwise-arithmetic-route-family-plan-owner-completion/prd.md`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`.
- Current statement consumer read:
  `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`.
- Current central functions to move include base memory movement classification,
  plan validation, plan derivation/application, operand binding ID/role logic,
  route-description mirror verification, and provider-plan compatibility in
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
