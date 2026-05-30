# Stage2 RVV MAcc route-family plan owner completion

## Goal

Move production route-family plan ownership for the active selected-body RVV
MAcc family out of central `RVVEmitCRoutePlanning.cpp` into the RVV
plugin-local MAcc owner module. The owner covers plain MAcc, scalar-broadcast
MAcc, computed-mask MAcc, and runtime-scalar computed-mask MAcc plan authority.
Central route planning may keep shared typed/config/runtime fact collection,
neutral owner dispatch, generic closure checks, materialization-fact
consumption, and provider handoff, but it must not retain MAcc-specific plan
constants, ABI order authority, accumulator/result layout authority,
computed-mask MAcc producer/suffix contract authority, intrinsic/header/type
mirror authority, provider-support mirror authority, or route-description
mirror semantics.

## What I Already Know

- The worktree started clean at `f2415433 rvv: move base memory route plans to owner`.
- No current Trellis task existed, so this task was created from the Hermes
  Direction Brief before source changes.
- The archived base-memory owner task moved route-family plan constants,
  derive/apply/validate logic, operand-binding authority, provider-plan
  compatibility, and route-description mirror validation into a dedicated
  owner module while leaving central route planning as neutral fact plumbing.
- `RVVEmitCMAccRouteFamilyPlanOwners.{h,cpp}` already exist and currently own
  MAcc route-family owner registry, provider-plan verification, and
  MAcc-specific route operand-binding plan IDs/roles.
- Current central `RVVEmitCRoutePlanning.cpp` still owns plain/scalar-broadcast
  MAcc route-family plan IDs/constants, computed-mask MAcc target/profile/type
  constants, MAcc runtime ABI order constants, plan validation bodies,
  derive/apply bodies, route-description mirror checks, and forwarding
  validation wrappers used by the MAcc owner.

## Requirements

- Extend `RVVEmitCMAccRouteFamilyPlanOwners` so it owns MAcc route-family plan
  classification, plan IDs/constants, runtime ABI order rules,
  typed/config/dtype checks, accumulator/result layout checks, provider mirror
  fields, derive/apply/validate bodies, route-description mirror validation,
  operand-binding plan ownership, and provider-plan compatibility for:
  - `MAccAdd`;
  - `ScalarBroadcastMAccAdd`;
  - `ComputedMaskedMAccAdd`;
  - `RuntimeScalarComputedMaskedMAccAdd`.
- The owner must derive facts from typed `tcrv_rvv` body/config/capability and
  runtime ABI facts already present in `RVVSelectedBodyRouteAnalysis`, not from
  operation names, route IDs, artifact metadata, descriptors, ABI strings,
  exact intrinsic spelling, common EmitC choices, or fixture residue.
- Central route planning may dispatch to the MAcc owner for plan construction,
  plan application, route-description mirror validation, operand bindings, and
  provider-plan verification, but must not own MAcc-specific semantic choices.
- Unsupported or inconsistent MAcc inputs must fail closed with targeted
  diagnostics for stale family ID, wrong ABI order or role, missing accumulator
  binding, wrong memory form, wrong dtype/config, wrong accumulator/result
  layout, wrong mask/compare binding, stale intrinsic/type/header mirrors,
  provider-plan mismatch, and stale central-route-planning semantic residue.
- The implementation must rewire the production/default selected-body provider
  path, not only add helper APIs or tests.

## Acceptance Criteria

- [x] `RVVEmitCMAccRouteFamilyPlanOwners.{h,cpp}` own plain MAcc,
      scalar-broadcast MAcc, computed-mask MAcc, and runtime-scalar
      computed-mask MAcc route-family plan derive/apply/validate logic.
- [x] `RVVEmitCRoutePlanning.cpp` no longer defines MAcc route-family plan
      IDs/constants, MAcc runtime ABI order constants, MAcc target/profile/type
      mirror constants, MAcc derive/apply/validate bodies, or MAcc
      route-description mirror semantics beyond neutral dispatch.
- [x] Central route planning dispatches to the MAcc owner for selected-body
      MAcc plan construction/application and route-description verification.
- [x] Focused C++ tests prove owner-derived route-family plan facts,
      route-description mirrors, operand binding facts, and provider-plan
      compatibility for representative plain, scalar-broadcast,
      computed-mask, and runtime-scalar computed-mask MAcc routes.
- [x] Focused fail-closed coverage includes stale family ID, wrong ABI
      order/role, missing binding facts, wrong memory form, wrong dtype/config,
      wrong accumulator/result layout, stale intrinsic/type/header mirrors,
      provider-plan mismatch, and MAcc owner exclusion of widening/contraction
      MAcc authority.
- [x] Bounded authority scan shows central route planning no longer owns MAcc
      route-family plan IDs/constants or MAcc derive/apply/validate/provider
      bodies except dispatch/generic fact plumbing.
- [x] `git diff --check`, focused RVV route-provider build, focused
      `tianchenrv-rvv-extension-plugin-test`, task validation, and
      `check-tianchenrv` pass unless an exact blocker is found.
- [x] Task status, journal/archive, and final commit are truthful.

## Out Of Scope

- No new MAcc coverage, direct-route-entry migration, selected-body
  realization variant, base-memory work, widening dot work, segment2 work,
  elementwise work, standalone reduction expansion, high-level
  Linalg/frontend lowering, one-intrinsic wrapper dialects, dashboards,
  broad smoke matrices, or evidence-only work.
- Do not move generic route-planning facts, common route materialization, or
  common EmitC mechanics into the MAcc owner.
- Do not require `ssh rvv` unless this round makes a new runtime,
  correctness, or performance claim.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/implementation-stack/compiler-stack-contract.md`.
- Previous owner pattern read:
  `.trellis/tasks/archive/2026-05/05-30-stage2-rvv-base-memory-route-family-plan-owner-completion/prd.md`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h`,
  and `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`.
- Current source files inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.
