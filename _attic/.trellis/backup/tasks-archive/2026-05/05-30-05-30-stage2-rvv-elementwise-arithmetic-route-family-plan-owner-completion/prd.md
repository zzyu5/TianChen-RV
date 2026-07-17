# Stage2 RVV elementwise arithmetic route-family plan owner completion

## Goal

Move production route-family plan ownership for plain vector elementwise
arithmetic and scalar-broadcast elementwise arithmetic out of central
`RVVEmitCRoutePlanning.cpp` into a dedicated RVV plugin-local owner surface.
The central planner may continue collecting shared typed/config/capability
facts, dispatching to owner APIs, performing neutral closure checks, and handing
validated analysis to the provider, but it must not retain elementwise route
family constants, ABI order authority, intrinsic/type/header mirror authority,
scalar-broadcast binding authority, or family-plan mirror verification bodies.

## What I Already Know

- The worktree started clean at `d25df000 rvv: move contraction route plans to owner`.
- There is no current Trellis task; this task was created from the current
  Hermes Direction Brief before source changes.
- The preceding contraction task moved widening-contraction plan construction,
  validation, ABI order, dtype/widening relation, target/profile, intrinsic
  mirrors, and provider-plan checks into
  `RVVEmitCContractionRouteFamilyPlanOwners`.
- Current `RVVEmitCRoutePlanning.cpp` still owns elementwise arithmetic and
  scalar-broadcast elementwise plan IDs, runtime ABI orders, target leaf
  profiles, header/C type summaries, intrinsic mirror checks, derive/apply
  functions, and provider-plan verification bodies.
- Existing elementwise arithmetic statement-owner files are consumers/shape
  references, not the route-family plan owner requested here.

## Requirements

- Add a dedicated RVV plugin-local route-family plan owner surface for:
  - plain vector elementwise arithmetic (`add`, `sub`, `mul`);
  - masked/strided elementwise arithmetic already covered by the existing
    elementwise arithmetic plan shape;
  - scalar-broadcast elementwise arithmetic (`scalar_broadcast_add`,
    `scalar_broadcast_sub`, `scalar_broadcast_mul`).
- Owner code must derive and validate family plan ID, operation support,
  runtime ABI order, operand-binding expectations, ABI roles, scalar-broadcast
  runtime role, dtype/config relation, target leaf/profile, required headers,
  C type mapping, intrinsic mirrors, route-description mirrors, and
  provider-plan compatibility from typed `tcrv_rvv` body/config/capability and
  runtime facts.
- Central route planning may call the owner, collect shared facts, perform
  generic route-description closure checks, and dispatch provider verification.
- Unsupported or inconsistent inputs must fail closed with targeted diagnostics.
- The implementation must be production rewiring, not tests-only coverage or a
  helper that is unused by the default provider path.

## Acceptance Criteria

- [x] New owner header/source are added for elementwise route-family plan
      ownership and are built through RVV EmitC CMake.
- [x] `RVVEmitCRoutePlanning.cpp` no longer defines elementwise arithmetic or
      scalar-broadcast elementwise route-family plan constants, derive/apply
      functions, provider-plan verification bodies, exact intrinsic choices,
      or scalar-broadcast runtime ABI authority.
- [x] Central route planning dispatches to the owner for plan construction,
      plan application, and elementwise/select provider-plan verification.
- [x] Focused C++ tests prove plain and scalar-broadcast elementwise plans are
      owner-derived, provider-plan verified, and fail closed for stale family
      ID, wrong ABI order, stale operand binding, wrong scalar runtime role,
      wrong dtype/config relation, wrong target leaf/profile, stale
      intrinsic/type/header mirrors, and route-description mirror mismatch.
- [x] Existing contraction, MAcc, segment2, elementwise statement-owner, and
      selected-body tests still pass.
- [x] Bounded authority scan shows no retained elementwise route-family
      constants, exact intrinsic strings, ABI order constants, or
      scalar-broadcast binding authority in central route planning beyond
      neutral dispatch.
- [x] `git diff --check` and focused build/test pass; run `check-tianchenrv`
      unless an exact blocker is found.
- [x] Task status, journal, archive, and final commit are truthful.

## Completion Evidence

- Added `RVVEmitCElementwiseRouteFamilyPlanOwners` as the production owner for
  plain, masked/strided, and scalar-broadcast elementwise arithmetic
  route-family plan derivation, application, validation, provider-plan
  verification, operand-binding expectations, route-description mirror checks,
  target leaf/profile, ABI order, C type/header mapping, and required intrinsic
  mirrors.
- Reduced `RVVEmitCRoutePlanning.cpp` to shared typed/config/runtime fact
  collection, neutral owner dispatch, generic closure checks, and provider
  handoff for this family.
- Focused authority scans showed no retained central elementwise route-family
  plan IDs/constants, ABI order constants, scalar-broadcast binding authority,
  or elementwise add/sub/mul/splat intrinsic choices outside the owner.
- Focused C++ build/test passed, `git diff --check` passed, Trellis task
  validation passed, and `check-tianchenrv` passed 464/464.

## Out Of Scope

- No new arithmetic operations, dtype/LMUL clone batches, compare/select,
  reduction, conversion, MAcc, contraction, memory movement, TensorExt,
  frontend/Linalg lowering, dashboard/report work, or broad smoke matrix.
- No runtime/correctness/performance claim without real generated artifact or
  `ssh rvv` evidence.
- Do not re-open contraction owner unless a compile/test failure directly
  blocks this migration.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`.
- Precedent files:
  `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`
  and
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`.
- Current central functions to move are around elementwise and scalar-broadcast
  route-family plan derivation/validation/application and provider-plan
  verification in `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
