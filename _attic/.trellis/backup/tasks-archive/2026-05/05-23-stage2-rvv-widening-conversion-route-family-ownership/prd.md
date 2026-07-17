# Stage2 RVV widening conversion route-family ownership

## Goal

Own the active RVV widening conversion route family through RVV plugin-local
planning and provider validation. The production provider must materialize
`widen_i32_to_i64` and `widen_i16_to_i32` only from validated widening
conversion family plans, and stale or missing plans must fail closed instead of
allowing route ids, mirror metadata, helper strings, or common EmitC/export code
to decide conversion semantics.

## Direction Source

Hermes direction brief supplied on 2026-05-23:

- module owner: RVV plugin-local widening conversion family boundary for active
  `widen_i32_to_i64` and `widen_i16_to_i32` routes
- previous baseline: `1d03baab rvv: own scalar broadcast splat-store families`
- required follow-up: give widening conversion the same explicit public
  family-plan, consumer predicate, and provider verifier boundary already used
  by memory, contraction, reduction, accumulation, select, broadcast, and
  splat-store families

## Current Facts Verified

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Worktree started clean at `1d03baab`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
- `widen_i32_to_i64` and `widen_i16_to_i32` are active production route kinds
  in provider/planning code and generated-bundle expectations.
- Explicit and pre-realized fixtures exist under `test/Target/RVV` and
  `test/Scripts` for both active widening conversion routes.
- Current planning headers have route-family plans for scalar-broadcast,
  splat-store, computed-mask select/memory, segment2 memory, accumulation,
  standalone reduction, and contraction, but no widening conversion family plan.
- Current conversion route logic already carries source/result SEW/LMUL,
  conversion relation, typed compute op, source load intrinsic, target/header
  mirrors, and `RouteOperandBindingPlan` facts; this task must move that
  authority behind an explicit widening conversion family boundary.

## Requirements

- Inventory only active `widen_i32_to_i64` and `widen_i16_to_i32` production
  routes plus directly adjacent provider predicates.
- Add an RVV planning-owned widening conversion consumer predicate covering
  exactly `WidenI32ToI64` and `WidenI16ToI32`.
- Add a widening conversion route-family plan carrying operation kind, memory
  form, family plan id, runtime ABI order, target leaf/profile mirrors, required
  headers, C type mapping, VL type, result vector type/C type, source
  SEW/LMUL/vector type/C type/source-load intrinsic, setvl/store/convert
  intrinsic leaves, result value name, conversion relation, and runtime ABI
  parameters.
- Populate the route description mirror fields from the validated widening
  conversion family plan during analysis.
- Make provider materialization for conversion routes depend on the widening
  conversion provider verifier before `TCRVEmitCLowerableRoute` construction.
- Reject missing widening conversion family plans for conversion consumers.
- Reject stale widening conversion family plans on non-conversion consumers.
- Preserve source/result dtype and SEW/LMUL facts, conversion relation,
  source/result operand roles, runtime ABI order, AVL/VL behavior, target
  leaf/header mirrors, conversion intrinsic leaf selection, and
  `RouteOperandBindingPlan` closure.
- Keep common EmitC/export neutral: RVV conversion semantics and intrinsic/type
  mapping stay in RVV planning/provider/realization/target support.

## Non-goals

- No dtype or LMUL clone batches beyond active `widen_i32_to_i64` and
  `widen_i16_to_i32`.
- No memory movement, compare/select, broadcast/splat-store, contraction,
  reduction, MAcc, widening dot, frontend/Linalg, source-front-door, dashboard,
  broad smoke-matrix, helper-only, or report-only work.
- No route id, ABI order, conversion semantics, runtime `n`/AVL behavior,
  dispatch/fallback behavior, target artifact contract, or common EmitC/export
  neutrality changes.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, descriptor
  residue, helper string, artifact name, or mirror-only acceptance authority.

## Acceptance Criteria

- [x] Active `widen_i32_to_i64` and `widen_i16_to_i32` routes are inventoried
      from current source and tests.
- [x] Provider materialization for widening conversion consumers requires a
      validated widening conversion family plan.
- [x] Missing widening conversion family plans on consumers fail closed with a
      targeted diagnostic.
- [x] Stale widening conversion family plans on non-consumers fail closed with a
      targeted diagnostic.
- [x] Focused C++ or lit/FileCheck coverage proves consumer classification,
      family plan id, conversion relation, source/result dtype and SEW/LMUL
      facts, runtime ABI order, conversion intrinsic leaf/type/header mirrors,
      binding closure, and missing/stale-plan rejection.
- [x] Generated-bundle dry-runs cover counts `7`, `16`, and `23` for
      representative explicit and pre-realized active widening conversion
      routes.
- [x] Real `ssh rvv` evidence covers executable `widen_i32_to_i64` and
      `widen_i16_to_i32`, including sign-extension/widening correctness,
      source/result lane widths, tail preservation, and runtime `n` variation.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      shows no legacy route authority regression.
- [x] `check-tianchenrv`, `git diff --check`, and clean git status pass.

## Completion Evidence

- Added `rvv-widening-conversion-route-family-plan.v1` as the explicit
  widening conversion family boundary for exactly `widen_i32_to_i64` and
  `widen_i16_to_i32`.
- Provider materialization now calls
  `verifyRVVSelectedBodyWideningConversionRouteFamilyProviderPlans` and uses a
  validated plan-derived `emitsWideningConversion` predicate, not a local
  provider route predicate.
- C++ coverage proves consumer classification, missing-plan failure for both
  conversion consumers, and stale-plan rejection on `add`.
- Target/FileCheck coverage proves family plan id, runtime control plan,
  target leaf profile, provider mirror, header/type mapping, source/dest
  SEW/LMUL, conversion relation, route operand binding closure, and header
  mirror stability for explicit `widen_i32_to_i64`, pre-realized
  `widen_i32_to_i64`, and pre-realized `widen_i16_to_i32`.
- Generated-bundle dry-runs passed for counts `7`, `16`, and `23`:
  explicit `widen_i32_to_i64`, pre-realized `widen_i32_to_i64`, and
  pre-realized `widen_i16_to_i32`.
- Real `ssh rvv` evidence passed for counts `7`, `16`, and `23`:
  explicit `widen_i32_to_i64` and pre-realized `widen_i16_to_i32`; the
  `widen_i16_to_i32` harness reported `sign_extension_checked tail_preserved`.
- Added-line active-authority scan found no new `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, descriptor, source-front-door, source-export, or direct-C
  authority in the touched diff.
- `cmake --build build --target check-tianchenrv` passed `361/361`, and
  `git diff --check` passed.

## Definition Of Done

- Implementation is bounded to the PRD module behavior.
- Focused tests and runtime evidence are recorded in task notes or journal.
- The Trellis task is finished and archived when complete.
- One coherent git commit records the completed task.

## Technical Starting Points

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-scalar-broadcast-splat-store-route-family-ownership/`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- focused widening conversion fixtures under `test/Target/RVV` and
  `test/Scripts`
