# Stage2 RVV widening conversion runtime and binding closure

## Goal

Close the RVV plugin-local widening conversion provider boundary for the active
`widen_i32_to_i64` and `widen_i16_to_i32` routes. Provider materialization must
depend on typed `tcrv_rvv` source/result element-width and vector config facts,
RVV-owned runtime AVL/VL control facts, widening conversion family-plan
validation, runtime ABI mirrors, and full `RouteOperandBindingPlan` closure
before constructing a `TCRVEmitCLowerableRoute`.

This is a bounded Stage2 provider-validation and runtime/binding closure repair
for existing widening conversion routes. It is not a new conversion, dtype,
LMUL, frontend, or evidence-packaging task.

## Current-head facts

- `.trellis/.current-task` was absent when this task was created.
- The worktree was clean before the task was created.
- Recent HEAD is `a111d01e rvv: close contraction provider binding`.
- The archived contraction task deepened an adjacent route family by requiring
  provider materialization to validate runtime AVL/VL mirrors, ABI mirrors,
  route/type/intrinsic facts, and full `RouteOperandBindingPlan` closure.
- Current widening conversion planning has
  `RVVSelectedBodyWideningConversionRouteFamilyPlan`, active explicit
  `widen_i32_to_i64` fixtures, and active pre-realized `widen_i32_to_i64` and
  `widen_i16_to_i32` fixtures.
- Current widening conversion plan validation already validates active
  operations, unit-stride conversion memory form, family plan id, runtime ABI
  order, target/header/type mirrors, source/result SEW/LMUL/vector C types,
  setvl/load/conversion/store intrinsic mirrors, result name, conversion
  relation, and runtime ABI parameters.
- Current provider-family verifier still compares only the runtime control plan
  id and route operand binding plan id. It does not yet compare all runtime
  AVL/VL control mirrors from the plan to the route description, and it does
  not call `verifyRVVRouteOperandBindingClosure` at the family-provider
  boundary.

## Requirements

- Inventory only active widening conversion consumers:
  `WidenI32ToI64` and `WidenI16ToI32`.
- Keep adjacent route families isolated. Non-consumers must reject stale
  widening conversion family plans, and widening conversion work must not
  couple contraction, computed-mask select, computed-mask MAcc, standalone
  reduction, plain compare-select, elementwise arithmetic, base memory
  movement, scalar broadcast, runtime splat-store, segment2, source-front-door,
  or descriptor paths.
- Validate before provider materialization that route description mirrors match
  the validated widening conversion plan for operation, memory form, runtime
  control, runtime ABI order/parameters, target leaf profile,
  `provider_supported_mirror`, required headers/header declarations, C type
  mapping, VL C type, source SEW/LMUL/vector type/load facts, result
  SEW/LMUL/vector type facts, setvl/conversion/store intrinsic mirrors, result
  naming, and conversion relation.
- Make runtime AVL/VL closure explicit at the widening conversion family
  provider boundary: SEW, LMUL, tail/mask policy, runtime control plan id,
  config contract id, runtime VL contract id, runtime AVL source, VL
  def/scope/use facts, EmitC loop facts, remaining AVL metadata, pointer
  advance metadata, bounded-slice marker, and multi-VL marker must match the
  validated family plan.
- Validate full `RouteOperandBindingPlan` closure for every widening conversion
  consumer: expected plan id, runtime ABI order, logical operand roles,
  materialized uses, parameter mirrors, and route binding summary mirror.
- Preserve mirror-only semantics. Checked description fields may mirror
  provider-owned planning output, but they must not become common EmitC/export
  route authority, dtype authority, compute authority, or intrinsic authority.

## Out of scope

- No new conversion operations, dtype/LMUL clone batches, high-level frontend or
  Linalg routes, global tuning, dashboards, or standalone evidence packaging.
- No redo of contraction, computed-mask select, computed-mask MAcc, standalone
  reduction, plain compare-select, elementwise arithmetic, base memory
  movement, scalar broadcast, runtime splat-store, segment2, or source
  front doors.
- No movement of conversion or dtype semantics into common EmitC/export,
  artifact names, route ids, helper strings, descriptors, source-front-door
  metadata, or mirror fields.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor-driven computation, source-front-door/source
  artifact paths, or exact intrinsic spelling as route authority.

## Acceptance Criteria

- [ ] `verifyRVVSelectedBodyWideningConversionRouteFamilyProviderPlans`
      validates full runtime AVL/VL mirrors from
      `RVVSelectedBodyWideningConversionRouteFamilyPlan` before provider
      materialization.
- [ ] The verifier calls `verifyRVVRouteOperandBindingClosure` for
      `widen_i32_to_i64` and `widen_i16_to_i32`, so binding plan id, runtime
      ABI order, parameter mirrors, logical operand roles, materialized uses,
      and summary mirrors fail closed at the provider boundary.
- [ ] Missing widening conversion family plans for active consumers and stale
      widening conversion plans on non-consumers fail closed with targeted
      diagnostics.
- [ ] Focused C++ and/or lit/FileCheck coverage proves provider validation,
      missing/stale plan failures, runtime control mismatch failures,
      source/result type mirror mismatch failures, conversion
      intrinsic/relation mismatch failures, runtime ABI mismatch failures,
      binding closure failures, and isolation from contraction and arithmetic
      routes.
- [ ] Generated-bundle dry-runs cover representative explicit and pre-realized
      `widen_i32_to_i64` and `widen_i16_to_i32` cases at counts 7, 16, and 23.
- [ ] Real `ssh rvv` evidence covers those representatives, including signed
      widening values, source/result element-width behavior, tail preservation,
      and runtime `n` variation.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      shows no new legacy `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, source-front-door/source-artifact, descriptor, or
      common/export route authority.
- [ ] `check-tianchenrv`, focused changed-behavior checks, `git diff --check`,
      and clean git status pass before finish.

## Relevant specs and evidence inputs

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- Archived task:
  `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-runtime-binding-closure/`
- Current code around `RVVSelectedBodyWideningConversionRouteFamilyPlan`
- Current verifier
  `verifyRVVSelectedBodyWideningConversionRouteFamilyProviderPlans`
- Current provider/realization/target-support paths:
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`, and
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- Generated-bundle runner `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing widening conversion fixtures under
  `test/Target/RVV/*widen-i16-to-i32*.mlir` and
  `test/Target/RVV/*widen-i32-to-i64*.mlir`

## Completion report requirements

The final report must state task id/title, current phase, module behavior
completed, changed files, provider validation fields now checked, active
widening conversion routes covered, adjacent route-family boundaries included
or excluded with reason, selected-body realization evidence, runtime/binding
closure evidence, generated-bundle and `ssh rvv` commands/results, checks,
active-authority scan, self-repair, finish/archive status, commit hash, and
the exact continuation point if unfinished.
