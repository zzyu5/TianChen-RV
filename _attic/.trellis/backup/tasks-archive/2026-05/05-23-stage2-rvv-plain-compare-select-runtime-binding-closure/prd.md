# Stage2 RVV plain compare-select runtime and binding closure

## Goal

Close the RVV plugin-local plain compare-select provider boundary for the
active `cmp_select` routes. Provider materialization must depend on typed
`tcrv_rvv.compare` / `tcrv_rvv.select` body facts, RVV-owned runtime AVL/VL
control facts, plain compare-select family-plan validation, runtime ABI mirrors,
and full `RouteOperandBindingPlan` closure before constructing a
`TCRVEmitCLowerableRoute`.

This is a bounded Stage2 provider-validation and runtime/binding closure repair
for existing explicit and pre-realized `cmp_select` / `cmp_select_sle` artifact
paths. It is not new compare/select coverage.

## Current-head facts

- `.trellis/.current-task` was absent when this task was created.
- The worktree was clean before the task was created.
- Recent HEAD is `25f05d73 rvv: close widening conversion provider binding`.
- The archived widening conversion task deepened an adjacent route family by
  requiring provider materialization to validate runtime AVL/VL mirrors, ABI
  mirrors, route/type/intrinsic facts, and full `RouteOperandBindingPlan`
  closure.
- Current plain compare-select planning has
  `RVVSelectedBodyPlainCompareSelectRouteFamilyPlan`, active explicit
  `cmp_select` / `cmp_select_sle` fixtures, and active pre-realized
  `cmp_select` / `cmp_select_sle` fixtures.
- Current plain compare-select plan validation already validates active
  operation, vector-rhs-load memory form, family plan id, runtime ABI order,
  target/header/type mirrors, vector/mask/VL C types, setvl/load/compare/select
  /store intrinsic mirrors, predicate kind, result and mask names, mask
  role/source/memory form, inactive-lane contract, passthrough/select layout,
  source/destination memory forms, and runtime ABI parameters.
- Current provider-family verifier still compares only the runtime control plan
  id and route operand binding plan id. It does not yet compare all runtime
  AVL/VL control mirrors from the plan to the route description, and it does not
  call `verifyRVVRouteOperandBindingClosure` at the plain compare-select
  family-provider boundary.

## Requirements

- Inventory only active plain compare-select consumers: `CmpSelect` selected
  bodies reached by explicit `cmp_select` / `cmp_select_sle` and their
  pre-realized equivalents.
- Keep adjacent route families isolated. `ComputedMaskSelect`,
  `RuntimeScalarCompareSelect`, `RuntimeScalarDualCompareMaskAndSelect`,
  elementwise arithmetic, widening conversion, contraction, computed-mask MAcc,
  standalone reduction, memory movement, scalar broadcast, runtime splat-store,
  segment2, source-front-door, and descriptor paths must not become plain
  compare-select consumers.
- Validate before provider materialization that route description mirrors match
  the validated plain compare-select plan for operation, memory form, runtime
  control, runtime ABI order/parameters, target leaf profile,
  `provider_supported_mirror`, required headers/header declarations, C type
  mapping, VL C type, vector/mask C types, setvl/load/compare/select/store
  intrinsic mirrors, compare predicate kind, result and mask names, mask
  role/source/memory form, inactive-lane contract, passthrough/select layout,
  source memory form, and destination memory form.
- Make runtime AVL/VL closure explicit at the plain compare-select family
  provider boundary: SEW, LMUL, tail/mask policy, runtime control plan id,
  config contract id, runtime VL contract id, runtime AVL source, VL
  def/scope/use facts, EmitC loop facts, remaining AVL metadata, pointer advance
  metadata, bounded-slice marker, and multi-VL marker must match the validated
  family plan.
- Validate full `RouteOperandBindingPlan` closure for every plain
  compare-select consumer: expected plan id, runtime ABI order, logical operand
  roles, materialized uses, parameter mirrors, and route binding summary mirror.
- Preserve mirror-only semantics. Checked description fields may mirror
  provider-owned planning output, but they must not become common EmitC/export
  route authority, dtype authority, compute authority, or intrinsic authority.

## Out of scope

- No new compare predicates, dtype/LMUL clone batches, high-level frontend or
  Linalg routes, global tuning, dashboards, or standalone evidence packaging.
- No redo of widening conversion, contraction, computed-mask select,
  computed-mask MAcc, standalone reduction, elementwise arithmetic, base memory
  movement, scalar broadcast, runtime splat-store, segment2, or source front
  doors.
- No movement of compare/select semantics into common EmitC/export, artifact
  names, route ids, helper strings, descriptors, source-front-door metadata, or
  mirror fields.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor-driven computation, source-front-door/source
  artifact paths, or exact intrinsic spelling as route authority.

## Acceptance Criteria

- [ ] `verifyRVVSelectedBodyPlainCompareSelectRouteFamilyProviderPlans`
      validates full runtime AVL/VL mirrors from
      `RVVSelectedBodyPlainCompareSelectRouteFamilyPlan` before provider
      materialization.
- [ ] The verifier calls `verifyRVVRouteOperandBindingClosure` for
      `cmp_select`, so binding plan id, runtime ABI order, parameter mirrors,
      logical operand roles, materialized uses, and summary mirrors fail closed
      at the provider boundary.
- [ ] Missing plain compare-select family plans for active consumers and stale
      plain compare-select plans on non-consumers fail closed with targeted
      diagnostics.
- [ ] Focused C++ and/or lit/FileCheck coverage proves provider validation,
      missing/stale plan failures, runtime control mismatch failures,
      predicate/type/intrinsic/layout mirror mismatch failures, runtime ABI
      mismatch failures, binding closure failures, and isolation from
      computed-mask select and elementwise arithmetic routes.
- [ ] Generated-bundle dry-runs cover representative explicit and pre-realized
      `cmp_select` and `cmp_select_sle` cases at counts 7, 16, and 23.
- [ ] Real `ssh rvv` evidence covers those representatives, including selected
      values, false-lane behavior, signed predicate behavior, tail preservation,
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
- `.trellis/spec/testing/index.md`
- Archived task:
  `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-widening-conversion-runtime-binding-closure/`
- Current code around `RVVSelectedBodyPlainCompareSelectRouteFamilyPlan`
- Current verifier
  `verifyRVVSelectedBodyPlainCompareSelectRouteFamilyProviderPlans`
- Current provider/realization/target-support paths:
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`, and
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- Generated-bundle runner `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing plain compare-select fixtures under
  `test/Target/RVV/*cmp-select*.mlir`, excluding runtime-scalar cmp-select
  except as an isolation boundary owned by computed-mask select.

## Completion report requirements

The final report must state task id/title, current phase, module behavior
completed, changed files, provider validation fields now checked, active plain
compare-select routes covered, adjacent route-family boundaries included or
excluded with reason, selected-body realization evidence, runtime/binding
closure evidence, generated-bundle and `ssh rvv` commands/results, checks,
active-authority scan, self-repair, finish/archive status, commit hash, and the
exact continuation point if unfinished.
