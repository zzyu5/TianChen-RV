# Stage2 RVV computed-mask select provider binding closure

## Goal

Complete the RVV plugin-local computed-mask select provider boundary for the
active `ComputedMaskSelect`, `RuntimeScalarCompareSelect`, and
`RuntimeScalarDualCompareMaskAndSelect` routes. Provider materialization must
depend on validated typed `tcrv_rvv` compare/mask/select body facts, computed-mask
select route-family plan mirrors, runtime ABI facts, and
`RouteOperandBindingPlan` closure before constructing a
`TCRVEmitCLowerableRoute`.

This is a bounded provider-validation repair for existing Stage2 RVV select
routes. It does not add new RVV coverage, dtype/LMUL expansion, source-front-door
support, or common EmitC semantics.

## Current-head facts

- `.trellis/.current-task` was absent when this task was created.
- The worktree was clean before the task was created.
- Recent HEAD is `01ba7bf3 rvv: deepen computed-mask macc provider validation`.
- The archived `05-23-stage2-rvv-computed-mask-macc-provider-validation` task
  deepened the adjacent computed-mask accumulation provider verifier and added
  binding closure before provider materialization.
- Current planning has an
  `RVVSelectedBodyComputedMaskSelectRouteFamilyPlan` for the active select
  routes.
- Current provider validation for computed-mask select already checks missing
  and stale plans, operation, family plan id, mask producer mirror, many target,
  type, intrinsic, memory, layout, and runtime ABI mirrors.
- Current provider validation still stops at the route operand binding plan id
  instead of validating full `RouteOperandBindingPlan` closure against the
  selected route description, unlike the repaired standalone-reduction and
  computed-mask MAcc paths.
- Active lit fixtures exist for explicit and pre-realized computed-mask select,
  runtime-scalar compare-select, runtime-scalar dual-compare mask-and-select,
  and adjacent plain compare-select routes.

## Requirements

- Inventory only active `ComputedMaskSelect`, `RuntimeScalarCompareSelect`, and
  `RuntimeScalarDualCompareMaskAndSelect` production routes, plus directly
  adjacent plain compare-select boundaries needed to prove family isolation.
- Require computed-mask select consumers to carry the computed-mask select
  route-family plan before provider materialization.
- Reject stale computed-mask select route-family plans on non-consumer routes.
- Validate route operation, memory form, family plan id, and producer-mode facts:
  vector compare producer, runtime scalar producer, and dual compare mask-and
  composition.
- Validate runtime control mirrors before provider materialization: SEW, LMUL,
  tail/mask policy, control plan id, config contract id, runtime VL contract id,
  runtime AVL source, VL definition/scope, VL uses, EmitC loop shape, remaining
  AVL metadata, pointer advance metadata, bounded-slice marker, and multi-VL
  marker.
- Validate runtime ABI order and runtime ABI parameter mirrors for each active
  route shape, including secondary compare operands for the dual-mask route.
- Validate target/profile mirrors: target leaf profile,
  `provider_supported_mirror`, required headers/header declarations, C type
  mapping, VL C type, vector type/C type, mask type/C type.
- Validate intrinsic mirrors: `setvl`, vector load, RHS scalar splat when
  present, primary compare, secondary compare, mask-and, select, and store.
- Validate computed-mask select facts: mask producer source, mask role/source,
  mask memory form, mask composition, mask name, result name, select layout,
  source memory form, destination memory form, and indexed memory layout.
- Validate `RouteOperandBindingPlan` closure against the route description
  before provider materialization, including plan id, runtime ABI order, binding
  roles, binding parameter mirrors, and binding summary mirror.
- Preserve plain compare-select as a separate family boundary; include only
  focused adjacent checks proving it does not consume the computed-mask select
  family plan.
- Preserve computed-mask MAcc and standalone-reduction boundaries; do not redo
  or broaden them in this task.
- Keep mirror fields mirror-only: they may be checked after RVV planning builds
  the provider family plan, but they must not become common EmitC/export route,
  dtype, compute, or intrinsic authority.

## Out of scope

- No redo of computed-mask MAcc, standalone reduction, plain compare-select,
  elementwise arithmetic, base memory movement, scalar broadcast,
  runtime splat-store, widening conversion, contraction, segment2, high-level
  frontend/Linalg, dtype/LMUL expansion, or broad evidence dashboards.
- No new route family or operation coverage beyond the existing active
  computed-mask select provider boundary.
- No movement of compare, mask, select, runtime-scalar, or intrinsic semantics
  into common EmitC/export, artifact names, route ids, helper strings,
  descriptors, or mirror fields.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-artifact, descriptor, or exact
  intrinsic spelling as route authority.

## Acceptance Criteria

- [ ] `verifyRVVSelectedBodyComputedMaskSelectRouteFamilyProviderPlans` performs
      deep computed-mask select plan validation comparable in depth to adjacent
      repaired family verifiers.
- [ ] Provider materialization rejects missing computed-mask select plans for
      active consumers and rejects stale computed-mask select plans on
      non-consumers.
- [ ] Provider validation rejects mirror mismatches for operation, memory form,
      runtime control, ABI order/parameters, target/header/type facts,
      vector/mask/VL type facts, setvl/load/splat/compare/secondary-compare/
      mask-and/select/store intrinsics, mask producer source, mask role/source/
      memory form, mask composition, select layout, source/destination/indexed
      memory facts, and `RouteOperandBindingPlan` closure.
- [ ] Active `ComputedMaskSelect`, `RuntimeScalarCompareSelect`, and
      `RuntimeScalarDualCompareMaskAndSelect` routes remain positive and
      coherent.
- [ ] Adjacent plain compare-select and computed-mask MAcc boundaries remain
      excluded with focused checks or code evidence showing no accidental
      coupling.
- [ ] Focused C++ and/or lit/FileCheck coverage proves deep validation,
      missing-plan and stale-plan fail-closed cases, mirror mismatch failures,
      runtime ABI/binding closure, vector vs runtime-scalar mask producer
      selection, dual-compare mask composition, and isolation from plain
      compare-select and computed-mask MAcc routes.
- [ ] Generated-bundle dry-runs cover representative explicit and pre-realized
      `computed_mask_select`, `runtime_scalar_cmp_select`, and
      `runtime_scalar_dual_cmp_mask_and_select` cases at counts 7, 16, and 23.
- [ ] Real `ssh rvv` evidence covers representative explicit and pre-realized
      computed-mask select cases, including selected lanes, false-lane behavior,
      runtime threshold variation, dual-compare mask behavior, tail
      preservation, and runtime `n` variation.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      shows no new legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
      source-front-door/source-artifact, descriptor, or common/export route
      authority.
- [ ] `check-tianchenrv`, focused changed-behavior checks, `git diff --check`,
      and clean git status pass before finish.

## Relevant specs and evidence inputs

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- Archived task:
  `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-macc-provider-validation/`
- Current code around
  `RVVSelectedBodyComputedMaskSelectRouteFamilyPlan`
- Current verifier
  `verifyRVVSelectedBodyComputedMaskSelectRouteFamilyProviderPlans`
- Current provider/realization/target-support paths:
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`, and
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- Generated-bundle runner `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing computed-mask select, runtime-scalar compare-select, and
  runtime-scalar dual-compare mask-and-select fixtures under `test/Target/RVV/`

## Completion report requirements

The final report must state task id/title, current phase, module behavior
completed, changed files, provider validation fields now checked, active
computed-mask select routes covered, adjacent plain compare-select and
computed-mask MAcc boundaries included or excluded with reason, selected-body
realization evidence, binding closure evidence, generated-bundle and `ssh rvv`
commands/results, checks, active-authority scan, self-repair, finish/archive
status, commit hash, and the exact continuation point if unfinished.
