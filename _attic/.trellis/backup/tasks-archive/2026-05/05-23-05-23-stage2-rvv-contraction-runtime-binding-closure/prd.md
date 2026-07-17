# Stage2 RVV contraction route-family runtime and binding closure

## Goal

Close the RVV plugin-local contraction route-family provider boundary for the
active widening multiply-accumulate and widening dot-reduction routes. Provider
materialization must depend on typed `tcrv_rvv` contraction body facts,
RVV-owned runtime AVL/VL control facts, contraction family-plan validation, and
`RouteOperandBindingPlan` closure before constructing a
`TCRVEmitCLowerableRoute`.

This is a bounded provider-validation and runtime/binding closure repair for
existing Stage2 contraction routes. It does not add new contraction operations,
dtype/LMUL coverage, source-front-door support, common EmitC semantics, or
standalone evidence packaging.

## Current-head facts

- `.trellis/.current-task` was absent when this task was created.
- The worktree was clean before the task was created.
- Recent HEAD is `b176d19e rvv: close computed-mask select provider binding`.
- The archived computed-mask select task deepened an adjacent route family by
  requiring provider materialization to validate runtime control, ABI mirrors,
  intrinsic/type/mask facts, and full `RouteOperandBindingPlan` closure.
- Current contraction planning has an
  `RVVSelectedBodyContractionRouteFamilyPlan` and active generated-bundle
  support for `widening_macc_add`, `widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`.
- Current contraction provider validation already requires missing-plan and
  stale-plan fail-closed behavior, validates operation, memory form, family
  plan id, ABI order/parameters, target/header/type mirrors, contraction
  intrinsic mirrors, computed-mask mirrors, strided-input mirrors, and the
  binding plan id.
- Current contraction provider validation still lacks the same explicit
  runtime AVL/VL control-plan validation and full route operand binding closure
  that adjacent select, standalone-reduction, and accumulation verifiers now
  perform.

## Requirements

- Inventory only active contraction consumers:
  `WideningMAccAdd`, `WideningDotReduceAdd`,
  `StridedInputWideningDotReduceAdd`,
  `ComputedMaskWideningDotReduceAdd`, and
  `ComputedMaskStridedInputWideningDotReduceAdd`.
- Keep non-contraction routes isolated. Non-consumers must reject stale
  contraction family plans, and contraction work must not couple computed-mask
  MAcc, standalone reduction, computed-mask select, plain compare-select,
  elementwise arithmetic, widening conversion, base memory movement, segment2,
  scalar broadcast, runtime splat-store, or source-front-door routes.
- Make contraction runtime AVL/VL control facts explicit in the contraction
  family plan and provider verifier: SEW, LMUL, tail/mask policy, runtime
  control plan id, config contract id, runtime VL contract id, runtime AVL
  source, VL def/scope/use facts, EmitC loop shape, remaining AVL metadata,
  pointer advance metadata, bounded-slice marker, and multi-VL marker.
- Validate before provider materialization that route description mirrors match
  the validated contraction plan for operation, memory form, runtime control,
  runtime ABI order/parameters, target leaf profile,
  `provider_supported_mirror`, required headers/header declarations, C type
  mapping, VL/vector/mask C types, source/result vector types, setvl/load/
  strided-load/store/compute/widening-product/masked-product/seed-splat/
  reduction-store mirrors, widening MAcc relation/layout facts, widening
  dot-reduction relation/layout facts, computed-mask facts, and strided-input
  facts.
- Validate full `RouteOperandBindingPlan` closure for every contraction
  consumer: expected plan id, runtime ABI order, logical operand roles,
  materialized uses, parameter mirrors, and route binding summary mirror.
- Preserve mirror-only semantics. Checked description fields may mirror
  provider-owned planning output, but they must not become common EmitC/export
  route authority, dtype authority, compute authority, or intrinsic authority.

## Out of scope

- No new contraction route coverage, new op kinds, dtype/LMUL clones, high-level
  frontend/Linalg routes, global tuning, dashboards, or broad evidence bundles.
- No redo of computed-mask select, computed-mask MAcc, standalone reduction,
  plain compare-select, elementwise arithmetic, base memory movement, scalar
  broadcast, runtime splat-store, widening conversion, segment2, or source
  front doors.
- No movement of contraction semantics into common EmitC/export, artifact
  names, route ids, helper strings, descriptors, source-front-door metadata, or
  mirror fields.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor-driven computation, source-front-door/source
  artifact paths, or exact intrinsic spelling as route authority.

## Acceptance Criteria

- [ ] `RVVSelectedBodyContractionRouteFamilyPlan` carries explicit runtime
      AVL/VL control facts derived from the selected typed body/config facts.
- [ ] `verifyRVVSelectedBodyContractionRouteFamilyProviderPlans` validates
      runtime control, route/type/intrinsic/mask/strided mirrors, runtime ABI
      parameters, and full `RouteOperandBindingPlan` closure before provider
      materialization.
- [ ] Missing contraction family plans for active consumers and stale
      contraction plans on non-consumers fail closed with targeted diagnostics.
- [ ] Widening MAcc and widening dot-reduction remain isolated: each validates
      its own relation/layout/compute mirrors, and dot-reduction validates
      widening-product, optional masked-product, scalar seed, and reduction
      store facts.
- [ ] Computed-mask dot-reduction and strided-input dot-reduction remain
      isolated: computed-mask routes validate mask facts only when present, and
      strided-input routes validate strided facts only when present.
- [ ] Focused C++ and/or lit/FileCheck coverage proves provider validation,
      missing/stale plan failures, runtime control mismatch failures, mirror
      mismatch failures, runtime ABI and binding closure, widening MAcc versus
      dot-reduction isolation, computed-mask isolation, and strided-input
      isolation.
- [ ] Generated-bundle dry-runs cover representative explicit and pre-realized
      `widening_macc_add`, `widening_dot_reduce_add`,
      `strided_input_widening_dot_reduce_add`, and
      `computed_masked_widening_dot_reduce_add` cases at counts 7, 16, and 23.
- [ ] Real `ssh rvv` evidence covers those representatives, including
      accumulated values, dot-reduction values, mask false-lane behavior,
      strided input behavior, tail preservation, and runtime `n` variation.
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
  `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-select-provider-binding-closure/`
- Current code around `RVVSelectedBodyContractionRouteFamilyPlan`
- Current verifier
  `verifyRVVSelectedBodyContractionRouteFamilyProviderPlans`
- Current provider/realization/target-support paths:
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`, and
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- Generated-bundle runner `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing contraction fixtures under `test/Target/RVV/*widening-macc*.mlir`
  and `test/Target/RVV/*widening-dot-reduce*.mlir`

## Completion report requirements

The final report must state task id/title, current phase, module behavior
completed, changed files, provider validation fields now checked, active
contraction routes covered, adjacent route-family boundaries included or
excluded with reason, selected-body realization evidence, runtime/binding
closure evidence, generated-bundle and `ssh rvv` commands/results, checks,
active-authority scan, self-repair, finish/archive status, commit hash, and
the exact continuation point if unfinished.
