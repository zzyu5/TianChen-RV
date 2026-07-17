# Stage2 RVV runtime-scalar compare-select provider boundary closure

## Goal

Close the runtime-scalar compare/select subfamily against the compare/select
provider preflight introduced by the previous owner-boundary task:

```text
selected tcrv.exec RVV variant
  -> owner-local elementwise/compare-select selected-body realization
  -> realized typed runtime-scalar tcrv_rvv compare/select body
  -> same-analysis computed-mask select family facts
  -> materialization facts
  -> route-control and elementwise/select operand-binding facts
  -> compare/select statement/provider plan
  -> provider preflight
  -> TCRVEmitCLowerableRoute
  -> neutral EmitC
  -> generated target artifact evidence
```

This is a bounded provider-boundary closure for
`runtime_scalar_cmp_select` and
`runtime_scalar_dual_cmp_mask_and_select`. It is not a new route family and
not a coverage expansion outside the compare/select owner/provider path.

## Direction Source

- Direction title: `Expand: Stage2 RVV runtime-scalar compare-select provider boundary closure`.
- Module owner: RVV elementwise/compare-select selected-body route-provider
  boundary for `runtime_scalar_cmp_select` and
  `runtime_scalar_dual_cmp_mask_and_select`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Session start facts: worktree was clean; HEAD was
  `0d886c2c rvv: close compare-select owner provider boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief.

## What I Already Know

- The previous archived task
  `05-31-05-31-stage2-rvv-elementwise-compare-select-owner-boundary-closure`
  added `verifyRVVSelectedBodyCompareSelectRouteProviderFacts(...)` and made
  `RVVEmitCRouteProvider` call it before route construction for the base
  compare/select boundary.
- Current code already has runtime-scalar compare/select route-family facts,
  operand-binding facts, and statement-plan support:
  `runtime_scalar_cmp_select` and
  `runtime_scalar_dual_cmp_mask_and_select` are consumers of the computed-mask
  select family and compare/select statement-plan boundary.
- Current provider preflight consumes `analysis`, materialization facts, and
  elementwise/select operand-binding facts, but it does not explicitly consume
  the compare/select statement/provider plan before the provider creates
  `TCRVEmitCLowerableRoute`.
- The RVV spec names runtime-scalar compare-select and runtime-scalar dual
  compare-mask-and-select in the selected-body realization boundary and in
  compare/select statement-plan scope.
- `tcrv.exec` ABI/runtime values are envelope facts only; runtime scalar
  producer, dtype/config/policy, memory form, mask composition, AVL/VL, and
  select semantics must come from typed `tcrv_rvv` body/family/provider facts.
- Common EmitC and target artifacts may mirror provider facts after route
  construction, but must not infer route support from mirrors, route ids,
  artifact names, exact intrinsics, ABI strings, descriptors, or source-front
  doors.

## Requirements

1. Keep the owner header and route-family planning header layered as in the
   completed compare/select split. Do not move selected-body owner APIs into
   route planning headers.
2. Extend the compare/select provider preflight so runtime-scalar
   compare/select route construction proves the compare/select statement plan
   belongs to the same analysis/materialization/operand-binding facts before
   `TCRVEmitCLowerableRoute` construction.
3. For `runtime_scalar_cmp_select`, require realized typed computed-mask select
   facts with runtime-scalar producer source, matching single runtime-scalar
   operand-binding facts, RHS scalar splat leaf, route-control/mask-tail facts,
   and a statement plan that marks runtime-scalar computed-mask select but not
   dual mask-and.
4. For `runtime_scalar_dual_cmp_mask_and_select`, require the same typed
   facts plus dual runtime-scalar operand-binding facts, secondary compare
   facts, mask-and facts, mask composition `and`, and a statement plan that
   marks both runtime-scalar computed-mask select and dual compare-mask-and.
5. Preserve plain `cmp_select` and vector `computed_mask_select` behavior and
   keep their previous positive/fail-closed coverage passing.
6. Fail closed on stale runtime scalar producer facts, single/dual mismatch,
   wrong mask composition, missing or stale RHS scalar splat leaf, stale
   materialization leaves, wrong dtype/config/policy, wrong runtime n/AVL/VL
   role, stale operand binding, stale statement plan, mirror metadata as
   authority, route-id/artifact-name/exact-intrinsic authority, direct
   route-entry-only claims, and common EmitC semantic invention where those
   can enter this bounded path.
7. Add or repair focused C++ coverage for positive and fail-closed provider
   preflight behavior for both runtime-scalar subcases.
8. Add or repair generated-bundle or target-artifact dry-run evidence through
   selected lowering-boundary realization for both runtime-scalar subcases.
9. Do not make runtime, correctness, or performance claims without real
   `ssh rvv` evidence.

## Acceptance Criteria

- [ ] `RVVEmitCRouteProvider` obtains compare/select statement/provider plan
      facts before constructing `TCRVEmitCLowerableRoute` for runtime-scalar
      compare/select routes.
- [ ] `verifyRVVSelectedBodyCompareSelectRouteProviderFacts(...)` or an
      equivalent provider preflight consumes the statement plan as an explicit
      input for compare/select consumers.
- [ ] The preflight accepts valid `runtime_scalar_cmp_select` and
      `runtime_scalar_dual_cmp_mask_and_select` analyses only when typed
      config, materialization facts, operand-binding facts, computed-mask
      family plan, route-control/mask-tail facts, and statement plan all
      agree.
- [ ] The preflight fails closed for stale runtime-scalar producer facts,
      single-vs-dual mismatch, missing RHS scalar splat leaf, stale statement
      plan, stale materialization facts, stale operand-binding facts, and
      stale runtime n/AVL/VL or policy facts before route construction.
- [ ] Existing plain `cmp_select` and vector `computed_mask_select`
      provider-preflight tests still pass.
- [ ] C++ tests cover runtime-scalar compare-select and runtime-scalar dual
      compare-mask-and-select positive and negative provider preflight cases.
- [ ] Generated-bundle or target-artifact dry-run checks cover
      `runtime_scalar_cmp_select` and
      `runtime_scalar_dual_cmp_mask_and_select` through selected
      lowering-boundary realization.
- [ ] Direct pre-realized route-entry-only claims remain fail-closed for both
      runtime-scalar subcases.
- [ ] Bounded owner/API scan shows selected-body owner declarations remain out
      of route-family planning headers.
- [ ] Bounded touched-file authority scan finds no new name-derived,
      metadata-derived, descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [ ] `git diff --check` passes.
- [ ] Focused RVV plugin and generated-bundle/target tests pass.
- [ ] `check-tianchenrv` passes, or the exact blocker is recorded.
- [ ] Task status, context, and journal are truthful; task is archived and one
      coherent commit is created if acceptance passes.

## Out Of Scope

- No arithmetic, contraction, reduction, memory movement, conversion, segment2,
  source-front-door, Linalg/frontend, dtype/LMUL clone, dashboard/report, or
  broad smoke-matrix work.
- No new extension family, backend, descriptor-driven compute path, direct C
  exporter, or one-intrinsic wrapper.
- No re-opening completed plain `cmp_select` or vector `computed_mask_select`
  work except for directly required non-regression.
- No runtime/correctness/performance claim without `ssh rvv` evidence.

## Technical Notes

- Specs read before PRD: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous archived task read:
  `.trellis/tasks/archive/2026-05/05-31-05-31-stage2-rvv-elementwise-compare-select-owner-boundary-closure/{task.json,prd.md,implement.jsonl,check.jsonl}`.
- Relevant workspace journal entries read from
  `.trellis/workspace/codex/journal-19.md`, especially Session 350.
- Initial repository inspection found no current task, a clean worktree, and
  recent commits ending at `0d886c2c`.
- Primary production files for inspection and possible edit:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.
- Directly related generated-bundle consumers:
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-select-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-dual-cmp-mask-and-select-dry-run.test`,
  and corresponding direct-pre-realized fail-closed tests.
- No blocking user question remains; the supplied Direction Brief and specs are
  specific enough to implement this bounded closure.
