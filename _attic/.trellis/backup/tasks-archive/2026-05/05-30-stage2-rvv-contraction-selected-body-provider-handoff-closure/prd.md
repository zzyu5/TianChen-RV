# Stage2 RVV Contraction Selected-Body Provider Handoff Closure

## Goal

Close the selected-body/provider handoff for the direct RVV widening
contraction family. Pre-realized widening contraction validation authority for
`widening_macc_add`, `widening_dot_reduce_add`,
`strided_input_widening_dot_reduce_add`,
`computed_masked_widening_dot_reduce_add`, and
`computed_masked_strided_input_widening_dot_reduce_add` must live in
`RVVEmitCContractionRouteFamilyPlanOwners` or an owner-local companion under
that route-family boundary. `RVVSelectedBodyRealization` should remain the
selected-boundary dispatcher and realized-body construction site, not the owner
of duplicate contraction dtype/config/runtime/layout legality.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state for this round is clean on `main` at `b96e8014`
  (`rvv: close MAcc selected-body handoff`).
- There is no active `.trellis/.current-task`; this task was created from the
  supplied Hermes Direction Brief.
- The previous completed MAcc handoff moved analogous MAcc pre-realized
  selected-body validation into `RVVEmitCMAccRouteFamilyPlanOwners` and removed
  duplicate central MAcc branches.
- The direction brief says analogous widening contraction validation constants
  and `validatePreRealizedRVVSelectedWidening*` helpers still live centrally in
  `RVVSelectedBodyRealization.cpp`, while
  `RVVEmitCContractionRouteFamilyPlanOwners` already owns contraction provider
  facts.
- The stable authority chain remains:
  `tcrv.exec envelope -> selected RVV variant -> typed low-level tcrv_rvv body
  -> RVV plugin-owned legality / selected-body realization / route provider ->
  TCRVEmitCLowerableRoute -> neutral EmitC materialization -> target artifact`.

## Requirements

1. Move or factor pre-realized widening contraction selected-body validation
   for the five active contraction variants into the contraction owner-local
   boundary.
2. Keep `RVVSelectedBodyRealization.cpp` limited to selected-body owner
   dispatch, neutral shared helpers, and realized-body construction.
3. Preserve validation of operation kind, memory form, dtype/SEW/LMUL
   relation, accumulator/result layout, runtime ABI role/order, stride binding,
   mask provenance, tail/mask policy, mixed pre-realized/realized body
   rejection, and capability metadata.
4. Preserve provider-plan and mirror verification through owner-derived
   contraction facts. Mirror fields must remain mirrors and fail closed when
   stale.
5. Avoid adding MAcc-specific or contraction-specific semantic branches in
   central route provider/planning code. Central code may call owner APIs as a
   neutral boundary.
6. Add focused tests showing representative unmasked, strided, computed-mask,
   and computed-mask-strided contraction bodies use owner-local validation and
   fail closed on mismatched facts.
7. Provide generated-bundle dry-run evidence for representative migrated
   contraction paths without claiming route-entry shortcut authority or runtime
   correctness.
8. Preserve the completed MAcc handoff as a non-regression.

## Acceptance Criteria

- [x] Contraction pre-realized selected-body validation declarations and
      definitions live in `RVVEmitCContractionRouteFamilyPlanOwners` or an
      owner-local companion, not as central contraction legality in
      `RVVSelectedBodyRealization.cpp`.
- [x] `RVVSelectedBodyRealization.cpp` retains only dispatcher/construction
      responsibilities for this family and calls the owner-local validation
      surface.
- [x] The five active contraction variants are classified exactly by the
      contraction owner and reject stale or mismatched typed/config/runtime
      facts with targeted diagnostics.
- [x] Provider plans, route descriptions, and mirrors still consume the same
      owner-derived contraction facts and reject stale plan/mirror metadata.
- [x] Focused C++ tests cover representative positive owner-local validation
      for unmasked, strided, computed-mask, and computed-mask-strided
      contraction bodies.
- [x] Focused fail-closed tests cover wrong op kind, memory form,
      dtype/SEW/LMUL mismatch, accumulator/result layout mismatch, runtime ABI
      role/order mismatch, stride binding mismatch, mask provenance mismatch,
      policy mismatch, mixed pre-realized/realized body, missing capability
      metadata, stale route/provider mirror, and central/provider semantic
      invention where applicable.
- [x] Representative generated-bundle dry-runs pass for migrated contraction
      paths, or exact blockers are recorded.
- [x] MAcc handoff focused non-regression remains green.
- [x] Bounded authority scans over touched files show no central ad hoc,
      name-derived, metadata-derived, descriptor-derived, route-id-derived,
      exact-intrinsic-as-authority, common-EmitC-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32
      authority for this family.
- [x] `git diff --check`, focused RVV plugin tests, and `check-tianchenrv`
      pass, or exact blockers are recorded.
- [x] Task is finished/archived and one coherent commit is created with clean
      final source status.

## Completion Evidence

- Owner-local validation APIs were added to
  `RVVEmitCContractionRouteFamilyPlanOwners` for widening MAcc, widening
  dot-reduce, strided widening dot-reduce, computed-mask widening dot-reduce,
  and computed-mask strided widening dot-reduce.
- `RVVSelectedBodyRealization.cpp` now dispatches the contraction family
  through the contraction owner hook, calls owner-local validation, and keeps
  only realization-plan construction plus realized `setvl`/`with_vl`/load/
  compute/store materialization.
- Central contraction op-kind legality checks were removed from selected-body
  construction helpers; contraction op kind, memory form, dtype/config,
  runtime ABI, mask, stride, policy, mixed-body, and requires metadata checks
  live in the owner-local validation surface.
- Existing RVV extension plugin coverage proves positive selected-boundary
  realization/provider handoff for widening MAcc, widening dot-reduce,
  strided-input widening dot-reduce, computed-mask widening dot-reduce, and
  computed-mask strided-input widening dot-reduce, plus owner fail-closed
  mutation coverage for stale mask/config/layout/op-kind/runtime/stride facts.
- Generated-bundle dry-run evidence:
  `artifacts/tmp/stage2_rvv_contraction_handoff/codex-contraction-handoff-all`
  for all five migrated contraction op kinds.
- Direct route-entry shortcut negative evidence:
  `artifacts/tmp/stage2_rvv_contraction_handoff/codex-contraction-handoff-direct-route-entry-negative`
  fails before bundle export with the retired shortcut diagnostic.
- MAcc non-regression dry-run evidence:
  `artifacts/tmp/stage2_rvv_contraction_handoff/codex-macc-handoff-nonregression`.
- Checks passed: `git diff --check`,
  `ninja -C build tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-rvv-extension-plugin-test`, and
  `ninja -C build check-tianchenrv` with 464/464 tests passed.

## Out Of Scope

- No new contraction operation coverage, dtype/LMUL clone batches, new
  intrinsic cases, source-front-door routes, Linalg/frontend lowering,
  runtime/performance claims without `ssh rvv` evidence, dashboards, broad
  smoke matrices, or report-only work.
- No churn to completed MAcc, memory, elementwise, compare/select, segment2,
  standalone reduction, or conversion owners except narrow shared-interface
  fallout required by this handoff.
- No descriptor-driven computation, route-id authority, artifact-name
  authority, common EmitC semantic invention, direct route-entry shortcut, or
  legacy `i32m1`/`tcrv_rvv.i32_*` authority.

## Technical Notes

- Specs to read for implementation/check:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/guides/plugin-locality-review-guide.md`.
- Archived task to read as nearest completed pattern:
  `.trellis/tasks/archive/2026-05/05-30-stage2-rvv-macc-selected-body-realization-provider-handoff-closure/`.
- Production files to inspect first:
  `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`, and relevant
  RVV extension plugin tests.

## Validation Plan

1. Validate Trellis context and read relevant specs/code.
2. Implement the bounded contraction owner-local validation handoff.
3. Update focused C++ tests for owner-local positive and fail-closed behavior.
4. Run focused build/test target for `tianchenrv-rvv-extension-plugin-test`.
5. Run generated-bundle dry-run evidence for representative contraction paths.
6. Run bounded authority scans and `git diff --check`.
7. Run `ninja -C build check-tianchenrv` or record the exact blocker.
8. Finish/archive the task and create one coherent commit.
