# Stage2 RVV Non-Segment Computed-Mask Memory Selected-Body Provider Handoff Closure

## Goal

Close the selected-body/provider handoff for the RVV non-segment computed-mask
memory movement family. Pre-realized selected-body validation authority for the
production-active runtime-scalar, unit, strided, and indexed computed-mask
memory bodies must live in
`RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners` or an owner-local companion
under that route-family boundary. `RVVSelectedBodyRealization.cpp` must remain
the selected-body owner dispatcher and realized IR materializer for
`setvl` / `with_vl` / compare / mask / load / store structure only.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state for this round is clean on `main` at `95377d1d`
  (`rvv: close base memory selected-body handoff`).
- There was no active `.trellis/.current-task`; this task was created from the
  supplied Hermes Direction Brief.
- The archived base-memory handoff task moved analogous pre-realized validation
  APIs into `RVVEmitCBaseMemoryRouteFamilyPlanOwners` and kept central selected
  body code limited to owner dispatch and realized IR materialization.
- Current bounded inspection shows
  `RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners` already owns the
  non-segment computed-mask memory route-family owner registry and provider-plan
  verifier, but exposes no selected-body validation APIs.
- Current bounded inspection shows `RVVSelectedBodyRealization.cpp` still owns
  selected-body validators for:
  `TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp`,
  `TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp`,
  `TypedComputedMaskMemoryPreRealizedBodyOp`,
  `TypedComputedMaskStridedStorePreRealizedBodyOp`,
  `TypedComputedMaskStridedLoadPreRealizedBodyOp`,
  `TypedComputedMaskIndexedGatherPreRealizedBodyOp`, and
  `TypedComputedMaskIndexedScatterPreRealizedBodyOp`.
- Segment2 computed-mask memory validators are adjacent but remain out of scope
  for this non-segment owner round.
- The stable authority chain remains:
  `tcrv.exec envelope -> selected RVV variant -> typed low-level tcrv_rvv body
  -> RVV plugin-owned legality / selected-body realization / route provider ->
  TCRVEmitCLowerableRoute -> neutral EmitC materialization -> target artifact`.

## Requirements

1. Add owner-local selected-body validation declarations and definitions for
   production-active non-segment computed-mask memory bodies in
   `RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners`.
2. Include the runtime-scalar computed-mask store/load-store validators in the
   same owner-local surface because those routes are part of the current
   non-segment computed-mask memory route-family provider and statement-plan
   boundary.
3. Keep `RVVSelectedBodyRealization.cpp` limited to family owner dispatch,
   selected-body realization calls into owner-local validators, and realized IR
   materialization.
4. Preserve validation of operation kind, memory form, predicate kind,
   compare-mask role/source/form, inactive-lane policy, runtime ABI role/order,
   mem_window/runtime_param imported values, source/destination/stride/index
   bindings, dtype/config/policy, VL/AVL relation, selected variant requires
   metadata, and mixed pre-realized/realized body rejection.
5. Unsupported or inconsistent facts must fail closed with targeted diagnostics
   before provider facts, statement plans, common EmitC, or target artifact
   export.
6. Keep current executable behavior and emitted route outputs stable unless a
   concrete bug is found. This is an ownership handoff, not new RVV coverage.
7. Preserve segment2 memory, base-memory, contraction, MAcc, reduction,
   compare/select, widening conversion, frontend lowering, dtype/LMUL clone
   batches, one-intrinsic wrappers, dashboards, reports, and broad smoke
   matrices as non-goals.
8. Do not add descriptor-derived, name-derived, route-id-derived,
   ABI-string-derived, artifact-name-derived, exact-intrinsic-derived,
   common-EmitC-derived, direct-route-entry-only, source-front-door-derived,
   script-derived, stale mirror-derived, or legacy-i32 authority.

## Acceptance Criteria

- [x] Non-segment computed-mask memory pre-realized selected-body validation
      declarations and definitions live in
      `RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners`, not as central
      computed-mask memory legality in `RVVSelectedBodyRealization.cpp`.
- [x] `RVVSelectedBodyRealization.cpp` dispatches runtime-scalar, unit,
      strided, and indexed computed-mask memory validation through owner-local
      APIs and retains only family dispatch plus realized IR materialization.
- [x] Runtime-scalar store/load-store, unit load/store, strided store,
      strided load/unit-store, indexed gather/unit-store, and indexed
      scatter/unit-load selected bodies keep exact positive behavior and fail
      closed on wrong mask binding, inactive-lane policy, memory form,
      runtime ABI role, dtype/config/policy, AVL/VL relation, stale mirror
      metadata, direct-route-only authority, artifact-name/script-derived
      authority, and common-EmitC semantic invention.
- [x] Computed-mask memory route-family plans, route-control provider plans,
      operand-binding facts, statement plans, provider facts, and mirrors still
      consume owner-derived computed-mask facts and reject stale plans/mirrors.
- [x] Focused C++ coverage proves central selected-body dispatch reaches the
      owner-local validators for representative runtime-scalar and non-segment
      computed-mask memory bodies.
- [x] Focused fail-closed coverage proves wrong memory form, missing or wrong
      runtime ABI role, wrong stride/index/mask binding, wrong dtype/config/
      policy, stale provider mirror, stale operand binding, and non-owner route
      claims where current focused tests can exercise them.
- [x] Generated-bundle dry-runs pass for representative affected computed-mask
      memory selected-boundary variants, or exact blockers are recorded.
- [x] Base-memory handoff and at least one contraction path remain green as
      non-regressions.
- [x] Bounded authority scans show central selected-body no longer owns
      non-segment computed-mask memory legality constants or validation
      authority and touched planning/provider/test files have no legacy
      i32/source-front-door/descriptor/direct-C/source-export/direct-route or
      mirror-only authority drift.
- [x] `git diff --check`, focused RVV plugin build/test, and
      `check-tianchenrv` pass, or exact blockers are recorded.
- [x] Task is finished/archived and one coherent commit is created with clean
      final source status.

## Completion Evidence

- Production handoff added owner-local selected-body validation APIs to
  `RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners` for runtime-scalar
  computed-mask store/load-store, computed-mask unit load/store, strided
  store, strided load/unit-store, indexed gather/unit-store, and indexed
  scatter/unit-load.
- `RVVSelectedBodyRealization.cpp` now includes the computed-mask memory owner
  header, calls those owner-local validators, and no longer defines the
  non-segment computed-mask memory selected-body validation helpers or
  operation/memory-form/inactive-lane constants. It still owns selected-body
  realization dispatch and realized IR materialization.
- Focused C++ coverage in `tianchenrv-rvv-extension-plugin-test` now directly
  calls the computed-mask memory owner-local validators for representative
  runtime-scalar masked-store and computed-mask unit load/store pre-realized
  bodies, then proves the selected-body producer and provider route still work.
- Generated-bundle dry-run passed for affected computed-mask memory paths under
  `artifacts/tmp/stage2_rvv_computed_mask_memory_handoff/codex-computed-mask-memory-handoff-all`:
  `runtime_scalar_cmp_masked_store`,
  `runtime_scalar_cmp_masked_load_store`,
  `computed_masked_unit_load_store`,
  `computed_masked_strided_store`,
  `computed_masked_strided_load_unit_store`,
  `computed_masked_indexed_gather_load_unit_store`, and
  `computed_masked_indexed_scatter_store_unit_load`.
- Direct route-entry negative evidence failed closed with the expected retired
  shortcut diagnostic under
  `artifacts/tmp/stage2_rvv_computed_mask_memory_handoff/codex-direct-route-entry-negative`.
- Base-memory non-regression generated-bundle dry-run passed under
  `artifacts/tmp/stage2_rvv_computed_mask_memory_handoff/codex-base-memory-nonregression`.
- Contraction non-regression generated-bundle dry-run passed for
  `computed_masked_strided_input_widening_dot_reduce_add` under
  `artifacts/tmp/stage2_rvv_computed_mask_memory_handoff/codex-contraction-nonregression`.
- Authority scans confirmed the central selected-body file no longer defines
  the non-segment computed-mask memory validators, while production touched
  owner/central files contain no legacy i32, source-front-door, descriptor,
  direct-C/source-export, artifact-name/script-derived, or route-id authority.
  Existing legacy/source-front-door strings in
  `test/Plugin/RVVExtensionPluginTest.cpp` remain pre-existing negative and
  fail-closed test inventory.
- Checks passed:
  `ninja -C build tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  generated-bundle dry-runs above;
  bounded authority scans;
  `git diff --check`;
  `ninja -C build check-tianchenrv` with 464/464 tests passed.
- No `ssh rvv` run was required because this round moved validation authority
  without changing runtime behavior or making a new runtime/correctness/
  performance claim.
- Spec update review completed: no `.trellis/spec/**` edit was needed because
  the existing computed-mask memory statement-plan and selected-body owner
  specs already require this owner-local handoff pattern.

## Out Of Scope

- No segment2 computed-mask selected-body handoff in this round.
- No new computed-mask memory coverage, emitted semantic change, runtime
  correctness/performance claim, `ssh rvv` claim, or target-hardware evidence
  unless a runtime behavior change is introduced.
- No base-memory, contraction, MAcc, reduction, compare/select, widening
  conversion, high-level frontend, source-front-door, report/dashboard, or
  broad-matrix work.

## Technical Notes

- Specs read for this PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/guides/plugin-locality-review-guide.md`.
- Nearest archived pattern:
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-base-memory-selected-body-provider-handoff-closure/prd.md`.
- Production files to inspect/modify first:
  `include/TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  and `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`.
- Focused test entry point:
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Validation Plan

1. Validate and start this Trellis task.
2. Implement the bounded non-segment computed-mask memory owner-local
   validation handoff.
3. Update focused C++ tests to prove the owner-local validation API boundary
   where current coverage does not.
4. Run focused build/test target for `tianchenrv-rvv-extension-plugin-test`.
5. Run representative generated-bundle dry-runs for affected computed-mask
   memory selected-boundary paths.
6. Run base-memory and contraction non-regression focused evidence.
7. Run bounded authority scans and `git diff --check`.
8. Run `ninja -C build check-tianchenrv` or record the exact blocker.
9. Finish/archive the task and create one coherent commit.
