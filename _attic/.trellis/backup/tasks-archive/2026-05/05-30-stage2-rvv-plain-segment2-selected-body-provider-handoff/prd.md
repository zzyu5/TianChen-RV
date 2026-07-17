# Stage2 RVV Plain Segment2 Selected-Body Provider Handoff

## Goal

Close the selected-body/provider handoff for plain RVV segment2 memory routes.
Validation authority for `segment2_deinterleave_unit_store` and
`segment2_interleave_unit_load` must move from central
`RVVSelectedBodyRealization.cpp` into the segment2 route-family owner surface.
Central selected-body realization may dispatch to owner-local validation and
materialize neutral realized `tcrv_rvv` structure, but it must not retain plain
segment2 legality constants, field-role/memory-form authority, provider-plan
construction decisions, or route acceptance logic.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state for this round is clean on `main` at `e77a5b10`
  (`rvv: close segment2 computed-mask selected-body handoff`).
- There was no active `.trellis/.current-task`; this task was created from the
  supplied Hermes Direction Brief.
- The previous archived task moved computed-mask segment2 selected-body
  validation/provider-plan authority into
  `RVVEmitCSegment2RouteFamilyPlanOwners`.
- `RVVEmitCSegment2RouteFamilyPlanOwners` already has plain segment2
  deinterleave/interleave route-family planning-owner registry entries and
  operand-binding plan IDs.
- `RVVSelectedBodyRealization.cpp` still owns the plain segment2
  deinterleave/interleave selected-body op-kind, memory-form, source/result
  memory-form, field-role predicates, and validators.
- The stable authority chain remains:
  `tcrv.exec envelope -> selected RVV variant -> typed low-level tcrv_rvv body
  -> RVV plugin-owned legality / selected-body realization / route provider ->
  TCRVEmitCLowerableRoute -> neutral EmitC materialization -> target artifact`.

## Requirements

1. Add owner-local selected-body validation declarations and definitions for
   `TypedSegment2DeinterleaveMemoryPreRealizedBodyOp` and
   `TypedSegment2InterleaveMemoryPreRealizedBodyOp` in
   `RVVEmitCSegment2RouteFamilyPlanOwners`.
2. Move plain segment2 legality constants, op-kind validation, memory-form
   validation, segment-count validation, field-role validation, distinct field
   role rejection, source/result memory-window validation, runtime `n`/AVL
   validation, SEW/LMUL/policy validation, mixed pre-realized/realized-body
   rejection, and selected variant `requires` validation into that segment2
   owner surface.
3. Keep `RVVSelectedBodyRealization.cpp` responsible for selected-body owner
   dispatch and neutral realized IR materialization only: `setvl`, `with_vl`,
   segment load, move, field stores, field loads, tuple/segment store, and
   erasing the consumed pre-realized body.
4. Preserve existing runtime behavior and generated artifact semantics. This is
   an ownership handoff, not new route coverage or runtime/performance work.
5. Unsupported or inconsistent facts must fail closed with targeted diagnostics
   before provider facts, statement plans, common EmitC, or target artifact
   export.
6. Preserve computed-mask segment2 behavior as non-regression; this round must
   not reopen or rewrite computed-mask segment2 except for shared helper reuse
   needed by the same segment2 owner surface.
7. Do not add descriptor-derived, name-derived, route-id-derived,
   ABI-string-derived, artifact-name-derived, exact-intrinsic-derived,
   common-EmitC-derived, direct-route-entry-only, source-front-door-derived,
   script-derived, stale mirror-derived, or legacy-i32 authority.

## Acceptance Criteria

- [x] Plain segment2 pre-realized selected-body validation declarations and
      definitions live in `RVVEmitCSegment2RouteFamilyPlanOwners`, not as
      central plain segment2 legality in `RVVSelectedBodyRealization.cpp`.
- [x] `RVVSelectedBodyRealization.cpp` dispatches plain segment2
      deinterleave/interleave validation through owner-local APIs and retains
      only selected-body dispatch plus realized IR materialization.
- [x] The two selected-boundary routes keep exact positive behavior:
      `segment2_deinterleave_unit_store` and
      `segment2_interleave_unit_load`.
- [x] Wrong operation kind, memory form, segment count, field role, duplicate
      field role, missing/wrong `mem_window`, missing/wrong runtime `n`/AVL,
      SEW/LMUL/policy mismatch, missing required capability, stale route
      metadata, mixed pre-realized/realized bodies, artifact-name or
      script-derived authority, and common-EmitC semantic invention fail closed
      before provider route construction or target export.
- [x] Focused C++ coverage proves segment2 owner-local validation APIs can be
      called directly for representative plain segment2 deinterleave and
      interleave pre-realized bodies, and that the selected-body producer plus
      provider route still work.
- [x] Generated-bundle dry-runs pass for `segment2_deinterleave_unit_store` and
      `segment2_interleave_unit_load`, or exact blockers are recorded.
- [x] Computed-mask segment2 generated-bundle dry-runs remain green as
      non-regression.
- [x] Bounded authority scans show central selected-body no longer owns plain
      segment2 legality constants or validation authority, while touched
      production/test files have no legacy i32/source-front-door/descriptor/
      direct-C/source-export/direct-route or mirror-only authority drift.
- [x] `git diff --check`, focused RVV plugin build/test, and
      `check-tianchenrv` pass, or exact blockers are recorded.
- [x] Task is finished/archived and one coherent commit is created with clean
      final source status.

## Out Of Scope

- No new segment2 route coverage, segment3/N support, or new op family.
- No dtype/LMUL clone batches.
- No high-level frontend/Linalg lowering.
- No descriptor/source-front-door positive route.
- No artifact metadata scheme, dashboard, report-only work, or broad smoke
  matrix.
- No runtime correctness/performance claim and no `ssh rvv` evidence unless a
  runtime behavior change is introduced.

## Technical Notes

- Specs read for this PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Nearest archived pattern:
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-segment2-computed-mask-selected-body-provider-handoff/prd.md`.
- Production files to inspect/modify first:
  `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`, and
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`.
- Focused test entry point:
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Validation Plan

1. Curate Trellis implement/check context for the relevant specs.
2. Implement the bounded plain segment2 owner-local validation handoff.
3. Update focused C++ tests to prove the owner-local validation API boundary
   where current coverage does not.
4. Run focused build/test target for `tianchenrv-rvv-extension-plugin-test`.
5. Run generated-bundle dry-runs for the two affected plain segment2
   selected-boundary paths.
6. Run computed-mask segment2 generated-bundle non-regression dry-runs.
7. Run bounded authority scans and `git diff --check`.
8. Run `ninja -C build check-tianchenrv` or record the exact blocker.
9. Finish/archive the task and create one coherent commit.

## Completion Evidence

- Production owner surface changed:
  `RVVEmitCSegment2RouteFamilyPlanOwners.h` declares owner-local validation
  APIs for plain segment2 deinterleave and interleave pre-realized bodies, and
  `RVVEmitCSegment2RouteFamilyPlanOwners.cpp` defines the op-kind,
  memory-form, segment-count, field-role, source/destination memory-form,
  dtype/config/policy, runtime ABI role, selected variant `requires`, and
  mixed-body checks.
- Central selected-body realization changed:
  `RVVSelectedBodyRealization.cpp` no longer defines plain segment2 legality
  predicates or validators. It calls the segment2 owner APIs before neutral
  `setvl`, `with_vl`, segment load/store, field load/store, move, tuple/store,
  and erase materialization.
- Focused C++ evidence:
  `ninja -C build tianchenrv-rvv-extension-plugin-test` passed, and
  `build/bin/tianchenrv-rvv-extension-plugin-test` printed
  `RVV extension plugin smoke test passed`.
- Generated-bundle evidence:
  `scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body` passed for `segment2_deinterleave_unit_store`,
  `segment2_interleave_unit_load`,
  `computed_masked_segment2_load_unit_store`,
  `computed_masked_segment2_store_unit_load`, and
  `computed_masked_segment2_update_unit_load` with runtime counts
  `0, 1, 7, 16, 23, 257`. Artifact directory:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260530T113141Z`.
- Authority scans:
  the central selected-body scan no longer finds plain segment2 op-kind,
  memory-form, or field-role legality constants/validators; remaining
  segment2 hits there are neutral source/destination attribute materialization
  and owner API calls. Touched production-file authority scan only found
  unrelated widening-conversion `i32m1` relation strings in central selected
  body code, not plain segment2 authority.
- Full check:
  `git diff --check` passed and `ninja -C build check-tianchenrv` passed
  464/464 tests.
- Runtime evidence:
  not run; this round did not claim new runtime correctness or performance and
  preserved generated artifact semantics through dry-run evidence.
- Spec update review:
  no `.trellis/spec/**` change was needed because the existing RVV plugin,
  EmitC route, exec contract, and testing specs already encode this owner-local
  selected-body/provider boundary.
