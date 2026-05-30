# Stage2 RVV Segment2 Computed-Mask Selected-Body Provider Handoff

## Goal

Close the selected-body/provider handoff for RVV computed-mask segment2 memory
routes. Validation authority for
`computed_masked_segment2_load_unit_store`,
`computed_masked_segment2_store_unit_load`, and
`computed_masked_segment2_update_unit_load` must move from central
`RVVSelectedBodyRealization.cpp` into the segment2 route-family owner surface.
Central selected-body realization must keep only owner dispatch and neutral
realized IR materialization for `setvl`, `with_vl`, compare/mask,
segment2 load/store/update structure.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state for this round is clean on `main` at `8e47834e`
  (`rvv: close computed-mask memory selected-body handoff`).
- There was no active `.trellis/.current-task`; this task was created from the
  supplied Hermes Direction Brief.
- The archived non-segment computed-mask memory handoff moved pre-realized
  selected-body validation APIs into
  `RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners` and kept central selected
  body code limited to owner dispatch and realized IR materialization.
- `RVVEmitCSegment2RouteFamilyPlanOwners` already owns segment2
  route-family provider-plan classification, operand-binding derivation, and
  provider-plan checks for plain and computed-mask segment2 routes.
- `RVVSelectedBodyRealization.cpp` still owns the computed-mask segment2
  selected-body validation constants and validators for
  `TypedComputedMaskSegment2LoadPreRealizedBodyOp` and
  `TypedComputedMaskSegment2StorePreRealizedBodyOp`; the store validator also
  gates the update form through `arithmetic_kind = "add"`.
- Plain segment2 deinterleave/interleave validation remains adjacent in
  central code and is not the named migration target for this round.
- The stable authority chain remains:
  `tcrv.exec envelope -> selected RVV variant -> typed low-level tcrv_rvv body
  -> RVV plugin-owned legality / selected-body realization / route provider ->
  TCRVEmitCLowerableRoute -> neutral EmitC materialization -> target artifact`.

## Requirements

1. Add owner-local selected-body validation declarations and definitions for
   computed-mask segment2 load and store/update bodies in
   `RVVEmitCSegment2RouteFamilyPlanOwners`.
2. Move computed-mask segment2 legality constants, ABI-role validation,
   mask-policy validation, segment-count validation, segment field-role
   validation, source/destination memory-form validation, inactive-lane policy
   validation, dtype/config/policy validation, runtime AVL binding validation,
   mixed pre-realized/realized-body rejection, selected variant `requires`
   validation, and update arithmetic validation into that segment2 owner
   surface.
3. Keep `RVVSelectedBodyRealization.cpp` responsible for selected-body owner
   dispatch and realized IR materialization only: setvl/with_vl construction,
   compare and mask construction, old-field passthrough loads, field payload
   loads, update arithmetic materialization, masked segment2 load/store, and
   final field stores.
4. Preserve existing runtime behavior and generated artifact semantics. This is
   an ownership handoff, not new route coverage or runtime/performance work.
5. Unsupported or inconsistent facts must fail closed with targeted diagnostics
   before provider facts, statement plans, common EmitC, or target artifact
   export.
6. Preserve the existing segment2 provider-plan boundary: computed-mask
   segment2 routes must consume the verified computed-mask memory family plan,
   segment2 provider plan, memory operand-binding facts, route-control provider
   plan, and owner-derived statement facts before `TCRVEmitCLowerableRoute`.
7. Keep the completed non-segment computed-mask memory owner boundary green as
   a non-regression.
8. Do not add descriptor-derived, name-derived, route-id-derived,
   ABI-string-derived, artifact-name-derived, exact-intrinsic-derived,
   common-EmitC-derived, direct-route-entry-only, source-front-door-derived,
   script-derived, stale mirror-derived, or legacy-i32 authority.

## Acceptance Criteria

- [x] Computed-mask segment2 pre-realized selected-body validation declarations
      and definitions live in `RVVEmitCSegment2RouteFamilyPlanOwners`, not as
      central computed-mask segment2 legality in
      `RVVSelectedBodyRealization.cpp`.
- [x] `RVVSelectedBodyRealization.cpp` dispatches computed-mask segment2 load
      and store/update validation through owner-local APIs and retains only
      selected-body dispatch plus realized IR materialization.
- [x] The three selected-boundary routes keep exact positive behavior:
      `computed_masked_segment2_load_unit_store`,
      `computed_masked_segment2_store_unit_load`, and
      `computed_masked_segment2_update_unit_load`.
- [x] Wrong operation kind, predicate, memory form, mask role/source/form,
      inactive-lane policy, segment count, field roles, source/destination
      memory forms, runtime ABI roles, dtype/config/policy, update arithmetic,
      stale mirror metadata, direct-route-only authority, artifact-name or
      script-derived authority, and common-EmitC semantic invention fail closed
      before provider route construction or target export.
- [x] Focused C++ coverage proves segment2 owner-local validation APIs can be
      called directly for representative computed-mask segment2 load and
      store/update pre-realized bodies, and that the selected-body producer plus
      provider route still work.
- [x] Generated-bundle dry-runs pass for
      `computed_masked_segment2_load_unit_store`,
      `computed_masked_segment2_store_unit_load`, and
      `computed_masked_segment2_update_unit_load`, or exact blockers are
      recorded.
- [x] Non-segment computed-mask memory handoff remains green as a
      non-regression.
- [x] Bounded authority scans show central selected-body no longer owns
      computed-mask segment2 legality constants or validation authority, while
      touched production/test files have no legacy i32/source-front-door/
      descriptor/direct-C/source-export/direct-route or mirror-only authority
      drift.
- [x] `git diff --check`, focused RVV plugin build/test, and
      `check-tianchenrv` pass, or exact blockers are recorded.
- [x] Task is finished/archived and one coherent commit is created with clean
      final source status.

## Out Of Scope

- No new segment2 route coverage or new segment2 operation.
- No migration of plain segment2 deinterleave/interleave validation beyond
  minimal shared factoring needed by the computed-mask segment2 owner API.
- No non-computed-mask segment2 artifact behavior change.
- No high-level frontend lowering, conversion/dtype/LMUL batches,
  one-intrinsic wrappers, dashboards, reports, broad smoke matrices, global
  tuning, or Stage3/future plugin work.
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
  `.trellis/tasks/archive/2026-05/05-30-stage2-rvv-computed-mask-memory-selected-body-provider-handoff-closure/prd.md`.
- Production files to inspect/modify first:
  `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`, and
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`.
- Focused test entry point:
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Validation Plan

1. Curate Trellis implement/check context for the relevant specs.
2. Implement the bounded computed-mask segment2 owner-local validation handoff.
3. Update focused C++ tests to prove the owner-local validation API boundary
   where current coverage does not.
4. Run focused build/test target for `tianchenrv-rvv-extension-plugin-test`.
5. Run generated-bundle dry-runs for the three affected computed-mask segment2
   selected-boundary paths.
6. Run a non-segment computed-mask memory non-regression dry-run or focused
   existing check.
7. Run bounded authority scans and `git diff --check`.
8. Run `ninja -C build check-tianchenrv` or record the exact blocker.
9. Finish/archive the task and create one coherent commit.

## Completion Evidence

- Production owner surface changed:
  `RVVEmitCSegment2RouteFamilyPlanOwners.h` declares owner-local validation
  APIs for computed-mask segment2 load and store/update bodies, and
  `RVVEmitCSegment2RouteFamilyPlanOwners.cpp` defines the legality, ABI-role,
  mask-policy, segment-field, memory-form, dtype/config/policy, selected
  variant `requires`, mixed-body rejection, and update arithmetic checks.
- Central selected-body realization changed:
  `RVVSelectedBodyRealization.cpp` no longer defines computed-mask segment2
  selected-body legality constants or validators. It dispatches validation to
  the segment2 owner APIs and retains neutral setvl/with_vl/compare/load/store/
  update materialization.
- Provider-plan facts consumed:
  the segment2 owner surface now validates typed body/config/runtime facts
  before the existing segment2 route-family provider plan, computed-mask memory
  family plan, route-control provider plan, operand-binding plan, and
  statement facts can authorize `TCRVEmitCLowerableRoute`.
- Focused C++ evidence:
  `ninja -C build tianchenrv-rvv-extension-plugin-test` passed, and
  `build/bin/tianchenrv-rvv-extension-plugin-test` printed
  `RVV extension plugin smoke test passed`.
- Generated-bundle evidence:
  `scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body` passed for
  `computed_masked_segment2_load_unit_store`,
  `computed_masked_segment2_store_unit_load`, and
  `computed_masked_segment2_update_unit_load` with runtime counts
  `0, 1, 7, 16, 23, 257`.
- Non-regression evidence:
  the same generated-bundle dry-run passed for
  `computed_masked_unit_load_store` as the completed non-segment
  computed-mask memory owner boundary.
- Authority scans:
  touched production files have no descriptor/direct-C/source-export/
  source-front-door/legacy-i32/rvv-i32m1/RVVI32M1/route-id/artifact-name/
  emission-plan/status authority hits. The central selected-body scan only
  reports calls to the owner-local computed-mask segment2 validation APIs.
  Existing test hits are retained negative/fail-closed inventory, not
  production authority.
- Full check:
  `git diff --check` passed and `ninja -C build check-tianchenrv` passed
  464/464 tests.
- Runtime evidence:
  not run; this round did not claim new runtime correctness or performance and
  preserved generated artifact semantics through dry-run evidence.
- Spec update review:
  no `.trellis/spec/**` change was needed because the existing RVV plugin,
  EmitC route, exec contract, and testing specs already encode this
  owner-local selected-body/provider boundary.
