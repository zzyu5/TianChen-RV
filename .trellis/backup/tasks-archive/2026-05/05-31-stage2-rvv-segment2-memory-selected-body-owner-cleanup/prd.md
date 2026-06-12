# Stage2 RVV segment2 memory selected-body owner-side cleanup

## Goal

Move segment2 memory selected-body realization ownership out of
`lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` into a dedicated RVV
selected-body owner module. The central selected-body file should retain only
owner registry / dispatch glue for this family, while the segment2 owner
recognizes and realizes the active segment2 selected-body routes:

- `computed_masked_segment2_load_unit_store`
- `computed_masked_segment2_store_unit_load`
- `computed_masked_segment2_update_unit_load`
- `segment2_deinterleave_unit_store`
- `segment2_interleave_unit_load`

The production behavior must remain the existing selected-boundary path:
selected typed pre-realized `tcrv_rvv` body -> RVV selected-body owner ->
realized typed `tcrv_rvv` body -> existing segment2 provider route-family facts
-> `TCRVEmitCLowerableRoute` -> neutral EmitC -> target artifact evidence.

## Direction Source

- Direction title: `Switch: Stage2 RVV segment2 memory selected-body owner-side cleanup`
- Module owner: RVV plugin-local selected-body realization owner for segment2
  memory bodies.
- Repository root: `/home/kingdom/phdworks/TianchenRV`
- `.trellis/.current-task` was absent at session start, so this task was
  created from the supplied Hermes Direction Brief.

## Current Repository Facts

- Current HEAD is `364c7ca6 rvv: move base memory selected-body owner-side`;
  the worktree was clean before task creation.
- The previous archived task added
  `RVVBaseMemoryMovementSelectedBodyRealizationOwner.{h,cpp}` and changed
  `RVVSelectedBodyRealization.cpp` so base-memory production realization is
  owner-side.
- `RVVSelectedBodyRealization.cpp` still defines the segment2 selected-body
  owner predicate and wrapper locally.
- `RVVSelectedBodyRealization.cpp` still realizes plain
  segment2-deinterleave and segment2-interleave bodies locally.
- Computed-mask segment2 validation and realization helpers already live in
  `RVVEmitCSegment2RouteFamilyPlanOwners`, and current tests treat those
  validators as segment2 owner-local validation surfaces.
- Segment2 route-family provider planning and statement-plan ownership already
  exist in `RVVEmitCSegment2RouteFamilyPlanOwners` and
  `RVVEmitCMemoryStatementPlanOwners`; this task must preserve, not replace,
  those route-provider boundaries.

## Requirements

1. Add a dedicated
   `RVVSegment2MemorySelectedBodyRealizationOwner.{h,cpp}` module.
2. Move the segment2 selected-body predicate out of
   `RVVSelectedBodyRealization.cpp` into the new owner module.
3. Move plain segment2 deinterleave/interleave realization logic out of
   `RVVSelectedBodyRealization.cpp` into the new owner module.
4. Route computed-mask segment2 load/store/update bodies through the new owner
   module while preserving the existing validation and realization behavior.
5. Update the central owner registry so the `segment2 memory` entry points at
   the new owner module.
6. Keep the central selected-body file free of segment2 production
   construction logic; it may only keep neutral registry / dispatch glue.
7. Preserve existing validation and fail-closed diagnostics for body shape,
   op kind, segment count, mask source, field roles, dtype/config/policy,
   ABI roles/order, selected variant requires, mixed pre-realized/realized
   bodies, stale route metadata, and direct route-entry-only claims.
8. Preserve the realized typed `tcrv_rvv` facts consumed by provider plans:
   runtime `n`/AVL, `setvl`, `with_vl`, segment2 load/store, masked segment2
   load/store, field load/store, compare mask, update arithmetic, field roles,
   memory-form direction, SEW/LMUL, and policy.
9. Do not change segment2 route-family planning, statement-plan construction,
   target artifact validation, common EmitC semantics, or unrelated selected
   body families except for include / CMake fallout caused by this owner split.

## Acceptance Criteria

- [ ] New segment2 selected-body owner header/source exist and are compiled in
      the RVV plugin library.
- [ ] `RVVSelectedBodyRealization.cpp` no longer defines the segment2
      selected-body predicate or segment2 production realization bodies.
- [ ] The central owner registry still contains exactly one `segment2 memory`
      entry, but its predicate and realization hook are provided by the new
      owner module.
- [ ] Existing C++ selected-body owner registry and segment2 owner-local
      negative tests pass.
- [ ] Focused segment2 selected-boundary and generated-bundle dry-runs pass for
      representative computed-mask load/store/update and plain
      deinterleave/interleave paths.
- [ ] Existing direct pre-realized segment2 route-entry fail-closed behavior is
      preserved.
- [ ] Base memory owner extraction has at least focused non-regression coverage.
- [ ] Bounded scans over touched files show no new central, name-derived,
      metadata-derived, descriptor-derived, ABI-string-derived,
      script/artifact-derived, source-front-door-derived, route-id-derived,
      exact-intrinsic-derived, direct-route-entry-only, or legacy-i32-derived
      segment2 authority.
- [ ] `git diff --check` passes.
- [ ] `cmake --build build --target check-tianchenrv -j2` passes, or the exact
      blocker is recorded.
- [ ] The task is finished/archived and the round ends with one coherent commit,
      unless an exact blocker prevents completion.

## Out of Scope

- No new segment2 semantics or segment counts.
- No dtype/LMUL clone batches.
- No high-level Linalg/frontend work.
- No one-intrinsic wrappers.
- No broad selected-body framework rewrite.
- No route-table or provider expansion outside the segment2 owner boundary.
- No source-front-door positive route.
- No dashboard/report/evidence-only round.
- No churn to completed base memory, contraction, MAcc, widening dot, or
  standalone reduction behavior except focused non-regression checks.

## Validation Plan

1. Run focused RVV plugin C++ tests that cover selected-body owner registry,
   segment2 owner-local validation/fail-closed behavior, provider-plan
   consumption, and base-memory non-regression.
2. Run representative generated-bundle dry-run tests:
   computed-mask segment2 load, store, update; pre-realized plain
   deinterleave/interleave; direct pre-realized fail-closed checks.
3. Run bounded authority scans over touched owner / central / test files.
4. Run `git diff --check`.
5. Run `cmake --build build --target check-tianchenrv -j2` if the focused
   checks pass.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous task reference:
  `.trellis/tasks/archive/2026-05/05-31-05-31-stage2-rvv-base-memory-selected-body-owner-cleanup/prd.md`.
- Existing segment2 route-family provider planning remains in
  `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`.
- Existing segment2 statement-plan consumption remains in
  `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`.
