# Stage2 RVV computed-mask memory route-family handoff

## Goal

Create a dedicated RVV plugin-local owner boundary for the non-segment
computed-mask memory route family so compare-produced masks become first-class
in masked load/store lowering. The production boundary must make the active
non-segment computed-mask memory consumers route through a dedicated
owner-local plan/provider verification surface instead of keeping the family
centralized in the generic route-planning path.

This round covers the active non-segment computed-mask memory consumers:

- `RuntimeScalarComputedMaskStore`
- `RuntimeScalarComputedMaskLoadStore`
- `ComputedMaskUnitLoadStore`
- `ComputedMaskStridedStore`
- `ComputedMaskStridedLoadUnitStore`
- `ComputedMaskIndexedGatherLoadUnitStore`
- `ComputedMaskIndexedScatterStoreUnitLoad`

The existing computed-mask segment2 memory boundary is out of scope for this
round and must remain on the segment2 path.

## What I Already Know

- The repository root is `/home/kingdom/phdworks/TianchenRV`.
- `.trellis/.current-task` was absent when this task was created, so the task
  was created from the Direction Brief before source edits.
- The working tree was already dirty when this task started, with unrelated
  MAcc route-family edits and a previously generated archive directory in the
  tree. Those changes must be preserved and not overwritten.
- The stable RVV authority chain remains:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body ->
  RVV plugin-owned legality / selected-body realization / route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC materialization -> target artifact.
- `RVVSelectedBodyComputedMaskMemoryRouteStatementPlan` and
  `getRVVSelectedBodyComputedMaskMemoryRouteStatementPlan(...)` already exist
  from the prior statement-plan ownership task and explicitly exclude
  computed-mask segment2 routes from the non-segment computed-mask memory
  statement boundary.
- Current central route-planning code still keeps the computed-mask memory
  route-family ownership logic in `RVVEmitCRoutePlanning.cpp` rather than in a
  dedicated owner module.
- The base memory and MAcc owner files already show the local shape to follow:
  dedicated owner-local classification, route-family validation, and
  provider-plan verification rather than central ad hoc branches.
- The current generated-bundle dry-run fixture for
  `computed_masked_unit_load_store` already proves the non-segment path can be
  exercised without `descriptor`, `direct-C`, `source-export`, or
  `tcrv_rvv.i32_` authority.

## Requirements

1. Add or repair a dedicated non-segment computed-mask memory route-family
   owner module.
2. The owner boundary must cover the active non-segment consumers listed in the
   goal and expose explicit consumer classification plus provider-plan
   verification hooks.
3. The owner boundary must validate compare-produced-mask provenance, runtime
   ABI order, SEW/LMUL/policy, memory form, mask role/source/form, and
   stride/index facts where applicable before provider materialization.
4. `runtime_scalar_cmp_masked_store` and
   `runtime_scalar_cmp_masked_load_store` must be first-class masked memory
   consumers in this owner boundary, not ad hoc branches hidden in the central
   planner.
5. Segment2 computed-mask routes remain excluded from this owner boundary and
   continue to use the existing segment2 memory path. This round must not
   broaden into segment2 route-family ownership or statement-plan changes.
6. Common EmitC/export must remain neutral. It may consume provider-built
   routes and mirrors, but it must not infer RVV semantics from route ids,
   metadata mirrors, helper strings, ABI strings, or artifact names.
7. Preserve the current emitted semantics, ABI order, intrinsic spelling,
   route ids, and generated artifacts for valid non-segment computed-mask
   memory routes.
8. Do not add new route coverage, new memory forms, MAcc, contraction,
   reduction, widening-conversion, source-front-door, descriptor/direct-C,
   source-export, or report/status work.

## Acceptance Criteria

- [ ] A dedicated non-segment computed-mask memory owner module exists and is
      used by the production route planning/provider path.
- [ ] The owner boundary covers the seven active non-segment computed-mask
      memory consumers listed in the goal.
- [ ] `runtime_scalar_cmp_masked_store` and
      `runtime_scalar_cmp_masked_load_store` are validated through the new
      owner boundary, including compare-produced-mask provenance and runtime
      ABI binding.
- [ ] The production route provider consumes the owner-local computed-mask
      memory plan/provider verification surface rather than reconstructing the
      family through central ad hoc branches.
- [ ] Computed-mask segment2 routes remain excluded from this owner boundary
      and continue through the existing segment2 memory path.
- [ ] Focused C++ tests cover owner registry membership, positive owner
      classification, fail-closed stale/mismatched facts, and production
      provider handoff for representative non-segment computed-mask memory
      routes.
- [ ] One representative generated-bundle dry-run for
      `computed_masked_unit_load_store` passes.
- [ ] `git diff --check` passes and `cmake --build build --target
      check-tianchenrv -j2` passes, or an exact blocker is recorded.
- [ ] The task is archived and one coherent commit records the work if the
      acceptance criteria are met.

## Out Of Scope

- No computed-mask segment2 ownership changes.
- No MAcc, contraction, reduction, widening-conversion, or other family
  expansion.
- No source-front-door positive routes, descriptor/direct-C/source-export
  resurrection, or helper-only cleanup.
- No runtime correctness/performance claim unless emitted behavior actually
  changes and the evidence is rerun.

## Technical Approach

1. Create a dedicated computed-mask memory owner header/cpp pair modeled on
   the existing RVV owner modules.
2. Route the non-segment computed-mask memory production path through the new
   owner boundary.
3. Keep the existing computed-mask memory statement-plan ownership boundary in
   place for the non-segment routes and leave segment2 on the segment2 path.
4. Add focused C++ tests for owner registry membership, positive/negative
   owner classification, and fail-closed stale or mismatched facts.
5. Re-run the representative computed-mask memory generated-bundle dry-run and
   the focused plugin test target.
6. Verify the touched scope with authority scans, `git diff --check`, and
   `check-tianchenrv`.

## Validation Plan

1. Validate and start the Trellis task.
2. Build the focused RVV plugin test target.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run a representative generated-bundle dry-run for
   `computed_masked_unit_load_store`.
5. Run targeted negative checks for stale or mismatched computed-mask memory
   facts.
6. Run a bounded authority scan over the touched RVV planning/provider/test
   files.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`.

## Completion Evidence

- Dedicated non-segment computed-mask memory owner boundary in production.
- Focused positive and fail-closed C++ coverage for the owner boundary.
- Representative computed-mask memory dry-run evidence.
- Authority scan and `git diff --check` evidence.
- `check-tianchenrv` evidence or an exact blocker.

## Definition Of Done

The non-segment computed-mask memory route family is owned by an explicit RVV
plugin-local boundary, the production route provider consumes that boundary
before common EmitC materialization, segment2 remains outside the round, the
focused tests and validation pass, the task is archived, and one coherent
commit records the work.
