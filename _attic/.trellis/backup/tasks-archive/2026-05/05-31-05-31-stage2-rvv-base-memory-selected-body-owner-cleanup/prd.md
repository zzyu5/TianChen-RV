# Stage2 RVV base memory-movement selected-body owner cleanup

## Goal

Move base memory-movement selected-body realization out of
`lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` into a dedicated owner module
so the central file keeps only registry and neutral dispatch glue for this
family.

The new owner module must own the selected-body predicate, validation,
realization, and fail-closed behavior for the active base-memory subfamilies:

- `strided_load_unit_store`
- `unit_load_strided_store`
- `indexed_gather_unit_store`
- `indexed_scatter_unit_load`
- `masked_unit_load_store`
- `masked_unit_store`

The owner must preserve runtime ABI order, `setvl` / `with_vl` placement,
memory-form, mask, stride, index, passthrough, and source/destination facts so
the existing RVV route-family planning and target-artifact layers continue to
consume the same structural evidence.

This task is about moving ownership, not expanding semantics. The existing
route-family plan owners in
`lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp` remain the
source of provider-plan facts.

## Direction Source

- Direction title: `RVV base memory movement selected-body realization owner cleanup`
- Module owner: RVV base memory movement selected-body realization owner and
  its route-family plan boundary
- Repository root: `/home/kingdom/phdworks/TianchenRV`
- The task was created from the supplied Hermes Direction Brief because
  `.trellis/.current-task` was absent at session start.

## Current Repository Facts

- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` still defines the base
  memory selected-body owner predicate and realization hook inline.
- The same file also carries a dead or redundant base-memory branch inside the
  owner-local realization helper, so the family is still centralized there.
- The base-memory route-family plan owners already exist in
  `include/TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h`
  and `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`.
- Existing plugin and generated-bundle tests already exercise the selected
  base-memory routes; this round should preserve them while moving ownership
  out of the central realization file.

## Requirements

1. Add a dedicated base-memory selected-body owner header and implementation
   following the existing RVV owner naming pattern.
2. Move the base-memory predicate and realization hook out of
   `RVVSelectedBodyRealization.cpp` into that owner module.
3. Update the central owner registry to point at the new owner module and keep
   the rest of the central dispatcher neutral.
4. Remove the base-memory realization branch body from the central owner-local
   helper so the base-memory family no longer has production realization logic
   in `RVVSelectedBodyRealization.cpp`.
5. Preserve the existing base-memory validation and materialization order for
   strided load/store, indexed gather/scatter, and masked memory bodies.
6. Keep route-family plan derivation and provider-plan facts unchanged; do not
   alter contraction, MAcc, reduction, widening conversion, segment2, or
   elementwise semantics in this round.
7. Keep unsupported or stale pre-realized bodies fail-closed, including
   direct-route-entry-only claims and metadata-derived authority.

## Acceptance Criteria

- [ ] New owner files exist and are compiled from the RVV plugin library.
- [ ] `RVVSelectedBodyRealization.cpp` no longer contains the base-memory
      predicate or realization implementation, and its registry entry binds to
      the new owner module.
- [ ] The base-memory owner still validates and realizes all six active
      subfamilies with the same runtime ABI order, `setvl` / `with_vl`
      placement, memory-form, mask, stride, index, and passthrough facts.
- [ ] Focused C++ tests still pass for the selected-body owner registry,
      selected pre-realized route path, and the base-memory route-family
      planning / generated-bundle coverage.
- [ ] Focused generated-bundle dry-runs still show the selected-boundary
      base-memory routes and fail closed for unsupported or stale claims.
- [ ] `cmake --build build --target check-tianchenrv -j2` passes, or the exact
      blocker is recorded.
- [ ] `git diff --check` passes and `git status --short` is clean at the end
      of the round.

## Technical Approach

1. Copy the base-memory owner-specific helpers and realization logic into a
   dedicated `RVVBaseMemoryMovementSelectedBodyRealizationOwner.{h,cpp}` pair.
2. Include the new header from `RVVSelectedBodyRealization.cpp` so the owner
   registry and any owner-local guard code can reference the new predicate.
3. Delete the base-memory realization block and associated helper residue from
   the central selected-body file.
4. Add the new source file to `lib/Plugin/RVV/CMakeLists.txt`.
5. Keep the existing base-memory route-family plan owner module untouched
   except for any compile-time include fallout caused by the move.
6. Run focused plugin tests, generated-bundle dry-runs, `git diff --check`,
   and `check-tianchenrv`; fix any regressions directly.

## Out of Scope

- Do not add new base-memory semantics, new RVV families, or new route-table
  entries.
- Do not change contraction, MAcc, reduction, widening conversion, segment2,
  or elementwise realization behavior.
- Do not move route-family plan ownership into the new selected-body owner
  module.
- Do not make this a report-only or test-only round; the production owner move
  is the point of the task.

## Validation Plan

1. Build the RVV plugin targets affected by the owner move.
2. Run the focused RVV plugin tests that cover owner selection and base-memory
   route-path behavior.
3. Run the selected generated-bundle dry-runs for strided load/store, indexed
   gather/scatter, and masked memory routes.
4. Run `git diff --check`.
5. Run `cmake --build build --target check-tianchenrv -j2`.

## Completion Evidence

- Production owner module created for base-memory selected-body realization.
- Central selected-body realization file reduced to neutral routing and shared
  non-base owner glue.
- Focused plugin and generated-bundle checks pass on the moved routes.
- The final repository status is clean and the task is archived with a
  coherent commit.
