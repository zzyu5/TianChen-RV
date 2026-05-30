# Stage2 RVV segment2 Generated-Bundle Direct-Entry Demotion

## Goal

Remove any remaining direct pre-realized route-entry authority from the
segment2 generated-bundle path while keeping the owner-built segment2
route-family plans authoritative. The selected lowering boundary must still
consume the pre-realized selected body, produce the same route/provider facts,
and export the same target/header artifacts without relying on route-id,
script-derived shortcut, or direct-entry authority.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current worktree started clean on branch `main`.
- The active task directory is
  `.trellis/tasks/05-30-stage2-rvv-segment2-generated-bundle-direct-entry-demotion/`.
- Relevant long-term authority chain from the RVV spec remains:
  `tcrv.exec envelope -> selected RVV variant -> typed low-level tcrv_rvv body
  -> RVV plugin-owned legality / selected-body realization / route provider ->
  TCRVEmitCLowerableRoute -> common EmitC materializer -> target artifact`.
- The RVV spec already treats selected-body realization as the production
  boundary and says direct route-entry support is retired unless a later task
  explicitly restores it with matching provider facts and evidence.
- The current generator and target fixtures already contain segment2
  pre-realized selected-body coverage and direct-entry fail-closed coverage;
  this task is about making the boundary explicit and ensuring no segment2
  route authority leaks back in through generated-bundle metadata, helper
  support flags, or test phrasing.
- Focused evidence files to inspect:
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-*segment2*`,
  `test/Target/RVV/*segment2*`.

## Requirements

1. Keep the change bounded to the RVV generated-bundle path and the matching
   segment2 tests/artifact fixtures.
2. Ensure segment2 direct pre-realized route-entry requests fail closed and do
   not act as a supported production path.
3. Keep owner-built segment2 route-family plans authoritative for the
   pre-realized selected-body path.
4. Preserve selected-boundary dry-runs for plain segment2 interleave /
   deinterleave and computed-mask segment2 load/store/update routes.
5. Preserve target artifact and header export for the same segment2 routes,
   with the emitted evidence still flowing from owner-built plan facts.
6. Keep route-entry / script-derived authority out of production metadata,
   evidence mirrors, and target assertions.
7. Do not add new segment2 op families, do not widen into indexed
   gather/scatter or contraction work, and do not touch unrelated RVV
   families.
8. Do not turn this into a report-only or test-only round; if a code path still
   advertises segment2 direct-entry authority, remove or fail-close it in the
   production path.

## Acceptance Criteria

- [ ] `scripts/rvv_generated_bundle_abi_e2e.py` no longer presents segment2
      pre-realized direct-entry support as an active route authority.
- [ ] Segment2 direct-pre-realized script probes fail closed with a retired
      direct-entry diagnostic and do not reach bundle generation.
- [ ] Selected-boundary dry-runs for plain segment2 interleave/deinterleave
      and computed-mask segment2 load/store/update still succeed.
- [ ] Segment2 target/header artifact checks still pass and continue to show
      owner-built route-family facts, `route_entry_realization = false`, and no
      script-derived shortcut authority.
- [ ] Bounded authority scans show no new route-id-derived, script-derived, or
      direct-entry-derived segment2 authority in production code, generated
      bundle output, or artifact fixtures.
- [ ] `git diff --check` passes.
- [ ] `check-tianchenrv` passes, or the exact blocker is reported.
- [ ] Final `git status --short` is clean after a coherent commit.

## Out Of Scope

- No new segment2 op families.
- No indexed gather/scatter expansion.
- No contraction work.
- No unrelated RVV family edits.
- No descriptor-driven, source-front-door, or route-id authority reintroduction.
- No runtime correctness or performance claim unless real `ssh rvv` evidence is
  produced, which this task does not require.

## Validation Plan

1. Inspect the current generated-bundle segment2 gate and the segment2 owner
   boundary files.
2. Update the production path or tests where a segment2 direct-entry shortcut
   still leaks into supported surface area.
3. Run focused script dry-runs for:
   - one direct-pre-realized segment2 negative case,
   - one pre-realized selected-body segment2 success case,
   - one computed-mask segment2 selected-body success case.
4. Run the relevant target/header artifact checks for the segment2 fixtures.
5. Run bounded authority scans over the touched script, plugin, tests, and
   target fixtures.
6. Run `git diff --check`.
7. Run `check-tianchenrv` or report the exact blocker if it cannot complete.

## Technical Notes

- Spec files already in scope:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/testing/index.md`.
- The owner-built segment2 route-family plan files are the source of truth for
  segment2 route facts and must stay in charge of provider-plan derivation.
- The generated-bundle script should continue to be the evidence driver, not a
  semantic authority for segment2 compute or route support.
