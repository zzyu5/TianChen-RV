# Stage2 RVV Provider Route-Family Planning Registry Owner

## Goal

Introduce one RVV plugin-local provider/planning owner boundary for active
segment2 route-entry families. The production RVV EmitC provider should build
segment2 statement/provider plans through registered segment2 family planning
owners instead of keeping per-family booleans and ad hoc checks in the central
segment2 statement-plan body.

The primary migrated consumer is
`computed_masked_segment2_update_unit_load`. At least one adjacent active
segment2 route, such as computed-mask segment2 store/load or plain segment2
interleave/deinterleave, must be migrated through the same provider/planning
registry.

## Direction Source

- Direction title: `Stage2 RVV provider route-family planning registry owner`.
- Module owner: RVV plugin-local provider/planning boundary that turns
  selected-body segment2 route-entry family ownership into provider-built
  `TCRVEmitCLowerableRoute` planning facts.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `af07b488 rvv: add segment2 route-entry family registry`.
- No `.trellis/.current-task` existed, so this task was created from the
  supplied Hermes direction brief before source edits.

## What I Already Know

- The spec authority chain is:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / selected-body realization / route provider ->
  `TCRVEmitCLowerableRoute` -> neutral common EmitC -> target artifact.
- The previous archived task
  `05-26-05-26-stage2-rvv-selected-body-route-entry-family-registry-owner`
  introduced `RVVSelectedBodySegment2RouteEntryFamilyOwner` and registered five
  selected-body route-entry families: computed-mask segment2 load, store,
  update, plain segment2 deinterleave, and plain segment2 interleave.
- Current `RVVEmitCRoutePlanning.cpp` already has broad memory-family provider
  owners and a migrated statement-plan owner for `segment2 memory`.
- Current `getRVVSelectedBodySegment2MemoryRouteStatementPlan(...)` still
  performs sub-family selection with local booleans for computed-mask segment2
  load/store/update and plain deinterleave/interleave. This is the provider
  boundary to make owner-driven.
- Existing route-family plans already carry typed/config/runtime/mask/memory
  facts:
  - computed-mask segment2 load/store/update use
    `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan`;
  - plain segment2 deinterleave/interleave use
    `RVVSelectedBodySegment2MemoryRouteFamilyPlan`;
  - all segment2 routes consume `RVVSelectedBodyMemoryRouteOperandBindingFacts`
    and `RVVSelectedBodyRouteControlProviderPlan` before statement construction.
- Common EmitC/export must remain neutral. Metadata, route ids, artifact names,
  scripts, ABI strings, descriptors, exact intrinsic spellings, and legacy i32
  helpers are mirror-only or forbidden as authority.

## Requirements

1. Add or extract a provider/planning owner registry for active segment2
   route-family planning entries.
2. Registry entries must expose an explicit family name, consumer predicate over
   the provider route description, and provider-plan builder/verifier over
   `RVVSelectedBodyRouteAnalysis`, route materialization facts, memory
   operand-binding facts, and route-control facts.
3. The registry must cover at least:
   - computed-mask segment2 update;
   - one adjacent active segment2 consumer.
   Prefer covering all five active selected-body segment2 route-entry families
   if it is a localized change.
4. `getRVVSelectedBodySegment2MemoryRouteStatementPlan(...)` must consume the
   provider/planning owner result instead of owning the sub-family dispatch
   itself.
5. The provider/planning owner must fail closed for missing or stale facts:
   mismatched selected-body family equivalent, wrong operation kind, segment
   factor mismatch, mask/VL mismatch, missing passthrough, wrong stream or ABI
   order, dtype/config mismatch, memory form mismatch, unsupported arithmetic
   kind, runtime `n`/AVL mismatch, unsupported policy, or stale route/control
   plan facts.
6. The interface may name the selected-body route-entry family as a mirror of
   the owner-selected provider plan, but it must not use route ids, metadata,
   scripts, ABI strings, artifact names, descriptors, common EmitC, exact
   intrinsic spelling, source-front-door fixtures, or legacy i32 helpers as
   support authority.
7. Do not add new RVV operation coverage, new segment width, dtype/LMUL clone
   batches, source-front-door positives, high-level frontend routes, common
   EmitC RVV semantics, dashboards, or evidence-only changes.

## Acceptance Criteria

- [x] Production C++ changes in `RVVEmitCRoutePlanning` and/or
      `RVVEmitCRouteProvider` introduce or strengthen a plugin-local segment2
      route-family planning owner interface.
- [x] Computed-mask segment2 update and at least one adjacent segment2 family
      are migrated behind the provider/planning owner boundary. If feasible,
      all five active segment2 route-entry families are covered exactly once.
- [x] The segment2 statement-plan construction path consumes the owner-built
      provider plan and no longer owns the central sub-family dispatch as an
      ad hoc boolean cluster.
- [x] Focused C++ tests prove owner registry membership/order/hooks,
      exact-one classification for migrated segment2 provider consumers, empty
      non-consumer behavior, and production statement-plan consumption.
- [x] Focused C++ tests prove fail-closed diagnostics for stale or missing
      family plans, route-control facts, operand binding facts, segment count,
      memory form, mask producer/source, arithmetic kind, runtime ABI order,
      typed config/policy, and stale analysis/materialization inputs.
- [x] Generated-bundle dry-run covers computed-mask segment2 update and one
      migrated adjacent segment2 family.
- [x] `ssh rvv` is run for computed-mask segment2 update unless the emitted
      target sequence and ABI are proven preserved; if preserved, run one
      migrated representative or record exact preservation evidence.
- [x] Focused non-regression covers computed-mask segment2 store/load,
      segment2 deinterleave/interleave, masked elementwise, reduction,
      scalar-broadcast, conversion, runtime-scalar MAcc, compare/select, MAcc,
      contraction, and base memory route-entry paths.
- [x] Bounded authority scan over touched files finds no new positive
      legacy-i32, source-front-door, descriptor, ABI-string, artifact-name,
      script-derived, metadata-derived, route-id-derived, exact-intrinsic-
      derived, or common-EmitC-derived authority.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded and the task
      remains open.
- [x] Task status, journal, archive, and one coherent commit are completed if
      all acceptance criteria are satisfied.

## Completion Evidence

- Added `RVVSelectedBodySegment2RouteFamilyPlanningOwner` and
  `RVVSelectedBodySegment2RouteFamilyProviderPlan` APIs in
  `RVVEmitCRoutePlanning`.
- Registered five exact segment2 planning owners:
  computed-mask segment2 load, computed-mask segment2 store,
  computed-mask segment2 update, plain segment2 deinterleave, and plain
  segment2 interleave.
- Rewired `getRVVSelectedBodySegment2MemoryRouteStatementPlan(...)` to consume
  the owner-built segment2 route-family provider plan before statement
  construction.
- Added focused C++ registry, exact-one classification, empty non-consumer, and
  missing-plan fail-closed tests in `RVVExtensionPluginTest.cpp`.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the durable
  segment2 route-family planning owner boundary.

## Validation Results

- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [x] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] generated-bundle dry-run for direct pre-realized
      `computed_masked_segment2_update_unit_load`
- [x] generated-bundle dry-run for direct pre-realized
      `computed_masked_segment2_store_unit_load`
- [x] direct pre-realized non-regression dry-run for segment2 load/store,
      segment2 deinterleave/interleave, compare/select, standalone reduction,
      conversion, MAcc, scalar-broadcast MAcc, computed-mask MAcc,
      runtime-scalar computed-mask MAcc, base memory, widening MAcc, and
      widening dot-reduction routes
- [x] selected-boundary non-regression dry-run for `masked_add` and
      `scalar_broadcast_add`
- [x] real `ssh rvv` direct pre-realized
      `computed_masked_segment2_update_unit_load`, counts `0,7,16,23,257`
- [x] bounded production/test touched-file authority scan
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2` (390/390)

## Validation Plan

1. Validate and start this Trellis task.
2. Build focused RVV plugin C++ test target.
3. Run `./build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run generated-bundle dry-runs for direct pre-realized
   `computed_masked_segment2_update_unit_load` and one adjacent migrated
   segment2 route.
5. Run focused direct route-entry generated-bundle non-regression for existing
   segment2 and adjacent active owners.
6. Run targeted authority scan over touched production/test files.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2` if focused checks
   are green and no blocker appears.

## Out Of Scope

- New operation support or broad Stage2 coverage expansion.
- New segment factor beyond segment2.
- New dtype/LMUL clone batches.
- Linalg/Vector/StableHLO frontend lowering.
- Source-front-door positive routes.
- Descriptor/direct-C/source-export route resurrection.
- Common EmitC/export RVV semantic inference.
- Dashboard/report-only/helper-only changes.
- Weakening existing owners for computed-mask segment2 update/store/load,
  plain segment2 deinterleave/interleave, masked elementwise, reduction,
  scalar-broadcast, conversion, runtime-scalar MAcc, compare/select, MAcc,
  contraction, or base memory.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/guides/index.md`
  - capability-first, plugin-locality, and compute-boundary guides
- Predecessor context read:
  - `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-selected-body-route-entry-family-registry-owner/prd.md`
  - `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-route-provider-family-owner-registry/prd.md`
  - `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-memory-route-family-owner-registry/prd.md`
  - `.trellis/workspace/codex/journal-16.md`
- Initial code surface:
  - `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Plugin/RVVExtensionPluginTest.cpp`

## Definition Of Done

The provider/planning boundary for active segment2 route-entry families is
explicit and registry-driven, real segment2 consumers use it in the production
route construction path, focused tests and validation pass, Trellis state is
truthful, and one coherent commit records the work.
