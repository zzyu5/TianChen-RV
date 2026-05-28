# Stage2 RVV segment2 memory route-family artifact validator migration

## Goal

Migrate segment2 memory target artifact validation from
`RVVTargetSupportBundle.cpp` into the target-owned RVV route-family artifact
validator registry. The registry entry must own segment2 provider-fact checks
and candidate metadata mirror checks for both plain segment2 memory and
computed-mask segment2 memory routes, while the central support bundle remains
a neutral target artifact bridge.

## Direction Source

- Direction title: `Switch: Stage2 RVV segment2 memory route-family artifact
  validator migration`.
- Module owner: target-owned RVV route-family artifact validator registry entry
  for segment2 memory movement routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `e1dc26ec rvv: migrate conversion artifact validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes direction brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflows.

## What I Already Know

- Commit `e1dc26ec` completed the conversion dtype-policy artifact validator
  migration, left the worktree clean, and moved conversion-specific provider
  fact and mirror validation behind the route-family validator registry.
- `RVVTargetArtifactRouteFamilyValidation.cpp` currently registers
  `compare-select-mask`, `conversion-dtype-policy`,
  `elementwise-arithmetic`, `macc`, and `widening-dot-reduction` validators.
- `RVVTargetSupportBundle.cpp` still owns segment2 semantic provider-fact
  validation:
  - `isRVVSegment2RouteFamilyOperation`;
  - `isRVVComputedMaskSegment2RouteFamilyOperation`;
  - `isRVVPlainSegment2RouteFamilyOperation`;
  - `validateRVVSegment2RouteHeaders`;
  - `validateRVVSegment2RouteTypeMappings`;
  - `validateRVVSegment2RouteABIMappings`;
  - `validateRVVSegment2RouteStatementPlan`;
  - `validateRVVSegment2RoutePayloadFacts`.
- `RVVTargetSupportBundle.cpp` also still owns segment2-specific candidate
  mirror validation inside `validateRVVRouteMetadataMirrorsSelectedBody`.
- Existing segment2 artifact fixtures already cover at least one plain
  segment2 artifact consumer and one computed-mask segment2 artifact consumer:
  - `pre-realized-selected-body-artifact-segment2-interleave-unit-load.mlir`;
  - `pre-realized-selected-body-artifact-computed-masked-segment2-store.mlir`.
- The RVV plugin spec has a dedicated `Segment2 Target Export Consumer
  Contract` requiring target export to rebuild the provider route, validate
  provider facts, validate mirror-only metadata, distinguish plain and
  computed-mask segment2 plans, and fail closed on stale segment2 facts.
- This round must not change RVV provider lowering, selected-body realization,
  generated C semantics, intrinsic spelling, route IDs, runtime ABI, artifact
  names, or runtime/correctness/performance claims unless a real production
  bug is found and separately evidenced.

## Requirements

1. Register a production `segment2-memory` route-family validator in
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
2. Dispatch active plain and computed-mask segment2 target artifact candidates
   through the target route-family validator registry.
3. Move segment2 memory provider-fact validation out of
   `RVVTargetSupportBundle.cpp`.
4. Move segment2 memory candidate mirror validation out of
   `RVVTargetSupportBundle.cpp`.
5. Keep `RVVTargetSupportBundle.cpp` limited to neutral bridge
   responsibilities:
   - selected-body route rebuild;
   - generic candidate shape and selected-boundary/runtime checks;
   - descriptor, direct-C, source-export, and source-front-door residue
     rejection;
   - neutral materialized route verification and artifact mechanics;
   - metadata evidence listing;
   - registry dispatch.
6. The new validator must distinguish plain segment2 memory routes from
   computed-mask segment2 memory routes:
   - plain segment2 routes require `segment2MemoryRouteFamilyPlanID` and must
     reject computed-mask route-family support residue;
   - computed-mask segment2 routes require
     `computedMaskMemoryRouteFamilyPlanID` plus mask producer/role/source/form
     facts and must not require the plain segment2 family plan.
7. The new validator must consume rebuilt `TCRVEmitCLowerableRoute` plus
   `RVVSelectedBodyEmitCRouteDescription` facts to validate:
   - segment2 operation classification;
   - provider support mirror label as mirror-only evidence;
   - route operand binding plan and summary;
   - runtime AVL/VL control plan and runtime ABI order;
   - source/destination memory roles and forms;
   - segment count and segment memory layout;
   - mask producer, role, source, memory form, and compare/mask statement
     facts for computed-mask segment2 routes;
   - provider-derived headers;
   - VL/vector/mask C type mappings;
   - ABI mapping count, order, roles, C names, and value names;
   - pre-loop setvl and one runtime AVL/VL loop with selected typed RVV source
     provenance;
   - plain deinterleave segment-load/extract/store statements;
   - plain interleave load/load/tuple/segment-store statements;
   - computed-mask segment2 load/store/update statement facts, including
     update arithmetic where applicable.
8. The validator must validate candidate metadata only as mirrors of rebuilt
   provider facts:
   - `tcrv_rvv.provider_supported_mirror`;
   - `tcrv_rvv.route_operand_binding_plan`;
   - `tcrv_rvv.route_operand_binding_operands`;
   - `tcrv_rvv.segment2_memory_route_family_plan`;
   - `tcrv_rvv.computed_mask_memory_route_family_plan`;
   - `tcrv_rvv.memory_form`;
   - `tcrv_rvv.runtime_control_plan`;
   - `tcrv_rvv.runtime_abi_order`;
   - `tcrv_rvv.required_header_declarations`;
   - `tcrv_rvv.c_type_mapping`;
   - `tcrv_rvv.segment_memory_layout`;
   - `tcrv_rvv.source_memory_form`;
   - `tcrv_rvv.destination_memory_form`;
   - `tcrv_rvv.segment_count`;
   - computed-mask segment2 mask producer/role/source/form mirrors.
9. The validator must fail closed on missing segment2 route-family facts, stale
   non-segment2 route-family residue, missing or stale computed-mask facts,
   wrong runtime ABI order, wrong memory roles, stale provider-supported
   mirrors, wrong headers, wrong type mappings, wrong statement-plan facts,
   descriptor residue, source-front-door residue, route-id-derived authority,
   artifact-name-derived authority, and exact-intrinsic-as-authority.
10. Preserve existing conversion, elementwise, compare/select, MAcc,
    widening-dot, runtime-scalar splat-store, standalone
    reduction/accumulation, base-memory, and other segment2 target artifact
    behavior as non-regressions.
11. Do not introduce central ad hoc, name-derived, metadata-derived,
    descriptor-derived, ABI-string-derived, script-derived, artifact-name-
    derived, common-EmitC-derived, source-front-door-derived, route-id-derived,
    exact-intrinsic-derived, direct-route-entry-only,
    pre-realized-fixture-only, or legacy-i32-derived route authority.

## Acceptance Criteria

- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` registers and dispatches a
      unique `segment2-memory` validator for active plain and computed-mask
      segment2 route descriptions.
- [x] `RVVTargetSupportBundle.cpp` no longer contains
      `validateRVVSegment2Route*` semantic helper ownership or a
      segment2-specific candidate mirror branch.
- [x] The new validator checks provider-derived route id agreement, headers,
      type mappings, ABI order/mapping, route operand binding mirrors,
      runtime AVL/VL plan, segment memory facts, mask facts where present, and
      statement-plan callees using rebuilt provider/body facts.
- [x] The new validator checks candidate mirrors for both one plain segment2
      case and one computed-mask segment2 case, including stale provider
      support, stale route-family plan, wrong ABI order, wrong memory/mask
      evidence, and stale non-segment2 route-family residue.
- [x] Existing plain and computed-mask segment2 artifact fixtures still pass
      without changing generated C/runtime semantics.
- [x] Existing conversion, elementwise, compare/select, MAcc, widening-dot,
      runtime-scalar splat-store, standalone reduction/accumulation, and
      base-memory route-family behavior remains covered and non-regressed.
- [x] `git diff --check` passes.
- [x] Focused build/test targets for `TianChenRVRVVTarget`, `tcrv-opt`,
      `tcrv-translate`, target artifact export tests, RVV plugin tests, and
      directly related segment2 lit tests pass.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] Bounded touched-file authority scan confirms no executable artifact
      claim depends on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [x] Trellis task status, journal, archive state, and commit state are
      truthful at the end of the round.

## Out Of Scope

- Adding new segment2 route coverage or new segment2 selected-body
  realization.
- Adding standalone reduction, runtime scalar splat-store, dtype/LMUL clone
  batches, high-level Linalg/frontend lowering, one-intrinsic wrapper dialects,
  dashboard/report work, or broad route-family registry rewrites.
- Changing RVV provider lowering, selected-body realization, runtime ABI,
  generated C semantics, intrinsic spelling, route IDs, source-front-door
  behavior, artifact naming, dispatch/fallback behavior, or common EmitC
  materialization unless this migration exposes a real production bug.
- Claiming new runtime/correctness/performance evidence without real `ssh rvv`
  execution.

## Technical Approach

1. Keep this Trellis task current and bounded to the supplied Hermes direction.
2. Move segment2 family classifiers and provider-fact validators from
   `RVVTargetSupportBundle.cpp` into
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
3. Add a `segment2-memory` registry entry and consumer predicate next to the
   existing migrated validators.
4. Remove the central bridge call to `validateRVVSegment2RoutePayloadFacts` and
   rely on registry provider-fact validation.
5. Remove the segment2-specific branch from
   `validateRVVRouteMetadataMirrorsSelectedBody` and rely on registry
   candidate-mirror validation.
6. Strengthen focused segment2 lit tests with stale provider, stale
   route-family, stale unrelated family residue, stale ABI/memory, and
   stale mask evidence checks as needed.
7. Run focused build/lit/C++ checks, `git diff --check`, bounded authority
   scan, and `check-tianchenrv` if feasible.
8. Finish/archive the task and create one coherent commit if acceptance is met.

## Validation Plan

1. Validate/start this Trellis task after PRD and context files are written.
2. Build focused targets:
   `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
3. Run focused C++ tests:
   `build/bin/tianchenrv-target-artifact-export-test` and
   `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run focused lit tests for directly related segment2 artifact consumers:
   `pre-realized-selected-body-artifact-segment2-interleave-unit-load.mlir`,
   `pre-realized-selected-body-artifact-segment2-deinterleave-unit-store.mlir`,
   `pre-realized-selected-body-artifact-computed-masked-segment2-load.mlir`,
   `pre-realized-selected-body-artifact-computed-masked-segment2-store.mlir`,
   and `pre-realized-selected-body-artifact-computed-masked-segment2-update.mlir`.
5. Run `git diff --check`.
6. Run a bounded authority scan over touched production and test files.
7. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Required specs and prior context:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- archived task
  `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-conversion-dtype-policy-artifact-validator-migration/prd.md`

Primary production files:

- `include/TianChenRV/Target/RVV/RVVTargetArtifactRouteFamilyValidation.h`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`

Primary evidence consumers:

- `test/Target/RVV/pre-realized-selected-body-artifact-segment2-interleave-unit-load.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-segment2-deinterleave-unit-store.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-load.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-store.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-update.mlir`
