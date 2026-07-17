# Stage2 RVV segment2 memory route-control provider-plan integration

## Goal

Make the existing production-active RVV segment2 memory statement-plan path
consume the shared RVV route-control provider plan before statement/provider
construction. Already-supported plain segment2 and computed-mask segment2
memory routes must validate runtime AVL/VL, typed config, selected capability,
SEW/LMUL, tail policy, mask policy, runtime ABI order, materialization,
memory-form, segment direction, computed-mask segment facts, and operand-binding
ownership through the RVV-owned route-control boundary.

## Direction Source

- Direction title: `Stage2 RVV segment2 memory route-control provider-plan integration`.
- Module owner: RVV plugin-local route-control provider-plan consumption by
  the existing segment2 memory route-family/provider path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `fc5fc3bd rvv: route-control computed-mask memory plans`.
- `.trellis/.current-task` was absent, so this task was created from the
  provided Direction Brief before source edits.

## Current Repository Facts

- The required RVV authority chain remains selected `tcrv.exec` envelope ->
  typed or realized low-level `tcrv_rvv` body -> RVV plugin-owned legality,
  family planning, selected-body realization, route-control provider plan, and
  provider-built `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact
  mirrors.
- `RVVSelectedBodyRouteControlProviderPlan` already validates typed config,
  selected capability, runtime AVL/VL control, SEW/LMUL, tail/mask policy,
  runtime ABI order, and mirror consistency for migrated families including
  ordinary elementwise, scalar-broadcast elementwise, base memory, standalone
  reduction, scalar-broadcast MAcc, compare/select, computed-mask select, and
  non-segment computed-mask memory.
- `RVVSelectedBodySegment2MemoryRouteFamilyPlan` already exists for plain
  segment2 deinterleave/unit-store and interleave/unit-load, and carries
  runtime control, memory-form, segment direction, runtime ABI, target leaf
  profile, provider mirror, type/header/intrinsic, field role, and segment
  layout facts.
- Computed-mask segment2 load/store routes are represented through
  `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan` with segment2 load/store
  booleans, mask producer facts, memory-form facts, runtime control, runtime
  ABI, type/header/intrinsic, field role, and segment layout facts.
- `getRVVSelectedBodySegment2MemoryRouteStatementPlan(...)` already owns the
  segment2 statement sequence, but it currently validates family/materialization
  and memory operand-binding facts directly without first requiring the
  route-control provider plan.
- The previous computed-mask memory route-control task intentionally excluded
  computed-mask segment2 routes from `controlsComputedMaskMemory`; this task is
  the segment2 owner that should now consume route-control for both plain and
  computed-mask segment2 statement planning.

## Scope

1. Add a segment2 route-control consumer marker or equivalent structural flag
   to `RVVSelectedBodyRouteControlProviderPlan`.
2. Extend route-control consumer detection to include the existing
   production-active segment2 memory routes:
   `Segment2DeinterleaveUnitStore`, `Segment2InterleaveUnitLoad`,
   `ComputedMaskSegment2LoadUnitStore`, and
   `ComputedMaskSegment2StoreUnitLoad`.
3. In `getRVVSelectedBodyRouteControlProviderPlan(...)`, validate the
   appropriate same-analysis family/materialization plan for plain segment2 or
   computed-mask segment2, validate segment direction, memory-form, mask
   producer where applicable, and attach the owning runtime control plan.
4. In `getRVVSelectedBodySegment2MemoryRouteStatementPlan(...)`, require the
   segment2 route-control provider plan before building statement steps.
5. Update focused C++ tests for positive route-control consumption and targeted
   fail-closed diagnostics for segment2 route-control dependencies.
6. Update the RVV plugin spec only if the implementation makes segment2 memory
   a durable route-control consumer in the long-term contract.

## Requirements

1. Plain segment2 memory statement planning must require a route-control plan
   whose runtime control pointer is the same runtime-control plan owned by the
   verified plain segment2 family plan.
2. Computed-mask segment2 memory statement planning must require a
   route-control plan whose runtime control pointer is the same runtime-control
   plan owned by the verified computed-mask memory family plan.
3. Route-control construction must fail closed when the segment2 family plan is
   missing, stale, from another selected route analysis, or classified with the
   wrong segment direction, memory form, mask producer, segment2 load/store flag,
   or operation.
4. Route-control construction must fail closed when typed config, selected
   target capability, runtime AVL role/source, SEW/LMUL, tail policy, mask
   policy, runtime ABI order, runtime VL contract, setvl/with_vl names, loop
   facts, or mirror fields disagree with the validated typed body/config/runtime
   facts.
5. Segment2 statement planning must still fail closed for missing/stale memory
   operand-binding facts, missing runtime `n`, missing field/source/destination
   ABI roles, stale direction markers, stale memory-form markers, missing
   materialization leaves, or wrong source operation provenance before common
   EmitC materialization.
6. Common EmitC, target export, scripts, route ids, artifact names, ABI strings,
   metadata, manifests, descriptors, and tests may only mirror provider-built
   facts after route construction; they must not become AVL/VL, mask, policy,
   memory-form, segment direction, dtype, or compute authority.
7. Do not add new segment2 operation kinds, new dtype/LMUL clone batches,
   frontend lowering, conversion, contraction, accumulation, reduction,
   source-front-door positive routes, dashboards, broad smoke matrices, or
   evidence-only fixture copying.

## Acceptance Criteria

- [x] `RVVSelectedBodyRouteControlProviderPlan` exposes a segment2 memory
      consumer flag or equivalent structural marker.
- [x] Route-control consumer detection includes only the four existing
      production-active segment2 memory routes listed in Scope.
- [x] `getRVVSelectedBodyRouteControlProviderPlan(...)` validates same-analysis
      segment2 family/materialization facts, runtime AVL/VL control, typed
      config, selected capability, tail/mask policy, runtime ABI order,
      segment direction, memory-form, and computed-mask segment facts where
      applicable.
- [x] `getRVVSelectedBodySegment2MemoryRouteStatementPlan(...)` consumes the
      route-control provider plan before constructing segment2 setvl/load/
      compare/tuple/segment/store statements.
- [x] Positive C++ tests prove route-control consumption and provider/statement
      attachment for plain segment2 deinterleave, plain segment2 interleave,
      computed-mask segment2 load, and computed-mask segment2 store.
- [x] Negative C++ tests fail closed for representative stale or missing
      dependencies: missing segment2 family/materialization plan, missing
      computed-mask segment2 family/materialization plan, stale same-analysis
      ownership, wrong runtime AVL role, policy mismatch, unsupported selected
      capability/config, runtime ABI mirror mismatch, stale direction marker,
      stale memory-form marker, stale computed-mask segment facts, and stale
      operand binding.
- [x] Existing non-segment computed-mask memory statement-plan coverage still
      proves segment2 routes stay outside the non-segment computed-mask memory
      statement-plan boundary.
- [x] If emitted mirrors or target artifacts change, focused FileCheck or
      generated-header evidence covers explicit mirror-only labels. If emitted
      code and ABI stay unchanged, final report states why historical `ssh rvv`
      evidence remains sufficient.
- [x] Bounded scans over touched RVV planning/provider/test/spec/target/script
      files find no new name-, route-id-, metadata-, descriptor-, ABI-string-,
      script-, artifact-, common-EmitC-, source-front-door-, or legacy-i32-
      derived AVL/VL, mask, memory-form, segment direction, policy, dtype, or
      compute authority.
- [x] `git diff --check` passes.
- [x] Focused RVV plugin tests and `check-tianchenrv` pass, or an exact blocker
      is recorded.
- [x] Task status, journal/archive, and one coherent commit complete if this
      task finishes.

## Out Of Scope

- New segment2 operation coverage or new segment memory forms.
- Moving computation semantics into `tcrv.exec`, common EmitC, target export,
  scripts, metadata, descriptors, or route ids.
- Source-front-door positive routes, direct-C/source-export paths, legacy
  i32 route authority, scalar/IME/offload/future plugin work, broad dashboards,
  performance claims, or unrelated cleanup.
- Runtime/correctness/performance claims unless emitted target code or ABI
  changes and real `ssh rvv` evidence is collected.

## Technical Approach

1. Start and validate the Trellis task.
2. Add segment2 route-control consumer detection and plan flag.
3. Extend `getRVVSelectedBodyRouteControlProviderPlan(...)` with a segment2
   branch that chooses the plain segment2 family plan or computed-mask memory
   family plan, validates same-analysis ownership and family-specific
   classification, then reuses the existing typed config/runtime control/
   selected capability validation tail.
4. Require the segment2 statement-plan boundary to acquire the route-control
   plan and confirm the segment2 flag and runtime-control pointer before
   emitting any statement steps.
5. Update existing C++ provider tests rather than adding broad fixture copies.
6. Update `rvv-plugin.md` route-control and segment2 memory sections to record
   segment2 as a migrated consumer if production code and tests establish it.
7. Run focused build/tests, bounded authority scans, `git diff --check`, and
   `check-tianchenrv`.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-segment2-route-control`
2. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
3. `./build/bin/tianchenrv-rvv-extension-plugin-test`
4. Focused lit/FileCheck for segment2 selected-body artifacts if target mirrors
   or emitted artifacts change.
5. Generated-bundle dry-run only if emitted output, ABI evidence, or artifact
   mirrors change.
6. Bounded authority scan over touched RVV planning/provider/test/spec files.
7. `git diff --check`
8. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous task context read:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-runtime-vl-policy-provider-plan/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-computed-mask-memory-route-control/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-base-memory-movement-route-family-boundary/prd.md`,
  and
  `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-segment2-memory-statement-plan-ownership/prd.md`.
- Initial code surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Implementation Result

- Added `controlsSegment2Memory` to
  `RVVSelectedBodyRouteControlProviderPlan`.
- Added route-control consumer detection for exactly the four existing
  segment2 memory statement-plan routes:
  `segment2_deinterleave_unit_store`, `segment2_interleave_unit_load`,
  `computed_masked_segment2_load_unit_store`, and
  `computed_masked_segment2_store_unit_load`.
- Extended `getRVVSelectedBodyRouteControlProviderPlan(...)` so plain segment2
  routes require the verified plain segment2 family/materialization plan from
  the same selected route analysis, validate segment direction/memory-form
  facts, and attach the plain segment2 runtime-control plan.
- Extended `getRVVSelectedBodyRouteControlProviderPlan(...)` so computed-mask
  segment2 routes require the verified computed-mask memory
  family/materialization plan from the same selected route analysis, validate
  vector-compare mask-producer facts, segment2 load/store direction,
  memory-form facts, and attach the computed-mask memory runtime-control plan.
- Required `getRVVSelectedBodySegment2MemoryRouteStatementPlan(...)` to consume
  the segment2 route-control provider plan before constructing any segment2
  setvl/load/compare/tuple/segment/store statements.
- Tightened segment2 statement planning to reject missing or stale
  same-analysis memory operand-binding facts before common EmitC.
- Extended focused C++ provider tests for positive route-control consumption
  and fail-closed diagnostics covering missing family/materialization plans,
  stale direction, stale memory form, stale policy, stale selected capability,
  stale runtime ABI mirror, stale runtime AVL role, stale same-analysis
  materialization, and stale operand binding.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to record segment2
  memory as a route-control consumer and to require the route-control plan in
  segment2 statement-plan ownership.
- No emitted target statement order, runtime ABI, generated artifact format, or
  runtime behavior changed. No new `ssh rvv` claim was made; historical runtime
  evidence remains sufficient for unchanged executable output.

## Validation Result

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-segment2-route-control`
- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [x] `./build/bin/tianchenrv-rvv-extension-plugin-test`: RVV extension plugin
      smoke test passed.
- [x] `git diff --check`
- [x] Bounded added-line authority scan over touched planning/provider/test/spec
      files. Matches were limited to RVV plugin spec mirror-only prohibition
      text, normal `description` field access, and the negative `metadata_n`
      stale ABI mirror test; no source-front-door, descriptor, common-EmitC,
      artifact/script, or legacy-i32 authority was introduced.
- [x] `cmake --build build --target check-tianchenrv -j2`: 379/379 tests passed.
- [x] Focused lit/FileCheck and generated-bundle dry-run were not rerun because
      this task did not change emitted target code, ABI order, target mirrors,
      generated artifacts, or runtime/correctness claims.
