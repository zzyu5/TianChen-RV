# Stage2 RVV memory route-family ownership extraction

## Goal

Extract an explicit RVV plugin-local ownership boundary for the completed active
memory route families without changing route semantics or broadening coverage.
The bounded families are:

- runtime/vector computed-mask memory;
- computed-mask strided load/store;
- computed-mask indexed gather/scatter;
- computed-mask segment2 load/store;
- plain segment2 deinterleave/interleave.

This round should make family-owned planning, validation, description
application, provider predicates, metadata mirrors, and evidence expectations
easier to audit and reuse before any further Stage2 coverage slice is added.

## Direction Source

- Direction title: `Stage2 RVV memory route-family ownership extraction`.
- Module owner: RVV plugin-local memory route-family planning/provider
  ownership boundary for already completed computed-mask memory, indexed,
  strided, computed-mask segment2, and plain segment2 families.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `7d5950b7 rvv: share plain segment2 memory route family`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes/Codex Direction Brief before source edits.

## What I Already Know

- Long-term specs require the active RVV path to flow from selected
  `tcrv.exec` variant, through typed `tcrv_rvv` body/config/runtime facts, RVV
  plugin-owned legality/realization/route provider, provider-built
  `TCRVEmitCLowerableRoute`, common EmitC materialization, target artifact, and
  real `ssh rvv` evidence for runtime/correctness claims.
- Common EmitC/export may consume provider output and mirror metadata, but must
  not infer RVV semantics, dtype/config, mask/segment/index/stride behavior,
  route support, or acceptance state from route ids, helper strings, ABI names,
  manifests, artifact names, or status fields.
- Current headers already expose:
  - `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan`;
  - `RVVSelectedBodySegment2MemoryRouteFamilyPlan`;
  - optional analysis storage for both plans;
  - description mirror fields
    `computedMaskMemoryRouteFamilyPlanID` and
    `segment2MemoryRouteFamilyPlanID`.
- Current target metadata already mirrors
  `tcrv_rvv.computed_mask_memory_route_family_plan` and
  `tcrv_rvv.segment2_memory_route_family_plan`.
- The active ownership remains difficult to audit because derivation,
  validation, description application, generic route-description verification,
  provider-side predicates, and metadata mirror emission are concentrated in
  large `RVVEmitCRoutePlanning.cpp` / `RVVEmitCRouteProvider.cpp` sections.
- The previous archived task made plain segment2 routes consume
  `RVVSelectedBodySegment2MemoryRouteFamilyPlan` while preserving typed body
  authority, ABI order, field roles, target leaf/header facts, provider
  mirrors, generated-bundle behavior, and real `ssh rvv` evidence.
- Computed-mask memory and plain segment2 semantics are distinct. This task may
  share neutral mechanics only; it must not collapse mask semantics into the
  plain segment2 family or vice versa.

## Scope

1. Inventory only the completed active memory-family planner/provider code
   paths listed in this PRD and the task brief.
2. Introduce or repair explicit plugin-local module boundaries for:
   - memory family plan derivation;
   - family plan validation;
   - description application;
   - provider-side family predicates;
   - route-description validation for family-owned facts;
   - metadata mirror production for family-owned facts;
   - generated-bundle/evidence expectations that consume the same mirrors.
3. Keep active consumers on their existing production route semantics:
   - `runtime_scalar_cmp_masked_store`;
   - `runtime_scalar_cmp_masked_load_store`;
   - `computed_masked_unit_load_store`;
   - `computed_masked_strided_store`;
   - `computed_masked_strided_load_unit_store`;
   - `computed_masked_indexed_gather_load_unit_store`;
   - `computed_masked_indexed_scatter_store_unit_load`;
   - `computed_masked_segment2_load_unit_store`;
   - `computed_masked_segment2_store_unit_load`;
   - `segment2_deinterleave_unit_store`;
   - `segment2_interleave_unit_load`.
4. Preserve route names, selected-body semantics, ABI order,
   `RouteOperandBindingPlan` closure, target leaf/header facts, generated
   artifact behavior, and real runtime results.
5. Keep common EmitC/export neutral. Common code may serialize mirrors supplied
   after route construction; it must not own RVV family semantics.

## Requirements

1. Family-owned components must make it explicit which facts are owned by the
   computed-mask memory family and which are owned by the plain segment2 memory
   family.
2. Computed-mask memory family ownership must remain responsible for:
   - producer source and mask facts;
   - runtime AVL/VL and ABI order;
   - store-only, load-merge, strided, indexed, and segment2 facets;
   - inactive-lane and passthrough contracts;
   - indexed and strided memory layout facts;
   - computed-mask segment2 layout/count/tuple/field facts;
   - target leaf/header facts, C type mapping, provider-supported mirror, and
     route operand binding closure.
3. Plain segment2 family ownership must remain responsible for:
   - deinterleave versus interleave direction;
   - runtime AVL/VL and ABI order;
   - segment count exactly 2;
   - segment layout and tuple C type;
   - segment2 load/store leaves and field extract/tuple-create facts;
   - field roles/names and per-field source/destination memory forms;
   - target leaf/header facts, C type mapping, provider-supported mirror, and
     route operand binding closure.
4. Provider-side family predicates must require the relevant family plan before
   materializing any migrated route. A route must not be materialized only from
   route id, helper string, metadata mirror, artifact name, ABI name, or common
   exporter inference.
5. Existing computed-mask and plain segment2 route semantics must stay distinct.
   Neutral mechanics may be shared only when they do not own mask, stride,
   index, segment, field-role, ABI, dtype/config, or inactive-lane semantics.
6. The extracted boundary must move or clarify active production ownership. A
   round that only renames helpers, adds reports, or adjusts tests is not
   complete.
7. No new positive legacy `RVVI32M1`, `rvv-i32m1`, finite
   `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
   source-front-door, helper-string, route-id, artifact-name, or mirror-only
   route authority may be introduced.

## Acceptance Criteria

- [x] PRD and task context reference the RVV plugin, EmitC route, plugin
      interface, and testing specs plus directly relevant archived memory
      family tasks.
- [x] Completed active memory-family code paths are inventoried without a broad
      all-repo route rewrite.
- [x] Explicit plugin-local ownership components exist for memory-family plan
      derivation, validation, description application, provider predicates, and
      metadata mirrors.
- [x] `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan` consumers still cover
      runtime/vector unit memory, strided store/load, indexed gather/scatter,
      and computed-mask segment2 load/store semantics without regression.
- [x] `RVVSelectedBodySegment2MemoryRouteFamilyPlan` consumers still cover
      plain segment2 deinterleave/interleave semantics without regression.
- [x] Provider materialization requires the corresponding family plan for every
      migrated active memory-family route.
- [x] Route-description validation checks family-owned fields through the new
      boundary rather than through scattered one-off route branches where those
      branches are now covered by a family.
- [x] Metadata mirror production for family-owned memory facts is explicit and
      remains mirror-only.
- [x] Common EmitC/export gains no RVV semantic authority.
- [x] Focused lit/FileCheck coverage for representative computed-mask store or
      load-store, strided-load, indexed gather/scatter, computed-mask segment2,
      and plain segment2 routes still proves family plan mirrors, materialized
      operands, headers, binding closure, and field/index/stride facts where
      applicable.
- [x] Focused C++ plugin/provider/target artifact tests for touched components
      pass.
- [x] Generated-bundle dry-runs pass for representative pre-realized and
      explicit routes where fixtures exist.
- [x] Real `ssh rvv` smoke passes for at least one computed-mask memory route
      and one plain segment2 route with multiple runtime counts, unless the
      final diff is proven non-runtime-affecting and narrower evidence is
      justified.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive legacy or mirror-only route authority.
- [x] `git diff --check`, focused checks, `check-tianchenrv`, task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No new route coverage.
- No new dtype/LMUL clones.
- No new segment counts.
- No frontend/source-front-door positive routes.
- No all-route rewrite.
- No dashboards, readiness state machines, report-only work, or evidence-only
  packaging.
- No change to memory semantics, mask semantics, field roles, index/stride
  semantics, runtime `n`/AVL behavior, route ids, ABI names, or target artifact
  contracts except for mechanically preserving them through the clarified
  ownership boundary.
- No routing through legacy `RVVI32M1`, `rvv-i32m1`, finite positive
  `tcrv_rvv.i32_*`, descriptor, direct-C, source-front-door, helper-string,
  route-id, artifact-name, or mirror-only authority.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test`.
3. Run focused lit/FileCheck tests for representative memory family routes and
   directly touched fail-closed fixtures.
4. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if planning/provider
   helpers are changed.
5. Run generated-bundle dry-runs for representative active memory-family
   routes with counts `7,16,23`.
6. Run real `ssh rvv` generated-bundle smoke for at least one computed-mask
   memory route and one plain segment2 route unless the final diff is proven
   non-runtime-affecting.
7. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Completion Evidence

Production changes:

- Added shared planning-surface predicates:
  - `isRVVSelectedBodyComputedMaskMemoryRouteFamilyConsumer`;
  - `isRVVSelectedBodyPlainSegment2MemoryRouteFamilyConsumer`;
  - `isRVVSelectedBodyMemoryRouteFamilyConsumer`.
- Added `verifyRVVSelectedBodyMemoryRouteFamilyProviderPlans` so provider
  materialization is gated by planning-owned family plan availability for every
  active computed-mask memory or plain segment2 memory family consumer.
- Removed duplicated provider-local computed-mask memory and plain segment2
  family-route predicates from `RVVEmitCRouteProvider.cpp`; provider code now
  uses the planning-owned predicates.
- Factored segment2 memory-family metadata mirror production into
  `addRVVSelectedBodySegment2MemoryRouteFamilyMetadataMirrors`, covering both
  computed-mask segment2 routes and both plain segment2 routes without changing
  mirror keys or emitted values.
- Left active route names, selected-body semantics, runtime ABI order,
  `RouteOperandBindingPlan` closure, target leaf/header facts, generated
  artifact behavior, and common EmitC/export semantics unchanged.

Active routes still consuming the family boundaries:

- Computed-mask memory family:
  - `runtime_scalar_cmp_masked_store`;
  - `runtime_scalar_cmp_masked_load_store`;
  - `computed_masked_unit_load_store`;
  - `computed_masked_strided_store`;
  - `computed_masked_strided_load_unit_store`;
  - `computed_masked_indexed_gather_load_unit_store`;
  - `computed_masked_indexed_scatter_store_unit_load`;
  - `computed_masked_segment2_load_unit_store`;
  - `computed_masked_segment2_store_unit_load`.
- Plain segment2 memory family:
  - `segment2_deinterleave_unit_store`;
  - `segment2_interleave_unit_load`.

Checks and evidence:

- Focused build passed:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- C++ plugin smoke passed:
  `build/bin/tianchenrv-rvv-extension-plugin-test`.
- Focused lit passed 29/29 selected tests via:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter ...`
  from `build/test`, covering representative computed-mask unit memory,
  strided load/store, indexed gather/scatter, computed-mask segment2, plain
  segment2, and relevant dataflow fixtures.
- Generated-bundle dry-run passed for representative pre-realized routes with
  counts `7,16,23`:
  - `runtime_scalar_cmp_masked_store`;
  - `computed_masked_strided_load_unit_store`;
  - `computed_masked_indexed_gather_load_unit_store`;
  - `computed_masked_indexed_scatter_store_unit_load`;
  - `computed_masked_segment2_load_unit_store`;
  - `computed_masked_segment2_store_unit_load`;
  - `segment2_deinterleave_unit_store`;
  - `segment2_interleave_unit_load`.
- Real `ssh rvv` generated-bundle smoke passed for
  `runtime_scalar_cmp_masked_store` with counts `7,16,23`, reporting active
  lanes, inactive lane preservation, payload distinguishing lanes, source
  preservation, and tail preservation.
- Real `ssh rvv` generated-bundle smoke passed for
  `segment2_deinterleave_unit_store` with counts `7,16,23`, reporting
  field-order distinguishing lanes and tail preservation.
- Active-authority added-line scan over touched RVV planning/provider files
  found no new positive `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export, source-front-door,
  helper-string, artifact-name, mirror-only, or exact i32m1 intrinsic route
  authority. Full touched-file scan only hit an existing fail-closed
  `tcrv_rvv.i32_*` guard.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed 349/349.
- `clang-format` was not available in this environment; formatting was
  checked through diff review, compiler warnings, `git diff --check`, and the
  full test target above.

## Relevant Specs And Prior Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-plain-segment2-memory-route-family/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-computed-mask-segmented-memory-route-family/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-indexed-memory-route-family/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-strided-load-route-family-interface/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-computed-mask-memory-route-family-interface/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-computed-mask-memory-producer-source-family/prd.md`

## Initial Implementation Surface

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp` only if mirror expectations need
  mechanical adjustment.
- `scripts/rvv_generated_bundle_abi_e2e.py` only if generated-bundle mirror
  expectations need mechanical adjustment.
- Focused tests under `test/Target/RVV`, `test/Conversion/EmitC`,
  `test/Dialect/RVV`, and plugin C++ tests as directly relevant.

## Definition Of Done

The active memory movement route families have an explicit plugin-local
ownership boundary for derive/validate/apply/provider predicate/mirror
responsibilities, every migrated route remains executable with unchanged
semantics, common EmitC/export remains neutral, focused tests and generated
bundle evidence pass, real `ssh rvv` evidence is collected or explicitly
justified as unnecessary, task state is finished/archived, and one coherent
commit records the work.
