# Stage2 RVV computed-mask segmented-memory route-family interface

## Goal

Migrate the existing computed-mask segmented-memory consumers onto the shared
RVV plugin-local computed-mask memory route-family interface. The active
repository routes are:

- `computed_masked_segment2_load_unit_store`
- `computed_masked_segment2_store_unit_load`

These correspond to:

- `RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore`
- `RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad`
- `RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore`
- `RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store`

This is a migration of existing segmented load/store route authority into the
computed-mask memory family. It is not new frontend work, not a broad
all-memory rewrite, and not a new dtype/LMUL route matrix.

## Direction Source

- Direction title: `Stage2 RVV computed-mask segmented-memory route-family interface`.
- Module owner: RVV plugin-local computed-mask segmented-memory route-family
  support for the existing segmented load/store consumers.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `d1fbadee rvv: share computed-mask indexed memory route family`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes/Codex Direction Brief before source edits.

## Current Inventory

- The previous indexed-memory task migrated
  `computed_masked_indexed_gather_load_unit_store` and
  `computed_masked_indexed_scatter_store_unit_load` onto
  `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan`.
- The shared family currently records runtime-scalar, vector compare, store
  only, load-merge/store, strided, and indexed facets. It does not carry
  segment-specific family facts such as segment count, tuple type, masked
  segment load/store leaves, field roles, field names, or per-field source and
  destination memory forms.
- Current segmented route description still has duplicated one-off provider
  description population and validation for segmented facts after route
  analysis. This task moves active segmented computed-mask memory authority
  into the family plan/provider boundary and demotes or removes the duplicated
  one-off route authority where covered by that family.
- `computed_masked_segment2_load_unit_store` structurally uses vector compare
  lhs/rhs loads to produce a mask, loads old field0/field1 destination
  passthrough vectors, performs `tcrv_rvv.masked_segment2_load` from an
  interleaved segment2 source, and stores field0/field1 unit-stride outputs.
- `computed_masked_segment2_store_unit_load` structurally uses vector compare
  lhs/rhs loads to produce a mask, loads two unit-stride source field payloads,
  and performs `tcrv_rvv.masked_segment2_store` into one interleaved segment2
  destination, preserving inactive lanes by not writing compare-false lanes.
- Existing explicit and pre-realized target tests already cover these two
  routes; this task must repair them so they assert family-derived segmented
  facts and not merely route-specific segmented mirrors.

## Relevant Specs And Prior Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-computed-mask-memory-producer-source-family/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-strided-load-route-family-interface/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-indexed-memory-route-family/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-route-operand-abi-binding/prd.md`

## Scope

1. Extend or repair `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan` so it
   can represent the two active computed-mask segment2 routes.
2. Preserve existing already-migrated computed-mask memory consumers:
   - `RuntimeScalarComputedMaskStore`
   - `RuntimeScalarComputedMaskLoadStore`
   - `ComputedMaskUnitLoadStore`
   - `ComputedMaskStridedStore`
   - `ComputedMaskStridedLoadUnitStore`
   - `ComputedMaskIndexedGatherLoadUnitStore`
   - `ComputedMaskIndexedScatterStoreUnitLoad`
3. Make `ComputedMaskSegment2LoadUnitStore` consume the shared family for:
   - vector compare RHS producer source;
   - runtime AVL/VL and ABI order;
   - segment count/layout and tuple C type;
   - masked segment2 load leaf and tuple field extraction/create facts;
   - source interleaved segment memory form;
   - field0/field1 output roles and unit-store destination forms;
   - old field passthrough and inactive-lane contract;
   - provider-supported mirror, target leaf profile, headers, C type mapping,
     and route operand binding closure.
4. Make `ComputedMaskSegment2StoreUnitLoad` consume the shared family for:
   - vector compare RHS producer source;
   - runtime AVL/VL and ABI order;
   - segment count/layout and tuple C type;
   - masked segment2 store leaf and tuple create facts;
   - field0/field1 input roles and unit-stride source forms;
   - interleaved destination memory form;
   - inactive-lane no-write contract;
   - provider-supported mirror, target leaf profile, headers, C type mapping,
     and route operand binding closure.
5. Keep common EmitC/export neutral. Common materialization may consume
   provider output only; it must not infer segmented layout, field roles,
   mask producer source, load/store/merge behavior, dtype/config, runtime
   roles, or support state from route ids, helper strings, artifact names,
   manifests, mirrors, or C ABI strings alone.

## Requirements

1. The family plan must carry or validate segmented facets:
   - segment2 load/store consumer kind;
   - segment count exactly 2;
   - segment memory layout;
   - segment tuple C type;
   - segment load/store/field extraction or tuple-create leaves;
   - field0/field1 roles;
   - field0/field1 generated value names;
   - field0/field1 source or destination memory forms;
   - source and destination memory forms for the route as a whole.
2. The family plan must continue to carry existing common memory facts:
   - family plan id;
   - vector compare mask producer source;
   - load-merge/store versus store-only facet;
   - runtime AVL/VL plan and runtime ABI order;
   - provider-supported mirror, target leaf profile, required headers, and C
     type mapping;
   - SEW32 LMUL m1 vector/mask type facts;
   - setvl, compare, vector load, mask role/source/form, inactive-lane
     contract, passthrough layout, and `RouteOperandBindingPlan` closure.
3. `computed_masked_segment2_load_unit_store` must require compare lhs/rhs
   loads, old field passthrough loads, `masked_segment2_load`, two final
   stores, runtime `n/AVL`, SEW32 LMUL m1, segment count 2, field0/field1
   output roles, inactive passthrough preservation, and tail preservation.
4. `computed_masked_segment2_store_unit_load` must require compare lhs/rhs
   loads, field0/field1 source payload loads, `masked_segment2_store`,
   runtime `n/AVL`, SEW32 LMUL m1, segment count 2, field0/field1 input
   roles, inactive no-write/output preservation, and tail preservation.
5. Provider materialization must require the shared computed-mask memory
   family plan before emitting either segmented route.
6. Duplicated one-off segmented computed-mask route authority should be removed
   or demoted behind the family plan. Segment-specific facts may remain as
   fields in the shared family plan.

## Acceptance Criteria

- [ ] One shared RVV plugin-local computed-mask memory route-family plan is
      consumed by both active computed-mask segmented-memory routes.
- [ ] The plan exposes explicit segmented facets for segment layout/count,
      tuple type, segment load/store leaves, field roles, field names, and
      per-field source/destination forms.
- [ ] `computed_masked_segment2_load_unit_store` derives family-owned
      producer-source, runtime/control/header/mirror, segment layout,
      passthrough, field-role, memory-form, and binding-closure facts.
- [ ] `computed_masked_segment2_store_unit_load` derives family-owned
      producer-source, runtime/control/header/mirror, segment layout,
      payload-field, destination, memory-form, and binding-closure facts.
- [ ] Existing runtime-scalar, unit, strided, strided-load, and indexed
      computed-mask memory routes remain supported.
- [ ] Old duplicated one-off segmented computed-mask description/provider
      authority is removed or demoted so segmented routes do not bypass the
      shared family.
- [ ] Focused explicit and pre-realized target/header FileCheck coverage proves
      family-derived segment layout/count, field roles, field memory forms,
      materialized operands, mirrors, headers, and fail-closed diagnostics.
- [ ] Negative/fail-closed coverage covers missing segment field role, wrong
      segment count/layout, missing compare input, missing mask source, missing
      payload or passthrough/output role, missing runtime n/AVL, wrong memory
      form, stale route id, helper-string fallback, mirror-only authority, and
      common/export semantic inference where current test surfaces can express
      them.
- [ ] Generated-bundle dry-runs pass for explicit and pre-realized
      `computed_masked_segment2_load_unit_store` and
      `computed_masked_segment2_store_unit_load`, counts `7,16,23`.
- [ ] Real `ssh rvv` evidence passes for both migrated segmented routes with
      multiple counts if emitted provider materialization or target
      header/source output changes.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, route-id, helper-string, artifact-name, mirror-only,
      or common/export RVV semantic authority.
- [ ] Focused checks, `git diff --check`, `check-tianchenrv`, task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No new indexed or strided forms.
- No new select, accumulation, reduction, arithmetic, mask algebra, dtype,
  LMUL, frontend, Linalg, source-front-door, dashboard, or report-only work.
- No broad all-memory rewrite and no collapse of segmented semantics into
  indexed or strided semantics. Only truly common computed-mask memory-family
  mechanics may be shared.
- No migration through legacy `RVVI32M1`, `rvv-i32m1`, finite positive
  `tcrv_rvv.i32_*`, descriptor, direct-C, source-front-door, helper-string,
  route-id, artifact-name, or mirror-only route authority.
- No common EmitC/export ownership of segmented layout, field roles,
  load/store/merge behavior, compare, mask producer source, dtype/config,
  runtime roles, result roles, or acceptance state.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test`.
3. Run focused lit/FileCheck coverage for explicit and pre-realized
   computed-mask segment2 target/header fixtures plus directly relevant
   segmented dataflow and fail-closed fixtures.
4. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if route planning or
   provider helpers are changed.
5. Run generated-bundle dry-runs for explicit and pre-realized segment2 load
   and store routes with counts `7,16,23`.
6. Run real `ssh rvv` generated-bundle evidence for both migrated segmented
   routes if emitted source/header/provider materialization changes.
7. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Initial Implementation Surface

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` only if realization must
  surface missing segmented facts.
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` and
  `lib/Dialect/RVV/IR/RVVDialect.cpp` only if verifier diagnostics need a
  narrow repair.
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py` only if generated-bundle metadata
  expectations require update.
- Focused tests under `test/Target/RVV`, `test/Conversion/EmitC`,
  `test/Dialect/RVV`, and `test/Plugin` as directly relevant.

## Definition Of Done

Both active computed-mask segment2 memory production routes consume the shared
plugin-local computed-mask memory route-family interface for segmented facts,
typed body/config/runtime-derived route support, provider mirrors, and
operand-binding closure. Existing non-segmented computed-mask memory routes
remain supported; common EmitC/export remains neutral; focused tests,
generated-bundle evidence, authority scan, `check-tianchenrv`, task archive,
clean status, and one coherent commit are completed.
