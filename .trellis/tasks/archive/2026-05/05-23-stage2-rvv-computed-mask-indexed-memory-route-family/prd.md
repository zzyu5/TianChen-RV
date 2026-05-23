# Stage2 RVV computed-mask indexed-memory route-family interface

## Goal

Migrate the existing computed-mask indexed memory consumers onto the shared
RVV plugin-local computed-mask memory route-family interface. The active
repository routes are:

- `computed_masked_indexed_gather_load_unit_store`
- `computed_masked_indexed_scatter_store_unit_load`

This task must keep the selected `tcrv_rvv` body as authority for compare,
mask, index vector, payload/passthrough, output role, runtime AVL/VL, dtype,
and memory form. The shared computed-mask memory family should own the
indexed-memory route support facts that are currently derived through
one-off indexed fallback branches.

## Direction Source

- Direction title: `Stage2 RVV computed-mask indexed-memory route-family interface`.
- Module owner: RVV plugin-local computed-mask indexed-memory route-family
  support for existing indexed load/store memory consumers.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `9123e536 rvv: add computed-mask strided-load family route`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes/Codex Direction Brief before source edits.

## Current Inventory

- The previous completed task migrated
  `computed_masked_strided_load_unit_store` onto
  `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan`.
- The shared computed-mask memory family currently covers:
  - `runtime_scalar_cmp_masked_store`
  - `runtime_scalar_cmp_masked_load_store`
  - `computed_masked_unit_load_store`
  - `computed_masked_strided_store`
  - `computed_masked_strided_load_unit_store`
- The active computed-mask indexed routes already exist and are supported by
  typed-body planning/provider logic:
  - explicit target tests:
    `test/Target/RVV/explicit-selected-body-artifact-computed-masked-indexed-gather-load.mlir`
    and
    `test/Target/RVV/explicit-selected-body-artifact-computed-masked-indexed-scatter-store.mlir`
  - pre-realized tests:
    `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-indexed-gather-load.mlir`
    and
    `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-indexed-scatter-store.mlir`
- The current family membership predicates in planning/provider exclude the
  indexed routes, so their index-vector source, indexed memory layout,
  masked-indexed load/store leaf, index EEW, offset unit, index source, and
  indexed source/destination memory-form validation are still owned by
  route-specific masked/indexed fallback branches.
- Indexed gather semantics are load-merge/store:
  compare lhs/rhs loads produce a mask, `index_load` materializes the index
  vector, old destination is loaded as passthrough, `masked_indexed_load`
  reads source active lanes, and final unit store writes the loaded vector.
- Indexed scatter semantics are store-only/no-passthrough:
  compare lhs/rhs loads produce a mask, source payload is unit-loaded,
  `index_load` materializes the index vector, and `masked_indexed_store`
  writes active lanes to indexed destination slots while false lanes preserve
  the output buffer.

## Scope

1. Extend or repair `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan` so it
   can represent the two active computed-mask indexed memory routes.
2. Keep the already migrated runtime-scalar, unit, strided-store, and
   strided-load computed-mask memory routes supported through the same family.
3. Make the indexed gather/scatter routes consume the family for:
   - vector compare producer source;
   - runtime AVL/VL and ABI order;
   - index-vector type/source, index load leaf, byte-offset scaling leaf,
     index EEW, offset unit, and scatter uniqueness;
   - masked indexed load/store leaf and source/destination memory-form facts;
   - inactive-lane and passthrough/no-passthrough contracts;
   - target leaf profile, provider-supported mirror, headers, C type mapping,
     and route operand binding closure.
4. Demote duplicated one-off indexed authority once the family owns the
   equivalent route-support facts.
5. Keep common EmitC/export neutral. The common path may consume provider
   output only; it must not infer indexed memory semantics from route ids,
   helper strings, artifact names, mirrors, descriptors, source-front-door
   metadata, or C ABI names.

## Requirements

1. Family membership must include both indexed operations without pulling in
   segmented memory.
2. The family plan must distinguish:
   - runtime-scalar producer versus vector compare producer;
   - store-only versus load-merge/store;
   - indexed gather versus indexed scatter;
   - indexed load source form versus indexed store destination form.
3. The gather route must require explicit typed structure: compare lhs/rhs
   loads, compare-produced mask, `index_load`, old output passthrough load,
   `tcrv_rvv.masked_indexed_load`, final unit store, and runtime `n/AVL`.
4. The scatter route must require explicit typed structure: compare lhs/rhs
   loads, compare-produced mask, source payload load, `index_load`,
   `tcrv_rvv.masked_indexed_store`, destination output role, and runtime
   `n/AVL`.
5. The family plan must validate index facts structurally from the selected
   body/config/runtime facts: index vector type, index EEW 32, offset unit
   `element`, index source `runtime_abi:index`, scatter uniqueness `unique`,
   source/destination memory forms, and route operand binding closure.
6. The provider must require the shared computed-mask memory family plan before
   materializing indexed gather/scatter routes.
7. Target/header mirrors must label family outputs explicitly, including
   route-family plan id, mask producer source, target leaf profile,
   provider-supported mirror, source/destination memory forms, indexed memory
   layout, index source, index EEW, offset unit, and indexed data or
   destination memory form.
8. Existing supported routes outside indexed memory must not regress.

## Acceptance Criteria

- [x] `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan` or an equivalent
      plugin-local interface carries indexed-memory facets for gather and
      scatter.
- [x] `ComputedMaskIndexedGatherLoadUnitStore` and
      `ComputedMaskIndexedScatterStoreUnitLoad` are included in shared
      computed-mask memory family derivation and validation.
- [x] Provider construction refuses those indexed routes without the shared
      computed-mask memory family plan.
- [x] Explicit and pre-realized target/header FileCheck coverage proves
      family-derived plan id, mask producer source, target leaf profile,
      provider-supported mirror, C type mapping, indexed memory layout,
      index source/EEW/offset, indexed data/destination forms, binding
      closure, and materialized operands.
- [x] Negative coverage covers missing index vector, missing compare input,
      missing mask source, missing payload or passthrough/output role,
      missing `n/AVL`, wrong memory form, wrong producer source, bad
      dtype/config, stale route id, helper-string fallback, mirror-only
      authority, and common/export semantic inference where current test
      surfaces can express them.
- [x] Generated-bundle dry-runs pass for migrated explicit and pre-realized
      indexed gather/scatter routes if the script exposes both forms.
- [x] Real `ssh rvv` evidence passes for the migrated indexed route family
      with multiple runtime counts and index patterns if runtime/correctness
      evidence is claimed in this round.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, route-id, helper-string, artifact-name, mirror-only,
      or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, task archive,
      clean final git status, and one coherent commit complete if this task
      finishes.

## Non-Goals

- No segmented memory migration.
- No new strided forms, select routes, accumulation routes, reductions, mask
  algebra, dtype/LMUL clones, frontend lowering, source-front-door positive
  routes, dashboards, report-only work, helper-only cleanup, or broad
  all-memory rewrite.
- No collapse of indexed gather/scatter semantics into strided or segmented
  memory semantics.
- No routing through legacy `RVVI32M1`, `rvv-i32m1`, finite positive
  `tcrv_rvv.i32_*`, descriptor, direct-C, source-front-door,
  helper-string, route-id, artifact-name, or mirror-only authority.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-extension-plugin-test`, and target artifact/export tests if
   touched.
3. Run focused FileCheck coverage for explicit and pre-realized computed-mask
   indexed gather/scatter target/header fixtures plus directly relevant
   negative fixtures.
4. Run focused generated-bundle dry-runs for indexed gather/scatter explicit
   and pre-realized routes if supported by `scripts/rvv_generated_bundle_abi_e2e.py`.
5. Run real `ssh rvv` generated-bundle evidence for the migrated indexed
   route family if executable correctness is claimed.
6. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Initial Implementation Surface

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- focused target/dialect tests for computed-mask indexed memory routes
- `scripts/rvv_generated_bundle_abi_e2e.py` only if generated-bundle mirror
  expectations require update

## Definition Of Done

The active computed-mask indexed gather/scatter production routes consume the
producer-source-aware computed-mask memory family for route support, indexed
memory facts, provider mirrors, headers, binding closure, and emitted artifact
evidence. Duplicated one-off indexed route authority is removed or demoted
where the family now owns it; segmented memory remains untouched; focused
tests, generated-bundle evidence where available, authority scan,
`check-tianchenrv`, task archive, clean status, and one coherent commit finish
the round.

## Implementation Results

- Extended `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan` with explicit
  indexed gather/scatter facets: indexed membership booleans, index vector
  type/C type, index load and index-scale leaves, indexed-store leaf, index
  EEW, offset unit, index source, scatter uniqueness, indexed data form, and
  indexed destination form.
- Added `ComputedMaskIndexedGatherLoadUnitStore` and
  `ComputedMaskIndexedScatterStoreUnitLoad` to the shared computed-mask memory
  family membership, load-merge/store-only classification, runtime ABI order,
  memory-form mapping, target leaf profile, provider-supported mirror, header
  declarations, C type mapping, memory layout, inactive-lane contract,
  passthrough/no-passthrough layout, and source/destination memory-form
  mapping.
- Made family derivation require indexed gather to carry `index_load`,
  `masked_indexed_load`, old-destination passthrough, final store, explicit
  compare-produced mask, `runtime_abi:index`, and runtime `n/AVL`.
- Made family derivation require indexed scatter to carry source load,
  `index_load`, `masked_indexed_store`, explicit compare-produced mask,
  `runtime_abi:index`, destination output role, and runtime `n/AVL`.
- Made the provider family predicate include the two indexed routes, so
  provider materialization requires the shared computed-mask memory plan before
  constructing either active indexed route.
- Updated explicit and pre-realized indexed gather/scatter target/header
  coverage to prove `computed_mask_memory_route_family_plan`,
  `computed_mask_memory_mask_producer_source`, target leaf profile,
  provider-supported mirror, header declarations, C type mapping, route
  operand binding closure, materialized index source, indexed memory layout,
  and source/destination indexed memory forms.
- Updated generated-bundle evidence tooling metadata expectations for the new
  indexed family facts.
- Left segmented memory intentionally unconverted; it remains outside this
  task's bounded scope.

## Validation Results

- [x] Trellis task context validation:
      `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-23-stage2-rvv-computed-mask-indexed-memory-route-family`.
- [x] Focused build:
      `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- [x] Focused explicit/pre-realized target/header pipelines for:
      `explicit-selected-body-artifact-computed-masked-indexed-gather-load.mlir`,
      `pre-realized-selected-body-artifact-computed-masked-indexed-gather-load.mlir`,
      `explicit-selected-body-artifact-computed-masked-indexed-scatter-store.mlir`,
      and
      `pre-realized-selected-body-artifact-computed-masked-indexed-scatter-store.mlir`.
- [x] Focused verifier diagnostics for existing indexed-memory negative
      surfaces:
      `masked-indexed-load-dataflow.mlir`,
      `computed-mask-indexed-scatter-store-dataflow.mlir`,
      `indexed-gather-memory-dataflow.mlir`, and
      `indexed-scatter-memory-dataflow.mlir`.
- [x] C++ tests:
      `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test`.
- [x] Script checks:
      `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [x] Generated-bundle dry-runs passed for explicit and pre-realized
      `computed_masked_indexed_gather_load_unit_store` and
      `computed_masked_indexed_scatter_store_unit_load`, counts `7,16,23`:
      `artifacts/tmp/stage2_cm_indexed_family/explicit` and
      `artifacts/tmp/stage2_cm_indexed_family/pre_realized`.
- [x] Real `ssh rvv` evidence passed for explicit and pre-realized
      `computed_masked_indexed_gather_load_unit_store` and
      `computed_masked_indexed_scatter_store_unit_load`, counts `7,16,23`.
      Both runs reported active-lane checks, inactive preservation,
      non-contiguous index lanes, source preservation, tail preservation, and
      `PASS` for both indexed routes:
      `artifacts/tmp/stage2_cm_indexed_family/explicit_ssh` and
      `artifacts/tmp/stage2_cm_indexed_family/pre_realized_ssh`.
- [x] Active-authority scan over touched RVV include/lib/script/test paths
      found no newly added positive legacy i32/RVVI32M1/source-front-door/
      descriptor/common-export authority. Existing matches are negative
      self-test or fail-closed guard strings.
- [x] `git diff --check`.
- [x] `cmake --build build --target check-tianchenrv -j2` passed 349/349.
- [x] `clang-format` was attempted but is not installed in this environment;
      the changed C++ still built and passed `git diff --check`.

## Spec Update Decision

No `.trellis/spec/**` change is required. This round applied the existing RVV
plugin-owned route-family, selected typed-body authority, route operand binding
closure, common EmitC neutrality, and real RVV evidence rules to the bounded
computed-mask indexed gather/scatter route family; it did not introduce a new
durable architecture rule.
