# Stage2 RVV computed-mask memory route-family interface

## Goal

Introduce one bounded RVV plugin-local computed-mask memory route-family
interface for the existing runtime-scalar computed-mask contiguous memory
consumers:

- `runtime_scalar_cmp_masked_store`
- `runtime_scalar_cmp_masked_load_store`

This is a migration of existing production paths, not new operation coverage.
Route support, closure roles, materialized operands, mirror/header fields,
target leaf selection, memory-form facts, inactive-lane behavior, and
fail-closed diagnostics must derive from one shared RVV planning/provider
boundary while preserving typed `tcrv_rvv` body authority and common
EmitC/export neutrality.

## Direction Source

- Direction title: `Stage2 RVV plugin-owned computed-mask memory route-family interface`.
- Module owner: RVV plugin-local route planning/provider interface for
  runtime-scalar computed-mask memory consumers.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `154ada5d rvv: share computed-mask select route family`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes Direction Brief before source edits.

## Current Inventory

Inventory is from current repo state and directly relevant archived tasks.

- `runtime_scalar_cmp_masked_store` is complete and production-supported for a
  runtime scalar threshold compare feeding `tcrv_rvv.masked_store`. It binds
  `lhs`, `rhs_scalar`, `src`, `dst`, and `n`, realizes
  `load lhs -> splat rhs_scalar -> compare sle -> load src -> masked_store`,
  closure-gates materialized uses/mirrors, and has generated-bundle plus real
  `ssh rvv` evidence in its archived task.
- `runtime_scalar_cmp_masked_load_store` is complete and production-supported
  for a runtime scalar threshold compare feeding `tcrv_rvv.masked_load` with
  old-destination passthrough and a following store. It binds `lhs`,
  `rhs_scalar`, `src`, `dst`, and `n`, realizes
  `load lhs -> splat rhs_scalar -> compare sle -> load old dst ->
  masked_load src/mask/passthrough -> store`, closure-gates
  materialized uses/mirrors, and has generated-bundle plus real `ssh rvv`
  evidence in its archived task.
- `154ada5d` introduced a shared computed-mask select route-family interface
  for the runtime-scalar select routes. That is the local pattern for this
  round: route-specific facts remain data in a shared family plan, not
  independent parallel route architectures.
- Current planning already has
  `RVVSelectedBodyRuntimeScalarComputedMaskStoreRouteFamilyPlan`, but it is
  store-named while carrying fields for both store and load-store. This round
  must repair that boundary into a memory-family abstraction and make both
  active memory routes consume it explicitly.
- Current code still has route-specific branches for store vs load-store in
  validation, description population, and provider operand binding. Some
  route-specific checks are legitimate, but duplicated route-family ownership
  should be demoted behind the shared computed-mask memory family plan.

## Scope

1. Rename, repair, or replace the store-named family plan with one RVV
   plugin-local runtime-scalar computed-mask memory route-family interface.
2. Migrate both active consumers:
   - `RuntimeScalarComputedMaskStore`
   - `RuntimeScalarComputedMaskLoadStore`
3. Preserve existing explicit and pre-realized selected-body semantics,
   runtime ABI order, compare/mask/load/store materialized uses, inactive-lane
   behavior, target leaf/header output, route operand binding mirrors,
   generated-bundle support, and fail-closed diagnostics.
4. Remove or demote duplicated one-off route-family logic for these two
   memory routes. Route-specific memory facts may remain as fields/variants in
   the shared family plan.

## Requirements

1. Route authority remains structural: typed `tcrv_rvv` body/config/runtime
   facts plus RVV plugin-owned selected-body realization, route planning,
   route provider, and `RVVRouteOperandBindingPlan` closure.
2. The shared memory family interface must derive or validate:
   - operation kind and memory form;
   - whether the route is store-only or load-merge/store;
   - runtime AVL/VL control plan;
   - runtime ABI order and parameter list;
   - provider-supported mirror;
   - required headers and C type mapping summary;
   - SEW/LMUL/policy-derived vector and mask types;
   - compare, runtime scalar splat, produced mask, source load or masked load,
     old-destination passthrough, masked store or final store target leaves;
   - mask role/source/memory form;
   - source and destination memory-form mirrors;
   - inactive-lane and passthrough layout contracts.
3. Store route requirements remain:
   - one lhs load;
   - one non-AVL `rhs_scalar` splat;
   - one `sle` compare;
   - one source payload load;
   - one `tcrv_rvv.masked_store` consuming the produced mask and source value;
   - runtime `n`/AVL and SEW32 LMUL m1 config;
   - false lanes and tail remain preserved.
4. Load-store route requirements remain:
   - one lhs load;
   - one non-AVL `rhs_scalar` splat;
   - one `sle` compare;
   - one old-destination load from output buffer;
   - one `tcrv_rvv.masked_load` consuming source, produced mask, and old
     destination passthrough;
   - one final unit store to output buffer;
   - runtime `n`/AVL and SEW32 LMUL m1 config;
   - false lanes preserve old destination and tail remains preserved.
5. Provider operand binding must be shared for common memory-family roles and
   explicitly distinguish store-only active-store value from load-store
   masked-load source/passthrough/final-store roles without bypassing
   `RVVRouteOperandBindingPlan`.
6. Common EmitC/export must remain neutral: no semantic inference from route
   ids, helper strings, mirrors, artifact names, descriptors, direct-C,
   source-front-door metadata, C ABI strings alone, or exact intrinsic
   spelling.
7. Generated-bundle behavior and `ssh rvv` evidence expectations must remain
   valid. If emitted target sequence or provider materialization changes,
   rerun real `ssh rvv` for at least one store and one load-store regression.

## Acceptance Criteria

- [x] PRD and task context point to the RVV/EmitC/testing specs plus directly
      relevant prior store/load-store/select-family and operand-binding
      evidence.
- [x] One shared RVV plugin-local runtime-scalar computed-mask memory
      route-family plan or equivalent interface exists and is consumed by both
      `runtime_scalar_cmp_masked_store` and
      `runtime_scalar_cmp_masked_load_store`.
- [x] The old store-named or duplicated one-off route-family logic for these
      two routes is removed or demoted so the two production consumers do not
      derive route support through parallel independent interfaces.
- [x] Planning/provider output for both routes still derives target leaf
      selection, headers, mirrors, ABI parameters, materialized operands,
      inactive-lane/passthrough contracts, source/destination memory forms,
      and fail-closed diagnostics from typed body/config/runtime facts plus
      `RVVRouteOperandBindingPlan` closure.
- [x] Explicit and pre-realized selected-body target/header FileCheck coverage
      for store and load-store proves family-derived closure, materialized
      operands, mirrors, headers, and source/destination memory-form fields.
- [x] Negative fail-closed coverage still covers missing compare input,
      missing runtime scalar, missing mask consumer, missing source payload,
      missing input/output memory role, missing old-destination passthrough for
      load-store, missing runtime n/AVL, wrong memory form, unsupported
      predicate, bad dtype/config, stale route id, helper-string fallback,
      mirror-only authority, and common/export semantic inference where current
      test surfaces can express them.
- [x] Generated-bundle dry-runs pass for explicit and pre-realized
      `runtime_scalar_cmp_masked_store` and
      `runtime_scalar_cmp_masked_load_store`, counts `7,16,23`, threshold
      values already used by the archived route tasks.
- [x] Real `ssh rvv` evidence is rerun if emitted target sequence or provider
      materialization changes.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, route-id, helper-string, artifact-name, mirror-only,
      or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No new mask algebra, select routes, macc/reduction migration, dtype/LMUL
  clones, strided/indexed/segmented memory matrices, frontend lowering,
  source-front-door positive routes, dashboards, report-only work, helper-only
  cleanup, or broad all-route rewrite.
- No migration of macc or standalone reduction unless required as a narrow
  regression repair.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`,
  descriptor, direct-C, source-front-door, helper-string, route-id,
  artifact-name, or mirror-only route authority.
- No common EmitC/export ownership of compare, mask, memory form, load-merge,
  store, dtype/config, runtime roles, result roles, or acceptance state.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test`.
3. Run focused FileCheck/lit coverage for explicit and pre-realized
   `runtime_scalar_cmp_masked_store` and
   `runtime_scalar_cmp_masked_load_store` target/header fixtures plus directly
   relevant negative fail-closed fixtures.
4. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if route planning or
   provider helpers are changed.
5. Run generated-bundle dry-runs for explicit and pre-realized store and
   load-store with counts `7,16,23`.
6. Run real `ssh rvv` generated-bundle evidence if emitted target sequence or
   ABI/materialized operand mapping changes.
7. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`

Relevant archived tasks read:

- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-select-route-family-interface/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-stage2-rvv-runtime-scalar-computed-mask-store-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-runtime-scalar-computed-mask-load-merge-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-route-operand-abi-binding/prd.md`

Initial source surfaces inspected:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- relevant target/plugin tests and generated-bundle support for the two target
  routes.

## Definition Of Done

Both active runtime-scalar computed-mask contiguous memory production routes
consume a shared plugin-local memory route-family interface, preserve existing
behavior/evidence, and leave common EmitC/export as neutral materialization
only. If this grows beyond one coherent round, finish store plus load-store
migration and leave macc/reduction family migration explicitly out of scope.

## Completion Evidence

Production changes:

- Replaced the store-named
  `RVVSelectedBodyRuntimeScalarComputedMaskStoreRouteFamilyPlan` with
  `RVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteFamilyPlan`.
- Added `usesLoadMerge` so the shared family explicitly distinguishes the
  store-only consumer from the load-merge/store consumer without splitting
  them into separate route-family interfaces.
- Migrated `RuntimeScalarComputedMaskStore` and
  `RuntimeScalarComputedMaskLoadStore` analysis to derive/apply the shared
  memory family plan.
- Made the RVV EmitC provider require the shared memory family plan before it
  materializes runtime-scalar computed-mask store or load-store routes, and
  consume plan-owned setvl/load/splat/compare/store/header facts.
- Removed the old store-named route-family type/function/optional names from
  active RVV planning/provider code.

Routes converted:

- `runtime_scalar_cmp_masked_store`
- `runtime_scalar_cmp_masked_load_store`

Routes intentionally not converted:

- macc, standalone reduction, strided/indexed/segmented computed-mask memory,
  select, scalar broadcast, runtime splat store, and ordinary memory routes.
  They were outside this task's bounded memory-family scope.

Selected-body realization and binding closure evidence:

- Explicit and pre-realized store/load-store lit tests passed through
  selected-body realization, emission plan materialization, target header
  export, provider-supported mirrors, route operand binding plan mirrors, and
  source/destination memory-form fields.
- `RVVRouteOperandBindingPlan` remains the materialized operand authority for
  `lhs`, `rhs_scalar`, `src`, `dst`, and `n`. Store uses `src` as masked-store
  active value and `dst` as masked-store destination. Load-store uses `src` as
  masked-load source, `dst` as old-destination passthrough and final store
  destination.
- Mirror fields remain downstream diagnostics only; provider materialization
  now requires the shared memory family plan and still verifies binding
  closure before constructing the lowerable route.

Checks run:

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-(computed-mask-(store|load-store)-dataflow|cmp-masked-(store|load-store))'` from `build/test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/stage2_rvv_computed_mask_memory_route_family --run-id explicit --overwrite --op-kind runtime_scalar_cmp_masked_store --op-kind runtime_scalar_cmp_masked_load_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_rvv_computed_mask_memory_route_family --run-id pre_realized --overwrite --op-kind runtime_scalar_cmp_masked_store --op-kind runtime_scalar_cmp_masked_load_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/stage2_rvv_computed_mask_memory_route_family --run-id ssh_explicit --overwrite --op-kind runtime_scalar_cmp_masked_store --op-kind runtime_scalar_cmp_masked_load_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91 --timeout 180 --connect-timeout 10`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_rvv_computed_mask_memory_route_family --run-id ssh_pre_realized --overwrite --op-kind runtime_scalar_cmp_masked_store --op-kind runtime_scalar_cmp_masked_load_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91 --timeout 180 --connect-timeout 10`
- Active-authority scan over touched RVV/plugin/export/script/test diff for
  `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor/direct-C/source-front-door/source-artifact,
  helper-string, artifact-name, mirror-only, and common/export semantic
  authority.
- Stale store-family symbol scan for old
  `RuntimeScalarComputedMaskStoreRouteFamilyPlan` names.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`

Self-repair:

- Initial focused ssh run with only `--runtime-count 16` was blocked by the
  evidence script because runtime ABI evidence requires several runtime
  counts. Reran explicit and pre-realized ssh with counts `7,16,23` and
  thresholds `-37,91`.
- `clang-format` was not available in this environment; the C++ changes were
  kept small and passed `git diff --check` plus compilation.

Runtime evidence:

- Explicit ssh RVV generated-bundle run passed for both routes with counts
  `7,16,23` and `rhs_scalar=-37,91`, proving active lanes, inactive lane
  preservation, source preservation, and tail preservation.
- Pre-realized ssh RVV generated-bundle run passed for both routes with the
  same count/threshold set, proving selected-body realization still feeds the
  shared provider route correctly.

Spec-update judgment:

- No `.trellis/spec/**` update was made. This task applies the existing RVV
  plugin-owned route-family, RouteOperandBindingPlan, and common EmitC
  neutrality contracts; it did not introduce a new durable architecture rule.
