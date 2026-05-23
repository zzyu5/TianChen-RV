# Stage2 RVV computed-mask memory producer-source family migration

## Goal

Extend the RVV plugin-local computed-mask memory route-family interface so it
is not runtime-scalar-only. The family must carry an explicit mask-producer
source dimension and be consumed by the existing non-runtime/vector compare
computed-mask memory consumers while preserving the already migrated
runtime-scalar memory routes:

- `runtime_scalar_cmp_masked_store`
- `runtime_scalar_cmp_masked_load_store`

The repository has no active route named exactly `computed_masked_store` or
`computed_masked_load_store`. The closest active bounded store-only and
load-merge/store consumers are:

- `computed_masked_strided_store`
- `computed_masked_unit_load_store`

This task migrates those two active routes onto the shared family boundary.
Strided/indexed/segmented forms beyond that nearest active store/load-store
pair remain out of scope unless a focused regression requires a narrow repair.

## Direction Source

- Direction title: `Stage2 RVV computed-mask memory producer-source family migration`.
- Module owner: RVV plugin-local computed-mask memory route-family support for
  existing vector/non-runtime-scalar compare-mask memory consumers.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `a96dee64 rvv: share computed-mask select producer source`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes/Codex Direction Brief before source edits.

## Current Inventory

- The archived runtime-scalar computed-mask memory-family task replaced the
  store-named plan with
  `RVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteFamilyPlan`, consumed by
  `runtime_scalar_cmp_masked_store` and
  `runtime_scalar_cmp_masked_load_store`.
- The current shared memory plan is still runtime-scalar-specific. It validates
  only runtime scalar threshold producer forms and only the two runtime-scalar
  memory consumers.
- The archived select producer-source task introduced the current pattern for
  carrying a producer-source dimension in a shared family:
  `familyPlanID`, `maskProducerSource`, and boolean producer-source facets.
- The archived accumulation producer-source task applied the same pattern to
  vector compare RHS and runtime scalar splat compare RHS accumulation routes.
- Current vector/non-runtime computed-mask memory routes already derive route
  facts from typed `tcrv_rvv` body/config/runtime facts, but store/load-store
  facts are still populated through route-specific planning/provider branches.
- `computed_masked_strided_store` is the active store-only vector compare-mask
  memory consumer: compare lhs/rhs vector loads produce a mask, source payload
  is loaded, and a masked strided store writes active lanes while preserving
  inactive/output lanes.
- `computed_masked_unit_load_store` is the active unit load-merge/store vector
  compare-mask memory consumer: compare lhs/rhs vector loads produce a mask,
  old destination is loaded as passthrough, masked load reads source on true
  lanes, and a final unit store writes the merged vector.
- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/index.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/index.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant prior task context read:
  - `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-computed-mask-select-producer-source-family/prd.md`
  - `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-computed-mask-memory-route-family-interface/prd.md`
  - `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-computed-mask-memory-dataflow/prd.md`
  - `.trellis/tasks/archive/2026-05/05-22-stage2-rvv-runtime-scalar-computed-mask-store-boundary/prd.md`
  - `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-runtime-scalar-computed-mask-load-merge-boundary/prd.md`
  - `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-route-operand-abi-binding/prd.md`
  - `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-operand-binding-contract-closure-gate/prd.md`
  - `.trellis/workspace/codex/journal-13.md`

## Scope

1. Rename, repair, or replace the runtime-scalar-only memory family plan with
   one computed-mask memory family plan that explicitly carries producer-source
   facts.
2. Preserve the already migrated runtime-scalar consumers:
   - `RuntimeScalarComputedMaskStore`
   - `RuntimeScalarComputedMaskLoadStore`
3. Migrate the active vector/non-runtime store/load-store consumers:
   - `ComputedMaskStridedStore`
   - `ComputedMaskUnitLoadStore`
4. Keep route support derived from typed `tcrv_rvv` body/config/runtime facts
   plus `RVVRouteOperandBindingPlan` closure.
5. Preserve store-only versus load-merge/store semantics:
   - store-only routes consume source payload and masked store leaves;
   - load-merge/store routes consume old destination passthrough, masked load
     leaves, and final store leaves.
6. Keep common EmitC/export neutral. Mirrors, headers, route ids, artifact
   metadata, and generated bundle expectations are outputs of plugin planning
   and provider materialization, not inputs to route support.

## Requirements

1. The shared memory family must carry or validate:
   - family plan id;
   - mask producer source: runtime scalar splat compare RHS or vector compare
     RHS load;
   - whether the route uses runtime scalar or vector compare producer;
   - whether the memory semantic is store-only or load-merge/store;
   - runtime AVL/VL control plan;
   - runtime ABI order and parameter list;
   - provider-supported mirror, target leaf profile, headers, and C type
     mapping;
   - SEW/LMUL/policy-derived vector and mask types;
   - setvl, vector load, optional runtime scalar splat, compare, masked load,
     masked store, and final store target leaves;
   - mask role/source/memory form;
   - source/destination memory forms;
   - inactive-lane and passthrough layout contracts.
2. `runtime_scalar_cmp_masked_store` must still require lhs load, runtime
   scalar splat, `sle` compare, source payload load, masked store, runtime
   `n/AVL`, SEW32 LMUL m1, false-lane preservation, and tail preservation.
3. `runtime_scalar_cmp_masked_load_store` must still require lhs load, runtime
   scalar splat, `sle` compare, old-destination load, masked load with
   passthrough, final store, runtime `n/AVL`, SEW32 LMUL m1, false-lane
   preservation, and tail preservation.
4. `computed_masked_strided_store` must consume the family through vector
   compare RHS producer facts and preserve strided store-only semantics,
   destination stride binding, inactive lane no-write/preservation behavior,
   and tail preservation.
5. `computed_masked_unit_load_store` must consume the family through vector
   compare RHS producer facts and preserve unit load-merge/store semantics,
   source load, old-destination passthrough, final store, inactive lane
   preservation, and tail preservation.
6. Provider operand binding must remain the materialized operand authority.
   The family may share common role checks, but it must keep store-only active
   source value roles distinct from load-merge/store masked-load source,
   passthrough, and final-store roles.
7. Common EmitC/export must not infer compare, mask producer source, memory
   form, store/load-merge semantics, dtype/config, runtime roles, result roles,
   or acceptance state from route ids, helper strings, mirrors, artifact names,
   descriptors, direct-C, source-front-door metadata, or C ABI strings alone.
8. Duplicated one-off vector memory route-family planning/provider logic should
   be removed or demoted behind the shared computed-mask memory family plan.

## Acceptance Criteria

- [ ] One shared RVV plugin-local computed-mask memory route-family plan or
      equivalent interface exists and is consumed by all four scoped active
      routes.
- [ ] The plan exposes producer-source facts for runtime scalar splat compare
      RHS and vector compare RHS load producers.
- [ ] `computed_masked_strided_store` consumes the shared family for
      producer-source, runtime/control/header/mirror, source/destination memory
      form, inactive-lane contract, and binding-closure facts.
- [ ] `computed_masked_unit_load_store` consumes the shared family for
      producer-source, runtime/control/header/mirror, source/destination memory
      form, inactive-lane/passthrough contract, and binding-closure facts.
- [ ] `runtime_scalar_cmp_masked_store` and
      `runtime_scalar_cmp_masked_load_store` remain supported, with only shared
      interface adjustments.
- [ ] Old runtime-scalar-only naming/validation/provider requirements are
      removed or demoted so scoped vector routes do not bypass the shared
      family.
- [ ] Focused explicit and pre-realized target/header FileCheck coverage proves
      family-derived producer-source handling, closure, materialized operands,
      mirrors, headers, memory semantics, and fail-closed diagnostics for the
      migrated routes.
- [ ] Generated-bundle dry-runs pass for at least migrated
      `computed_masked_strided_store` and `computed_masked_unit_load_store`,
      counts `7,16,23`. Runtime-scalar dry-runs are rerun if provider output or
      metadata changes affect them.
- [ ] Real `ssh rvv` evidence passes for at least one migrated store route and
      one migrated load-merge/store route if provider materialization or target
      leaf/header output changes.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, route-id, helper-string, artifact-name, mirror-only,
      or common/export RVV semantic authority.
- [ ] Focused checks, `git diff --check`, `check-tianchenrv`, task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No new memory forms, no new strided/indexed/segmented route coverage, no new
  mask algebra, no select or accumulation route changes except regression
  repair caused by shared headers/types.
- No reduction kinds, arithmetic operation kinds, dtype/LMUL clones, frontend
  lowering, source-front-door positive routes, dashboards, report-only work,
  helper-only cleanup, or broad all-route rewrite.
- No collapse of store-only and load-merge/store into one ambiguous memory
  semantic. Only producer-source and truly common memory-family mechanics are
  shared.
- No migration through legacy `RVVI32M1`, `rvv-i32m1`, finite positive
  `tcrv_rvv.i32_*`, descriptor, direct-C, source-front-door, helper-string,
  route-id, artifact-name, or mirror-only route authority.
- No common EmitC/export ownership of compare, mask producer source, memory
  form, load-merge/store semantics, dtype/config, runtime roles, result roles,
  or acceptance state.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test`.
3. Run focused lit/FileCheck coverage for explicit and pre-realized
   `computed_masked_strided_store`, `computed_masked_unit_load_store`,
   `runtime_scalar_cmp_masked_store`, and
   `runtime_scalar_cmp_masked_load_store` target/header fixtures plus directly
   relevant negative fail-closed fixtures.
4. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if route planning or
   provider helpers are changed.
5. Run generated-bundle dry-runs for explicit and pre-realized migrated vector
   routes with counts `7,16,23`; include runtime-scalar store/load-store if
   shared output changes.
6. Run real `ssh rvv` generated-bundle evidence for one migrated store and one
   migrated load-store regression if emitted provider materialization or target
   header/mirror output changes.
7. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Initial Implementation Surface

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- focused target/dialect tests for scoped computed-mask memory routes
- `scripts/rvv_generated_bundle_abi_e2e.py` only if generated-bundle mirror
  expectations require update

## Definition Of Done

The scoped runtime-scalar and vector/non-runtime computed-mask memory
store/load-store production routes consume one producer-source-aware
plugin-local memory family interface; old runtime-scalar-only family ownership
is removed or demoted; store-only and load-merge/store semantics remain
distinct; focused tests, generated-bundle evidence, authority scan,
`check-tianchenrv`, task archive, clean status, and one coherent commit are
completed.
