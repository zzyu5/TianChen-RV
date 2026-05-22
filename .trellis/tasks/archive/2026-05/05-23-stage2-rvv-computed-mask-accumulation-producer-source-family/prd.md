# Stage2 RVV computed-mask accumulation producer-source family migration

## Goal

Extend the existing RVV plugin-local computed-mask accumulation family so it is
not runtime-scalar-only. The family must carry an explicit mask-producer source
dimension and be consumed by the existing vector/vector computed-mask
accumulation routes:

- `computed_masked_macc_add`
- `computed_mask_standalone_reduce_add`

The already migrated runtime-scalar accumulation routes remain supported and
may change only through shared-interface adjustments:

- `runtime_scalar_cmp_masked_macc_add`
- `runtime_scalar_cmp_masked_standalone_reduce_add`

This is a production-path migration of existing routes, not new RVV operation
coverage.

## Direction Source

- Direction title: `Stage2 RVV computed-mask accumulation producer-source family migration`.
- Module owner: RVV plugin-local computed-mask accumulation route-family support
  for existing non-runtime-scalar/vector-compare accumulation consumers.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `4b976d80 rvv: share computed-mask accumulation route family`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes/Codex Direction Brief before source edits.

## Current Inventory

Inventory is from current HEAD, the relevant specs, the previous archived
accumulation-family task, and focused planning/provider/test inspection.

- The previous archived task
  `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-accumulation-route-family-interface/`
  introduced `RVVSelectedBodyRuntimeScalarComputedMaskAccumulationRouteFamilyPlan`
  for the two runtime-scalar routes only. It explicitly left vector/vector
  `computed_masked_macc_add` and `computed_mask_standalone_reduce_add` outside
  scope.
- `computed_masked_macc_add` is an existing active route with compare lhs/rhs
  vector loads, payload `lhs`/`rhs` vector loads, accumulator vector load,
  `tcrv_rvv.masked_macc`, output store, runtime `n`, explicit and pre-realized
  target fixtures, and generated-bundle dry-run tests.
- `computed_mask_standalone_reduce_add` is an existing active route with compare
  lhs/rhs vector loads, source vector load, scalar accumulator seed, scalar
  output, `tcrv_rvv.masked_standalone_reduce`, runtime `n`, explicit and
  pre-realized target fixtures, and computed-mask min/max sibling coverage.
- Current runtime-scalar macc and standalone-reduction routes derive common
  compare/mask/runtime/header facts through the runtime-scalar accumulation
  family. The vector/vector routes do not consume that family; vector macc uses
  direct description population and provider operand binding, while vector
  standalone reduction consumes only the standalone-reduction route family.
- The required producer-source dimension is already structurally visible in
  typed bodies:
  - vector/vector producer: compare RHS is a vector `tcrv_rvv.load`;
  - runtime scalar producer: compare RHS is a non-AVL runtime scalar
    `tcrv_rvv.splat`.
- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/index.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/implementation-stack/index.md`

## Scope

1. Introduce or repair one RVV plugin-local computed-mask accumulation
   route-family interface that supports both producer sources:
   vector/vector compare and runtime scalar splat-compare.
2. Migrate these existing vector/vector accumulation consumers onto that family:
   - `ComputedMaskedMAccAdd`
   - `ComputedMaskStandaloneReduceAdd`
3. Keep these existing runtime-scalar consumers on the same family:
   - `RuntimeScalarComputedMaskedMAccAdd`
   - `RuntimeScalarComputedMaskStandaloneReduceAdd`
4. Preserve suffix semantics:
   - vector macc remains masked multiply-add accumulation with inactive lanes
     preserving accumulator values;
   - standalone reduction remains horizontal scalar reduction with scalar carry
     across runtime VL chunks and inactive-lane exclusion for add.
5. Keep common EmitC/export neutral. The family may populate mirrors and target
   header fields, but materialization must remain provider-built from typed body
   facts plus `RVVRouteOperandBindingPlan` closure.

## Requirements

1. Route authority remains structural: typed `tcrv_rvv` body/config/runtime
   facts, RVV plugin-owned realization/planning/provider logic, and
   `RVVRouteOperandBindingPlan` closure.
2. The shared family must carry or validate:
   - producer source kind: vector compare RHS or runtime scalar threshold;
   - operation kind and memory form;
   - suffix kind: vector masked macc or scalar horizontal standalone reduction;
   - runtime AVL/VL control plan;
   - runtime ABI order and runtime ABI parameter list;
   - provider-supported mirror and target leaf/header facts;
   - SEW/LMUL/policy-derived vector and mask types;
   - setvl, vector load, optional runtime scalar splat, compare, mask, store;
   - compare predicate, mask role/source/memory form;
   - accumulator/result contracts and inactive-lane/scalar-carry contracts.
3. Vector/vector `computed_masked_macc_add` must still require compare lhs/rhs
   loads, payload lhs/rhs loads, accumulator load, produced mask,
   `tcrv_rvv.masked_macc`, output store, SEW32 LMUL m1 config, and runtime `n`.
4. Vector/vector `computed_mask_standalone_reduce_add` must still require compare
   lhs/rhs loads, source load, scalar accumulator seed, produced mask,
   `tcrv_rvv.masked_standalone_reduce`, scalar output store, SEW32 LMUL m1
   config, runtime `n`, inactive-lane zeroing, and scalar carry.
5. Runtime-scalar routes must continue to require compare lhs load, runtime
   scalar splat, produced mask, their route-specific payload/accumulator/result
   operands, and their current ABI order.
6. Old duplicated vector/vector one-off producer binding in the provider should
   be removed or demoted into a shared family helper. Route-specific payload,
   accumulator, result, and scalar-output binding may remain suffix-specific.
7. Negative diagnostics must fail closed for missing compare input, missing RHS
   vector or runtime scalar source, missing mask producer, missing payload,
   missing accumulator/scalar output, missing runtime `n`, wrong producer source,
   wrong compute suffix, bad dtype/config, stale route id, helper-string
   fallback, mirror-only authority, and common/export semantic inference where
   current surfaces can express them.

## Acceptance Criteria

- [x] A shared computed-mask accumulation route-family plan or equivalent
      producer-source interface exists and is consumed by all four active
      accumulation routes listed in this PRD.
- [x] `computed_masked_macc_add` consumes the shared family for compare/mask,
      producer-source, runtime/control/header, and binding-closure facts while
      preserving macc suffix semantics.
- [x] `computed_mask_standalone_reduce_add` consumes the shared family for
      compare/mask, producer-source, runtime/control/header, and binding-closure
      facts while preserving standalone scalar reduction semantics.
- [x] Runtime-scalar macc and standalone reduction remain supported and consume
      the same family through the runtime-scalar producer source.
- [x] Focused explicit and pre-realized target/header FileCheck coverage proves
      family-derived producer-source handling, route operand closure, mirrors,
      headers, accumulator/result contracts, and scalar carry where relevant.
- [x] Focused fail-closed tests cover the touched producer-source and accumulation
      authority paths without adding broad unrelated smoke coverage.
- [x] Generated-bundle dry-runs pass for migrated explicit and pre-realized
      vector/vector routes. Runtime-scalar dry-runs are rerun if interface mirror
      or provider output changes affect their expected metadata.
- [x] Real `ssh rvv` evidence is run for at least one migrated macc and one
      migrated standalone reduction if provider materialization or target leaf
      output changes.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-front-door/source-artifact,
      helper-string, artifact-name, mirror-only, or common/export RVV semantic
      authority.
- [x] `git diff --check`, focused checks, `check-tianchenrv`, clean final status,
      task finish/archive, and one coherent commit complete if the task
      finishes.

## Non-Goals

- No new arithmetic, reduction, mask algebra, select, memory, dtype, LMUL,
  strided/indexed/segmented, frontend, source-front-door, dashboard, or report
  routes.
- No collapse of vector macc and horizontal scalar reduction into one ambiguous
  compute suffix.
- No migration through legacy `RVVI32M1`, `rvv-i32m1`, finite positive
  `tcrv_rvv.i32_*`, descriptor, direct-C, source-front-door, helper-string,
  route-id, artifact-name, or mirror-only route authority.
- No common EmitC/export ownership of compare, mask producer source, macc,
  reduction, accumulator, scalar output, dtype/config, runtime roles, result
  roles, or acceptance state.

## Validation Plan

1. Start the Trellis task after PRD/context files exist.
2. Build focused compiler targets if needed:
   `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
3. Run focused lit/FileCheck tests for:
   - explicit/pre-realized `computed_masked_macc_add`;
   - explicit/pre-realized `computed_mask_standalone_reduce_add`;
   - runtime-scalar accumulation fixtures touched by shared-interface changes;
   - directly relevant negative pre-realized/dataflow tests.
4. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if provider/planning API
   behavior changes are not fully visible through lit.
5. Run generated-bundle dry-runs for explicit and pre-realized migrated routes
   with counts `7,16,23`.
6. Run real `ssh rvv` generated-bundle evidence for one migrated macc and one
   migrated standalone reduction if target materialization changes.
7. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Definition Of Done

The vector/vector computed-mask macc and standalone-reduction routes consume the
same producer-source-aware computed-mask accumulation family as the
runtime-scalar routes. The old vector/vector one-off common producer/binding
logic is removed or demoted, route-specific suffix semantics remain distinct,
tests/evidence are focused and passing, the task is finished/archived, and the
work is committed.
