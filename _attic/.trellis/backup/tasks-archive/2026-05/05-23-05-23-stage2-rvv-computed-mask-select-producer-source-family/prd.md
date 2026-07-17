# Stage2 RVV computed-mask select producer-source family migration

## Goal

Extend the existing RVV plugin-local computed-mask select route family so it is
not runtime-scalar-only. The family must carry an explicit mask-producer source
dimension and be consumed by the existing vector/non-runtime-scalar
`computed_mask_select` production path while preserving the already migrated
runtime-scalar select routes:

- `runtime_scalar_cmp_select`
- `runtime_scalar_dual_cmp_mask_and_select`

This is a production-path migration of an existing select route, not new RVV
operation coverage.

## Direction Source

- Direction title: `Stage2 RVV computed-mask select producer-source family migration`.
- Module owner: RVV plugin-local computed-mask select route-family support for
  existing vector/non-runtime-scalar compare-select consumers.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `1cf8f602 rvv: share computed-mask accumulation producer source`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes/Codex Direction Brief before source edits.

## Current Inventory

Inventory is from current HEAD, the relevant specs, and directly relevant
archived select / accumulation / closure tasks.

- The archived computed-mask select route-family task introduced
  `RVVSelectedBodyRuntimeScalarComputedMaskSelectRouteFamilyPlan` for the two
  runtime-scalar routes only. It explicitly left vector `computed_mask_select`
  outside scope as a regression anchor.
- Current `computed_mask_select` is an active production route with compare
  lhs/rhs vector loads, true/false vector loads, `tcrv_rvv.compare`,
  `tcrv_rvv.select`, output store, runtime `n`, explicit and pre-realized
  target/header fixtures, generated-bundle dry-runs, and real `ssh rvv`
  evidence from its archived executable-slice task.
- Current `computed_mask_select` still receives some compare/mask/source/result
  description fields through route-specific branches in
  `RVVEmitCRoutePlanning.cpp`, and its operand binding closure is separate from
  the shared runtime-scalar select helper.
- Current runtime-scalar single and dual select routes derive common route
  facts through the runtime-scalar computed-mask select family and share
  `addRuntimeScalarComputedMaskSelectOperandBindings`.
- The accumulation producer-source task established the current pattern for
  carrying a producer-source dimension across vector compare RHS and runtime
  scalar splat compare RHS producers. This select task should follow that
  pattern without importing accumulation suffix semantics.
- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`

## Scope

1. Introduce or repair one RVV plugin-local computed-mask select route-family
   interface that supports all current producer-source forms:
   - vector compare RHS load for `computed_mask_select`;
   - runtime scalar splat compare RHS for `runtime_scalar_cmp_select`;
   - dual runtime scalar splat compare RHS producers plus mask-and composition
     for `runtime_scalar_dual_cmp_mask_and_select`.
2. Migrate the existing vector `computed_mask_select` consumer onto that family
   for compare/mask, producer-source, runtime/control/header/mirror, and
   binding-closure facts.
3. Preserve the existing runtime-scalar single and dual select routes, allowing
   only shared-interface adjustments required by the producer-source migration.
4. Keep route support derived from typed `tcrv_rvv` body/config/runtime facts
   plus `RVVRouteOperandBindingPlan` closure.
5. Keep common EmitC/export neutral. Mirrors, target headers, route ids, and
   artifact metadata remain outputs of provider planning/materialization, not
   inputs to route support.

## Requirements

1. Route authority remains structural: typed `tcrv_rvv` body/config/runtime
   facts, RVV plugin-owned realization/planning/provider code, and
   `RVVRouteOperandBindingPlan` closure.
2. The shared select family must carry or validate:
   - producer source kind: vector compare RHS, runtime scalar threshold, or
     dual runtime scalar thresholds with mask-and composition;
   - operation kind and memory form;
   - runtime AVL/VL control plan;
   - runtime ABI order and runtime ABI parameter list;
   - provider-supported mirror and target leaf/header facts;
   - SEW/LMUL/policy-derived vector and mask types;
   - setvl, vector load, optional runtime scalar splat, compare, optional
     secondary compare, optional mask_and, select, and store target leaves;
   - mask role/source/memory form/composition mirrors;
   - select layout, true/false payload layout, source/destination memory form,
     and unit-stride layout facts.
3. Vector `computed_mask_select` must still require compare lhs/rhs loads,
   true/false payload loads, compare-produced mask, select, store, SEW32 LMUL
   m1 config, supported predicate, and runtime `n`.
4. Runtime-scalar `runtime_scalar_cmp_select` must still require lhs load,
   runtime scalar splat, compare-produced mask, true/false payload loads,
   select, store, SEW32 LMUL m1 config, predicate `sle`, and runtime `n`.
5. Runtime-scalar `runtime_scalar_dual_cmp_mask_and_select` must still require
   two compare lhs loads, two runtime scalar splats, two compare masks, one
   typed `mask_and`, true/false payload loads, select, store, SEW32 LMUL m1
   config, predicate `sle`, and runtime `n`.
6. Old duplicated vector `computed_mask_select` one-off planning/provider
   logic should be removed or demoted into a shared select-family helper.
   Route-specific facts may remain as data in the shared plan.
7. Negative diagnostics must fail closed for missing compare input, missing
   vector RHS or runtime scalar source, missing mask producer, missing true or
   false payload, missing output, missing runtime `n`, wrong producer source,
   wrong select suffix/layout, bad dtype/config, stale route id, helper-string
   fallback, mirror-only authority, and common/export semantic inference where
   current surfaces can express them.

## Acceptance Criteria

- [x] One shared computed-mask select route-family plan or equivalent
      producer-source interface exists and is consumed by all three active
      select routes listed in this PRD.
- [x] `computed_mask_select` consumes the shared family for compare/mask,
      producer-source, runtime/control/header/mirror, and binding-closure facts
      while preserving vector compare-select semantics.
- [x] Runtime-scalar single and dual select routes remain supported and consume
      the same family through runtime-scalar producer sources.
- [x] The old vector `computed_mask_select` one-off common producer/binding
      logic is removed or demoted.
- [x] Focused explicit and pre-realized target/header FileCheck coverage proves
      family-derived producer-source handling, route operand closure, mirrors,
      headers, true/false materialized operands, output materialization, and
      lane semantics.
- [x] Focused fail-closed coverage covers touched producer-source and
      select-family authority paths without adding broad unrelated smoke
      coverage.
- [x] Generated-bundle dry-runs pass for migrated explicit and pre-realized
      vector `computed_mask_select` routes. Runtime-scalar dry-runs are rerun
      if shared-interface mirrors or provider output changes affect their
      expected metadata.
- [x] Real `ssh rvv` evidence is run for at least one migrated vector
      `computed_mask_select` regression if provider materialization or target
      leaf output changes.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-front-door
      or source-artifact, helper-string, artifact-name, mirror-only, or
      common/export RVV semantic authority.
- [x] `git diff --check`, focused checks, `check-tianchenrv`, clean final
      status, task finish/archive, and one coherent commit complete if the task
      finishes.

## Non-Goals

- No new mask algebra, memory routes, accumulation routes, reduction kinds,
  arithmetic operation kinds, dtype/LMUL clones, frontend lowering,
  source-front-door positive routes, dashboards, report-only work, helper-only
  cleanup, or broad all-route rewrite.
- No collapse of single-mask select, dual-mask-and-select, and vector compare
  select into one ambiguous compute semantic. Only producer-source and
  select-family mechanics that are truly common are shared.
- No migration through legacy `RVVI32M1`, `rvv-i32m1`, finite positive
  `tcrv_rvv.i32_*`, descriptor, direct-C, source-front-door, helper-string,
  route-id, artifact-name, or mirror-only route authority.
- No common EmitC/export ownership of compare, mask producer source, select
  semantics, dtype/config, runtime roles, result roles, or acceptance state.

## Validation Plan

1. Validate the Trellis task metadata/context after PRD creation.
2. Build focused compiler/test targets if needed:
   `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`.
3. Run focused lit/FileCheck coverage for:
   - explicit and pre-realized vector `computed_mask_select`;
   - runtime-scalar single and dual select fixtures if their expected metadata
     changes;
   - directly relevant negative fail-closed fixtures.
4. Run focused C++ tests for route planning/provider/binding contract if
   touched.
5. Run generated-bundle dry-runs for explicit and pre-realized
   `computed_mask_select`, counts `7,16,23`.
6. Run real `ssh rvv` generated-bundle evidence for one migrated vector
   `computed_mask_select` regression if target materialization changes.
7. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Completion Evidence

- Production owners changed:
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- Shared interface introduced:
  `RVVSelectedBodyComputedMaskSelectRouteFamilyPlan` with
  `familyPlanID`, `maskProducerSource`,
  `usesVectorCompareProducer`, `usesRuntimeScalarProducer`, and
  `usesDualCompareMaskAnd`.
- Active routes consuming the shared family:
  - `computed_mask_select` via `vector-compare-rhs-load`;
  - `runtime_scalar_cmp_select` via `runtime-scalar-splat-compare-rhs`;
  - `runtime_scalar_dual_cmp_mask_and_select` via
    `dual-runtime-scalar-splat-compare-rhs-mask-and`.
- Old duplicated vector select planning/provider authority was demoted into the
  shared select-family derive/apply path and shared binding closure. Route
  semantics remain distinct per producer source.
- Focused lit/FileCheck:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='computed-mask-select|runtime-scalar-cmp-select|runtime-scalar-dual-cmp-mask-and-select|rvv-generic-stage2-runtime-scalar-cmp-select-negative|runtime-scalar-dual-compare-mask-and-select-dataflow'`
  from `build/test` passed 11/11.
- C++ focused tests:
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-construction-protocol-common-test`, and
  `build/bin/tianchenrv-target-artifact-export-test` passed.
- Generated-bundle dry-runs:
  - explicit selected body:
    `--op-kind computed_mask_select_sle --runtime-count 7 --runtime-count 16 --runtime-count 23`
    passed at
    `artifacts/tmp/stage2_rvv_computed_mask_select_producer_source_family/explicit-computed-mask-select-sle-producer-source`;
  - pre-realized selected body:
    `--pre-realized-selected-body --op-kind computed_mask_select --runtime-count 7 --runtime-count 16 --runtime-count 23`
    passed at
    `artifacts/tmp/stage2_rvv_computed_mask_select_producer_source_family/pre-realized-computed-mask-select-producer-source`.
- Real `ssh rvv` evidence:
  `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_mask_select --runtime-count 7 --runtime-count 16 --runtime-count 23`
  passed with `PASS op=computed_mask_select counts=7,16,23`;
  all counts reported true and false selected lanes plus tail preservation.
- Active-authority scan over touched RVV/plugin/export/script/test paths found
  only existing fail-closed legacy-op and forbidden-residue guard/self-test
  references, with no new positive legacy authority.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed 349/349.

## Definition Of Done

The vector/non-runtime-scalar `computed_mask_select` route consumes the same
producer-source-aware computed-mask select family as the runtime-scalar select
routes. Old duplicated vector select common producer/binding logic is removed or
demoted, route-specific select semantics remain distinct, tests/evidence are
focused and passing, the task is finished/archived, and the work is committed.
