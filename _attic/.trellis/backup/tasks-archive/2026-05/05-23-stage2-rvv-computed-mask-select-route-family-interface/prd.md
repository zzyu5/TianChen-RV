# Stage2 RVV computed-mask select route-family interface

## Goal

Introduce a bounded RVV plugin-local computed-mask select route-family
interface for the two existing runtime-scalar select production paths:
`runtime_scalar_cmp_select` and
`runtime_scalar_dual_cmp_mask_and_select`.

This is not new Stage2 operation coverage. The goal is to make route support,
target leaf/header fields, materialized operand closure, mirrors, and
fail-closed diagnostics for single-mask and dual-mask runtime-scalar select
bodies derive from one shared RVV planning/provider boundary while preserving
typed `tcrv_rvv` body authority and common EmitC/export neutrality.

## Direction Source

- Direction title: `Stage2 RVV plugin-owned computed-mask select route-family interface`.
- Module owner: RVV plugin-local route planning/provider interface for
  computed-mask select families.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `160cd2f4 rvv: add runtime scalar mask composition route`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes Direction Brief before source edits.

## Current Inventory

Inventory is from current repo state and directly relevant archived tasks.

- `runtime_scalar_cmp_select` is complete and production-supported. It carries
  a runtime scalar threshold through `tcrv_rvv.splat`, compares `lhs` with the
  splatted scalar, consumes the produced mask in `tcrv_rvv.select`, stores the
  selected value, exports route operand binding mirrors, dry-runs generated
  bundles, and has real `ssh rvv` evidence.
- `runtime_scalar_dual_cmp_mask_and_select` is complete and
  production-supported. It carries two runtime scalar thresholds, two compare
  lhs buffers, two compare masks, one typed `tcrv_rvv.mask_and`, true/false
  select payloads, output store, route operand binding mirrors, generated
  bundle evidence, and real `ssh rvv` evidence.
- Current planning has separate one-off plan types and optionals:
  `RVVSelectedBodyRuntimeScalarCompareSelectRouteFamilyPlan` and
  `RVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRouteFamilyPlan`.
- Current planning has separate derive/apply/validate functions for those two
  routes even though both are runtime-scalar compare-produced-mask select
  families with shared config, headers, select layout, ABI closure shape, and
  common EmitC/export ownership constraints.
- Current provider binding closure has separate route-specific branches for
  `runtime_scalar_cmp_select` and
  `runtime_scalar_dual_cmp_mask_and_select`.
- Existing vector-vs-vector `computed_mask_select` is a useful regression
  anchor but is not in this task's migration scope.

## Scope

1. Add or repair one RVV plugin-local route-family interface for runtime-scalar
   computed-mask select routes.
2. Migrate both active production consumers:
   - `runtime_scalar_cmp_select`
   - `runtime_scalar_dual_cmp_mask_and_select`
3. Preserve existing explicit and pre-realized selected-body semantics,
   runtime ABI role order, materialized mask/select/store uses, target
   leaf/header output, route operand binding mirrors, generated bundle support,
   and fail-closed diagnostics.
4. Remove or demote duplicated one-off planning/provider logic for these two
   routes. Route-specific facts may remain as data in the shared family plan;
   they must not remain as parallel independent route architectures.

## Requirements

1. Route authority remains structural: typed `tcrv_rvv` body/config/runtime
   facts plus RVV plugin-owned selected-body realization, route planning,
   route provider, and `RVVRouteOperandBindingPlan` closure.
2. The shared family interface must derive or validate:
   - operation kind and memory form;
   - runtime AVL/VL control plan;
   - runtime ABI order and ABI parameter list;
   - provider-supported mirror;
   - required headers and C type mapping summary;
   - SEW/LMUL/policy-derived vector and mask types;
   - compare, optional secondary compare, optional mask-and, select, splat,
     load, store target leaves;
   - mask role/source/memory form/composition mirrors;
   - selected true/false payload layout.
3. The single-mask route must still require exactly one runtime scalar splat,
   one compare mask, true/false loads, select, store, runtime `n`, and SEW32
   LMUL m1 config.
4. The dual-mask route must still require two runtime scalar splats, two
   compare masks, one typed `mask_and`, true/false loads, select, store,
   runtime `n`, and SEW32 LMUL m1 config.
5. Provider operand binding must be shared for the common select-family roles
   and explicitly extend to the dual compare/mask-and roles without bypassing
   `RVVRouteOperandBindingPlan`.
6. Common EmitC/export must remain neutral: no semantic inference from route
   ids, helper strings, mirrors, artifact names, descriptors, direct-C,
   source-front-door metadata, C ABI strings alone, or exact intrinsic spelling.
7. Existing generated-bundle behavior and `ssh rvv` evidence expectations must
   remain valid. If emitted target sequence changes, rerun real `ssh rvv` for
   dual explicit/pre-realized and one single-mask regression.

## Acceptance Criteria

- [x] PRD and task context point to the relevant RVV/EmitC/testing specs and
      prior completed route evidence.
- [x] One shared RVV plugin-local computed-mask select route-family plan or
      equivalent interface exists and is consumed by both
      `runtime_scalar_cmp_select` and
      `runtime_scalar_dual_cmp_mask_and_select`.
- [x] The old independent one-off route-family plan path for these two routes
      is removed or demoted so the two production consumers do not derive route
      support through separate parallel interfaces.
- [x] Planning/provider output for both routes still derives target leaf
      selection, headers, mirrors, ABI parameters, materialized operands, and
      fail-closed diagnostics from typed body/config/runtime facts plus
      `RVVRouteOperandBindingPlan` closure.
- [x] Explicit and pre-realized selected-body target/header FileCheck coverage
      for both routes proves family-derived closure, materialized operands,
      mirrors, and headers.
- [x] Negative fail-closed coverage still covers missing compare input,
      missing runtime scalar role, missing `mask_and` on dual route, wrong
      predicate, missing true/false/out/n operands, bad dtype/config, stale
      route id, helper-string fallback, mirror-only authority, and common/export
      semantic inference where current test surfaces can express them.
- [x] Generated-bundle dry-runs pass for explicit and pre-realized
      `runtime_scalar_cmp_select` and
      `runtime_scalar_dual_cmp_mask_and_select`.
- [x] Real `ssh rvv` evidence is rerun if emitted target sequence changes.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, route-id, helper-string, artifact-name, mirror-only,
      or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No new mask algebra such as OR/XOR/NOT.
- No new memory, macc, reduction, conversion, dtype, LMUL, frontend, source
  front-door, dashboard, report-only, broad all-route, or future plugin work.
- No migration of masked store/load/macc/reduction unless required as a
  narrowly scoped regression repair.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`,
  descriptor, direct-C, source-front-door, helper-string, route-id, artifact
  name, or mirror-only route authority.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets:
   `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`.
3. Run focused FileCheck/lit coverage for:
   - explicit runtime scalar compare-select target/header fixture;
   - pre-realized runtime scalar compare-select target/header fixture;
   - explicit dual runtime scalar compare mask-and select target/header
     fixture;
   - pre-realized dual runtime scalar compare mask-and select target/header
     fixture;
   - directly relevant negative fail-closed fixtures.
4. Run focused C++ tests for route planning/provider/binding contract if
   touched.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes.
6. Run generated-bundle dry-runs for explicit and pre-realized
   `runtime_scalar_cmp_select` and
   `runtime_scalar_dual_cmp_mask_and_select`, counts `7,16,23`.
7. Run real `ssh rvv` generated-bundle evidence if emitted target sequence or
   ABI materialization changes.
8. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Relevant archived tasks read:

- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-computed-mask-composition-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-runtime-scalar-compare-select-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-operand-binding-contract-closure-gate/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-operand-binding-coverage-closure/prd.md`

Initial source surfaces inspected:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- relevant target/plugin tests and generated-bundle support for the two target
  routes.

## Definition Of Done

Both active runtime-scalar select production routes consume a shared
plugin-local route-family interface, preserve existing behavior/evidence, and
leave common EmitC/export as neutral materialization only.

## Completion Evidence

Production changes:

- Added `RVVSelectedBodyRuntimeScalarComputedMaskSelectRouteFamilyPlan` as the
  shared RVV plugin-local route-family plan for
  `runtime_scalar_cmp_select` and
  `runtime_scalar_dual_cmp_mask_and_select`.
- Removed the independent one-off plan optionals for
  `RVVSelectedBodyRuntimeScalarCompareSelectRouteFamilyPlan` and
  `RVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRouteFamilyPlan`.
- Consolidated route-family derive/validate/apply logic in RVV EmitC route
  planning, including runtime ABI order, target leaf profile,
  provider-supported mirror, headers, C type mapping, mask role/source/form,
  optional secondary compare, optional `mask_and`, select layout, and runtime
  ABI parameters.
- Consolidated `RVVRouteOperandBindingPlan` construction for the single-mask
  and dual-mask routes through `addRuntimeScalarComputedMaskSelectOperandBindings`.
- Consolidated provider binding closure for both production routes while
  keeping dual-specific `cmp_lhs_b`, `rhs_scalar_b`, `mask-and-*`, and
  `out` header-mirror requirements fail-closed.
- Intentionally did not migrate computed-mask vector select, masked store/load,
  macc, reduction, memory movement, or future route families because this PRD
  is bounded to the two active runtime-scalar select production consumers.

Validation results:

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-23-stage2-rvv-computed-mask-select-route-family-interface`
  passed.
- Focused build passed:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`.
- Focused C++ tests passed:
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-construction-protocol-common-test`, and
  `build/bin/tianchenrv-target-artifact-export-test`.
- Script checks passed:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- Focused lit/FileCheck pass:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='runtime-scalar-cmp-select|runtime-scalar-dual-cmp-mask-and-select|rvv-generic-stage2-runtime-scalar-cmp-select-negative|runtime-scalar-dual-compare-mask-and-select-dataflow'`
  from `build/test` passed 6 selected tests.
- Generated-bundle dry-runs passed:
  explicit and pre-realized `runtime_scalar_cmp_select`, plus explicit and
  pre-realized `runtime_scalar_dual_cmp_mask_and_select`, using counts
  `7,16,23` and runtime scalar values `-37,91`.
- Continuation retry live dry-run run ids:
  `20260523-cmsel-single-explicit`, `20260523-cmsel-single-pre`,
  `20260523-cmsel-dual-explicit`, and `20260523-cmsel-dual-pre`.
- Real `ssh rvv` evidence passed:
  explicit dual route, pre-realized dual route, and explicit single-mask
  regression route, each with counts `7,16,23` and scalar values `-37,91`.
- Continuation retry live `ssh rvv` run ids:
  `20260523-cmsel-dual-explicit-ssh`,
  `20260523-cmsel-dual-pre-ssh`, and
  `20260523-cmsel-single-explicit-ssh`.
- Active-authority scan over touched source/task files found no new positive
  legacy route authority. The only touched production-source match was the
  existing fail-closed guard for `tcrv_rvv.i32_*`.
- Spec-update review found no `.trellis/spec/**` change was needed because the
  existing RVV plugin, unified EmitC route, and MLIR testing contracts already
  define this plugin-owned family-interface boundary.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed 349/349 tests.
