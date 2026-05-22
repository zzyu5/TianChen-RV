# Stage2 RVV computed-mask accumulation route-family interface

## Goal

Introduce one bounded RVV plugin-local runtime-scalar computed-mask
accumulation route-family interface for the two existing arithmetic
accumulation consumers:

- `runtime_scalar_cmp_masked_macc_add`
- `runtime_scalar_cmp_masked_standalone_reduce_add`

This is a migration of existing production paths, not new operation coverage.
The shared interface must own the common runtime-scalar threshold compare,
produced-mask, typed config/runtime, header/mirror, operand-closure, and
fail-closed planning facts while preserving the distinct suffix semantics:

- vector masked multiply-add accumulation/store for macc;
- scalar-output horizontal masked standalone reduction with multi-VL scalar
  carry for reduction.

## Direction Source

- Direction title: `Stage2 RVV plugin-owned computed-mask accumulation route-family interface`.
- Module owner: RVV plugin-local route planning/provider interface for
  runtime-scalar computed-mask arithmetic accumulation consumers.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `7be8ef48 rvv: share computed-mask memory route family`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes Direction Brief before source edits.

## Current Inventory

Inventory is from current HEAD, relevant specs, archived tasks, and the active
planning/provider code.

- `runtime_scalar_cmp_masked_macc_add` is production-supported for a runtime
  scalar threshold compare feeding `tcrv_rvv.masked_macc`. It binds
  `cmp_lhs`, `rhs_scalar`, payload `lhs`, payload `rhs`, `acc`, `out`, and
  `n`, preserves inactive lanes from the accumulator, and has explicit and
  pre-realized generated-bundle plus real `ssh rvv` evidence from its archived
  task.
- `runtime_scalar_cmp_masked_standalone_reduce_add` is
  production-supported for a runtime scalar threshold compare feeding
  `tcrv_rvv.masked_standalone_reduce`. It binds `cmp_lhs`, `rhs_scalar`,
  payload `src`, scalar seed `acc`, scalar result `out`, and `n`, excludes
  inactive lanes from the scalar sum, preserves tail sentinels, and has
  explicit and pre-realized generated-bundle plus real `ssh rvv` evidence from
  its archived task.
- Current code derives the macc route through
  `RVVSelectedBodyContractionRouteFamilyPlan` with
  `usesComputedMask = true`.
- Current code derives the standalone reduction route through
  `RVVSelectedBodyStandaloneReductionRouteFamilyPlan` with
  `usesRuntimeScalarThreshold = true`.
- The two routes currently duplicate runtime-scalar computed-mask family facts
  across route planning, description population, binding closure, provider leaf
  selection, and diagnostics. The route-specific suffix plans remain useful,
  but the common computed-mask accumulation producer/closure mechanics are not
  yet represented by one shared family interface.
- Recent completed tasks established local patterns:
  - `154ada5d` shared the runtime-scalar computed-mask select family.
  - `7be8ef48` shared the runtime-scalar computed-mask memory family.
  This task applies the same production-path migration discipline to arithmetic
  accumulation, without converting select or memory routes.

## Scope

1. Add or repair one RVV plugin-local runtime-scalar computed-mask
   accumulation route-family interface.
2. Migrate both active production consumers:
   - `RuntimeScalarComputedMaskedMAccAdd`
   - `RuntimeScalarComputedMaskStandaloneReduceAdd`
3. Preserve existing explicit and pre-realized selected-body semantics,
   runtime ABI role order, materialized operand closure, target leaf/header
   output, generated-bundle behavior, and fail-closed diagnostics.
4. Keep macc and standalone reduction suffix semantics distinct. The shared
   family may carry a suffix kind or sub-plan references, but it must not merge
   vector macc and scalar horizontal reduction into one ambiguous compute op.
5. Remove or demote duplicated one-off runtime-scalar computed-mask
   accumulation logic. Route-specific suffix facts may remain in contraction or
   standalone-reduction suffix plans only where they are genuinely suffix
   specific.

## Requirements

1. Route authority remains structural: typed `tcrv_rvv` body/config/runtime
   facts plus RVV plugin-owned selected-body realization, route planning,
   route provider, and `RVVRouteOperandBindingPlan` closure.
2. The shared accumulation family interface must derive or validate:
   - operation kind and memory form;
   - suffix kind: vector masked macc or scalar horizontal standalone reduction;
   - runtime AVL/VL control plan;
   - runtime ABI order and runtime ABI parameter list;
   - provider-supported mirror;
   - required headers and C type mapping summary;
   - SEW/LMUL/policy-derived vector and mask types;
   - `setvl`, vector load, runtime scalar splat, compare, produced-mask, and
     store/header facts;
   - compare predicate, mask role, mask source, mask memory form;
   - accumulator role, result role, and inactive-lane or scalar-carry contract;
   - source/payload memory form and destination/output memory form where
     applicable.
3. Macc route requirements remain:
   - `cmp_lhs` unit load;
   - non-AVL `rhs_scalar` splat;
   - `sle` compare;
   - payload `lhs` and `rhs` unit loads;
   - accumulator load/passthrough;
   - `tcrv_rvv.masked_macc`;
   - result store to `out`;
   - runtime `n`/AVL and SEW32 LMUL m1 config;
   - false lanes preserve `acc[i]`; tail lanes remain preserved.
4. Standalone reduction route requirements remain:
   - `cmp_lhs` unit load;
   - non-AVL `rhs_scalar` splat;
   - `sle` compare;
   - payload `src` unit load;
   - scalar seed/result state from `acc` and `out`;
   - `tcrv_rvv.masked_standalone_reduce`;
   - scalar result store to `out`;
   - runtime `n`/AVL and SEW32 LMUL m1 config;
   - inactive lanes are excluded from the sum;
   - multi-VL scalar carry and tail sentinel preservation remain intact.
5. Provider operand binding must be shared for common accumulation-family roles
   (`cmp_lhs`, `rhs_scalar`, `acc`, `out`, `n`, produced mask) and explicitly
   distinguish macc payload roles (`lhs`, `rhs`) from standalone-reduction
   payload/scalar-output roles (`src`, scalar result state/store) without
   bypassing `RVVRouteOperandBindingPlan`.
6. Common EmitC/export must remain neutral: no semantic inference from route
   ids, helper strings, mirrors, artifact names, descriptors, direct-C,
   source-front-door metadata, C ABI strings alone, or exact intrinsic spelling.
7. Generated-bundle behavior and `ssh rvv` evidence expectations must remain
   valid. If emitted target sequence or provider materialization changes,
   rerun real `ssh rvv` for at least one macc and one standalone reduction
   regression, preferably explicit and pre-realized if both are touched.

## Acceptance Criteria

- [x] PRD and task context point to the RVV/EmitC/testing specs plus directly
      relevant prior select-family, memory-family, macc, standalone reduction,
      and operand-binding evidence.
- [x] One shared RVV plugin-local runtime-scalar computed-mask accumulation
      route-family plan or equivalent interface exists and is consumed by both
      `runtime_scalar_cmp_masked_macc_add` and
      `runtime_scalar_cmp_masked_standalone_reduce_add`.
- [x] The old duplicated one-off runtime-scalar computed-mask accumulation
      logic for these two routes is removed or demoted so the production
      consumers do not derive their common compare/mask/closure/header facts
      through parallel independent interfaces.
- [x] Planning/provider output for both routes still derives target leaf
      selection, headers, mirrors, ABI parameters, materialized operands,
      accumulator/result roles, and fail-closed diagnostics from typed
      body/config/runtime facts plus `RVVRouteOperandBindingPlan` closure.
- [x] Macc and standalone reduction suffix semantics remain distinct in
      planning/provider output, generated target code, and tests.
- [x] Explicit and pre-realized selected-body target/header FileCheck coverage
      for both routes proves family-derived closure, materialized operands,
      mirrors, headers, accumulator/result roles, and scalar carry where
      relevant.
- [x] Negative fail-closed coverage still covers missing compare input, missing
      runtime scalar, missing produced mask, missing payload, missing
      accumulator, missing scalar output/result for reduction, missing
      runtime n/AVL, wrong compute suffix, unsupported predicate, bad
      dtype/config, stale route id, helper-string fallback, mirror-only
      authority, and common/export semantic inference where current test
      surfaces can express them.
- [x] Generated-bundle dry-runs pass for explicit and pre-realized
      `runtime_scalar_cmp_masked_macc_add` and
      `runtime_scalar_cmp_masked_standalone_reduce_add`, counts `7,16,23`, with
      existing value-distinguishing scalar thresholds.
- [x] Real `ssh rvv` evidence is rerun if emitted target sequence or
      ABI/materialized operand mapping changes.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, route-id, helper-string, artifact-name, mirror-only,
      or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No new mask algebra, select routes, memory routes, arithmetic/reduction
  operation kinds, predicate clones, dtype/LMUL clones, frontend lowering,
  source-front-door positive routes, dashboards, report-only work, helper-only
  cleanup, or broad all-route rewrite.
- No migration of select or memory routes except narrow regression repairs.
- No collapse of vector macc and horizontal scalar reduction semantics into a
  single compute operation.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`,
  descriptor, direct-C, source-front-door, helper-string, route-id,
  artifact-name, or mirror-only route authority.
- No common EmitC/export ownership of compare, mask, macc, reduction,
  accumulator, scalar output, dtype/config, runtime roles, result roles, or
  acceptance state.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test`.
3. Run focused FileCheck/lit coverage for explicit and pre-realized
   `runtime_scalar_cmp_masked_macc_add` and
   `runtime_scalar_cmp_masked_standalone_reduce_add` target/header fixtures
   plus directly relevant negative fail-closed fixtures.
4. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if route planning,
   provider helpers, or binding-plan helpers are changed.
5. Run generated-bundle dry-runs for explicit and pre-realized macc and
   standalone reduction with counts `7,16,23` and the existing scalar values
   used by their archived route tasks.
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
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/implementation-stack/index.md`

Relevant archived tasks read:

- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-computed-mask-memory-route-family-interface/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-select-route-family-interface/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-runtime-scalar-computed-mask-macc-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-runtime-scalar-computed-mask-horizontal-reduction-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-route-operand-abi-binding/prd.md`

Relevant workspace journal entries read:

- Session 161 computed-mask select route-family interface.
- Session 161 computed-mask memory route family.
- Session 159 runtime scalar computed-mask masked store.
- Session 159 closure-gated masked horizontal reduce-sum accumulation.

Initial source surfaces inspected:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- relevant runtime scalar computed-mask macc and standalone reduction tests and
  generated-bundle support as directly relevant anchors.

## Definition Of Done

Both active runtime-scalar computed-mask arithmetic accumulation production
routes consume a shared plugin-local accumulation route-family interface,
preserve existing behavior/evidence, and leave common EmitC/export as neutral
materialization only. If this grows beyond one coherent round, finish the
shared computed-mask producer and closure submodule for these two arithmetic
consumers and leave exact continuation state for route-specific suffix cleanup.

## Completion Evidence

Production changes:

- Added `RVVSelectedBodyRuntimeScalarComputedMaskAccumulationRouteFamilyPlan`
  as the shared RVV plugin-local route-family plan for the existing
  `runtime_scalar_cmp_masked_macc_add` and
  `runtime_scalar_cmp_masked_standalone_reduce_add` consumers.
- The shared plan now validates and carries the common runtime AVL/VL control,
  runtime ABI order, provider-supported mirror, required headers, C type
  mapping, SEW32/LMUL m1 vector and mask types, setvl/load/splat/compare/store
  leaves, mask role/source/memory form, accumulator/result contracts, and
  route-specific suffix kind.
- Migrated runtime scalar computed-mask macc away from one-off target/profile
  and runtime-control description population. It now consumes the shared
  accumulation plan for common compare/mask/runtime/header facts while keeping
  `tcrv_rvv.masked_macc` accumulator and result suffix facts distinct.
- Migrated runtime scalar computed-mask standalone reduction to consume the
  same accumulation family for common threshold compare/mask/runtime/header
  facts while preserving standalone-reduction scalar seed/result layout,
  inactive-lane zeroing, scalar carry, and reduction suffix facts.
- Made the RVV EmitC provider require the shared accumulation family plan
  before materializing either route, select plan-owned leaves/headers/types for
  the common path, and share operand binding for `cmp_lhs`, `rhs_scalar`, and
  `n` through one provider helper.
- Added target/header mirror fields for the accumulation family plan, compute
  suffix, accumulator contract, result contract, and scalar carry contract.
- Left select, memory, non-runtime computed-mask macc, non-runtime
  standalone reduction, widening contraction, source-front-door, common
  EmitC/export semantics, and generated-bundle script behavior unchanged.

Routes converted:

- `runtime_scalar_cmp_masked_macc_add`
- `runtime_scalar_cmp_masked_standalone_reduce_add`

Routes intentionally not converted:

- computed-mask select and memory route families, because they already have
  separate shared family plans and are outside this PRD.
- vector-vs-vector `computed_masked_macc_add` and
  `computed_mask_standalone_reduce_add`, because this PRD is bounded to
  runtime-scalar computed-mask arithmetic accumulation consumers.
- widening dot/contraction routes, because their source dtype and contraction
  suffix are separate Stage2 coverage.

Selected-body realization and binding closure evidence:

- Explicit and pre-realized macc/reduction lit fixtures now check
  `tcrv_rvv.accumulation_route_family_plan`, suffix, accumulator/result
  contracts, route operand binding mirrors, target leaf profile, and runtime
  control plan.
- `RVVRouteOperandBindingPlan` remains the materialized operand authority.
  Macc binds `cmp_lhs`, `rhs_scalar`, payload `lhs`, payload `rhs`, `acc`,
  `out`, and `n`; standalone reduction binds `cmp_lhs`, `rhs_scalar`, `src`,
  `acc`, `out`, and `n`.
- Mirror fields remain diagnostics only; provider materialization requires the
  shared family plan and still verifies binding closure before constructing the
  lowerable route.

Checks run:

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-cmp-masked-(macc|standalone)|runtime-scalar-computed-mask-(macc|standalone)'` from `build/test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/stage2_rvv_computed_mask_accumulation_route_family --run-id explicit --overwrite --op-kind runtime_scalar_cmp_masked_macc_add --op-kind runtime_scalar_cmp_masked_standalone_reduce_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_rvv_computed_mask_accumulation_route_family --run-id pre_realized --overwrite --op-kind runtime_scalar_cmp_masked_macc_add --op-kind runtime_scalar_cmp_masked_standalone_reduce_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/stage2_rvv_computed_mask_accumulation_route_family --run-id ssh_explicit --overwrite --op-kind runtime_scalar_cmp_masked_macc_add --op-kind runtime_scalar_cmp_masked_standalone_reduce_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91 --timeout 180 --connect-timeout 10`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_rvv_computed_mask_accumulation_route_family --run-id ssh_pre_realized --overwrite --op-kind runtime_scalar_cmp_masked_macc_add --op-kind runtime_scalar_cmp_masked_standalone_reduce_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91 --timeout 180 --connect-timeout 10`
- Active-authority scan over touched RVV/plugin/export/test diff for
  `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor/direct-C/source-front-door/source-artifact,
  helper-string, artifact-name, mirror-only, and common/export semantic
  authority.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`

Self-repair:

- Fixed initial compile failure from `llvm::StringRef` / `StringLiteral`
  ternary type mismatch by explicitly wrapping string literals.
- Fixed focused reduction lit failures caused by placing reduction-specific
  inactive-lane/source-memory contracts into generic description fields whose
  existing verifier expects them to remain empty. The reduction-specific facts
  now stay in accumulation-family mirror fields and standalone reduction
  fields.

Runtime evidence:

- Explicit `ssh rvv` generated-bundle run passed for both routes with counts
  `7,16,23` and `rhs_scalar=-37,91`.
- Pre-realized `ssh rvv` generated-bundle run passed for both routes with the
  same count/threshold set.
- Macc evidence proved active-lane `acc + lhs * rhs`, inactive accumulator
  preservation, add-only/mul-only distinguishing values, and tail preservation.
- Standalone reduction evidence proved scalar output for seeds `-11,17`, active
  lane accumulation, inactive-lane exclusion, runtime `n`/AVL variation, and
  tail preservation.

Spec-update judgment:

- No `.trellis/spec/**` update was made. This task applies existing RVV
  plugin-owned route-family, RouteOperandBindingPlan, typed body authority,
  and common EmitC neutrality contracts; it did not introduce a new durable
  architecture rule.
