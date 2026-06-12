# Stage2 RVV arithmetic and select operand binding adoption

## Goal

Adopt the RVV plugin-owned `RVVRouteOperandBindingPlan` contract for the
remaining active arithmetic, compare/select, and masked-arithmetic RVV routes
that are route-supported or executable today, so provider materialization and
artifact/header mirrors use one operand authority instead of ad-hoc role
lookup.

This round continues the route operand binding owner from the archived
`05-21-stage2-rvv-route-operand-abi-binding` and
`05-21-stage2-rvv-operand-binding-coverage-closure` tasks. It does not add new
Stage2 operation families.

## Direction Source

- Direction title: `Stage2 RVV arithmetic and select operand binding adoption`.
- Module owner: RVV plugin-owned `RouteOperandBindingPlan` adoption for the
  remaining ordinary arithmetic, compare/select, and masked-arithmetic routes
  that are active route-supported or executable today.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `a3e2adad rvv: extend operand binding coverage`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.

## What I Already Know

- Existing converted binding-plan routes at current HEAD:
  - `macc_add`
  - `strided_load_unit_store`
  - `unit_load_strided_store`
  - `scalar_broadcast_add`
  - `standalone_reduce_add`
  - `masked_unit_store`
  - `computed_masked_strided_store`
- The archived operand-binding coverage task intentionally left the following
  arithmetic / compare / select / masked-arithmetic routes unconverted:
  - `add`
  - `sub`
  - `mul`
  - `cmp_select`
  - `computed_mask_select`
  - `masked_add`
  - `reduce_add`
  - `strided_add`
- Current `verifyRVVSelectedBodyEmitCRouteDescription` requires binding-plan
  mirrors for only the converted routes and requires every other route to
  carry no binding plan.
- Provider materialization still defaults to direct `slice.*ABI` operands for
  ordinary binary, compare/select, `masked_add`, `reduce_add`, and
  `strided_add`.
- Specs require the route authority chain to remain:
  `tcrv.exec` envelope -> selected RVV variant -> typed low-level
  `tcrv_rvv` body -> RVV legality/realization/provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI roles only.
  RVV route operands must be validated and consumed by the RVV plugin from
  typed body/config/runtime facts, not from route IDs, helper strings,
  artifact names, source-front-door metadata, descriptors, or common EmitC.

## Scope

Convert this coherent cluster:

- `add`
- `sub`
- `mul`
- `cmp_select`
- `computed_mask_select`
- `masked_add`
- `reduce_add`
- `strided_add`

For every converted route, record logical operand, runtime ABI role, C
parameter, materialized expression/use, and mirror/header field in
`RVVRouteOperandBindingPlan`, and make provider emission use the contract for
the real emitted operands.

## Requirements

1. Inventory active arithmetic, compare/select, and masked-arithmetic routes
   and classify them as already converted, converted in this round, or
   intentionally deferred.
2. Add binding-plan IDs and logical-operand-to-runtime-role validation for the
   scoped routes.
3. Derive each scoped route's binding summary from route analysis after typed
   body/runtime ABI validation, and reject missing roles, duplicate roles,
   unsupported logical operands, wrong logical-operand roles, and mirror order
   mismatch.
4. Rewire provider emission so materialized setvl/control, loads, compare
   operands, select operands, masked arithmetic operands, reduction
   accumulator/output uses, strided load/store bases/strides, and stores are
   supplied through `RVVRouteOperandBindingPlan`.
5. Require route description mirrors and generated artifact/header metadata for
   converted routes to carry the same binding plan ID and summary.
6. Add positive structural checks proving converted route plans and
   materialized operands match the binding contract.
7. Add or extend negative fail-closed tests for expressible swap/mismatch cases
   in this cluster, especially lhs/rhs swaps, predicate/result swaps, missing
   or duplicate ABI roles, mirror/header mismatch, stale route-id authority,
   descriptor/direct-C/source-front-door authority, and common/export semantic
   inference.
8. Keep common EmitC/export neutral. Common code may consume provider-built
   route payloads and mirrors, but must not infer RVV roles, dtype, policy,
   intrinsic choices, or memory semantics.

## Acceptance Criteria

- [x] Current active route inventory is recorded in this PRD completion notes.
- [x] `add`, `sub`, and `mul` derive lhs/rhs/out/n materialized operands and
      mirrors from `RVVRouteOperandBindingPlan`.
- [x] `cmp_select` derives lhs/rhs/out/n, compare operands, select operands,
      store operands, and mirrors from `RVVRouteOperandBindingPlan`.
- [x] `computed_mask_select` derives cmp_lhs/cmp_rhs/true_value/false_value/
      out/n, compare operands, select operands, store operands, and mirrors
      from `RVVRouteOperandBindingPlan`.
- [x] `masked_add` derives lhs/rhs/out/n, compare operands, active arithmetic
      operands, masked merge passthrough/result operands, store operands, and
      mirrors from `RVVRouteOperandBindingPlan`.
- [x] `reduce_add` derives lhs/rhs/out/n, reduction call operands, reduction
      store VL/output, store operands, and mirrors from
      `RVVRouteOperandBindingPlan`.
- [x] `strided_add` derives lhs/rhs/out/n/lhs_stride/rhs_stride/out_stride,
      strided load/store operands, byte/element stride operands, and mirrors
      from `RVVRouteOperandBindingPlan`.
- [x] Positive structural checks prove converted routes carry binding plan IDs,
      summaries, and materialized operands from the contract.
- [x] Negative fail-closed checks cover role swaps, missing/duplicate roles,
      materialized-use mismatch, mirror/header mismatch, stale route-id
      authority, descriptor/direct-C/source-front-door authority, and
      common/export semantic inference where current surfaces can express
      those failures.
- [x] Generated-bundle dry-runs pass for representative converted routes using
      mixed signed values, predicates, tail sentinels, and strided operands.
- [x] Real `ssh rvv` PASS evidence exists for representative converted routes
      when runtime/correctness is claimed.
- [x] Active-authority scan shows no new positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public exact
      intrinsic route authority, route-id authority, artifact-name authority,
      or common/export RVV semantic authority.
- [x] Focused checks, `check-tianchenrv`, `git diff --check`, Trellis finish/
      archive, clean git status, and one coherent commit are completed if the
      task finishes.

## Non-Goals

- No new Stage2 operation families or route coverage.
- No dtype/LMUL clone batches.
- No reductions beyond existing `reduce_add` binding adoption.
- No indexed, segmented, widening, widening-contraction, or conversion cluster
  conversion in this round.
- No high-level Linalg/Vector/StableHLO/frontend lowering.
- No source-front-door positive RVV routes.
- No dashboards, report-only work, helper-only cleanup, or broad smoke tests as
  the main achievement.
- No descriptor-driven compute, direct-C/source-export route restoration, or
  compatibility wrapper preserving old route authority.
- No movement of RVV operand semantics into common EmitC/export, target
  metadata, artifact names, route IDs, descriptors, ABI strings alone, tests,
  or exact intrinsic spelling.

## Validation Plan

1. Validate and maintain this Trellis task.
2. Run focused C++ coverage for route operand binding plan validation.
3. Run focused lit/FileCheck coverage for converted route plan mirrors, target
   artifacts, generated headers, EmitC materialization, and negative
   fail-closed cases.
4. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   or generated-bundle expectations change.
5. Run generated-bundle dry-runs for representative converted routes in
   pre-realized selected-body mode, with explicit selected-body mode when the
   route has an existing explicit fixture.
6. Run real `ssh rvv` evidence for representative converted routes after
   dry-runs pass.
7. Run active-authority scans over touched active RVV include/lib/script/test
   paths.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Code Surface

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing target artifact and EmitC fixtures for arithmetic, compare/select,
  masked-add, reduction, and strided-add routes.

## Definition Of Done

- The scoped arithmetic/select cluster consumes `RVVRouteOperandBindingPlan`
  for provider materialization and mirrors.
- Previously converted routes remain intact.
- Deferred active routes are named with reason and continuation point.
- Focused checks, hardware evidence, active-authority scan, and
  `check-tianchenrv` are complete or any skip is justified by concrete
  environment failure.
- Trellis task status is truthful, archived when complete, and one coherent
  commit records the work.

## Completion Notes

### Active Route Inventory

- Already converted before this task:
  - `macc_add`
  - `strided_load_unit_store`
  - `unit_load_strided_store`
  - `scalar_broadcast_add`
  - `standalone_reduce_add`
  - `masked_unit_store`
  - `computed_masked_strided_store`
- Converted in this task:
  - `add`
  - `sub`
  - `mul`
  - `cmp_select`
  - `computed_mask_select`
  - `masked_add`
  - `reduce_add`
  - `strided_add`
- Intentionally not converted in this task:
  - Indexed, segmented, widening, contraction, conversion, and later Stage2
    clusters remain outside this PRD. They were not part of the active
    ordinary arithmetic / compare-select / masked-arithmetic cluster and would
    be a new coverage expansion rather than adoption of the current cluster.

### Production Owners Changed

- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - Added route operand binding plan IDs and role validation for the converted
    cluster.
  - Derived binding summaries from typed body/runtime ABI facts for binary,
    compare-select, computed-mask-select, masked-add, reduce-add, and
    strided-add routes.
  - Required the converted routes to carry binding-plan mirrors during route
    description verification.
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - Rewired materialized operands for the converted cluster through
    `RVVRouteOperandBindingPlan` before provider emission.
  - Added fail-closed binding/use checks for swapped roles, missing operands,
    missing materialized uses, and mismatched runtime-role consumption.
- `scripts/rvv_generated_bundle_abi_e2e.py`
  - Extended generated-bundle metadata expectations for the converted routes.
- Target artifact fixtures and RVV plugin tests were updated to prove plan
  mirrors, header mirrors, and fail-closed route-plan validation.

### Binding Contract Evidence

- `add`, `sub`, `mul`: `lhs`, `rhs`, `out`, and `n` are bound through one
  binary plan and consumed for load-base, binary call, store-base, setvl,
  loop-control, and header mirrors.
- `cmp_select`: `lhs` and `rhs` are bound once and consumed for compare inputs,
  select true/false values, output store, and header mirrors.
- `computed_mask_select`: `cmp_lhs`, `cmp_rhs`, `true_value`, `false_value`,
  `out`, and `n` are bound through one route plan and consumed for compare,
  select, store, setvl, and header mirrors.
- `masked_add`: `lhs` and `rhs` are bound once for compare, masked arithmetic,
  passthrough merge, store, and header mirrors.
- `reduce_add`: `lhs`, `rhs`, `out`, and `n` are bound for reduction input,
  accumulator seed, result store, setvl, and header mirrors.
- `strided_add`: `lhs`, `rhs`, `out`, `n`, `lhs_stride`, `rhs_stride`, and
  `out_stride` are bound for strided address calculation, materialized loads,
  binary operation, store, setvl, and header mirrors.

### Validation Completed

- Focused artifact checks:
  - Ran `tcrv-opt` + FileCheck `PLAN` for all converted explicit and
    pre-realized route fixtures.
  - Ran `tcrv-opt` + `tcrv-translate --tcrv-export-target-header-artifact` +
    FileCheck `HEADER` for the same fixtures.
  - Ran `tcrv-opt --tcrv-materialize-selected-lowering-boundaries` +
    FileCheck `REALIZED` for fixtures that declare realization checks.
- Script checks:
  - `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- C++ focused check:
  - `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  - `build/bin/tianchenrv-rvv-extension-plugin-test`
- Generated-bundle dry-run:
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --dry-run --overwrite --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-arithmetic-select-binding-dry-run --op-kind add --op-kind sub --op-kind mul --op-kind cmp_select --op-kind computed_mask_select --op-kind masked_add --op-kind reduce_add --op-kind strided_add --runtime-count 7 --runtime-count 16 --runtime-count 23`
  - Result: `dry_run_success`
  - Artifact: `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-arithmetic-select-binding-dry-run/20260521T105331Z`
- Real RVV hardware evidence:
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --overwrite --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-arithmetic-select-binding-ssh --op-kind add --op-kind sub --op-kind mul --op-kind cmp_select --op-kind computed_mask_select --op-kind masked_add --op-kind reduce_add --op-kind strided_add --runtime-count 7 --runtime-count 16 --runtime-count 23`
  - Result: `success`
  - Artifact: `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-arithmetic-select-binding-ssh/20260521T105347Z`
  - Remote PASS markers: `add`, `sub`, `mul`, `cmp_select`,
    `computed_mask_select`, `masked_add`, `reduce_add`, and `strided_add`.
- Active-authority scan:
  - `git diff -U0` showed no new additions matching `RVVI32M1`,
    `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
    `source-front-door`, `source_front`, `descriptor-driven`, `direct-C`,
    `direct_c`, or exact `__riscv_*_i32m1` authority strings.
  - No common/export semantic owner paths were touched.
  - Existing legacy/intrinsic hits in touched test and route-planning files
    remain pre-existing fail-closed checks, residue detectors, or current
    intrinsic spelling mirrors, not new positive authority from this task.
- Global checks:
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2`
  - Result: 261/261 lit tests passed.
