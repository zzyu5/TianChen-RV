# Stage2 RVV Runtime Scalar Compare/Select Executable Boundary

## Goal

Close the selected-boundary executable boundary for
`runtime_scalar_cmp_select`. The runtime `rhs_scalar` operand must flow from a
selected `tcrv.exec` RVV variant into typed `tcrv_rvv` splat/compare/select
structure, then through RVV provider facts, `TCRVEmitCLowerableRoute`, neutral
EmitC, target artifact ABI/header/object export, generated RVV C harness, and
real `ssh rvv` correctness evidence.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current branch is `main`; the working tree was clean before task creation.
- Latest commit before task creation is
  `3919dae2 rvv: close compare select executable artifacts`.
- No `.trellis/.current-task` existed when this task was created.
- The archived task `05-28-stage2-rvv-compare-select-executable-artifacts`
  closed plain and computed-mask compare/select executable evidence and kept
  direct pre-realized route-entry shortcuts fail-closed.
- Existing production files already expose a runtime-scalar compare/select
  path that must be validated rather than assumed:
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` validates
    `typed_runtime_scalar_compare_select_pre_realized_body`, requires
    `lhs,rhs_scalar,true_value,false_value,out,n`, derives runtime AVL/VL, and
    realizes `setvl`, `with_vl`, `load`, `splat`, `compare`, `select`, and
    `store`.
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` classifies
    `RuntimeScalarCompareSelect` as a compare/select statement-plan consumer,
    requires runtime-scalar operand-binding facts, consumes the computed-mask
    select family plan, route-control plan, mask/tail policy plan,
    materialization facts, RHS scalar splat leaf, and ABI order before
    constructing provider-ready route statements.
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp` validates selected-body route
    metadata mirrors against the rebuilt provider route before target artifact
    export.
  - `test/Plugin/RVVExtensionPluginTest.cpp` already covers provider facts for
    runtime-scalar compare/select, including runtime scalar producer facts and
    stale producer-source rejection.
  - `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-select.mlir`
    and `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-select.mlir`
    cover realized typed dataflow and artifact/header mirrors.
  - `scripts/rvv_generated_bundle_abi_e2e.py` already has
    `runtime_scalar_cmp_select` expectations and an external C ABI harness path
    that accepts explicit `--rhs-scalar` values.

## Requirements

- Use the public selected lowering-boundary path for the pre-realized
  `runtime_scalar_cmp_select` body.
- Preserve direct pre-realized route-entry unsupported behavior for
  `runtime_scalar_cmp_select`; do not turn the direct route-entry shortcut into
  positive authority.
- Preserve and validate:
  - operation kind `runtime_scalar_cmp_select`;
  - `rhs_scalar` as a runtime ABI scalar value, not a vector buffer or ABI-name
    inference;
  - typed RHS scalar `tcrv_rvv.splat` inside the selected RVV body;
  - predicate kind, mask source, mask role, mask memory form, select layout,
    lhs/true/false/out buffer roles, runtime `n`/AVL/VL, setvl placement,
    policy, SEW/LMUL, route facts, and ABI order;
  - provider route facts and generated target artifact mirrors as mirrors of
    typed/provider facts only.
- If production code is already sufficient, prove it with focused production
  validation and executable evidence rather than adding a helper-only change.
- If validation reveals a production gap, fix the owning production boundary:
  selected-body realization, provider planning, target artifact ABI validation,
  or generated RVV C harness logic as appropriate.

## Acceptance Criteria

- [x] The PRD accurately records the module goal, boundaries, non-goals, and
      production owner proof or production fix.
- [x] Focused selected-boundary generated-bundle dry-run passes for pre-realized
      `runtime_scalar_cmp_select` with `route_entry_realization=false` and
      `pre_realized_body_consumed=true`.
- [x] Dry-run evidence checks typed runtime scalar splat/compare/select facts,
      provider route facts, runtime ABI order
      `lhs,rhs_scalar,true_value,false_value,out,n`, target artifact ABI/header
      facts, and absence of descriptor/direct-C/source-front-door authority.
- [x] Direct pre-realized route-entry for `runtime_scalar_cmp_select` remains
      fail-closed with an op-specific selected-boundary-only diagnostic.
- [x] `ssh rvv` generated-bundle compile/run/correctness passes for counts
      including `0`, `1`, exact-vector, tail, and stress cases.
- [x] Runtime evidence uses at least two `rhs_scalar` values and covers
      predicate true, predicate false, equality/boundary behavior, signed
      negative behavior, and tail preservation.
- [x] Non-regression for completed compare/select selected-boundary artifacts
      passes.
- [x] Focused C++/lit/script checks pass for the changed or validated boundary,
      including provider route planning and target artifact ABI consumers.
- [x] Bounded touched-file authority scan finds no central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      executable authority.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or any blocker is exact and recorded.
- [x] Final `git status --short` is clean after the coherent commit.

## Definition Of Done

- Production path is either fixed or explicitly proven to own the runtime scalar
  compare/select selected-boundary behavior.
- Generated-bundle dry-run and `ssh rvv` executable evidence are recorded.
- Relevant tests and bounded scans are run.
- Trellis task status is truthful and archived only after acceptance is met.
- One coherent commit records the round.

## Completion Evidence

- Production owner result: no C++ production owner changes were required.
  Existing RVV selected-body realization, route planning, route provider, and
  target artifact ABI code already carried the selected-boundary
  `runtime_scalar_cmp_select` facts. This round tightened the generated-bundle
  evidence script and tests so the proof is explicit instead of implicit.
- Implemented evidence tightening in `scripts/rvv_generated_bundle_abi_e2e.py`:
  runtime scalar compare/select now requires two RHS scalar values, extracts
  materialized `tcrv_rvv.splat`/`compare`/`select` facts, extracts emitted RVV
  C++ scalar-splat/compare/select facts, includes runtime scalar producer facts
  in `compare_select_predicate_boundary`, and preserves direct route-entry
  fail-closed behavior.
- Added focused lit coverage:
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-select-dry-run.test`
  and
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-select-fail-closed.test`.
- Generated-bundle dry-run:
  `artifacts/tmp/stage2_runtime_scalar_cmp_select_closure/runtime-scalar-cmp-select-dry-run`.
- `ssh rvv` executable evidence:
  `artifacts/tmp/stage2_runtime_scalar_cmp_select_closure/runtime-scalar-cmp-select-ssh`;
  counts `0,1,16,23,257`; RHS scalars `-37,91`; remote compile/run passed;
  pass marker `tcrv_rvv_generated_bundle_abi_runtime_scalar_cmp_select_ok`.
- Compare/select non-regression dry-run:
  `artifacts/tmp/stage2_runtime_scalar_cmp_select_closure/compare-select-non-regression-dry-run`.
- Checks passed: `python3 -m py_compile`, script `--self-test`, focused
  runtime-scalar lit filter, RVV extension and selected-lowering-boundary C++
  tests, direct fail-closed dry-run, `git diff --check`, bounded authority scan,
  and final serial `check-tianchenrv` with 407/407 tests passed.

## Out Of Scope

- `runtime_scalar_dual_cmp_mask_and_select`.
- `runtime_scalar_cmp_masked_store`.
- `runtime_scalar_cmp_masked_load_store`.
- `runtime_scalar_cmp_masked_standalone_reduce_add`.
- Segment2, standalone reduction, widening conversion, new dtype/LMUL clone
  batches, high-level Linalg/frontend lowering, one-intrinsic wrapper dialects,
  broad selected-body framework rewrites, dashboards, broad smoke matrices, or
  another proof-only plain compare/select round.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
- Relevant prior task:
  - `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-compare-select-executable-artifacts/prd.md`
- Relevant files:
  - `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-select.mlir`
  - `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-select.mlir`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
