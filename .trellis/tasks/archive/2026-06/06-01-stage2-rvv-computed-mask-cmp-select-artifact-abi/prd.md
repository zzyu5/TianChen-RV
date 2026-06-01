# Stage2 RVV computed-mask compare-select artifact ABI boundary

## Goal

Carry exactly one existing `computed_mask_select` selected body from a selected
`tcrv.exec` RVV variant through RVV plugin-local selected-body realization,
provider-owned compare/select route facts, `TCRVEmitCLowerableRoute`, common
EmitC materialization, RVV target artifact bundle generation, and real
`ssh rvv` correctness evidence.

This task follows the completed runtime-scalar compare/select ABI boundary and
closes the adjacent computed-mask form: compare operands are vector inputs,
the predicate mask is produced by the realized typed body in the same VL scope,
and true/false value roles plus output memory must survive structurally into
provider/export evidence and generated artifact execution.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief as
  `.trellis/tasks/06-01-stage2-rvv-computed-mask-cmp-select-artifact-abi`.
* The repository started clean on `main` at
  `5cf367a1 rvv: prove runtime-scalar compare-select artifact abi`.
* The immediately preceding archived task
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi/`
  proved `runtime_scalar_cmp_select` through selected-body realization,
  provider route facts, common EmitC, generated RVV artifact ABI execution, and
  real `ssh rvv` correctness.
* `.trellis/spec/index.md` requires the RVV authority chain to remain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body ->
  RVV plugin legality/realization/provider -> common EmitC -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` explicitly includes
  `computed_mask_select` in the elementwise/compare-select selected-body
  realization owner, provider-facts preflight, and compare/select statement-plan
  boundaries.
* `.trellis/spec/lowering-runtime/emitc-route.md` forbids common EmitC/export
  from inferring RVV predicate, dtype/config, true/false roles, policy, route
  support, or ABI semantics from route ids, artifact names, manifests,
  descriptors, emission-plan mirrors, C strings, or intrinsic spellings.
* `.trellis/spec/testing/mlir-testing-contract.md` requires generated-bundle
  compare/select evidence to expose provider-derived predicate, mask source,
  true/false value operands, materialized body, emitted C/C++ structure,
  route metadata mirrors, and real `ssh rvv` runtime evidence when correctness
  is claimed.
* Initial code search shows the relevant production surfaces already exist:
  `typed_computed_mask_select_pre_realized_body`,
  `RVVElementwiseSelectedBodyRealizationOwner`,
  `RVVEmitCCompareSelectStatementPlanOwners`,
  `RVVEmitCRoutePlanning`, `RVVEmitCRouteProvider`, common EmitC
  materialization, RVV target artifact validation/support bundle logic, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.

## Requirements

* Use exactly one selected body for this round: `computed_mask_select`.
* The positive path must start from the selected-boundary pre-realized body and
  realize before route construction:

  ```text
  selected tcrv.exec RVV variant
    -> typed pre-realized computed_mask_select body
    -> RVV elementwise/compare-select realization owner
    -> realized tcrv_rvv setvl/with_vl/load/load/load/load/compare/select/store body
    -> computed-mask select family plan
    -> operand-binding facts and compare/select statement plan
    -> provider preflight
    -> TCRVEmitCLowerableRoute
    -> common EmitC materializer
    -> RVV target artifact bundle
    -> external callable ABI consumer
    -> ssh rvv evidence
  ```

* Compare LHS/RHS inputs, predicate kind, predicate/mask source, mask role,
  mask memory form, true input, false input, output memory, runtime `n`/AVL,
  dtype/config, SEW, LMUL, and tail/mask policy must come from the typed
  selected `tcrv_rvv` body and provider-validated facts.
* Provider/export evidence may mirror route metadata after provider route
  construction, but route ids, artifact names, manifests, test names,
  descriptors, emission-plan fields, C strings, exact intrinsic spellings, and
  common EmitC code must not become compare predicate, true/false role,
  dtype/config, policy, ABI, or route authority.
* The generated bundle harness must check `out[i] = true_value[i]` when the
  vector compare predicate is true, `out[i] = false_value[i]` otherwise,
  runtime `n = 0` skips writes, and storage beyond `n` remains sentinel
  preserved.
* Real `ssh rvv` correctness evidence must cover representative counts
  `0,1,16,17,257` and at least two compare-data patterns, including lanes where
  the predicate is true and lanes where it is false.
* Direct pre-realized route-entry or artifact export for this body must fail
  closed before provider route construction/export unless production evidence
  shows a spec-backed direct route-entry API was intentionally reintroduced.
* Keep changes bounded to the single production/export/runtime boundary needed
  for computed-mask compare/select evidence. If the existing path already
  expresses the ABI faithfully, prove it and add only the missing focused
  guard, harness repair, or test pin.

## Acceptance Criteria

* [x] Trellis PRD/context files validate successfully and the task is started.
* [x] Focused evidence shows `computed_mask_select` is realized by the RVV
  plugin selected-body owner before route construction.
* [x] Focused provider/export evidence shows compare input roles, predicate
  kind, predicate/mask source and role, true input, false input, output memory,
  runtime `n`/AVL, dtype/config, and tail/mask policy survive into the
  generated bundle.
* [x] Generated-bundle dry-run for `--pre-realized-selected-body --op-kind
  computed_mask_select` passes for counts `0,1,16,17,257`.
* [x] Direct pre-realized route-entry mode for `computed_mask_select` fails
  closed with the selected lowering-boundary diagnostic before route/provider
  export.
* [x] Real `ssh rvv` compile/run correctness evidence passes for counts
  `0,1,16,17,257` with at least two compare-data patterns and both predicate
  true/false lanes.
* [x] The generated harness checks true/false select behavior and destination
  tail sentinel preservation beyond runtime `n`.
* [x] Focused lit/FileCheck, C++ target/plugin tests, or script self-tests
  covering the touched path pass.
* [x] A bounded old-authority scan over touched plugin/provider/materializer/
  target/script/test/spec/task files classifies any remaining hits for
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, `descriptor`, and `selected route`.
* [x] `git diff --check` passes.
* [x] The task is finished/archived and one coherent commit is created, or the
  exact blocker and continuation point are recorded.

## Definition of Done

* The executable compiler path for `computed_mask_select` is proven through
  selected-body realization, generated RVV target artifact bundle export, and
  external ABI execution on `ssh rvv`.
* Runtime correctness claims are backed by real `ssh rvv` output.
* No broad compare/select matrix, runtime-scalar clone batch, dtype/LMUL clone
  batch, high-level frontend authority, source-front-door path, descriptor
  compute path, common EmitC semantic inference, performance claim, or broad
  smoke matrix is added.
* Task status, workspace journal, archived task record, final commit, and final
  report truthfully describe what was completed and what was not.

## Out of Scope

* `runtime_scalar_cmp_select`, `runtime_scalar_dual_cmp_mask_and_select`,
  dtype/LMUL clone expansion, masked memory, reduction, MAcc, contraction,
  conversion, or high-level Linalg/frontend authority.
* New descriptor-driven computation, direct-C/source-export route support,
  source-front-door positive routes, dashboards, tuning databases, readiness
  state machines, or performance claims.
* Treating route ids, artifact names, manifests, exact intrinsic spellings,
  status fields, emission-plan metadata, scripts, or test names as predicate,
  true/false role, dtype, policy, ABI, or route authority.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi/`.
* Inspected/search targets:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and focused tests.

## Completion Notes

* The selected body for this round was exactly `computed_mask_select`.
* The existing production C++ chain already carried the selected body through
  RVV plugin-local realization, computed-mask select route-family facts,
  operand-binding facts, compare/select statement-plan facts, provider
  preflight, `TCRVEmitCLowerableRoute`, common EmitC, and target artifact
  bundle export. No C++ owner/provider/materializer/target code change was
  needed.
* The evidence gap was in generated-bundle runtime coverage and script-local
  guardrails: the computed-mask select harness previously used one compare-data
  initializer pattern. It now runs two compare-data patterns for every runtime
  count and records `compare_data_patterns=2` in real `ssh rvv` output.
* `compare_select_predicate_boundary` now records
  `computed_mask_select_compare_data_patterns` and
  `compare_data_patterns_required_minimum = 2` for computed-mask select
  evidence.
* `--self-test` now explicitly pins the retired direct pre-realized
  `computed_mask_select` route-entry diagnostic, including the requirement to
  use the public selected lowering-boundary producer before target bundle
  export.
* Final dry-run evidence:
  `artifacts/tmp/06-01-stage2-rvv-computed-mask-cmp-select-artifact-abi/final-dry-run-v3/20260601T151556Z`.
  It records `input_mode = pre-realized-selected-body`,
  `pre_realized_body_consumed = true`, runtime ABI order
  `cmp_lhs,cmp_rhs,true_value,false_value,out,n`, predicate source
  `compare-produced-mask-same-vl-scope`, selected value operands
  `cmp_lhs`, `cmp_rhs`, `true_value`, `false_value`, `out`, and
  `compare_data_patterns_required_minimum = 2`.
* Direct pre-realized route-entry negative evidence:
  `artifacts/tmp/06-01-stage2-rvv-computed-mask-cmp-select-artifact-abi/final-direct-fail-v2/20260601T151655Z`.
  The command exited 1 with the expected retired direct route-entry diagnostic
  for `computed_mask_select`.
* Real `ssh rvv` evidence:
  `artifacts/tmp/06-01-stage2-rvv-computed-mask-cmp-select-artifact-abi/final-ssh-rvv-v1/20260601T151638Z`.
  Counts `0,1,16,17,257` passed for compare patterns `0` and `1`. Multi-lane
  cases reported both true and false select lanes, and all cases preserved the
  tail sentinel. Final remote output reported
  `tcrv_rvv_generated_bundle_abi_computed_mask_select_ok counts=0,1,16,17,257 compare_data_patterns=2`.
* Bounded old-authority scan:
  added lines contain no hits for `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`, `__riscv_.*_i32m1`,
  `source-front-door`, `source-artifact`, `emission_plan`, `descriptor`, or
  `selected route`. Existing context hits for exact `__riscv_*_i32m1`
  intrinsic spellings remain provider-derived FileCheck evidence, not route
  authority.
* Spec update decision: no `.trellis/spec/` update was needed. The existing
  RVV plugin, EmitC route, and MLIR testing contracts already described this
  boundary; this round made the computed-mask compare/select generated-bundle
  evidence stricter and durable.

## Checks

* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-stage2-rvv-computed-mask-cmp-select-artifact-abi`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind computed_mask_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-computed-mask-cmp-select-artifact-abi/final-dry-run-v3 --overwrite`
* [x] Direct route-entry negative command exited 1 with the expected retired
  direct route-entry diagnostic for `computed_mask_select`.
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_mask_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-computed-mask-cmp-select-artifact-abi/final-ssh-rvv-v1 --overwrite`
* [x] Manual FileCheck equivalents for
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-mask-select-dry-run.test`
  prefixes `STDOUT`, `ROOT`, `CMS`, `SLE`, `HARNESS`, and `HARNESS-SLE`.
* [x] Manual FileCheck equivalent for
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-mask-select-unsupported.test`
  prefix `CMS`.
* [x] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* [x] `rtk sh -lc 'artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select.mlir --tcrv-materialize-selected-lowering-boundaries | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select.mlir --check-prefix=REALIZED'`
* [x] `rtk sh -lc 'artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select.mlir --check-prefix=PLAN'`
* [x] `rtk sh -lc 'artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | artifacts/tmp/tianchenrv-build/bin/tcrv-translate --tcrv-export-target-header-artifact | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select.mlir --check-prefix=HEADER'`
* [x] Bounded old-authority scan over added lines in touched script/test/task
  files.
* [x] `rtk git diff --check`
