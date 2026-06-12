# Stage2 RVV dual runtime-scalar mask-and-select artifact ABI boundary

## Goal

Carry exactly one existing `runtime_scalar_dual_cmp_mask_and_select` selected
body from a selected `tcrv.exec` RVV variant through RVV plugin-local
selected-body realization, provider-owned compare/select and mask-composition
facts, `TCRVEmitCLowerableRoute`, common EmitC materialization, RVV target
artifact bundle generation, and real `ssh rvv` correctness evidence.

This round follows the completed computed-mask compare/select artifact ABI
boundary. It exercises the next structured compare/select bottleneck: two
runtime scalar thresholds are imported through explicit ABI bindings, splatted
inside the same VL scope, compared against two vector inputs, composed through
`tcrv_rvv.mask_and`, and used to select between true/false value vectors before
storing to output memory.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief as
  `.trellis/tasks/06-01-06-01-stage2-rvv-dual-runtime-scalar-mask-select-artifact-abi`.
* The repository started clean on `main` at
  `4c10e0ce rvv: prove computed-mask compare-select artifact abi`.
* The immediately preceding archived task
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-computed-mask-cmp-select-artifact-abi/`
  proved `computed_mask_select` through selected-body realization, provider
  route facts, common EmitC, generated RVV artifact ABI execution, two
  compare-data patterns, and real `ssh rvv` correctness.
* `.trellis/spec/index.md` requires the RVV authority chain to remain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body ->
  RVV plugin legality/realization/provider -> common EmitC -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` explicitly includes
  `runtime_scalar_dual_cmp_mask_and_select` in the elementwise/compare-select
  selected-body realization boundary and provider-facts preflight.
* `.trellis/spec/lowering-runtime/emitc-route.md` forbids common EmitC/export
  from inferring runtime thresholds, predicate composition, true/false roles,
  dtype/config, policy, route support, or ABI semantics from route ids,
  artifact names, manifests, descriptors, emission-plan mirrors, C strings, or
  intrinsic spellings.
* `.trellis/spec/testing/mlir-testing-contract.md` requires generated-bundle
  compare/select evidence to expose provider-derived predicate, mask source,
  true/false value operands, materialized body, emitted C/C++ structure, route
  metadata mirrors, and real `ssh rvv` runtime evidence when correctness is
  claimed.
* Initial code search shows the base production/evidence surfaces already
  exist for this exact body:
  `TypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOp`,
  `RVVElementwiseSelectedBodyRealizationOwner`,
  `RVVEmitCCompareSelectStatementPlanOwners`,
  `RVVEmitCRoutePlanning`, `RVVEmitCRouteProvider`, common EmitC
  materialization, RVV target artifact validation/support bundle logic,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and focused Target/Script tests.

## Requirements

* Use exactly one selected body for this round:
  `runtime_scalar_dual_cmp_mask_and_select`.
* The positive path must start from the selected-boundary pre-realized body and
  realize before route construction:

  ```text
  selected tcrv.exec RVV variant
    -> typed pre-realized runtime_scalar_dual_cmp_mask_and_select body
    -> RVV elementwise/compare-select realization owner
    -> realized tcrv_rvv setvl/with_vl/load/splat/load/splat/load/load/compare/compare/mask_and/select/store body
    -> computed-mask select family plan
    -> operand-binding facts and compare/select statement plan
    -> provider preflight
    -> TCRVEmitCLowerableRoute
    -> common EmitC materializer
    -> RVV target artifact bundle
    -> external callable ABI consumer
    -> ssh rvv evidence
  ```

* Runtime `n`/AVL, both runtime scalar thresholds, primary compare predicate,
  secondary compare predicate, mask-and composition, true input, false input,
  output memory, dtype/config, SEW, LMUL, and tail/mask policy must come from
  the typed selected `tcrv_rvv` body and provider-validated facts.
* Provider/export evidence may mirror route metadata after provider route
  construction, but route ids, artifact names, manifests, test names,
  descriptors, emission-plan fields, C strings, exact intrinsic spellings, and
  common EmitC code must not become threshold, predicate-composition,
  true/false role, dtype/config, policy, ABI, or route authority.
* The generated bundle harness must check:
  `out[i] = true_value[i]` only when both runtime scalar comparisons are true,
  `out[i] = false_value[i]` otherwise, runtime `n = 0` skips writes, output
  tail storage beyond `n` remains sentinel preserved, and aggregate cases cover
  primary-mask true/false, secondary-mask true/false, composed-mask true/false,
  single-mask-only lanes, and true/false payload selection.
* Real `ssh rvv` correctness evidence must cover representative counts
  `0,1,16,23,257` and at least two threshold-pair patterns. The default
  focused evidence should use threshold A values `-37,91` and threshold B
  values `-37,91`, yielding four runtime scalar pairs without adding a broad
  dtype/LMUL matrix.
* Direct pre-realized route-entry or artifact export for this body must fail
  closed before provider route construction/export unless production evidence
  shows a spec-backed direct route-entry API was intentionally reintroduced.
* Keep changes bounded to the single production/export/runtime boundary needed
  for the base dual runtime-scalar mask-and-select evidence. If the existing
  path already expresses the ABI faithfully, prove it and add only the missing
  focused fail-closed guard, harness repair, or evidence pin.

## Acceptance Criteria

* [x] Trellis PRD/context files validate successfully and the task is started.
* [x] Focused evidence shows `runtime_scalar_dual_cmp_mask_and_select` is
  realized by the RVV plugin selected-body owner before route construction.
* [x] Focused provider/export evidence shows runtime `n`, both runtime scalar
  thresholds, primary compare, secondary compare, mask-and composition, true
  input, false input, output memory, dtype/config, and tail/mask policy survive
  into the generated bundle.
* [x] Generated-bundle dry-run for `--pre-realized-selected-body --op-kind
  runtime_scalar_dual_cmp_mask_and_select` passes for counts `0,1,16,23,257`
  and two-by-two runtime scalar threshold pairs.
* [x] Direct pre-realized route-entry mode for
  `runtime_scalar_dual_cmp_mask_and_select` fails closed with the selected
  lowering-boundary diagnostic before route/provider export.
* [x] Real `ssh rvv` compile/run correctness evidence passes for counts
  `0,1,16,23,257` and at least two threshold-pair patterns, with aggregate
  mask and select payload coverage.
* [x] The generated harness checks true/false select behavior, mask-and
  intersection behavior, single-mask-only lanes, and destination tail sentinel
  preservation beyond runtime `n`.
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

* The executable compiler path for the base
  `runtime_scalar_dual_cmp_mask_and_select` selected body is proven through
  selected-body realization, generated RVV target artifact bundle export, and
  external ABI execution on `ssh rvv`.
* Runtime correctness claims are backed by real `ssh rvv` output.
* No broad compare/select matrix, dtype/LMUL clone batch, high-level frontend
  authority, source-front-door path, descriptor compute path, common EmitC
  semantic inference, performance claim, or broad smoke matrix is added.
* Task status, workspace journal, archived task record, final commit, and final
  report truthfully describe what was completed and what was not.

## Out of Scope

* `runtime_scalar_dual_cmp_mask_and_select_i64`,
  `runtime_scalar_dual_cmp_mask_and_select_lmul_m2`, other dtype/LMUL clone
  expansion, computed-mask clone expansion, masked memory, reduction, MAcc,
  contraction, conversion, or high-level Linalg/frontend authority.
* New descriptor-driven computation, direct-C/source-export route support,
  source-front-door positive routes, dashboards, tuning databases, readiness
  state machines, or performance claims.
* Treating route ids, artifact names, manifests, exact intrinsic spellings,
  status fields, emission-plan metadata, scripts, or test names as threshold,
  predicate-composition, true/false role, dtype, policy, ABI, or route
  authority.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-computed-mask-cmp-select-artifact-abi/`.
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

* The selected body for this round was exactly the base
  `runtime_scalar_dual_cmp_mask_and_select`.
* The existing production C++ chain already carried the selected body through
  RVV plugin-local realization, dual runtime-scalar computed-mask select
  route-family facts, operand-binding facts, compare/select statement-plan
  facts, provider preflight, `TCRVEmitCLowerableRoute`, common EmitC, and RVV
  target artifact bundle export. No C++ owner/provider/materializer/target code
  change was needed.
* The evidence gap was script-local: the generated-bundle evidence already
  enumerated four threshold pairs, but it did not explicitly pin a required
  minimum threshold-pair contract in JSON/self-test for the dual
  runtime-scalar path.
* `compare_select_predicate_boundary`, per-op evidence, root evidence, and the
  harness summary now expose `*_threshold_pairs_required_minimum = 2` for
  `runtime_scalar_dual_cmp_mask_and_select`.
* `--self-test` now explicitly pins the dual runtime-scalar single-threshold
  failure, the retired direct pre-realized
  `runtime_scalar_dual_cmp_mask_and_select` route-entry diagnostic, and the
  generated harness aggregate mask/mask-and/select-payload checks.
* Final dry-run evidence:
  `artifacts/tmp/06-01-stage2-rvv-dual-runtime-scalar-mask-select-artifact-abi/focused-dry-run-v1/20260601T153721Z`.
  It records `input_mode = pre-realized-selected-body`,
  `pre_realized_body_consumed = true`, runtime ABI order
  `cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n`,
  primary and secondary compare predicate `sle`, composed mask source
  `mask-and-of-two-runtime-scalar-compare-produced-masks`, mask composition
  `and`, selected value roles, and threshold-pair required minimum `2`.
* Direct pre-realized route-entry negative evidence:
  `artifacts/tmp/06-01-stage2-rvv-dual-runtime-scalar-mask-select-artifact-abi/focused-direct-fail-v2/direct-pre-realized-runtime-scalar-dual-cmp-mask-and-select`.
  The command exited through FileCheck with the expected retired direct
  route-entry diagnostic.
* Real `ssh rvv` evidence:
  `artifacts/tmp/06-01-stage2-rvv-dual-runtime-scalar-mask-select-artifact-abi/final-ssh-rvv-v1/20260601T153921Z`.
  Counts `0,1,16,23,257` passed for threshold pairs
  `(-37,-37)`, `(-37,91)`, `(91,-37)`, and `(91,91)`. Multi-lane cases reported
  primary-mask, secondary-mask, composed-mask, and single-mask-only coverage,
  and all cases preserved the tail sentinel. Final remote output reported
  `tcrv_rvv_generated_bundle_abi_runtime_scalar_dual_cmp_mask_and_select_ok counts=0,1,16,23,257 rhs_scalar_a_values=-37,91 rhs_scalar_b_values=-37,91`.
* Bounded old-authority scan:
  added tracked diff lines contain no hits for `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`, `__riscv_.*_i32m1`,
  `source-front-door`, `source-artifact`, `emission_plan`, `descriptor`, or
  `selected route`. Full-file hits in the task PRD are guardrail/non-goal
  references. Existing exact intrinsic spelling hits in the script/test remain
  provider-derived emitted-C evidence checks, not route authority.
* Spec update decision: no `.trellis/spec/` update was needed. Existing RVV
  plugin, EmitC route, variant pipeline, and MLIR testing contracts already
  describe this boundary; this round made the dual runtime-scalar generated
  bundle evidence stricter and durable.

## Checks

* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-06-01-stage2-rvv-dual-runtime-scalar-mask-select-artifact-abi`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind runtime_scalar_dual_cmp_mask_and_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --artifact-root artifacts/tmp/06-01-stage2-rvv-dual-runtime-scalar-mask-select-artifact-abi/focused-dry-run-v1 --overwrite`
* [x] Manual FileCheck equivalents for
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-dual-cmp-mask-and-select-dry-run.test`
  prefixes `ROOT`, `RSD`, and `HARNESS`.
* [x] Direct route-entry negative command matched
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-dual-cmp-mask-and-select-fail-closed.test`.
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind runtime_scalar_dual_cmp_mask_and_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --artifact-root artifacts/tmp/06-01-stage2-rvv-dual-runtime-scalar-mask-select-artifact-abi/final-ssh-rvv-v1 --overwrite`
* [x] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* [x] `rtk proxy bash -lc 'artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir --tcrv-materialize-selected-lowering-boundaries | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir --check-prefix=REALIZED'`
* [x] `rtk proxy bash -lc 'artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir --check-prefix=PLAN'`
* [x] `rtk proxy bash -lc 'artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | artifacts/tmp/tianchenrv-build/bin/tcrv-translate --tcrv-export-target-header-artifact | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir --check-prefix=HEADER'`
* [x] Bounded old-authority scan over added tracked diff lines and touched task
  PRD/script/test files.
* [x] `rtk git diff --check`
