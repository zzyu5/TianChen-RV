# Stage2 RVV runtime-scalar compare-select artifact ABI boundary

## Goal

Carry exactly one existing `runtime_scalar_cmp_select` selected body from a
selected `tcrv.exec` RVV variant through RVV plugin-local selected-body
realization, provider-owned route facts, common EmitC materialization, RVV
target artifact bundle generation, and real `ssh rvv` correctness evidence.

The bounded case is `runtime_scalar_cmp_select`, because it exercises runtime
scalar threshold binding, runtime `n`/AVL, true-value and false-value memory
roles, compare predicate and mask facts, select layout, dtype/config, and the
generated callable artifact ABI without relying on common/export semantic
inference.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief as
  `.trellis/tasks/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi`.
* The repository started clean on `main` at
  `92a11502 rvv: prove runtime-scalar masked memory artifact abi`.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi/`
  proved `runtime_scalar_cmp_masked_load_store` through selected-body
  realization, provider route facts, common EmitC, target artifact bundle
  export, runtime scalar threshold coverage, old-destination preservation, and
  real `ssh rvv` correctness.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` require RVV compare/select
  semantics, runtime scalar binding, true/false roles, dtype/config,
  mask/policy facts, route facts, and artifact metadata mirrors to remain
  RVV/plugin/provider-owned.
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` defines
  `typed_runtime_scalar_compare_select_pre_realized_body`; it must be realized
  into explicit `setvl`, `with_vl`, `load`, `splat`, `load`, `load`,
  `compare`, `select`, and `store` structure before route construction.
* `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp` already
  validates and realizes the runtime-scalar compare/select pre-realized body,
  binding `lhs`, `rhs_scalar`, `true_value`, `false_value`, `out`, and `n`
  through explicit runtime ABI values.
* `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` already expose and consume
  elementwise/select operand-binding facts, the compare/select statement plan,
  and provider preflight for the runtime-scalar computed-mask select family.
* Initial dry-run evidence for `runtime_scalar_cmp_select` in pre-realized
  selected-body mode already succeeds locally at
  `artifacts/tmp/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi/initial-dry-run/20260601T144515Z`.
  The evidence records `input_mode = pre-realized-selected-body`,
  `pre_realized_body_consumed = true`, runtime ABI order
  `lhs,rhs_scalar,true_value,false_value,out,n`, predicate `sle`,
  runtime scalar source `runtime-scalar-splat-compare-rhs`, true/false value
  roles, select layout `select-true-value-when-mask-else-false-value`, runtime
  counts `0,1,16,17,257`, and RHS scalar values `-500,-37,91`.
* Initial direct route-entry negative evidence exits 1 with the retired direct
  route-entry diagnostic for `runtime_scalar_cmp_select` at
  `artifacts/tmp/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi/initial-direct-fail-v2/20260601T144532Z`.
* The remaining focused gap is not a production C++ route gap: the e2e script
  self-test currently checks runtime-scalar compare/select multi-threshold
  coverage, but does not explicitly pin the `runtime_scalar_cmp_select`
  direct pre-realized route-entry fail-closed diagnostic.

## Requirements

* Use exactly one selected body: `runtime_scalar_cmp_select`.
* Preserve the authority chain:

  ```text
  selected tcrv.exec RVV variant
    -> typed pre-realized runtime-scalar compare/select body
    -> RVV selected-body realization owner
    -> realized tcrv_rvv setvl/with_vl/load/splat/load/load/compare/select/store body
    -> RVV provider-owned route facts
    -> TCRVEmitCLowerableRoute
    -> common EmitC materializer
    -> RVV target artifact bundle
    -> external callable ABI consumer
    -> ssh rvv evidence
  ```

* The positive path must start from the selected-boundary pre-realized body and
  must realize before route construction.
* Runtime `n`/AVL and runtime scalar threshold must come from explicit
  `tcrv.exec`/`tcrv_rvv.runtime_abi_value` binding and selected-body dataflow,
  not route ids, artifact names, manifests, test names, descriptor residue,
  C strings, or common EmitC inference.
* Provider/export evidence must carry `lhs`, `rhs_scalar`, `true_value`,
  `false_value`, `out`, and `n` ABI roles; compare predicate; mask role/source
  and memory form; select true/false roles and layout; tail/mask policy;
  dtype/config; route operand binding; route-control facts; statement plan;
  and required RVV headers/intrinsics.
* Common EmitC/materialization and target export may consume provider-built
  route payloads and mirrors, but must not infer runtime threshold, true/false
  value roles, predicate facts, dtype, policy, or selected-body operation from
  metadata or generated C spellings.
* The generated bundle harness must check `out[i] = true_value[i]` when
  `lhs[i] <= rhs_scalar`, `out[i] = false_value[i]` otherwise, runtime `n=0`
  skips writes, and storage beyond `n` remains sentinel preserved.
* Real `ssh rvv` correctness evidence must cover representative counts
  `0,1,16,17,257` and at least two runtime scalar threshold patterns.
* Direct pre-realized route-entry or artifact export for this body must fail
  closed before route construction/export, and the script self-test must
  explicitly pin this fail-closed diagnostic for `runtime_scalar_cmp_select`.
* Keep changes bounded to directly required evidence script guard/test logic,
  task context, and workspace notes unless production evidence exposes a real
  C++ boundary defect.

## Acceptance Criteria

* [x] Trellis PRD/context files are created and validate successfully.
* [x] Focused evidence shows `runtime_scalar_cmp_select` is realized by the
  RVV plugin selected-body owner before route construction.
* [x] Focused provider/export evidence shows runtime `n`, runtime scalar
  threshold, true input, false input, output memory, compare predicate, select
  layout, tail/mask policy, and typed config survive into the generated bundle.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body --op-kind runtime_scalar_cmp_select` passes for
  counts `0,1,16,17,257` and RHS scalar values `-500,-37,91`.
* [x] Direct pre-realized route-entry mode for `runtime_scalar_cmp_select`
  fails closed with the expected selected lowering-boundary diagnostic before
  bundle export.
* [x] Real `ssh rvv` compile/run correctness evidence passes for counts
  `0,1,16,17,257` and at least two threshold patterns, including predicate-true
  and predicate-false lanes.
* [x] The generated harness checks true/false select behavior and destination
  tail sentinel preservation beyond runtime `n`.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes after
  pinning the runtime-scalar compare/select direct route-entry diagnostic.
* [x] Focused C++ build/test targets for RVV target artifact export and RVV
  plugin pass after the change.
* [x] A bounded old-authority scan over touched plugin/provider/materializer/
  target/script/test/spec/task files classifies any remaining hits for
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, `descriptor`, and `selected route`.
* [x] `git diff --check` passes.
* [x] The task is finished/archived and one coherent commit is created, or the
  exact blocker and continuation point are recorded.

## Definition of Done

* The executable compiler path for `runtime_scalar_cmp_select` is proven
  through generated RVV target artifact bundle export and external ABI
  execution on `ssh rvv`.
* Runtime correctness claims are backed by real `ssh rvv` output.
* No broad compare/select matrix, dual compare/select expansion, dtype/LMUL
  clone batch, high-level frontend authority, source-front-door path,
  descriptor-driven computation path, common EmitC semantic inference,
  performance claim, or broad smoke matrix is added.
* Task status, workspace journal, archived task record, final commit, and final
  report truthfully describe what was completed and what was not.

## Out of Scope

* `runtime_scalar_dual_cmp_mask_and_select`, `computed_mask_select`,
  plain `cmp_select` matrix expansion, masked memory, reduction, MAcc,
  contraction, conversion, or high-level Linalg frontend authority.
* New dtype/LMUL clone batches, dashboards, tuning databases, readiness state
  machines, source-front-door positive routes, descriptor residue, direct-C or
  source-export routes, or common EmitC semantic inference.
* Treating route ids, artifact names, manifests, exact intrinsic spellings,
  status fields, emission-plan metadata, or test names as predicate,
  true/false role, runtime-threshold, dtype, policy, ABI, or route authority.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi/prd.md`.
* Inspected:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and focused target tests.
* Initial generated dry-run evidence:
  `artifacts/tmp/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi/initial-dry-run/20260601T144515Z`.
* Initial direct route-entry negative evidence:
  `artifacts/tmp/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi/initial-direct-fail-v2/20260601T144532Z`.

## Completion Notes

* The chosen selected body was `runtime_scalar_cmp_select`.
* The existing production C++ path already realized the pre-realized selected
  body through `RVVElementwiseSelectedBodyRealizationOwner`, provider route
  facts, compare/select statement planning, common EmitC, and RVV target
  artifact export. No production owner/provider/materializer/target C++ change
  was required.
* Fixed the evidence script so `--self-test` explicitly pins the retired
  direct pre-realized `runtime_scalar_cmp_select` route-entry diagnostic.
* The first real `ssh rvv` run exposed an evidence-harness bug, not a compiler
  route failure: RHS scalar `-500` produced a valid all-false predicate case,
  but the generated harness required every multi-lane case to contain both
  predicate-true and predicate-false lanes. The harness now aggregates
  runtime-scalar compare/select coverage across all counts and thresholds,
  accepts all-false threshold patterns, and still requires aggregate true
  lanes, false lanes, and at least one mixed case.
* Final dry-run evidence:
  `artifacts/tmp/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi/final-dry-run-v2/20260601T145115Z`.
  The evidence records `input_mode = pre-realized-selected-body`,
  `pre_realized_body_consumed = true`, realized `tcrv_rvv.setvl`,
  `tcrv_rvv.with_vl`, `tcrv_rvv.load`, `tcrv_rvv.splat`,
  `tcrv_rvv.compare`, `tcrv_rvv.select`, and `tcrv_rvv.store`; runtime ABI
  order `lhs,rhs_scalar,true_value,false_value,out,n`; predicate `sle`;
  runtime scalar producer source `runtime-scalar-splat-compare-rhs`; select
  layout `select-true-value-when-mask-else-false-value`; runtime counts
  `0,1,16,17,257`; and RHS scalar values `-500,-37,91`.
* Direct pre-realized route-entry negative evidence:
  `artifacts/tmp/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi/final-direct-fail-v2/20260601T145140Z`.
  The command exited 1 with the expected diagnostic naming
  `runtime_scalar_cmp_select` and the required public selected
  lowering-boundary producer before target bundle export.
* Real `ssh rvv` evidence:
  `artifacts/tmp/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi/final-ssh-rvv-v2/20260601T145123Z`.
  Counts `0,1,16,17,257` passed for RHS scalar thresholds `-500`, `-37`, and
  `91`. The generated external ABI harness checked true-value selection,
  false-value selection, destination tail sentinel preservation, all-false
  threshold cases, and mixed predicate cases. Final remote output reported
  `true_lanes=352`, `false_lanes=518`, `mixed_cases=6`,
  `all_true_cases=0`, and `all_false_cases=3`.
* Bounded old-authority scan:
  the tracked diff adds no new hits for the requested legacy-authority strings.
  Existing relevant hits in `scripts/rvv_generated_bundle_abi_e2e.py` are
  pre-existing provider-derived intrinsic evidence, descriptor-residue
  rejection checks, and source-front-door fail-closed diagnostics; no new
  positive legacy route authority was added.
* Spec update decision: no `.trellis/spec/` update was needed. The existing
  RVV plugin, EmitC route, and MLIR testing contracts already cover this
  boundary; this round made the runtime-scalar compare/select generated-bundle
  evidence and direct-route fail-closed regression durable.

## Checks

* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind runtime_scalar_cmp_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -500 --rhs-scalar -37 --rhs-scalar 91 --artifact-root artifacts/tmp/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi/final-dry-run-v2`
* [x] Direct route-entry negative command exited 1 with the expected retired
  direct route-entry diagnostic for `runtime_scalar_cmp_select`.
* [x] Initial real `ssh rvv` command was self-repaired after exposing the
  overly strict all-false threshold harness check.
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind runtime_scalar_cmp_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -500 --rhs-scalar -37 --rhs-scalar 91 --artifact-root artifacts/tmp/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi/final-ssh-rvv-v2`
* [x] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* [x] `rtk sh -lc 'artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-select.mlir --tcrv-materialize-selected-lowering-boundaries | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-select.mlir --check-prefix=REALIZED'`
* [x] `rtk sh -lc 'artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-select.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-select.mlir --check-prefix=PLAN'`
* [x] `rtk sh -lc 'artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-select.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | artifacts/tmp/tianchenrv-build/bin/tcrv-translate --tcrv-export-target-header-artifact | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-select.mlir --check-prefix=HEADER'`
* [x] Bounded old-authority scan over touched script/task files and relevant
  owner/provider/materializer/target/test files.
* [x] `rtk git diff --check`
