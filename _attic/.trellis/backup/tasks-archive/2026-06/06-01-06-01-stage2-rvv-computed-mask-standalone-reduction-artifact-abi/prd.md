# Stage2 RVV computed-mask standalone-reduction artifact ABI boundary

## Goal

Carry exactly one existing supported computed-mask standalone reduction selected
body from a selected `tcrv.exec` RVV variant through RVV plugin-local
selected-body realization, provider-owned route facts, common EmitC
materialization, RVV target artifact bundle generation, and real `ssh rvv`
scalar-result ABI correctness evidence.

The chosen bounded case is `computed_mask_standalone_reduce_add`. The task must
prove that compare-produced mask facts, source memory, scalar seed, scalar
result output, runtime `n`/AVL, inactive-lane neutralization, and typed
reduction semantics survive the selected-body-to-generated-artifact path without
route names, artifact names, descriptors, common export inference, exact
intrinsic spellings, or test names becoming authority.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief as
  `.trellis/tasks/06-01-06-01-stage2-rvv-computed-mask-standalone-reduction-artifact-abi`.
* The repository started clean on `main` at
  `1c8cad9a rvv: prove standalone reduction artifact abi`.
* The previous standalone reduction artifact ABI task proved unmasked
  `standalone_reduce_add` through generated bundle export, direct route-entry
  fail-closed regression, and real `ssh rvv` counts `0,1,16,17,257`.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` require RVV semantics,
  dtype/config, route facts, mask/policy facts, scalar-result ABI, and artifact
  metadata mirrors to remain RVV/plugin/provider-owned.
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` defines
  `typed_computed_mask_standalone_reduce_pre_realized_body`; it must be
  realized into explicit `setvl`, `with_vl`, three loads, `compare`,
  `masked_standalone_reduce`, and scalar-result `store` before provider route
  construction.
* Initial dry-run evidence for `computed_mask_standalone_reduce_add` in
  pre-realized selected-body mode already succeeds locally and records ABI
  order `cmp_lhs,cmp_rhs,src,acc,out,n`, compare predicate `sle`, mask role
  `predicate-mask-produced-by-compare`, scalar seed source `acc[0]`, scalar
  output role `out`, and runtime counts `0,1,16,17,257`.

## Requirements

* Use exactly one existing supported selected body:
  `computed_mask_standalone_reduce_add`.
* Preserve the authority chain:

  ```text
  selected tcrv.exec RVV variant
    -> typed pre-realized computed-mask standalone reduction body
    -> RVV selected-body realization owner
    -> realized tcrv_rvv setvl/with_vl/load/load/load/compare/masked_standalone_reduce/store body
    -> RVV provider-owned route facts
    -> TCRVEmitCLowerableRoute
    -> common EmitC materializer
    -> RVV target artifact bundle
    -> external scalar-result ABI consumer
    -> ssh rvv evidence
  ```

* The positive path must start from the selected-boundary pre-realized body and
  must realize before route construction.
* Provider/export evidence must carry runtime `n`/AVL, compare lhs/rhs roles,
  source input role, scalar accumulator seed role/layout, scalar result
  output role/layout, compare predicate, mask role/source/form, inactive-lane
  neutralization, operation kind, dtype/config, scalar result vector facts,
  route operand binding, statement plan, and required RVV headers/intrinsics.
* Common EmitC/materialization and target export may consume provider-built
  route payloads and mirrors, but must not infer computed-mask standalone
  reduction semantics from route ids, artifact names, manifests, C strings, or
  test names.
* The generated bundle and harness must preserve scalar-result ABI behavior:
  compare input memory, source input memory, scalar seed/identity input,
  scalar output storage, runtime `n`, mixed active/inactive masks,
  all-inactive masks, at least two seeds, and no writes outside the scalar
  output contract.
* Direct pre-realized route-entry or artifact export for this computed-mask
  standalone reduction must fail closed before route construction/export.
* Keep changes bounded to the module owner files, directly required tests,
  task context, and workspace notes.

## Acceptance Criteria

* [x] Trellis PRD/context files are created and validate successfully.
* [x] Focused evidence shows `computed_mask_standalone_reduce_add` is realized
  by the RVV plugin selected-body owner before route construction.
* [x] Focused route/provider or generated-bundle evidence shows runtime `n`,
  compare lhs/rhs roles, source memory role, mask/compare role, scalar result
  role, scalar seed/identity facts, inactive-lane neutralization, and typed
  reduction facts survive export.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body --op-kind computed_mask_standalone_reduce_add`
  passes for counts `0,1,16,17,257`.
* [x] Direct pre-realized route-entry mode for
  `computed_mask_standalone_reduce_add` fails closed with the expected selected
  lowering-boundary diagnostic before bundle export.
* [x] Real `ssh rvv` compile/run correctness evidence passes for counts
  `0,1,16,17,257`, two seeds, mixed active/inactive mask cases, and all-inactive
  mask cases.
* [x] Focused C++ build/test targets for target artifact export and RVV plugin
  pass after the change.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes.
* [x] A bounded old-authority scan over touched plugin/provider/materializer/
  target/script/test/spec/task files classifies any remaining hits for
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, `descriptor`, and `selected route`.
* [x] `git diff --check` passes.
* [x] The task is finished/archived and one coherent commit is created, or the
  exact blocker and continuation point are recorded.

## Definition of Done

* The executable compiler path for `computed_mask_standalone_reduce_add` is
  proven through generated RVV target artifact bundle export and external
  scalar-result ABI execution on `ssh rvv`.
* Runtime correctness claims are backed by real `ssh rvv` output.
* No new reduction family, min/max expansion, MAcc/contraction expansion,
  dtype/LMUL clone batch, source-front-door path, descriptor-driven computation
  path, common EmitC semantic inference, performance claim, or broad smoke
  matrix is added.
* Task status, workspace journal, archived task record, final commit, and final
  report truthfully describe what was completed and what was not.

## Out of Scope

* New reduction families or broad coverage expansion.
* `computed_mask_standalone_reduce_min/max`, runtime-scalar masked standalone
  reduction, MAcc, computed-mask MAcc, contraction, high-level Linalg frontend
  authority, or performance claims.
* Dtype/LMUL clone batches, dashboards, tuning databases, readiness state
  machines, source-front-door positive routes, descriptor residue, direct-C or
  source-export routes, or common EmitC semantic inference.
* Treating route ids, artifact names, manifests, exact intrinsic spellings,
  status fields, emission-plan metadata, or test names as mask, reduction kind,
  dtype, identity, scalar-result, policy, ABI, or route authority.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-standalone-reduction-artifact-abi/prd.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-standalone-reduction-realization-boundary/prd.md`.
* Inspected:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and the initial generated
  dry-run evidence under
  `artifacts/tmp/06-01-stage2-rvv-computed-mask-standalone-reduction-artifact-abi/initial-dry-run/20260601T141002Z`.

## Completion Notes

* The chosen selected body was `computed_mask_standalone_reduce_add`.
* The existing production compiler/export path already carried the selected
  computed-mask standalone reduction body through RVV selected-body
  realization, provider route facts, common EmitC, RVV target artifact bundle
  export, and external scalar-result ABI execution. No production C++ owner,
  provider, materializer, target, or remote-probe change was required.
* Added a durable `scripts/rvv_generated_bundle_abi_e2e.py --self-test`
  regression for the retired direct pre-realized
  `computed_mask_standalone_reduce_add` route-entry mode. The diagnostic must
  name `computed_mask_standalone_reduce_add` and the required public selected
  lowering-boundary producer.
* Final dry-run evidence:
  `artifacts/tmp/06-01-stage2-rvv-computed-mask-standalone-reduction-artifact-abi/final-dry-run/20260601T141327Z`.
  The evidence records `input_mode = pre-realized-selected-body`,
  `pre_realized_body_consumed = true`, materialized `tcrv_rvv.setvl`,
  `tcrv_rvv.with_vl`, three `tcrv_rvv.load` ops, `tcrv_rvv.compare`,
  `tcrv_rvv.masked_standalone_reduce`, and scalar-result `tcrv_rvv.store`.
  The header prototype is
  `void tcrv_emitc_pre_cm_standalone_reduce_kernel_rvv_pre_cm_standalone_reduce(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src, const int32_t *acc, int32_t *out, size_t n);`.
  Runtime ABI order is `cmp_lhs,cmp_rhs,src,acc,out,n`; mask facts are
  `predicate-mask-produced-by-compare`,
  `compare-produced-mask-same-vl-scope`, and `compare-produced-mask`; scalar
  seed source is `acc[0]`; scalar output is `out[0]`; runtime counts are
  `0,1,16,17,257`.
* Direct pre-realized route-entry negative evidence:
  `artifacts/tmp/06-01-stage2-rvv-computed-mask-standalone-reduction-artifact-abi/final-direct-fail/20260601T141327Z`.
  The command exited 1 with the expected retired direct route-entry diagnostic
  before target bundle export.
* Real `ssh rvv` evidence:
  `artifacts/tmp/06-01-stage2-rvv-computed-mask-standalone-reduction-artifact-abi/final-ssh-rvv/20260601T141350Z`.
  Counts `0,1,16,17,257` passed for seeds `-11` and `17`. The external
  generated-bundle ABI harness checked mixed active/inactive masks and
  all-inactive masks. Representative multi-lane counts produced active/inactive
  splits `10/6`, `11/6`, and `155/102`; all-inactive cases preserved the seed;
  scalar-output tail sentinels were preserved.
* Bounded old-authority scan:
  the tracked script diff contains no matches for the requested legacy-authority
  strings. New task-context hits are guardrail/acceptance text. Existing
  relevant module/test/script hits are pre-existing provider-derived intrinsic
  leaf evidence, fail-closed legacy tests, descriptor rejection checks, selected
  route consistency diagnostics, or old-authority scan vocabulary; no new
  positive legacy route authority was added.
* Spec update decision: no `.trellis/spec/` update was needed. The existing RVV
  plugin, EmitC route, and MLIR testing contract already cover this boundary;
  this round made the computed-mask standalone-reduction direct-route
  fail-closed regression and end-to-end evidence durable.

## Checks

* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-06-01-stage2-rvv-computed-mask-standalone-reduction-artifact-abi`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind computed_mask_standalone_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-computed-mask-standalone-reduction-artifact-abi/final-dry-run`
* [x] Direct route-entry negative command exited 1 with the expected retired
  direct route-entry diagnostic for `computed_mask_standalone_reduce_add`.
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_mask_standalone_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-computed-mask-standalone-reduction-artifact-abi/final-ssh-rvv`
* [x] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* [x] Bounded old-authority scan over touched script/task files and relevant
  owner/provider/materializer/target/test files.
* [x] `rtk git diff --check`
