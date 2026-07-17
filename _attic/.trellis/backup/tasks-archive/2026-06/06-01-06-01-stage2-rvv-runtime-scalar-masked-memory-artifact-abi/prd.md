# Stage2 RVV runtime-scalar masked memory artifact ABI boundary

## Goal

Carry exactly one existing runtime-scalar compare masked memory selected body
from a selected `tcrv.exec` RVV variant through RVV plugin-local selected-body
realization, provider-owned route facts, common EmitC materialization, RVV
target artifact bundle generation, and real `ssh rvv` correctness evidence.

The bounded case is `runtime_scalar_cmp_masked_load_store`, because it exercises
runtime scalar threshold binding, runtime `n`/AVL, compare-mask construction,
source memory, destination output memory, and old-destination passthrough memory
roles in one generated artifact ABI path.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief as
  `.trellis/tasks/06-01-06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi`.
* The repository started clean on `main` at
  `01e2973d rvv: prove computed-mask standalone reduction artifact abi`.
* The previous completed task proved `computed_mask_standalone_reduce_add`
  through selected-body realization, provider route facts, common EmitC, target
  artifact bundle export, scalar-result ABI harness, and real `ssh rvv`
  correctness with mixed and all-inactive masks.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` require RVV semantics,
  dtype/config, mask/policy facts, runtime ABI binding, memory roles, route
  facts, and artifact metadata mirrors to remain RVV/plugin/provider-owned.
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` defines
  `typed_runtime_scalar_computed_mask_load_store_pre_realized_body`; it must be
  realized into explicit `setvl`, `with_vl`, `load`, `splat`, `load`,
  `compare`, `masked_load`, and `store` structure before route construction.
* `lib/Plugin/RVV/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.cpp`
  already has a selected-body owner for the runtime-scalar computed-mask
  load-store family. The owner validates the pre-realized body, binds runtime
  AVL from `n`, loads `lhs`, splats `rhs_scalar`, loads old `dst`, compares,
  masked-loads from `src` with old `dst` passthrough, and stores the merged
  result to `dst`.
* Initial dry-run evidence for
  `runtime_scalar_cmp_masked_load_store` in pre-realized selected-body mode
  already succeeds locally at
  `artifacts/tmp/06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi/initial-dry-run/20260601T142617Z`.
  The evidence records `input_mode = pre-realized-selected-body`,
  `pre_realized_body_consumed = true`, runtime ABI order
  `lhs,rhs_scalar,src,dst,n`, mask role
  `predicate-mask-produced-by-compare`, mask source
  `compare-produced-mask-same-vl-scope`, inactive lane contract
  `masked-off-lanes-preserve-old-destination`, old passthrough role
  `output-buffer`, and runtime counts `0,1,16,17,257`.

## Requirements

* Use exactly one existing selected body:
  `runtime_scalar_cmp_masked_load_store`.
* Preserve the authority chain:

  ```text
  selected tcrv.exec RVV variant
    -> typed pre-realized runtime-scalar computed-mask load-store body
    -> RVV selected-body realization owner
    -> realized tcrv_rvv setvl/with_vl/load/splat/load/compare/masked_load/store body
    -> RVV provider-owned route facts
    -> TCRVEmitCLowerableRoute
    -> common EmitC materializer
    -> RVV target artifact bundle
    -> external memory ABI consumer
    -> ssh rvv evidence
  ```

* The positive path must start from the selected-boundary pre-realized body and
  must realize before route construction.
* Runtime `n`/AVL and runtime scalar threshold must come from explicit
  `tcrv.exec`/`tcrv_rvv.runtime_abi_value` binding and selected-body
  dataflow, not route ids, artifact names, manifests, test names, descriptor
  residue, C strings, or common EmitC inference.
* Provider/export evidence must carry `lhs`, `rhs_scalar`, `src`, `dst`, and
  `n` ABI roles; source/destination/passthrough memory roles; compare
  predicate; mask role/source/form; inactive-lane old-destination preservation;
  tail/mask policy; dtype/config; route operand binding; route-control facts;
  statement plan; and required RVV headers/intrinsics.
* Common EmitC/materialization and target export may consume provider-built
  route payloads and mirrors, but must not infer runtime threshold, memory
  roles, mask facts, dtype, policy, or selected body operation from metadata or
  generated C spellings.
* The generated bundle harness must check active lanes copy `src[i]` to
  `dst[i]` when `lhs[i] <= rhs_scalar`, inactive lanes preserve old `dst[i]`,
  runtime `n=0` skips writes, and storage beyond `n` remains sentinel
  preserved.
* Real `ssh rvv` correctness evidence must cover representative counts
  `0,1,16,17,257` and at least two threshold/mask patterns.
* Direct pre-realized route-entry or artifact export for this body must fail
  closed before route construction/export.
* Keep changes bounded to the module owner files, directly required tests or
  scripts, task context, and workspace notes.

## Acceptance Criteria

* [x] Trellis PRD/context files are created and validate successfully.
* [x] Focused evidence shows `runtime_scalar_cmp_masked_load_store` is realized
  by the RVV plugin selected-body owner before route construction.
* [x] Focused provider/export evidence shows runtime `n`, runtime scalar
  threshold, `lhs`, `src`, `dst`, old-destination passthrough, mask/compare
  facts, inactive-lane contract, tail/mask policy, and typed config survive
  into the generated artifact bundle.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body --op-kind runtime_scalar_cmp_masked_load_store`
  passes for counts `0,1,16,17,257`.
* [x] Direct pre-realized route-entry mode for
  `runtime_scalar_cmp_masked_load_store` fails closed with the expected selected
  lowering-boundary diagnostic before bundle export.
* [x] Real `ssh rvv` compile/run correctness evidence passes for counts
  `0,1,16,17,257` and at least two threshold/mask patterns, including mixed
  active/inactive and all-inactive behavior.
* [x] The generated harness checks old-destination preservation for inactive
  lanes and sentinel preservation beyond runtime `n`.
* [x] Focused C++ build/test targets for RVV target artifact export and RVV
  plugin pass after the change.
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

* The executable compiler path for `runtime_scalar_cmp_masked_load_store` is
  proven through generated RVV target artifact bundle export and external
  memory ABI execution on `ssh rvv`.
* Runtime correctness claims are backed by real `ssh rvv` output.
* No broad memory matrix, strided/indexed/segment expansion, dtype/LMUL clone
  batch, high-level frontend authority, source-front-door path,
  descriptor-driven computation path, common EmitC semantic inference,
  performance claim, or broad smoke matrix is added.
* Task status, workspace journal, archived task record, final commit, and final
  report truthfully describe what was completed and what was not.

## Out of Scope

* New memory forms beyond `runtime_scalar_cmp_masked_load_store`.
* `runtime_scalar_cmp_masked_store`, strided/indexed/segment memory,
  compare-select expansion, standalone reduction, MAcc, contraction, high-level
  Linalg frontend authority, or performance claims.
* Dtype/LMUL clone batches, dashboards, tuning databases, readiness state
  machines, source-front-door positive routes, descriptor residue, direct-C or
  source-export routes, or common EmitC semantic inference.
* Treating route ids, artifact names, manifests, exact intrinsic spellings,
  status fields, emission-plan metadata, or test names as mask, memory-role,
  runtime-threshold, dtype, policy, ABI, or route authority.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-computed-mask-standalone-reduction-artifact-abi/prd.md`.
* Inspected:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Plugin/RVV/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `scripts/rvv_remote_probe.py`, and
  `test/Target/TargetArtifactExportTest.cpp`.
* Initial generated dry-run evidence:
  `artifacts/tmp/06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi/initial-dry-run/20260601T142617Z`.

## Completion Notes

* The chosen selected body was `runtime_scalar_cmp_masked_load_store`.
* The existing production C++ path already realized the pre-realized selected
  body through `RVVRuntimeScalarMemorySelectedBodyRealizationOwner`, provider
  route facts, common EmitC, and RVV target artifact export. No production
  owner/provider/materializer/target C++ change was required.
* The evidence-tool gap was in
  `scripts/rvv_generated_bundle_abi_e2e.py`: runtime-scalar computed-mask
  memory accepted a single RHS scalar value, and the generated harness rejected
  all-inactive threshold cases before aggregate threshold-pattern evidence
  could be collected.
* Fixed the evidence script so runtime-scalar computed-mask store/load-store
  evidence requires at least two RHS scalar values, self-test checks the
  retired direct pre-realized `runtime_scalar_cmp_masked_load_store`
  route-entry diagnostic, and the generated harness supports all-inactive
  threshold cases while aggregating mixed active/inactive and payload
  distinguishing coverage.
* Final dry-run evidence:
  `artifacts/tmp/06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi/final-dry-run-v3/20260601T143230Z`.
  The evidence records `input_mode = pre-realized-selected-body`,
  `pre_realized_body_consumed = true`, realized `tcrv_rvv.setvl`,
  `tcrv_rvv.with_vl`, `tcrv_rvv.load`, `tcrv_rvv.splat`,
  `tcrv_rvv.compare`, `tcrv_rvv.masked_load`, and `tcrv_rvv.store`; runtime
  ABI order `lhs,rhs_scalar,src,dst,n`; inactive lane contract
  `masked-off-lanes-preserve-old-destination`; old passthrough role
  `output-buffer`; and runtime counts `0,1,16,17,257`.
* Direct pre-realized route-entry negative evidence:
  `artifacts/tmp/06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi/final-direct-fail-v2/20260601T143243Z`.
  The command exited 1 with the expected diagnostic naming
  `runtime_scalar_cmp_masked_load_store` and the required public selected
  lowering-boundary producer before target bundle export.
* Real `ssh rvv` evidence:
  `artifacts/tmp/06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi/final-ssh-rvv-v2/20260601T143230Z`.
  Counts `0,1,16,17,257` passed for RHS scalar thresholds `-500`, `-37`, and
  `91`. The generated external ABI harness checked all-inactive cases
  (`-500`), mixed active/inactive cases (`-37`, `91`), old-destination
  passthrough preservation, payload-distinguishing active lanes, source buffer
  preservation, and destination tail sentinel preservation. Final remote output
  reported `mixed_mask_cases=6`, `all_inactive_cases=3`, and
  `payload_distinguishing_lanes=354`.
* Bounded old-authority scan:
  the tracked diff adds no new hits for the requested legacy-authority strings.
  Existing relevant hits are pre-existing fail-closed legacy tests, descriptor
  rejection checks, provider selected-route consistency diagnostics, provider
  derived intrinsic leaf evidence in existing coverage, and task guardrail
  vocabulary; no new positive legacy route authority was added.
* Spec update decision: no `.trellis/spec/` update was needed. The existing
  RVV plugin, EmitC route, and MLIR testing contract already cover this
  boundary; this round made the runtime-scalar masked memory generated-bundle
  evidence and direct-route fail-closed regression durable.

## Checks

* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind runtime_scalar_cmp_masked_load_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -500 --rhs-scalar -37 --rhs-scalar 91 --artifact-root artifacts/tmp/06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi/final-dry-run-v3`
* [x] Direct route-entry negative command exited 1 with the expected retired
  direct route-entry diagnostic for `runtime_scalar_cmp_masked_load_store`.
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind runtime_scalar_cmp_masked_load_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -500 --rhs-scalar -37 --rhs-scalar 91 --artifact-root artifacts/tmp/06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi/final-ssh-rvv-v2`
* [x] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* [x] `rtk sh -lc 'artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-load-store.mlir --tcrv-materialize-selected-lowering-boundaries | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-load-store.mlir --check-prefix=REALIZED'`
* [x] `rtk sh -lc 'artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-load-store.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-load-store.mlir --check-prefix=PLAN'`
* [x] `rtk sh -lc 'artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-load-store.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | artifacts/tmp/tianchenrv-build/bin/tcrv-translate --tcrv-export-target-header-artifact | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-load-store.mlir --check-prefix=HEADER'`
* [x] `rtk sh -lc 'artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Dialect/RVV/runtime-scalar-computed-mask-load-store-dataflow.mlir --split-input-file --verify-diagnostics | /usr/lib/llvm-20/bin/FileCheck test/Dialect/RVV/runtime-scalar-computed-mask-load-store-dataflow.mlir'`
* [x] Bounded old-authority scan over touched script/task files and relevant
  owner/provider/materializer/target/test files.
* [x] `rtk git diff --check`
