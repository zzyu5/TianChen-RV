# Stage2 RVV owner-driven direct contraction executable artifact closure

## Goal

Close executable artifact evidence for one bounded owner-driven direct-provider
contraction route. The chosen route is
`computed_masked_strided_input_widening_dot_reduce_add`, carried from a selected
`tcrv.exec` RVV variant with a pre-realized typed `tcrv_rvv` body through RVV
selected-body realization, the direct contraction route-provider owner,
`TCRVEmitCLowerableRoute`, common EmitC materialization, generated RVV C
artifact/header/harness, and real `ssh rvv` correctness execution.

## What I Already Know

* Repository state before task creation was clean on `main`, with HEAD
  `06f2a4de rvv: add direct contraction route-provider owner`.
* `.trellis/.current-task` did not exist, so this task was created from the
  Hermes direction brief.
* The predecessor task
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-direct-provider-contraction-route-provider-owner/`
  moved the five active direct-provider contraction routes behind the RVV
  plugin-owned direct contraction route-provider owner.
* The predecessor explicitly left `ssh rvv` executable evidence out of scope
  because it changed owner/provider structure only.
* The five active direct-provider contraction routes remain:
  `widening_macc_add`, `widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`.
* Current repository evidence shows the chosen route already has generated
  bundle/harness support in `scripts/rvv_generated_bundle_abi_e2e.py` and
  selected-body fixtures under `test/Target/RVV/`.

## Context Read

* `.trellis/spec/index.md` defines the RVV-first authority chain:
  `tcrv.exec` envelope -> selected RVV variant -> typed low-level
  `tcrv_rvv` body -> RVV plugin-owned legality/realization/route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires direct-provider
  contraction statement construction to flow through the direct contraction
  route-provider owner and forbids route ids, ABI strings, artifact names,
  descriptors, common EmitC, source-front-door metadata, or legacy i32 surfaces
  from becoming route authority.
* `.trellis/spec/lowering-runtime/emitc-route.md` keeps common EmitC neutral:
  RVV headers, vector C types, intrinsic strings, runtime ABI bindings, and
  route payloads are provider-owned and only materialized by common code.
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md` requires
  plugin-local route construction and fail-closed behavior for metadata-only,
  source-front-door, descriptor, route-id, artifact-name, or legacy i32 inputs.
* `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  output for runtime/correctness claims and treats generated-bundle metadata as
  mirror-only after provider route construction.
* `.trellis/workspace/codex/journal-15.md` records the recent route-control,
  selected-body realization owner, route-provider owner, and direct contraction
  owner progression.

## Requirements

* Use one representative direct-provider contraction route:
  `computed_masked_strided_input_widening_dot_reduce_add`.
* Prove the selected path starts from a selected `tcrv.exec` RVV fixture and a
  typed pre-realized `tcrv_rvv` body, not a source-front-door artifact.
* Prove selected-body realization consumes the pre-realized body before route
  facts are collected.
* Prove the direct contraction route-provider owner remains the route statement
  construction authority for the chosen route.
* Prove route-control provider-plan facts, materialization facts, math
  operand-binding facts, typed config/capability facts, runtime AVL/VL facts,
  and ABI/mem_window roles are carried through provider route construction.
* Prove common EmitC only materializes provider output and does not invent RVV
  semantics.
* Produce a generated RVV artifact/header/harness and run it on `ssh rvv` with a
  focused correctness oracle over multiple runtime `n` values.
* Run one representative migrated statement-plan owner dry-run to show the
  owner-driven direct contraction closure did not regress the existing migrated
  statement-plan path.
* If current production code already satisfies the executable boundary, do not
  make unnecessary compiler or script edits. Record the exact evidence instead.
* If execution fails, repair only the production runtime/artifact boundary
  needed for the chosen route, or record the exact blocker if hardware/toolchain
  access prevents completion.

## Acceptance Criteria

* [x] Task context files contain real spec/task context and no placeholder JSONL.
* [x] The chosen direct-provider contraction route is named and bounded.
* [x] Current production route generation for the chosen route succeeds through
  pre-realized selected-body materialization, emission-plan construction,
  provider-built route, common EmitC materialization, and target bundle export.
* [x] The generated artifact/header/harness compiles and runs on `ssh rvv`.
* [x] The correctness oracle checks compare-produced active/inactive mask lanes,
  runtime strided dot-product source positions, scalar seed addition, scalar
  output, runtime AVL/VL counts, and tail/non-scalar sentinel preservation.
* [x] A migrated statement-plan owner path dry-run still succeeds.
* [x] Bounded touched-file/current-path scan shows no new legacy
  i32/source-front-door/descriptor/direct-C/source-export/ABI-string/
  artifact-name/script/common-EmitC/metadata/route-id authority leaks.
* [x] `git diff --check` passes.
* [x] `check-tianchenrv` passes, or an exact blocker is recorded.

## Out Of Scope

* New RVV operation families, new direct contraction variants, dtype/LMUL clone
  batches, source-front-door positive routes, frontend/Linalg routes, common
  EmitC RVV semantics, descriptor/direct-C/source-export computation,
  dashboards, broad smoke matrices, performance claims, non-RVV plugin work,
  and changes that weaken owner-driven route construction or fail-closed
  diagnostics.

## Definition Of Done

* PRD, `implement.jsonl`, and `check.jsonl` are truthful.
* The task either records real executable evidence on the existing production
  path or repairs the exact production boundary needed to obtain it.
* Focused and final checks pass or blockers are exact.
* Task status and journal are truthful.
* Completed work is archived and committed as one coherent commit, or the
  unfinished continuation point is explicit.

## Evidence Captured So Far

* Dry-run local bundle generation for chosen route:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 1 --runtime-count 7 --runtime-count 17 --runtime-count 33 --runtime-count 257 --artifact-root artifacts/tmp/stage2_rvv_direct_contraction_executable_artifact_closure --run-id pre-realized-computed-mask-strided-dry --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
  succeeded under
  `artifacts/tmp/stage2_rvv_direct_contraction_executable_artifact_closure/pre-realized-computed-mask-strided-dry`.
* Non-dry-run `ssh rvv` generated-bundle execution for chosen route:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 1 --runtime-count 7 --runtime-count 17 --runtime-count 33 --runtime-count 257 --artifact-root artifacts/tmp/stage2_rvv_direct_contraction_executable_artifact_closure --run-id pre-realized-computed-mask-strided-ssh --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --timeout 180 --connect-timeout 10`
  succeeded and printed
  `PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=1,7,17,33,257 lhs_stride=2 rhs_stride=3`.
* Migrated statement-plan owner non-regression dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --op-kind add --runtime-count 1 --runtime-count 17 --artifact-root artifacts/tmp/stage2_rvv_direct_contraction_executable_artifact_closure --run-id migrated-add-statement-plan-dry --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
  succeeded under
  `artifacts/tmp/stage2_rvv_direct_contraction_executable_artifact_closure/migrated-add-statement-plan-dry`.
* Current-path authority scan:
  `rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|source-front-door|source-artifact|descriptor|__riscv_.*_i32m1" include/TianChenRV/Plugin/RVV lib/Plugin/RVV/EmitC lib/Target/RVV include/TianChenRV/Target/RVV scripts/rvv_generated_bundle_abi_e2e.py test/Target/RVV`
  reported existing provider-derived intrinsic mirrors, descriptor/source-front
  door negative checks, and legacy fail-closed fixtures; no production files
  were changed in this task, and `git diff --name-only -- ':!.trellis/tasks/**'`
  was empty.
* Task-file authority scan:
  `rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|source-front-door|source-artifact|descriptor|direct-C|source-export|ABI-string|artifact-name|route-id|common-EmitC|metadata" .trellis/tasks/05-26-stage2-rvv-direct-contraction-executable-artifact-closure`
  found only explicit non-goal, fail-closed, and mirror-only wording.
* `git diff --check` passed.
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-stage2-rvv-direct-contraction-executable-artifact-closure`
  passed.
* `cmake --build build --target check-tianchenrv -j2` passed: 379/379 tests.
