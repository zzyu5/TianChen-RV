# Stage2 RVV composite gather-MAcc-scatter generated-bundle runtime evidence

## Goal

Prove the selected `tcrv.exec` envelope version of the
`runtime_scalar_cmp_masked_indexed_gather_macc_scatter` RVV composite route
through generated target artifact bundle compilation and real `ssh rvv`
correctness execution for both explicit and pre-realized selected bodies, or
leave an exact production blocker if the executable boundary is not yet valid.

This task continues after commit `3ae4859e`, which made the composite selected
envelope ABI/export boundary structural and fail-closed but explicitly did not
claim new `ssh rvv` runtime correctness.

## What I Already Know

- There was no active `.trellis/.current-task`; this task records the Hermes
  direction before source changes.
- Current HEAD at task creation is
  `3ae4859e rvv: bind composite selected envelope abi`.
- The previous selected-envelope task updated explicit and pre-realized
  composite fixtures with `tcrv.exec.mem_window` / `tcrv.exec.runtime_param`
  declarations, `exec_binding` references, selected dispatch/fallback mirrors,
  and `tcrv_rvv.exec_abi_bindings` target artifact validation.
- The earlier runtime-boundary task proved the pre-envelope composite generated
  bundle on `ssh rvv`, but that evidence predates the selected-envelope ABI
  binding and mirror validation from `3ae4859e`.
- RVV runtime/correctness claims require real `ssh rvv` evidence. Dry-run
  bundle/header evidence is insufficient for this task's claim.
- The production authority chain must remain:
  `tcrv.exec` envelope -> selected RVV variant -> typed or realized
  `tcrv_rvv` body -> RVV provider route facts -> `TCRVEmitCLowerableRoute` ->
  common EmitC materialization -> target artifact bundle -> generated harness
  -> `ssh rvv` execution.

## Requirements

- Use the selected-envelope explicit fixture
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
  and the selected-envelope pre-realized fixture
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`.
- Run the generated-bundle path through
  `scripts/rvv_generated_bundle_abi_e2e.py` in non-dry-run mode for both
  explicit and pre-realized selected bodies.
- Evidence must cover the runtime ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`, selected
  `exec_abi_bindings` mirror consistency, runtime scalar compare mask behavior,
  noncontiguous indexed gather/scatter lanes, inactive destination
  preservation, signed product accumulation, source/payload/accumulator
  preservation, tail preservation, runtime AVL/VL counts, and provider support
  mirror consistency.
- If `ssh rvv` exposes a real compiler, artifact, header/prototype, harness,
  ABI-order, mask/index/MAcc, runtime AVL/VL, or provider mirror bug, fix the
  focused owning production seam.
- Do not move RVV semantics into route ids, artifact names, descriptors,
  source-front-door markers, test names, common EmitC inference, or helper-only
  metadata.
- Keep fail-closed evidence for selected-envelope stale provider mirror,
  runtime ABI order, missing/stale `exec_abi_bindings`, and missing selected
  body `exec_binding`.

## Acceptance Criteria

- [x] Explicit selected-envelope composite generated bundle runs non-dry-run on
  `ssh rvv` and reports route-specific success, or the task records the exact
  production blocker and continuation point.
- [x] Pre-realized selected-envelope composite generated bundle runs
  non-dry-run on `ssh rvv` and reports route-specific success, or the task
  records the exact production blocker and continuation point.
- [x] Runtime evidence includes counts `0,1,16,17,257`, RHS scalars `-37,91`,
  patterns `0,1`, noncontiguous indexed lanes, signed product lanes, inactive
  lane preservation, source/payload/accumulator preservation, and tail
  preservation.
- [x] Evidence JSON records `ssh_evidence = true`, remote compile success,
  remote run success, and `PASS op=runtime_scalar_cmp_masked_indexed_gather_macc_scatter`.
- [x] Focused generated-bundle lit/FileCheck for the composite route still
  passes.
- [x] Focused fail-closed evidence for stale/missing selected-envelope ABI and
  provider mirrors still passes.
- [x] Build targets `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test` build.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passes.
- [x] Bounded old-authority scan over touched files and added diff lines finds
  no new positive legacy `i32m1`, `RVVI32M1`, `rvv-i32m1`, descriptor,
  direct-C/source-export, or source-front-door route authority.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] Final worktree is clean after one coherent commit if the task completes.

## Completion Notes

- No production source change was needed. The selected-envelope artifact/export
  seam from `3ae4859e` already produced executable generated bundles for both
  explicit and pre-realized composite routes.
- Explicit selected-envelope `ssh rvv` evidence:
  `artifacts/tmp/stage2-selected-envelope-composite-gms-ssh-rvv/explicit-selected-envelope-composite-gms-ssh-rvv/runtime_scalar_cmp_masked_indexed_gather_macc_scatter/evidence.json`.
- Pre-realized selected-envelope `ssh rvv` evidence:
  `artifacts/tmp/stage2-selected-envelope-composite-gms-ssh-rvv/pre-selected-envelope-composite-gms-ssh-rvv/runtime_scalar_cmp_masked_indexed_gather_macc_scatter/evidence.json`.
- Both runs used runtime counts `0,1,16,17,257`, RHS scalars `-37,91`, and
  patterns `0,1`. Both recorded `ssh_evidence = true`, `status = success`,
  remote RISC-V clang compile success, remote run success, and
  `PASS op=runtime_scalar_cmp_masked_indexed_gather_macc_scatter`.
- Runtime stdout covered active lanes, inactive destination preservation,
  noncontiguous indexed gather/scatter lanes, signed product lanes,
  source/payload/accumulator preservation, and tail preservation.
- Bundle indexes for both runs carried the selected-envelope
  `tcrv_rvv.exec_abi_bindings` mirror:
  `cmp_lhs=lhs-input-buffer->@abi_cmp_lhs_input_buffer;rhs_scalar=rhs-scalar-value->@abi_rhs_scalar_value;gather_src=source-input-buffer->@abi_source_input_buffer;payload=dot-rhs-input-buffer->@abi_dot_rhs_input_buffer;acc=accumulator-input-buffer->@abi_accumulator_input_buffer;index=index-input-buffer->@abi_index_input_buffer;dst=output-buffer->@abi_output_buffer;n=runtime-element-count->@abi_runtime_element_count`.
- Focused fail-closed evidence remains in the explicit and pre-realized target
  artifact lit tests for stale provider mirror, stale runtime ABI order,
  stale/missing selected-envelope exec ABI mirrors, and missing selected-body
  `exec_binding`.
- Spec update review found no new durable contract to add: the current RVV
  plugin, EmitC route, emission runtime, and MLIR testing specs already state
  the composite route, selected-envelope mirror, generated-bundle, and `ssh rvv`
  evidence requirements exercised here.

## Definition Of Done

- The selected-envelope composite generated-bundle runtime path is truthfully
  proven with real `ssh rvv` evidence for both explicit and pre-realized
  selected bodies, or the task remains open with the exact blocker.
- Any code changes are limited to the generated-bundle/runtime/export seam or
  directly related fail-closed tests.
- The task status, journal, and archive state match the actual evidence.

## Out Of Scope

- No broad composite matrix.
- No dtype/LMUL clone batch.
- No new gather/scatter/MAcc family expansion.
- No performance tuning or dashboard/report-only closure.
- No high-level Linalg/Vector/StableHLO frontend work.
- No source-front-door positive route.
- No per-Linalg route authority or common EmitC invention of RVV semantics.
- No unrelated memory, segment2, reduction, compare/select, conversion, or
  MAcc rewrites outside this composite runtime boundary.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Previous task context read:
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-selected-envelope-composite-gather-macc-scatter/prd.md`
  - `.trellis/tasks/archive/2026-06/06-07-06-07-stage2-rvv-composite-gather-macc-scatter-ssh-rvv-runtime-boundary/prd.md`
- Read-first implementation/test references:
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run.test`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
