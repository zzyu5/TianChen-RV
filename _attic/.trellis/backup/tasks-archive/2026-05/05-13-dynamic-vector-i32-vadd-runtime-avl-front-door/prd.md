# Dynamic vector i32-vadd runtime AVL front door

## Goal

Add exactly one dynamic-length MLIR vector/SCF i32-vadd source front door that
carries a real source `%n` extent into the existing finite RVV selected
boundary, materialized `tcrv_rvv` body, common EmitC artifacts, dispatch ABI,
and `ssh rvv` evidence. This is a bounded source-authority repair for runtime
AVL, not a generic vector backend or scalable-vector maturity claim.

The production path for this task is:

```text
marked vector/SCF i32-vadd source wrapper with %n
  -> tcrv.exec.kernel with finite binary ABI roles and dynamic source extent authority
  -> RVV plugin selected finite i32-vadd variant
  -> materialized tcrv_rvv.i32_vadd_microkernel body
  -> common EmitC route
  -> source/header/object or bundle artifacts
  -> ssh rvv evidence for counts 7, 16, and 23
```

## What I Already Know

* Initial repository state for this task: `pwd` is
  `/home/kingdom/phdworks/TianchenRV`, the worktree was clean, and HEAD was
  `c4cf3f2 fix(vector): enforce fixed AVL boundary`.
* No `.trellis/.current-task` existed at session start. This task was created
  from the Hermes Direction Brief before source edits and started as the
  current Trellis task.
* The immediately previous archived vector PRD closed the fixed
  `vector<16xi32>` front door by recording source extent `16`, requiring
  selected RVV descriptor-local `element_count = 16`, and emitting an
  `n == 16` fail-closed guard in direct and dispatch artifacts.
* Current specs separate runtime SSA/control values such as `n`, AVL, and VL
  from selected RVV vector-shape config and descriptor-local `element_count`.
* The existing vector lowering pass accepts one fixed source wrapper through
  `--tcrv-lower-vector-rvv-i32-vadd-to-exec` and already shares the same
  finite i32-vadd ABI, RVV plugin planning, materialized `tcrv_rvv` body, and
  common EmitC route as linalg-fronted binary inputs.
* The dynamic source authority required in this round is a marked `func.func`
  with three `memref<?xi32>` buffers and one source `%n: index`, containing
  one bounded `scf.for` from zero to `%n` in step `16`, with
  `vector.transfer_read`, `arith.addi` over `vector<16xi32>`, and
  `vector.transfer_write`.

## Requirements

* Preserve compiler implementation in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
* Add exactly one dynamic vector i32-vadd source-fronted route. Do not add a
  generic MLIR vector backend, dtype/shape matrix, scalable-vector claim,
  descriptor-driven computation, or direct descriptor-to-C exporter.
* Keep the existing fixed vector front door truthful: fixed
  `vector<16xi32>` artifacts must still enforce `n == 16` and reject
  mismatched runtime counts.
* Require the dynamic source wrapper shape to be explicit and bounded:
  three `memref<?xi32>` buffers, one `%n: index`, one `scf.for` with lower
  bound zero, upper bound `%n`, step `16`, two transfer reads at the induction
  variable, one vector `arith.addi`, one transfer write at the induction
  variable, and no legacy descriptor metadata.
* Materialize the dynamic source `%n` authority onto the generated
  `tcrv.exec.kernel` and the runtime-element-count `tcrv.exec.runtime_param`
  so downstream planning/export can reject missing, partial, or stale runtime
  extent metadata before artifact output.
* Keep selected RVV `element_count` as selected vector-shape/chunk config and
  not as the dynamic source trip count. The dynamic `%n` becomes runtime AVL
  consumed by the materialized `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` control
  plane.
* Direct export and plan-and-export bundle must consume the same production
  path: source lowering first, then existing execution planning, then the same
  target artifact validation/export route.
* Generated direct and bundle source/header/object artifacts must expose the
  dynamic runtime AVL contract and must not emit the fixed `n == 16` trap for
  the dynamic route.
* Fresh RVV evidence must prove the dynamic vector-fronted artifact for
  runtime counts `7`, `16`, and `23`, and a focused regression must prove the
  fixed route still rejects mismatched counts.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits, and the task is current.
* [x] `--tcrv-lower-vector-rvv-i32-vadd-to-exec` accepts the new dynamic
      vector/SCF source fixture and lowers it to a parseable
      `tcrv.exec.kernel` with runtime `%n` source-authority metadata on both
      the kernel and `@abi_runtime_element_count`.
* [x] The lowering rejects malformed dynamic wrappers, including missing
      `%n`, stale/missing source authority metadata, wrong loop upper bound,
      wrong loop step, unsupported vector type, unsupported body arithmetic,
      or legacy descriptor metadata.
* [x] The execution-planning pipeline carries selected-plan metadata showing
      source `%n`/SCF upper-bound runtime AVL authority separately from
      selected RVV vector-shape metadata and descriptor-local `element_count`.
* [x] RVV direct artifact export rejects stale, missing, partial, or
      mismatched dynamic runtime extent metadata before source/header/object
      output.
* [x] Dynamic direct source/header/object artifacts expose the runtime AVL
      contract, selected RVV config, descriptor-local `element_count`, and
      callable runtime `n`, without the fixed `n == 16` trap.
* [x] Dynamic plan-and-export bundle source/header/object artifacts expose the
      same dynamic boundary through the same production path.
* [x] Existing fixed vector direct and bundle tests still show fixed source
      extent `16` and `n == 16` fail-closed behavior.
* [x] Focused lit covers the new dynamic `VectorToExec` path, the existing
      fixed vector regression, vector plan-and-export bundle,
      `RVVMicrokernel`, `TargetArtifactBundleExport`, `EmissionManifest`, and
      representative `LinalgToExec` regressions.
* [x] Focused C++ build covers `TianChenRVTransforms`, `TianChenRVTarget`,
      `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test` as applicable.
* [x] Exact direct and bundle artifact commands for the dynamic vector-fronted
      input are recorded.
* [x] Fresh `ssh rvv` evidence proves the dynamic vector-fronted artifact for
      runtime counts `7`, `16`, and `23`, plus a focused fixed-route mismatch
      rejection check.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] The task is finished/archived and one coherent commit records the round
      if complete.

## Definition Of Done

* Source `%n` / `scf.for` upper bound is the dynamic runtime AVL authority.
* Typed `tcrv_rvv` extension ops remain the compute/body authority.
* `tcrv.exec` remains an execution envelope for ABI/control organization, not
  a high-level vector compute dialect.
* Generated artifacts make dynamic source extent vs selected RVV config vs
  runtime AVL/VL readable without claiming generic vector lowering,
  performance, or dynamic-shape maturity.
* Hardware claims are bounded to named `ssh rvv` evidence paths and runtime
  counts.

## Completion Evidence

* Dynamic direct artifact commands:
  `build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/dynamic_vector_i32_vadd_runtime_avl_front_door/direct/vector-dynamic-i32-vadd.planned.mlir`;
  `build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/dynamic_vector_i32_vadd_runtime_avl_front_door/direct/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/dynamic_vector_i32_vadd_runtime_avl_front_door/direct/vector-dynamic-i32-vadd.c`;
  `build/bin/tcrv-translate --tcrv-export-target-header-artifact artifacts/tmp/dynamic_vector_i32_vadd_runtime_avl_front_door/direct/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/dynamic_vector_i32_vadd_runtime_avl_front_door/direct/vector-dynamic-i32-vadd.h`;
  `build/bin/tcrv-translate --tcrv-export-rvv-microkernel-object artifacts/tmp/dynamic_vector_i32_vadd_runtime_avl_front_door/direct/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/dynamic_vector_i32_vadd_runtime_avl_front_door/direct/vector-dynamic-i32-vadd.o`.
* Dynamic bundle artifact command:
  `build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/dynamic_vector_i32_vadd_runtime_avl_front_door/bundle test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`.
* Dynamic RVV evidence:
  `python3 scripts/rvv_microkernel_e2e.py --input test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir --lower-vector-i32-vadd-frontend --arithmetic-family i32-vadd --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --artifact-root artifacts/tmp/dynamic_vector_i32_vadd_runtime_avl_front_door/e2e --run-id dynamic_vector_i32_vadd_direct_rvv --overwrite --ssh-target rvv --timeout 120`.
* Dynamic RVV evidence result:
  `artifacts/tmp/dynamic_vector_i32_vadd_runtime_avl_front_door/e2e/dynamic_vector_i32_vadd_direct_rvv/evidence.json`
  has `status=success`, `ssh_evidence.success=true`,
  `selected_kernel=frontend_vector_dynamic_i32_vadd`, and
  `runtime_element_counts=[7,16,23]`.
* Fixed mismatch rejection command:
  `python3 scripts/rvv_microkernel_e2e.py --input test/Transforms/VectorToExec/vector-i32-vadd-to-exec.mlir --lower-vector-i32-vadd-frontend --arithmetic-family i32-vadd --runtime-count 7 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --artifact-root artifacts/tmp/dynamic_vector_i32_vadd_runtime_avl_front_door/e2e --run-id fixed_vector_i32_vadd_mismatch_reject --overwrite --dry-run --timeout 120`.
* Fixed mismatch rejection result:
  `rvv_microkernel_e2e: fixed source-vector extent external caller evidence requires exactly one runtime count equal to 16; observed 7`.

## Out Of Scope

* Generic MLIR vector lowering, scalable vector support, new vector arithmetic
  families, dtype matrices, performance claims, LLVM/RISC-V backend lowering,
  inline asm, or new hardware targets.
* Descriptor-driven computation, descriptor-to-C export, or moving arithmetic
  semantics into `tcrv.exec`.
* Prompt/report/helper-only closeout as the main achievement.

## Technical Notes

* Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior task context read:
  `.trellis/tasks/archive/2026-05/05-13-vector-frontend-runtime-avl-authority-boundary/prd.md`.
* Initial code inspection covered:
  `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h`,
  `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`,
  `include/TianChenRV/Target/RVV/RVVVectorShape.h`,
  `include/TianChenRV/Transforms/Passes.td`,
  `lib/Transforms/LowerLinalgRVVBinaryToExec.cpp`,
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`,
  `test/Transforms/VectorToExec/vector-i32-vadd-to-exec.mlir`, and
  `test/Target/TargetArtifactBundleExport/plan-vector-i32-vadd-and-export-target-artifact-bundle.mlir`.
