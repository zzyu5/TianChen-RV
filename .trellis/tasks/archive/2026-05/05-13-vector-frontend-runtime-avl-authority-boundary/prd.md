# Vector frontend runtime AVL authority boundary

## Goal

Repair and harden the bounded vector i32-vadd front door so the fixed
`vector<16xi32>` source shape, selected RVV finite config, descriptor-local
`element_count`, and runtime callable `n`/AVL boundary are modeled as distinct
layers and cannot be silently conflated before artifact export or runtime
evidence.

The production path for this task remains:

```text
marked vector/arith i32 add source wrapper
  -> tcrv.exec.kernel with finite binary ABI roles
  -> RVV plugin selected finite i32-vadd variant
  -> materialized tcrv_rvv.i32_vadd_microkernel body
  -> common EmitC route
  -> source/header/object or bundle artifacts
  -> ssh rvv evidence
```

## What I Already Know

* Initial repository state for this task: `pwd` is
  `/home/kingdom/phdworks/TianchenRV`, the worktree was clean, and HEAD was
  `6877c26 feat(rvv): add bounded vector i32 vadd front door`.
* No `.trellis/.current-task` existed at session start. This task was created
  from the Hermes Direction Brief before source edits.
* The previous archived bounded vector front-door task accepted exactly one
  `func.func` vector wrapper with `vector.transfer_read`,
  `arith.addi`, and `vector.transfer_write` over `vector<16xi32>`.
* That previous task reused the finite binary `tcrv.exec.mem_window` plus
  `tcrv.exec.runtime_param @abi_runtime_element_count` ABI plan and collected
  `ssh rvv` evidence for runtime counts `7,16,23`.
* Current specs explicitly separate hardware facts, compile-time selected RVV
  vector-shape config, descriptor-local `element_count`, and runtime ABI
  values such as `n`, AVL, and VL.
* The current vector front door checks `vector<16xi32>` structurally but does
  not preserve that accepted source extent as an auditable IR/artifact
  authority after lowering.
* The current generated RVV source already records selected config/runtime AVL
  metadata and uses `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` from the
  materialized typed RVV body.

## Decision

This round keeps the source fixed-width. It does not add dynamic vector extent
semantics. Therefore vector-fronted artifacts must treat `16` as the accepted
source-frontdoor extent and constrain runtime `n` to exactly that value.

Multiple runtime counts are still valid for linalg-fronted dynamic source
fixtures. For the vector-fronted fixture in this task, fresh hardware evidence
must use only runtime count `16`.

## Requirements

* Preserve compiler implementation in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
* Keep the vector frontend bounded to exactly the existing i32-vadd source
  shape; do not add generic vector lowering, new vector ops, dtype/shape
  matrices, descriptor-driven compute, or a direct descriptor-to-C exporter.
* Record the accepted vector source extent as source-frontdoor authority on
  the lowered execution boundary.
* Keep selected RVV vector-shape config and descriptor-local
  `tcrv_rvv.element_count` separate from runtime `n`/AVL.
* Require the selected RVV i32-vadd element count and selected-plan metadata to
  agree with the fixed source extent for vector-fronted kernels.
* Constrain generated direct and plan-and-export bundle artifacts so the
  public runtime callable fails closed for vector-fronted runtime counts that
  differ from the fixed source extent.
* Make missing, stale, partial, or inconsistent source extent/runtime-count
  metadata fail before source/header/object or bundle artifact output.
* Keep the plan-and-export bundle front door on the same production path as
  direct export: vector source lowering first, then planning, then the same
  exporter validation.
* Expose the resulting boundary in source/header/object comments and bundle
  index metadata clearly enough to audit.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits, and the task is started as the current Trellis task.
* [x] `--tcrv-lower-vector-rvv-i32-vadd-to-exec` lowers the vector fixture to
      a `tcrv.exec.kernel` that records fixed source vector extent `16` and a
      runtime-element-count constraint distinct from selected RVV config.
* [x] The execution-planning pipeline carries matching selected-plan metadata
      for the vector source extent and runtime count constraint.
* [x] RVV target export rejects stale, missing, or mismatched vector source
      extent/runtime count metadata before artifact output.
* [x] Direct source/header/object artifacts expose the fixed source extent,
      selected RVV config, descriptor-local element count, runtime `n`, and
      `n == 16` constraint.
* [x] Plan-and-export bundle source/header/object artifacts expose the same
      boundary and use the same production lowering/planning/export path.
* [x] Generated direct RVV callable and RVV+scalar dispatch callable fail
      closed at runtime if vector-fronted `n` is not the fixed source extent.
* [x] Focused lit covers `VectorToExec`, vector plan-and-export bundle,
      representative `RVVMicrokernel`, `TargetArtifactBundleExport`,
      `EmissionManifest`, and `LinalgToExec` regressions.
* [x] Focused C++ build covers `TianChenRVTransforms`, `TianChenRVTarget`,
      `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test` as applicable.
* [x] Exact direct and bundle artifact commands for the vector-fronted input
      are recorded.
* [x] Fresh `ssh rvv` evidence for the vector-fronted artifact uses runtime
      count `16` only and records why `7` and `23` are no longer truthful for
      this fixed source fixture.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] The task is finished/archived and one coherent commit records the round
      if complete.

## Definition Of Done

* The source vector/arith body and typed `tcrv_rvv` extension ops remain the
  compute authority.
* `tcrv.exec` remains an execution envelope for ABI/control organization, not
  a high-level vector compute dialect.
* Generated artifacts make fixed source extent vs selected RVV config vs
  runtime AVL/VL readable without claiming generic vector lowering,
  performance, or dynamic shape maturity.
* Hardware claims are bounded to the named `ssh rvv` evidence path and are not
  generalized beyond the fixed vector i32-vadd source fixture.

## Out Of Scope

* Dynamic vector extent modeling, generic MLIR vector lowering, scalable-vector
  support, new vector arithmetic families, new dtype support, broad matrices,
  performance claims, LLVM/RISC-V backend lowering, inline asm, or new hardware
  targets.
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
  `.trellis/tasks/archive/2026-05/05-13-bounded-vector-to-rvv-frontdoor/prd.md`.
* Relevant journal entries read from `.trellis/workspace/codex/journal-4.md`:
  RVV runtime AVL/VL boundary authority, source-fronted RVV runtime artifact
  proof, RVV materialized i32-vadd artifact runtime ABI proof, and bounded
  vector-to-RVV front-door notes.
* Initial code inspection covered:
  `include/TianChenRV/Transforms/Passes.td`,
  `lib/Transforms/LowerLinalgRVVBinaryToExec.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Transforms/VectorToExec/vector-i32-vadd-to-exec.mlir`,
  `test/Transforms/VectorToExec/vector-i32-vadd-invalid.mlir`,
  `test/Target/TargetArtifactBundleExport/plan-vector-i32-vadd-and-export-target-artifact-bundle.mlir`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`, and the EmitC source-authority
  materializer.

## Completion Evidence

* Boundary chosen: the vector front door remains fixed-width. The accepted
  `vector<16xi32>` source extent is recorded as source-frontdoor authority,
  selected RVV `element_count` must equal `16`, and generated public runtime
  callables trap before executing if runtime `n != 16`.
* Direct artifact command:
  `build/bin/tcrv-opt test/Transforms/VectorToExec/vector-i32-vadd-to-exec.mlir --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/vector_frontend_runtime_avl_authority_boundary/direct/vector-i32-vadd.planned.mlir`
  followed by `build/bin/tcrv-translate --tcrv-export-target-source-artifact`,
  `--tcrv-export-target-header-artifact`, and
  `--tcrv-export-rvv-microkernel-object` over that planned MLIR.
* Bundle artifact command:
  `build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/vector_frontend_runtime_avl_authority_boundary/bundle test/Target/TargetArtifactBundleExport/plan-vector-i32-vadd-and-export-target-artifact-bundle.mlir`
* Fresh RVV evidence:
  `artifacts/tmp/vector_frontend_runtime_avl_authority_boundary/e2e/20260513T-vector-frontend-runtime-avl-authority-boundary/evidence.json`
  with `runtime_element_counts = [16]`,
  `fixed_source_extent_contract.source_vector_extent = 16`, and
  `stdout_marker_observed = true`.
* Rejected inconsistent cases: stale selected-plan source extent `15`,
  missing runtime-param source extent metadata, and runner-requested runtime
  count `7` for a fixed vector source all fail before accepted evidence.
