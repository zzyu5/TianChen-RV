# Bounded MLIR vector-to-RVV materialization

## Goal

Build the bounded i32-vadd source/selected boundary proof so a non-final
frontend source path materializes a selected RVV variant into plugin-owned
`tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, `tcrv_rvv.i32_load`,
`tcrv_rvv.i32_add`, and `tcrv_rvv.i32_store` ops before reusing the existing
RVV target artifact/export route.

## What I Already Know

* Current HEAD at task start is
  `c9f0441 fix(rvv): consume dispatch avl contract`; the worktree was clean
  before the task was created.
* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief and set as current before source edits.
* The previous archived PRD
  `.trellis/tasks/archive/2026-05/05-13-rvv-dispatch-avl-vl-contract-consumption/prd.md`
  completed dispatch consumption of the direct RVV selected-config/runtime AVL
  contract without changing executable behavior.
* Existing production code already contains descriptorless selected RVV
  microkernel materialization in
  `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp` and
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`.
* Existing positive tests already cover direct frontend-to-RVV source export,
  and the bundle test starts from a `linalg.generic` source boundary, but the
  bundle IR check does not directly prove the materialized i32-vadd
  `setvl/with_vl/load/add/store` body.
* The i64 selected-emission attachment path validates selected-vector-shape
  mirror metadata on the explicit microkernel op. The i32 path validates the
  structured body/control-plane against the selected shape but does not
  validate the microkernel op's selected-vector-shape mirror metadata before
  export.

## Requirements

* The bounded i32-vadd frontend/source boundary must materialize typed RVV
  microkernel ops from selected RVV variant metadata, not rely on an input that
  already contains the final RVV body.
* The selected RVV i32 microkernel body must expose exactly one runtime AVL
  body argument, one `tcrv_rvv.setvl`, one `tcrv_rvv.with_vl`, two loads, one
  `tcrv_rvv.i32_add`, and one store for the bounded i32-vadd slice.
* The direct artifact route must continue to consume the typed RVV body and
  expose selected binary config, runtime AVL provenance, EmitC route
  provenance, and callable ABI order.
* Export must fail closed when the selected RVV microkernel carries stale or
  incomplete selected-vector-shape mirror metadata, rather than accepting a
  descriptor/config mismatch beside an otherwise valid body.
* Descriptor-only compute authority must remain rejected before RVV source,
  header, object, or bundle artifact output.
* This task must stay in the C++/MLIR/TableGen/CMake/lit/FileCheck compiler
  stack. Python may only be used for bounded test text mutation or task
  tooling.

## Acceptance Criteria

* [x] PRD and Trellis context are created before source edits.
* [x] i32 selected-emission attachment validation rejects stale or incomplete
      selected-vector-shape metadata on the matched `tcrv_rvv.i32_*`
      microkernel op, matching the existing i64 fail-closed behavior.
* [x] A focused positive pipeline/bundle test that starts before the final RVV
      body checks materialized `tcrv_rvv.i32_vadd_microkernel`,
      `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, `tcrv_rvv.i32_load`,
      `tcrv_rvv.i32_add`, and `tcrv_rvv.i32_store` ops before artifact export.
* [x] Focused negative coverage fails before RVV artifact output when the
      microkernel selected-vector-shape mirror metadata is stale or incomplete.
* [x] Existing descriptor-only rejection coverage remains valid.
* [x] Generated source/header/object or bundle evidence still exposes
      selected binary config, runtime AVL provenance, EmitC route provenance,
      and callable ABI order.
* [x] Focused build, C++ target artifact export test if touched targets require
      it, focused lit for RVV materialization/RVVMicrokernel/EmissionManifest/
      TargetArtifactBundleExport affected routes, exact manual
      `tcrv-opt`/`tcrv-translate` commands, `git diff --check`,
      `git diff --cached --check`, and Trellis validation pass.
* [x] If executable correctness is claimed, one bounded `ssh rvv` run is
      collected. If this round only tightens compiler validation/provenance,
      final notes state that no runtime correctness claim is made.
* [x] Task is finished/archived and one coherent commit records the round if
      complete.

## Definition Of Done

* Compiler behavior remains implemented in C++/MLIR/LLVM/TableGen/CMake/lit.
* No descriptor-driven computation or direct descriptor-to-C fallback is added.
* No computation semantics are added to `tcrv.exec`.
* No RVV-specific semantic branch is added to shared core orchestration.
* The new check stays plugin/target-local to RVV selected-emission planning.
* The positive test proves a real production path, not helper-only metadata.

## Out Of Scope

* Generic MLIR vector lowering, arbitrary tensor/tile IR, new dtype/family
  expansion, performance evidence, broad smoke matrices, or a generic RVV
  backend claim.
* New runtime hardware correctness claims unless generated executable behavior
  changes and is validated on `ssh rvv`.
* Python implementation of compiler internals.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Initial source/test surfaces inspected:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`,
  `lib/Transforms/VariantMaterialization.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `test/Target/RVVMicrokernel/`,
  `test/Target/EmissionManifest/`, and
  `test/Target/TargetArtifactBundleExport/plan-linalg-i32-vadd-and-export-target-artifact-bundle.mlir`.

## Completion Notes

* Added plugin-local i32 selected-emission validation in
  `RVVBinarySelectedEmissionPlanning.cpp`: when an explicit selected i32 RVV
  microkernel is matched, its `selected_vector_*` mirror metadata is now
  validated against the selected RVV vector-shape contract before emission
  planning or artifact export can proceed. This mirrors the existing i64
  selected-emission behavior.
* Extended the positive plan-and-export bundle test so the IR phase starting
  from `linalg.generic` proves materialization of
  `tcrv_rvv.i32_vadd_microkernel`, runtime AVL block argument,
  `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, two `tcrv_rvv.i32_load` ops,
  `tcrv_rvv.i32_add`, and `tcrv_rvv.i32_store` before the bundle source/header/
  object export checks.
* Added `rvv-microkernel-selected-shape-metadata-fails.mlir` with split-input
  negative coverage for complete-but-stale and incomplete i32 microkernel
  selected-vector-shape mirror metadata. Both fail before source emission.
* Existing descriptor-only rejection coverage remains in
  `rvv-microkernel-descriptor-only-production-rejects.mlir` and the focused
  RVVMicrokernel lit slice passed.
* Generated executable behavior did not intentionally change. This round
  tightens compiler validation and FileCheck coverage for the same bounded
  i32-vadd artifact route, so no new `ssh rvv` correctness or performance
  claim is made.

## Manual Artifact Evidence

* Artifact directory:
  `artifacts/tmp/bounded_vector_to_rvv_materialization/direct/`.
* Materialized IR from a non-final linalg/source boundary:
  `build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/bounded_vector_to_rvv_materialization/direct/materialized_i32_vadd.mlir`
* Direct RVV source artifact:
  `build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-source-artifact > artifacts/tmp/bounded_vector_to_rvv_materialization/direct/direct_i32_vadd.c`
* Direct RVV header artifact:
  `build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-header-artifact > artifacts/tmp/bounded_vector_to_rvv_materialization/direct/direct_i32_vadd.h`
* Direct RVV object artifact:
  `build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-artifact > artifacts/tmp/bounded_vector_to_rvv_materialization/direct/direct_i32_vadd.o`
* `rg` evidence from the materialized IR found
  `tcrv_rvv.i32_vadd_microkernel`, `tcrv_rvv.setvl`,
  `tcrv_rvv.with_vl`, two `tcrv_rvv.i32_load` ops,
  `tcrv_rvv.i32_add`, and `tcrv_rvv.i32_store`.
* `rg` evidence from generated source found `selected_binary_config`,
  `control_plane_runtime_avl`, `emitc_route`, `emitc_route_source_ops`,
  ordered `runtime_abi_parameter[0..3]`, `tcrv_emitc.source_authority`, and
  `__riscv_vadd_vv_i32m1`.
* `file` reports the direct object as an ELF 64-bit RISC-V relocatable object.
  `llvm-readobj-20` reports `Format: elf64-littleriscv`, `Arch: riscv64`,
  `Type: Relocatable`, `Machine: EM_RISCV`, and symbol
  `tcrv_rvv_i32_vadd_microkernel_frontend_i32_vadd_rvv_first_slice` with no
  `main` symbol.

## Validation

* `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'RVVMicrokernel|EmissionManifest|TargetArtifactBundleExport|LinalgToExec'`
  with 72/72 selected tests passed.
* Manual artifact commands listed above.
* `git diff --check`
* `git diff --cached --check`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-05-13-bounded-mlir-vector-to-rvv-materialization`
