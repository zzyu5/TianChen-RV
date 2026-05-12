# RVV selected variant materialization to extension ops

## Goal

Materialize the bounded source-fronted RVV i32-vadd selected path through the
production selected-path materialization boundary so the selected RVV config and
runtime AVL/VL contract become concrete `tcrv_rvv` extension ops, not only
validated metadata. The first slice is the existing i32-vadd route family:
selected `tcrv.exec.variant` plus runtime ABI element-count authority must
produce one `tcrv_rvv.i32_vadd_microkernel` body with `setvl`, `with_vl`,
load/add/store dataflow, and export-visible selected-route agreement.

## What I Already Know

* Initial repository state for this task: `pwd` was
  `/home/kingdom/phdworks/TianchenRV`; the worktree was clean; HEAD was
  `55a0883 fix(rvv): selected config/runtime VL boundary authority`.
* No `.trellis/.current-task` existed at session start. This task was created
  from the Hermes Direction Brief before source edits.
* The previous selected-config/runtime-VL boundary task made selected RVV
  vector shape, runtime element-count C name, runtime AVL source/role, runtime
  VL source/scope, and descriptor-local element-count mirror metadata
  authoritative in planning, export validation, and tests.
* `RVVBinarySelectedConfigContract` already represents selected family, dtype,
  vector shape, SEW, LMUL, policy, selected path, descriptor-local element
  count, runtime element-count C name, dispatch guard C name, and runtime
  AVL/VL metadata.
* `materializeSelectedLoweringBoundaries` is the generic selected-path
  transform boundary. It routes selected variants through the origin plugin
  without RVV-specific core branching.
* The RVV plugin currently owns selected lowering-boundary materialization and
  has a helper that can build `setvl` / `with_vl` / load / arithmetic / store
  facts from `RVVBinarySelectedConfigContract`.
* The current task should harden and exercise this production path. It must not
  pivot to a descriptor-driven exporter, Python IR model, generic RVV backend
  claim, broad matrix, or standalone evidence packaging.

## Requirements

* The production selected-path materialization must consume the selected RVV
  variant, selected vector shape, typed RVV policy, runtime element-count ABI
  boundary, runtime AVL source/role, runtime VL source/scope, and
  descriptor-local element-count mirror before creating exportable artifacts.
* The bounded i32-vadd selected route must materialize a concrete
  `tcrv_rvv.i32_vadd_microkernel` body whose control/dataflow structure is
  derived from the selected contract:
  `tcrv_rvv.setvl` from the runtime element-count block argument,
  matching `tcrv_rvv.with_vl`, two `tcrv_rvv.i32_load` ops, one
  `tcrv_rvv.i32_add`, and one `tcrv_rvv.i32_store`.
* Missing or stale selected vector-shape metadata, runtime element-count ABI,
  runtime AVL/VL metadata, descriptor-local element-count mirror metadata, or
  RVV policy metadata must fail before source/header/object/bundle artifact
  output.
* The source-fronted bundle/front-door path must still emit the selected RVV
  source/header/object records through the same materialization authority.
* Generic transform/export tests must agree on selected-route metadata and the
  concrete `tcrv_rvv` body. Any compatibility path kept for existing explicit
  fixtures must remain explicitly bounded.

## Acceptance Criteria

* [ ] Trellis PRD, implement context, and check context exist before source
      edits, and the task is started as the current Trellis task.
* [ ] The selected-path materialization code has one production C++ authority
      from selected RVV config/runtime VL contract to i32-vadd `tcrv_rvv`
      body materialization.
* [ ] The generated i32-vadd body visibly contains selected `setvl`,
      matching `with_vl`, `i32_load`, `i32_add`, and `i32_store` operations
      derived from the selected config/runtime contract.
* [ ] Focused negative coverage proves stale or missing selected config,
      runtime element-count ABI, runtime AVL/VL metadata, descriptor-local
      element-count mirror metadata, or RVV policy metadata fails before
      artifact output.
* [ ] The source-fronted plan-and-export bundle path still emits the proven
      RVV source/header/object artifacts and records selected-route metadata
      consistently with the generic export front door.
* [ ] Focused build, target artifact export test when target/export code
      changes, focused lit for the affected areas, exact `tcrv-opt` or
      `tcrv-translate` bounded i32-vadd command, `git diff --check`,
      `git diff --cached --check`, and Trellis validation pass.
* [ ] The task is finished/archived and one coherent commit records the round
      if complete.

## Definition Of Done

* Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
* Python remains tooling only.
* Computation semantics do not move into `tcrv.exec`, descriptors, generated
  harnesses, or Python.
* Shared core transforms do not gain RVV-family semantic branches; generic
  passes continue routing through plugin interfaces.
* Descriptor metadata remains a legacy mirror/quarantine surface, not the
  selected computation authority.
* No runtime correctness, hardware, or performance claim is made without fresh
  `ssh rvv` evidence.

## Out Of Scope

* New arithmetic families, dtype expansion, generic RVV backend claims, broad
  smoke matrices, or performance evidence.
* Moving the current main route away from extension family ops -> EmitC ->
  intrinsic/runtime C/C++.
* Reintroducing descriptor-only computation or direct descriptor-to-C export as
  production authority.
* Refactoring unrelated scalar/offload/IME/plugin code.

## Technical Notes

* Specs read for this task:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior task context read:
  `.trellis/tasks/archive/2026-05/05-13-rvv-selected-config-runtime-vl-boundary/prd.md`.
* Relevant journal context read:
  `.trellis/workspace/codex/journal-3.md` session 21 and
  `.trellis/workspace/codex/journal-2.md` session 34.
* Initial source inspection focused on:
  `include/TianChenRV/Transforms/VariantMaterialization.h`,
  `include/TianChenRV/Transforms/VariantSelection.h`,
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`,
  `include/TianChenRV/Target/RVV/RVVVectorShape.h`,
  `lib/Transforms/VariantMaterialization.cpp`,
  `lib/Transforms/VariantSelection.cpp`,
  `lib/Transforms/LoweringBoundary.cpp`,
  `lib/Transforms/ExecutionPlanningPipeline.cpp`,
  `lib/Transforms/ExecutionPlanCoherence.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Transforms/LowerLinalgRVVBinaryToExec.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `test/Target/RVVMicrokernel/rvv-microkernel-auto-materialization.mlir`,
  `test/Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir`,
  `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`,
  `test/Transforms/LoweringBoundary/`,
  `test/Target/TargetArtifactBundleExport/`, and
  `test/Target/EmissionManifest/`.
