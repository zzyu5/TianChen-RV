# Source frontend lowering ownership split

## Goal

Split the existing bounded source-frontdoor production path into a clear
source-frontend lowering owner with reusable adapter boundaries for the current
linalg, fixed vector, and dynamic vector i32-vadd routes, while preserving
current RVV selected-body, EmitC, artifact, dispatch, and source-authority
behavior.

The intended production ownership is:

```text
marked source IR adapter
  -> common finite-binary source frontend contract
  -> tcrv.exec ABI / selected boundary
  -> RVV plugin materialization
  -> direct and plan-and-export target artifact routes
```

This task is an ownership/refactor round, not a new operation, dtype, generic
vector backend, or runtime/performance milestone.

## What I Already Know

* Initial repository state for this task: `pwd` is
  `/home/kingdom/phdworks/TianchenRV`, the worktree was clean, and HEAD was
  `b0f1cb4 feat(vector): add dynamic runtime AVL front door`.
* No `.trellis/.current-task` existed at session start. This task was created
  from the Hermes Direction Brief before source edits.
* The previous fixed-vector task proved a bounded `vector<16xi32>` i32-vadd
  front door and requires generated public callables to fail closed unless
  runtime `n == 16`.
* The previous dynamic-vector task proved a bounded vector/SCF i32-vadd front
  door where source `%n` is the dynamic runtime AVL authority and remote
  `ssh rvv` evidence covered counts `7`, `16`, and `23`.
* The existing production implementation has linalg, fixed-vector, and
  dynamic-vector source-frontdoor behavior accumulated in
  `lib/Transforms/LowerLinalgRVVBinaryToExec.cpp`, while public tool
  plan-and-export routing is manually chained through linalg/vector-specific
  lowering steps.
* `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h` already owns
  important shared ABI/source-authority construction helpers and is the natural
  boundary to grow into a source-frontend contract.

## Requirements

* Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python remains limited to tooling and evidence support.
* Move production source-frontdoor semantics out of linalg-named ownership.
  Linalg-specific names must no longer own vector/source-frontdoor semantics in
  the default path.
* Factor common finite-binary source frontend validation, source-authority
  metadata, runtime extent handling, and `tcrv.exec` ABI construction behind a
  shared source-frontend contract.
* Keep linalg, fixed vector, and dynamic vector source adapters explicit and
  bounded. Do not add a generic MLIR vector backend, new dtype, new op, or new
  finite-family matrix row.
* Preserve fixed-vector `n == 16` enforcement and dynamic `%n` runtime AVL
  authority while refactoring.
* Keep `tcrv.exec` as an execution/ABI/control envelope only. Arithmetic
  compute authority remains the selected/materialized `tcrv_rvv` family body
  consumed through the common EmitC route.
* Ensure `tcrv-opt` direct lowering and `tcrv-translate` plan-and-export use the
  same production source-frontdoor route instead of divergent hand-chained
  lowering.
* Preserve fail-closed behavior for stale/missing source authority before
  source/header/object or bundle artifact output.
* Generated direct and bundle artifacts for representative linalg, fixed
  vector, and dynamic vector routes must remain equivalent to the current
  behavior.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits, and the task is started as current.
* [x] A production source-frontend lowering owner exists with neutral naming
      and owns common finite-binary source adapter routing.
* [x] The linalg-named pass/file no longer owns vector or source-frontdoor
      semantics in the default path; compatibility names may delegate to the
      neutral owner when required by existing CLI/tests.
* [x] Common finite-binary source frontend validation, ABI construction,
      source-authority metadata, and fixed/dynamic runtime extent handling are
      factored behind the shared source-frontend contract.
* [x] Linalg, fixed-vector, and dynamic-vector adapters remain explicit,
      bounded, and covered; the dynamic vector adapter is completed at minimum
      if the full adapter split proves too large.
* [x] `tcrv-opt` and `tcrv-translate` plan-and-export use the same production
      source-frontdoor route for linalg, fixed vector, and dynamic vector
      fixtures.
* [x] Existing source-authority fail-closed behavior remains intact for stale,
      missing, partial, or mismatched fixed/dynamic extent metadata before
      artifact output.
* [x] Direct and bundle artifact commands are recorded for one linalg route,
      the fixed vector route, and the dynamic vector route.
* [x] Focused build covers `TianChenRVTransforms`, `TianChenRVTarget`,
      `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test` as applicable.
* [x] Focused lit covers `LinalgToExec`, `VectorToExec`, fixed and dynamic
      vector plan-and-export bundles, `RVVMicrokernel`,
      `TargetArtifactBundleExport`, `EmissionManifest`, and lowering-boundary
      regressions.
* [x] If generated artifact or plan-and-export routing materially changes,
      fresh `ssh rvv` evidence is collected for the dynamic vector route;
      otherwise the PRD/final report gives a truthful local artifact-equivalence
      explanation and references the existing dynamic evidence path.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] The task is finished/archived and one coherent commit records the round
      if complete.

## Definition Of Done

* Source frontend ownership is production code, not helper-only scaffolding.
* The shared contract is consumed by the direct transform path and the
  plan-and-export front door in the same round.
* The refactor does not weaken fixed vector or dynamic vector source extent
  authority.
* Existing RVV selected-body, common EmitC, source/header/object, bundle, and
  dispatch semantics remain behaviorally equivalent for the covered routes.

## Completed This Round

* Added the production neutral
  `--tcrv-lower-source-rvv-binary-to-exec` pass and registered it in
  `tcrv-opt`.
* Moved the transform implementation from
  `lib/Transforms/LowerLinalgRVVBinaryToExec.cpp` to
  `lib/Transforms/LowerSourceRVVBinaryToExec.cpp`.
* Rewired `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` to run
  the single production source-frontdoor pass instead of hand-chaining vector
  and linalg-specific passes.
* Kept the old linalg and vector pass names as explicit adapter/compatibility
  entry points.
* Factored the common finite-binary source lowering contract into
  `FiniteBinarySourceFrontendLoweringContract` in
  `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h`, covering
  runtime ABI mem windows, runtime element-count ABI, fixed source extent
  metadata, and dynamic source `%n` runtime-AVL metadata.
* Reworked linalg and vector adapters to produce a shared
  `SourceFrontendLoweringRequest`, so selected target/profile validation,
  supplemental provider imports, duplicate kernel checks, exec kernel
  creation, ABI materialization, and source-authority runtime-param metadata
  are handled once.
* Added representative `--tcrv-lower-source-rvv-binary-to-exec` lit coverage
  for linalg, fixed-vector, and dynamic-vector direct lowering.
* Generated representative direct source/header/object artifacts and
  plan-and-export bundles for linalg i32-vadd, fixed vector i32-vadd, and
  dynamic vector i32-vadd under
  `artifacts/tmp/source_frontend_lowering_ownership_split/`.
* Collected fresh `ssh rvv` evidence for the dynamic vector route through the
  plan-and-export bundle front door.

## Exact Artifact Commands

```bash
build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/source_frontend_lowering_ownership_split/direct/linalg/linalg-i32-vadd.planned.mlir
build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/source_frontend_lowering_ownership_split/direct/linalg/linalg-i32-vadd.planned.mlir > artifacts/tmp/source_frontend_lowering_ownership_split/direct/linalg/linalg-i32-vadd.c
build/bin/tcrv-translate --tcrv-export-target-header-artifact artifacts/tmp/source_frontend_lowering_ownership_split/direct/linalg/linalg-i32-vadd.planned.mlir > artifacts/tmp/source_frontend_lowering_ownership_split/direct/linalg/linalg-i32-vadd.h
build/bin/tcrv-translate --tcrv-export-target-artifact artifacts/tmp/source_frontend_lowering_ownership_split/direct/linalg/linalg-i32-vadd.planned.mlir > artifacts/tmp/source_frontend_lowering_ownership_split/direct/linalg/linalg-i32-vadd.o

build/bin/tcrv-opt test/Transforms/VectorToExec/vector-i32-vadd-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/source_frontend_lowering_ownership_split/direct/vector_fixed/vector-i32-vadd.planned.mlir
build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/source_frontend_lowering_ownership_split/direct/vector_fixed/vector-i32-vadd.planned.mlir > artifacts/tmp/source_frontend_lowering_ownership_split/direct/vector_fixed/vector-i32-vadd.c
build/bin/tcrv-translate --tcrv-export-target-header-artifact artifacts/tmp/source_frontend_lowering_ownership_split/direct/vector_fixed/vector-i32-vadd.planned.mlir > artifacts/tmp/source_frontend_lowering_ownership_split/direct/vector_fixed/vector-i32-vadd.h
build/bin/tcrv-translate --tcrv-export-target-artifact artifacts/tmp/source_frontend_lowering_ownership_split/direct/vector_fixed/vector-i32-vadd.planned.mlir > artifacts/tmp/source_frontend_lowering_ownership_split/direct/vector_fixed/vector-i32-vadd.o

build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/source_frontend_lowering_ownership_split/direct/vector_dynamic/vector-dynamic-i32-vadd.planned.mlir
build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/source_frontend_lowering_ownership_split/direct/vector_dynamic/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/source_frontend_lowering_ownership_split/direct/vector_dynamic/vector-dynamic-i32-vadd.c
build/bin/tcrv-translate --tcrv-export-target-header-artifact artifacts/tmp/source_frontend_lowering_ownership_split/direct/vector_dynamic/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/source_frontend_lowering_ownership_split/direct/vector_dynamic/vector-dynamic-i32-vadd.h
build/bin/tcrv-translate --tcrv-export-target-artifact artifacts/tmp/source_frontend_lowering_ownership_split/direct/vector_dynamic/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/source_frontend_lowering_ownership_split/direct/vector_dynamic/vector-dynamic-i32-vadd.o

build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/source_frontend_lowering_ownership_split/bundle/linalg test/Target/TargetArtifactBundleExport/plan-linalg-i32-vadd-and-export-target-artifact-bundle.mlir > artifacts/tmp/source_frontend_lowering_ownership_split/bundle/linalg/stdout.txt
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/source_frontend_lowering_ownership_split/bundle/vector_fixed test/Target/TargetArtifactBundleExport/plan-vector-i32-vadd-and-export-target-artifact-bundle.mlir > artifacts/tmp/source_frontend_lowering_ownership_split/bundle/vector_fixed/stdout.txt
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/source_frontend_lowering_ownership_split/bundle/vector_dynamic test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir > artifacts/tmp/source_frontend_lowering_ownership_split/bundle/vector_dynamic/stdout.txt
```

## Dynamic RVV Evidence

Command:

```bash
python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --input test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir --lower-vector-i32-vadd-frontend --arithmetic-family i32-vadd --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --artifact-root artifacts/tmp/source_frontend_lowering_ownership_split/e2e --run-id dynamic_vector_plan_export_source_owner --overwrite --ssh-target rvv --timeout 120
```

Result:

* Evidence file:
  `artifacts/tmp/source_frontend_lowering_ownership_split/e2e/dynamic_vector_plan_export_source_owner/evidence.json`.
* `status = success`, `ssh_evidence.success = true`,
  `selected_kernel = frontend_vector_dynamic_i32_vadd`,
  `runtime_element_counts = [7, 16, 23]`.
* Both bundle-source and bundle-object remote callers printed
  `tcrv_rvv_microkernel_external_abi_ok counts=7,16,23`.

## Checks Run

* `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'LinalgToExec|VectorToExec|plan-vector-dynamic-i32-vadd|plan-vector-i32-vadd|RVVMicrokernel|TargetArtifactBundleExport|EmissionManifest|LoweringBoundary'`
  with 92/92 selected tests passed.
* Direct and bundle artifact commands listed above.
* Dynamic vector `ssh rvv` command listed above.
* `file` on generated direct linalg/fixed-vector/dynamic-vector objects and
  dynamic bundle object confirmed RISC-V ELF relocatables.
* `git diff --check`
* Trellis validation before finish/archive.

## Out Of Scope

* New finite-family matrix rows, new arithmetic ops, new dtypes, generic MLIR
  vector backend work, scalable-vector maturity claims, performance claims, or
  new hardware targets.
* Descriptor-driven computation, descriptor-to-C export, Python compiler
  implementation, or moving compute semantics into `tcrv.exec`.
* Report-only, helper-only, prompt-only, or broad smoke-test closeout as the
  main achievement.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior task context read:
  `.trellis/tasks/archive/2026-05/05-13-dynamic-vector-i32-vadd-runtime-avl-front-door/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-13-bounded-vector-to-rvv-frontdoor/prd.md`.
* Initial code files to inspect before implementation:
  `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h`,
  `include/TianChenRV/Transforms/Passes.td`,
  `lib/Transforms/LowerLinalgRVVBinaryToExec.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Transforms/LinalgToExec/`,
  `test/Transforms/VectorToExec/`,
  `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`,
  and
  `test/Target/TargetArtifactBundleExport/plan-vector-i32-vadd-and-export-target-artifact-bundle.mlir`.
