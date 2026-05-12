# RVV selected config and runtime VL boundary

## Goal

Make the bounded source/linalg i32-vadd to selected RVV path carry one
production C++ selected-boundary contract for compile-time RVV config and
runtime element-count/AVL/VL authority, then consume that contract from
plugin-owned `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` materialization through
target artifact source/header/object export.

This round stays within the existing IR shape. It does not change RVV op
definitions unless the current C++ contract proves insufficient.

## What I Already Know

* Initial repository state for this task: `pwd` was
  `/home/kingdom/phdworks/TianchenRV`, worktree was clean, and HEAD was
  `5150dba fix(rvv): unify plugin-owned artifact route authority`.
* No `.trellis/.current-task` existed at session start. This task was created
  from the Direction Brief before source edits.
* The previous route-authority task made RVV source/header/object route facts
  plugin-owned and shared by generic bundle export plus direct compatibility
  routes.
* The current code already has `RVVBinarySelectedConfigContract`, but selected
  vector-shape metadata, runtime AVL/VL selected-plan metadata, descriptor-local
  element-count metadata, and export comments are still assembled through
  several separate helpers.
* Existing materialization already creates an i32 body with one runtime index
  argument, one `tcrv_rvv.setvl`, one `tcrv_rvv.with_vl`, two loads, one
  arithmetic op, and one store. This task should harden the contract boundary
  around that behavior rather than adding a new lowering route.
* Generated runtime semantics are not expected to change if this round only
  tightens selected-boundary validation and provenance.

## Requirements

* The selected boundary must expose or resolve, through production C++ contract
  code, the selected family, dtype, selected variant, selected role, SEW, LMUL,
  tail policy, mask policy, selected vector shape, vector suffix, setvl suffix,
  runtime element-count C name, runtime AVL source/role, runtime VL source, VL
  scope, and descriptor-local element count.
* RVV binary materialization must consume that contract when constructing the
  bounded `setvl` / `with_vl` / load / arithmetic / store dataflow.
* RVV emission-plan selected metadata and target/export selected metadata must
  be validated from the same contract fields instead of reconstructing
  conflicting shape/runtime/descriptor facts.
* Artifact export must fail before source/header/object/bundle output when
  selected vector shape, runtime AVL source, runtime element-count C name, or
  descriptor-local element-count mirror metadata is missing or stale.
* The source-fronted plan-and-export i32-vadd bundle path must still emit the
  proven source/header/object artifact records.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits.
* [x] `RVVBinarySelectedConfigContract` owns helper surfaces for selected
      vector-shape metadata, runtime AVL/VL selected-boundary metadata, and
      descriptor-local element-count mirror checks.
* [x] Plugin-selected emission planning builds selected-plan metadata from the
      selected config/runtime VL contract and includes descriptor-local count
      provenance for the bounded binary path.
* [x] Target artifact source route validation consumes the same contract and
      rejects missing/stale runtime AVL/VL metadata, selected vector-shape
      metadata, runtime element-count C name, or descriptor-local count before
      output.
* [x] Generated source comments and bundle index records expose the selected
      config/runtime VL contract fields for the source-fronted i32-vadd path.
* [x] Focused negative coverage proves stale or missing descriptor-local count
      metadata fails before artifact output.
* [x] Focused build, `tianchenrv-target-artifact-export-test`, focused lit,
      exact plan-and-export bundle command, `git diff --check`,
      `git diff --cached --check`, and Trellis validation pass.
* [x] If generated runtime behavior changes materially, one bounded `ssh rvv`
      correctness run is collected; otherwise the final report states why the
      prior runtime proof still applies.
* [ ] The task is finished/archived and one coherent commit records the round
      if complete.

## Definition Of Done

* Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit.
* Python remains runner/tooling only.
* No computation semantics move into descriptors, Python, or `tcrv.exec`.
* No RVV-specific semantic branch is added to shared core orchestration.
* No new arithmetic family, dtype matrix, generic RVV backend claim,
  performance claim, broad smoke matrix, or standalone evidence repackaging is
  treated as completion.
* Generated binary/runtime artifacts remain under `artifacts/tmp` or test temp
  directories and are not committed.

## Out Of Scope

* New RVV arithmetic families, dtype expansion, generic linalg lowering, or
  arbitrary RVV vector modeling.
* ODS/op verifier changes unless the C++ boundary cannot close the required
  fail-closed cases.
* Direct descriptor-to-C computation or descriptor-driven authority.
* New hardware runtime evidence unless the generated runtime C semantics change
  materially.

## Technical Notes

* Specs read for this task:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior task context read:
  `.trellis/tasks/archive/2026-05/05-13-plugin-owned-rvv-artifact-route-authority/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-13-rvv-runtime-avl-vl-boundary-authority/prd.md`.
* Initial source inspection focused on:
  `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`,
  `include/TianChenRV/Target/RVV/RVVMicrokernel.h`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`,
  `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`,
  `test/Target/RVVMicrokernel/`,
  `test/Target/TargetArtifactBundleExport/`, and
  `test/Target/EmissionManifest/`.

## Completion Notes

* `RVVBinarySelectedConfigContract` now exposes runtime AVL source/role,
  runtime VL source/scope, and a formatted runtime VL boundary comment in the
  same contract object that already owns family, selected variant/role,
  selected vector shape, SEW, LMUL, policy, vector suffixes, runtime
  element-count C name, dispatch guard C name, and descriptor-local count.
* RVV selected emission planning now appends selected vector-shape metadata and
  runtime AVL/VL metadata through contract helpers, and always records
  `tcrv_rvv.descriptor_element_count` for bounded binary selected plans.
* RVV source route metadata now requires descriptor-local element-count
  metadata in addition to selected config, runtime element-count C name, and
  runtime AVL/VL selected-plan metadata.
* RVV source candidate preflight validates selected-plan metadata against the
  contract, then, when a matching microkernel attachment exists, resolves the
  actual module-backed selected config/runtime AVL contract and checks the
  descriptor-local count for stale metadata before source/header/object/bundle
  output.
* Generated RVV source now prints a `selected_runtime_vl_boundary` comment next
  to the existing `selected_binary_config` comment, making the runtime
  element-count C name, runtime AVL source/role, `tcrv_rvv.setvl`,
  `tcrv_rvv.with_vl`, descriptor-local count, selected variant, and selected
  role visible in the artifact.
* C++ coverage now rejects missing descriptor-local count route metadata and
  stale descriptor-local count metadata on the dispatch composite path before
  bundle metadata export. The role-binding lit fixture now supplies
  `tcrv_rvv.descriptor_element_count` in hand-authored selected-plan metadata.
* Generated runtime C semantics did not materially change: this round only
  tightened contract metadata, preflight validation, and provenance comments.
  No new `ssh rvv` correctness claim is made; the prior bounded
  source-fronted runtime proof remains applicable to the unchanged generated
  callable semantics.

## Manual Artifact Evidence

* Artifact directory:
  `artifacts/tmp/rvv_selected_config_runtime_vl_boundary_bundle/`.
* Exact command:

```bash
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_selected_config_runtime_vl_boundary_bundle test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir
```

* `stdout.txt` reported `tianchenrv.target_artifact_bundle_export: complete`.
* The generated bundle index reports three RVV plugin-owned artifacts:
  source route `tcrv-export-rvv-microkernel-c`, header route
  `tcrv-export-rvv-microkernel-header`, and object route
  `tcrv-export-rvv-microkernel-object`.
* The generated source contains `selected_binary_config` with
  `runtime_element_count_c_name=n` and `descriptor_element_count=16`, plus
  `selected_runtime_vl_boundary` with
  `runtime_avl_source=runtime-element-count-abi-parameter`,
  `runtime_avl_role=runtime-element-count`,
  `runtime_vl_source=tcrv_rvv.setvl`, and
  `runtime_vl_scope=tcrv_rvv.with_vl`.
* The bundle index records `tcrv_rvv.runtime_avl_source`,
  `tcrv_rvv.runtime_vl_source`, `tcrv_rvv.runtime_element_count_c_name`, and
  `tcrv_rvv.descriptor_element_count` for the source/header/object records.

## Validation

* `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-microkernel-auto-materialization|rvv-microkernel-runtime-abi-role-binding|rvv-microkernel-pipeline|plan-linalg-i32-vadd-and-export-target-artifact-bundle|linalg-i32-vadd-to-exec|linalg-i32-vsub-to-rvv-artifact|linalg-i32-vmul-to-rvv-artifact'`
  with 8/8 selected tests passed.
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'TargetArtifactBundleExport|EmissionManifest|RVVMicrokernel|LinalgToExec|rvv-microkernel-e2e'`
  with 73/73 selected tests passed.
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-selected-config-runtime-vl-boundary`
* `git diff --check`
* `git diff --cached --check`
