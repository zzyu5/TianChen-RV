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
* [x] The task is finished/archived and one coherent commit records the round
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

## Continuation: Dynamic Add/Sub Dispatch Bundle Runtime VL Metadata

This continuation started from `HEAD = 8625bbe feat(rvv): share finite binary
family registry` with a clean worktree and no active `.trellis/.current-task`.
The Hermes brief requested the existing dynamic vector add/sub route, not a
new arithmetic family, to make selected RVV config and runtime VL authority a
single production-consumed boundary.

Completed in this continuation:

* Added dispatch bundle metadata
  `tcrv_rvv.dispatch_contract_runtime_vl_boundary`, derived from the same
  `RVVBinarySelectedConfigContract` consumed by the direct RVV component. The
  bundle index now records runtime element-count C name, runtime AVL source,
  runtime AVL role, runtime VL source `tcrv_rvv.setvl`, and runtime VL scope
  `tcrv_rvv.with_vl` beside the existing dispatch selected vector config and
  descriptor-local element-count metadata.
* Tightened RVV binary microkernel materialization so the policy attr attached
  to materialized `tcrv_rvv.setvl` and `tcrv_rvv.with_vl` is derived from the
  selected-config contract and validated there, instead of being an unrelated
  local hard-coded materialization fact.
* Strengthened dynamic vector vsub lit coverage to assert runtime AVL/VL
  selected-plan metadata and generated `selected_runtime_vl_boundary` source
  comments.
* Strengthened dynamic vector vadd/vsub plan-and-export bundle coverage to
  assert selected vector config and runtime VL boundary metadata in the bundle
  index.

Exact artifact commands:

```bash
mkdir -p artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vadd artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vsub artifacts/tmp/rvv_selected_config_runtime_vl_boundary/bundle/vector_dynamic_i32_vadd artifacts/tmp/rvv_selected_config_runtime_vl_boundary/bundle/vector_dynamic_i32_vsub

build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir
build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.c
build/bin/tcrv-translate --tcrv-export-target-header-artifact artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.h
build/bin/tcrv-translate --tcrv-export-target-artifact artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.o

build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir
build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.c
build/bin/tcrv-translate --tcrv-export-target-header-artifact artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.h
build/bin/tcrv-translate --tcrv-export-target-artifact artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.o

build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_selected_config_runtime_vl_boundary/bundle/vector_dynamic_i32_vadd test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir > artifacts/tmp/rvv_selected_config_runtime_vl_boundary/bundle/vector_dynamic_i32_vadd/stdout.txt
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_selected_config_runtime_vl_boundary/bundle/vector_dynamic_i32_vsub test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir > artifacts/tmp/rvv_selected_config_runtime_vl_boundary/bundle/vector_dynamic_i32_vsub/stdout.txt
```

Artifact checks:

* Direct dynamic vadd source contains `__riscv_vadd_vv_i32m1`.
* Direct dynamic vsub source contains `__riscv_vsub_vv_i32m1`.
* Direct vadd/vsub sources contain `selected_runtime_vl_boundary`.
* Dynamic vadd/vsub bundle indexes contain
  `tcrv_rvv.dispatch_contract_runtime_vl_boundary`.
* Direct vadd/vsub objects and bundle dispatch objects are RISC-V ELF
  relocatables.

Dynamic RVV evidence command:

```bash
python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family i32-vsub --input test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --expect-selected-kernel=frontend_vector_dynamic_i32_vsub --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv --artifact-root artifacts/tmp/rvv_selected_config_runtime_vl_boundary/e2e --run-id 20260513T-rvv-selected-config-runtime-vl-boundary-vsub --overwrite --timeout 120
```

Result:

* Evidence file:
  `artifacts/tmp/rvv_selected_config_runtime_vl_boundary/e2e/20260513T-rvv-selected-config-runtime-vl-boundary-vsub/evidence.json`.
* `status = success`, `ssh_evidence = true`,
  `selected_kernel = frontend_vector_dynamic_i32_vsub`,
  `runtime_element_counts = [7, 16, 23]`,
  `vector_shape = i32m1`.

Checks run in this continuation:

* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-selected-config-runtime-vl-boundary`
* `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'vector-dynamic-i32-vadd-to-exec|vector-dynamic-i32-vsub-to-exec|vector-dynamic-i32-binary-invalid|vector-i32-vadd-to-exec|vector-i32-vadd-invalid|plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle|plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle|RVVMicrokernel|EmissionManifest|plan-linalg-i32-vadd-and-export-target-artifact-bundle|plan-linalg-i32-vsub-and-export-target-artifact-bundle'`
  with 52 selected tests passed after one FileCheck order repair.
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'VectorToExec|LinalgToExec|TargetArtifactBundleExport|RVVMicrokernel|EmissionManifest'`
  with 80 selected tests passed.
* Direct and bundle artifact commands listed above.
* `ssh rvv` evidence command listed above.
* `file` confirmed generated direct and bundle objects are RISC-V ELF
  relocatables.
* `git diff --check`
* `git diff --cached --check`
* Trellis validation before finish/archive and after archive.

Self-repair:

* `clang-format` was not available in PATH or common version-suffixed names on
  this host, so C++ style was checked by diff/manual review and the focused
  C++ build.
* The first focused lit run failed because the new vsub runtime-VL FileCheck
  assertions were placed after source-frontdoor metadata, while the actual
  selected-plan metadata order is selected vector shape, capacity facts,
  runtime VL boundary, source-frontdoor runtime authority, then typed source
  metadata. The assertions were moved to the real order and the focused and
  broader lit sets passed.

Spec update judgment:

No `.trellis/spec/` update was needed. This continuation implemented and
tested an already documented contract: dispatch bundle metadata now exposes
the runtime AVL/VL boundary that the RVV plugin and target artifact specs
already require, and materialization now consumes the existing selected-config
contract more directly.
