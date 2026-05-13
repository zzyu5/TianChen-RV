# RVV selected-config fail-closed artifact validation

## Goal

Make selected RVV config and runtime AVL/VL metadata a validated production
contract at the artifact boundary for the existing dynamic vector i32-vsub
route, with dynamic vector i32-vadd as the required regression. Direct
source/header/object export and plan-and-export bundle export must reject stale
or missing selected-config/runtime-VL metadata before writing artifact output.

This round extends the existing selected-config/runtime-VL boundary work. It
does not add a new arithmetic family, dtype, LMUL, generic scalable-vector
backend, performance claim, or descriptor-driven export path.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial worktree state for this task was clean.
* Initial HEAD was `4ab4f41 fix(rvv): expose dispatch runtime VL boundary`.
* No `.trellis/.current-task` existed at session start, so this task was
  created from the Hermes Direction Brief before source edits.
* The prior selected-config/runtime-VL task carried selected RVV config into
  `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` materialization and exposed runtime VL
  boundary metadata in direct artifacts and dispatch bundle metadata.
* The next bottleneck is fail-closed validation at the consumed artifact
  boundary: route metadata must not merely document selected config/runtime VL;
  stale or missing metadata must block source/header/object/bundle output.
* The current production scope is existing dynamic vector i32m1 vadd/vsub
  through the neutral source front door and shared RVV finite binary family
  registry.

## Scope

In scope:

* Existing dynamic vector i32-vsub direct source/header/object artifact export.
* Existing dynamic vector i32-vsub plan-and-export target artifact bundle
  export.
* Dynamic vector i32-vadd as the required regression for the same validation
  surface.
* Shared validation for directly shared fixed-vector vadd or linalg
  compatibility only where needed to avoid duplicate selected-config truth.
* Negative lit/FileCheck coverage for missing or stale selected RVV
  config/runtime AVL/VL metadata before artifact output.

Out of scope:

* New `vmul`, `i64`, LMUL, shape, vector family expansion, or generic
  scalable-vector backend claims as the main result.
* Performance claims.
* Descriptor-driven computation or direct descriptor-to-C export.
* Python implementation of compiler core, dialects, passes, plugin registry,
  lowering, or emission.
* Moving computation semantics into `tcrv.exec`.
* Report-only, helper-only, prompt-only, or broad smoke-test closeout.

## Requirements

* Direct artifact export and bundle export must validate that selected SEW,
  LMUL, tail policy, mask policy, runtime AVL source, runtime VL source,
  runtime VL scope, family identity, and materialized `tcrv_rvv.setvl` /
  `tcrv_rvv.with_vl` policy agree before any source/header/object/bundle output
  is accepted.
* Missing selected-config metadata must fail closed before artifact output.
* Stale selected-config metadata must fail closed before artifact output.
* Missing or stale runtime AVL/VL selected metadata must fail closed before
  artifact output.
* Descriptor-local `tcrv_rvv.element_count` or descriptor-local count metadata
  must not substitute for runtime AVL authority.
* Target hardware facts such as vlenb/base lanes/march remain separate from
  selected config and must not become runtime trip-count authority.
* Generated source comments and bundle index records must still expose the
  selected-config/runtime-VL boundary after validation.
* Dynamic vector vadd must emit vadd and dynamic vector vsub must emit vsub
  through the neutral source front door and shared family registry.
* Fixed-vector vadd extent enforcement and dynamic transfer-tail authority must
  remain intact.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits, and the task is started as current before implementation.
* [x] The direct dynamic i32-vsub artifact path validates selected config,
      runtime AVL/VL metadata, and materialized `setvl` / `with_vl` policy
      agreement before source/header/object output.
* [x] The plan-and-export dynamic i32-vsub bundle path validates the same
      contract before writing artifact files or a complete bundle index.
* [x] Dynamic i32-vadd remains covered as a regression through direct and/or
      bundle artifact paths.
* [x] Negative tests prove missing/stale selected SEW, LMUL, tail policy,
      mask policy, runtime AVL source, runtime VL source, runtime VL scope, or
      runtime element-count authority fail before artifact output.
* [x] Descriptor-local element count cannot satisfy runtime AVL authority in
      the validation surface.
* [x] Generated source and bundle index still expose selected-config/runtime-VL
      metadata after the fail-closed gate.
* [x] Focused build covers `TianChenRVTransforms`, `TianChenRVTarget`,
      `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test`.
* [x] Focused lit covers dynamic vector vadd/vsub, stale selected-config or
      runtime-VL negative cases, vector invalid diagnostics, RVV microkernel
      materialization, `TargetArtifactBundleExport`, `EmissionManifest`, fixed
      vector vadd regression, and representative linalg compatibility if
      touched.
* [x] Exact direct and bundle artifact commands are recorded for dynamic vector
      i32-vsub and at least one vadd regression.
* [x] Fresh `ssh rvv` evidence is collected for dynamic vector i32-vsub if
      emitted source/object behavior changes; otherwise the final report states
      why local fail-closed validation did not create a new runtime claim.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] If complete, the task is finished, archived, and committed as one
      coherent commit. If incomplete, the task remains open with an exact next
      continuation point.

## Definition Of Done

* Selected RVV config/runtime AVL/VL is a consumed validation contract at the
  artifact boundary, not only emitted documentation.
* The implementation stays in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
* Shared orchestration remains target-neutral; RVV-specific semantic checks
  stay in RVV plugin/target/export boundaries.
* No descriptor-local value is promoted to runtime AVL or correctness
  authority.
* Evidence is bounded to the changed compiler behavior.

## Technical Notes

Specs read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/core-dialect/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

* `.trellis/tasks/archive/2026-05/05-13-rvv-selected-config-runtime-vl-boundary/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-finite-binary-family-contract-registry/prd.md`

Initial code inspection target:

* `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
* `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`
* `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
* `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
* `lib/Target/RVV/RVVMicrokernel.cpp`
* `lib/Target/Builtin/RVVScalarDispatch.cpp`
* `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
* dynamic vector add/sub tests under `test/Transforms/VectorToExec/`
* dynamic vector add/sub bundle tests under
  `test/Target/TargetArtifactBundleExport/`

## Completed This Round

* Reworked RVV+scalar dispatch bundle component validation so the RVV dispatch
  candidate consumes the full `RVVBinarySelectedConfigContract` before bundle
  metadata export. The contract now checks selected vector-shape metadata,
  runtime AVL/VL metadata, source-frontdoor runtime extent metadata, typed RVV
  family/source metadata, and descriptor-local element-count mirror metadata
  through one contract-level helper instead of a shape-only plus partial typed
  metadata split.
* Added direct i32-vsub target artifact C++ negative coverage for stale SEW,
  LMUL, tail policy, mask policy, runtime VL source, runtime VL scope, runtime
  element-count C-name authority, and missing runtime AVL source metadata.
  Descriptor-local `descriptor_element_count` is explicitly rejected as a
  runtime element-count authority substitute.
* Added RVV+scalar i32-vsub dispatch composite C++ preflight coverage for
  stale runtime VL metadata on source/header/object dispatch bundle routes.
* Added dynamic vector i32-vsub lit negative coverage proving stale/missing
  selected-config/runtime-VL selected-plan metadata fails before direct source
  output and before target artifact bundle completion/index output.
* Kept dynamic i32-vadd and i32-vsub artifact output family-correct through the
  neutral source front door and shared family registry.

## Exact Artifact Commands

```bash
mkdir -p artifacts/tmp/rvv_selected_config_fail_closed/direct/vector_dynamic_i32_vsub artifacts/tmp/rvv_selected_config_fail_closed/direct/vector_dynamic_i32_vadd artifacts/tmp/rvv_selected_config_fail_closed/bundle/vector_dynamic_i32_vsub artifacts/tmp/rvv_selected_config_fail_closed/bundle/vector_dynamic_i32_vadd

build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/rvv_selected_config_fail_closed/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir
build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/rvv_selected_config_fail_closed/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_selected_config_fail_closed/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.c
build/bin/tcrv-translate --tcrv-export-target-header-artifact artifacts/tmp/rvv_selected_config_fail_closed/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_selected_config_fail_closed/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.h
build/bin/tcrv-translate --tcrv-export-target-artifact artifacts/tmp/rvv_selected_config_fail_closed/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_selected_config_fail_closed/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.o

build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/rvv_selected_config_fail_closed/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir
build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/rvv_selected_config_fail_closed/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/rvv_selected_config_fail_closed/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.c

build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_selected_config_fail_closed/bundle/vector_dynamic_i32_vsub test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir > artifacts/tmp/rvv_selected_config_fail_closed/bundle/vector_dynamic_i32_vsub/stdout.txt
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_selected_config_fail_closed/bundle/vector_dynamic_i32_vadd test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir > artifacts/tmp/rvv_selected_config_fail_closed/bundle/vector_dynamic_i32_vadd/stdout.txt
```

Artifact checks:

* Direct dynamic vsub source contains `__riscv_vsub_vv_i32m1` and
  `selected_runtime_vl_boundary`.
* Direct dynamic vadd source contains `__riscv_vadd_vv_i32m1` and
  `selected_runtime_vl_boundary`.
* Dynamic vsub/vadd bundle indexes contain
  `tcrv_rvv.dispatch_contract_selected_vector_config` and
  `tcrv_rvv.dispatch_contract_runtime_vl_boundary`.
* Direct vsub object and vsub/vadd dispatch bundle objects are RISC-V ELF
  relocatables.

## Dynamic RVV Evidence

Command:

```bash
python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family i32-vsub --input test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --expect-selected-kernel=frontend_vector_dynamic_i32_vsub --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv --artifact-root artifacts/tmp/rvv_selected_config_fail_closed/e2e --run-id 20260513T-rvv-selected-config-fail-closed-vsub --overwrite --timeout 120
```

Result:

* Evidence file:
  `artifacts/tmp/rvv_selected_config_fail_closed/e2e/20260513T-rvv-selected-config-fail-closed-vsub/evidence.json`.
* `status = success`, `ssh_evidence = true`,
  `selected_kernel = frontend_vector_dynamic_i32_vsub`,
  `runtime_element_counts = [7, 16, 23]`, `vector_shape = i32m1`.

## Checks Run

* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-selected-config-fail-closed-artifact-validation`
* `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'vector-dynamic-i32-vsub-to-exec|plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle'`
  with 2 selected tests passed.
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'vector-dynamic-i32-vadd-to-exec|vector-dynamic-i32-vsub-to-exec|vector-dynamic-i32-binary-invalid|vector-i32-vadd-to-exec|vector-i32-vadd-invalid|plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle|plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle|RVVMicrokernel|EmissionManifest|linalg-i32-vadd-to-exec|linalg-i32-vsub-to-exec'`
  with 52 selected tests passed.
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'VectorToExec|LinalgToExec|TargetArtifactBundleExport|RVVMicrokernel|EmissionManifest'`
  with 80 selected tests passed.
* Direct and bundle artifact commands listed above.
* `ssh rvv` evidence command listed above.
* `file` confirmed generated direct and bundle objects are RISC-V ELF
  relocatables.
* `git diff --check`
* `git diff --cached --check`
* Trellis validation before finish/archive.

## Self-Repair

* The first direct artifact probe printed planned MLIR to stdout. I reran the
  same command with an explicit artifact output path and generated source,
  header, and object artifacts from that planned file.

## Spec Update Judgment

No `.trellis/spec/` update was needed. This round tightened implementation and
tests for already documented RVV selected-config/runtime AVL/VL artifact
fail-closed behavior without adding a new architectural rule.
