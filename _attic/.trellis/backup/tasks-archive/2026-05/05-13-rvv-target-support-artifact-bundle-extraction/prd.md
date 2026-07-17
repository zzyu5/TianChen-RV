# RVV target-support artifact bundle extraction

## Goal

Extract the production RVV finite-binary target artifact registration surface
into an RVV plugin / target-support bundle. The central built-in target
artifact exporter code should compose generic extension bundles and delegate RVV
route ownership to RVV target-support code instead of hand-assembling RVV direct
microkernel and RVV+scalar dispatch route truth itself.

This round continues the previous RVV-owned route-registration task. It is a
structural ownership migration for existing bounded dynamic i32-vadd/i32-vsub
direct artifacts and RVV+scalar dispatch bundle artifacts. It is not a new
family, dtype, LMUL, generic plugin framework rewrite, descriptor-driven
computation path, or performance task.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial worktree state for this task was clean.
* Initial HEAD was `11153e3 fix(rvv): own dispatch artifact routes in rvv bundle`.
* No `.trellis/.current-task` existed at session start, so this task was
  created from the supplied Direction Brief before source edits.
* The previous completed task moved RVV+scalar dispatch route registration
  ownership metadata from `scalar-plugin` to `rvv-plugin` and preserved
  `scalar-plugin` as an explicit dependency.
* Current code still has RVV route truth in
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`: the RVV extension
  bundle loops over `rvv::getRVVMicrokernelArtifactRouteAuthority()` and
  `rvv_scalar::getRVVScalarDispatchRouteManifest()` to add route metadata
  requirements, and central built-in code combines RVV direct and dispatch
  artifact exporter bundle registration.
* `lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp` still manually combines
  RVV direct and RVV+scalar dispatch direct helper route registrations.
* RVV direct microkernel route authority exists in
  `include/TianChenRV/Target/RVV/RVVMicrokernel.h` /
  `lib/Target/RVV/RVVMicrokernel.cpp`.
* RVV+scalar dispatch route authority currently lives under
  `include/TianChenRV/Target/RVVScalarDispatch.h` and
  `lib/Target/Builtin/RVVScalarDispatch.cpp`, even though the route group is
  RVV-primary and now registered under `rvv-plugin` with a scalar dependency.

## Scope

In scope:

* Existing RVV direct binary microkernel source/header/object routes for the
  finite family registry, with dynamic i32-vadd and i32-vsub as required
  regressions.
* Existing RVV+scalar dispatch source/header/object bundle routes for the
  finite dispatch family registry, with dynamic i32-vadd and i32-vsub as
  required regressions.
* RVV target-support registration code that owns plugin target exporter bundle
  contribution, extension-bundle target artifact route metadata requirements,
  and direct helper translate-route contribution for these RVV-owned route
  groups.
* Focused C++ registry coverage proving owner/dependency behavior and absence
  of central RVV-specific route metadata truth.
* Focused lit/FileCheck and artifact commands preserving direct and
  plan-and-export bundle behavior for dynamic i32-vsub plus at least one
  i32-vadd regression.

Out of scope:

* New vmul, i64, LMUL, vector-shape, ToyExt/TensorExt, or generic plugin
  framework expansion as the main result.
* Route-name compatibility purges unless required by the extraction.
* Descriptor-driven computation, direct descriptor-to-C architecture, or moving
  computation semantics into `tcrv.exec`.
* Python implementation of compiler core, dialects, passes, plugin registry,
  lowering, or emission.
* Performance claims or broad benchmark matrices.
* Report-only, metadata-only, prompt-only, or helper-only closeout.

## Requirements

* Compiler implementation stays in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
* The RVV target-support layer must expose one coherent production registration
  owner for RVV direct microkernel artifact routes and RVV-primary
  RVV+scalar dispatch artifact routes.
* `BuiltinTargetArtifactExporters.cpp` must stop iterating RVV direct or
  RVV+scalar dispatch manifests to add route metadata or compose RVV route
  registration truth. It may create the RVV extension bundle and delegate RVV
  target-support configuration through a generic target-owned helper.
* RVV direct source/header/object routes remain owned by `rvv-plugin`.
* RVV+scalar dispatch source/header/object routes remain owned by `rvv-plugin`
  with `scalar-plugin` as an explicit enabled-plugin dependency.
* Scalar standalone fallback source/header/object routes remain scalar-owned.
* Missing or disabled `rvv-plugin` must not publish RVV direct or RVV+scalar
  dispatch routes.
* Missing or disabled `scalar-plugin` must not publish RVV+scalar dispatch
  routes, but must not disable RVV direct routes.
* Public translate route registration should consume a generic target-support
  route contribution rather than manually combining RVV direct and dispatch
  manifests in the tool-facing built-in route file.
* `tcrv-translate` artifact export continues to use generic
  `TargetArtifactExporterRegistry` lookup and direct helper route registration;
  no RVV-specific artifact-selection branch may be added to the tool.
* Selected-config fail-closed validation, runtime-VL metadata, finite binary
  family registry behavior, fixed-vector vadd extent enforcement, and dynamic
  transfer-tail authority remain intact.
* Direct export and plan-and-export bundle export remain equivalent for dynamic
  i32-vadd and i32-vsub.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits, and the task is started as current before implementation.
* [x] RVV target-support code owns the route registration composition for RVV
      direct microkernel source/header/object routes and RVV+scalar dispatch
      source/header/object routes.
* [x] Central built-in artifact exporter code delegates RVV route metadata and
      plugin target exporter bundle registration to the RVV target-support
      helper, with no manifest iteration over RVV direct or RVV+scalar dispatch
      routes left in `BuiltinTargetArtifactExporters.cpp`.
* [x] RVV+scalar dispatch implementation and/or route registration source is
      placed in RVV target-support ownership rather than central built-in route
      registration ownership.
* [x] Scalar standalone fallback routes remain registered by the scalar bundle
      and are not moved under RVV.
* [x] C++ registry tests prove duplicate registration, missing/disabled
      `rvv-plugin`, missing/disabled `scalar-plugin`, route metadata owner,
      route metadata dependency, and direct RVV independence from scalar
      dependency.
* [x] `tcrv-translate` direct helper registration consumes the generic
      target-support route contribution and artifact-backed direct helper
      routes still use generic registry lookup.
* [x] Dynamic vector i32-vadd and i32-vsub direct artifacts still emit
      `__riscv_vadd_vv_i32m1` and `__riscv_vsub_vv_i32m1` respectively.
* [x] Dynamic vector i32-vadd and i32-vsub plan-and-export bundles still record
      registry-derived RVV+scalar source/header/object route metadata and
      selected-config/runtime-VL metadata.
* [x] Selected-config fail-closed regressions, family registry regressions,
      fixed-vector vadd extent checks, and dynamic transfer-tail authority
      checks pass.
* [x] Focused build covers `TianChenRVTransforms`, `TianChenRVTarget`,
      `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test`.
* [x] Exact direct and bundle artifact commands are recorded for dynamic vector
      i32-vsub plus at least one i32-vadd regression.
* [x] Fresh `ssh rvv` evidence is collected for dynamic i32-vsub if emitted
      source/object or bundle materialization behavior changes; otherwise the
      final report states why the extraction was registration ownership only
      and did not create a new runtime claim.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] If complete, the task is finished, archived, and committed as one
      coherent commit. If incomplete, the task remains open with the exact next
      continuation point.

## Definition Of Done

* RVV finite-binary direct and RVV-primary dispatch artifact registration is
  owned by RVV target/plugin support code and consumed through generic
  registries.
* Central built-in code only performs generic bundle aggregation for RVV route
  metadata and exporter bundle registration.
* Existing dynamic vadd/vsub direct and bundle behavior is preserved with
  focused local checks and bounded RVV evidence where emitted artifacts change.
* No descriptor-local value is promoted to runtime authority, correctness
  evidence, or performance evidence.

## Technical Notes

Specs read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/plugin-protocol/index.md`
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
* `.trellis/spec/plugin-protocol/locality-contract.md`
* `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

* `.trellis/tasks/archive/2026-05/05-13-rvv-plugin-owned-artifact-route-registration/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-selected-config-fail-closed-artifact-validation/prd.md`
* `.trellis/workspace/codex/journal-5.md`

Initial code inspection target:

* `include/TianChenRV/Target/RVV/RVVMicrokernel.h`
* `include/TianChenRV/Target/RVVScalarDispatch.h`
* `lib/Target/RVV/RVVMicrokernel.cpp`
* `lib/Target/Builtin/RVVScalarDispatch.cpp`
* `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`
* `lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp`
* `lib/Target/RVV/CMakeLists.txt`
* `lib/Target/Builtin/CMakeLists.txt`
* `tools/tcrv-translate/tcrv-translate.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* dynamic vector add/sub tests under `test/Transforms/VectorToExec/`
* dynamic vector add/sub bundle tests under
  `test/Target/TargetArtifactBundleExport/`

## Completed This Round

* Added an RVV target-support bundle API in
  `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h` and
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`.
* Moved RVV+scalar dispatch target/export implementation from
  `lib/Target/Builtin/RVVScalarDispatch.cpp` to
  `lib/Target/RVV/RVVScalarDispatch.cpp`, with the public header now under
  `include/TianChenRV/Target/RVV/RVVScalarDispatch.h` and a compatibility
  forwarding header left at `include/TianChenRV/Target/RVVScalarDispatch.h`.
* Rewired `TianChenRVRVVTarget` to compile the RVV+scalar dispatch target
  support code and depend on `TianChenRVScalarTarget`.
* Changed `BuiltinTargetArtifactExporters.cpp` so the RVV extension bundle
  delegates route metadata requirements and plugin target exporter bundle
  registration to `rvv::configureRVVTargetSupportExtensionBundle`.
* Changed `BuiltinTargetTranslateRoutes.cpp` so direct helper route
  registration delegates to
  `rvv::registerRVVTargetSupportTargetTranslateRoutes` instead of manually
  combining RVV direct and RVV+scalar dispatch route contributors.
* Added C++ coverage proving the RVV target-support bundle contributes both
  direct RVV routes and RVV+scalar dispatch routes, preserves the scalar
  dependency for dispatch routes, and keeps RVV direct routes available without
  scalar.
* Added a narrow lowering-runtime spec note that central built-in composition
  must not iterate RVV direct or RVV+scalar dispatch manifests as central route
  truth.

## Route Registration Owner Before / After

Before:

* RVV direct source/header/object route metadata was added by
  `BuiltinTargetArtifactExporters.cpp` by iterating
  `rvv::getRVVMicrokernelArtifactRouteAuthority()`.
* RVV+scalar dispatch source/header/object route metadata was added by
  `BuiltinTargetArtifactExporters.cpp` by iterating
  `rvv_scalar::getRVVScalarDispatchRouteManifest()`.
* Built-in target translate route registration manually called the RVV direct
  route contributor and RVV+scalar dispatch route contributor.
* RVV+scalar dispatch implementation lived in the built-in target exporter
  library source tree.

After:

* RVV direct source/header/object route metadata is contributed by
  `rvv::configureRVVTargetSupportExtensionBundle`.
* RVV+scalar dispatch source/header/object route metadata is contributed by the
  same RVV target-support bundle helper, still owned by `rvv-plugin` and still
  requiring `scalar-plugin`.
* Scalar standalone fallback source/header/object route metadata remains
  scalar-owned.
* Built-in artifact exporter and translate-route composition each call one RVV
  target-support helper and no longer iterate RVV route manifests directly.
* RVV+scalar dispatch target/export implementation is compiled as part of the
  RVV target-support library.

Central RVV-specific route truth removed:

* `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` no longer includes
  `RVVMicrokernel.h` or `RVVScalarDispatch.h`.
* `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` no longer calls
  `getRVVMicrokernelArtifactRouteAuthority`,
  `getRVVScalarDispatchRouteManifest`,
  `registerRVVMicrokernelPluginTargetExporterBundle`, or
  `registerRVVScalarDispatchPluginTargetExporterBundle`.
* `lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp` no longer calls
  `registerRVVMicrokernelTargetTranslateRoutes` or
  `registerRVVScalarDispatchTargetTranslateRoutes` directly.

## Exact Artifact Commands

```bash
ART=artifacts/tmp/rvv_target_support_bundle_extraction
mkdir -p "$ART/direct/vector_dynamic_i32_vsub" "$ART/direct/vector_dynamic_i32_vadd" "$ART/bundle/vector_dynamic_i32_vsub" "$ART/bundle/vector_dynamic_i32_vadd"

build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > "$ART/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir"
build/bin/tcrv-translate --tcrv-export-target-source-artifact "$ART/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir" > "$ART/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.c"
build/bin/tcrv-translate --tcrv-export-target-header-artifact "$ART/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir" > "$ART/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.h"
build/bin/tcrv-translate --tcrv-export-target-artifact "$ART/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir" > "$ART/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.o"

build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > "$ART/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir"
build/bin/tcrv-translate --tcrv-export-target-source-artifact "$ART/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir" > "$ART/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.c"

build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir="$ART/bundle/vector_dynamic_i32_vsub" test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir > "$ART/bundle/vector_dynamic_i32_vsub/stdout.txt"
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir="$ART/bundle/vector_dynamic_i32_vadd" test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir > "$ART/bundle/vector_dynamic_i32_vadd/stdout.txt"
```

Artifact checks:

* Direct dynamic vsub source contains `__riscv_vsub_vv_i32m1` and
  `selected_runtime_vl_boundary`.
* Direct dynamic vadd source contains `__riscv_vadd_vv_i32m1` and
  `selected_runtime_vl_boundary`.
* Dynamic vsub/vadd bundle indexes contain RVV+scalar source/header/object
  routes plus `tcrv_rvv.dispatch_contract_selected_vector_config` and
  `tcrv_rvv.dispatch_contract_runtime_vl_boundary`.
* Direct vsub object and dynamic vsub/vadd dispatch bundle objects are RISC-V
  ELF relocatables.

## Dynamic RVV Evidence

Command:

```bash
python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family i32-vsub --input test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --expect-selected-kernel=frontend_vector_dynamic_i32_vsub --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv --artifact-root artifacts/tmp/rvv_target_support_bundle_extraction/e2e --run-id 20260513T-rvv-target-support-bundle-extraction-vsub --overwrite --timeout 120
```

Result:

* Evidence directory:
  `artifacts/tmp/rvv_target_support_bundle_extraction/e2e/20260513T-rvv-target-support-bundle-extraction-vsub/`.
* `status = success`, `ssh_evidence = true`,
  `expected_selected_kernel = frontend_vector_dynamic_i32_vsub`,
  `selected_kernel = frontend_vector_dynamic_i32_vsub`,
  `runtime_element_counts = [7, 16, 23]`,
  `source_export_route = tcrv-export-rvv-i32-vsub-microkernel-c`,
  `vector_shape = i32m1`.

## Checks Run

* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-target-support-artifact-bundle-extraction`
  before starting implementation.
* `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `cmake --build build --target tianchenrv-i32-binary-family-registry-test -j2`
* `./build/bin/tianchenrv-i32-binary-family-registry-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'vector-dynamic-i32-vadd-to-exec|vector-dynamic-i32-vsub-to-exec|vector-dynamic-i32-binary-invalid|vector-i32-vadd-to-exec|vector-i32-vadd-invalid|plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle|plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle|RVVMicrokernel|EmissionManifest|linalg-i32-vadd-to-exec|linalg-i32-vsub-to-exec|TargetArtifactBundleExport'`
  with 68 selected tests passed.
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
* Trellis validation after archive and archived JSONL path repair.

## Self-Repair

* While rerunning combined checks from `build/test`, I first used the repo-root
  path `./build/bin/tianchenrv-target-artifact-export-test`, which is invalid
  from `build/test`. I reran with
  `../bin/tianchenrv-target-artifact-export-test` and the broad lit filter
  passed.
* After archive, Trellis validation found that `implement.jsonl` and
  `check.jsonl` still referenced the pre-archive PRD path. I updated both
  archived JSONL files to point at the archived PRD path and reran validation.

## Spec Update Judgment

Spec update was useful because the extraction creates a durable registration
boundary: central built-in composition must delegate RVV direct and
RVV+scalar dispatch route metadata to RVV target-support code. The update is
limited to `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
