# Extension manifest target-support route activation

## Goal

Activate the existing RVV target-support artifact and translate route
contributions through the extension plugin / manifest boundary. Central
built-in target route setup should aggregate generic plugin declarations and
must no longer know which RVV target-support helper to call for RVV direct
microkernel routes or RVV+scalar dispatch routes.

This is a structural route-activation task for existing bounded RVV direct
source/header/object routes and RVV+scalar dispatch source/header/object bundle
routes. It is not a new arithmetic family, dtype, LMUL, descriptor export path,
or performance task.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial worktree state for this task was clean.
* Initial HEAD was `7c8ff54 refactor(rvv): extract target-support artifact bundle`.
* No `.trellis/.current-task` existed at session start, so this task was
  created from the supplied Hermes direction brief before source edits.
* The previous round extracted RVV direct and RVV+scalar dispatch target
  artifact registration into `rvv::configureRVVTargetSupportExtensionBundle`
  and `rvv::registerRVVTargetSupportTargetTranslateRoutes`.
* Current central built-in target artifact setup still constructs the RVV
  extension bundle in `BuiltinTargetArtifactExporters.cpp` and directly calls
  the RVV target-support helper.
* Current central built-in translate route setup still directly calls the RVV
  target-support translate-route helper.
* The target-support route facts themselves are already RVV-owned; the
  remaining bottleneck is activation through a generic plugin/manifest surface.

## Scope

In scope:

* A small C++ plugin/manifest hook that lets an extension plugin configure its
  target-support `ExtensionBundle` contribution and optional target translate
  routes.
* RVV plugin implementation of that hook for existing RVV direct
  source/header/object routes and RVV+scalar dispatch source/header/object
  bundle routes.
* Central built-in artifact and translate route setup rewritten to call the
  generic plugin/manifest hook for RVV, rather than an RVV-named target-support
  helper.
* Focused C++ coverage proving RVV target-support routes are activated through
  the plugin/manifest boundary and fail closed when RVV or scalar dependencies
  are missing/disabled.
* Focused local artifact and lit checks preserving dynamic i32-vadd/i32-vsub
  direct and plan-and-export bundle behavior.
* Fresh `ssh rvv` evidence for dynamic i32-vsub after route activation changes.

Out of scope:

* New vmul, i64, LMUL, vector-family expansion, or route-name purge as the main
  result.
* A broad rewrite of every extension plugin or the entire plugin framework.
* Descriptor-driven computation, direct descriptor-to-C as architecture, or
  moving computation semantics into `tcrv.exec`.
* Python implementation of compiler core, dialects, passes, plugin registry,
  lowering, or emission.
* Report-only, metadata-only, prompt-only, helper-only, or broad smoke-test
  closeout.

## Requirements

* Compiler implementation remains in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
* RVV owns the declaration that its target-support bundle contributes artifact
  route metadata, plugin-owned target exporter bundles, and target translate
  routes.
* Central built-in artifact setup may perform generic built-in plugin
  registration/linkage, but it must not directly call the RVV target-support
  helper or iterate RVV direct/dispatch route manifests.
* Central built-in translate route setup must consume target translate
  contributions through the generic plugin/manifest hook, not through an
  RVV-named helper call.
* Adding a future target-support bundle that fits this hook should require a
  plugin-local hook implementation plus build linkage, not editing central
  target route truth.
* RVV direct source/header/object routes remain owned by `rvv-plugin`.
* RVV+scalar dispatch source/header/object routes remain owned by `rvv-plugin`
  and gated on enabled `scalar-plugin` dependency.
* Scalar standalone fallback routes remain scalar-owned.
* `tcrv-translate` direct and plan-and-export artifact route consumption stays
  registry-based and target-neutral.
* Selected-config fail-closed validation, runtime-VL metadata, finite binary
  family registry behavior, fixed-vector vadd extent enforcement, and dynamic
  transfer-tail authority remain intact.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits, and the task is started as current before implementation.
* [x] `ExtensionPlugin` or an equivalent manifest surface exposes a generic
      target-support contribution hook for extension bundles and translate
      routes.
* [x] `RVVExtensionPlugin` owns the declaration that activates
      `configureRVVTargetSupportExtensionBundle` and
      `registerRVVTargetSupportTargetTranslateRoutes`.
* [x] `BuiltinTargetArtifactExporters.cpp` no longer includes
      `RVVTargetSupportBundle.h` or directly calls
      `configureRVVTargetSupportExtensionBundle`.
* [x] `BuiltinTargetTranslateRoutes.cpp` no longer includes
      `RVVTargetSupportBundle.h` or directly calls
      `registerRVVTargetSupportTargetTranslateRoutes`.
* [x] Built-in target exporter registration still publishes RVV direct routes
      when `rvv-plugin` is enabled, omits them when RVV is missing/disabled,
      and omits only RVV+scalar dispatch routes when `scalar-plugin` is
      missing/disabled.
* [x] Built-in target translate route registration still publishes RVV direct
      and RVV+scalar dispatch direct helper routes through generic registry
      aggregation.
* [x] C++ tests prove the RVV target-support contribution is reached through
      the plugin/manifest boundary and no central RVV route truth remains for
      artifact/translate activation.
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
* [x] Fresh `ssh rvv` evidence is collected for dynamic vector i32-vsub.
* [x] `git diff --check`, `git diff --cached --check`, Trellis validation
      before finish/archive, and Trellis validation after archive pass.
* [x] If complete, the task is finished, archived, and committed as one
      coherent commit. If incomplete, the task remains open with the exact next
      continuation point.

## Definition Of Done

* RVV finite-binary direct and RVV-primary dispatch route activation is owned
  by RVV plugin/target-support surfaces and consumed through generic registries.
* Central built-in code performs generic plugin/manifest aggregation rather
  than carrying RVV target-support route truth.
* Existing dynamic vadd/vsub direct and bundle behavior is preserved with
  focused local checks and fresh RVV hardware evidence.

## Technical Notes

Specs read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/plugin-protocol/index.md`
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
* `.trellis/spec/plugin-protocol/locality-contract.md`
* `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
* `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

* `.trellis/tasks/archive/2026-05/05-13-rvv-target-support-artifact-bundle-extraction/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-plugin-owned-artifact-route-registration/prd.md`
* `.trellis/workspace/codex/journal-5.md`

Initial code inspection target:

* `include/TianChenRV/Plugin/ExtensionPlugin.h`
* `lib/Plugin/ExtensionPlugin.cpp`
* `include/TianChenRV/Plugin/RVV/RVVExtensionPlugin.h`
* `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
* `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`
* `lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp`
* `tools/tcrv-translate/tcrv-translate.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* dynamic vector add/sub tests under `test/Transforms/VectorToExec/`
* dynamic vector add/sub bundle tests under
  `test/Target/TargetArtifactBundleExport/`

## Completed This Round

* Added generic `ExtensionPlugin` target-support hooks for extension bundle
  configuration and target translate route contribution.
* Implemented the RVV hook in `RVVExtensionPlugin`, making the RVV plugin own
  activation of `configureRVVTargetSupportExtensionBundle` and
  `registerRVVTargetSupportTargetTranslateRoutes`.
* Rewired `BuiltinTargetArtifactExporters.cpp` so RVV extension bundle
  activation comes from the generic plugin manifest hook instead of a central
  RVV target-support helper call.
* Rewired `BuiltinTargetTranslateRoutes.cpp` so built-in translate routes are
  collected by iterating enabled extension plugins and invoking their generic
  target-support translate hook.
* Added C++ coverage proving RVV plugin manifest activation configures the
  target-support extension bundle, publishes direct RVV and RVV+scalar dispatch
  translate routes, and is consumed by built-in translate aggregation.
* Updated specs for the durable plugin/manifest target-support contribution
  boundary.

## Route Activation Owner Before / After

Before:

* `BuiltinTargetArtifactExporters.cpp` constructed the RVV extension bundle and
  directly called `rvv::configureRVVTargetSupportExtensionBundle`.
* `BuiltinTargetTranslateRoutes.cpp` directly called
  `rvv::registerRVVTargetSupportTargetTranslateRoutes`.
* RVV route facts were already target-support-owned, but central built-in route
  setup still knew which RVV helper activated them.

After:

* `RVVExtensionPlugin::configureTargetSupportExtensionBundle` owns the
  declaration that activates RVV direct and RVV+scalar dispatch artifact route
  metadata and plugin target exporter bundles.
* `RVVExtensionPlugin::registerTargetSupportTranslateRoutes` owns the
  declaration that activates RVV direct and RVV+scalar dispatch direct helper
  translate routes.
* Central built-in artifact setup performs generic plugin manifest aggregation
  for RVV and no longer includes or calls `RVVTargetSupportBundle.h` helpers.
* Central built-in translate setup iterates enabled extension plugins and no
  longer includes or calls `RVVTargetSupportBundle.h` helpers.
* RVV+scalar dispatch routes still record `rvv-plugin` ownership with explicit
  `scalar-plugin` dependency; scalar standalone fallback routes remain
  scalar-owned.

Central RVV-specific route truth removed:

* `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` no longer includes
  `TianChenRV/Target/RVV/RVVTargetSupportBundle.h`.
* `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` no longer calls
  `configureRVVTargetSupportExtensionBundle`.
* `lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp` no longer includes
  `TianChenRV/Target/RVV/RVVTargetSupportBundle.h`.
* `lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp` no longer calls
  `registerRVVTargetSupportTargetTranslateRoutes`.

## Exact Artifact Commands

```bash
ART=artifacts/tmp/extension_manifest_target_support_route_activation
mkdir -p "$ART/direct/vector_dynamic_i32_vsub" "$ART/direct/vector_dynamic_i32_vadd" "$ART/bundle/vector_dynamic_i32_vsub" "$ART/bundle/vector_dynamic_i32_vadd"

build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > "$ART/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir"
build/bin/tcrv-translate --tcrv-export-target-source-artifact "$ART/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir" > "$ART/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.c"
build/bin/tcrv-translate --tcrv-export-target-header-artifact "$ART/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir" > "$ART/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.h"
build/bin/tcrv-translate --tcrv-export-target-artifact "$ART/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir" > "$ART/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.o"

build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > "$ART/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir"
build/bin/tcrv-translate --tcrv-export-target-source-artifact "$ART/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir" > "$ART/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.c"
build/bin/tcrv-translate --tcrv-export-target-header-artifact "$ART/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir" > "$ART/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.h"
build/bin/tcrv-translate --tcrv-export-target-artifact "$ART/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir" > "$ART/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.o"

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
* Direct vsub/vadd objects and dynamic vsub/vadd dispatch bundle objects are
  RISC-V ELF relocatables.

## Dynamic RVV Evidence

Command:

```bash
python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family i32-vsub --input test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --expect-selected-kernel=frontend_vector_dynamic_i32_vsub --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv --artifact-root artifacts/tmp/extension_manifest_target_support_route_activation/e2e --run-id 20260513T-extension-manifest-target-support-route-activation-vsub --overwrite --timeout 120
```

Result:

* Evidence directory:
  `artifacts/tmp/extension_manifest_target_support_route_activation/e2e/20260513T-extension-manifest-target-support-route-activation-vsub/`.
* `status = success`, `ssh_evidence = true`,
  `expected_selected_kernel = frontend_vector_dynamic_i32_vsub`,
  `selected_kernel = frontend_vector_dynamic_i32_vsub`,
  `runtime_element_counts = [7, 16, 23]`,
  `source_export_route = tcrv-export-rvv-i32-vsub-microkernel-c`,
  `vector_shape = i32m1`.

## Checks Run

* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-05-13-extension-manifest-target-support-route-activation`
  before implementation.
* `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `cmake --build build --target tianchenrv-plugin-registry-test tianchenrv-rvv-extension-plugin-test tianchenrv-i32-binary-family-registry-test -j2`
* `./build/bin/tianchenrv-plugin-registry-test`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
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

## Self-Repair

* No failed focused checks required code self-repair after the hook and central
  aggregation rewiring.

## Spec Update Judgment

Spec update was required because this round added a durable plugin/manifest
target-support contribution hook. The updates are limited to:

* `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
