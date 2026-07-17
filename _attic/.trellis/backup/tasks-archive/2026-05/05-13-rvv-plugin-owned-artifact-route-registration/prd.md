# RVV plugin-owned artifact route registration

## Goal

Make the existing RVV finite-binary artifact route registration clearly owned
by the RVV plugin / target-support boundary for the dynamic vector i32-vadd and
i32-vsub production paths. Direct RVV source/header/object routes and the
RVV-primary scalar-dispatch source/header/object bundle routes must be
registered through RVV-owned route owners or target-support bundles, and public
tools must consume the generic route registries rather than carrying their own
RVV route truth.

This round is a route-registration ownership migration for existing bounded
finite-binary routes. It is not a new arithmetic family, dtype, LMUL, generic
plugin framework rewrite, descriptor-driven export path, or performance task.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial worktree state for this task was clean.
* Initial HEAD was `2244725 fix(rvv): fail closed on selected config artifacts`.
* No `.trellis/.current-task` existed at session start, so this task was
  created from the Hermes direction brief before source edits.
* RVV direct microkernel source/header/object routes already have a
  target-owned route authority in `include/TianChenRV/Target/RVV/` and are
  exported by `registerRVVMicrokernelPluginTargetExporterBundle` under
  `rvv-plugin`.
* `tcrv-translate` already consumes `TargetTranslateRouteRegistry` for direct
  RVV and RVV+scalar helper routes instead of manually iterating route
  manifests in the tool.
* The remaining ownership mismatch is the RVV+scalar dispatch artifact route
  bundle: it is semantically an RVV-primary finite-binary dispatch route, but
  the current plugin-owned target exporter bundle is keyed under
  `scalar-plugin` with `rvv-plugin` as a dependency. The scalar plugin should
  continue to own scalar standalone fallback callable routes, but the
  RVV-primary dispatch source/header/object artifact registration should be
  moved to an RVV-owned bundle with `scalar-plugin` as an explicit dependency.
* The selected-config/runtime AVL/VL fail-closed contract from the previous
  task must continue to block stale or missing metadata before direct or bundle
  artifact output.

## Scope

In scope:

* Existing dynamic vector i32-vadd and i32-vsub direct RVV microkernel
  source/header/object routes.
* Existing dynamic vector i32-vadd and i32-vsub RVV+scalar dispatch
  source/header/object bundle routes.
* Target artifact exporter plugin-bundle ownership and extension-bundle route
  metadata requirements for these routes.
* C++ registry tests proving owner/dependency behavior and preventing central
  route truth duplication.
* Focused lit/FileCheck and artifact commands that show the existing vadd/vsub
  direct and plan-and-export bundle paths still consume the registered routes.

Out of scope:

* New vmul, i64, LMUL, vector-shape, ToyExt/TensorExt, or generic plugin
  framework expansion as the main result.
* Moving computation semantics into `tcrv.exec`.
* Descriptor-driven computation or direct descriptor-to-C export as
  architecture.
* Python implementation of compiler core, dialects, passes, plugin registry,
  lowering, or emission.
* Performance claims or broad benchmark matrices.
* Removing legacy helper command names unless required by this ownership move.

## Requirements

* Compiler implementation remains in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
* RVV direct source/header/object routes must continue to be registered through
  the RVV target-support route authority and keyed by `rvv-plugin`.
* RVV+scalar dispatch source/header/object routes must be registered by an
  RVV-owned plugin target exporter bundle or equivalent RVV target-support
  bundle, with scalar participation represented as an explicit enabled-plugin
  dependency.
* Scalar standalone fallback source/header/object routes must remain
  scalar-owned.
* Built-in extension bundle route metadata requirements must match the new
  registration ownership: RVV owns RVV direct and RVV+scalar dispatch
  source/header/object route metadata; scalar owns scalar standalone fallback
  route metadata.
* Missing or disabled `rvv-plugin` must not publish RVV direct or RVV+scalar
  dispatch routes.
* Missing or disabled `scalar-plugin` must not publish RVV+scalar dispatch
  routes, but must not disable RVV direct routes.
* `tcrv-translate` direct helper routes and plan-and-export bundle routes must
  keep consuming generic route registries; no RVV-specific route selection
  truth may be reintroduced into the tool.
* Route metadata must continue to identify the RVV plugin / extension family
  consistently for direct RVV routes, and dispatch route metadata must preserve
  conservative no-runtime/performance claims.
* Selected family/config/runtime-VL validation must remain fail-closed after
  the ownership move.
* Dynamic vadd must still emit vadd and dynamic vsub must still emit vsub
  through the neutral source front door and shared finite-binary contracts.
* Fixed-vector vadd extent enforcement and dynamic transfer-tail authority
  must remain intact.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits, and the task is started as current before implementation.
* [x] RVV direct source/header/object artifact routes remain registered under
      `rvv-plugin` through the RVV target-support bundle.
* [x] RVV+scalar dispatch source/header/object artifact routes are registered
      under an RVV-owned plugin target exporter bundle with an explicit
      `scalar-plugin` dependency.
* [x] Scalar standalone fallback artifact routes remain registered under
      `scalar-plugin`.
* [x] Built-in extension bundle metadata requirements move RVV+scalar dispatch
      route metadata from the scalar bundle to the RVV bundle, while retaining
      dependency gating on `scalar-plugin`.
* [x] C++ tests cover duplicate registration, missing/disabled owner plugin,
      missing/disabled dependency plugin, extension-bundle metadata ownership,
      and route metadata preservation for RVV direct and RVV+scalar dispatch
      routes.
* [x] `tcrv-translate` continues to register direct helper routes through
      `TargetTranslateRouteRegistry` and uses generic target artifact route
      export for artifact-backed direct helper routes.
* [x] Dynamic vector i32-vadd and i32-vsub direct artifacts still emit
      `__riscv_vadd_vv_i32m1` and `__riscv_vsub_vv_i32m1` respectively.
* [x] Dynamic vector i32-vadd and i32-vsub plan-and-export bundles still record
      registry-derived source/header/object route metadata and selected
      config/runtime-VL metadata.
* [x] Selected-config fail-closed regressions, family registry regressions,
      fixed-vector vadd extent checks, and dynamic transfer-tail authority
      checks pass.
* [x] Focused build covers `TianChenRVTransforms`, `TianChenRVTarget`,
      `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test`.
* [x] Exact direct and bundle artifact commands are recorded for dynamic vector
      i32-vsub plus at least one vadd regression.
* [x] Fresh `ssh rvv` evidence is collected for dynamic vector i32-vsub if the
      route ownership move changes emitted source/object or bundle
      materialization behavior; otherwise the final report states why the move
      was registry ownership only and did not create a new runtime claim.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] If complete, the task is finished, archived, and committed as one
      coherent commit. If incomplete, the task remains open with an exact next
      continuation point.

## Definition Of Done

* RVV finite-binary direct and RVV-primary dispatch route registration is
  owned by RVV target/plugin surfaces and consumed through generic registries.
* Scalar ownership is limited to scalar standalone fallback routes.
* Generic orchestration stays target-neutral and does not gain RVV/scalar
  family branches.
* Existing dynamic vadd/vsub direct and bundle behavior is preserved with
  focused local checks and bounded RVV evidence where the emitted route changes.

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

* `.trellis/tasks/archive/2026-05/05-13-rvv-selected-config-fail-closed-artifact-validation/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-finite-binary-family-contract-registry/prd.md`

Initial code inspection target:

* `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`
* `include/TianChenRV/Target/RVV/RVVMicrokernel.h`
* `include/TianChenRV/Target/RVVScalarBinaryFamily.h`
* `include/TianChenRV/Target/RVVScalarDispatch.h`
* `include/TianChenRV/Target/TargetArtifactExport.h`
* `include/TianChenRV/Target/TargetTranslateRegistration.h`
* `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`
* `lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp`
* `lib/Target/Builtin/RVVScalarDispatch.cpp`
* `lib/Target/RVV/RVVMicrokernel.cpp`
* `lib/Target/TargetArtifactExport.cpp`
* `lib/Target/TargetTranslateRegistration.cpp`
* `tools/tcrv-translate/tcrv-translate.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* dynamic vector add/sub tests under `test/Transforms/VectorToExec/`
* dynamic vector add/sub bundle tests under
  `test/Target/TargetArtifactBundleExport/`

## Completed This Round

* Rewired built-in target artifact exporter bundle composition so the RVV
  extension bundle registers both RVV direct microkernel source/header/object
  routes and RVV-primary RVV+scalar dispatch source/header/object routes.
* Changed `registerRVVScalarDispatchPluginTargetExporterBundle` so dispatch
  routes are keyed by `rvv-plugin` and require `scalar-plugin` as an enabled
  dependency. Missing or disabled RVV remains the owner failure; missing or
  disabled scalar is now a dependency failure.
* Kept scalar standalone fallback source/header/object registration under the
  scalar extension bundle only.
* Moved RVV+scalar dispatch route metadata requirements from the scalar
  extension bundle to the RVV extension bundle, with `scalar-plugin` declared
  as the route metadata dependency.
* Updated `TargetArtifactExportTest.cpp` to prove the new owner/dependency
  boundary, duplicate bundle rejection, missing/disabled owner and dependency
  behavior, and extension-bundle route metadata ownership.
* Updated the lowering-runtime spec sentence that previously described
  RVV+scalar dispatch as scalar-owned; it now records RVV-owned dispatch route
  registration with scalar dependency.
* Preserved `tcrv-translate` route consumption through
  `TargetTranslateRouteRegistry` and generic exact target artifact route export;
  no route-family truth was added to the tool.

## Route Registration Owner Before / After

Before:

* RVV direct source/header/object routes: `rvv-plugin`.
* Scalar standalone fallback routes: `scalar-plugin`.
* RVV+scalar dispatch source/header/object routes: `scalar-plugin`, requiring
  `rvv-plugin`.

After:

* RVV direct source/header/object routes: `rvv-plugin`.
* Scalar standalone fallback routes: `scalar-plugin`.
* RVV+scalar dispatch source/header/object routes: `rvv-plugin`, requiring
  `scalar-plugin`.

Central route truth removed:

* The scalar extension bundle no longer owns RVV+scalar dispatch route metadata
  requirements or dispatch target exporter bundle registration.
* `tcrv-translate` continues to use target translate route registration and
  generic target artifact route lookup rather than route-specific loops.

## Exact Artifact Commands

```bash
ART=artifacts/tmp/rvv_plugin_owned_route_registration
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
python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family i32-vsub --input test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --expect-selected-kernel=frontend_vector_dynamic_i32_vsub --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv --artifact-root artifacts/tmp/rvv_plugin_owned_route_registration/e2e --run-id 20260513T-rvv-plugin-owned-route-registration-vsub --overwrite --timeout 120
```

Result:

* Evidence file:
  `artifacts/tmp/rvv_plugin_owned_route_registration/e2e/20260513T-rvv-plugin-owned-route-registration-vsub/evidence.json`.
* `status = success`, `ssh_evidence = true`,
  `expected_selected_kernel = frontend_vector_dynamic_i32_vsub`,
  `runtime_element_counts = [7, 16, 23]`,
  `source_export_route = tcrv-export-rvv-i32-vsub-microkernel-c`.

## Checks Run

* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-plugin-owned-artifact-route-registration`
* `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
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
* Trellis validation before finish/archive and after archive.

## Self-Repair

* The first evidence-file summary probe queried fields that are emitted in the
  command summary rather than as top-level evidence fields. I re-read the
  evidence JSON fields actually used by the runner and recorded the bounded
  result fields above.

## Spec Update Judgment

Spec update was required because this round intentionally changed the
cross-layer owner/dependency contract for RVV+scalar dispatch artifact route
registration. The update is limited to
`.trellis/spec/lowering-runtime/emission-runtime-contract.md`: RVV+scalar
dispatch routes are now documented as `rvv-plugin` target-exporter routes with
an enabled `scalar-plugin` dependency.
