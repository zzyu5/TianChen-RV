# Scalar finite source route metadata coverage

## Goal

Generalize compiler-owned `TargetArtifactRouteMetadata` and generic route
preflight to every finite scalar fallback source route currently contributed by
`scalar-plugin`. This closes the scalar side of the route metadata contract
after the RVV finite source-route work: supported scalar source candidates must
carry descriptor-derived runtime ABI, selected scalar descriptor, runtime count,
and conservative no-claim metadata before scalar C source/header/object helpers
can consume them.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial worktree was clean at `4e956c4 feat(rvv): cover finite source route
  metadata`.
* No `.trellis/.current-task` existed at takeover; this task was created as
  `.trellis/tasks/05-11-05-11-scalar-finite-source-route-metadata-coverage`.
* Latest complete supervisor audit/review input is under
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0093-20260511T025021Z/`.
  It shows the previous RVV finite source-route metadata task completed,
  archived, committed, and left a clean worktree.
* Current RVV source routes register descriptor-derived
  `TargetArtifactRouteMetadata` for all finite RVV source routes and the RVV
  bundle declares those route metadata requirements.
* Current scalar source routes in
  `lib/Target/Scalar/ScalarMicrokernel.cpp` register plain
  `TargetArtifactExporter` instances without route metadata, and
  `registerScalarExtensionBundle` in
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` declares no scalar
  route metadata requirement.
* Scalar source candidate validation already has a route-local typed ABI
  preflight, but stale selected scalar descriptor metadata is not consumed by
  the generic `TargetArtifactRouteMetadata` validator before route-local scalar
  emission.
* The finite scalar source family surface is currently the scalar half of the
  RVV/scalar binary family descriptors: add/sub/mul for i32 and add/sub/mul
  for i64.

## Requirements

* Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only run existing task/test tooling.
* Preserve `tcrv.exec` as execution/capability/variant organization only; do
  not add high-level tensor/tile compute semantics.
* Add scalar-plugin-owned source-route metadata for every finite scalar binary
  source route currently returned by `getScalarMicrokernelFamilySpecs()`.
* The metadata must be descriptor-derived from
  `rvv_scalar::ScalarBinaryMicrokernelDescriptor`: runtime ABI identity, scalar
  dtype/family/operator/lowering descriptor requirements, runtime element-count
  metadata requirement, and conservative compiler-only/no runtime/no hardware/no
  performance claim fields.
* Generic target export may continue validating route metadata shape, runtime
  ABI equality, selected-plan metadata requirements, and claim fields. It must
  not gain scalar/RVV/dtype/operator/family semantic branches.
* Scalar-specific descriptor fields and route-local validation remain in scalar
  target/export code. Header/object composite helpers must keep reusing the
  validated scalar source candidate through their existing route-local preflight
  callback.
* `Scalar` built-in extension bundle registration must declare metadata
  requirements for all finite scalar source routes once those exporters publish
  registered route metadata.
* Existing Toy, Template, Offload, RVV, RVV+Scalar dispatch, and RVV smoke-probe
  behavior must not be weakened.
* Do not claim scalar runtime correctness, linked runtime integration, RVV
  hardware execution, throughput, latency, or performance from this task.

## Acceptance Criteria

* Every finite scalar source route registered by
  `registerScalarMicrokernelTargetExporters` publishes non-empty
  `TargetArtifactRouteMetadata` with:
  * scalar runtime ABI, runtime ABI kind/name, and runtime glue role;
  * exact selected-plan requirements for scalar dtype, family, operator, and
    lowering descriptor;
  * presence requirement for scalar runtime element-count C name or equivalent
    runtime count boundary metadata;
  * conservative claim fields showing compiler artifact only and no runtime
    correctness, hardware execution, or performance claim.
* `registerScalarExtensionBundle` declares target artifact route metadata
  requirements for all finite scalar source routes.
* Focused C++ coverage enumerates all finite scalar source routes and asserts
  descriptor metadata is present, consumed by generic preflight, and fails
  closed for stale runtime ABI, stale selected family/operator/dtype, missing
  scalar runtime-count metadata, and duplicate registration.
* Existing scalar source/header/object route-local preflight continues to reject
  stale family ABI metadata before scalar source/header/object output.
* Focused lit/FileCheck coverage includes at least one non-legacy scalar route
  and proves deterministic source export plus conservative evidence wording.
* Trellis task context validates before finish/archive.

## Non-Goals

* No new scalar/RVV dtype, operator, kernel family, runtime ABI shape, or
  frontend lowering.
* No generic scalar backend, LLVM lowering rewrite, linker/runtime integration,
  or broad benchmark matrix.
* No new RVV hardware run or `ssh rvv` evidence claim.
* No core pass branch on scalar, RVV, dtype, route id, operator, or vendor
  details.
* No docs-only, report-only, smoke-only, helper-only, or test-harness-only
  closeout without active C++ producer/consumer behavior.

## Validation Plan

* Build focused targets:
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-scalar-extension-plugin-test`, `tcrv-opt`, and
  `tcrv-translate` as needed by touched tests.
* Run `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`.
* Run focused scalar plugin C++ tests if scalar descriptor/planning surfaces are
  touched.
* Run focused lit/FileCheck coverage for scalar source/header/object export and
  target artifact route metadata regressions, including at least one non-legacy
  scalar family.
* Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-05-11-scalar-finite-source-route-metadata-coverage`.
* Run `git diff --check`.
* Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` after focused checks pass if the build directory remains usable.
* Do not run `ssh rvv` unless a later change explicitly creates a new RVV
  hardware correctness/performance claim. This task should not require it.

## Technical Notes

* Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior task context read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-finite-source-route-metadata-coverage/prd.md`,
  `.trellis/tasks/archive/2026-05/05-11-extension-bundle-registration-frontdoor/prd.md`,
  and
  `.trellis/tasks/archive/2026-05/05-11-template-extension-zero-core-integration-slice/prd.md`.
* Initial code inspection found the implementation target in
  `lib/Target/Scalar/ScalarMicrokernel.cpp` around
  `validateScalarMicrokernelSourceCandidate` and
  `registerScalarMicrokernelTargetExporters`.
* Initial code inspection found the bundle declaration target in
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` around
  `registerScalarExtensionBundle`.
* `test/Target/TargetArtifactExportTest.cpp` already has reusable helpers for
  RVV route metadata preflight and scalar stale ABI validation.

## Definition Of Done

The task is done when finite scalar source routes publish descriptor-derived
route metadata, the Scalar bundle declares and validates those metadata
requirements, focused C++ and lit tests prove generic preflight consumption and
existing scalar route-local preflight behavior, Trellis context validates and
archives, and one coherent commit records the module.

If unfinished, leave the task open and record the exact continuation point:
scalar route metadata builder, Scalar bundle requirement, generic preflight
test coverage, scalar source fixture update, focused build/test failure,
Trellis validation, archive, or commit.

## Implementation Completed

* Added shared scalar selected-plan metadata names, roles, notes, and descriptor
  appender helpers in `include/TianChenRV/Target/RVVScalarBinaryFamily.h`.
* Made `scalar-plugin` emission plans append scalar dtype, family, operator,
  lowering descriptor, and runtime element-count C-name metadata derived from
  the selected finite scalar binary descriptor and runtime ABI parameters.
* Registered descriptor-derived `TargetArtifactRouteMetadata` for every finite
  scalar source route in `lib/Target/Scalar/ScalarMicrokernel.cpp`, including
  exact scalar selected-plan requirements, runtime count C-name presence, and
  compiler-only/no-runtime/no-hardware/no-performance claim fields.
* Kept scalar route-local validation in scalar target code while adding the
  route-specific runtime element-count C-name equality check after generic
  route metadata preflight succeeds.
* Made the built-in Scalar extension bundle declare target artifact route
  metadata requirements for all finite scalar source routes.
* Extended focused C++ coverage to enumerate all finite scalar source routes,
  assert registered route metadata and conservative claim fields, and prove
  generic preflight rejects stale runtime ABI, stale scalar dtype/family/operator
  metadata, and missing runtime count metadata.
* Updated scalar lit/FileCheck coverage so the non-legacy `i32-vmul` source
  route proves deterministic source export and visible scalar
  `selected_plan_metadata` without claiming runtime correctness, hardware
  execution, throughput, latency, or performance.

## Validation Results

* `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-target-artifact-export-test tianchenrv-scalar-extension-plugin-test
  tcrv-opt tcrv-translate -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`
* Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `Target/ArtifactExport/scalar-target-source-artifact-routes.test`,
  `Target/ArtifactExport/scalar-target-vmul-source-artifact-routes.test`,
  `Target/ArtifactExport/scalar-target-header-object-artifact-routes.test`,
  `Transforms/EmissionReadiness/materialize-emission-plans-scalar-microkernel.mlir`,
  `Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-generic-route.mlir`,
  `Target/RVVScalarDispatch/rvv-scalar-i32-vmul-dispatch-generic-route.mlir`,
  `Target/RVVScalarDispatch/rvv-scalar-i64-vsub-dispatch-generic-route.mlir`,
  and `Target/ArtifactExport/target-artifact-export-registry.test` passed.
* Focused adjacent regression lit from `artifacts/tmp/tianchenrv-build/test`:
  `Target/ArtifactExport/artifact-export-coherence-preflight.test`,
  `Target/ArtifactExport/offload-runtime-descriptor-artifact-route.test`,
  `Target/TargetArtifactBundleExport/target-artifact-bundle-guards.mlir`,
  `Transforms/ExecutionPlanCoherence/execution-plan-coherence-negative.mlir`,
  `Transforms/ExecutionPlanning/execution-planning-pipeline-builtin.mlir`,
  `Transforms/LinalgToExec/linalg-i32-vsub-to-scalar-artifact.mlir`,
  `Transforms/LinalgToExec/linalg-i32-vmul-to-scalar-artifact.mlir`, and
  `Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir` passed.
* `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-05-11-scalar-finite-source-route-metadata-coverage`
* `git diff --check`
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2`: 205/205 tests passed.

Self-repair performed: updated the hand-authored missing-scalar-microkernel
fixture so it carries the scalar selected-plan metadata now required by generic
source-route preflight, preserving the intended missing executable diagnostic.

No `ssh rvv` run was required or performed because this task made no new RVV
runtime correctness, hardware execution, throughput, latency, or performance
claim.

Spec update decision: no `.trellis/spec/` change was needed; this round
implements the existing plugin locality, scalar fallback source-route,
target-artifact preflight, and no-claim evidence contracts.
