# RVV finite source route metadata coverage

## Goal

Generalize RVV target-owned `TargetArtifactRouteMetadata` coverage across every
currently supported finite RVV binary source route in the direct-route
manifest. The task removes the prior single-route `i32-vsub` metadata proof and
makes generic target artifact export preflight all supported RVV source routes
through descriptor-derived route metadata before route-local RVV emission.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial worktree was clean at `4f91923 feat(template): add zero-core extension
  integration slice`.
* No `.trellis/.current-task` existed at takeover; this task was created and
  started as `.trellis/tasks/05-11-rvv-finite-source-route-metadata-coverage`.
* The archived `rvv-executable-artifact-route-metadata-registration` task
  adopted `TargetArtifactRouteMetadata` for only the bounded
  `tcrv-export-rvv-i32-vsub-microkernel-c` source route.
* The archived `extension-bundle-registration-frontdoor` task made the RVV
  extension bundle declare that one `i32-vsub` source route as its route
  metadata regression requirement.
* The archived `template-extension-zero-core-integration-slice` task confirmed
  built-in extension bundles and route metadata registration now run through a
  clean generic frontdoor, with full `check-tianchenrv` passing.
* Current code already has a descriptor-driven
  `buildRVVMicrokernelSourceRouteMetadata(const RVVBinaryFamilyDescriptor &)`
  helper, but `registerRVVMicrokernelTargetExporters` attaches its metadata only
  when `family.familyID == getI32VSubFamilyDescriptor().familyID`.
* `getRVVMicrokernelDirectRouteManifest()` enumerates all finite RVV binary
  families from `getRVVBinaryFamilyDescriptors()` and route kinds source,
  header, and object.
* The currently supported finite RVV binary family surface is add/sub/mul for
  i32 and add/sub/mul for i64. This task must not add new families, dtypes,
  operators, LMULs, kernels, or a generic RVV backend.

## Requirements

* Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only orchestrate existing scripts/tests.
* Preserve `tcrv.exec` as execution/capability/variant organization only; do
  not add compute semantics.
* Make every supported finite RVV binary source route register non-empty
  `TargetArtifactRouteMetadata` derived from its
  `RVVBinaryFamilyDescriptor`.
* Remove or replace the prior `i32-vsub`-only metadata branch. If the legacy
  generic i32-vadd route remains for CLI compatibility, it must still register
  descriptor-derived metadata and must not bypass selected-plan source-route
  preflight.
* Generic target export code may validate registered route metadata shape,
  runtime ABI equality, selected-plan metadata requirements, and claim fields.
  It must not gain hard-coded RVV, dtype, family, operator, LMUL, or route
  semantic branches.
* RVV-specific descriptor fields and validation remain in RVV target/plugin
  code: family/operator/dtype, vector shape, SEW/LMUL/tail/mask policy,
  vector/setvl suffixes, capability mirror fields, runtime AVL/VL boundary,
  runtime element-count ABI name, runtime ABI identity, and conservative claim
  fields.
* Missing or stale runtime ABI, wrong selected family/operator/dtype, missing
  selected vector shape, missing SEW/LMUL capability mirrors, missing runtime
  AVL/VL boundary, and unknown source route must fail closed before
  route-local source emission.
* Header/object composite routes must remain compatible with the source-route
  preflight path by reusing the validated source candidate. This task must not
  become a full header/object route redesign.
* Dry-run source artifacts and bundle evidence must keep the boundary explicit:
  compile/export success is compiler artifact evidence only and is not RVV
  runtime correctness, hardware execution, or performance evidence.

## Acceptance Criteria

* All source entries in `getRVVMicrokernelDirectRouteManifest()` that have
  `routeKind == Source` publish route metadata with runtime ABI fields,
  selected binary descriptor requirements, selected vector-shape requirements,
  capability mirror requirements, runtime AVL/VL boundary requirements,
  runtime element-count C-name requirement, and no-runtime/no-performance claim
  fields.
* The RVV extension bundle declares target artifact route metadata requirements
  for all finite RVV source-route manifest entries, not only `i32-vsub`.
* Focused C++ coverage enumerates all finite RVV direct source routes and
  asserts registered metadata is present and consumed by generic preflight.
* Focused fail-closed coverage includes stale runtime ABI kind/name, wrong
  selected family/operator/dtype, missing selected vector shape, missing
  SEW/LMUL capability mirror, missing runtime AVL/VL boundary, and unknown
  route.
* Existing `i32-vsub` route metadata coverage remains a regression.
* Lit/FileCheck coverage includes at least one source route that was not the
  original `i32-vsub` proof route and proves deterministic dry-run source
  export plus compile/export-vs-runtime-evidence boundary text.
* Focused RVV scalar dispatch and bundle tests affected by route metadata
  continue to pass.
* Trellis task context validates before finish/archive.

## Non-Goals

* No new RVV dtype, LMUL, operator, kernel family, vector shape, or broad test
  matrix beyond current finite descriptors.
* No MLIR vector/scalable-vector lowering route, generic RVV backend, or new
  extension dialect semantics.
* No new runtime correctness claim, hardware execution claim, throughput,
  latency, or performance work.
* No IME, AME, Sophgo/offload, Template, Toy, or Scalar behavior changes except
  narrow compatibility fixes for shared route metadata tests.
* No Python compiler internals.
* No docs-only, helper-only, smoke-only, report-only, or evidence-packaging-only
  closeout without active C++ RVV route metadata producer/consumer behavior.

## Validation Plan

* Build focused targets:
  `tianchenrv-target-artifact-export-test`, `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-binary-planning-test`, `tcrv-opt`, and `tcrv-translate` as
  needed by touched code/tests.
* Run `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`.
* Run focused RVV plugin tests if planning surfaces are touched.
* Run focused lit/FileCheck coverage for RVV scalar dispatch and bundle routes
  affected by source-route metadata preflight, including a non-`i32-vsub`
  route.
* Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-rvv-finite-source-route-metadata-coverage`.
* Run `git diff --check`.
* Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` after focused checks pass if the build directory remains usable.
* Do not run `ssh rvv` unless this task intentionally creates a new runtime
  correctness claim. The current plan does not require fresh RVV hardware
  evidence because it only strengthens compiler export preflight metadata.

## Technical Notes

* Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/validation/experiment-reference.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Archived PRDs read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-executable-artifact-route-metadata-registration/prd.md`,
  `.trellis/tasks/archive/2026-05/05-11-extension-bundle-registration-frontdoor/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-11-template-extension-zero-core-integration-slice/prd.md`.
* Initial code inspection found the one-off source-route metadata branch in
  `lib/Target/RVV/RVVMicrokernel.cpp` inside
  `registerRVVMicrokernelTargetExporters`.
* Initial code inspection found the RVV extension bundle one-off metadata
  requirement in `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` inside
  `registerRVVExtensionBundle`.
* Generic route metadata validation already exists in
  `lib/Target/TargetArtifactExport.cpp` and should remain generic.

## Definition Of Done

The task is done when descriptor-driven RVV source-route metadata covers all
finite RVV binary source-route manifest entries, generic target export consumes
that metadata and fails stale/missing selected-plan surfaces before RVV source
emission, focused C++ and lit tests pass, dry-run artifacts remain explicit
compiler-only evidence, Trellis context validates and archives, and one
coherent commit records the module.

If unfinished, leave this task open and record the exact continuation point:
descriptor-driven metadata builder, manifest iteration, exporter preflight
hookup, stale/missing-field diagnostic, non-`i32-vsub` lit coverage,
header/object compatibility, focused test failure, or Trellis archive step.

## Implementation Completed

* Replaced the `i32-vsub`-only source-route metadata registration in
  `lib/Target/RVV/RVVMicrokernel.cpp` with descriptor-driven
  `buildRVVMicrokernelSourceRouteMetadata(family)` registration for every
  finite RVV direct-route manifest source entry.
* Made RVV source-route candidate validation always require the selected
  binary descriptor metadata, selected vector-shape metadata, capability mirror
  metadata, runtime AVL/VL boundary metadata, and runtime element-count C-name
  metadata before route-local source emission.
* Generalized the built-in RVV extension bundle frontdoor in
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` so generic target
  export preflights all RVV source routes with route metadata requirements
  instead of only the old `i32-vsub` proof route.
* Extended `test/Target/TargetArtifactExportTest.cpp` to enumerate all finite
  RVV source-route manifest entries, assert descriptor-derived metadata
  coverage, and prove generic preflight rejects missing/stale selected-plan
  metadata before route-local emission.
* Updated lit/FileCheck fixtures so hand-authored RVV source artifact plans
  carry the descriptor metadata now required by source-route preflight while
  preserving their intended negative diagnostics.

## Validation Results

* `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-target-artifact-export-test -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-binary-planning-test
  tcrv-opt tcrv-translate -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
* Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-i32m2-generic-route.mlir`,
  `Target/RVVScalarDispatch/rvv-scalar-i32-vmul-dispatch-generic-route.mlir`,
  `Target/RVVScalarDispatch/rvv-scalar-i64-vmul-dispatch-generic-route.mlir`,
  `Scripts/rvv-scalar-dispatch-bundle-e2e.test`, and
  `Scripts/rvv-microkernel-bundle-e2e.test` passed.
* Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `Scripts/rvv-scalar-dispatch-e2e.test`,
  `Target/ArtifactExport/target-artifact-export-registry.test`, and
  `Target/TargetArtifactBundleExport/target-artifact-bundle-guards.mlir`
  passed.
* Focused repair lit from `artifacts/tmp/tianchenrv-build/test`:
  `Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir`,
  `Transforms/LinalgToExec/linalg-i64-vmul-to-rvv-artifact.mlir`,
  `Target/ArtifactExport/target-source-artifact-routes.test`, and
  `Target/TargetArtifactBundleExport/target-artifact-bundle-guards.mlir`
  passed after fixture updates.
* `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-rvv-finite-source-route-metadata-coverage`
* `git diff --check`
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2`: 205/205 tests passed.

Self-repair performed: updated stale FileCheck diagnostics and hand-authored
artifact fixtures whose selected plans did not yet carry the descriptor
metadata required by the generalized source-route preflight.

No `ssh rvv` run was required or performed because this task made no new RVV
runtime correctness, hardware execution, throughput, latency, or performance
claim.

Spec update decision: no `.trellis/spec/` change was needed; this round
implements existing RVV/plugin/export locality, metadata preflight, and evidence
boundary specs rather than introducing a new convention.
