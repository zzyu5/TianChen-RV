# RVV binary artifact manifests as target-owned family source of truth

## Goal

Refactor finite RVV binary artifact route manifests so target-owned C++
manifest APIs derive direct RVV microkernel and RVV+scalar dispatch route
records from the finite family registries. Registration and export tests should
consume those manifests instead of repeating family/route totals or
family-by-family route lists.

This is a C++/MLIR target-support source-of-truth cleanup. It is not an
evidence runner, script packaging, runtime proof, performance claim, or Python
compiler-core task.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial state for this round is clean at `538d8a0 refactor(rvv): derive dispatch routes from family registry`.
* `.trellis/.current-task` was missing, so this task was created from the Hermes brief.
* The archived prior task is `.trellis/tasks/archive/2026-05/05-10-hermes-fallback-compiler-continuation`.
* The prior task made `getRVVScalarDispatchRouteManifest()` iterate `getRVVScalarBinaryFamilyDescriptors()` and made one translate-route count test sum the direct and dispatch manifests.
* Current direct RVV microkernel route manifest in `lib/Target/RVV/RVVMicrokernel.cpp` still repeats the six finite families as a handwritten 18-entry source/header/object array.
* The finite RVV binary family registry owns the supported family set:
  `i32-vadd`, `i32-vsub`, `i32-vmul`, `i64-vadd`, `i64-vsub`, `i64-vmul`.

## Requirements

* Preserve the finite family set and existing vector-shape behavior exactly.
* Keep direct RVV microkernel routes and RVV+scalar dispatch routes distinct.
* Make direct RVV microkernel route manifest derivation consistent with the
  dispatch manifest style: family registry owns family facts; route manifest
  derives per-route records from those facts.
* Remove the remaining direct RVV handwritten finite family/route list from
  target C++ code.
* Provide or refine a small target-owned manifest API so registration/export
  tests can ask for route records, route kinds, and expected finite counts
  instead of hard-coding route totals.
* Update focused tests to verify representative route names and at least one
  all-family invariant derived from the manifest.
* Do not add generic RVV backend support, new RVV families, new dtype support,
  new vector shapes, or new runtime/performance claims.
* Keep compiler implementation in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck;
  Python remains limited to tooling/runners if invoked by existing tests.

## Acceptance Criteria

* [x] `getRVVMicrokernelDirectRouteManifest()` is derived by iterating
  `getRVVBinaryFamilyDescriptors()` rather than by listing all six families.
* [x] Direct manifest API exposes target-owned route-kind/count information
  that tests can consume without raw `18` route totals.
* [x] RVV+scalar dispatch manifest retains registry-derived construction and
  exposes matching route-kind/count information where useful for tests.
* [x] Focused C++ tests verify representative route names and all-family
  source/header/object or source/header/object/self-check invariants from the
  manifests.
* [x] Existing bundle/export coverage remains intact, with no new runtime,
  correctness, or performance claim.
* [x] Focused build, C++ tests, lit filters, and required route-registration
  check target pass, or failures are recorded truthfully with the task left
  open.

## Definition Of Done

* `git diff --check` passes.
* Focused affected targets build, including
  `tianchenrv-target-artifact-export-test` and
  `tianchenrv-i32-binary-family-registry-test`.
* Focused test binaries pass.
* Focused lit filters covering target artifact export, RVV microkernel bundle,
  and RVV scalar dispatch bundle pass from the build-tree test directory if
  repo-root lit pathing is not viable.
* `check-tianchenrv` is run if route registration, target export, or public
  tool registration is touched.
* Task validates before finish and archived task validates after finish.
* One coherent commit is created only if the acceptance criteria are met.

## Out Of Scope

* New Python evidence runner or script-only evidence packaging.
* Remote `ssh rvv` runtime/correctness/performance evidence.
* New RVV operation family, dtype, vector shape, mask/tail policy, backend
  capability, or generic RVV backend claim.
* Compute semantics in `tcrv.exec`.
* Extension-specific branches in generic core passes.
* Broad smoke matrices, dashboards, report-only changes, or task-metadata-only
  closeout.

## Technical Notes

* Relevant specs:
  * `.trellis/spec/index.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  * `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
  * `.trellis/spec/testing/mlir-testing-contract.md`
* Relevant code:
  * `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`
  * `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
  * `include/TianChenRV/Target/RVV/RVVMicrokernel.h`
  * `include/TianChenRV/Target/RVVScalarBinaryFamily.h`
  * `include/TianChenRV/Target/RVVScalarDispatch.h`
  * `lib/Target/RVV/RVVMicrokernel.cpp`
  * `lib/Target/Builtin/RVVScalarDispatch.cpp`
  * `lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp`
  * `test/Target/TargetArtifactExportTest.cpp`
  * `test/Target/I32BinaryFamilyRegistryTest.cpp`
  * `test/Scripts/rvv-microkernel-bundle-e2e.test`
  * `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`

## Completion Evidence

* Removed the direct RVV microkernel manifest's handwritten 18-entry
  source/header/object list and replaced it with
  `getRVVBinaryFamilyDescriptors()` x `getRVVMicrokernelDirectRouteKinds()`.
* Added target-owned route-kind/count APIs for direct RVV microkernel routes and
  RVV+scalar dispatch routes.
* Updated dispatch manifest construction to use the same route-kind x family
  derivation pattern while preserving dispatch source/header/object/self-check
  metadata.
* Updated focused C++ tests to consume manifest count APIs, verify
  route-kind completeness per family, and check representative direct and
  dispatch route names.
* No remote `ssh rvv` run was needed or claimed; this round only changed local
  C++ target-support route manifest ownership.
* Checks passed:
  * `git diff --check`
  * `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`
  * `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
  * `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
  * From `artifacts/tmp/tianchenrv-build/test`:
    `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --show-pass . --filter 'target-artifact-export|i32-binary-family-registry|rvv-microkernel-bundle|rvv-scalar-dispatch-bundle'`
  * `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
