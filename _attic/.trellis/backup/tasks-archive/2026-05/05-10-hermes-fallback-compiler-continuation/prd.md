# Manifest-derived RVV+scalar dispatch route registration

## Goal

Make RVV+scalar dispatch artifact and direct translate route registration consume the finite RVV+scalar binary family registry as the single source of truth. This keeps the actual target-owned route surface aligned with the descriptor registry used by planning, emission metadata, bundle records, and Python evidence tooling.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial state on 2026-05-10: `git status --short` is clean and `HEAD` is `b04e5a1 feat(rvv): add dispatch bundle ssh evidence`.
* `.trellis/.current-task` is absent, so there is no active task to continue.
* Latest Hermes audit and review input are under `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0049-20260510T095832Z/`.
* The latest completed commit added front-door RVV+scalar dispatch bundle ssh evidence for a bounded `i64-vmul` path.
* The brief forbids selecting smoke/probe/guardrail/test-harness work unless it is the only blocker for a real compiler path.
* The relevant spec requires the RVV+scalar dispatch route manifest to cover exactly `i32-vadd`, `i32-vsub`, `i32-vmul`, `i64-vadd`, `i64-vsub`, and `i64-vmul`, with five route kinds per family.
* Current `lib/Target/Builtin/RVVScalarDispatch.cpp` already has a manifest object, but it hand-appends the six families directly instead of iterating `getRVVScalarBinaryFamilyDescriptors()`.
* Current C++ tests validate manifest shape, but the built-in translate route count uses a hard-coded total rather than deriving it from the two route manifests that feed `registerBuiltinTargetTranslateRoutes`.

## Assumptions

* This round should harden the active RVV target-artifact / dispatch-bundle route surface, because that is the newest committed mainline in the audit.
* No RVV runtime or correctness claim is needed for this task, because the behavior change is route registration / manifest ownership, not generated runtime output.
* Python may be used for runners and artifact parsing, but compiler core behavior must remain in C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck.

## Open Questions

* None.

## Requirements (evolving)

* `getRVVScalarDispatchRouteManifest()` must derive its route entries by iterating `getRVVScalarBinaryFamilyDescriptors()`, not by maintaining a separate explicit six-family append list.
* Source/header/object/self-check source/self-check object route metadata must remain identical for all existing finite families.
* Built-in target translate route tests must derive the expected route count from `getRVVMicrokernelDirectRouteManifest().size()` plus `getRVVScalarDispatchRouteManifest().size()` instead of a stale magic number.
* The change must stay in target-owned route registration and test code; no core `tcrv.exec` compute semantics, generic linalg lowering expansion, Python-owned compiler behavior, or new RVV evidence claim.

## Acceptance Criteria (evolving)

* [x] `lib/Target/Builtin/RVVScalarDispatch.cpp` dispatch route manifest construction consumes `getRVVScalarBinaryFamilyDescriptors()`.
* [x] Existing manifest metadata remains stable for all six finite families and five route kinds.
* [x] `test/Target/TargetArtifactExportTest.cpp` derives built-in translate route count from actual manifests.
* [x] Focused C++/lit checks covering target artifact export / route registry pass.
* [x] Trellis task context validates and task status is truthful.

## Definition of Done

* Relevant CMake build and focused tests pass after any self-repair.
* Task is finished and archived if the acceptance criteria are met.
* One coherent commit is created for the completed round.

## Out of Scope

* Performance claims.
* Broad matrix validation unrelated to the selected module behavior.
* New RVV runtime or correctness evidence.
* Implementing compiler core, dialects, passes, plugin registry, capability model, lowering, or emission as Python data structures.
* New high-level tensor/tile IR semantics in `tcrv.exec`.

## Technical Notes

* Latest audit: `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0049-20260510T095832Z/repo_audit.md`.
* Latest review input: `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0049-20260510T095832Z/review_input.md`.
* Project spec index: `.trellis/spec/index.md`.
* Relevant specs:
  * `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/plugin-protocol/locality-contract.md`
  * `.trellis/spec/implementation-stack/compiler-stack-contract.md`
  * `.trellis/spec/testing/mlir-testing-contract.md`
* Relevant code:
  * `include/TianChenRV/Target/RVVScalarDispatch.h`
  * `include/TianChenRV/Target/RVVScalarBinaryFamily.h`
  * `lib/Target/Builtin/RVVScalarDispatch.cpp`
  * `lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp`
  * `test/Target/TargetArtifactExportTest.cpp`
  * `test/Target/I32BinaryFamilyRegistryTest.cpp`

## Completion Evidence

* `lib/Target/Builtin/RVVScalarDispatch.cpp` now builds the dispatch route manifest by iterating `getRVVScalarBinaryFamilyDescriptors()` and appending the five route kinds for each finite family.
* `test/Target/TargetArtifactExportTest.cpp` derives the expected built-in target translate route count from `getRVVMicrokernelDirectRouteManifest().size() + getRVVScalarDispatchRouteManifest().size()`.
* Checks passed:
  * `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`
  * `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
  * `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
  * `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'target-artifact-export|i32-binary-family-registry|rvv-scalar-dispatch'` from `artifacts/tmp/tianchenrv-build/test`
  * `git diff --check`
  * `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
