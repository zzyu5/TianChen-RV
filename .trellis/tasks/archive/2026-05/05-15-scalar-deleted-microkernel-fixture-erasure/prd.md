# Scalar Deleted Microkernel Fixture Erasure

## Goal

Remove active scalar deleted-microkernel fixture authority from live tests and durable spec text. The scalar fallback surface should remain metadata-only and non-executable, without preserving exact historical finite-family scalar microkernel op spellings as current route anchors.

## Requirements

* Delete or rewrite the active scalar dialect lit fixture that hand-authors deleted `tcrv_scalar.i32/i64_*_microkernel` syntax.
* Remove scalar plugin C++ test helpers and assertions that scan live IR for deleted scalar `_microkernel` op names as named absence fixtures.
* Update directly related active spec/TableGen comments that preserve exact historical scalar microkernel op names as durable scalar route contracts.
* Preserve current scalar fallback metadata-only and legality behavior through positive metadata-only tests.
* Keep this round deletion/refactor-only.

## Acceptance Criteria

* [x] Active tests no longer hand-author deleted `tcrv_scalar.i32/i64_*_microkernel` syntax.
* [x] Active tests no longer scan for `_microkernel` as a named deleted-op absence fixture.
* [x] Active specs/comments keep the rule that scalar fallback is metadata-only and non-executable.
* [x] Active specs/comments no longer preserve exact old finite-family scalar microkernel op spellings as contract anchors.
* [x] No replacement op, alias, wrapper, compatibility path, descriptor path, direct C exporter, runtime lowering, route id, or rebuild implementation is added.
* [x] Focused active-surface reference scan shows no authoritative remaining hits for the deleted scalar microkernel spellings, `hasDeletedScalarMicrokernelOp`, or `deleted-microkernel.mlir`, except explicitly justified non-live governance text if any.
* [x] Focused build/test checks for affected scalar/plugin surfaces pass, or failures are recorded as unrelated baseline gaps.
* [x] `check-tianchenrv` is attempted or its current status is reported.
* [x] `git diff --check` and Trellis task validation pass.
* [x] Task is finished/archived and one coherent commit is created if the round completes.

## Definition of Done

* PRD and task context reflect the deletion-only scope.
* Source/spec/test changes are limited to the active scalar deleted-microkernel fixture authority surface and directly related metadata-only wording.
* Focused validation has been run and any self-repair is recorded.
* The worktree is clean after the final commit.

## Technical Approach

Delete the obsolete negative scalar microkernel syntax fixture rather than replacing it with another deleted-op fixture. Rework scalar plugin tests to verify current metadata-only plugin behavior directly, without string-scanning for old microkernel op names. Edit scalar fallback spec or active TableGen comments only where they preserve exact historical scalar microkernel spellings as durable contract text.

## Decision (ADR-lite)

**Context**: The Wrong Logic Deletion Campaign requires deleting obsolete descriptor/direct-exporter/finite-route authority before rebuild. The current scalar surface still preserves old scalar microkernel op names in tests/spec text.

**Decision**: Treat old scalar microkernel spellings as obsolete route authority. Remove their active fixtures and named absence scans while preserving metadata-only scalar fallback legality coverage.

**Consequences**: This round will not create a replacement scalar execution path. Any resulting missing-architecture failures should be reported as deletion gaps instead of restoring deleted microkernel syntax.

## Out of Scope

* RVV rebuild or scalar runtime lowering.
* Common EmitC implementation.
* Executable plugin template.
* New scalar microkernel op, pass option, route id, alias, wrapper, compatibility mode, descriptor route, direct C exporter, source/header/object/bundle route, or ssh evidence.
* Edits to `artifacts/tmp`, archived Trellis tasks, workspace journals, or supervisor prompt guardrails just to reduce scan counts.

## Technical Notes

* Direction source: current user/Hermes task brief for `Scalar Deleted Microkernel Fixture Erasure`.
* Required first reads: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`, `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, `.trellis/spec/testing/mlir-testing-contract.md`, `include/TianChenRV/Dialect/Scalar/IR/ScalarOps.td`, `test/Dialect/Scalar/deleted-microkernel.mlir`, `test/Plugin/ScalarExtensionPluginTest.cpp`, `test/Plugin/scalar-extension-plugin.test`.
* Read `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp` only if deleting test assertions exposes directly related active plugin code residue.
* Focused build passed: `ninja -C build tcrv-opt tianchenrv-scalar-extension-plugin-test`.
* Focused C++ test passed: `./build/bin/tianchenrv-scalar-extension-plugin-test`.
* Focused lit passed: `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Plugin/scalar-extension-plugin.test Dialect/Scalar` from `build/test`.
* Focused live ref scan excluding `.trellis/tasks`, `.trellis/workspace`, `.git`, `build`, and artifacts returned no hits for the exact deleted scalar microkernel op names, `hasDeletedScalarMicrokernelOp`, or `deleted-microkernel.mlir`.
* Initial `ninja -C build check-tianchenrv -- -v` invocation was invalid because this Ninja invocation parsed `-v` as a target.
* Real full-check attempt `cmake --build build --target check-tianchenrv -- -v` failed in the existing 12 RVV planning/lowering/script tests: `Plugin/plugin-emission-plan.test`, `Scripts/rvv-probe-to-mlir.test`, `Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`, `Transforms/EmissionReadiness/emission-readiness-rvv-builtin.mlir`, `Transforms/EmissionReadiness/emission-readiness.test`, `Transforms/EmissionReadiness/materialize-emission-plans-rvv-builtin.mlir`, `Transforms/ExecutionPlanCoherence/rvv-capacity-stale-boundary-fails.mlir`, `Transforms/LoweringBoundary/rvv-i32m1-policy-capability-fails.mlir`, `Transforms/LoweringBoundary/rvv-lowering-boundary-compat.mlir`, `Transforms/LoweringBoundary/rvv-lowering-boundary-malformed.mlir`, `Transforms/PluginVariantLegality/plugin-variant-legality-pass-invalid.mlir`, and `Transforms/PluginVariantLegality/plugin-variant-legality-pass.mlir`.
* `git diff --check` passed.
* Trellis context validation passed.
