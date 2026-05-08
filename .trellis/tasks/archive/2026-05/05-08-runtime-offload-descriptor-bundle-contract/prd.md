# runtime-offload descriptor bundle contract

## Goal

Strengthen the runtime-offload descriptor handoff from a metadata-only first slice into a bounded, versioned, machine-readable compiler artifact contract that can be consumed by an external runtime adapter. The selected offload path must continue to be modeled as a runtime-offload capability and descriptor handoff, not as a custom RISC-V ISA, vendor runtime implementation, hardware execution, correctness proof, or performance claim.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- HEAD before work is `0f8bd62 feat: add RVV bundle external ABI evidence bridge`.
- The worktree was clean before this task was created.
- The current offload first slice already proposes `offload-plugin`, materializes `tcrv_offload.lowering_boundary`, materializes a supported descriptor emission plan, and exports `runtime-offload-handoff-descriptor` through the generic target artifact route.
- Existing descriptor fields include source kernel, selected variant, role, origin plugin, route, emission kind, artifact kind, lowering boundary, runtime ABI kind/name, runtime glue role, handoff kind, required capabilities, and bounded non-claim evidence scope.
- Existing bundle export is registry-derived, but offload dispatch plus scalar fallback should produce a one-artifact offload descriptor bundle rather than also exporting the fallback single artifact when no composite route applies.

## Requirements

- Keep implementation in C++/MLIR/TableGen/CMake/lit; do not implement compiler semantics in Python.
- Strengthen the offload runtime descriptor with explicit schema/kind/status/adapter/non-claim handoff fields.
- Derive descriptor content from selected path, offload lowering boundary, and emission-plan metadata; do not use a disconnected hard-coded descriptor table.
- Preserve plugin locality: offload-specific descriptor validation remains in offload target/plugin code, while generic bundle routing remains registry-derived and target-neutral.
- Export selected offload descriptor artifacts through the generic target artifact bundle route as a deterministic one-artifact bundle for the selected non-fallback offload path.
- Bundle index metadata must expose stable artifact kind, route, owner/origin, runtime ABI, and handoff kind for descriptor handoff.
- Fail closed on missing runtime ABI, missing handoff kind, stale selected variant/plan/boundary metadata, unsafe URL/secret/raw-log text, and RVV/scalar spoofing of the offload descriptor route.
- Do not implement Sophgo, AME/IME, DMA, vendor runtime calls, accelerator kernels, runtime execution, hardware correctness, or performance.
- Do not broaden RVV/scalar runtime claims or rerun `ssh rvv`.

## Acceptance Criteria

- [ ] `tcrv-opt --tcrv-execution-planning-pipeline` on offload-capable input materializes selected offload path, lowering boundary, emission plan, and descriptor metadata.
- [ ] `tcrv-translate --tcrv-export-target-artifact` emits the versioned descriptor fields.
- [ ] `tcrv-translate --tcrv-export-target-artifact-bundle` emits a bundle index and descriptor file for the selected offload path.
- [ ] Positive FileCheck coverage proves descriptor schema/kind/status/adapter/non-claim fields and bundle index descriptor metadata.
- [ ] Negative coverage proves missing runtime ABI or handoff kind, stale metadata, unsafe URL/secret/raw-log text, and RVV/scalar spoofing fail before descriptor output.
- [ ] `git diff --check`, `cmake --build build --target tcrv-opt tcrv-translate -j2`, and `cmake --build build --target check-tianchenrv -j2` pass.

## Out Of Scope

- Vendor runtime integration, Sophgo-specific runtime calls, DMA, accelerator kernels, object generation for offload, hardware execution, correctness evidence, and performance evidence.
- New compute operations in `tcrv.exec`.
- New RVV runtime/correctness/performance claims or remote RVV evidence.

## Technical Notes

- Relevant specs: `.trellis/spec/core-dialect/tcrv-exec-contract.md`, `.trellis/spec/extension-plugins/offload-runtime-plugin.md`, `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, `.trellis/spec/plugin-protocol/locality-contract.md`, `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, `.trellis/spec/testing/mlir-testing-contract.md`.
- Relevant implementation: `lib/Plugin/Offload/OffloadExtensionPlugin.cpp`, `lib/Target/Offload/OffloadRuntimeDescriptor.cpp`, `lib/Target/TargetArtifactExport.cpp`, `lib/Transforms/ExecutionPlanCoherence.cpp`.
- Relevant tests: `test/Target/ArtifactExport/offload-runtime-descriptor-artifact-route.test`, `test/Target/TargetArtifactBundleExport/*`, and offload execution-planning tests.
