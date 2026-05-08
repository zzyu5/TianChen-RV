# module target profile planning consumption

## Goal

Make explicit module-level `tcrv.exec.target` capability-provider profiles
usable by the public execution-planning path, not only by isolated verifier or
capability-model checks.

## What I Already Know

- The previous committed head is expected to be `f062bbd feat: support module
  target profile capabilities`.
- `tcrv.exec.kernel target = @profile` may bind exactly one direct module-level
  `tcrv.exec.target` into the kernel capability scope.
- Core consumers should use `TargetCapabilitySet` or an equivalent generic
  capability-provider scope rather than duplicating symbol rules.
- `tcrv.exec` must stay execution/capability/variant/dispatch/fallback focused
  and compute-free.
- No RVV runtime, correctness, or performance claim is required for this task.

## Requirements

- A module-level capability-provider target explicitly referenced by
  `kernel target = @profile` must be visible to the active public planning
  pipeline.
- Plugin proposal/materialization, legality, selection, selected lowering
  boundary materialization, emission-plan materialization, and coherence checks
  must continue to consume the generic `TargetCapabilitySet`.
- Module-level targets must remain explicit per-kernel bindings. Unreferenced
  module targets must not enter unrelated kernels.
- Parse-only, missing, shadowed, duplicate-id, or unavailable providers must
  continue to fail closed through existing generic diagnostics.
- The implementation must remain target-neutral and must not introduce
  hard-coded RVV/IME/Sophgo/AME/scalar/vendor/offload branches in core passes.

## Acceptance Criteria

- A lit/FileCheck test runs through the public
  `--tcrv-execution-planning-pipeline` route from a module-level
  `tcrv.exec.target @profile` and a `tcrv.exec.kernel target = @profile`.
- The test proves the module-level profile is consumed as the required provider
  anchor for a materialized selected variant and downstream planning metadata.
- Existing verifier, capability-model, plugin proposal/materialization,
  variant selection, dispatch, emission, coherence, target export, and script
  tests continue to pass.
- Required commands run:
  - `git diff --check`
  - `cmake -S . -B artifacts/tmp/tianchenrv-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
  - `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

## Out Of Scope

- New lowering, runtime ABI, object/header export semantics, ssh rvv execution,
  performance/correctness evidence, or new extension dialect operations.
- A new custom `requires` attribute form.
- Implicit collection of all module-level target profiles into every kernel.

## Technical Notes

- Initial inspection found the primary gap in
  `lib/Transforms/VariantMaterialization.cpp`: the public materialization pass
  still requires direct kernel-local capability-provider anchors before using
  `TargetCapabilitySet`.
- The existing public pipeline name is `--tcrv-execution-planning-pipeline`.
- The durable spec language already defines the target-profile scope, but pass
  descriptions that still mention only direct anchors may need minimal cleanup.
