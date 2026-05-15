# RVV Lowering-Boundary Compatibility Wrapper Erasure

## Goal

Erase the public RVV-specific lowering-boundary compatibility wrapper as an
active compiler route. The canonical selected lowering-boundary materialization
route must remain the generic
`--tcrv-materialize-selected-lowering-boundaries` pass through
`ExtensionPluginRegistry`, with RVV-specific boundary semantics owned only by
the RVV plugin implementation.

## What I Already Know

- The previous committed round is `f63a210 chore(rvv): erase deferred runtime
  abi fixtures`.
- Repository state at task creation was clean.
- Current active wrapper authority exists in:
  - `include/TianChenRV/Transforms/Passes.td`
    `MaterializeRVVLoweringBoundary`
  - `include/TianChenRV/Plugin/RVV/RVVLoweringBoundary.h`
  - `lib/Plugin/RVV/RVVLoweringBoundary.cpp`
  - `tools/tcrv-opt/tcrv-opt.cpp` public registration for
    `--tcrv-materialize-rvv-lowering-boundary`
  - `test/Transforms/LoweringBoundary/rvv-lowering-boundary-compat.mlir`
  - `test/Transforms/LoweringBoundary/RVVLoweringBoundaryTest.cpp` wrapper
    equivalence coverage
- `.trellis/spec/extension-plugins/rvv-plugin.md` and
  `.trellis/spec/testing/mlir-testing-contract.md` still contain old
  compatibility-wrapper wording that must be aligned with this deletion round.
- This round is part of the Wrong Logic Deletion Campaign. It is deletion /
  refactor only.

## Module Goal

Remove active public API, pass registration, tool registration, tests, and spec
language that preserve the RVV-specific lowering-boundary wrapper/front door:

- `--tcrv-materialize-rvv-lowering-boundary`
- `MaterializeRVVLoweringBoundary`
- `createMaterializeRVVLoweringBoundaryPass`
- `materializeRVVLoweringBoundaries`
- `rvv-lowering-boundary-compat` compatibility fixture/contract

## Requirements

- Active source and tests must no longer define or register the RVV-specific
  lowering-boundary wrapper pass/front door.
- Active C++ APIs must no longer expose RVV-specific wrapper helper functions
  that forward to generic selected-boundary materialization.
- The canonical `--tcrv-materialize-selected-lowering-boundaries` path remains
  the selected-boundary materialization route.
- The RVV plugin's real `materializeSelectedLoweringBoundary` and
  `validateSelectedLoweringBoundary` hook implementations must remain intact.
- Wrapper-only lit/C++ compatibility tests must be deleted or rewritten so they
  no longer prove wrapper equivalence.
- Specs must describe the generic public route and RVV plugin hook without
  preserving the old wrapper compatibility contract.
- Do not add a replacement wrapper, pass alias, compatibility route, new
  lowering route, descriptor route, direct C exporter, executable RVV lowering,
  runtime ABI implementation, Common EmitC implementation, helper-only route,
  plugin template, or `ssh rvv` evidence as the main result.

## Acceptance Criteria

- [ ] Focused active-surface ref scan, excluding `artifacts/tmp`,
  `.trellis/tasks/archive`, `.trellis/workspace`, `.git`, `build`, and run
  artifacts, finds no active authoritative occurrences of:
  - `--tcrv-materialize-rvv-lowering-boundary`
  - `MaterializeRVVLoweringBoundary`
  - `createMaterializeRVVLoweringBoundaryPass`
  - `materializeRVVLoweringBoundaries`
  - `rvv-lowering-boundary-compat`
- [ ] Any remaining hits are absent or explicitly justified as non-authoritative
  governance.
- [ ] `ninja -C build tcrv-opt` is attempted and passes or reports an unrelated
  baseline/build gap.
- [ ] Affected LoweringBoundary C++ or lit coverage is built/run through the
  surviving generic route.
- [ ] CMake/test registration is cleaned up if wrapper-only files or tests are
  removed; no dummy target is kept solely to preserve the old wrapper.
- [ ] `check-tianchenrv` is attempted or its status is reported without
  restoring the RVV wrapper.
- [ ] `git diff --check` passes.
- [ ] Trellis task validates, is finished/archived when complete, and the round
  is committed as one coherent commit.

## Out Of Scope

- RVV executable lowering.
- Common EmitC implementation.
- Runtime ABI implementation.
- New pass, pass alias, wrapper, compatibility path, descriptor route, direct C
  exporter, source/header/object route, plugin template, or replacement route.
- Unrelated RVV selected-shape legality, probe-replay, runtime, correctness, or
  performance fixes unless they are direct references to the wrapper/front door.
- `ssh rvv` runtime/correctness/performance evidence.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/plugin-protocol/index.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/extension-plugins/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/testing/index.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Current generic route:
  - `include/TianChenRV/Transforms/Passes.td`
    `MaterializeSelectedLoweringBoundaries`
  - `lib/Transforms/LoweringBoundary.cpp`
  - `lib/Plugin/LoweringBoundary.cpp`
  - `RVVExtensionPlugin::materializeSelectedLoweringBoundary`
- Initial focused scan found wrapper authority in the source/test files listed
  above plus compatibility wording in active specs.

## Completion Notes

- Deleted the RVV-specific public pass definition and `tcrv-opt` registration
  for `--tcrv-materialize-rvv-lowering-boundary`.
- Deleted the RVV wrapper API/header and forwarding implementation:
  `RVVLoweringBoundary.h`, `RVVLoweringBoundary.cpp`,
  `materializeRVVLoweringBoundaries`, and
  `createMaterializeRVVLoweringBoundaryPass`.
- Deleted the wrapper lit fixture
  `test/Transforms/LoweringBoundary/rvv-lowering-boundary-compat.mlir`.
- Removed the C++ wrapper-equivalence test while retaining generic
  selected-boundary and RVV plugin-hook coverage.
- Updated active specs to state that the public selected-boundary route is
  only `--tcrv-materialize-selected-lowering-boundaries` through
  `ExtensionPluginRegistry`; RVV-specific interpretation remains in the RVV
  plugin hook.
- Added no replacement wrapper, alias, compatibility path, descriptor route,
  direct C exporter, runtime ABI, Common EmitC route, or RVV executable
  lowering.
- Focused wrapper ref scan found no active source/spec/test hits outside this
  task's non-authoritative Trellis PRD/task metadata.
- Focused build and LoweringBoundary checks passed. Full `check-tianchenrv`
  was attempted and still fails in 7 existing baseline tests outside this
  owner: `Plugin/plugin-emission-plan.test`,
  `Scripts/rvv-probe-to-mlir.test`,
  `Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`,
  `Transforms/LoweringBoundary/rvv-i32m1-policy-capability-fails.mlir`,
  `Transforms/LoweringBoundary/rvv-lowering-boundary-malformed.mlir`,
  `Transforms/PluginVariantLegality/plugin-variant-legality-pass-invalid.mlir`,
  and `Transforms/PluginVariantLegality/plugin-variant-legality-pass.mlir`.
