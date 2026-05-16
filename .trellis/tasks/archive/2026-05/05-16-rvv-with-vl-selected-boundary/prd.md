# RVV with_vl selected lowering boundary

## Goal

Make the RVV plugin selected-lowering-boundary interface recognize and
validate the existing `tcrv_rvv.with_vl` region as the selected boundary for
the bounded explicit typed i32m1 add/sub/mul arithmetic family.

The intended compiler path for this round is:

```text
selected RVV variant
  -> validated tcrv_rvv.with_vl lowering boundary
  -> plugin-owned i32m1 arithmetic EmitC route
  -> common EmitC artifact front door
  -> object/header artifact
```

## What I Already Know

- The current task brief targets one coherent RVV selected lowering-boundary
  submodule, not a new RVV family or new high-level lowering path.
- Commit `ba7fe66` already centralized the i32m1 SEW32 / LMUL m1 / agnostic
  policy / runtime AVL-to-VL contract in `RVVConfigContract`.
- `RVVEmitCRouteProvider` already consumes that contract and supports bounded
  i32m1 add/sub/mul route construction from explicit typed RVV IR.
- `RVVExtensionPlugin::materializeSelectedLoweringBoundary` currently returns
  no-boundary for legal explicit typed RVV variants.
- `RVVExtensionPlugin::validateSelectedLoweringBoundary` currently rejects all
  validation requests unconditionally.
- Existing target artifact tests already use emission-plan metadata
  `lowering_boundary = "tcrv_rvv.with_vl"` for i32m1 add/sub/mul.
- The relevant specs require plugin-owned RVV semantics, a common EmitC route,
  MLIR/lit/FileCheck coverage, and no descriptor/direct-C/source-export
  revival.

## Design Boundaries

- `materializeSelectedLoweringBoundary` must not synthesize computation or
  create a compatibility wrapper.
- The recognized boundary is the existing explicit typed `tcrv_rvv.with_vl`
  operation in the selected variant body.
- `validateSelectedLoweringBoundary` must fail closed through RVV-owned
  diagnostics for missing, duplicate, mismatched config/VL, stale route/op, and
  unsupported selected boundary shapes.
- Common/core orchestration must stay extension-agnostic. Any common interface
  repair must be generic and must not add RVV intrinsic/header names or
  RVV-specific semantic branches.
- Descriptor-driven computation, direct handwritten C semantic export, Python
  compiler-core logic, GCC-default routing, and compatibility wrappers are out
  of scope.

## Requirements

- Valid selected add/sub/mul variants with exactly one supported
  `tcrv_rvv.with_vl` boundary no longer receive no-boundary materialization or
  unconditional validation failure.
- The selected boundary must be validated with the shared
  `validateRVVI32M1ArithmeticConfigVLContract` contract.
- Boundary validation must confirm the route family remains one of the
  supported i32m1 arithmetic routes already provided by the RVV EmitC route
  provider.
- Stale route/op mismatches must continue to fail before artifact export.
- Missing or multiple `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` surfaces must fail.
- Unsupported boundary operation shapes must fail and must not be silently
  accepted as generic RVV.
- The selected emission plan and target artifact export must continue to
  consume the plugin-owned EmitC route and common artifact front door.

## Acceptance Criteria

- [ ] `--tcrv-materialize-selected-lowering-boundaries` recognizes selected
      i32m1 add/sub/mul `tcrv_rvv.with_vl` boundaries.
- [ ] `--tcrv-check-emission-readiness`,
      `--tcrv-materialize-emission-plans`, or execution-planning pipeline
      coverage validates the selected `with_vl` boundary instead of reporting
      missing/no boundary for valid add/sub/mul cases.
- [ ] Negative tests cover missing/multiple/mismatched config/VL selected
      boundary shapes and stale route/op mismatch behavior.
- [ ] Existing selected artifact export for add/sub/mul remains supported.
- [ ] A changed-surface scan shows descriptor/direct-C/source-export legacy
      terms were not restored and common/core code does not gain RVV intrinsic
      or RVV header names.
- [ ] Focused build/lit checks pass for the touched RVV plugin, dialect,
      lowering-boundary/emission planning, and target artifact surfaces.

## Definition Of Done

- Trellis task context is truthful and current.
- Implementation is in C++ / MLIR / TableGen / CMake / lit / FileCheck only.
- Focused validation has been run and failures self-repaired where practical.
- `ssh rvv` is run only if generated C/header/object output changes; otherwise
  final evidence includes a clear unchanged-artifact rationale.
- The task is finished/archived if complete.
- One coherent commit is created when complete.

## Out Of Scope

- New RVV dtype, LMUL, SEW, op families, i32m2 execution, generic RVV lowering,
  MLIR vector lowering, high-level tensor lowering, TensorExt/IME
  implementation, performance matrices, descriptor or binary-family registries,
  direct C compute printers, Python compiler-core logic, GCC-default routes,
  compatibility wrappers, and extension-specific semantic branches in common
  orchestration.

## Technical Notes

- Read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Key files:
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `include/TianChenRV/Conversion/EmitC/`,
  `lib/Conversion/EmitC/`,
  `test/Target/RVV/i32m1-selected-dispatch-artifact.mlir`,
  `test/Target/RVV/i32m1-sub-selected-dispatch-artifact.mlir`,
  `test/Target/RVV/i32m1-mul-selected-dispatch-artifact.mlir`,
  `test/Target/RVV/i32m1-object-stale-route-op.mlir`.
- There is a spec-history conflict: the older "Deleted Selected Lowering
  Boundary Route" section describes RVV no-boundary behavior, while the current
  task brief requires promoting the existing `tcrv_rvv.with_vl` boundary for
  the already rebuilt i32m1 EmitC object/header slice. This task treats the
  user-provided Direction Brief as the current round authority and should
  update spec text if the implementation lands.
