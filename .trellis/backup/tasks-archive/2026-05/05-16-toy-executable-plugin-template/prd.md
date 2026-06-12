# Toy executable plugin construction template

## Goal

Turn the existing Toy construction-protocol surface from a metadata-only
template into one bounded executable construction template: a selected Toy
variant must validate construction manifest metadata, materialize a Toy-owned
typed compute role boundary, build a plugin-owned `TCRVEmitCLowerableRoute`,
and materialize an EmitC module through the common EmitC materializer.

## What I already know

- The repo root is `/home/kingdom/phdworks/TianchenRV`.
- The starting worktree was clean at HEAD `dc3eecb`.
- No Trellis current task existed when this task began.
- The current Toy plugin already registers `toy-plugin`, capability
  `toy.template`, and proposal metadata for the construction protocol.
- `ToyConstructionProtocol` verifies the manifest and typed role graph, and
  `tcrv_toy.compute_skeleton` already implements
  `TCRVEmitCLowerableOpInterface`.
- `ToyExtensionPlugin` currently reports no active selected boundary and no
  active materialized EmitC route for the valid Toy template path.
- RVV has the nearest executable exemplar: typed family ops are validated,
  route mapping is plugin-owned, and the common EmitC materializer consumes a
  `TCRVEmitCLowerableRoute`.

## Requirements

- For the valid Toy first-slice path, Toy must no longer stop at the
  unsupported-emission placeholder before selected boundary and EmitC route
  materialization.
- Toy legality must continue to verify capability metadata, construction
  manifest metadata, typed role/interface realization, and evidence profile
  before readiness, planning, boundary validation, or route construction.
- The selected Toy path must materialize or validate a Toy-owned
  `tcrv_toy.compute_skeleton` boundary whose source op, source role,
  typed role, role order, and role-specific interface agree with the C++
  construction manifest and typed role realization.
- Toy must build a plugin-owned `TCRVEmitCLowerableRoute` from the typed
  Toy role op and route mapping, then prove that the common EmitC materializer
  emits a valid EmitC module for that route.
- Missing or stale construction metadata, wrong typed role/interface
  realization, missing selected boundary, and invalid route mapping must fail
  closed before supported readiness or materialization.
- Common/core orchestration must stay extension-neutral; do not add Toy/RVV
  semantic branches to core passes.

## Acceptance Criteria

- [x] Toy construction manifest and typed role realization tests cover the new
      active route mapping and fail-closed route mismatches.
- [x] Toy plugin C++ tests prove selected boundary materialization creates or
      validates `tcrv_toy.compute_skeleton` for the valid selected path.
- [x] Toy plugin C++ tests prove `buildVariantEmitCLowerableRoute` consumes
      the selected Toy role op and materializes an EmitC module through the
      common materializer.
- [x] lit/FileCheck coverage proves the valid Toy selected path reaches a
      materialized EmitC module, and negative cases cover missing selected
      boundary, wrong source role, wrong typed role/interface, stale route
      mapping, and unsupported non-template paths.
- [x] Focused checks for Toy plugin, common construction protocol, EmitC
      materialization, and touched lit tests pass, or failures are documented
      as bounded environment gaps.
- [x] A changed-surface scan shows descriptor/direct-C/source-export legacy
      paths were not restored and core orchestration did not gain Toy/RVV
      semantic branches.

## Out of Scope

- No new RVV dtype/LMUL/SEW/op family.
- No TensorExt, IME, offload, scalar, high-level tensor lowering, MLIR vector
  lowering, or target object compilation for Toy.
- No runtime, correctness, performance, or `ssh rvv` claim for Toy.
- No descriptor-driven computation, descriptor/binary-family registry,
  direct C source printer, source skeleton generation, Python compiler-core
  logic, GCC-default route, checkpoint/state-machine ledger, or compatibility
  wrapper.
- No common/core Toy/RVV semantic branch.

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Primary implementation files:
  `include/TianChenRV/Plugin/Toy/ToyConstructionProtocol.h`,
  `lib/Plugin/Toy/ToyConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/Toy/ToyExtensionPlugin.h`,
  `lib/Plugin/Toy/ToyExtensionPlugin.cpp`,
  `include/TianChenRV/Dialect/Toy/IR/ToyOps.td`,
  `lib/Dialect/Toy/IR/ToyDialect.cpp`.
- Common route files:
  `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h`,
  `lib/Conversion/EmitC/TCRVEmitCLowerableInterface.cpp`,
  `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h`,
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`,
  `lib/Transforms/EmitCLowerableMaterialization.cpp`.
- Focused tests:
  `test/Plugin/ToyExtensionPluginTest.cpp`,
  `test/Dialect/Toy/compute-skeleton.mlir`,
  new or updated `test/Conversion/EmitC/*toy*` lit coverage if needed,
  and `test/Plugin/ConstructionProtocolCommonTest.cpp` if route validation
  needs generic coverage.

## Definition of Done

- Implement the bounded Toy executable construction template.
- Update focused C++ and lit/FileCheck tests.
- Run focused build/tests; run broader `check-tianchenrv` if practical.
- Keep Trellis task status truthful, finish/archive if complete, and create
  one coherent commit.
