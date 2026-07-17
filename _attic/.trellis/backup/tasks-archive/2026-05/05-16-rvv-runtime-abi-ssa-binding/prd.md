# RVV runtime ABI SSA binding

## Goal

For the existing bounded RVV i32m1 add slice, make the callable runtime ABI
values `lhs`, `rhs`, `out`, and `n` explicit compiler IR values consumed by
the RVV extension-family IR and plugin-owned EmitC route provider. The route
must stop relying on synthetic AVL placeholders or buffer-role-only metadata
as artifact handoff authority.

## What I already know

* Current route provider is plugin-owned in
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
* Current first-slice materialization tests create `%n` with
  `builtin.unrealized_conversion_cast`.
* Current `tcrv_rvv.i32_load` and `tcrv_rvv.i32_store` carry only
  `buffer_role` metadata; the route provider maps that metadata to hard-coded
  `lhs`, `rhs`, and `out` C ABI names.
* Specs require extension-family ops -> common EmitC route -> target artifact,
  with RVV-specific naming isolated inside RVV plugin/target code.
* Common target artifact front doors must remain RVV-agnostic and must not
  infer route semantics from descriptors or target-side family branches.

## Requirements

* Introduce a bounded RVV-owned runtime ABI value binding surface for the
  current slice, without adding descriptor-driven computation or direct C
  source export.
* Make `tcrv_rvv.setvl` consume a real SSA value for `n` that is defined by an
  explicit runtime ABI binding op in the tested RVV route fixtures.
* Make `tcrv_rvv.i32_load` consume explicit `lhs` and `rhs` ABI value SSA
  operands rather than depending only on `buffer_role` metadata.
* Make `tcrv_rvv.i32_store` consume an explicit `out` ABI value SSA operand.
* Route provider must bind the callable C ABI parameters from these explicit
  IR values by role, type, ownership, and C name, then fail closed on missing,
  duplicate, malformed, or unsupported bindings.
* Preserve external C ABI names and order: `lhs`, `rhs`, `out`, `n`.
* Keep common target artifact export code free of RVV intrinsic/header names
  and RVV-specific semantic branches.

## Acceptance Criteria

* [x] RVV first-slice materialization and selected artifact fixtures no longer
      require `builtin.unrealized_conversion_cast` for AVL `n`.
* [x] `lhs`, `rhs`, `out`, and `n` provenance is visible in IR and consumed by
      the RVV route provider.
* [x] Missing or malformed runtime ABI bindings fail before EmitC/artifact
      export with RVV plugin/provider diagnostics.
* [x] Selected dispatch and selected-path sibling artifact tests still pass
      through the common RVV-agnostic target artifact front door.
* [x] Unsupported RVV shape and ambiguous/unselected selected artifact tests
      remain fail-closed.
* [x] Generated callable ABI names/order remain compatible with the existing
      bounded C ABI, or any change is explicitly documented and verified.

## Out of Scope

* New RVV dtype, LMUL, op family, or general RVV lowering.
* MLIR vector lowering, TensorExt/IME/offload executable routes, or high-level
  frontend lowering.
* Descriptor or binary-family registries, direct C compute printers, GCC
  default routes, Python compiler-core logic, or compatibility wrappers.
* RVV performance matrices or broadened runtime/correctness claims.

## Technical Notes

* Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Relevant code read:
  `include/TianChenRV/Support/RuntimeABI.h`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Conversion/EmitC/TCRVEmitCLowerableInterface.cpp`,
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`, and
  `lib/Target/TargetArtifactExport.cpp`.
* Focused evidence should include build targets plus lit/FileCheck for
  `test/Conversion/EmitC/rvv-first-slice-materialization*.mlir` and
  `test/Target/RVV/i32m1-*artifact*.mlir`.
