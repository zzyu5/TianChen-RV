# RVV i32m1 arithmetic EmitC route family

## Goal

Generalize the existing plugin-owned RVV i32m1 add EmitC route into a bounded
arithmetic route family for the already-modeled `tcrv_rvv.i32_add`,
`tcrv_rvv.i32_sub`, and `tcrv_rvv.i32_mul` typed ops. The production path must
remain selected RVV variant -> explicit ABI SSA values -> plugin-owned
arithmetic route -> common EmitC artifact front door -> target object/header
artifact -> `ssh rvv` correctness evidence.

## What I already know

* Current HEAD is `514da0e target(rvv): bind i32m1 runtime ABI values in IR`.
* The working tree was clean before this task started.
* No `.trellis/.current-task` existed, so this task was created from the Hermes
  direction brief.
* `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, and `tcrv_rvv.i32_mul` already exist
  as typed RVV dialect ops and implement the EmitC lowerable op interface.
* The current provider in `RVVEmitCRouteProvider.cpp` recognizes only the
  load-add-store slice and hard-codes `__riscv_vadd_vv_i32m1`.
* The current target RVV support bundle, translate routes, artifact route ids,
  header comments, component group, and C++ tests are add-specific.
* Common target artifact export already has a selected EmitC artifact front
  door; this task should reuse it rather than adding RVV semantic branches to
  common target/export orchestration.

## Requirements

* Support exactly the bounded SEW32 / LMUL m1 / tail-agnostic / mask-agnostic
  i32m1 arithmetic scope.
* Preserve the explicit runtime ABI SSA contract for `lhs`, `rhs`, `out`, and
  `n`; do not derive ABI values from descriptor metadata.
* Select exactly one supported arithmetic op in the load-op-store slice.
* Map each supported op to the correct RVV intrinsic:
  `i32_add -> __riscv_vadd_vv_i32m1`,
  `i32_sub -> __riscv_vsub_vv_i32m1`,
  `i32_mul -> __riscv_vmul_vv_i32m1`.
* Generate operation-specific route ids, runtime ABI names, component groups,
  header routes, object routes, and translate routes for selected artifact
  export.
* Keep common target/export code generic; operation-specific intrinsic and
  route naming belong to the RVV plugin/target support bundle.
* Fail closed for unsupported, mixed, ambiguous, missing, or multiple
  arithmetic bodies.
* Preserve add behavior while adding selected artifact coverage for sub and
  mul.

## Acceptance Criteria

* Add still materializes through the existing explicit ABI SSA path.
* Sub and mul materialize EmitC `call_opaque` steps with the correct RVV
  intrinsic names.
* Missing ABI, missing store, unsupported shape, mixed arithmetic, multiple
  arithmetic, and ambiguous selected artifact cases fail closed.
* Selected dispatch and selected-path artifact export work for sub and mul via
  the common front door without RVV intrinsic/header names in common target
  code.
* Generated object/header ABI remains `void fn(const int32_t *lhs,
  const int32_t *rhs, int32_t *out, size_t n)`, with only route/function naming
  identifying the operation.
* Focused lit/FileCheck and target artifact C++ tests pass.
* Real `ssh rvv` link/run harnesses pass for every newly supported arithmetic
  op and add is rerun because the shared route builder changes.
* A changed-surface scan shows descriptor/direct-C/source-export legacy terms
  were not restored and common target code still contains no RVV intrinsic or
  RVV header names.

## Out of Scope

* New SEW, LMUL, dtype, mask, tail-policy, i32m2 executable route, vector
  lowering, high-level tensor lowering, TensorExt/IME implementation, or
  performance claims.
* Descriptor or binary-family registries, direct C compute printers, Python
  compiler-core logic, GCC-default routes, compatibility wrappers, or
  extension-specific semantic branches in common/core orchestration.
* Rebuilding high-level frontend lowering or adding new extension families.

## Technical Notes

* Relevant specs: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Relevant implementation files:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`, and
  `lib/Target/TargetArtifactExport.cpp`.
* Main tests to update or add:
  `test/Conversion/EmitC/rvv-first-slice-materialization.mlir`,
  `test/Conversion/EmitC/rvv-first-slice-materialization-missing-abi.mlir`,
  `test/Conversion/EmitC/rvv-first-slice-materialization-missing-store.mlir`,
  `test/Conversion/EmitC/rvv-first-slice-materialization-negative.mlir`,
  `test/Target/RVV/i32m1-add-object-artifact.mlir`,
  `test/Target/RVV/i32m1-selected-dispatch-artifact.mlir`,
  `test/Target/RVV/i32m1-selected-path-sibling-artifact.mlir`,
  `test/Target/RVV/i32m1-object-unsupported-shape.mlir`, and
  `test/Target/TargetArtifactExportTest.cpp`.
