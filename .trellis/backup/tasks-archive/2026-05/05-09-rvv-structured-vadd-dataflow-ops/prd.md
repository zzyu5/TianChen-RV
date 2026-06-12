# rvv structured vadd dataflow ops

## Goal

Make the bounded RVV i32-vadd executable microkernel body more structurally real by replacing the single opaque `tcrv_rvv.i32_vadd_dataflow` marker in automatically materialized RVV microkernels with explicit plugin-local RVV load, add, and store dataflow operations that the exporter validates before emitting C.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current HEAD at task start is `f47215a` (`feat: support scalar offload relation providers`).
- Latest Hermes audit says the previous round completed scalar/offload relation-provider consumption and left the repo clean.
- The current RVV executable slice already materializes `tcrv_rvv.i32_vadd_microkernel` from the finite descriptor `i32-vadd-microkernel.v1`.
- The current microkernel body has `setvl -> with_vl -> i32_vadd_dataflow`; the marker carries ABI roles but does not expose plugin-local load/add/store structure.
- The architecture boundary requires computation and hardware execution details to remain in extension dialects such as `tcrv.rvv`, not in `tcrv.exec`.

## Requirements

- Add finite RVV dialect operations for the bounded i32-vadd body:
  - load lhs input buffer by ABI role,
  - load rhs input buffer by ABI role,
  - add the two loaded values,
  - store the result to output buffer by ABI role.
- Keep runtime `n`/AVL/VL as runtime SSA/control values through the existing microkernel block argument, `tcrv_rvv.setvl`, and `tcrv_rvv.with_vl`.
- Keep descriptor-local `element_count` as microkernel metadata only.
- Keep mem_window pointer ABI roles and runtime count role as `tcrv.exec` ABI boundary objects, not RVV operands or target capabilities.
- Update RVV plugin auto-materialization to emit the explicit load/add/store body.
- Update plugin and exporter validation to consume that structured body and fail closed for stale marker-only or malformed dataflow bodies.
- Preserve existing source/object/header artifact semantics and generated C ABI.
- Update durable specs and README only where behavior descriptions change.
- Add focused lit/FileCheck coverage for dialect syntax/materialization/export validation.

## Out Of Scope

- No generic high-level tensor, tile, or vector IR.
- No generic RVV lowering, arbitrary RVV op coverage, reductions, masks, layouts, or performance tuning.
- No Python implementation of dialects, operations, passes, capability model, plugin registry, or lowering/emission internals.
- No new RVV correctness/performance claim without real `ssh rvv` evidence.
- No core orchestration branch that hardcodes RVV dataflow semantics.

## Acceptance Criteria

- [ ] New RVV ops are declared in TableGen/ODS and verified in C++.
- [ ] Auto-materialized RVV i32-vadd microkernel contains explicit load/add/store body under `with_vl`.
- [ ] RVV exporter/plugin validation rejects marker-only or malformed dataflow bodies before source output.
- [ ] Existing RVV i32-vadd generated C ABI remains deterministic and source/header/object route behavior remains compatible.
- [ ] Relevant lit/FileCheck tests pass.
- [ ] `cmake --build build --target check-tianchenrv -j2` passes, or exact missing local toolchain diagnostics are reported.
- [ ] `git diff --check` passes.
- [ ] One coherent commit is created and the worktree is clean.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Likely implementation files:
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `lib/Dialect/RVV/IR/RVVDialect.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - RVV dialect/target/transform lit tests under `test/`.
