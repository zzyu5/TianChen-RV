# rvv runtime abi role dataflow export

## Goal

Replace the bounded RVV i32-vadd dataflow/export stringly C-name boundary with a finite role-based binding. The RVV dataflow body should name runtime ABI roles, and the target-owned exporter should resolve concrete C parameter names from structured runtime ABI parameter metadata when that metadata is present.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial precondition passed: worktree was clean, HEAD was `0db226b feat: consume RVV i32 vadd dataflow body`, and the three supervisor-policy files were clean.
* The current `tcrv_rvv.i32_vadd_dataflow` op carries fixed C-name attributes `lhs`, `rhs`, `out`, and `runtime_n`.
* `RVVExtensionPlugin` materializes those fixed names and validates the same fixed strings before reporting a supported microkernel emission plan.
* `RVVMicrokernel.cpp` validates fixed dataflow C names and emits the generated C function using default `lhs/rhs/out/n` parameters.
* Structured runtime ABI metadata already exists in `support::RuntimeABIParameter`, with roles, C names, C type spellings, and ownership.

## Requirements

* Keep the implementation in C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck.
* Keep `tcrv.exec` compute-free; computation stays in bounded RVV extension-dialect dataflow and target-owned RVV exporter code.
* Change `tcrv_rvv.i32_vadd_dataflow` to record role references rather than fixed C parameter names.
* Materialize plugin-owned RVV i32-vadd microkernel bodies with role-based dataflow metadata.
* Validate the finite dataflow roles fail-closed for missing, unknown, duplicated, or mismatched role metadata.
* Resolve RVV generated C parameter declarations and uses from structured runtime ABI parameter metadata by role when a matching supported emission plan is present.
* Allow an alternate C ABI name fixture such as `a/b/dst/len` to emit correct RVV C by structural role lookup.
* Keep pointer buffers target/export ABI-owned; do not model them as SSA buffers, memrefs, or generic memory operations.
* Preserve runtime n/AVL/VL layering through the microkernel body argument, `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, and ABI metadata; keep `element_count` descriptor-local.
* Keep RVV-specific role/dataflow/emission logic plugin-local or target-owned.

## Acceptance Criteria

* [x] Plugin materialization creates role-based `tcrv_rvv.i32_vadd_dataflow` metadata under `setvl` / `with_vl`.
* [x] RVV exporter emits default generated C unchanged for the no-emission-plan direct route.
* [x] RVV exporter resolves alternate C parameter names from runtime ABI metadata by role when a supported emission-plan diagnostic is present.
* [x] Missing, duplicate, unknown, wrong-ownership, and wrong-type runtime ABI role metadata fail before source output.
* [x] RVV+scalar dispatch source/object fixtures still pass with the default ABI route.
* [x] Local validation includes `git diff --check` and `cmake --build build --target check-tianchenrv -j2`.

## Out of Scope

* No Python implementation of compiler/runtime behavior.
* No generic RVV lowering, memory model, scheduler, runtime scheduler, profile lattice, or conflict solver.
* No new RVV runtime/correctness/performance claim unless an explicit bounded `ssh rvv` run is executed and recorded separately.
* No supervisor-policy edits because the initial policy-file dirty precondition did not trigger.

## Technical Notes

* Relevant specs: `.trellis/spec/capability-model/capability-contract.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, `.trellis/spec/testing/mlir-testing-contract.md`.
* Main code surfaces: `include/TianChenRV/Support/RuntimeABI.h`, `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`, `lib/Dialect/RVV/IR/RVVDialect.cpp`, `lib/Plugin/RVV/RVVExtensionPlugin.cpp`, `lib/Target/RVV/RVVMicrokernel.cpp`, and generic target artifact route validation.
