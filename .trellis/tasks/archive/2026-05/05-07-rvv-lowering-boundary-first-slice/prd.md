# RVV Lowering Boundary First Slice

## Goal

Add the first plugin-owned RVV extension-dialect lowering boundary for selected RVV execution variants. The slice must expose a minimal `tcrv_rvv` TableGen/ODS operation and a C++/MLIR pass that materializes boundary metadata for selected RVV dispatch paths, without implementing executable RVV lowering, runtime ABI, correctness, or performance evidence.

## What I Already Know

- HEAD is `8d4f213 feat: synthesize conservative dispatch fallbacks`.
- Current repo state is clean before this task.
- Existing `tcrv.exec` core remains compute-free and owns kernel/capability/variant/dispatch/fallback/diagnostic structure.
- Existing RVV dialect concrete namespace is `tcrv_rvv`, not textual `tcrv.rvv`.
- Existing RVV first slice provides `!tcrv_rvv.vl` and `#tcrv_rvv.policy`.
- Existing RVV plugin reports emission readiness and emission plans as unsupported.
- Existing scalar fallback plugin is explicit-capability-gated and metadata-only.

## Requirements

- Add a minimal RVV extension-dialect ODS op representing a selected RVV lowering boundary or lowering plan.
- The op may carry source kernel symbol, selected variant symbol, dispatch role, capability/profile summary, lowering status, and unsupported reason.
- The op must not encode arithmetic, tensor, RVV intrinsic, memory, runtime ABI, object generation, correctness, or performance semantics.
- Register the new RVV op in C++ dialect initialization and TableGen/CMake generation.
- Add a plugin-aware C++ pass/API that consumes already selected/materialized `tcrv.exec.variant` and `tcrv.exec.dispatch` state.
- The pass must identify RVV-owned selected dispatch-case/direct variants through generic origin metadata and the plugin registry, then materialize RVV boundary ops at stable kernel locations.
- Scalar fallback dispatch targets must remain `tcrv.exec.fallback` metadata-only and must not receive RVV lowering-boundary ops.
- RVV unsupported emission-readiness and emission-plan diagnostics must remain unsupported.
- Add lit/FileCheck and C++ tests for RVV boundary parsing/verification/materialization and fallback preservation.
- Do not use Python for compiler internals.
- Do not run `ssh rvv` or claim runtime/correctness/performance evidence.

## Acceptance Criteria

- [ ] `tcrv_rvv` parses/prints/verifies the new boundary op.
- [ ] Malformed boundary ops reject empty source/variant/role/status or executable-claim wording with stable diagnostics.
- [ ] Selected RVV plus scalar fallback materializes an RVV boundary for the selected dispatch case and preserves scalar fallback.
- [ ] Scalar-only fallback materializes no RVV boundary.
- [ ] RVV-only selected variant can materialize an RVV boundary while preserving existing missing-fallback diagnostics.
- [ ] RVV emission readiness/plan remains unsupported and scalar fallback remains metadata-only.
- [ ] `git diff --check`, CMake configure, `check-tianchenrv`, and focused tests pass or failures are reported precisely.
- [ ] Task state is validated and archived before final commit.

## Out Of Scope

- Actual RVV intrinsics, LLVM/RISC-V lowering, object generation, runtime ABI, hardware execution, correctness proof, or benchmarking.
- Any new RVV/scalar/IME/offload target-family branch in generic core support, selection, or dispatch code.
- Any compute op or computation semantics inside `tcrv.exec`.
- Any Python implementation of compiler IR, dialects, passes, plugins, capability model, lowering, or emission.

## Technical Notes

- Required inspection completed before editing: repo status/log/HEAD, AGENTS, README, root CMake, Trellis spec index, core/plugin/lowering/testing specs, RVV dialect/plugin, scalar plugin, variant selection/dispatch/emission code, tcrv-opt, and relevant tests.
- User-provided paths `include/TianChenRV/Dialect/RVV/IR/RVVAttrs.td`, `include/TianChenRV/Dialect/RVV/IR/RVVTypes.td`, and `test/Dialect/RVV/verify.mlir` do not exist in current HEAD; the actual RVV ODS file is `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` and current RVV tests live under `test/Dialect/RVV/`.
