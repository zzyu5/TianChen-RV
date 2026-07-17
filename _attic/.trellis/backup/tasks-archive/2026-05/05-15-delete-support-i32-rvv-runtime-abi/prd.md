# Delete support-layer I32 RVV runtime ABI residue

## Goal

Remove support-layer authority for hard-coded I32/RVV runtime ABI selection. Core Support should keep only extension-agnostic finite-binary ABI primitives. It must not publish selected `i32-vadd` / `i32-vsub` / `i32-vmul` runtime ABI contracts, RVV/scalar dispatch C ABI identities, `rvv_available` defaults, or direct-route-deleted invocation labels as reusable Support defaults.

## What I already know

- Current HEAD is `d7304f6 chore(plugin): delete source emission-plan authority`.
- Worktree was clean before this task was created.
- `.trellis/.current-task` was absent; this task was created from the Hermes direction brief and set current.
- `.trellis/spec/index.md` states TianChen-RV is a unified capability-driven RISC-V MLIR, not per-hardware backend dialects, and Python cannot implement compiler core.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` says direct RVV/scalar/dispatch C source exporters are deleted or fail-closed until a materialized MLIR EmitC route exists.
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md` says core/common passes must use plugin interfaces and must not branch on RVV/scalar family details.
- Current Support headers publish `I32BinaryRuntimeABIContract`, `getI32BinaryRuntimeABIContract`, I32-specific parameter/window/runtime-param helpers, I32 callable plan wrappers, and `rvv_available` defaults.
- Current Support implementation hard-codes `i32-vadd`, `i32-vsub`, `i32-vmul`, `rvv-scalar-dispatch-runtime-callable-c-abi`, and the three RVV/scalar dispatch runtime C function names.
- Current Support and Target tests assert the above as valid support-core behavior.

## Requirements

- Delete `I32BinaryRuntimeABIContract` and `getI32BinaryRuntimeABIContract` from the public Support API.
- Delete Support-layer I32 helper wrappers/defaults that select `i32-vadd`, selected I32 families, or `rvv_available`.
- Remove Support-owned RVV/scalar dispatch runtime C ABI identity strings and direct-route-deleted invocation labels from code and tests.
- Keep only extension-agnostic finite-binary ABI primitives in Support, such as explicit `FiniteBinaryRuntimeABIContract`, runtime ABI parameter roles, mem-window/runtime-param validation, and generic invocation-contract formatting.
- Rewrite directly affected Support and Target tests so they validate generic finite-binary ABI shape/binding behavior, not RVV/I32 selected-route identity ownership.
- Update relevant spec text that currently documents Support-owned I32 runtime ABI authority.

## Acceptance Criteria

- [ ] Support headers no longer expose `I32BinaryRuntimeABIContract` or `getI32BinaryRuntimeABIContract`.
- [ ] Focused Support/Target ref-scan finds no remaining production/test ownership of `getI32BinaryRuntimeABIContract`, `I32BinaryRuntimeABIContract`, `getI32BinaryRuntimeABIParameters`, `getI32BinaryDispatchRuntimeABIParameters`, `rvv-scalar-dispatch-runtime-callable-c-abi`, `rvv-direct-route-deleted`, or `rvv-scalar-i32-v*` dispatch function names.
- [ ] Remaining Support runtime ABI helpers are extension-agnostic primitives requiring explicit caller/plugin-owned contracts and explicit dispatch guard C names.
- [ ] Directly affected Support and Target tests no longer assert RVV scalar dispatch C function identities from Support.
- [ ] Focused build/test checks for changed Support/Target behavior pass, or any failures are reported as expected missing new-architecture gaps without restoring deleted logic.

## Out of Scope

- No replacement plugin ABI provider.
- No new EmitC route, RVV lowering, descriptor path, compatibility wrapper, quarantine, or legacy mode.
- No broad documentation cleanup outside the relevant spec correction.
- No RVV runtime, correctness, or performance claims.

## Technical Notes

- Main files inspected before implementation:
  - `include/TianChenRV/Support/RuntimeABIContract.h`
  - `include/TianChenRV/Support/RuntimeABI.h`
  - `include/TianChenRV/Support/RuntimeABICallablePlan.h`
  - `include/TianChenRV/Support/RuntimeABIParam.h`
  - `include/TianChenRV/Support/RuntimeABIMemWindow.h`
  - `lib/Support/RuntimeABIContract.cpp`
  - `lib/Support/RuntimeABICallablePlan.cpp`
  - `test/Support/RuntimeABICallablePlanTest.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`
- Minimal evidence should include focused ref-scans over `include/TianChenRV/Support`, `lib/Support`, `test/Support`, and directly affected Target tests, plus focused Support/Target build/test targets.
