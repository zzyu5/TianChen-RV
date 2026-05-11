# Make i32 binary callable ABI parameters production-owned by exec IR

## Goal

Make the bounded i32 RVV and scalar binary callable source/export production path derive callable ABI parameters from direct `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` boundaries. Descriptor and selected-route metadata may continue to identify family, route, ABI identity, and cross-check facts, but must not supply an independent default callable parameter list when exec IR is missing, duplicate, malformed, or stale.

## What I Already Know

- The previous completed head is `0574d4f feat(scalar): render source from emitc route authority`.
- `.trellis/.current-task` did not exist at task start, so this task was created as `05-12-i32-binary-ir-backed-callable-abi-production-authority`.
- `support::buildI32BinaryCallableABIPlan` already exists and validates direct `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` boundaries for the bounded i32 binary ABI.
- `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp` currently has `buildScalarEmissionRuntimeABIParameters`, which starts from descriptor/default ABI parameters and consumes IR collection errors before using descriptor defaults.
- `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp` and `lib/Target/RVV/RVVMicrokernel.cpp` still contain route-local RVV binary ABI parameter builders; the i32 production path should consume the shared i32 binary support-layer ABI plan.
- `lib/Target/Scalar/ScalarMicrokernel.cpp` already validates scalar source export against an IR-backed callable plan, so scalar source/header/object export is closer to the target state than scalar emission-plan construction.

## Requirements

- For supported i32 RVV direct callable routes, emission-plan and export-time runtime ABI parameter lists must be built from the shared IR-backed i32 binary callable ABI plan.
- For supported i32 scalar fallback callable routes, production emission planning must fail closed when the direct exec IR ABI boundary is missing, duplicate, malformed, or stale. It must not silently replace the boundary with descriptor/default ABI parameters.
- Supported emission-plan `runtime_abi_parameters` metadata must mirror the IR-backed callable ABI plan and keep using existing typed mirror validation before source/header/object/bundle export.
- Runtime `n` C name may be non-default when provided by `tcrv.exec.runtime_param`, but missing or empty runtime parameter C name/type/ownership metadata must fail before source export.
- Preserve the current EmitC body authority: typed extension family ops define compute and the common EmitC route renders source bodies.
- Keep descriptor fields as selected family/config/ABI identity/cross-check data only.

## Non-Goals

- Do not add a descriptor-to-C or descriptor-to-ABI path.
- Do not broaden the required deliverable to i64 unless needed to keep shared code compiling without changing i64 authority.
- Do not add RVV runtime, correctness, or performance claims.
- Do not implement compiler internals in Python.
- Do not add compute semantics to `tcrv.exec` or family-specific semantic branches in common passes.

## Acceptance Criteria

- [x] Shared ABI-plan tests cover valid i32 mem_window/runtime_param IR and fail-closed cases for missing, duplicate, malformed, or stale ABI roles and runtime param metadata.
- [x] RVV i32 direct callable emission planning uses `support::buildI32BinaryCallableABIPlan` or an equivalent shared support-layer API, and route-local i32 descriptor/default ABI construction is no longer the production authority.
- [x] Scalar i32 fallback emission planning returns errors for missing/stale exec IR ABI boundaries instead of emitting descriptor/default `runtime_abi_parameters`.
- [x] Focused lit/FileCheck coverage proves at least one RVV i32 route and one scalar i32 route still emit common EmitC route provenance while ABI parameters come from exec IR boundaries.
- [x] Focused target/export or bundle coverage is updated if candidate preflight behavior changes.
- [x] `git diff --check` passes.
- [x] Focused C++/lit tests pass.
- [x] `check-tianchenrv` is run if the configured build directory exists or can be configured with local LLVM/MLIR paths.

## Technical Notes

- Relevant specs read for this task:
  - `.trellis/spec/index.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant source entry points:
  - `include/TianChenRV/Support/RuntimeABICallablePlan.h`
  - `lib/Support/RuntimeABICallablePlan.cpp`
  - `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
  - `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`

## Task Status Notes

- Default i32 RVV direct callable and scalar fallback callable production paths are rewired to consume the shared IR-backed callable ABI plan for runtime ABI parameters.
- Descriptor data remains selected family/config/ABI identity and cross-check data only; it no longer supplies i32 production callable parameter truth for the rewired paths.
- Focused C++ support coverage, focused lit/FileCheck coverage, `git diff --check`, and full `check-tianchenrv` passed on 2026-05-12.
