# fallback-preserving plugin proposal collection

## Goal

Make plugin proposal collection fallback-preserving: a plugin-local, recoverable inability to propose for the current target/kernel must not abort collection before later plugins can propose valid fallback variants.

## Requirements

- Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit or C++ tests.
- Preserve the target-neutral core transform boundary: no RVV, scalar, IME, Sophgo, AME, vendor, dtype, shape, runtime, or microarchitecture branching in generic transforms/support.
- Add a generic C++ representation for per-plugin collection outcomes if needed, covering valid proposals and recoverable decline diagnostics.
- Treat plugin-local missing, unavailable, or malformed proposal-generation evidence as recoverable declines when the plugin owns that evidence interpretation.
- Keep malformed/invalid plugin proposals, duplicate proposal symbols, unresolved or unavailable claimed requirements, malformed core IR, rerun mismatches, and typed materialization failures fatal.
- Fail public `--tcrv-materialize-plugin-variants` when no viable proposal is collected/materialized, with a useful generic diagnostic.
- Preserve deterministic ordering of proposals and diagnostics.
- Keep diagnostics safe and bounded; do not echo raw property values or secret-like text.
- Do not implement lowering, emission, runtime ABI, object generation, hardware execution, correctness validation, or performance measurement.

## Acceptance Criteria

- C++ registry tests cover recoverable decline followed by valid proposal, fatal invalid proposal, all-decline no-viable diagnostic, and deterministic diagnostic ordering.
- lit/FileCheck tests cover malformed/missing RVV evidence plus scalar fallback materialization, malformed/missing RVV-only no-viable failure, positive built-in materialization order, empty registry failure, and rerun mismatch failure.
- RVV plugin tests reflect recoverable proposal-generation declines while keeping explicit RVV legality errors fatal.
- `git diff --check`, CMake configure, and `check-tianchenrv -j2` pass.
- Completed Trellis task is archived before final report.

## Out of Scope

- New high-level frontend conversion or compute dialect design.
- Generic compute operations in `tcrv.exec`.
- RVV lowering/intrinsics/runtime/correctness/performance.
- Python implementation of compiler decisions or plugin registry behavior.

## Technical Notes

- Main interfaces: `include/TianChenRV/Plugin/ExtensionPlugin.h`, `lib/Plugin/ExtensionPlugin.cpp`.
- Main pass path: `lib/Transforms/VariantMaterialization.cpp`.
- RVV proposal evidence stays plugin-local in `lib/Plugin/RVV/RVVExtensionPlugin.cpp`.
- Scalar fallback semantics stay plugin-local in `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`.
