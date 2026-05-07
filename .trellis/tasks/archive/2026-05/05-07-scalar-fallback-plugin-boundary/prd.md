# Scalar Fallback Plugin Boundary

## Goal

Make the selected scalar fallback path explicit as plugin-owned MLIR lowering-boundary metadata, without adding scalar compute semantics to `tcrv.exec` or claiming executable scalar lowering.

## Requirements

- Add a minimal scalar plugin-owned MLIR surface, preferably a `tcrv_scalar` dialect if no scalar dialect exists.
- Define the scalar boundary op through TableGen/ODS and C++ verification.
- The boundary records selected variant reference, source kernel, origin/plugin metadata, selected-path role, metadata-only status, fallback reason, and capability requirement references.
- `ScalarExtensionPlugin::materializeSelectedLoweringBoundary` must materialize this scalar boundary op after plugin-local legality validation.
- Keep the generic selected lowering-boundary pass target-neutral and routed only through the existing plugin boundary API.
- Preserve fallback-preserving proposal behavior after recoverable RVV proposal decline.
- Keep RVV boundary behavior unchanged.
- Do not implement scalar executable lowering, LLVM lowering, runtime ABI, object generation, benchmarks, correctness validation, performance claims, or RVV hardware claims.
- Update stable specs only for durable scalar boundary contract and tests.

## Acceptance Criteria

- [x] ODS/lit coverage for scalar boundary parse/print.
- [x] Verifier rejects missing selected variant/ref or malformed required attrs.
- [x] Lit evidence shows scalar boundary is in scalar extension surface, not `tcrv.exec`.
- [x] C++ `ScalarExtensionPluginTest` covers scalar selected-boundary materialization, metadata preservation, deterministic malformed metadata diagnostics, and RVV decline plus scalar fallback materialization.
- [x] `--tcrv-execution-planning-pipeline` prints scalar boundary for scalar-only and RVV-decline fallback paths.
- [x] Valid RVV input continues to print RVV boundary.
- [x] Re-run/stale selected-surface diagnostics remain deterministic and do not duplicate scalar boundaries.
- [x] `git diff --check` passes.
- [x] CMake configure with LLVM/MLIR 20 passes.
- [x] `cmake --build build --target check-tianchenrv -j2` passes.

## Definition of Done

- Active C++/MLIR/ODS/lit or C++ test changes are committed in one coherent commit.
- Trellis task state is archived before final report.
- Worktree is clean after commit.

## Technical Approach

Add a small `include/lib/TianChenRV/Dialect/Scalar/IR` dialect mirroring the RVV dialect layout, but with only non-executable scalar fallback boundary metadata. Register it through `ScalarExtensionPlugin::registerDialects`, not through default core dialect registration. Update scalar plugin materialization to create `tcrv_scalar.lowering_boundary` and return a materialized plugin boundary result. Core transforms keep using the generic registry API unchanged.

## Out of Scope

- Scalar executable lowering, LLVM lowering, runtime ABI, object generation, runtime validation, performance work.
- RVV runtime/correctness/performance evidence.
- Core target-family branches or scalar/RVV-specific logic in generic transforms.
- New high-level compute dialect or compute op in `tcrv.exec`.

## Technical Notes

- Required inspection confirmed repo root `/home/kingdom/phdworks/TianchenRV`, HEAD `156e191`, and clean worktree before task creation.
- `lib/Transforms/EmissionPlan.cpp` does not exist; emission-plan materialization lives in `lib/Transforms/EmissionReadiness.cpp`.
- Existing `scalar-fallback-plugin.md` described scalar as a valid no-boundary route; this task intentionally replaces silent no-boundary absence with explicit plugin-owned scalar metadata boundary while keeping it non-executable.
