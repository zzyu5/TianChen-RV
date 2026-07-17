# minimal tcrv rvv dialect skeleton

## Goal

Add the first compiler-local RVV extension dialect slice and wire it through the existing RVV plugin dialect registration path. The slice must prove plugin-local dialect registration and parse/print support for a minimal non-compute RVV control-plane construct without adding RVV lowering, runtime ABI, executable kernels, benchmarks, or hardware claims.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- `HEAD` is `ce84969 feat: add RVV extension plugin first slice`.
- Current worktree was clean before this task was created.
- The previous RVV plugin slice intentionally left `RVVExtensionPlugin::registerDialects` as a no-op.
- `tcrv.exec` is implemented in MLIR/C++/TableGen/CMake and must remain execution/capability/variant/dispatch/fallback focused.
- The next owner is a minimal plugin-local RVV dialect skeleton, not lowering/emission/runtime/performance work.
- MLIR concrete dialect namespaces must not contain `.` characters; the architectural family remains `tcrv.rvv`, but the concrete MLIR namespace needs a dot-free spelling.

## Requirements

- Add a conventional C++/MLIR/TableGen RVV dialect package under `include/TianChenRV/Dialect/RVV/IR/` and `lib/Dialect/RVV/IR/`.
- Keep `registerAllDialects` core-only by default; do not add RVV to the default core dialect registration path.
- Implement `RVVExtensionPlugin::registerDialects` so the RVV plugin registers the RVV dialect through `ExtensionPluginRegistry` / `registerPluginDialects`.
- Provide at least one dialect-owned non-compute construct; this round uses a vector-length token type.
- Keep all RVV-specific behavior behind `ExtensionPluginRegistry` / `RVVExtensionPlugin`.
- Do not add RVV compute ops, load/store ops, lowering, emission, runtime ABI, benchmarking, or executable behavior.
- Update durable specs only for the real dialect surface and registration contract.
- Add C++ and/or lit coverage for positive registration/round-trip and negative missing/malformed/disabled-plugin cases.
- Run local CMake/lit checks and leave a clean committed worktree.

## Acceptance Criteria

- [ ] RVV plugin dialect registration makes the RVV dialect available through `registerPluginDialects`.
- [ ] The RVV dialect-owned vector-length type parses and prints round-trip when the registry is populated by the RVV plugin path.
- [ ] The same type is unavailable under default `registerAllDialects` alone.
- [ ] Disabled plugin registration does not make enabled-only plugin dialects available.
- [ ] Malformed RVV type syntax is rejected.
- [ ] Existing RVV proposal/materialization/legality/selection tests still pass.
- [ ] Existing `tcrv.exec` lit tests continue to parse/verify unchanged.
- [ ] `git diff --check`, CMake configure, and `check-tianchenrv` pass.

## Out Of Scope

- RVV arithmetic, load/store, fma, reductions, gather, compress, masks, setvl ops, or other compute operations.
- RVV lowering, code generation, runtime glue, ABI, executable kernels, hardware probes, correctness claims, or performance claims.
- Core target-family branches or RVV-specific branches in generic passes.
- Python implementations of compiler IR, dialects, registries, passes, capability model, variant pipeline, lowering, or emission.

## Technical Notes

- Required inspection commands and files were read before implementation.
- Relevant specs: architecture design boundaries, core dialect contract, RVV plugin, plugin registry/locality, implementation stack, testing, and shared plugin/compute/capability guides.
- Concrete namespace decision must be documented because MLIR rejects dotted dialect namespaces; this is an implementation compatibility detail, not a change to the architectural `tcrv.rvv` ownership boundary.
