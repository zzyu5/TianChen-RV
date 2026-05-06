# Scalar Fallback Plugin First Slice

## Goal

Add the first C++/MLIR scalar fallback extension plugin slice so TianChen-RV has a plugin-local fallback variant path in the same registry, legality, cost, selection, and emission-plan protocols already used by RVV.

## What I Already Know

- Repo root is `/home/kingdom/phdworks/TianchenRV`; current HEAD before this task is `a292055 feat: add RVV capability profile integration`.
- The latest supervisor audit/review input is under `artifacts/tmp/hermes_codex_supervisor/runs/20260506T144732Z-r0026-20260506T230209Z/`.
- The previous round completed RVV capability profile integration and left RVV lowering/runtime/executable emission explicitly unsupported.
- Specs require `tcrv.exec` to remain execution/capability/variant/dispatch/fallback metadata only.
- Extension-specific behavior must remain plugin-local and be reached through `ExtensionPluginRegistry`, not core `if RVV/scalar/...` branches.
- Fallback is required for system completeness, but fallback performance/runtime/correctness claims need real compiler/runtime evidence; this first slice should only claim compiler routing and metadata behavior.

## Requirements

- Add a plugin-local scalar fallback plugin implemented in C++.
- Declare stable scalar fallback plugin identity, version, capability id/kind, preferred capability symbol, and first-slice variant symbol.
- Make the plugin propose a `tcrv.exec.variant` only when a high-level MLIR op, enclosing `tcrv.exec.kernel`, and available scalar fallback capability are present.
- Verify scalar fallback variants through plugin-owned legality, including origin and required fallback capability.
- Provide plugin-owned cost metadata that makes scalar fallback a conservative coverage path, not a performance claim.
- Provide plugin-owned emission readiness and emission-plan metadata for a portable scalar fallback route without claiming object generation, linked runtime, hardware execution, correctness, or performance.
- Register the scalar fallback plugin in the built-in plugin registry alongside RVV.
- Keep core passes target-neutral; no extension-specific branches in core orchestration code.

## Acceptance Criteria

- [ ] C++ tests cover scalar plugin registration, proposal gating, materialization, legality, selection, emission readiness, and emission-plan fields.
- [ ] lit/FileCheck tests prove public `tcrv-opt` built-in registry can route scalar fallback selected paths through emission readiness and materialize supported fallback-plan diagnostics.
- [ ] CMake owns the new plugin library and test target under `check-tianchenrv`.
- [ ] Durable specs document the scalar fallback first-slice boundary and explicitly distinguish metadata support from executable correctness/performance evidence.
- [ ] Existing RVV capability/profile/readiness tests continue to pass and RVV first-slice emission remains unsupported.

## Definition Of Done

- Configure/build succeeds with local LLVM/MLIR packages, or missing toolchain diagnostics are reported exactly.
- `check-tianchenrv` passes.
- `git diff --check` passes.
- The task is finished/archived if Trellis tooling supports it, and the repo is left clean after one coherent commit.

## Out Of Scope

- No new high-level tensor/tile/compute op in `tcrv.exec`.
- No scalar dialect ops or real scalar lowering pass in this round.
- No RVV lowering, runtime ABI, generated executable, correctness, or performance claim.
- No IME, offload runtime, Sophgo, AME, or future hardware implementation.
- No Python implementation of compiler internals.

## Technical Approach

- Add `include/TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h` and `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`.
- Use the existing `ExtensionPlugin` virtual hooks: capabilities, dialect registration, variant proposal, legality, cost, emission readiness, and emission plan.
- Wire the plugin into `registerBuiltinExtensionPlugins` without changing core pass logic.
- Add C++ coverage in `test/Plugin/ScalarExtensionPluginTest.cpp` and lit coverage for the public tool path.
- Update `.trellis/spec/extension-plugins/` and lowering/testing specs to record the first-slice scalar fallback boundary.

## Technical Notes

- Relevant specs: `.trellis/spec/architecture/design-boundaries.md`, `.trellis/spec/capability-model/capability-contract.md`, `.trellis/spec/core-dialect/tcrv-exec-contract.md`, `.trellis/spec/extension-plugins/index.md`, `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, `.trellis/spec/testing/mlir-testing-contract.md`.
- Latest review noted remaining risks after `a292055`: no JSON-to-C++ ingestion tool, incomplete detailed RVV profiling, and RVV lowering/runtime/executable emission still intentionally unsupported.
