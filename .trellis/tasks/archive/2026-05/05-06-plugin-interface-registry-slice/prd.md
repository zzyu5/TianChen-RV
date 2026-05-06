# C++ Extension Plugin Interface Registry Slice

## Goal

Add the first minimal, compile-tested C++ surface for TianChen-RV extension plugins so later core orchestration can communicate through plugin interfaces and a deterministic registry instead of hard-coded RVV/IME/Sophgo/AME branches.

## Requirements

- Define a lightweight compiler-visible capability descriptor/value object for plugin-provided capability metadata in C++.
- Define an abstract extension plugin interface with stable plugin identity/name, capability declaration, and a dialect registration hook using `mlir::DialectRegistry`.
- Keep any later variant/legality concepts out of scope unless they are compile-only placeholders and clearly separated from concrete selection/lowering.
- Implement a small deterministic plugin registry that registers plugin references, rejects duplicate plugin names with `llvm::Error`, supports lookup by name, and exposes all/enabled plugins.
- Optionally expose simple capability lookup by id/kind if it remains small and tested.
- Integrate the new library through CMake without changing `tcrv-opt` runtime behavior except normal linking/compilation.
- Add a test-only mock plugin in test code to validate ordering, duplicate rejection, lookup, dialect hook invocation, and capability visibility.
- Do not implement concrete RVV/IME/offload dialects, variant generation/selection passes, lowering, emission, or hardware runtime paths in this round.

## Acceptance Criteria

- [x] New plugin interface and registry live in C++ headers/sources under `include/TianChenRV/Plugin/` and `lib/Plugin/` or an equivalent conventional compiler location.
- [x] Registry code has no target-specific RVV/IME/Sophgo/AME branches.
- [x] `tcrv.exec` ODS/C++ remains compute-free and unchanged unless a build include fix is truly required.
- [x] A compile/run smoke test is integrated into `check-tianchenrv` and validates the mock plugin behavior.
- [x] Local configure/build/check passes with LLVM/MLIR CMake packages.
- [x] `git diff --check` passes.
- [x] Trellis task validation passes and task is archived before the final commit.
- [x] One coherent commit is created and the final worktree is clean.

## Definition of Done

- Primary implementation remains C++ / MLIR / LLVM / CMake / lit or C++ tests.
- Python is used only for Trellis/task support scripts, not compiler semantics.
- No RVV runtime/correctness/performance claim is made, so no `ssh rvv` evidence is required.
- Build artifacts, logs, editor files, and `artifacts/tmp` outputs are not committed.

## Technical Approach

Create a small `TianChenRVPlugin` library containing:

- `PluginCapability` metadata with id/kind/description fields.
- `ExtensionPlugin` abstract interface with `getName`, `getVersion`, `getCapabilities`, `registerDialects`, and `isEnabled`.
- `ExtensionPluginRegistry` storing plugin references in insertion order, using `llvm::StringMap` for duplicate detection and lookup.
- Helper APIs to register all plugin dialect hooks into an `mlir::DialectRegistry` and query capabilities by id/kind.

Add a test executable containing only mock plugins and invoke it from lit via a `%tianchenrv-plugin-registry-test` substitution so the smoke test is part of `check-tianchenrv`.

## Out of Scope

- Concrete RVV, IME, offload, Sophgo, AME, or scalar plugin implementations.
- Variant generation, capability-aware selection, tuning, legality orchestration, lowering, emission, or runtime glue.
- Python plugin registry/capability model implementation.
- RVV hardware/runtime validation.

## Technical Notes

- User-required inspection completed before editing: repo root, git status/log/show, Trellis task state, root AGENTS/README/CMake, relevant specs, Exec ODS/C++/init, and current lit tests.
- Relevant specs: `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, `.trellis/spec/plugin-protocol/locality-contract.md`, `.trellis/spec/capability-model/capability-contract.md`, `.trellis/spec/core-dialect/tcrv-exec-contract.md`, `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
