# emission readiness routing diagnostics

## Goal

Add a bounded C++/MLIR plugin-owned emission-readiness protocol slice before final lowering, routed generically through ExtensionPluginRegistry by variant origin, with diagnostics and tests that do not claim RVV lowering/runtime support.

## Requirements

* Add a generic emission readiness request/result/status interface to the C++ plugin protocol.
* Carry at minimum materialized `tcrv.exec.variant`, enclosing `tcrv.exec.kernel`, and generic `TargetCapabilitySet` in the request.
* Route emission readiness through `ExtensionPluginRegistry` by variant origin using the existing legality/cost routing style.
* Validate missing variant/kernel, non-sibling variant/kernel relationships, missing/empty origin, unknown plugin, disabled plugin, malformed plugin result, and plugin-local unsupported/failure states with context-rich diagnostics.
* Add a reusable helper and public MLIR pass such as `tcrv-check-emission-paths`, registered in `Passes.td`/`Passes.h` and available through `tcrv-opt` with an empty default registry.
* For kernels with `tcrv.exec.dispatch`, verify every referenced case target and fallback target after structural symbol validation.
* For kernels without dispatch or selected marker, verify direct `tcrv.exec.variant` children.
* Keep dispatch/fallback reference validation generic: references must resolve to direct sibling `tcrv.exec.variant` ops in the same kernel and duplicate/missing/non-variant references must be diagnosed before plugin routing.
* Integrate `RVVExtensionPlugin` so metadata-only first-slice RVV variants return explicit unsupported emission status/reason, not fake emission support.
* Add mock/test plugins with supported emission paths for positive tests.
* Do not auto-register built-in or RVV plugins in `tcrv-opt` by default.
* Do not implement actual lowering, runtime ABI, toolchain invocation, executable kernels, hardware probes, or performance/correctness claims.
* Keep compiler implementation in C++/MLIR/TableGen/CMake/lit/C++ tests; Python only for Trellis/task/test support.

## Acceptance Criteria

* [x] Mock supported plugin routes readiness for a materialized direct `tcrv.exec.variant`.
* [x] Injected-registry `tcrv-check-emission-paths` succeeds for direct variants owned by the supported mock plugin.
* [x] Injected-registry pass succeeds for dispatch where every case and fallback target resolves to direct sibling supported variants.
* [x] Diagnostics/results preserve plugin, kernel, variant, and emission path context without target-family branching.
* [x] Public `tcrv-opt` default pass diagnoses unregistered origins with an empty registry.
* [x] Missing origin, empty origin, unknown origin, disabled origin, missing kernel, cross-kernel variant, and malformed plugin result are rejected.
* [x] Dispatch case/fallback missing, non-variant, or non-sibling targets are diagnosed before plugin routing.
* [x] RVV plugin returns clear unsupported-emission diagnostic for metadata-only first-slice variants and makes no lowering/runtime claim.
* [x] Existing legality, cost, selection, dispatch, materialization, RVV dialect, and RVV plugin tests keep passing.
* [x] `git diff --check`, configure, build, and `check-tianchenrv` pass.

## Definition of Done

* C++/MLIR implementation and tests are committed as one coherent commit.
* Build artifacts/logs are not committed.
* Trellis task state is archived and validated if changed.
* Final report includes changed files, spec requirements, checks, no-ssh-rvv evidence statement, risks, task coherence, clean repo, and invariant status.

## Out of Scope

* Actual RVV/IME/offload/scalar lowering or runtime generation.
* RVV hardware/toolchain/runtime/correctness/performance evidence or claims.
* New generic compute ops in `tcrv.exec`.
* RVV-specific branches or target-family interpretation in core passes/protocols.
* Auto-registration of built-in/RVV plugins in public `tcrv-opt` default registry.

## Technical Notes

* User explicitly requested one serial full-access non-TUI worker and no subagents/multi-agent workflow.
* Required repository inspection and required source/spec reads must be completed before code edits.
