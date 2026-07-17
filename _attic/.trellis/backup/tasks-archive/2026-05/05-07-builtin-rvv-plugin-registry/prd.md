# tcrv-opt built-in RVV plugin registry integration

## Goal

Integrate a deterministic C++ built-in extension plugin registry surface into public `tcrv-opt` passes so registry-dependent public diagnostics can route `origin = "rvv"` variants through the existing RVV plugin and report plugin-owned unsupported readiness/emission-plan metadata instead of generic unregistered-origin failures.

## What I already know

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Expected HEAD: `3c623cd feat: materialize emission plan diagnostics`.
* The previous round added structured emission-plan diagnostics and left RVV first-slice emission unsupported.
* This round must remain single-worker, serial, non-TUI, with no subagents or parallel agent workflows.
* Core implementation must remain C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck.
* Python is allowed only for Trellis/task/support scripting, not compiler internals.

## Requirements

* Add or strengthen a reusable C++ registration helper for built-in extension plugins, starting with the existing RVV plugin.
* Wire `tcrv-opt` so public registry-dependent passes can use a deterministic tool-level `ExtensionPluginRegistry` populated with built-ins.
* Keep core selection/readiness/plan logic target-neutral and free of `origin == "rvv"` target-family branches.
* Ensure `tcrv-check-emission-paths` and `tcrv-materialize-emission-plans` route selected RVV-origin variants through the RVV plugin using the existing repository contract `origin = "rvv-plugin"` and produce unsupported plugin-owned diagnostics.
* Preserve generic unregistered-origin diagnostics for unknown origins.
* Preserve deterministic dispatch case order and fallback-after-cases ordering.
* Do not implement RVV lowering, RVV intrinsics, assembly, object generation, runtime ABI glue, or executable emission.
* Update specs/tests to clarify built-in plugin registration and unsupported RVV first-slice diagnostic scope.

## Acceptance Criteria

* [x] Public `tcrv-opt` lit coverage proves selected RVV variants route through the RVV plugin for emission readiness.
* [x] Public `tcrv-opt` lit coverage proves selected RVV variants route through the RVV plugin for emission-plan materialization.
* [x] Unknown origins still produce generic unregistered-origin diagnostics.
* [x] Unselected RVV variants are not routed when the selected marker points elsewhere.
* [x] Dispatch case order and fallback-after-cases ordering stay deterministic in public output.
* [x] C++ tests cover RVV built-in registration, duplicate policy, and empty/unavailable generic behavior.
* [x] Required local configure/build/check commands pass or failures are reported precisely.

## Definition of Done

* Tests added/updated for public tool and C++ helper behavior.
* Specs updated for public built-in plugin registration and unsupported diagnostic scope.
* `git diff --check`, CMake configure, `check-tianchenrv`, focused binaries, and focused lit checks run.
* Trellis task validates, is finished/archived, and included in the final coherent commit.
* Worktree is clean after the final commit.

## Out of Scope

* RVV lowering or code generation.
* RISC-V/RVV runtime execution, correctness, or performance evidence.
* Python compiler internals.
* Compute semantics inside `tcrv.exec`.
* Core orchestration target-family special cases.

## Technical Notes

* Required inspection command set was run before this PRD was created; worktree was clean and HEAD matched `3c623cd`.
* Required files and specs are being read before code edits.
* Any Trellis task state created in this round must be validated, finished, archived, and included in the final coherent commit.
* Final implementation adds `TianChenRVBuiltinPlugins`, wires `tcrv-opt` to a deterministic tool-owned built-in registry by default, and preserves `--tcrv-disable-builtin-plugins` for empty-registry negative tests.
* Final validation: `git diff --check`, CMake configure, `check-tianchenrv`, focused C++ tests, and focused lit filter passed.
