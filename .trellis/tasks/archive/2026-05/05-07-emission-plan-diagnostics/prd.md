# materialize emission plan diagnostics

## Goal

Materialize plugin-owned `VariantEmissionPlan` results for selected execution paths into structured `tcrv.exec.diagnostic` metadata that is visible through MLIR / `tcrv-opt` while staying non-executable and target-neutral.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* HEAD is `fdc14ba feat: add plugin emission plan collection` and the initial worktree is clean.
* Previous round added `VariantEmissionPlan`, registry routing, selected-path plan collection, RVV unsupported plan behavior, C++ tests, specs, and archived Trellis state.
* Current live gap: plans are collected as C++ objects but do not yet have durable MLIR-visible structured diagnostic materialization or `tcrv-opt` verification surface.
* This round must remain C++ / MLIR / TableGen / CMake / lit / C++ tests; Python is only support tooling.

## Requirements

* Reuse and minimally extend `tcrv.exec.diagnostic` with generic optional emission-plan attributes when needed.
* Define central constants/helpers for the emission-plan diagnostic convention rather than scattered string literals.
* Add a reusable C++ API such as `materializeKernelEmissionPlanDiagnostics` that uses `collectKernelEmissionPlans` and mutates IR only after collection and validation succeeds.
* Preserve deterministic selected-path order: dispatch cases in order, fallback after cases, selected marker target alone, or all direct variants in IR order for conservative mode.
* Validate emission-plan diagnostics generically: supported plans require non-empty origin, target, role, emission kind, lowering pipeline, runtime ABI, and artifact kind; unsupported plans require non-empty origin, target, role, and diagnostic/explanation text.
* Diagnostic targets must resolve to direct sibling `tcrv.exec.variant` operations in the same `tcrv.exec.kernel`.
* Reject duplicate/conflicting materialized emission-plan diagnostics rather than silently appending stale diagnostics.
* Add a real public `tcrv-materialize-emission-plans` pass if useful; default registration must use an empty registry and diagnose unregistered origins rather than invent core semantics.
* Keep `tcrv-check-emission-paths` strict and keep RVV first-slice emission unsupported.
* Do not generate LLVM/RISC-V/RVV lowering, assembly, object files, runtime calls, toolchain invocation, executable emission, or RVV runtime/correctness/performance claims.

## Acceptance Criteria

* Lit/FileCheck covers valid/invalid `tcrv.exec.diagnostic` emission-plan attributes and verifier behavior.
* C++ tests cover supported materialization, dispatch ordering, selected marker behavior, conservative all-variant behavior, RVV unsupported behavior, and failure-before-partial-mutation cases.
* If public pass is added, `tcrv-opt` lit coverage proves default empty-registry diagnostics and structured output through a registry-injected C++ path.
* Specs in core dialect, plugin protocol, variant pipeline, lowering runtime, and testing are updated to describe emission-plan diagnostics as compiler-visible metadata only.
* Required local checks pass: `git diff --check`, CMake configure, `check-tianchenrv`, focused binaries, focused lit filter, and Trellis task validation.

## Out of Scope

* Actual lowering or executable emission.
* RVV runtime/correctness/performance evidence or `ssh rvv` use.
* Core-side target-family, vendor, dtype, shape, runtime, or toolchain semantics.
* Python compiler internals.

## Technical Notes

* Required inspection files are listed in the user request and were read before implementation.
* Relevant implementation files include `ExecOps.td/cpp`, `ExtensionPlugin.h/cpp`, `EmissionReadiness.h/cpp`, `Passes.h/td`, transform tests, exec verifier lit tests, CMake, and `tcrv-opt` registration.
