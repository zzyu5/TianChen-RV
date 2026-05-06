# emission plan first slice

## Goal

Add a bounded C++/MLIR first slice for generic plugin-owned emission plans after emission readiness: describe plugin-owned lowering/runtime intent and structured diagnostics for selected paths without generating executable code or making RVV runtime/correctness/performance claims.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* HEAD is `b108f60 feat: add selected variant path markers`.
* Current worktree was clean before task creation.
* The previous milestone added selected-path markers and plugin emission readiness checks.
* The live gap is the absence of a generic plugin-owned emission plan interface carrying selected-path lowering/runtime boundary metadata.

## Requirements

* Add a target-neutral C++ plugin protocol object for emission plans.
* Plans are compiler decision metadata/intent, not executable artifacts.
* Add plugin-owned hooks and registry routing by generic variant origin.
* Validate malformed plans with `llvm::Error` / `llvm::Expected`.
* Reuse selected-path traversal from emission readiness instead of duplicating divergent logic.
* Add selected-path emission-plan collection over real MLIR ops.
* Keep RVV first-slice emission unsupported with explicit structured diagnostics.
* Do not add core origin/target-family branches or Python compiler internals.

## Acceptance Criteria

* [x] Supported mock plugin returns a well-formed plan for a selected static variant.
* [x] Supported mock plugin returns deterministic plans for dispatch cases and fallback.
* [x] Selected marker plans only the selected target; unselected unsupported variants do not fail selected-path planning.
* [x] No dispatch and no selected marker conservatively plans all direct variants.
* [x] RVV remains unsupported with structured unsupported diagnostic.
* [x] Unregistered origin, disabled plugin, empty origin, mismatched variant symbol, malformed fields, duplicate selected markers, and missing dispatch targets are diagnosed generically.
* [x] No new tcrv-opt plan pass was exposed; existing default readiness lit coverage remains intact.
* [x] Normal local checks pass.

## Out of Scope

* No lowering to LLVM/RISC-V/RVV intrinsics/assembly/object files.
* No executable emission, runtime execution, correctness claim, or performance claim.
* No `ssh rvv` evidence unless actual runtime claims are intentionally introduced.
* No core compute ops or compute semantics in `tcrv.exec`.

## Technical Notes

* Implementation stack remains C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck.
* Python is limited to Trellis/support scripts and test runners.
* Required specs and code files are listed in the user handoff and will be inspected before code changes.
