# IR-linked RVV scalar dispatch fallback export

## Goal

Make the RVV+scalar runtime-callable dispatch exporter consume both sides of the selected `tcrv.exec.dispatch` surface before C source or object export. The RVV branch must stay linked through `tcrv.exec.case runtime_guard`; the scalar fallback callable must now be validated against the actual selected `tcrv.exec.fallback` target rather than detached route metadata alone.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Expected HEAD is `839b17f feat: export dispatch library object route`.
* Initial worker inspection found the worktree clean.
* The three supervisor-policy files are clean and must not be touched in this compiler round.
* Previous round exported a runtime-callable RVV+scalar dispatch library object route and kept the self-check object as explicit evidence helper.
* No ssh `rvv` runtime, correctness, or performance claim is required for this round.

## Requirements

* Locate the unique direct `tcrv.exec.dispatch` for the selected RVV+scalar dispatch pair.
* Preserve the existing selected RVV case `runtime_guard` to `runtime_param` validation.
* Resolve exactly one `tcrv.exec.fallback` inside that dispatch.
* Validate that the fallback target symbol matches the selected scalar fallback callable route variant symbol.
* Validate that the fallback target resolves to a direct `tcrv.exec.variant` under the same kernel.
* Fail closed with clear target-owned diagnostics when dispatch/fallback resolution is missing, ambiguous, unknown, not a variant, or mismatched.
* Keep the default library object route and explicit self-check object helper behavior.
* Keep implementation in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck; Python is only allowed for runner/support utilities.
* Do not add generic compute operations to `tcrv.exec` or generic target-family branches to core code.

## Acceptance Criteria

* Positive FileCheck coverage shows RVV runtime guard linkage and scalar fallback linkage in the generated C export.
* Focused negative coverage fails before source/object export when selected scalar route metadata names one variant and `tcrv.exec.fallback` targets another, with diagnostics naming both symbols.
* Existing object route test still proves library object output is relocatable, riscv64, has dispatcher/callable symbols, and has no `main`.
* Required checks pass: `git diff --check`, `cmake --build build --target tcrv-opt tcrv-translate -j2`, `cmake --build build --target check-tianchenrv -- -j2`, and `build/bin/tianchenrv-target-artifact-export-test` if applicable.

## Out of Scope

* ssh `rvv` runtime, correctness, or performance validation.
* Rewriting broad RVV/IME/offload lowering.
* Re-promoting self-check object as the generic artifact front door.
* Python implementations of compiler IR, dialects, verifier, plugin registry, target exporter, or runtime ABI decisions.
* Large unrelated spec or test rewrites.

## Technical Notes

* Primary target-owned implementation files are `lib/Target/Builtin/RVVScalarDispatch.cpp` and `include/TianChenRV/Target/RVVScalarDispatch.h` if a narrow public surface update is needed.
* Focused tests should extend existing RVVScalarDispatch lit tests and `TargetArtifactExportTest.cpp`.
* Specs/docs updates should be durable and limited to lowering-runtime/testing/core-dialect wording needed to document dispatch export consuming both selected case and fallback from `tcrv.exec.dispatch`.
