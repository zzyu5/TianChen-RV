# Dispatch-Aware Capability Legality

## Goal

Extend the existing `--tcrv-check-capability-requires` pass so generic capability availability checks understand structured `tcrv.exec.dispatch`, `tcrv.exec.case`, and `tcrv.exec.fallback` without adding selection, cost, tuning, lowering, runtime ABI, concrete plugin, or target-family logic.

## Requirements

- Build the `TargetCapabilitySet` per `tcrv.exec.kernel` as the existing pass does.
- Preserve verifier boundaries: malformed `requires` attributes and unknown capability symbols remain `tcrv.exec` verifier diagnostics.
- Reject variants with unavailable requirements when they are not protected by a structured dispatch case.
- Accept unavailable requirements for dispatch case targets only when the case carries at least one non-empty generic `condition`, `guard`, or `policy` attribute.
- Reject dispatch cases that target unavailable variants without any generic guard field.
- Require each dispatch fallback target to be generically available under the same target capability set.
- Succeed when all non-fallback dispatch cases are guarded and unavailable while the fallback variant remains available.
- Keep all logic target-independent and generic, with no RVV/IME/Sophgo/AME/offload target-family branches.

## Acceptance Criteria

- Existing positive and negative `--tcrv-check-capability-requires` tests continue to pass.
- Add positive lit coverage for a guarded unavailable dispatch case with an available fallback.
- Add negative lit coverage for an unguarded unavailable dispatch case.
- Add negative lit coverage for an unavailable dispatch fallback.
- Add positive neutral non-RVV capability names to show target-independent behavior.
- Run local CMake configure and `check-tianchenrv`.
- Run `git diff --check`.
- Validate and archive Trellis task state before final commit if supported.

## Out of Scope

- No IR rewrite or selected-variant marking.
- No selection, cost model, tuning, lowering, emission, runtime glue, runtime ABI, or concrete plugin work.
- No parsing or target-specific interpretation of dispatch condition strings.
- No Python implementation of compiler internals.

## Technical Notes

- Primary implementation files are expected to remain C++ / MLIR / TableGen / CMake / lit.
- Relevant specs: core dialect contract, capability contract, variant pipeline, plugin protocol locality/interface contracts, and MLIR testing contract.
