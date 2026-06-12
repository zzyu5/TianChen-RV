# Selected Target Artifact Front-Door Coherence

## Goal

Make selected target-artifact front-door coherence an active C++ compiler/export preflight before `tcrv-translate` writes bundle outputs. A selected execution plan must prove that selected variants, dispatch/fallback roles, plugin lowering boundaries, emission-plan diagnostics, and registry-derived target artifact bundle records agree on the exportable handoff.

## Scope

- Implement bounded C++ coherence/preflight behavior using existing MLIR plan surfaces.
- Prefer the existing `ExecutionPlanCoherence` transform/helper and Target/export APIs without creating dependency cycles.
- Ensure `tcrv-translate --tcrv-export-target-artifact-bundle` rejects incoherent selected artifact surfaces before accepting bundle output.
- Preserve target-owned final exporter validation and descriptor safety checks.
- Add focused lit/FileCheck coverage through real `tcrv-opt` and `tcrv-translate` front doors.

## Required Positive Behavior

- Selected offload runtime descriptor path is accepted as one descriptor front door.
- RVV plus scalar dispatch composite path remains accepted when a registered composite route supports it.
- Scalar fallback-only path remains accepted when it is the selected fallback front door.

## Required Fail-Closed Behavior

- Unsupported, stale, or ambiguous selected artifacts are rejected.
- An unselected fallback artifact must not leak into a bundle when a supported non-fallback selected artifact is the front door.
- Multiple selected standalone artifacts without a registered composite route are rejected with a clear diagnostic.
- Descriptor-only offload path continues to reject missing runtime ABI, missing handoff kind, stale lowering boundary, unsafe raw-log/secret-like metadata, and RVV/scalar spoofing of the offload descriptor route.

## Non-Goals

- No Python implementation of compiler internals.
- No new lowering, hardware execution, runtime calls, DMA, accelerator kernels, Sophgo, IME, or AME implementation.
- No new `tcrv.exec` compute op or high-level tensor/tile IR.
- No new RVV runtime/correctness/performance claim or remote `ssh rvv` run.

## Evidence

Minimum local checks:

- `git diff --check`
- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- `cmake --build build --target check-tianchenrv -j2`
