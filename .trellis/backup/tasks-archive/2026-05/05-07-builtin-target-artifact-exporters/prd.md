# Builtin Target Artifact Exporter Registration Boundary

## Goal

Add a reusable C++ registration boundary for built-in target artifact exporters so generic `tcrv-translate` artifact routes no longer enumerate every RVV, scalar, and offload exporter directly.

## Requirements

- Add a target support API such as `registerBuiltinTargetArtifactExporters(TargetArtifactExporterRegistry &registry)`.
- Register the current built-in target-owned artifact exporters through that API:
  - RVV explicit i32 vector-add microkernel standalone C source.
  - Scalar explicit i32 vector-add standalone C source.
  - Offload runtime handoff descriptor.
- Keep RVV/scalar/offload export implementation and route semantics in their target-specific modules.
- Wire `tcrv-export-target-source-artifact` and `tcrv-export-target-artifact` through the built-in registration boundary.
- Preserve the legacy direct `tcrv-export-rvv-microkernel-c` command.
- Preserve source-only behavior: `tcrv-export-target-source-artifact` must not export the offload descriptor.
- Preserve artifact-kind aware behavior: `tcrv-export-target-artifact` must export source artifacts and the offload descriptor only when the selected plan and artifact kind are legal.
- Do not add target-family branches to generic `TargetArtifactExport.cpp` or core transforms.
- Keep the slice bounded to C++/CMake/tests/spec notes needed for this registration boundary.

## Acceptance Criteria

- [ ] C++ registration boundary exists under the Target support layer.
- [ ] Generic `tcrv-translate` artifact helpers use the boundary instead of direct per-target registration calls.
- [ ] Tests prove all current built-in routes are registered and duplicate registration still fails.
- [ ] Registered route metadata remains target-neutral, deterministic, and queryable without interpreting target semantics in generic routing.
- [ ] Existing lit route coverage still proves RVV source, scalar source, offload descriptor, source-only filtering, and fail-closed spoof cases.
- [ ] Inspection confirms generic target artifact helper code in `tcrv-translate` no longer directly calls all target-specific registration helpers.
- [ ] `git diff --check`, CMake configure, and `check-tianchenrv` pass.
- [ ] Task is validated and archived before the final commit.

## Definition Of Done

- Code, CMake, tests, and any narrowly necessary spec notes are updated.
- Build/test evidence is recorded in the final report.
- No generated artifacts, build outputs, logs, credentials, or temporary evidence are committed.
- Worktree is clean after one coherent commit.

## Technical Approach

Create a small Target-layer built-in exporter bundle that includes target-owned exporter headers and delegates registration to each target-owned registration function. The bundle itself owns only registration composition; it does not inspect MLIR plans or interpret RVV/scalar/offload semantics.

Use `TargetArtifactExporterRegistry` lookup/introspection narrowly in tests to assert registered route metadata. Keep existing generic artifact validation in `TargetArtifactExport.cpp` unchanged unless a tiny generic helper is needed.

## Out Of Scope

- New target routes, new microkernels, new runtime/offload execution, object generation, linking, or driver integration.
- RVV hardware runs or new RVV runtime/correctness/performance claims.
- Python compiler internals.
- Generic compute operations in `tcrv.exec`.
- Broad CMake/library rewrites.

## Technical Notes

- Required inspection confirmed repo root `/home/kingdom/phdworks/TianchenRV`, HEAD `4e1852f`, clean initial `git status --short`, and `predoc/` absent.
- Relevant specs: plugin protocol registration boundary, lowering/runtime target artifact route, RVV/scalar/offload plugin boundaries, and MLIR testing contract.
- Current bottleneck: `tools/tcrv-translate/tcrv-translate.cpp` directly includes target-specific artifact headers and manually calls RVV/scalar/offload exporter registration from generic artifact helper functions.
