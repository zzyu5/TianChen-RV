# RVV scalar dispatch self-check object artifact route

## Goal

Advance the bounded RVV+scalar i32-vadd dispatch self-check path from source
export to the first target-owned object-file artifact boundary.

## Requirements

- Implement the object route in C++ target/export/translate code.
- Reuse the existing validated RVV+scalar dispatch self-check C source path.
- Preserve selected-path, lowering-boundary, emission-plan, route-id,
  artifact-kind, runtime ABI parameter, dispatch/fallback role, and bounded
  source validation before object creation.
- Keep generic routing target-neutral. Extension-specific compile facts and
  object emission belong in target-owned RVV+scalar dispatch exporter code.
- Scope is finite: only the existing bounded RVV+scalar i32-vadd dispatch
  self-check object route.
- Do not add generic RVV lowering, arbitrary kernels, dynamic loading,
  automatic hardware probing, linking/runtime installation, or performance
  claims.
- If local RISC-V/RVV clang support is unavailable, fail closed with an exact
  diagnostic.

## Acceptance Criteria

- Public tool surface exposes the object route through a deterministic
  translation path.
- Minimal lit/C++ coverage verifies route visibility, source/self-check split
  preservation, object-route fail-closed behavior for malformed metadata, and
  object bytes when local toolchain support exists.
- Required project checks run:
  - `git diff --check`
  - CMake configure under `artifacts/tmp/tianchenrv-build`
  - build `tcrv-translate`
  - `check-tianchenrv`
- Bounded `ssh rvv` evidence is collected if available and kept under
  `artifacts/tmp`.

## Out Of Scope

- Python implementation of compiler decisions or object routing.
- New standalone smoke probe or helper as the owner of the round.
- Generic high-level op lowering, arbitrary RVV kernel support, dynamic runtime
  loader, performance measurement, or broad correctness claims.

## Technical Notes

- Required live inspection completed before code edits.
- User-requested path `lib/Target/BuiltinTargetArtifactExporters.cpp` maps to
  actual repo file `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`.
