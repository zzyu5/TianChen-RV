# RVV/Scalar Callable ABI Preflight Parity

## Goal

Make RVV and scalar callable target-artifact routes validate compiler-owned
runtime ABI role contracts through the generic target-artifact preflight
boundary, matching the offload descriptor route's front-door behavior.

## Scope

- Keep implementation in C++/MLIR/CMake/lit/spec files.
- Reuse `support::RuntimeABIContract` and `support::RuntimeABIParameter`.
- Keep generic `TargetArtifactExport` code target-family-free.
- Keep RVV/scalar/offload route-specific decisions in plugin-local or
  target-export-local owners.
- Preserve `rvv_available` as a dispatch runtime guard/control input only.
- Preserve `n`/runtime-element-count as an explicit ABI/control input.
- Do not change emitted C code semantics or make RVV runtime/correctness/
  performance claims.

## Acceptance

- Valid RVV/scalar callable source, header, object, dispatch, or bundle routes
  still export deterministic artifacts.
- A stale/missing/conflicting RVV or scalar runtime ABI role contract fails at
  the target-artifact front-door preflight boundary with a precise diagnostic.
- Offload descriptor ABI preflight tests remain passing.
- Final validation runs `git diff --check`, CMake configure under
  `artifacts/tmp/tianchenrv-build`, and `check-tianchenrv`.
