# capability query model and legality pass

## Goal

Add the first reusable C++ capability query layer and a target-independent MLIR pass that makes `tcrv.exec.capability` availability affect compiler legality decisions for `tcrv.exec.variant` requirements.

## Requirements

- Implement capability query objects in C++/MLIR, not Python.
- Build capability state from `tcrv.exec.kernel` contents by reading `tcrv.exec.capability` operations.
- Capture capability symbol name, `id`, `kind`, and an optional generic availability/status field.
- Default missing status to available/present.
- Treat generic status strings such as unavailable, disabled, or missing as unavailable.
- Support lookup by symbol name and id, collection/query by kind, and availability checks.
- Add a target-independent pass callable from `tcrv-opt`.
- Walk `tcrv.exec.kernel`, inspect `tcrv.exec.variant` `requires`, and diagnose required capabilities that are declared but generically unavailable.
- Preserve the existing dialect verifier ownership for malformed `requires` and unknown capability symbols.
- Do not add variant generation, selection, cost, tuning, lowering, emission, runtime dispatch, production plugins, or extension-specific branches.
- Keep `tcrv.exec` execution/capability/variant focused and compute-free.

## Acceptance Criteria

- `tcrv-opt` exposes a pass such as `--tcrv-check-capability-requires`.
- lit/FileCheck covers a positive available-capability case.
- lit/FileCheck covers a negative declared-but-unavailable capability diagnostic.
- lit/FileCheck includes at least one generic non-RVV capability example such as `toolchain` or `runtime-offload`.
- CMake wires new Support/Transforms libraries and tests into `check-tianchenrv`.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passes.
- `git diff --check` passes.
- Trellis task validation passes if supported by the local script.

## Out of Scope

- No concrete RVV, IME, Sophgo, AME, offload dialect implementation.
- No runtime/correctness/performance RVV claim or `ssh rvv` run.
- No Python implementation of compiler-owned IR, capability model, pass, registry, legality, lowering, or emission.
- No extension-specific hard-coded branches in core code.

## Technical Notes

- Required source inspection was completed before editing.
- Relevant specs: capability model, plugin protocol, variant pipeline, testing, and shared capability/plugin/compute-boundary guides.
- Direct single-worker implementation is used for this round because the user explicitly prohibited subagents and multi-agent workflows.
