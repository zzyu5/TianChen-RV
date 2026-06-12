# Generic plugin lowering-boundary interface

## Goal

Factor selected-path lowering-boundary materialization into a generic C++ extension-plugin interface and registry-routed public pass, with RVV as the first concrete implementation and scalar fallback as a valid metadata-only/no-boundary result.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- HEAD is `3541d8d feat: materialize RVV lowering boundary metadata` and the worktree was clean before this task.
- The existing RVV lowering-boundary path is RVV-specific and materializes `tcrv_rvv.lowering_boundary` as pre-executable metadata only.
- `tcrv.exec` must remain compute-free; RVV-specific structure stays in the RVV dialect/plugin.
- Scalar fallback is explicit-capability-gated and metadata-only; it must not be treated as executable lowering.
- No ssh rvv runtime/correctness/performance evidence is required or planned for this MLIR/plugin-interface round.

## Requirements

- Add strongly typed C++ request/result APIs for selected-path lowering-boundary materialization near existing plugin emission APIs.
- Add `ExtensionPlugin` and `ExtensionPluginRegistry` routing methods that delegate by selected variant origin through the owning enabled plugin.
- Validate generic error cases: missing/empty origin, unknown plugin, disabled plugin, malformed request, non-direct variant/kernel relationship, malformed dispatch references, unsupported/no-op responses, and plugin result mismatches.
- Refactor RVV implementation so RVV-specific interpretation is behind the RVV plugin interface.
- Keep RVV materialization pre-executable: selected RVV direct/dispatch variants produce `tcrv_rvv.lowering_boundary` with `status = "unsupported"` and non-empty `unsupported_reason`.
- Route scalar fallback through the same generic registry path as a valid no-boundary/metadata-only response.
- Add a public generic pass such as `--tcrv-materialize-selected-lowering-boundaries` and wire it into `tcrv-opt` with built-in plugin registry support.
- Retire or wrap the old RVV-specific pass without leaving divergent implementations.
- Add/update C++ and lit/FileCheck coverage for registry routing, scalar fallback no-op, RVV+fallback dispatch preservation, malformed selected paths, and generic diagnostics.
- Preserve project boundary: no Python compiler internals, no executable RVV lowering, no runtime ABI, no correctness/performance claims.

## Acceptance Criteria

- [x] C++ plugin interface and registry path compile and are covered by focused tests.
- [x] Generic pass materializes RVV boundaries through the registry and contains no target-family branches.
- [x] Scalar fallback-only selected paths produce no `tcrv_rvv.lowering_boundary` and no false missing-plugin error.
- [x] Existing RVV lowering-boundary public behavior remains available either through generic canonical tests or a thin wrapper.
- [x] Local checks pass: `git diff --check`, CMake configure, `check-tianchenrv`, focused test binaries, and focused lit filter.
- [x] Trellis task is validated and archived before final commit.

## Out of Scope

- RVV intrinsic lowering, LLVM/RISC-V lowering, object generation, runtime ABI, hardware execution, correctness proof, benchmarking, or executable emission.
- Any target-family branches in generic support, selection, dispatch, or the generic lowering-boundary pass.
- Weakening `tcrv_rvv.lowering_boundary` verifier restrictions.
- Changing RVV emission readiness from unsupported to supported.

## Technical Notes

- Must inspect and follow `AGENTS.md`, `.trellis/spec/index.md`, core dialect, plugin protocol, variant pipeline, lowering/runtime, RVV/scalar plugin, and testing specs.
- Implementation stack remains C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck.
