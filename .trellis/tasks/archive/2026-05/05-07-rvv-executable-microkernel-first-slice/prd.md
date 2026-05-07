# RVV executable microkernel first slice

## Goal

Add one bounded RVV extension-dialect executable microkernel slice and a
target-local C exporter that can generate a deterministic standalone RVV i32
vector-add smoke executable from post-planning MLIR. The claim is limited to
that explicit microkernel path and must not broaden generic RVV kernel lowering,
runtime ABI support, or performance evidence.

## Requirements

- Keep compiler structure in C++ / MLIR / TableGen / ODS / CMake / lit; Python
  remains limited to existing probe/artifact helper roles.
- Add one explicit RVV extension dialect op for a finite i32 vector-add
  microkernel body. The op must live under `tcrv_rvv`, not `tcrv.exec`.
- The op must carry only bounded selected-path metadata: source kernel,
  selected variant, origin, selected role, required capability refs, required
  march, optional selected mabi, and a tiny element count.
- Verify malformed metadata, stale selected variant references, mismatched
  variant requirements, invalid element counts, missing required capability
  refs, unavailable/non-RVV capability refs, unbounded/secret-like strings, and
  generic tensor/tile/benchmark-style attributes.
- Add a distinct RVV target-local exporter reachable through
  `tcrv-translate --tcrv-export-rvv-microkernel-c`.
- The exporter consumes post-planning MLIR with a selected `rvv-plugin` path,
  matching `tcrv_rvv.lowering_boundary`, preserved selected march metadata, and
  exactly one matching executable microkernel op for the selected kernel and
  variant.
- Generated C must include `riscv_vector.h`, use RVV intrinsics, initialize
  bounded arrays, run the generated microkernel function, and self-check
  deterministic i32 vector-add results.
- Preserve the existing `--tcrv-export-rvv-smoke-probe-c` behavior.
- Do not update generic manifest/export or generic core orchestration for RVV
  executable support in this round. Default RVV emission readiness/plan remains
  unsupported/deferred unless a later slice explicitly wires supported emission.
- Capture real `ssh rvv` compile/run evidence when reachable, under
  `artifacts/tmp/rvv_microkernel/`, without committing artifacts.

## Acceptance Criteria

- [ ] RVV dialect lit/FileCheck covers positive parse/verify for the new op.
- [ ] RVV dialect negative tests cover malformed selected-path metadata,
      invalid element count, missing/invalid capability refs, unavailable RVV
      capability, secret/unbounded metadata, and forbidden generic compute
      attributes.
- [ ] RVV target lit/FileCheck covers pipeline-to-export deterministic C with
      `riscv_vector.h`, selected kernel/variant comments, RVV intrinsics,
      bounded function name, `main`, self-check, and no unsafe evidence strings.
- [ ] RVV target negative tests fail before source for missing selected RVV
      path, scalar/offload-only paths, missing/stale boundary, missing
      microkernel op, duplicate microkernel ops, malformed march metadata, and
      malformed op metadata.
- [ ] Existing smoke-probe exporter behavior remains covered and unchanged.
- [ ] Local checks pass: `git diff --check`, RVV Python helper self-tests,
      CMake configure, and `check-tianchenrv`.
- [ ] If `ssh rvv` is reachable, generated microkernel C is compiled with the
      selected march/mabi and its self-check exits 0, with bounded sanitized
      evidence saved under `artifacts/tmp/rvv_microkernel/`.

## Out Of Scope

- Generic high-level MLIR lowering.
- Generic vector math, reductions, memory analysis, bufferization, runtime
  library linking, benchmarking, performance measurement, or arbitrary RVV
  kernel executable emission.
- RVV branches in core orchestration, generic plugin registry, generic
  manifest export, scalar/offload plugins, or `tcrv.exec`.
- Changing default RVV emission readiness/manifest support for metadata-only
  selected paths.

## Technical Notes

- Current HEAD before implementation: `6762068 feat: add RVV smoke probe target export`.
- Existing selected-path and metadata validation patterns live in
  `lib/Target/RVV/RVVSmokeProbe.cpp` and `lib/Dialect/RVV/IR/RVVDialect.cpp`.
- The new executable claim is bounded to the explicit RVV extension dialect op
  plus target-local exporter and, if available, real `ssh rvv` compile/run
  evidence.
