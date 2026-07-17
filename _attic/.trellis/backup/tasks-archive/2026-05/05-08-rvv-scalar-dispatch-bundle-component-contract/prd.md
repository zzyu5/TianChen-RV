# rvv scalar dispatch bundle component contract

## Goal

Make the RVV+scalar dispatch target artifact bundle index expose a compiler-owned, typed component contract for the external ABI group. External consumers should select the generated dispatch source, header, and relocatable object from explicit bundle metadata rather than route-name or file-name heuristics.

## Requirements

- Extend the C++ target artifact export layer so bundle records can carry explicit component group, component role, external ABI name, and runtime ABI metadata for dispatch-capable artifacts.
- Generate the RVV+scalar dispatch source/header/object grouping contract from target exporters or generic bundle/index builder state, not from Python tooling.
- Validate the group before emitting a complete bundle index: source/header/object roles must be present once, role names must not duplicate inside the same external ABI group, runtime ABI identity must be present and consistent, and no incomplete complete bundle should be written.
- Keep RVV/scalar-specific facts inside target exporter registration/dispatch exporter code. Keep generic grouping validation inside target artifact support.
- Update `scripts/rvv_scalar_dispatch_e2e.py` only as an index consumer. It must consume the explicit role/group fields when selecting source/header/object.
- Preserve existing source-built and bundle-object e2e behavior and runtime guard coverage.

## Out Of Scope

- No arbitrary kernels, performance work, new RVV dialect feature, dynamic runtime integration, or broad lowering claim.
- No Python implementation of compiler IR, target artifact semantics, plugin registry, capability model, variant selection, lowering, emission, runtime ABI, or compiler decisions.
- No generated artifacts or remote logs committed.

## Acceptance Criteria

- Local C++/lit tests prove RVV+scalar dispatch bundle index emits explicit group/role metadata for source/header/object under one external ABI group.
- Local tests prove incoherent bundle component grouping is rejected or diagnosed before a complete index is emitted.
- Dry-run lit wrapper proves the Python script consumes the explicit index fields rather than only route/path names.
- Required checks run: `git diff --check`, Python compile/self-test, `tcrv-opt`/`tcrv-translate` build, `check-tianchenrv`.
- If bundle index fields consumed by the runner change, run the `ssh rvv` bundle e2e and report the artifact directory; otherwise report why no new RVV runtime claim is made.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`, `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, `.trellis/spec/testing/mlir-testing-contract.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`.
- Existing C++ bundle schema lives in `include/TianChenRV/Target/TargetArtifactExport.h` and `lib/Target/TargetArtifactExport.cpp`.
- Existing dispatch composite exporters live in `lib/Target/Builtin/RVVScalarDispatch.cpp`.
- Existing Python bundle consumer is `scripts/rvv_scalar_dispatch_e2e.py`.
