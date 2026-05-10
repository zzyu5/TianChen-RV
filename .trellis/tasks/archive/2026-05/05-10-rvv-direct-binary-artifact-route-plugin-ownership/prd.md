# RVV direct binary artifact route plugin ownership

## Goal

Move the RVV direct binary microkernel source/header/object artifact-route
registration slice to an RVV-owned target-support contribution path. The same
RVV-owned route source of truth must drive `tcrv-translate` direct helper route
population and `TargetArtifactExport` registry integration for the supported
i32/i64 add/sub/mul direct microkernel families.

## Current Central Wiring Points

- `tools/tcrv-translate/tcrv-translate.cpp` currently has RVV-specific direct
  microkernel route registration code that iterates only the legacy i32 binary
  family registry for non-vadd family routes and separately hard-codes the
  legacy vadd source/header/object commands.
- `lib/Target/RVV/RVVMicrokernel.cpp` currently hand-registers each RVV direct
  binary source/header/object artifact route in
  `registerRVVMicrokernelTargetExporters`.
- `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` is the built-in
  Target-layer aggregation point. It may call RVV target-support registration,
  but it must not duplicate individual RVV family route facts.

## Requirements

- Add the smallest C++ target-support interface that exposes RVV direct binary
  microkernel route entries for source/header/object routes.
- Keep route ids, artifact kinds, runtime ABI metadata, direct helper flags,
  component-group metadata, and export behavior stable.
- Register RVV direct source/header/object artifact exporters by iterating the
  RVV-owned direct route contribution data rather than hand-writing each
  i32/i64 add/sub/mul route in the central registry code.
- Register `tcrv-translate` RVV direct helper commands by iterating the same
  RVV-owned route contribution data rather than enumerating individual RVV
  binary family route ids in the tool.
- Preserve existing legacy direct route names:
  `tcrv-export-rvv-microkernel-c`, `tcrv-export-rvv-microkernel-header`, and
  `tcrv-export-rvv-microkernel-object`.
- Preserve i32 family route names for vsub/vmul and i64 route names for
  add/sub/mul.
- Leave RVV+scalar dispatch route migration out of scope.

## Acceptance Criteria

- [x] `TargetArtifactExport` registry contains all existing RVV direct binary
  i32/i64 add/sub/mul source/header/object routes after RVV target-support
  registration.
- [x] Duplicate RVV direct route contribution fails through the existing
  generic registry duplicate-route diagnostics.
- [x] A registry that does not receive the RVV direct contribution does not
  silently provide RVV direct routes.
- [x] `tcrv-translate --help` still exposes representative i32 and i64 RVV
  direct helper routes through the RVV-owned route population path.
- [x] Focused source/header/object tests still prove behavior for at least one
  i32 family and one i64 family.
- [x] Existing RVV+scalar dispatch route behavior remains unchanged.

## Completion Notes

- Added `RVVMicrokernelDirectRouteManifestEntry` and
  `getRVVMicrokernelDirectRouteManifest()` in RVV target support as the direct
  binary source/header/object route contribution source of truth.
- Replaced hand-maintained RVV direct source/header/object exporter
  registration with iteration over the RVV-owned manifest.
- Replaced `tcrv-translate` per-family RVV direct helper enumeration with the
  same manifest-driven registration path.
- Preserved the legacy generic route ids
  `tcrv-export-rvv-microkernel-c`,
  `tcrv-export-rvv-microkernel-header`, and
  `tcrv-export-rvv-microkernel-object` as selected-family compatibility
  aliases, while family-specific routes fail closed on selected-family mismatch.
- Added registry-coherence coverage for explicit RVV contribution, missing
  contribution, duplicate rejection, and route uniqueness.
- Added i64 direct source/header/object coverage while preserving existing i32
  direct route behavior and RVV+scalar dispatch tests.
- Updated `.trellis/spec/lowering-runtime/emission-runtime-contract.md` with
  the durable target-local RVV direct route contribution boundary.
- No new `ssh rvv` evidence was required because generated RVV
  runtime-callable source/header/object semantics and runtime correctness claims
  were preserved.

## Non-goals

- No new RVV arithmetic families, vector shapes, SEW/LMUL coverage, runtime ABI
  semantics, correctness claims, or performance evidence.
- No Python compiler implementation.
- No migration of RVV+scalar dispatch composite route ownership in this round.
- No compute semantics added to `tcrv.exec`.
- No RVV-specific semantic branch in core target artifact routing.

## Minimal Validation

- `git diff --check`
- Build focused changed targets: `tcrv-translate`,
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-i32-binary-family-registry-test`
- Focused tests for `TargetArtifactExport`, `Target/RVVMicrokernel`, and the
  RVV microkernel e2e scripts as applicable
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- `python3 ./.trellis/scripts/task.py validate <archived-task-path>` after
  archive

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Current implementation already has a manifest-like route model for
  RVV+scalar dispatch in `RVVScalarDispatch`; this task applies the same shape
  only to the direct RVV binary microkernel route slice.
- No new `ssh rvv` evidence is required unless generated RVV runtime-callable
  source/header/object behavior changes. This round is intended to preserve
  generated behavior while changing route-registration ownership.
