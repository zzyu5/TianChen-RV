# Direct C Deleted-Route Shell Erasure

## Goal

Delete stale direct-C semantic exporter route shells for scalar microkernel and
RVV+scalar dispatch paths. These paths are already deleted/fail-closed in
behavior; this round removes their callable APIs, empty registration bundles,
dead CMake entries, and tests whose only purpose is preserving those old route
IDs as quarantine surfaces.

## Background

The current architecture requires executable artifact rebuild work to flow from
extension-family ops through the common EmitC route and then through C/C++
emission. Direct descriptor/family-record-to-C semantic exporters are deletion
campaign residue. Existing evidence at `6209e2b` shows:

- `include/TianChenRV/Target/Scalar/ScalarMicrokernel.h` and
  `lib/Target/Scalar/ScalarMicrokernel.cpp` expose scalar source/header/object
  exporter APIs that only return deleted-route errors, plus an empty
  plugin-owned target exporter bundle.
- `include/TianChenRV/Target/RVV/RVVScalarDispatch.h`,
  `include/TianChenRV/Target/RVVScalarDispatch.h`, and
  `lib/Target/RVV/RVVScalarDispatch.cpp` expose RVV+scalar dispatch
  target-exporter and translate-route registration APIs that only no-op.
- `test/Target/TargetArtifactExportTest.cpp` still directly tests these
  deleted-route shell registration surfaces.

## Scope

- Remove scalar microkernel direct source/header/object exporter front doors.
- Remove scalar microkernel empty target artifact exporter bundle registration.
- Remove RVV+scalar dispatch empty target artifact exporter and translate-route
  registration shells.
- Remove direct includes and CMake entries whose only purpose is building those
  shells.
- Rewrite focused target artifact tests so they validate generic registry
  behavior and built-in route absence without calling the deleted shell APIs.
- Keep generic target artifact registry mechanics, extension bundles, plugin
  registries, support-layer ABI contracts, extension-family ops, dialect
  verification, and non-semantic metadata/package artifact routes.

## Non-Goals

- Do not implement a new EmitC route, RVV lowering, scalar lowering, artifact
  emitter, runtime ABI, compatibility wrapper, legacy mode, helper exporter, or
  descriptor fallback.
- Do not add replacement direct-C exporters beside the deleted routes.
- Do not delete generic target artifact registry infrastructure or current
  metadata-only Toy/Template/TensorExtLite artifact routes.
- Do not expand RVV finite-family coverage or add tests for old direct-C route
  behavior.
- Do not make RVV runtime/correctness/performance claims.

## Acceptance Criteria

- `exportScalarMicrokernelC`, `exportScalarMicrokernelHeader`,
  `exportScalarMicrokernelObject`, and
  `registerScalarMicrokernelPluginTargetExporterBundle` are absent from active
  callable APIs and production registration surfaces.
- `registerRVVScalarDispatchTargetExporters`,
  `registerRVVScalarDispatchPluginTargetExporterBundle`,
  `registerRVVScalarDispatchTargetTranslateRoutes`, and the direct
  `RVVScalarDispatch` route shell header/source are absent from active callable
  APIs and production registration surfaces.
- Scalar microkernel direct route IDs and RVV+scalar dispatch direct route IDs
  are not registered by built-in target artifact exporters.
- Focused tests no longer call deleted shell registration functions merely to
  assert that they no-op.
- Any remaining mentions of direct C semantic exporters or deleted direct route
  IDs are clearly historical/deletion-campaign documentation, unsupported
  diagnostics, or generic route-absence checks rather than compatibility
  surfaces.
- If deletion exposes a missing EmitC-route implementation gap, report it
  truthfully and do not restore the old route shell.

## Validation Plan

- Run focused ref-scans for the deleted scalar and RVV+scalar dispatch symbols
  and direct route IDs named in the brief.
- Build/run the affected target artifact C++ test target if available.
- Run any touched scalar/RVV plugin or support test target needed by CMake link
  changes.
- Run `git diff --check`, `git diff --cached --check`, and Trellis task
  validation.
