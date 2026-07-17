# first-class target header artifact route

## Goal

Make the bounded RVV+scalar i32-vadd runtime-callable dispatch ABI header
available through the generic target artifact registry front door, alongside
the existing dispatch C source and relocatable object routes.

## What I Already Know

* The current direct header command is
  `tcrv-export-rvv-scalar-i32-vadd-dispatch-header`.
* The source and object routes already participate in
  `TargetArtifactExporterRegistry` through RVV+scalar composite exporters.
* The new header route must not make `tcrv-export-target-source-artifact`
  ambiguous with C source, and must not make `tcrv-export-target-artifact`
  select the header instead of the object.
* Header emission must still be derived from the exec-IR-backed
  `DispatchABIPlan` and target-owned validations.

## Requirements

* Add a generic `exportTargetHeaderArtifact` C++ target artifact export API.
* Add a distinct `runtime-callable-c-header` artifact kind.
* Register the RVV+scalar dispatch header as a target-owned composite route
  using the same candidate match and direct header exporter.
* Add a `tcrv-translate` generic header command.
* Preserve existing direct source/header/object/self-check commands.
* Update focused durable lowering-runtime/testing specs for first-class header
  artifact selection.
* Add lit/FileCheck coverage for the new generic header command.
* Add C++ registry/export tests that prove header selection is registry-backed
  and fail-closed on an existing dispatch IR boundary error.

## Acceptance Criteria

* `git diff --check` passes.
* `cmake --build build --target tcrv-opt tcrv-translate -j2` passes.
* `cmake --build build --target check-tianchenrv -- -j2` passes.
* `build/bin/tianchenrv-target-artifact-export-test` passes if not already
  covered by `check-tianchenrv`.

## Out Of Scope

* No tcrv.exec ODS semantic changes unless strictly required.
* No Python implementation of compiler registry, exporter, IR, selection, or
  ABI decisions.
* No promotion of self-check object/source helpers into the generic artifact
  front door.
* No runtime, correctness, or performance claim requiring `ssh rvv`.
