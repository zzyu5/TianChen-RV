# Composite dispatch route metadata authority

## Goal

Make registry-owned route metadata available for composite target artifact
routes, then use it for the bounded RVV+scalar dispatch source/header/object
routes so the compiler-emitted bundle index carries conservative no-claim
fields from the target exporter registry instead of leaving Python evidence
helpers or route names to infer the claim boundary.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state for this round is clean on `main` at
  `91427cd feat(rvv): enforce dispatch component body authority`.
- No `.trellis/.current-task` existed at session start, so this is a new
  Hermes fallback continuation task rather than a resumed local Trellis task.
- Latest supervisor audit/input read:
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0099-20260511T050444Z/repo_audit.md`
  and
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0099-20260511T050444Z/review_input.md`.
- The latest completed task closed dispatch component body authority for the
  bounded `i32-vmul` RVV+scalar dispatch path.
- `TargetArtifactRouteMetadata` currently belongs to standalone
  `TargetArtifactExporter` routes. It is validated at standalone route
  registration, used by generic candidate preflight, and copied into bundle
  records as route claim fields.
- `TargetArtifactCompositeExporter` already owns route id, artifact kind,
  owner, runtime ABI kind/name, runtime ABI parameters, direct-helper status,
  component group, external ABI name, and route-local candidate validation, but
  it has no registry-owned route metadata surface.
- `appendCompositeBundleRecords` currently copies component selected-plan
  metadata and runtime ABI fields into the bundle index, but it does not copy
  route claim fields because composite routes have no route metadata object.
- The built-in scalar extension bundle declares metadata requirements for
  scalar source routes only. RVV+scalar dispatch source/header/object routes
  are plugin-owned composite routes and are not currently enforceable by the
  extension-bundle route metadata requirement contract.

## Requirements

- Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python remains tooling/test runner only.
- Preserve TianChen-RV boundaries: `tcrv.exec` stays compute-free; RVV/scalar
  dispatch behavior remains target/export-owned and plugin-local.
- Add a registry-owned route metadata surface for composite target artifact
  exporters without moving RVV/scalar family semantics into generic registry
  code.
- Validate composite route metadata shape at registration with the same bounded
  text, duplicate-field, runtime ABI descriptor, selected-plan requirement,
  and route-claim-field checks used by standalone routes.
- Allow extension-bundle target artifact route metadata requirements to resolve
  either standalone exporters or composite exporters by route id and artifact
  kind, and to require registered route metadata on either kind.
- Add RVV+scalar dispatch composite route metadata for source/header/object
  route registrations. It must include conservative claim fields:
  `compile_export_claim = compiler-artifact-only`,
  `runtime_correctness_claim = none`, `hardware_execution_claim = none`, and
  `performance_claim = none`.
- Preserve existing dispatch runtime ABI kind/name, component group, external
  ABI name, direct-helper flag, route-local candidate validation, and runtime
  ABI parameter derivation callbacks.
- Copy composite route claim fields into target artifact bundle records so the
  bundle index exposes the compiler-owned no-claim boundary for dispatch
  source/header/object records.
- Update focused C++ and lit/FileCheck coverage for the new composite metadata
  authority. Tests must prove the dispatch composite routes carry route
  metadata, the scalar extension bundle requires those dispatch route metadata
  entries, malformed/missing composite metadata fails closed, and the dispatch
  bundle index records the conservative route claim fields.

## Non-Goals

- No new RVV runtime correctness, hardware execution, throughput, latency, or
  performance claim.
- No fresh `ssh rvv` requirement unless this task changes runtime behavior and
  the final report claims runtime correctness.
- No broad benchmark/test matrix.
- No new high-level tensor/tile IR and no compute semantics in `tcrv.exec`.
- No Python implementation of compiler internals, dialects, passes, plugin
  registry, capability model, lowering, emission, or target artifact export.
- No generic hard-coded RVV/scalar branches in shared target artifact registry
  code.
- No change to direct self-check helper route semantics; this task targets
  generic source/header/object composite routes and bundle records.

## Acceptance Criteria

- `TargetArtifactCompositeExporter` can carry `TargetArtifactRouteMetadata`
  and exposes it through a getter.
- Composite exporter registration rejects malformed route metadata and
  duplicate route-claim fields before any artifact export.
- Extension-bundle route metadata requirements can validate composite routes as
  well as standalone routes, including artifact kind and required metadata.
- The scalar extension bundle declares metadata requirements for finite scalar
  source routes and RVV+scalar dispatch source/header/object composite routes.
- Valid RVV+scalar dispatch composite source/header/object route registrations
  retain their existing matcher, exporter, candidate preflight, runtime ABI
  callback, direct-helper, component group, and external ABI behavior.
- C++ target artifact export tests cover at least one dispatch composite route
  metadata shape, built-in bundle metadata requirements, and malformed
  composite metadata fail-closed diagnostics.
- A focused target artifact bundle lit/FileCheck test observes the
  conservative route claim fields on RVV+scalar dispatch source/header/object
  records and continues to reject runtime/performance wording in dry-run
  bundle output.
- `git diff --check` passes.
- Focused build/test targets and relevant lit tests pass.
- Full `check-tianchenrv` is run if the local build remains usable.

## Validation Plan

- Build focused targets:
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`.
- Run `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`.
- Run focused lit/FileCheck for affected RVV+scalar dispatch bundle output,
  especially `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test` or the
  narrower target bundle route test updated in this task.
- Run `git diff --check`.
- Run Trellis task validation for this task path.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if the build tree remains usable.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Primary implementation surfaces:
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.
- Primary lit surface:
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test` or a narrower
  `test/Target/TargetArtifactBundleExport/*.mlir` dispatch bundle fixture.

## Definition Of Done

The task is done when composite dispatch source/header/object routes have
registry-owned route metadata, extension-bundle validation requires that
metadata, target artifact bundle indexes preserve the conservative claim fields
for dispatch composite records, focused tests pass, the Trellis task is
validated/finished/archived, and one coherent commit records the work.

If unfinished, leave the task open and record the exact continuation point:
PRD/context repair, composite metadata API, registration validation, extension
bundle validation, RVV+scalar dispatch metadata, bundle index propagation,
C++ tests, lit checks, full check, Trellis validation, archive, or commit.

## Completion Notes

- Added `TargetArtifactRouteMetadata` ownership to
  `TargetArtifactCompositeExporter`, including registration-time metadata
  shape validation and a public getter.
- Extended extension-bundle route metadata requirements so they can validate
  either standalone or composite target artifact routes. Requirements may now
  declare additional required enabled plugins, which keeps scalar-owned
  RVV+scalar dispatch metadata conditional on `rvv-plugin`.
- Made RVV+scalar dispatch source/header/object composite routes carry
  registry-owned conservative claim fields:
  `compile_export_claim = compiler-artifact-only`,
  `runtime_correctness_claim = none`, `hardware_execution_claim = none`, and
  `performance_claim = none`.
- Propagated composite route claim fields into target artifact bundle records
  so the generated bundle index records the no-claim boundary for dispatch
  source/header/object artifacts.
- Updated C++ target artifact export tests for composite metadata validation,
  conditional bundle metadata requirements, malformed composite metadata
  rejection, and dispatch composite claim fields.
- Updated focused dispatch bundle lit/FileCheck coverage to observe the new
  route claim fields in the compiler-emitted bundle index.
- Updated the lowering/runtime spec to record composite route metadata and
  conditional route metadata requirement behavior.
- No fresh `ssh rvv` evidence was produced because this task changed compiler
  registry/bundle metadata authority only and did not make a new RVV runtime
  correctness, hardware execution, throughput, latency, or performance claim.
