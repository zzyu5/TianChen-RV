# Extension plugin artifact route registration template

## Goal

Complete one bounded extension-plugin artifact route registration template that
proves an enabled plugin can own proposal, selection, lowering-boundary,
emission-plan, and artifact/export route metadata through a reusable
plugin-target exporter bundle. The implementation should reduce future plugin
route wiring friction by making route metadata explicit enough for generic
target artifact export preflight to validate it before invoking a route-local
exporter.

## Background

The archived task
`.trellis/tasks/archive/2026-05/05-11-offload-runtime-plugin-boundary-template/`
completed the active offload runtime boundary template and descriptor export.
Current code already has a `PluginTargetArtifactExporterBundle` layer and
Toy/Offload route bundle registration. The remaining bottleneck is that route
ownership metadata is still partly implicit in route-local validators and
artifact text. This task should turn the route metadata into a reusable C++
registration contract consumed generically by target artifact export.

## Requirements

- Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Preserve `tcrv.exec` as execution organization only; do not add compute ops
  or high-level compute semantics.
- Extend the existing target artifact registration surface rather than adding
  Toy/offload/vendor-specific branches in generic core passes.
- Add or refine a reusable route descriptor/metadata contract that can record:
  route id, artifact kind, origin plugin, emission kind, runtime ABI kind/name,
  runtime glue role, handoff/selected-plan metadata requirements, explicit
  no-claim or conservative claim fields, and exporter/preflight callbacks.
- Make generic target artifact export preflight consume the registered metadata
  for at least one proof route. The preferred proof route is the offload
  runtime descriptor because it already carries typed runtime ABI and handoff
  selected-plan metadata; Toy metadata may also be adjusted if naturally
  adjacent.
- Preserve active producer and consumer evidence:
  plugin proposal/selection/lowering/emission metadata on the producer side,
  deterministic artifact/export route on the consumer side.
- Duplicate route registration, missing exporter callback, unknown route,
  stale runtime ABI metadata, and stale handoff/selected-plan metadata must
  fail closed with useful diagnostics.
- Artifact output must continue to state explicit non-runtime,
  non-correctness, non-performance claims. This task must not claim runtime
  correctness or performance without real hardware evidence.

## Acceptance Criteria

- A plugin-local or target-support-owned route descriptor registers route
  identity, artifact kind, origin plugin, emission kind, runtime ABI metadata,
  selected handoff metadata requirements, explicit no-claim fields, and exporter
  callback/preflight hook.
- `tcrv-translate --tcrv-export-target-artifact` and/or target artifact bundle
  export consumes the registered route descriptor generically.
- The offload descriptor route remains deterministic and emits no-runtime,
  no-correctness, no-hardware-execution, and no-performance claim fields.
- Existing Toy and offload plugin producer paths remain active through
  proposal, selected lowering boundary, emission plan, and artifact export.
- Negative coverage proves duplicate route registration, missing callback,
  unknown route, stale runtime ABI route metadata, and stale selected
  handoff/selected-plan metadata fail closed.
- Existing RVV, scalar, Toy, and offload artifact/export tests are not weakened.
- Trellis task context files validate before finishing/archive.

## Non-Goals

- No broad plugin framework rewrite.
- No dynamic shared-library plugin loading or external packaging.
- No new hardware execution, RVV runtime evidence, Sophgo runtime, IME, AME, or
  performance experiment.
- No Python compiler internals.
- No docs-only, helper-only, smoke-only, report-only, or metadata-only closeout
  without an active C++ producer and consumer route proof.

## Validation Plan

- Build focused targets for `tcrv-opt`, `tcrv-translate`, plugin tests, and
  target artifact export tests.
- Run focused lit/FileCheck tests for Toy/offload plugin route export and
  negative route diagnostics.
- Run C++ registry/preflight tests proving route descriptor metadata is
  registered and consumed generically.
- Run `git diff --check`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target
  check-tianchenrv -j2` if focused checks pass and the build directory is
  usable.
- Validate and archive the Trellis task only after the active C++ route proof is
  complete.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/offload-runtime-plugin.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Archived context read:
  `.trellis/tasks/archive/2026-05/05-11-offload-runtime-plugin-boundary-template/prd.md`.
- Current code already exposes:
  `TargetArtifactExporterRegistry`,
  `PluginTargetArtifactExporterRegistry`,
  `PluginTargetArtifactExporterBundle`,
  Toy metadata route registration, and offload descriptor route registration.
- Implementation should therefore refine the existing route descriptor contract
  and generic preflight, not duplicate route lists in `tcrv-translate` or add
  extension-specific generic branches.

## Completion Notes

- Added `TargetArtifactRouteMetadata` to the C++ target artifact registration
  API. A route can now register runtime ABI kind/name/glue metadata,
  selected-plan metadata requirements, and explicit conservative claim fields
  alongside its route id, artifact kind, owner, emission kind, exporter
  callback, runtime ABI role requirements, and route-local candidate validator.
- Generic target artifact preflight now consumes the route descriptor before
  invoking route-local validation. It rejects stale runtime ABI metadata and
  stale selected-plan/handoff metadata without adding Toy, offload, vendor,
  runtime, dtype, or target-specific branches.
- The Toy metadata route and the offload runtime descriptor route both register
  route descriptor metadata through their plugin-owned target exporter bundles.
  Offload additionally publishes registered no-claim fields into the generic
  bundle index.
- Existing artifact output remains deterministic and explicit about
  no-runtime/no-correctness/no-hardware-execution/no-performance claims. No
  runtime correctness, hardware execution, or performance evidence was claimed.
- `tcrv.exec` remained compute-free; no compiler core behavior was implemented
  in Python.

## Checks Run

- `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-target-artifact-export-test tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt -j2`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv .
  --filter='(offload-runtime-descriptor-artifact-route|offload-runtime-descriptor-bundle|toy-metadata-artifact-route|toy-metadata-artifact-runtime-abi-kind-fails|target-artifact-export-registry)'`
  from `artifacts/tmp/tianchenrv-build/test`: 5 focused lit tests passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-target-artifact-export-test
  tianchenrv-toy-extension-plugin-test
  tianchenrv-offload-extension-plugin-test tcrv-opt tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-toy-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-offload-extension-plugin-test`
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-extension-plugin-artifact-route-registration-template`
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  200/200 lit tests passed.
