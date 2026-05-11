# Extension bundle registration frontdoor

## Goal

Add one bounded C++ extension-bundle registration frontdoor so a built-in
extension can register its plugin hooks, dialect dependencies, lowering-boundary
support, and target artifact route metadata through one reusable integration
surface consumed by public tools and registries. The proof extension should be
Toy metadata unless code inspection shows Offload is the lower-risk active
producer/consumer proof. RVV executable route metadata must remain a regression
consumer.

## Background

The archived `extension-plugin-artifact-route-registration-template` task added
`TargetArtifactRouteMetadata` and generic export preflight for Toy/Offload route
metadata. The archived `rvv-executable-artifact-route-metadata-registration`
task adopted the same metadata contract for one bounded executable RVV route.
The remaining integration friction is that enabling future built-in extensions
can still require hand-edited tool, registry, dialect, and target exporter
wiring. This task introduces a reusable frontdoor for built-in extension bundle
registration without moving extension semantics into core passes.

## Requirements

- Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.
- Preserve `tcrv.exec` as execution organization only; do not add compute ops or
  high-level tensor/tile semantics.
- Add or refactor a C++ bundle/registration contract using existing project
  naming conventions where possible.
- The bundle contract must cover plugin identity, required dialect/init
  dependency when applicable, proposal/legality/lowering support hooks, target
  artifact route metadata registration, and fail-closed diagnostics for
  duplicate or missing fields.
- Generic tool or registry setup must consume the bundle through a generic list
  or registration call. Do not add new Toy, Offload, RVV, vendor, dtype, shape,
  runtime, or microarchitecture semantic branches to generic core passes.
- The proof extension must include an active producer and consumer:
  plugin/registry behavior remains observable through `tcrv-opt` or relevant
  C++ registry tests, and target artifact export / `tcrv-translate` observes
  route metadata through the bundle frontdoor.
- The chosen proof route must continue to emit deterministic artifact output and
  validate runtime ABI / handoff metadata through `TargetArtifactRouteMetadata`.
- Preserve existing Offload and RVV route metadata behavior; compatibility
  changes must be mechanical and must not become a route rewrite.

## Acceptance Criteria

- A bounded proof extension, preferably Toy metadata, registers through the new
  bundle frontdoor rather than only through scattered direct calls.
- The bundle descriptor or registration function records enough data to register
  plugin behavior and target artifact route metadata through one reusable
  extension-owned or target-support-owned entry point.
- Generic tool/registry setup uses a bundle list/frontdoor and does not grow
  concrete extension semantic branches in shared pass/export logic.
- `TargetArtifactRouteMetadata` remains the generic route metadata validator for
  the proof route, including runtime ABI and selected-plan/handoff metadata
  checks where applicable.
- Focused C++ tests cover successful bundle registration and fail-closed
  duplicate bundle/plugin id, duplicate route registration through a bundle,
  missing route metadata for a bundle that claims export support, and stale
  runtime ABI or handoff metadata.
- Lit/FileCheck coverage shows the proof extension pipeline/export behavior
  still works through the bundle frontdoor.
- Existing Toy, Offload, and RVV target artifact route metadata tests are not
  weakened.

## Non-Goals

- No dynamic shared-library plugin loading or external packaging.
- No broad plugin framework rewrite beyond one built-in bundle frontdoor and one
  proof extension migration.
- No new RVV kernel, hardware run, performance claim, Sophgo runtime, IME, AME,
  or AME-like implementation.
- No compute semantics in `tcrv.exec`.
- No docs-only, helper-only, smoke-only, report-only, or metadata-only closeout
  without active C++ producer/consumer behavior.

## Validation Plan

- Build focused changed targets, including relevant plugin tests,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`.
- Run focused C++ tests for the proof extension, target artifact export, and
  any touched Offload/RVV route metadata behavior.
- Run focused lit/FileCheck coverage for proof extension export behavior and
  route metadata fail-closed diagnostics.
- Run `git diff --check`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` after focused checks pass if the build directory remains usable.
- Validate Trellis context with `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-extension-bundle-registration-frontdoor` before
  finish/archive.

## Technical Notes

- Initial repository state: root `/home/kingdom/phdworks/TianchenRV`, branch
  `main`, clean working tree, HEAD `5fb7ace feat(target): register RVV
  executable route metadata`.
- No current task existed at takeover; this task was created and started as
  `.trellis/tasks/05-11-extension-bundle-registration-frontdoor`.
- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Archived PRDs read for context:
  `.trellis/tasks/archive/2026-05/05-11-extension-plugin-artifact-route-registration-template/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-11-rvv-executable-artifact-route-metadata-registration/prd.md`.

## Definition Of Done

The task is done when one proof extension registers plugin behavior and target
artifact route metadata through the bundle frontdoor, public tools or registries
consume that frontdoor generically, focused C++ and lit tests cover success and
fail-closed cases, Offload/RVV route metadata regressions continue to pass,
Trellis context validates, the task is finished/archived, and one coherent
commit records the module.

If unfinished, leave the task open and record the exact continuation point:
bundle descriptor API, proof extension migration, generic registry consumer,
tool registration hook, target route metadata hookup, duplicate/missing-field
diagnostic, lit/C++ test integration, or compatibility with existing RVV/Offload
routes.

## Completion Notes

- Added a C++ `ExtensionBundle` / `ExtensionBundleRegistry` frontdoor on top of
  the existing plugin and target artifact registries. A bundle now records
  plugin identity, plugin registration callback, dialect/init dependency names,
  lowering-boundary support names, plugin-owned target exporter bundle callback,
  and expected target artifact route metadata requirements.
- Reworked built-in target artifact registration so RVV, Offload, Toy, and
  Scalar are registered through built-in extension bundles. Toy is the bounded
  proof extension: its plugin behavior, dialect/lowering-boundary dependency,
  target artifact route, and `TargetArtifactRouteMetadata` requirement all flow
  through the bundle frontdoor.
- Updated `tcrv-opt` and `tcrv-translate` to populate built-in plugin registries
  through the bundle plugin frontdoor before registering dialects, planning
  passes, and target artifact exporters.
- Preserved the existing plugin-owned target exporter bundle layer. The new
  frontdoor aggregates those extension-owned or target-support-owned callbacks
  and checks declared route metadata after enabled-plugin exporter population,
  instead of adding Toy/Offload/RVV semantic branches to generic passes.
- Preserved RVV executable route metadata as a regression consumer by requiring
  the bounded RVV `i32-vsub` source route to publish registered
  `TargetArtifactRouteMetadata` through the RVV extension bundle.
- Added focused C++ coverage for successful built-in bundle registration,
  Toy route metadata consumption, stale runtime ABI and stale selected-plan
  metadata rejection, duplicate bundle id, duplicate plugin id, duplicate route
  registration through a bundle, missing route metadata declaration, and a route
  registered without required `TargetArtifactRouteMetadata`.
- No runtime correctness, hardware execution, or performance evidence was
  claimed.

## Checks Run

- `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-toy-extension-plugin-test tianchenrv-offload-extension-plugin-test
  tianchenrv-rvv-extension-plugin-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-toy-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-offload-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Plugin/toy-extension-plugin.test
  Target/ToyMetadataArtifact/toy-metadata-artifact-route.mlir
  Target/ToyMetadataArtifact/toy-metadata-artifact-runtime-abi-kind-fails.mlir
  Target/ArtifactExport/target-artifact-export-registry.test
  Target/ArtifactExport/offload-runtime-descriptor-artifact-route.test
  Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-i32m2-generic-route.mlir`
  from `artifacts/tmp/tianchenrv-build/test`: 6 focused lit tests passed.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-extension-bundle-registration-frontdoor`
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  200/200 lit tests passed.

Self-repair performed:

- Initial focused build exposed that the bundle frontdoor needed a distinct
  function pointer type for registering plugin-owned target exporter bundles
  into `PluginTargetArtifactExporterRegistry`, separate from registering
  concrete exporters into `TargetArtifactExporterRegistry`; the type was split.
- The next focused build exposed an unqualified nested RVV route manifest type
  in the built-in bundle registration code; it was qualified as
  `rvv::RVVMicrokernelDirectRouteManifestEntry`.
- A source-tree lit invocation failed because the build-site config was not
  loaded (`tianchenrv_obj_root` missing); the same focused tests passed when run
  from `artifacts/tmp/tianchenrv-build/test`.
- `clang-format` was not available in this environment, including common
  versioned names; `git diff --check` passed and formatting was kept consistent
  manually.
