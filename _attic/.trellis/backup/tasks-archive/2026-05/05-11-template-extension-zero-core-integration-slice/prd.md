# Template extension zero-core integration slice

## Goal

Add one deliberately small built-in `Template` extension that proves a future
extension can integrate through the existing `ExtensionBundle` frontdoor with
localized plugin, dialect, lowering-boundary, target route metadata, and
deterministic manifest code. The slice is non-executable and metadata-only, but
it must still exercise an active compiler path from capability discovery through
plugin proposal, selected `tcrv.exec.variant`, plugin-owned lowering boundary,
supported manifest route metadata, and target artifact export.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial worktree is clean at `eeb48b0 feat(target): add extension bundle registration frontdoor`.
* No `.trellis/.current-task` was present at takeover; this task was created as
  `.trellis/tasks/05-11-template-extension-zero-core-integration-slice`.
* The archived `extension-bundle-registration-frontdoor` task completed the
  generic `ExtensionBundle` / `ExtensionBundleRegistry` frontdoor and migrated
  Toy, Offload, RVV, and Scalar bundle registration.
* Existing Toy code is useful as a structural template, but this task must add
  a distinct future-extension proof rather than simply rename or weaken Toy.

## Requirements

* Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
* Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.
* Preserve `tcrv.exec` as execution/capability/variant organization only; do
  not add tensor/tile compute semantics or high-level compute ops.
* Add a new built-in extension named `Template`, with plugin id
  `template-plugin`, dialect namespace `tcrv_template`, and capability id
  `template.extension`.
* The Template plugin must own capability interpretation, proposal metadata,
  legality checks, cost/preference metadata, emission readiness/plan metadata,
  selected lowering-boundary materialization, and selected-boundary validation.
* The Template target exporter must own deterministic manifest validation and
  artifact output, including route metadata and explicit no runtime,
  correctness, hardware execution, or performance claims.
* Register the Template extension through `ExtensionBundle`; public tool and
  registry setup must consume it through the existing bundle frontdoor.
* Generic core passes and generic target artifact routing must not gain
  Template-specific semantic branches.
* Preserve existing Toy, Offload, RVV, and Scalar behavior.

## Acceptance Criteria

* A kernel with an available `template.extension` capability and required
  bounded capability properties can run through the public execution planning
  pipeline and produce a selected Template variant.
* The selected Template path materializes one `tcrv_template.lowering_boundary`
  with plugin-owned metadata that matches the selected variant and capability.
* Template emission planning records a supported non-executable manifest route
  with registered `TargetArtifactRouteMetadata`.
* `tcrv-translate --tcrv-export-target-artifact` exports a deterministic
  Template manifest that explicitly says it is a compiler handoff/template
  artifact and carries no runtime correctness, hardware execution, or
  performance claim.
* Focused C++ tests cover bundle registration, plugin proposal/selection
  behavior, route metadata registration, artifact export success, missing
  Template capability, stale runtime ABI/handoff metadata, duplicate extension
  bundle/plugin id, missing route metadata for export support, and unknown
  route.
* Focused lit/FileCheck coverage shows the Template path through `tcrv-opt`
  and `tcrv-translate`, including proposal/selection/lowering-boundary metadata
  and deterministic artifact output.
* The diff demonstrates low integration friction: new Template-owned files plus
  bundle/CMake/test registration are expected; broad generic core pass changes
  require explicit justification.

## Non-Goals

* No real tensor/tile IR or high-level compute dialect.
* No RVV kernel, RVV hardware evidence, performance work, Sophgo runtime, IME,
  AME, or hardware execution.
* No dynamic shared-library plugin loading or external packaging.
* No broad plugin framework rewrite beyond the Template extension integration.
* No docs-only, helper-only, smoke-only, report-only, or metadata-only closeout
  without an active C++ producer/consumer path.

## Validation Plan

* Build focused changed targets:
  `tianchenrv-template-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and
  `tcrv-translate`.
* Run the Template C++ plugin test and target artifact export C++ test.
* Run focused lit/FileCheck coverage for Template plugin planning and Template
  manifest artifact export, plus affected Toy/Offload/RVV route metadata
  regressions.
* Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-template-extension-zero-core-integration-slice`.
* Run `git diff --check`.
* Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if focused checks pass and the existing build directory remains usable.

## Technical Notes

* Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/extension-plugins/future-plugins.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior context read:
  `.trellis/tasks/archive/2026-05/05-11-extension-bundle-registration-frontdoor/prd.md`.
* Core integration points inspected:
  `ExtensionPlugin`, `ExtensionBundle`, `TargetArtifactExport`, built-in bundle
  registration, `tcrv-opt`, `tcrv-translate`, and existing Toy extension
  plugin/dialect/target/test surfaces.

## Definition Of Done

The task is done when the Template extension has a working bundle-registered
compiler path, focused C++ and lit tests pass, Trellis context validates, the
task is finished/archived, and one coherent commit records the module. If
unfinished, leave this task open and record the exact continuation point:
PRD boundary, Template plugin API, bundle descriptor, tool registration,
lowering boundary/handoff marker, target route metadata, artifact exporter,
fail-closed diagnostics, lit/C++ test integration, or compatibility with
existing bundles.

## Completion Notes

* Added a new built-in Template extension with `template-plugin`,
  `template.extension`, `tcrv_template.lowering_boundary`, and a
  target-owned `template-extension-zero-core-manifest` route.
* The Template path is active C++ compiler behavior:
  capability properties are checked by the plugin, variant proposal and
  selected static variant materialization flow through the existing registry,
  the plugin materializes and validates its lowering boundary, emission
  planning records route metadata, and `tcrv-translate
  --tcrv-export-target-artifact` emits a deterministic manifest.
* The manifest explicitly describes a compiler handoff/template artifact and
  records no runtime execution, no hardware execution, no correctness claim,
  and no performance claim.
* Registered the Template plugin and target exporter through the existing
  built-in `ExtensionBundle` frontdoor. Public tools continue to populate
  plugin and target artifact registries through the generic bundle frontdoor.
* Generic `tcrv.exec`, shared planning passes, generic target artifact routing,
  `tcrv-opt`, and `tcrv-translate` did not gain Template-specific semantic
  branches. The only generic-test update was the expected built-in plugin and
  route count after adding one new built-in bundle.
* Reviewed the Trellis spec-update workflow. No `.trellis/spec/` change was
  needed because this slice instantiates existing plugin-protocol,
  lowering-runtime, target-artifact, and testing contracts rather than changing
  those contracts.

## Checks Run

* `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-template-extension-plugin-test
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-template-extension-plugin-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Plugin/template-extension-plugin.test
  Target/TemplateMetadataArtifact/template-metadata-artifact-route.mlir
  Target/TemplateMetadataArtifact/template-metadata-artifact-runtime-abi-kind-fails.mlir
  Target/TemplateMetadataArtifact/template-metadata-artifact-handoff-kind-fails.mlir
  Target/TemplateMetadataArtifact/template-metadata-artifact-unknown-route.mlir
  Plugin/toy-extension-plugin.test
  Target/ToyMetadataArtifact/toy-metadata-artifact-route.mlir
  Target/ToyMetadataArtifact/toy-metadata-artifact-runtime-abi-kind-fails.mlir
  Target/ArtifactExport/target-artifact-export-registry.test
  Target/ArtifactExport/offload-runtime-descriptor-artifact-route.test
  Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-i32m2-generic-route.mlir`
  from `artifacts/tmp/tianchenrv-build/test`: 11 focused lit tests passed.
* `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-template-extension-zero-core-integration-slice`
* `git diff --check`
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  first run found one stale RVV built-in plugin count assertion; after updating
  it for the new Template plugin, the rerun passed 205/205 lit tests.

Self-repair performed:

* Updated the RVV plugin C++ test's built-in registration assertion from four
  plugins to five plugins so it recognizes the new `template-plugin` lifetime
  alongside RVV, Offload, Scalar, and Toy.
