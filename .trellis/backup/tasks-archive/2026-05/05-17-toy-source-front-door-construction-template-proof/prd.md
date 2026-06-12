# Toy Source-Front-Door Construction Template Proof

## Goal

Add one bounded positive Toy source front door that exercises the same common
`tcrv-source-artifact-front-door-pipeline` shape as the promoted RVV source
front door:

```text
Toy-owned source recognition
  -> Toy selected variant / selected boundary materialization
  -> Toy-owned EmitC route mapping
  -> materialized EmitC route evidence
  -> existing Toy target header artifact export validation
```

This proves the common `SourceFrontDoorPassRegistration` and source-artifact
front-door pipeline are reusable extension-family construction-template
surfaces, not RVV-only plumbing.

## Current Repository Facts

- HEAD `0810dfe` promoted the bounded RVV vector i32 add route to a production
  source front door and left the old source-seed public pipeline as a deleted
  option negative.
- The common plugin registry already validates and collects
  `SourceFrontDoorPassRegistration` objects from enabled plugins, and
  `tcrv-opt` registers those passes plus
  `--tcrv-source-artifact-front-door-pipeline`.
- `buildSourceArtifactFrontDoorPipeline` currently runs collected source
  front-door passes, then legality/capability/emission-plan/coherence checks.
  It does not select Toy or RVV semantics itself.
- The Toy plugin already owns `toy.template` capability metadata, Toy variant
  legality, selected `tcrv_toy.compute_skeleton` boundary materialization,
  Toy `TCRVEmitCLowerableRoute` construction, runtime ABI metadata, and the
  existing Toy target header artifact route.
- The existing Toy target header fixture starts from an already materialized
  `tcrv.exec.kernel` capability anchor and uses
  `--tcrv-execution-planning-pipeline`; there is not yet a positive Toy source
  fixture that enters through the common source-artifact front-door pipeline.
- Existing stale Toy seed coverage must remain fail-closed and must not become
  a compatibility entry point.

## Requirements

- Add a Toy-owned source front-door pass registered through
  `SourceFrontDoorPassRegistration`.
- The public pass argument must be production-scoped source-front-door wording,
  not `seed` wording or a compatibility alias.
- The accepted Toy source shape must be intentionally bounded and plugin-owned.
  It may be a small Toy construction-template marker/source function, but the
  source recognition and all semantic materialization must live in Toy plugin
  code rather than common/core pipeline code.
- The pass must materialize a Toy execution surface that the existing Toy
  plugin path consumes:
  - `tcrv.exec.kernel`;
  - available `toy.template` capability with preserved ABI/handoff properties;
  - selected Toy variant with `origin = "toy-plugin"`;
  - explicit Toy extension-family boundary op
    `tcrv_toy.compute_skeleton`;
  - Toy runtime ABI and emission-plan provenance;
  - Toy EmitC route metadata and source-op provenance.
- The positive source fixture must pass through
  `--tcrv-source-artifact-front-door-pipeline` and then reach the existing
  Toy target header artifact exporter.
- Negative coverage must fail closed for disabled built-in plugins, stale or
  deleted Toy seed entry points, stale Toy source-front-door registration,
  missing Toy boundary provenance, and attempts to satisfy the common pipeline
  through RVV-only assumptions.
- Focused C++ tests must prove RVV and Toy source front doors are collected
  through the same common plugin registry interface without duplicate or stale
  registrations.

## Acceptance Criteria

- [ ] The Toy plugin exposes exactly one bounded Toy source front-door pass
      registration through the common registry when built-in plugins are
      enabled.
- [ ] Registry C++ tests prove enabled Toy and RVV plugins both contribute
      source front-door registrations through the same
      `collectSourceFrontDoorPasses` path, and disabled or duplicate/stale
      registrations still fail closed.
- [ ] A positive Toy source fixture enters through
      `--tcrv-source-artifact-front-door-pipeline` and produces selected Toy
      IR containing `origin = "toy-plugin"`,
      `tcrv_toy.compute_skeleton`, selected variant linkage, required
      capability metadata, runtime ABI ownership metadata, supported emission
      plan metadata, and Toy EmitC route/source-op provenance.
- [ ] The same Toy source fixture can be piped to
      `tcrv-translate --tcrv-export-target-header-artifact`, producing the
      existing Toy materialized EmitC header evidence and no descriptor,
      direct-C, source-export, RVV intrinsic, or RVV-only residue.
- [ ] Disabled built-in plugins keep the Toy source front-door pass unavailable
      as a public option, and the common source-artifact front-door pipeline
      fails closed with an empty plugin registry.
- [ ] Deleted/stale Toy seed public entry points remain unavailable and stale
      `tcrv_toy.lowering_seed` metadata alone does not create a Toy selected
      route.
- [ ] Negative Toy source-front-door coverage rejects stale pre-materialized
      `tcrv.exec` or `tcrv_toy` selected-boundary residue, missing Toy source
      marker/provenance, and malformed Toy source markers before artifact
      output.
- [ ] Targeted scans over changed common/plugin/tool/test surfaces show no
      `SourceSeed` public API resurrection, no descriptor route authority, no
      direct C semantic exporter/source-export route, and no core semantic
      branch on Toy or RVV semantics.

## Out Of Scope

- No new RVV finite-family coverage, RVV source-front-door variants, generic
  RVV lowering, TensorExt/IME implementation, descriptor adapters, direct C
  semantic exporters, source-export routes, compatibility wrappers, or legacy
  source-seed aliases.
- No Python compiler-core behavior.
- No core/common Toy/RVV semantic branch. Common code may collect and run
  registered source front-door passes, but source semantics remain
  plugin-owned.
- No Toy runtime, correctness, performance, object, bundle, or hardware claim.
  Toy remains a construction-template proof that validates the extension
  plugin workflow through the existing header artifact route.
- No docs-only or template-only completion. The positive path must be
  executable through `tcrv-opt` and the target header exporter.

## Minimal Evidence

- `git diff --check`.
- Focused build/checks for touched plugin, registry, transform, tool, and
  target test binaries if available.
- Focused C++ tests:
  - Toy plugin source-front-door registration;
  - registry collection of RVV and Toy front doors through the same common
    interface;
  - duplicate/stale registration rejection if touched.
- Focused lit/FileCheck:
  - positive Toy source front door through
    `--tcrv-source-artifact-front-door-pipeline`;
  - Toy source front door piped to
    `--tcrv-export-target-header-artifact`;
  - disabled built-in plugin negative;
  - deleted/stale Toy seed entry-point negative;
  - malformed/missing Toy source provenance negative;
  - RVV source-front-door regression if common registration/pipeline code is
    touched.
- Targeted scans over changed common/plugin/tool/test surfaces for:
  `SourceSeed`, `source-seed`, `descriptor`, `source-export`, `direct-C`,
  and core semantic branches on Toy/RVV.
- `check-tianchenrv` if practical after focused checks pass.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous PRD read:
  `.trellis/tasks/archive/2026-05/05-17-05-17-rvv-vector-source-front-door-promotion/prd.md`.
- Initial code and test facts inspected:
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`,
  `lib/Transforms/ExecutionPlanningPipeline.cpp`,
  `tools/tcrv-opt/tcrv-opt.cpp`,
  `include/TianChenRV/Plugin/Toy/ToyExtensionPlugin.h`,
  `lib/Plugin/Toy/ToyExtensionPlugin.cpp`,
  `lib/Plugin/Toy/ToyConstructionProtocol.cpp`,
  `lib/Plugin/Toy/ToyEmitCRouteProvider.cpp`,
  `test/Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-stale-toy.mlir`,
  and `test/Target/Toy/toy-target-artifact-header.mlir`.
