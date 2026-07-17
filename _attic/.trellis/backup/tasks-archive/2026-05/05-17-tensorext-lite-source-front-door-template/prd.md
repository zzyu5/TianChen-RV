# TensorExtLite Source-Front-Door Construction Template Consumption

## Goal

Add a TensorExtLite-owned bounded source front door that consumes the same
common source-front-door registration and source-artifact pipeline surface
already proven by RVV and Toy. The path must make this route real:

```text
TensorExtLite source marker
  -> TensorExtLite-owned source front-door pass
  -> selected tcrv.exec variant with origin = "tensorext-lite-plugin"
  -> ordered TensorExtLite role ops
  -> plugin-local selected lowering-boundary marker for generic emission gates
  -> TensorExtLite EmitC route provenance
  -> existing materialized-EmitC declaration header artifact route
```

This proves the construction template is not Toy-only while keeping
TensorExtLite semantics inside the TensorExtLite plugin, dialect, EmitC route
provider, and target exporter.

## Current Repository Facts

- HEAD `6920a87` proves the common source-front-door registry and
  `--tcrv-source-artifact-front-door-pipeline` can expose a Toy-owned source
  materializer without adding Toy/RVV branches in common orchestration.
- The common pipeline collects enabled plugin source-front-door pass factories,
  runs them deterministically, then runs generic hart-parallel, plugin legality,
  capability, emission-plan, and execution-plan coherence gates.
- TensorExtLite already has a plugin, dialect, construction protocol,
  explicit role ops, EmitC route provider, target support bundle, and
  declaration-only header artifact exporter.
- TensorExtLite already accepts an explicit role sequence inside the selected
  variant body:
  `config_skeleton -> load_frag_skeleton -> tile_mma_skeleton -> store_frag_skeleton`.
- TensorExtLite currently lacks a source-front-door materializer and public pass
  registration that can create the selected TensorExtLite IR from a narrow
  source marker.
- Existing TensorExtLite header artifact evidence starts from already
  materialized TensorExtLite IR and emission-plan diagnostics.

## Requirements

- Add a TensorExtLite-owned source-front-door pass registered through
  `SourceFrontDoorPassRegistration`.
- The public pass argument must be production source-front-door wording and must
  not introduce a seed alias or compatibility wrapper.
- The accepted input syntax is limited to module attributes:
  `tcrv_tensorext_lite.source_front_door = "fragment_mma_template"` and an
  optional valid `tcrv_tensorext_lite.source_kernel`.
- The source marker and source-kernel attribute are input syntax only. They must
  be removed after materialization and must not become descriptor, direct-C,
  source-export, route-id, runtime-ABI, or artifact authority.
- The pass must fail closed for missing/unknown marker values, invalid source
  kernel symbols, stale TensorExtLite lowering seed metadata, and stale
  pre-materialized `tcrv.exec`, TensorExtLite, RVV, or Toy selected-boundary or
  variant residue.
- The pass must materialize:
  - one `tcrv.exec.kernel`;
  - one available `tensorext_lite.tile_mma` capability with preserved
    `fragment_abi` and `handoff_kind`;
  - one selected TensorExtLite variant with `origin =
    "tensorext-lite-plugin"` and existing construction protocol metadata;
  - the ordered TensorExtLite role sequence inside the selected variant body;
  - one direct `tcrv_tensorext_lite.lowering_boundary` marker carrying the
    selected path metadata required by the generic emission-plan gate;
  - one selected-path diagnostic targeting the TensorExtLite first-slice
    variant.
- The common source-artifact front-door pipeline must consume the new pass and
  produce TensorExtLite emission-plan provenance through existing generic gates.
- The resulting IR must pipe to the existing
  `tcrv-translate --tcrv-export-target-header-artifact` path and export the
  existing declaration-only TensorExtLite materialized EmitC header.
- C++ registry evidence must show built-in RVV, Toy, and TensorExtLite
  source-front-door registrations are collected through the same common
  registry interface in deterministic order.

## Acceptance Criteria

- [ ] TensorExtLite exposes exactly one source-front-door pass registration when
      built-in plugins are enabled.
- [ ] Registry tests prove RVV, Toy, and TensorExtLite source-front-door
      registrations are collected through `collectSourceFrontDoorPasses`,
      without stale source-seed arguments or extension-specific common
      orchestration branches.
- [ ] A positive TensorExtLite source fixture enters through
      `--tcrv-tensorext-lite-materialize-fragment-mma-source-front-door` and
      produces the selected TensorExtLite variant and ordered role sequence.
- [ ] The same source fixture enters through
      `--tcrv-source-artifact-front-door-pipeline` and produces supported
      TensorExtLite emission-plan metadata, route provenance, runtime ABI
      metadata, and source-op/source-role evidence.
- [ ] Piping the source fixture through
      `tcrv-opt --tcrv-source-artifact-front-door-pipeline` and
      `tcrv-translate --tcrv-export-target-header-artifact` exports the
      declaration-only TensorExtLite header artifact.
- [ ] Negative lit evidence rejects unknown marker values, invalid source
      kernel symbols, stale TensorExtLite seed metadata, stale pre-materialized
      `tcrv.exec`/TensorExtLite/RVV/Toy residue, missing role ops, reordered
      role ops, and missing route provenance before artifact output.
- [ ] Disabled built-in plugins keep the TensorExtLite public pass unavailable,
      and the common source-artifact front-door pipeline remains fail-closed
      with an empty plugin registry.
- [ ] Targeted scans over changed TensorExtLite/Toy/RVV/common front-door
      surfaces show no descriptor route authority, no direct-C semantic
      exporter, no source-export route, no legacy source-seed alias, and no new
      core/common semantic branch on TensorExtLite/Toy/RVV names.

## Definition Of Done

- Focused build/checks for `tcrv-opt`, `tcrv-translate`, TensorExtLite plugin
  tests, Toy source-front-door regression tests, and source-front-door/header
  lit tests pass.
- `git diff --check` passes.
- `check-tianchenrv` passes if practical after focused checks.
- Trellis task status and journal notes truthfully record the implementation,
  validation, and any remaining gap.
- One coherent commit is created if the task is complete.

## Technical Approach

- Mirror Toy's source-front-door pattern structurally, but keep all
  TensorExtLite facts in TensorExtLite-owned files and constants.
- Add `TensorExtLiteSourceFrontDoor.{h,cpp}` with a module pass that matches
  only the bounded TensorExtLite source marker, validates source-only input,
  materializes the TensorExtLite capability and selected variant, then places
  the existing ordered role ops inside the selected variant body.
- Materialize the existing TensorExtLite selected lowering-boundary op as a
  direct kernel child so the generic emission-plan and coherence gates can
  validate the selected path without learning TensorExtLite role semantics.
- Extend `TensorExtLiteExtensionPlugin` to register the pass through
  `registerSourceFrontDoorPasses`.
- Keep common/tool code unchanged unless a generic bug is found. Common code may
  only collect and run registrations.
- Add focused lit tests for direct pass output, common pipeline output, target
  header export, and fail-closed input cases.
- Extend existing C++ plugin/registry coverage rather than creating a parallel
  registry model.

## Out Of Scope

- No general TensorExt frontend, linalg lowering, tensor/tile IR, runtime
  execution, performance path, object/bundle packaging, RVV changes,
  IME/offload work, or Python compiler core.
- No descriptor adapters, descriptor-driven computation, source-export routes,
  direct C semantic exporters, compatibility aliases, or legacy lowering seeds.
- No common/core branches on TensorExtLite, Toy, RVV, dtype, operation names,
  route ids, runtime ABI names, or artifact kinds to infer semantics.
- No runtime correctness, hardware, or performance claims. RVV hardware
  evidence is not part of this task.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
- Previous task PRD read:
  `.trellis/tasks/archive/2026-05/05-17-toy-source-front-door-construction-template-proof/prd.md`.
- Initial code/test surfaces inspected:
  `include/TianChenRV/Plugin/Toy/ToySourceFrontDoor.h`,
  `lib/Plugin/Toy/ToySourceFrontDoor.cpp`,
  `include/TianChenRV/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h`,
  `include/TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h`,
  `include/TianChenRV/Plugin/TensorExtLite/TensorExtLiteEmitCRouteProvider.h`,
  `include/TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteOps.td`,
  `lib/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.cpp`,
  `lib/Plugin/TensorExtLite/Construction/TensorExtLiteConstructionProtocol.cpp`,
  `lib/Plugin/TensorExtLite/EmitC/TensorExtLiteEmitCRouteProvider.cpp`,
  `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`,
  `test/Plugin/TensorExtLiteExtensionPluginTest.cpp`,
  `test/Plugin/ToyExtensionPluginTest.cpp`,
  `test/Transforms/Toy/toy-template-source-front-door.mlir`,
  and `test/Target/TensorExtLite/tensorext-lite-target-artifact-header.mlir`.
