# Toy plugin metadata artifact route through the target exporter front door

## Goal

Carry the selected Toy plugin emission-plan metadata produced by
`tcrv-opt --tcrv-execution-planning-pipeline` into a deterministic
non-executable text artifact emitted by the existing C++ `tcrv-translate`
target artifact front door.

## Background

Commit `2b93451` added the plugin-local Toy dialect and Toy extension plugin.
That task proves builtin plugin registration, variant materialization,
selection, plugin-owned `tcrv_toy.lowering_boundary`, and emission-plan
metadata. This task extends the Toy template to the target artifact/exporter
surface without claiming Toy executable lowering, runtime glue, hardware
execution, correctness, or performance.

## Requirements

- Add a Toy-owned target artifact exporter component under the Toy target or
  plugin-local support surface.
- Register a `TargetArtifactExporter` for the Toy emission-plan route already
  produced by the Toy plugin through the builtin target artifact exporter
  aggregation path.
- Consume real `TargetArtifactCandidate` records derived from selected Toy
  emission-plan diagnostics. Do not emit from an ad hoc file, Python helper, or
  standalone bypass around `tcrv-translate`.
- Validate the Toy candidate before output:
  origin plugin, route id, emission kind, artifact kind, runtime ABI kind/name,
  runtime glue role, runtime ABI string, lowering-boundary op name, selected
  variant symbol, required capability metadata, and selected-plan metadata.
- Emit a deterministic bounded text artifact that explicitly identifies itself
  as non-executable metadata evidence. It may include kernel symbol, selected
  variant, role, origin plugin, route id, artifact kind, Toy template ABI,
  handoff kind, required capability, lowering-boundary op name, runtime ABI
  metadata, and selected-plan metadata.
- Keep generic target export/front-door logic free of Toy semantic branches.
  Central changes are allowed only for Toy exporter registration.
- Add a focused lit/FileCheck test that runs the Toy execution-planning
  pipeline and then `tcrv-translate --tcrv-export-target-artifact`.
- Add at least one fail-closed target artifact export test for malformed or
  mismatched Toy artifact metadata.

## Non-Goals

- Do not implement a Toy executable backend, runtime ABI glue, benchmark,
  hardware execution, correctness proof, or performance proof.
- Do not add RVV functionality, RVV evidence fixtures, broad smoke matrices, or
  new RVV claims.
- Do not add compute semantics to `tcrv.exec` or make Toy a generic compute
  dialect.
- Do not add Toy/RVV/scalar/offload semantic branches to generic core passes or
  generic target export logic beyond builtin exporter registration.
- Do not implement compiler core, dialects, passes, plugin registry, lowering,
  emission, or artifact export in Python.

## Acceptance Criteria

- [ ] A Toy-owned target artifact exporter is implemented in C++ and registered
      with `TargetArtifactExporterRegistry`.
- [ ] The Toy execution-planning pipeline produces a selected Toy path,
      Toy lowering boundary, and exportable Toy metadata artifact candidate.
- [ ] `tcrv-translate --tcrv-export-target-artifact` emits deterministic
      non-executable Toy metadata evidence from the real candidate path.
- [ ] Positive lit coverage checks both selected IR/emission-plan metadata and
      exported artifact content.
- [ ] Negative lit coverage fails closed for malformed Toy artifact metadata.
- [ ] Focused C++/build checks pass for `tcrv-opt`, `tcrv-translate`, Toy
      plugin tests, and target artifact tests as applicable.
- [ ] `git diff --check` passes.
- [ ] `check-tianchenrv -j2` is run if feasible after focused checks pass.

## Definition of Done

- Source changes are implemented in C++/MLIR/TableGen/CMake/lit/FileCheck.
- Trellis task notes reflect the actual outcome.
- The task is finished and archived only if artifact route implementation,
  focused checks, and a coherent commit are complete.
- No RVV runtime/correctness/performance claim is made.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/validation/experiment-reference.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Prior Toy task reference:
  `.trellis/tasks/archive/2026-05/05-11-toy-extension-plugin-template/prd.md`.
- Key source surfaces:
  `ToyExtensionPlugin`, `tcrv_toy.lowering_boundary`,
  `TargetArtifactExport`, builtin target exporter registration, and
  `tcrv-translate` target artifact translations.
- Existing generic candidate collection only turns supported emission-plan
  diagnostics into `TargetArtifactCandidate` records. For Toy export, the Toy
  metadata artifact route must therefore be a supported artifact-export route
  whose artifact kind and output text make the non-executable evidence boundary
  explicit.
