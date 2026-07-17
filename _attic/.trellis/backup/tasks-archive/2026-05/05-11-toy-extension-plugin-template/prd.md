# Toy extension plugin template through the real registry pipeline

## Goal

Add a minimal plugin-local Toy extension integration that proves a new
non-RVV extension can enter TianChen-RV through the existing C++/MLIR plugin
registry, variant materialization, selected lowering-boundary, emission
readiness/plan, and test surfaces without adding extension-specific semantic
branches to generic core passes.

## Background

The previous RVV selected-config/VL-boundary work completed on `main` at
`cf56efb`. This task intentionally does not add another RVV family or runtime
claim. It introduces a small Toy extension template as an integration example
for future plugin owners.

## Requirements

- Provide a plugin-local Toy extension dialect or boundary operation surface.
- Keep Toy IR execution/lowering metadata only; do not add tensor, tile, or
  compute semantics to `tcrv.exec`.
- Provide a C++ Toy extension plugin registered through the existing
  `ExtensionPluginRegistry` and builtin plugin registration path.
- The Toy plugin must declare one bounded capability and propose at most one
  Toy variant only when that capability is present and available.
- Toy proposal and legality must fail closed for missing, unavailable,
  malformed, or capability-property-mismatched inputs using existing generic
  capability validation where possible.
- The Toy variant must materialize as `tcrv.exec.variant`, pass plugin
  legality, be selectable by the existing selection path when appropriate, and
  materialize a plugin-local Toy lowering boundary through plugin-owned hooks.
- Emission readiness and emission plan handling must route by selected variant
  origin and return plugin-owned Toy metadata without claiming executable
  artifacts, hardware execution, correctness, or performance.
- Add focused tests proving positive pipeline behavior and at least one
  fail-closed case. Include a lit/FileCheck pipeline test through `tcrv-opt`
  and focused C++ coverage for registry/plugin API behavior where useful.

## Non-Goals

- Do not add RVV functionality, RVV evidence fixtures, broad smoke matrices, or
  new runtime/hardware claims.
- Do not implement a real Toy backend, generic vector lowering, runtime ABI
  integration, benchmarking, or artifact export.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission in Python.
- Do not add Toy/RVV/Scalar/Offload semantic branches to generic core passes.
- Do not put Toy compute semantics into `tcrv.exec`.
- Do not ship docs-only or CMake-only scaffolding that never exercises the real
  compiler pipeline.

## Acceptance Criteria

- [ ] A Toy plugin-local dialect/op surface exists and is registered through
      plugin dialect registration.
- [ ] A Toy C++ plugin registers through the builtin extension registry.
- [ ] With an available Toy capability, the existing variant materialization
      pass produces a Toy-origin `tcrv.exec.variant`.
- [ ] Plugin legality accepts the valid Toy variant and rejects malformed or
      unavailable capability cases.
- [ ] Existing selection/lowering-boundary/emission-readiness path reaches Toy
      plugin hooks and materializes Toy-owned metadata.
- [ ] Tests demonstrate a positive Toy pipeline and fail-closed behavior.
- [ ] `git diff --check` passes.
- [ ] Focused Toy/plugin/transform tests pass; `check-tianchenrv -j2` is run
      if feasible after focused checks pass.

## Definition of Done

- Source changes are implemented in C++/MLIR/TableGen/CMake/lit/FileCheck.
- Trellis task state and notes reflect the actual outcome.
- The task is finished and archived only if the full module behavior above is
  complete and checked.
- One coherent commit records the completed module, or the task remains open
  with a precise continuation point.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Existing examples inspected: `ExtensionPlugin`, builtin plugin aggregation,
  Scalar and Offload plugin slices, extension dialect CMake/TableGen layout,
  and registry-dependent transform tests.
