# TensorExtLite common EmitC route activation

## Goal

Graduate the existing TensorExtLite first-slice construction-template family
from passive no-active-route metadata to one bounded selected TensorExtLite
explicit typed-body route that can be consumed by the common
`TCRVEmitCLowerableRoute` materializer and produce a materialized MLIR EmitC
module.

This proves the Extension-Family Plugin Construction Protocol is not RVV-only
while keeping TensorExtLite plugin-owned and explicitly non-target-artifact for
this round.

## What I Already Know

- Current HEAD is `55be954 rvv: package materialized emitc header bundle`.
- The worktree was clean before task creation.
- TensorExtLite already has a dialect, construction manifest, proposal,
  legality, cost, and explicit no-active-route emission diagnostics.
- The current TensorExtLite dialect only has a `tile_mma_skeleton` typed role
  op; the construction manifest already names the expected role sequence
  `configure->load_frag->tile_mma->store_frag`.
- Toy is the closest minimal existing template for plugin-owned
  selected-boundary plus common EmitC route materialization.
- RVV is useful only as a mature route-shape reference; this task must not copy
  RVV intrinsic/object/header/package logic.

## Requirements

- TensorExtLite plugin-owned code must define one active first-slice EmitC route
  for a selected explicit TensorExtLite body.
- The selected body must be an ordered typed role sequence:
  `config_skeleton`, `load_frag_skeleton`, `tile_mma_skeleton`,
  `store_frag_skeleton`.
- Each typed TensorExtLite role op must expose bounded
  `TCRVEmitCLowerableOpInterface` provenance.
- TensorExtLite construction manifest, typed role graph, active route metadata,
  emission readiness, emission plan, and route materialization must be consumed
  by C++ code, not only documented.
- `--tcrv-materialize-emitc-lowerable-routes` must produce a materialized EmitC
  module for a selected TensorExtLite fixture with origin
  `tensorext-lite-plugin`.
- The common/core materialization path may only use existing generic origin
  plugin and `TCRVEmitCLowerableRoute` interfaces.
- Target artifact export for TensorExtLite remains unsupported and must fail
  closed; this round does not create object/header/package routes.

## Acceptance Criteria

- [ ] Positive C++ test proves TensorExtLite construction manifest, role graph,
      active route metadata, role validation, emission readiness, emission plan,
      and common EmitC materialization are all code-consumed.
- [ ] Positive lit/FileCheck fixture shows a selected TensorExtLite explicit
      role body materializes through
      `--tcrv-materialize-emitc-lowerable-routes` into EmitC include, function
      boundary, route-source provenance, call-source provenance, and
      TensorExtLite call-opaque steps.
- [ ] Negative coverage fails closed for missing selected body, wrong origin,
      stale no-active-route route metadata, missing/duplicate typed role ops,
      role order or interface mismatch, descriptor/source-export/direct-C
      residue, and attempted TensorExtLite target artifact export.
- [ ] Focused build and tests pass for TensorExtLite plugin/dialect route
      changes plus affected common route consumers.
- [ ] Targeted scans over touched TensorExtLite/common/test surfaces show no
      descriptor-driven computation, direct C semantic exporter, source-export
      route, target artifact route, Python compiler-core logic, or common/core
      TensorExtLite semantic branch residue.

## Out Of Scope

- No TensorExtLite object/header/bundle/package target artifact routes.
- No RVV route expansion, RVV intrinsic copying, or ssh rvv claim.
- No real vendor ISA semantics, runtime correctness, performance, or hardware
  evidence for TensorExtLite.
- No high-level `linalg`, `stablehlo`, or `tosa` lowering.
- No descriptor catalogs, direct C semantic exporters, source-export routes,
  Python compiler-core behavior, compatibility wrappers, or common/core
  `if TensorExtLite` branches.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/guides/plugin-locality-review-guide.md`.
- Relevant existing model inspected:
  TensorExtLite plugin/dialect/test surfaces, Toy plugin EmitC route provider,
  Toy construction protocol, Toy plugin tests, common
  `TCRVEmitCLowerableRoute` materializer, and
  `EmitCLowerableMaterialization.cpp`.

## Definition Of Done

- Code implements the bounded TensorExtLite route through C++/MLIR/ODS/CMake.
- Focused C++ and lit tests pass.
- `git diff --check` passes.
- Task status and journal reflect the actual outcome.
- One coherent commit is created if the round finishes.
