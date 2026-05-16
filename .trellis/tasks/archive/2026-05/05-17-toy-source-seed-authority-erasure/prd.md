# Toy source-seed metadata authority erasure

## Goal

Delete the Toy metadata-seed selected-boundary route so
`tcrv_toy.lowering_seed` can no longer act as compute, route, selected-boundary,
EmitC, or artifact authority. This is a Wrong Logic Deletion Campaign round:
delete stale authority before any Toy/source-frontdoor rebuild.

## Why Now

The RVV i32m1 add source-frontdoor path now depends on bounded unseeded source
body recognition rather than `tcrv_rvv.lowering_seed`. The remaining Toy path
still treats `tcrv_toy.lowering_seed = "template_compute"` as a positive input
that materializes `tcrv.exec`, Toy selected-boundary IR, EmitC route evidence,
and target artifacts. That is metadata-as-route-authority residue.

## Current Repository Facts

- `ToyExtensionPlugin::registerSourceSeedPasses` exposes
  `tcrv-toy-materialize-template-selected-boundary-seed`.
- `ToySelectedBoundarySeed.cpp` matches `tcrv_toy.lowering_seed` and creates
  `tcrv.exec.kernel`, `tcrv.exec.variant`, `tcrv_toy.compute_skeleton`, and
  selection diagnostics.
- `test/Transforms/Toy/toy-template-selected-boundary-seed.mlir` protects the
  positive Toy seed path through the direct pass, source-seed front door, EmitC,
  and target artifact export.
- `test/Transforms/SourceSeed/source-seed-artifact-front-door-pipeline-mixed.mlir`
  and `source-seed-artifact-front-door-pipeline-negative.mlir` currently depend
  on Toy source-seed behavior.
- RVV source materialization is separately specified as source-body-authorized;
  this task must not weaken the RVV unseeded structural materializer.

## Requirements

- Remove Toy source-seed pass registration from the active built-in plugin
  registry path.
- Remove the Toy source-seed pass implementation and public pass header if no
  active code should call it after deletion.
- Remove or rewrite tests that require positive Toy seed materialization from
  `tcrv_toy.lowering_seed`.
- Rewrite SourceSeed/front-door tests so common front-door behavior no longer
  requires Toy seed success.
- Preserve RVV source-seed behavior and tests that prove unseeded structural
  RVV source materialization.
- Keep deletion-only scope: do not add a replacement Toy materializer, alias,
  compatibility mode, wrapper, descriptor adapter, new plugin construction
  template, or new frontend lowering.

## Acceptance Criteria

- [ ] No positive Toy path materializes `tcrv.exec`,
  `tcrv_toy.compute_skeleton`, emission plans, EmitC, or artifacts from
  `tcrv_toy.lowering_seed`.
- [ ] `tcrv-toy-materialize-template-selected-boundary-seed` is no longer an
  active public pass route, or it fails closed without preserving Toy seed
  behavior.
- [ ] `--tcrv-source-seed-artifact-front-door-pipeline` does not depend on Toy
  metadata seed success; RVV unseeded structural source still works.
- [ ] Focused scans no longer find active positive Toy seed authority in
  source, tests, or specs.
- [ ] Focused Toy/SourceSeed/RVV/plugin checks pass, or failures are reported
  as missing new-architecture rebuild gaps without restoring the old path.

## Out Of Scope

- No replacement Toy unseeded materializer.
- No new common source-materializer protocol.
- No new RVV family/source coverage.
- No high-level frontend lowering.
- No compatibility alias, legacy mode, descriptor adapter, direct C exporter, or
  Python compiler-core behavior.
- No broad rename-only cleanup unrelated to Toy source-seed authority deletion.

## Technical Notes

- Relevant specs: `.trellis/spec/index.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- The source-seed front-door pipeline may still collect RVV source-seed passes
  from enabled plugins; common code must not inspect plugin marker names or
  source semantics.
- Remaining Toy construction/EmitC/target tests may stay only if they start
  from explicit Toy execution surfaces, not from `tcrv_toy.lowering_seed`.
