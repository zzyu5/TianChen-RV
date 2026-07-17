# RVV deleted metadata-route diagnostic erasure

## Goal

Erase active RVV deleted metadata-route diagnostic contracts and tests. The RVV
plugin and public pass/tool tests must no longer preserve
`metadata-only first-slice`, `rvv_deleted_metadata_path`,
`deleted_metadata_route`, or deleted lowering-boundary route names as named
proposal, legality, emission, lowering-boundary, or export behavior. Missing
new architecture should surface as generic absence of explicit typed RVV IR or
missing materialized EmitC route.

## What I already know

- The previous committed round is `2261502 chore(rvv): erase probe-derived route
  config`, and the worktree was clean before this task was created.
- Current RVV spec says the deleted metadata-only proposal route is not active
  compiler authority.
- The task is part of a deletion campaign: delete or rewrite stale contracts
  first, without rebuilding a replacement architecture in the same round.
- The active surface to inspect includes RVV plugin C++ tests and lit tests
  under PluginVariantMaterialization, PluginVariantLegality,
  ExecutionPlanning, LoweringBoundary, and TargetArtifactBundleExport.

## Requirements

- Remove active RVV plugin branches that emit old metadata-only first-slice
  route names or stale deleted metadata-route diagnostics.
- Rewrite or delete active tests that require old RVV metadata route names to
  decline proposal, fail legality, fail emission planning, fail lowering-boundary
  validation, or fail export by name.
- Keep valid explicit `tcrv_rvv` extension-family syntax tests that do not
  protect the deleted metadata route.
- Keep generic fail-closed behavior only when it is phrased as missing explicit
  typed RVV IR, no viable proposal, unsupported emission, or missing
  materialized EmitC route.
- Update specs only if they still make the deleted route a durable diagnostic
  contract rather than describing its absence.

## Acceptance Criteria

- [ ] No active RVV plugin branch emits `metadata-only first-slice`,
  `metadata-only RVV first-slice`, `rvv_deleted_metadata_path`,
  `deleted_metadata_route`, `RVV automatic metadata-only`, `RVV metadata-only
  lowering-boundary route`, `tcrv_rvv.lowering_boundary is not active compiler
  authority`, or `collect deleted RVV` as active behavior.
- [ ] No active tests require those old names for proposal decline, legality
  failure, emission planning failure, lowering-boundary validation, or target
  artifact bundle export behavior.
- [ ] Specs do not promote the old metadata route names into durable API or
  diagnostic contracts.
- [ ] Focused ref-scan for the stale strings is clean outside excluded archive,
  workspace, artifact, build, and git paths.
- [ ] Affected build targets pass: `tcrv-opt`, `tcrv-translate`, and
  `tianchenrv-rvv-extension-plugin-test`.
- [ ] Affected lit suites under `test/Transforms` and `test/Target` pass or any
  remaining failures are truthful missing-architecture gaps caused by deletion.
- [ ] `check-tianchenrv` is attempted.
- [ ] `git diff --check` and Trellis validation pass.
- [ ] Task is finished/archived and one coherent commit records the round.

## Out of Scope

- No RVV EmitC rebuild.
- No new RVV lowering implementation.
- No new selected-boundary implementation.
- No replacement proposal route.
- No compatibility, deprecated-route, or legacy-mode diagnostic wrapper.
- No new helper API.
- No `ssh rvv` evidence expansion.
- No deletion of valid explicit `tcrv_rvv` extension-family IR syntax merely
  because it is RVV-specific.

## Technical Notes

- Required specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Initial code/test surfaces from the direction brief:
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
  - `test/Transforms/VariantMaterialization/plugin-variant-materialization-rvv-deleted-route.mlir`
  - `test/Transforms/PluginVariantLegality/plugin-variant-legality-pass-invalid.mlir`
  - `test/Transforms/ExecutionPlanning/execution-planning-pipeline-no-viable.mlir`
  - `test/Transforms/LoweringBoundary/rvv-lowering-boundary-malformed.mlir`
  - `test/Transforms/LoweringBoundary/rvv-lowering-boundary-missing-selection.mlir`
  - `test/Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`
