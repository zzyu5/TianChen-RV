# Direct-C Source Artifact Kind Erasure

## Goal

Delete the remaining active `runtime-callable-c-source` and
`standalone-c-source` artifact-kind authority from plugin construction,
plugin emission-plan validation, target artifact export, and the focused
tests that preserve those old route-kind names. This is a deletion-only Wrong
Logic Deletion Campaign round: future source output must come from a real
materialized MLIR EmitC route, not descriptor metadata, selected-route
metadata, or deleted direct-C route IDs.

## Background

Current repo state is clean on `main` after recent deletion-campaign commits.
The remaining tracked-file scan still shows old direct-C artifact-kind names
inside:

- `lib/Plugin/Construction/ConstructionProtocol.cpp`
- `lib/Plugin/ExtensionPlugin.cpp`
- `lib/Target/TargetArtifactExport.cpp`
- `test/Plugin/ConstructionProtocolCommonTest.cpp`
- `test/Plugin/PluginEmissionPlanTest.cpp`
- `test/Target/TargetArtifactExportTest.cpp`
- `test/Target/ArtifactExport/target-source-artifact-routes.test`

The old names are not allowed to remain as first-class route branches or test
fixtures, even when the current behavior rejects them. The remaining
validation should be positive current-route validation or generic fail-closed
shape validation that does not encode deleted direct-C route IDs.

## Scope

- Remove old-name-specific direct-C source artifact-kind constants and helper
  predicates from active plugin construction, plugin emission-plan, and target
  artifact export validation code.
- Replace old-name-specific negative validation with positive current-route
  validation or generic fail-closed validation.
- Rewrite focused plugin and target tests so they no longer require
  `runtime-callable-c-source` or `standalone-c-source` as route fixtures.
- Keep current metadata/header/object/generic artifact validation behavior
  where it does not preserve deleted direct-C route-kind names.
- Keep task metadata, Trellis context, and journal truthful.

## Non-Goals

- No RVV emission rebuild.
- No high-level frontend lowering.
- No new artifact route, bundle/index ledger, executable plugin template,
  EmitC route implementation, descriptor compatibility path, wrappers,
  quarantine mode, or legacy compatibility layer.
- No edits whose main result is only prompt/report/status metadata.
- Do not chase `artifacts/tmp` archaeology or archived Trellis tasks.
- Do not treat `artifacts/grill-consensus-20260515.md` as source truth.
- Do not make RVV runtime, correctness, or performance claims without
  `ssh rvv` evidence.

## Acceptance Criteria

- Active production code under `lib/Plugin` and `lib/Target` no longer carries
  `runtime-callable-c-source` or `standalone-c-source` as direct-C semantic
  exporter route authority.
- Directly related active tests under `test/Plugin` and `test/Target` no
  longer require or protect the deleted source artifact-kind names as route
  fixtures.
- Any remaining validation is expressed as positive allowed current-route
  shape or generic fail-closed behavior, not compatibility with old direct-C
  routes.
- A focused tracked-file ref-scan over `lib/Plugin`, `lib/Target`,
  `test/Plugin`, and `test/Target` for
  `runtime-callable-c-source|standalone-c-source|deleted source artifact kind`
  has remaining hits explained or eliminated.
- Focused build/test coverage for changed plugin construction, plugin
  emission-plan, and target artifact export tests passes, or any failure is
  reported as a real missing-architecture gap without restoring the deleted
  route authority.
- Task is finished/archived and one coherent deletion-only commit is created
  if completion is possible.

## Validation Plan

- Run focused ref-scan:
  `rg -n "runtime-callable-c-source|standalone-c-source|deleted source artifact kind" lib/Plugin lib/Target test/Plugin test/Target`.
- Build the smallest existing CMake targets covering changed C++ tests.
- Run the focused changed C++ tests.
- Run the touched lit test or focused lit directory if the lit file remains
  relevant after rewrite.
- Run `git diff --check`, Trellis task validation, finish/archive, commit,
  and final `git status --short`.
