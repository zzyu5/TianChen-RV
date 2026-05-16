# Target artifact route-metadata authority erasure

## Goal

Erase target artifact route metadata and selected-plan metadata as artifact-route
authority. Target artifact export may still validate concrete artifact kinds,
export callbacks, runtime ABI fields, duplicate route IDs, and bundle component
shape, but extension bundles and exporters must not publish or require
selected-plan metadata descriptors as the authority for route selection,
preflight, or bundle acceptance.

## Why

The previous deletion round removed direct-C route absence contracts, but active
target/export surfaces still expose `TargetArtifactRouteMetadata`,
`ExtensionBundleTargetArtifactRouteMetadata`,
`requiresTargetArtifactRouteMetadata`,
`addTargetArtifactRouteMetadataRequirement`,
`getTargetArtifactRouteMetadata`, route metadata preflight, route claim fields,
and selected-plan metadata descriptor helpers as artifact-route authority. That
keeps export validation anchored in descriptor/metadata claims instead of
concrete artifacts produced from materialized extension-family EmitC/object/
header/metadata outputs.

## Scope

- Delete target artifact route-metadata requirement APIs from extension bundle
  registration.
- Delete exporter/composite-exporter route metadata as target artifact route
  authority.
- Stop target artifact export and execution-plan coherence from consuming
  `selected_plan_metadata` as route-selection or route-preflight contract.
- Rewrite tests that currently protect missing, duplicate, stale, or required
  route-metadata descriptor behavior.
- Update directly affected spec/test wording that says target artifact export
  validates route metadata authority.

## Non-goals

- No Common EmitC rebuild.
- No executable plugin template.
- No new source artifact route.
- No runtime ABI implementation.
- No replacement route-metadata schema.
- No bundle ledger or checkpoint protocol.
- No restoration of deleted direct-C absence contracts.
- No broad redesign outside the target artifact metadata authority surface.

## Requirements

- Target artifact export must not require extension bundles or composite
  exporters to declare route metadata requirements before artifact export.
- `selected_plan_metadata` may remain compiler-visible diagnostic or manifest
  self-description, but target artifact export must not use it to select,
  preflight, or authorize an artifact route.
- Registry validation may still reject malformed route IDs, artifact kinds,
  callbacks, duplicate routes, source artifact kinds without materialized EmitC
  routes, runtime ABI parameter contract mismatches, and bundle component
  shape mismatches.
- Remaining bundle serialization may include selected-plan metadata only as
  copied diagnostic description, not route authority.
- If deletion exposes missing new-architecture failures, keep the deletion
  truthful and report the gap instead of adding compatibility wrappers.

## Acceptance Criteria

- [ ] Active-surface ref-scan finds no live use of
  `TargetArtifactRouteMetadata`,
  `ExtensionBundleTargetArtifactRouteMetadata`,
  `requiresTargetArtifactRouteMetadata`,
  `addTargetArtifactRouteMetadataRequirement`,
  `getTargetArtifactRouteMetadata`, `route metadata preflight`, or
  `candidate route metadata requirements` outside archived tasks/workspace/
  build/artifact temp directories.
- [ ] Target artifact export no longer consumes `selected_plan_metadata` as a
  route-selection or route-preflight contract.
- [ ] Tests no longer protect missing/duplicate/stale route-metadata descriptor
  behavior.
- [ ] Concrete validation remains covered for artifact kind, callbacks,
  runtime ABI fields/parameters, duplicate route IDs, and fail-closed source
  artifacts.
- [ ] Focused target build/test passes or any failure is reported as a
  deletion-exposed architecture gap without restoring metadata authority.
- [ ] `git diff --check`, Trellis task validation, finish/archive, clean status,
  and one coherent commit are completed if the task is complete.

## Read First

- `.trellis/spec/index.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `include/TianChenRV/Target/TargetArtifactExport.h`
- `lib/Target/TargetArtifactExport.cpp`
- `include/TianChenRV/Target/RVV/RVVVectorShape.h`
- `include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h`
- `include/TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h`
- `include/TianChenRV/Dialect/Exec/IR/ExecOps.td`
- `test/Target/TargetArtifactExportTest.cpp`

## Technical Notes

- Initial repo state: `/home/kingdom/phdworks/TianchenRV`, clean worktree,
  HEAD `b2c13d6 chore(target): erase direct-c route absence contracts`.
- Initial active-surface scan confirmed live references in target export
  headers/source, RVV selected-plan metadata helpers, execution-plan coherence,
  active specs, and target artifact export tests.
- This is part of the Wrong Logic Deletion Campaign: deletion before rebuild,
  no new route-metadata replacement authority in this round.
