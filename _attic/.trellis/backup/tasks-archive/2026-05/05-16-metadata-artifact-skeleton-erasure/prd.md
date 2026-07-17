# Metadata artifact skeleton erasure

## Goal

Erase obsolete Template, Toy, and TensorExtLite metadata-artifact target
skeletons and public metadata-route helper APIs so the target artifact surface
no longer advertises a supported or compatibility metadata route after the
metadata-only route contract deletion. Unsupported paths must fail closed
through generic missing-EmitC-route diagnostics, not through preserved no-op
target exporter packages.

## Requirements

- Delete the Template, Toy, and TensorExtLite `*MetadataArtifact` target
  headers and sources.
- Remove CMake targets, subdirectories, and library links whose only purpose is
  to build those metadata-artifact skeletons.
- Remove public Template, Toy, and TensorExtLite plugin helper APIs named as
  metadata route, emission kind, artifact kind, runtime ABI kind, or runtime
  glue role accessors.
- Rewrite or delete tests that protect no-op metadata artifact bundle
  registration as meaningful route behavior.
- Preserve generic fail-closed target artifact exporter registry diagnostics
  for missing route metadata and duplicate routes without using the deleted
  Toy metadata route helpers.
- Preserve plugin unsupported emission readiness/planning behavior as
  no-active EmitC lowering diagnostics.

## Acceptance Criteria

- [ ] No active `include/TianChenRV/Target/{Template,Toy,TensorExtLite}/*MetadataArtifact.h`
      headers remain.
- [ ] No active `lib/Target/{Template,Toy,TensorExtLite}/*MetadataArtifact.cpp`
      sources remain.
- [ ] No active CMake target remains solely for Template, Toy, or
      TensorExtLite metadata-artifact skeletons.
- [ ] No active public plugin API declarations or definitions remain for
      `get*MetadataRouteID`, `get*MetadataEmissionKind`,
      `get*MetadataArtifactKind`, `get*MetadataRuntimeABIKind`, or
      `get*MetadataRuntimeGlueRole` for Template, Toy, or TensorExtLite.
- [ ] `test/Target/TargetArtifactExportTest.cpp` no longer includes or calls a
      Toy metadata artifact bundle as a no-op success path.
- [ ] Focused active-surface ref scan for the deleted symbols returns no active
      code/test hits outside this task directory.
- [ ] Affected target/plugin libraries and tools build, at least
      `ninja -C build tcrv-opt tcrv-translate`.
- [ ] Affected C++ tests under `test/Target` and `test/Plugin` pass after
      rewriting stale metadata-route expectations.
- [ ] `ninja -C build check-tianchenrv` is attempted; any failure is reported
      as an expected deletion gap only if it is caused by missing rebuild
      architecture rather than restored metadata-route compatibility.
- [ ] `git diff --check` and Trellis task validation pass.

## Definition of Done

- Source changes are deletion/refactor-only inside the metadata-artifact
  skeleton boundary.
- Focused checks pass or any remaining failure is explicitly tied to a
  missing new-architecture gap.
- The Trellis task is truthful, finished or archived per repo convention, and
  one coherent commit is created.

## Technical Approach

Remove the three metadata-artifact target packages instead of leaving empty
no-op registration functions. Then delete the corresponding public plugin
metadata helper declarations/definitions and update tests to use local generic
test constants for target artifact registry validation. Keep construction
protocol and plugin unsupported readiness/planning behavior intact where it
describes missing materialized EmitC lowering, because this task is not a
Common EmitC rebuild.

## Decision

Context: The previous deletion rounds removed metadata-only route authority,
but the target artifact surface still exposed fail-closed metadata-artifact
packages and helper names that looked like supported route ownership.

Decision: Delete the skeleton packages and helper APIs outright. Tests may
still cover generic registry validation with local fake routes, but must not
consume Template, Toy, or TensorExtLite metadata-artifact APIs.

Consequences: Built-in extension bundles remain plugin-registration surfaces
only for these families. Any future executable artifact path must be rebuilt
from explicit extension-family IR through a materialized EmitC route and a new
target export contract.

## Out of Scope

- No Common EmitC rebuild.
- No executable plugin template.
- No new Template, Toy, or TensorExtLite implementation.
- No replacement metadata artifact exporter.
- No wrapper, legacy mode, compatibility registration, new artifact kind,
  source artifact route, or broad cleanup outside this skeleton boundary.
- No RVV runtime, correctness, or performance claims.

## Technical Notes

- Read `.trellis/spec/index.md`.
- Read `.trellis/spec/plugin-protocol/interfaces-and-registry.md`.
- Read `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Read `.trellis/spec/testing/mlir-testing-contract.md`.
- Read the current metadata-artifact headers/sources, CMake targets, plugin
  headers/sources, and directly related `test/Target` / `test/Plugin` code.
- Relevant durable rule: descriptor/direct-C/metadata artifact residues are
  deletion targets; deletion rounds must not mix in rebuild work.
