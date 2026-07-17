# Future source-artifact placeholder erasure

## Goal

Delete the future source-artifact placeholder channel as active compiler,
plugin, construction-protocol, target-export, test, and spec state. Current
artifact routing must fail closed through explicit materialized artifact kinds
instead of preserving a pre-EmitC source-output placeholder.

## What I Already Know

- The repository is `/home/kingdom/phdworks/TianchenRV`.
- Current HEAD is `95ec27b chore(plugin): erase selected-plan metadata channel`.
- The worktree was clean before this task was created.
- There was no `.trellis/.current-task`; this task was created from the Hermes
  Direction Brief.
- `.trellis/spec/index.md` makes C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck the compiler stack, with Python limited to tooling and evidence.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` states that
  source output requires explicit extension-family IR plus a materialized MLIR
  EmitC module route; direct C string export and metadata-driven source
  artifacts are deletion targets.
- Focused scan found active `isSourceArtifactKind` /
  `hasArtifactKindToken(..., "source")` route-control logic in:
  `lib/Plugin/ExtensionPlugin.cpp`,
  `lib/Plugin/Construction/ConstructionProtocol.cpp`, and
  `lib/Target/TargetArtifactExport.cpp`.
- Focused scan found active tests registering or asserting
  `future-emitc-source-*` / `future-source-artifact-*` placeholders in:
  `test/Plugin/PluginEmissionPlanTest.cpp`,
  `test/Plugin/ConstructionProtocolCommonTest.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.

## Requirements

- Remove source-artifact placeholder detection as an active route-control
  mechanism.
- Remove active route IDs, artifact kinds, diagnostics, fixtures, and tests
  that preserve `future-emitc-source-*` or `future-source-artifact-*`.
- Keep real materialized artifact packaging boundaries for current object,
  header, and metadata artifacts.
- Replace source-token blacklist behavior with an explicit current artifact
  kind boundary where the deletion exposes an overly broad route surface.
- Keep plugin emission-plan, construction-manifest, target-export, and bundle
  validation fail-closed for unsupported materialized artifact kinds.
- Update specs only to remove wording that reserves the deleted placeholder
  channel as active authority; do not use specs as a status log.

## Acceptance Criteria

- [ ] Active include/lib/test/spec surfaces outside archived Trellis history no
      longer contain `future-emitc-source`, `future-source-artifact`,
      `isSourceArtifactKind`, or source-token artifact-kind control checks.
- [ ] `lib/Plugin/ExtensionPlugin.cpp` no longer rejects a source placeholder
      by string token; supported plan artifact kinds are bounded by explicit
      current materialized artifact kinds.
- [ ] `lib/Plugin/Construction/ConstructionProtocol.cpp` no longer preserves a
      source placeholder rejection path; manifest EmitC route artifact kinds
      are bounded by explicit current materialized artifact kinds.
- [ ] `lib/Target/TargetArtifactExport.cpp` no longer treats "not source" as
      the default artifact selector; object/header/metadata artifacts are the
      explicit supported surface.
- [ ] Affected C++ tests no longer register or assert future source placeholder
      routes and instead cover unsupported materialized artifact kind
      fail-closed behavior.
- [ ] Focused plugin, construction-protocol, and target artifact export checks
      pass, or any failure is reported as a missing new-architecture gap rather
      than restored source-placeholder compatibility.

## Out Of Scope

- Do not implement a new EmitC source route.
- Do not implement common lower-to-EmitC, RVV lowering, generated C semantic
  exporters, source compatibility modes, wrappers, legacy modes, evidence
  matrices, or artifact ledgers.
- Do not delete unrelated uses of "source" such as MLIR parse-source fixtures,
  `source_kernel` identity fields, or non-semantic artifact readers.
- Do not add new extension features or rebuild executable routes in this
  deletion-only round.

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`, and
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`.
- Relevant production files:
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`,
  `include/TianChenRV/Plugin/ConstructionProtocol.h`,
  `lib/Plugin/Construction/ConstructionProtocol.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`, and
  `lib/Target/TargetArtifactExport.cpp`.
- Relevant tests:
  `test/Plugin/PluginEmissionPlanTest.cpp`,
  `test/Plugin/ConstructionProtocolCommonTest.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.
