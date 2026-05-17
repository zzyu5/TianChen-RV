# Executable construction gate closure for remaining builtin plugins

## Goal

Close the registration-time executable construction conformance gate for all
current construction-capable builtin plugins. Toy and Template are the concrete
missing owners unless live repository evidence proves another
construction-capable builtin plugin is the actual gap.

This round makes stale construction manifests fail before proposal, lowering,
EmitC route mapping, or artifact export can consume a plugin. It does not add a
new lowering route or rebuild plugin semantics.

## What I Already Know

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Worktree was clean before this task.
- Current HEAD before this task was
  `33a0879 plugin: gate executable construction conformance`.
- No `.trellis/.current-task` existed at session start, so this task was
  created from the Hermes Direction Brief before source edits.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-18-executable-construction-protocol-registry-gate/prd.md`
  made the common conformance verifier and registry hook production-real.
- That previous task wired RVV and TensorExtLite through
  `ExtensionPlugin::verifyExecutableConstructionConformance()` but left the
  default implementation as success for plugins that do not override the hook.
- Direction Brief evidence says Toy and Template already have construction
  protocol readiness and EmitC route providers, so they must not bypass the
  canonical `ExtensionPluginRegistry::registerPlugin()` conformance boundary.
- The Brief's requested spec path
  `.trellis/spec/extension-plugins/plugin-construction-protocol.md` does not
  exist in the current repo. The live construction protocol spec is
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`.

## Requirements

- Identify every current builtin plugin that is construction-capable, using
  live header/implementation evidence rather than old task notes.
- Wire Toy and Template, or any other live construction-capable builtin gap,
  through the registry-time executable construction conformance hook.
- Keep the common gate extension-family agnostic. Common code must not gain
  RVV, TensorExtLite, Toy, Template, scalar, offload, or other family semantic
  branches.
- Reuse existing construction protocol readiness, manifest, role/interface,
  selected boundary, EmitC route, and construction artifact metadata surfaces
  where they already exist.
- Do not rely only on ad hoc plugin-local `verify*ConstructionProtocolReady()`
  calls inside proposal or EmitC providers when the plugin can be registered
  first and consumed later.
- Add focused fail-closed coverage proving stale Toy/Template-style
  construction specs are rejected by canonical plugin registration with
  actionable diagnostics.
- Add focused success coverage proving valid construction-capable builtins
  register through the same registry boundary.
- Preserve the previous RVV and TensorExtLite registry gate behavior.
- Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python remains only for Trellis/tooling/probe support.

## Acceptance Criteria

- No current construction-capable builtin executable plugin can rely on the
  default-success executable construction conformance hook.
- Toy and Template either override the registry-time gate or live repository
  evidence proves one of them has no executable construction protocol and
  should remain ungated.
- `ExtensionPluginRegistry::registerPlugin()` rejects deliberately stale
  Toy/Template-style construction manifests, typed-role/interface metadata
  mismatches, selected-boundary mismatches, or artifact metadata mismatches
  before the plugin is accepted.
- Valid RVV, TensorExtLite, Toy, and Template builtin plugin registration paths
  still succeed if each is construction-capable.
- Negative diagnostics name the failing plugin/construction component clearly
  enough to repair the stale manifest or metadata.
- Common construction/registry code remains free of extension-family semantic
  branches and does not introduce descriptor, direct-C, source-export, new
  artifact kind, compatibility wrapper, or Python compiler-core authority.
- Focused construction protocol common and affected plugin tests build and run.
- `git diff --check` passes.
- `check-tianchenrv` is run if practical; otherwise the exact blocker is
  recorded.

## Out Of Scope

- New RVV SEW/LMUL/dtype/op coverage.
- New Toy, Template, TensorExtLite, RVV, IME, scalar, or offload semantics.
- New EmitC route kinds, target artifact packaging, runtime execution, source
  exporters, direct C compute generation, descriptor tables, compatibility
  wrappers, or legacy modes.
- New artifact kinds or artifact ledger/checkpoint protocol.
- Runtime correctness or performance claims. No `ssh rvv` evidence is required
  unless emitted RVV artifacts or runtime packaging behavior changes.
- Broad smoke/report-only work.

## Definition Of Done

- PRD and task context describe this round truthfully.
- Live code scan identifies construction-capable builtin plugin status and the
  gated/ungated outcome.
- Code changes rewire the missing construction-capable builtin plugins through
  the existing registry-time gate.
- Focused C++ tests pass for valid registration and stale-manifest rejection.
- Focused affected plugin tests pass for Toy, Template, RVV, and
  TensorExtLite where corresponding test targets exist.
- `git diff --check` passes.
- Residue scans prove no descriptor/direct-C/source-export authority or
  extension-specific common semantic branch was added on touched surfaces.
- Trellis task is finished/archived when complete.
- One coherent commit is created when the task is complete.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/guides/plugin-locality-review-guide.md`.
- Previous task context read:
  `.trellis/tasks/archive/2026-05/05-18-executable-construction-protocol-registry-gate/prd.md`.
- Main source surfaces to inspect before implementation:
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`,
  `include/TianChenRV/Plugin/ConstructionProtocol.h`,
  `lib/Plugin/Construction/ConstructionProtocol.cpp`,
  Toy plugin construction/protocol files, Template plugin
  construction/protocol files, RVV and TensorExtLite plugin hook examples, and
  `test/Plugin/ConstructionProtocolCommonTest.cpp`.
- Minimal expected evidence:
  focused construction protocol common tests, affected Toy/Template/RVV/
  TensorExtLite plugin tests where present, `git diff --check`, practical
  `check-tianchenrv`, and bounded branch/residue scans.

## Completion Notes

- Live scan found current construction-capable builtin plugin surfaces for RVV,
  TensorExtLite, Toy, and Template. RVV and TensorExtLite already had
  `verifyExecutableConstructionConformance()` overrides; Toy and Template were
  the missing registration-time owners. Scalar and Offload had no construction
  protocol / construction manifest / construction artifact metadata surface in
  their builtin plugin directories.
- Toy and Template now override
  `ExtensionPlugin::verifyExecutableConstructionConformance()` and route
  registration through their existing `verify*ConstructionProtocolReady()`
  functions.
- Toy and Template readiness checks now construct a common
  `ConstructionConformanceGateSpec`, validating the manifest, typed-role graph
  realization, validation spec, and construction artifact metadata before the
  existing route and target bundle mapping checks.
- Added public Toy/Template construction artifact metadata helpers and
  metadata verifiers so stale registration fixtures and future plugin-local
  checks use the same construction metadata contract as production readiness.
- Extended `ConstructionProtocolCommonTest` to prove valid RVV,
  TensorExtLite, Toy, and Template builtins all register through the canonical
  `ExtensionPluginRegistry::registerPlugin()` boundary.
- Extended registry fail-closed coverage for a stale Toy construction manifest,
  stale Toy artifact metadata, and stale Template artifact metadata, all
  rejected before plugin acceptance with actionable diagnostics.
- Added the missing `TianChenRVRVVPlugin` test link dependency because the
  common construction test now directly proves RVV builtin registration, not
  only RVV construction-protocol helpers.
- Added no new lowering behavior, extension semantics, EmitC route kind,
  target artifact kind, descriptor/direct-C/source-export authority,
  compatibility wrapper, Python compiler-core logic, or family-specific branch
  in common construction code.

## Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-05-18-executable-construction-gate-remaining-builtins`
- `cmake --build build --target tianchenrv-construction-protocol-common-test -j2`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- `cmake --build build --target tianchenrv-toy-extension-plugin-test tianchenrv-template-extension-plugin-test tianchenrv-rvv-extension-plugin-test tianchenrv-tensorext-lite-extension-plugin-test -j2`
- `./build/bin/tianchenrv-toy-extension-plugin-test`
- `./build/bin/tianchenrv-template-extension-plugin-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- Construction-capable builtin scan:
  `rg -n "verify.*ConstructionProtocolReady|verifyExecutableConstructionConformance" include/TianChenRV/Plugin lib/Plugin`
  showed RVV, TensorExtLite, Toy, and Template have construction readiness
  surfaces and registration-time overrides, while the base default remains only
  the generic non-construction-capable hook.
- Scalar/Offload no-construction scan:
  `rg -n "ConstructionProtocol|ConstructionManifest|ConstructionArtifactMetadata" include/TianChenRV/Plugin/Scalar include/TianChenRV/Plugin/Offload lib/Plugin/Scalar lib/Plugin/Offload`
  returned no matches.
- Common construction family-name scan:
  `rg -n "RVV|TensorExt|Toy|Template|Offload|Scalar|rvv|tensorext|toy|template|offload|scalar" include/TianChenRV/Plugin/ConstructionProtocol.h lib/Plugin/Construction/ConstructionProtocol.cpp`
  returned no matches.
- Touched-surface descriptor/direct-C/source-export residue scan:
  `rg -n "descriptor-driven|descriptor|direct-C|direct C|source-export|source exporter|Python compiler-core|python compiler-core" include/TianChenRV/Plugin/Toy include/TianChenRV/Plugin/Template lib/Plugin/Toy lib/Plugin/Template test/Plugin/ConstructionProtocolCommonTest.cpp test/CMakeLists.txt`
  returned no matches.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` passed 122/122 lit
  tests.
