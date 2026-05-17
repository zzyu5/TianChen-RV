# Built-in non-plugin target artifact placeholder erasure

## Goal

Delete the remaining built-in non-plugin target artifact exporter placeholder
from the production target registration path. After this round,
`registerBuiltinTargetArtifactExporters` must be an explicit bundle-aware
composition surface:

```text
ExtensionBundleRegistry + ExtensionPluginRegistry
  -> enabled extension bundle target exporters
```

There must be no empty pre-bundle helper, compatibility lane, comment, spec
wording, or test expectation implying that future non-plugin built-in target
artifact exporters can register ahead of or outside enabled extension bundles.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round: HEAD
  `368b35d plugin: erase builtin target registration wrappers`; worktree clean;
  no `.trellis/.current-task` existed.
- The previous completed target-registration task removed public
  target-side aggregate wrapper overloads and rewired active tools/tests to
  pass explicit `ExtensionBundleRegistry` plus `ExtensionPluginRegistry`.
- One production residue remains in
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`:
  `registerBuiltinNonPluginTargetArtifactExporters`.
- That helper currently returns success after discarding the registry, but its
  call before `registerTargetArtifactExportersForEnabledExtensionBundles`
  preserves a hidden non-plugin built-in target artifact lane.
- Current target translate route registration is already bundle-aware and
  should remain so.
- Current Toy, RVV, TensorExtLite, and Template target artifact exporters are
  expected to register through enabled extension bundle target-support hooks.

## Boundaries

- Deletion/refactor-only continuation of target registration wrapper erasure.
- Keep the common `TargetArtifactExporterRegistry` and the existing explicit
  bundle-aware public target registration API.
- Keep tools/tests constructing visible built-in extension bundle/plugin
  registries at the boundary.
- Remove only the non-plugin placeholder and directly protective residue.
- Preserve fail-closed behavior for disabled or missing plugins and malformed
  bundle target-support contributions.
- Do not add new target artifact kinds, target translate routes, extension
  plugin features, compatibility wrappers, aliases, descriptor routes,
  direct-C/source-export paths, Python compiler-core logic, or replacement
  architecture.
- Do not rebuild RVV, header, bundle, source, Toy, TensorExtLite, Template,
  Scalar, Offload, IME, or future extension behavior in this round.

## Requirements

- Delete `registerBuiltinNonPluginTargetArtifactExporters` from
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`.
- Make `registerBuiltinTargetArtifactExporters` directly delegate to
  `registerTargetArtifactExportersForEnabledExtensionBundles`, or to an
  equivalently explicit path that consumes the supplied bundle and plugin
  registries.
- Remove active spec/test/comment residue that frames non-plugin built-in
  target artifact exporters as a current or future registration lane before
  enabled extension bundle processing.
- Keep `registerBuiltinTargetTranslateRoutes` bundle-aware.
- Keep existing Toy/RVV/TensorExtLite/Template target artifact route visibility
  flowing through enabled extension bundle registration.
- Keep disabled and missing plugin cases fail-closed.

## Acceptance Criteria

- [x] `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` no longer
  defines or calls `registerBuiltinNonPluginTargetArtifactExporters`.
- [x] `registerBuiltinTargetArtifactExporters` directly uses the supplied
  `ExtensionBundleRegistry` and `ExtensionPluginRegistry` to register enabled
  extension bundle target exporters.
- [x] Focused scans over `include/TianChenRV/Target`, `lib/Target/Builtin`,
  `tools/tcrv-translate`, and `test/Target` show no
  `registerBuiltinNonPluginTargetArtifactExporters` and no public one- or
  two-argument built-in target wrapper signatures.
- [x] Focused scans over active target specs/tests show no wording or coverage
  protecting a non-plugin built-in target artifact exporter lane.
- [x] Existing Toy, RVV, TensorExtLite, and Template target artifact exporters
  still register through enabled extension bundle target-support hooks.
- [x] Disabled/missing plugin cases still fail closed.
- [x] No compatibility wrapper, legacy alias, descriptor route,
  direct-C/source-export path, new artifact route, or new target translate
  route is introduced.
- [x] Focused build passes for `tcrv-translate` and
  `tianchenrv-target-artifact-export-test`.
- [x] `tianchenrv-target-artifact-export-test` passes.
- [x] Focused lit/help checks prove current target artifact and materialized
  EmitC translate routes remain visible through bundle-aware registration.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` is run if practical; otherwise the reason is
  recorded.

## Completion Evidence

- Removed the empty anonymous-namespace helper
  `registerBuiltinNonPluginTargetArtifactExporters` from
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`.
- Rewired `registerBuiltinTargetArtifactExporters` to directly return
  `registerTargetArtifactExportersForEnabledExtensionBundles(bundles, plugins,
  registry)`.
- Updated `.trellis/spec/lowering-runtime/emission-runtime-contract.md` so the
  durable target artifact registration boundary now states that built-in
  target artifact routes register only through enabled extension bundle
  target-support contributions, and that no pre-bundle or non-plugin built-in
  target artifact exporter lane exists.
- No public target API signature changed in this round; the surviving public
  signature remains the explicit bundle-aware
  `registerBuiltinTargetArtifactExporters(TargetArtifactExporterRegistry &,
  const ExtensionBundleRegistry &, const ExtensionPluginRegistry &)`.
- No compatibility layer, renamed helper, target route, artifact kind,
  descriptor route, direct-C/source-export path, Python compiler-core behavior,
  or extension-specific common/core/tool branch was added.

## Checks

- [x] Trellis context validation:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-05-17-builtin-non-plugin-target-artifact-placeholder-erasure`
  -> implement/check JSONL valid, 4 entries each.
- [x] Placeholder scan:
  `rg -n "registerBuiltinNonPluginTargetArtifactExporters" include/TianChenRV/Target lib/Target/Builtin tools/tcrv-translate test/Target .trellis/spec`
  -> no matches.
- [x] Deleted public wrapper signature scan:
  `rg -n -U "<one-/two-argument built-in target wrapper signatures>" include/TianChenRV/Target lib/Target/Builtin tools/tcrv-translate test/Target .trellis/spec`
  -> no matches.
- [x] Protective residue scan:
  `rg -n "non-plugin target|non plugin target|single-candidate route set|registered ahead|outside enabled extension|register.*ahead|pre-bundle|non-plugin built-in target artifact exporter lane" include/TianChenRV/Target lib/Target/Builtin tools/tcrv-translate test/Target .trellis/spec/plugin-protocol .trellis/spec/lowering-runtime`
  -> only the new prohibitive spec sentence remains:
  `There is no pre-bundle or non-plugin built-in target artifact exporter lane.`
- [x] Focused build:
  `cmake --build build --target tcrv-translate tianchenrv-target-artifact-export-test -j2`
  -> passed.
- [x] Focused C++ test:
  `./build/bin/tianchenrv-target-artifact-export-test`
  -> passed.
- [x] Help surface probe:
  `./build/bin/tcrv-translate --help 2>&1 | rg "tcrv-export-target-artifact|tcrv-export-target-header-artifact|tcrv-export-target-artifact-bundle|tcrv-rvv-emitc-to-cpp|tcrv-template-emitc-to-cpp|tcrv-tensorext-lite-emitc-to-cpp"`
  -> passed and showed the current generic target artifact routes plus RVV,
  Template, and TensorExtLite materialized EmitC translate routes.
- [x] Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='Target/(RVV/emitc-to-cpp-handoff|Template/template-emitc-to-cpp|TensorExtLite/tensorext-lite-source-front-door-emitc-to-cpp|RVV/vector-materialized-target-artifact-exporters|Toy/toy-materialized-target-artifact-object|Template/template-target-artifact-object|TensorExtLite/tensorext-lite-target-artifact-header)'`
  from `build/test`
  -> 8/8 focused tests passed.
- [x] `git diff --check`
  -> passed.
- [x] `cmake --build build --target check-tianchenrv -j2`
  -> 122/122 lit tests passed.

## Definition of Done

- Trellis task status and notes accurately describe the deletion/refactor.
- Journal records exact residue removed, scans, checks, self-repair, archive
  status, and commit hash.
- One coherent commit is created if the task completes.

## Out of Scope

- New executable plugin routes or target artifact kinds.
- New RVV/header/bundle/source rebuild work.
- Extension family expansion or finite-family enumeration.
- Broad test matrix expansion beyond focused validation unless required by a
  failure.
- Restoring old built-in wrapper overloads or preserving a renamed non-plugin
  helper.

## Technical Notes

- Specs read for this round:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Previous task PRD read:
  `.trellis/tasks/archive/2026-05/05-17-05-17-builtin-target-registration-wrapper-erasure/prd.md`.
- Relevant files inspected:
  `include/TianChenRV/Target/BuiltinTargetArtifactExporters.h`,
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`,
  `lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.
