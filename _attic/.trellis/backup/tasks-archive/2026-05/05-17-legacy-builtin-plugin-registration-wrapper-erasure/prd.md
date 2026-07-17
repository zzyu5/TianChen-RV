# Legacy built-in plugin registration wrapper erasure

## Goal

Delete the legacy plugin-only built-in registration compatibility API
`registerBuiltinExtensionPlugins` and the tests/comments that protect it as a
public construction path. Preserve the current canonical built-in route through
extension bundles and plugin registries.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round: HEAD
  `7a3c40b test: erase toy selected-boundary seed absence fixture`; worktree
  clean; no `.trellis/.current-task` existed.
- This is a Wrong Logic Deletion Campaign continuation: deletion before rebuild,
  no replacement architecture, no compatibility layer, and no helper-only
  progress.
- The previous archived task deleted the last directly related Toy
  selected-boundary seed named-absence fixture and did not add old option
  aliases, wrappers, descriptor adapters, source front doors, EmitC routes, or
  direct-C exporters.
- `include/TianChenRV/Plugin/BuiltinExtensionPlugins.h` declares
  `registerBuiltinExtensionPlugins`.
- `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp` defines that API only as a
  wrapper that creates a local `ExtensionBundleRegistry` and delegates to
  `registerBuiltinExtensionBundlePlugins`.
- `tools/tcrv-opt/tcrv-opt.cpp` and `tools/tcrv-translate/tcrv-translate.cpp`
  already use `registerBuiltinExtensionBundlePlugins`, not the legacy wrapper.
- Direct scan found legacy wrapper callers in active C++ tests beyond the
  specifically named target test:
  `test/Plugin/ScalarExtensionPluginTest.cpp`,
  `test/Plugin/ToyExtensionPluginTest.cpp`,
  `test/Plugin/OffloadExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`, and
  `test/Transforms/VariantSelection/VariantSelectionTest.cpp`.
- `test/Target/TargetArtifactExportTest.cpp` contains explicit compatibility
  coverage named `legacy built-in plugin registration delegates through bundle
  frontdoor` and a `compatibilityPlugins` registry.

## Boundaries

- Deletion/refactor-only. Remove the legacy wrapper declaration,
  implementation, active callers, tests, and comments that preserve it as a
  public plugin construction route.
- Keep the canonical `registerBuiltinExtensionBundlePlugins` and
  `registerBuiltinExtensionBundles` APIs.
- Existing hidden callers may be rewritten directly to the existing canonical
  bundle frontdoor by introducing a local `ExtensionBundleRegistry` at the call
  site. Do not introduce a new alias, wrapper, adapter, compatibility API, or
  legacy mode.
- Keep built-in extension bundle and plugin counts unchanged unless deletion
  exposes a real missing-architecture gap.
- Do not implement new plugin registry features, extension templates, target
  routes, source-front-door behavior, EmitC routes, descriptor adapters,
  direct-C exporters, Python compiler-core behavior, or broad documentation
  rewrites.
- Do not use this round to change extension-family semantics.

## Requirements

- Remove `registerBuiltinExtensionPlugins` from
  `BuiltinExtensionPlugins.h`.
- Remove the `registerBuiltinExtensionPlugins` wrapper definition from
  `BuiltinExtensionPlugins.cpp`.
- Rewrite active test callers to use
  `registerBuiltinExtensionBundlePlugins` with an explicit
  `ExtensionBundleRegistry`, or otherwise remove the test if its sole purpose
  was compatibility coverage.
- Remove the explicit `compatibilityPlugins` coverage and legacy registration
  strings from `TargetArtifactExportTest.cpp`.
- Confirm `tcrv-opt` and `tcrv-translate` continue to register built-ins
  through `registerBuiltinExtensionBundlePlugins`.
- If any non-test production caller still depends on the legacy wrapper, delete
  or rewire it to the existing canonical bundle route. If that cannot be done
  without design work, report it as a rebuild gap and do not restore the
  wrapper.

## Acceptance Criteria

- [x] `include/TianChenRV/Plugin/BuiltinExtensionPlugins.h` no longer declares
  `registerBuiltinExtensionPlugins`.
- [x] `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp` no longer defines the
  wrapper.
- [x] Active tests no longer assert
  `legacy built-in plugin registration delegates through bundle frontdoor` and
  no longer use `compatibilityPlugins` coverage.
- [x] Active include/lib/tools/test scans show no
  `registerBuiltinExtensionPlugins`, `compatibilityPlugins`, or
  `legacy built-in plugin registration` residue outside archived task history
  if unavoidable.
- [x] `tcrv-opt` and `tcrv-translate` still use the canonical
  `registerBuiltinExtensionBundlePlugins` frontdoor.
- [x] Focused C++ tests covering the touched plugin, target artifact exporter,
  variant selection, and tool registration behavior pass, or any failure is
  classified as an expected deletion gap without restoring the wrapper.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` is run if practical; otherwise the reason is recorded.

## Completion Evidence

- Removed the public legacy plugin-only wrapper declaration from
  `include/TianChenRV/Plugin/BuiltinExtensionPlugins.h`.
- Removed the wrapper implementation from
  `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp`; the remaining built-in
  registration APIs are `registerBuiltinExtensionBundles` and
  `registerBuiltinExtensionBundlePlugins`.
- Rewired hidden active C++ test callers in scalar, Toy, offload, target
  artifact export, and variant selection tests to construct an
  `ExtensionBundleRegistry` and call
  `registerBuiltinExtensionBundlePlugins` directly.
- Deleted the explicit `compatibilityPlugins` coverage and the legacy
  registration assertion from `TargetArtifactExportTest.cpp`.
- Confirmed `tools/tcrv-opt/tcrv-opt.cpp` and
  `tools/tcrv-translate/tcrv-translate.cpp` already use the canonical bundle
  frontdoor.
- Updated plugin-protocol specs to encode the durable rule: aggregate built-in
  registration must preserve bundle identity, and tests must exercise the
  canonical bundle frontdoor rather than a plugin-only compatibility alias.
- No compatibility layer, alias, wrapper, registry feature, extension template,
  source front door, target route, EmitC route, descriptor adapter, direct-C
  exporter, Python compiler-core behavior, or extension-family semantic change
  was added.

## Checks

- [x] Context validation:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-legacy-builtin-plugin-registration-wrapper-erasure`
  -> implement/check JSONL valid, 4 entries each.
- [x] Active legacy-residue scan:
  `rg -n "registerBuiltinExtensionPlugins|compatibilityPlugins|legacy built-in plugin registration" include lib tools test`
  -> no matches.
- [x] Active code/spec legacy-residue scan after spec update:
  `rg -n "registerBuiltinExtensionPlugins|compatibilityPlugins|legacy built-in plugin registration" include lib tools test .trellis/spec`
  -> no matches.
- [x] Canonical frontdoor scan confirmed remaining built-in aggregate calls are
  `registerBuiltinExtensionBundlePlugins` or
  `registerBuiltinExtensionBundles` in include/lib/tools/tests.
- [x] Focused build:
  `cmake --build build --target tianchenrv-scalar-extension-plugin-test tianchenrv-toy-extension-plugin-test tianchenrv-offload-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-variant-selection-test tcrv-opt tcrv-translate -j2`
  -> passed.
- [x] Focused C++ tests:
  `build/bin/tianchenrv-scalar-extension-plugin-test`,
  `build/bin/tianchenrv-toy-extension-plugin-test`,
  `build/bin/tianchenrv-offload-extension-plugin-test`,
  `build/bin/tianchenrv-target-artifact-export-test`, and
  `build/bin/tianchenrv-variant-selection-test` -> all passed.
- [x] Tool registration probes:
  `build/bin/tcrv-opt --help-hidden` showed the current built-in source
  front-door passes and source-artifact pipeline; `build/bin/tcrv-translate
  --help` showed built-in target artifact/emission translations.
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2`
  -> 122/122 passed.

## Spec Update Review

- Updated `.trellis/spec/plugin-protocol/extension-plugin-integration.md` with
  the built-in registration frontdoor rule.
- Updated `.trellis/spec/plugin-protocol/interfaces-and-registry.md` to replace
  the removed wrapper signature with the canonical bundle-aware signatures and
  test convention.

## Out of Scope

- Rebuilding plugin construction architecture.
- New extension/plugin templates or registry features.
- New source-front-door, target route, EmitC route, or runtime behavior.
- Descriptor-driven computation, direct C semantic export, source exporter
  adapters, compatibility shims, aliases, wrappers, or broad smoke matrices.
- New runtime correctness, performance, or `ssh rvv` claims.

## Technical Notes

- Specs read for this round:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`, and
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`.
- Previous task PRD read:
  `.trellis/tasks/archive/2026-05/05-17-legacy-source-front-door-named-absence-fixture-erasure/prd.md`.
- Relevant files read:
  `include/TianChenRV/Plugin/BuiltinExtensionPlugins.h`,
  `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp`,
  `tools/tcrv-opt/tcrv-opt.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Plugin/ScalarExtensionPluginTest.cpp`,
  `test/Plugin/ToyExtensionPluginTest.cpp`,
  `test/Plugin/OffloadExtensionPluginTest.cpp`, and
  `test/Transforms/VariantSelection/VariantSelectionTest.cpp`.
- Initial residue scan:
  `rg -n "registerBuiltinExtensionPlugins|compatibilityPlugins|legacy built-in plugin registration|registerBuiltinExtensionBundlePlugins|registerBuiltinExtensionBundles" include lib tools test CMakeLists.txt cmake`.
