# Built-in target registration wrapper erasure

## Goal

Delete the target-side built-in aggregate registration wrappers that construct
or hide the canonical extension bundle front door. Preserve the existing
bundle-aware target artifact exporter and target translate route registration
surfaces, and rewrite active tool/test callers so bundle and plugin identity is
explicit at the call site.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round: HEAD
  `373fe0e plugin: erase legacy builtin registration wrapper`; worktree clean;
  no `.trellis/.current-task` existed.
- The previous completed task removed the plugin-only
  `registerBuiltinExtensionPlugins` compatibility API and established
  `registerBuiltinExtensionBundles` plus
  `registerBuiltinExtensionBundlePlugins` as the canonical built-in front door.
- Current target-facing wrappers still hide the same bundle/plugin front door:
  `registerBuiltinTargetArtifactExporters(TargetArtifactExporterRegistry &)`,
  `registerBuiltinTargetArtifactExporters(TargetArtifactExporterRegistry &, const ExtensionPluginRegistry &)`,
  and `registerBuiltinTargetTranslateRoutes(TargetTranslateRouteRegistry &)`.
- Current bundle-aware target surfaces already exist:
  `registerBuiltinTargetArtifactExporters(TargetArtifactExporterRegistry &, const ExtensionBundleRegistry &, const ExtensionPluginRegistry &)`
  and
  `registerBuiltinTargetTranslateRoutes(TargetTranslateRouteRegistry &, const ExtensionBundleRegistry &, const ExtensionPluginRegistry &)`.
- `tools/tcrv-translate/tcrv-translate.cpp` and
  `test/Target/TargetArtifactExportTest.cpp` still exercise hidden aggregate
  wrappers that this round must delete or rewrite.
- This is part of the Wrong Logic Deletion Campaign: deletion before rebuild,
  no new target route, no new artifact behavior, no compatibility alias, and no
  replacement architecture in the same round.

## Boundaries

- Deletion/refactor-only. Remove public target wrapper overloads that internally
  construct built-in bundles/plugins or accept plugins without the matching
  bundle registry.
- Keep the existing bundle-aware target artifact exporter and target translate
  route registration APIs.
- Rewrite active callers/tests to explicitly create and populate
  `ExtensionBundleRegistry` and `ExtensionPluginRegistry` before registering
  target artifact exporters or target translate routes.
- Preserve duplicate, missing-plugin, disabled-plugin, and malformed route
  fail-closed coverage through the bundle-aware surfaces.
- Do not add new target routes, artifact kinds, plugin behavior, evidence
  matrices, EmitC routes, descriptors, direct C semantic exporters, source
  export paths, Python compiler-core logic, wrappers, aliases, or compatibility
  overloads.
- Do not rebuild RVV, Toy, TensorExtLite, Template, Scalar, Offload, IME, or
  any future extension emission behavior.

## Requirements

- Remove the zero-argument target artifact exporter wrapper declaration and
  definition:
  `registerBuiltinTargetArtifactExporters(TargetArtifactExporterRegistry &)`.
- Remove the plugin-only target artifact exporter wrapper declaration and
  definition:
  `registerBuiltinTargetArtifactExporters(TargetArtifactExporterRegistry &, const ExtensionPluginRegistry &)`.
- Remove the zero-argument target translate route wrapper declaration and
  definition:
  `registerBuiltinTargetTranslateRoutes(TargetTranslateRouteRegistry &)`.
- Keep and use only the bundle-aware target registration surfaces that accept
  both `ExtensionBundleRegistry` and `ExtensionPluginRegistry`.
- Rewire `tcrv-translate` setup so each active registration path exposes the
  same explicit built-in bundle and plugin registries it consumes.
- Rewrite target tests so wrapper-protecting cases no longer call zero-argument
  or plugin-only built-in target registration helpers.
- Update durable specs that still present hidden aggregate target wrappers as
  the correct public API.

## Acceptance Criteria

- [x] Public target headers no longer expose wrapper overloads that internally
  construct built-in bundles/plugins or accept plugins without the matching
  bundle registry.
- [x] `lib/Target/Builtin` no longer defines those hidden wrapper overloads.
- [x] `tcrv-translate` uses explicit `ExtensionBundleRegistry` plus
  `ExtensionPluginRegistry` setup for target artifact exporter and target
  translate route registration.
- [x] Target C++ tests use explicit bundle-aware target registration surfaces
  and no longer protect zero-argument or plugin-only target built-in
  registration wrappers.
- [x] Duplicate, missing-plugin, disabled-plugin, malformed exporter, and
  malformed translate-route fail-closed behavior remains covered through the
  bundle-aware surfaces.
- [x] Focused scans over `include`, `lib`, `tools`, `test`, and
  `.trellis/spec` show the deleted target wrapper signatures and
  wrapper-protecting tests are gone, while `registerBuiltinExtensionBundles`,
  `registerBuiltinExtensionBundlePlugins`, and explicit bundle-aware target
  registration surfaces remain.
- [x] Focused build for `tcrv-translate` and the target artifact export C++
  test passes.
- [x] Rewritten target C++ test and relevant translate help/probe checks pass.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` is run if practical; otherwise the reason is recorded.

## Completion Evidence

- Removed the public zero-argument and plugin-only target artifact exporter
  overloads from `include/TianChenRV/Target/BuiltinTargetArtifactExporters.h`.
- Removed the public zero-argument target translate route overload from
  `include/TianChenRV/Target/BuiltinTargetTranslateRoutes.h`.
- Deleted the corresponding wrapper definitions from
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` and
  `lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp`.
- Rewired `tools/tcrv-translate/tcrv-translate.cpp` so target translate route
  registration explicitly builds the built-in extension bundle front door and
  passes the bundle/plugin pair into the bundle-aware target route surface.
- Rewired `test/Target/TargetArtifactExportTest.cpp` so built-in target
  exporter and target translate route checks use explicit
  `ExtensionBundleRegistry` plus `ExtensionPluginRegistry` setup.
- Preserved fail-closed coverage for missing plugins, disabled plugins,
  failing translate contributors, duplicate target artifact routes, repeated
  route registration, and offload route absence through the bundle-aware
  surfaces.
- Updated `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md` so durable
  target registration contracts no longer present hidden aggregate wrappers as
  correct API.
- No compatibility layer, alias, wrapper, target route, artifact kind, plugin
  behavior, evidence matrix, EmitC route, descriptor, direct C exporter,
  source-export path, Python compiler-core behavior, or extension-specific
  common/core/tool branch was added.

## Checks

- [x] Context validation:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-05-17-builtin-target-registration-wrapper-erasure`
  -> implement/check JSONL valid, 4 entries each.
- [x] Focused build:
  `cmake --build build --target tcrv-translate tianchenrv-target-artifact-export-test -j2`
  -> passed.
- [x] Rewritten target C++ test:
  `build/bin/tianchenrv-target-artifact-export-test`
  -> passed.
- [x] Translate command surface probe:
  `build/bin/tcrv-translate --help 2>&1 | rg "tcrv-export-target-artifact|tcrv-export-target-header-artifact|tcrv-export-target-artifact-bundle|tcrv-source-artifact-bundle-front-door|tcrv-rvv-emitc-to-cpp|tcrv-template-emitc-to-cpp|tcrv-tensorext-lite-emitc-to-cpp"`
  -> passed and showed the current target artifact and materialized EmitC
  translate commands.
- [x] Deleted target wrapper signature scan:
  `rg -n -U "registerBuiltinTargetArtifactExporters\\s*\\(\\s*TargetArtifactExporterRegistry\\s*&\\s*registry\\s*\\)|registerBuiltinTargetArtifactExporters\\s*\\(\\s*TargetArtifactExporterRegistry\\s*&\\s*registry\\s*,\\s*const\\s+plugin::ExtensionPluginRegistry\\s*&\\s*plugins\\s*\\)|registerBuiltinTargetTranslateRoutes\\s*\\(\\s*TargetTranslateRouteRegistry\\s*&\\s*registry\\s*\\)" include lib tools test .trellis/spec`
  -> no matches.
- [x] Wrapper-protecting call/test scan:
  `rg -n "plugin-only target|zero-argument target|wrapper-protecting|registerBuiltinTargetArtifactExporters\\([^,)]*\\)|registerBuiltinTargetTranslateRoutes\\([^,)]*\\)" include lib tools test .trellis/spec`
  -> no matches.
- [x] Remaining front-door scan:
  `rg -n "registerBuiltinExtensionBundles|registerBuiltinExtensionBundlePlugins|registerBuiltinTargetArtifactExporters|registerBuiltinTargetTranslateRoutes" include/TianChenRV/Target lib/Target/Builtin tools/tcrv-translate test/Target .trellis/spec/plugin-protocol .trellis/spec/lowering-runtime`
  -> only canonical bundle front-door helpers and explicit bundle-aware target
  registration surfaces remain.
- [x] `git diff --check`
- [x] `cmake --build build --target tcrv-opt check-tianchenrv -j2`
  -> `tcrv-opt` built and `check-tianchenrv` passed 122/122.

## Definition of Done

- Trellis task status and notes reflect the deletion/refactor result.
- Any spec update is limited to durable target/bundle front-door rules, not
  current progress narration.
- Completion evidence records exact wrappers removed, caller/test rewrites,
  scans, checks, self-repair, archive status, and commit hash.

## Out of Scope

- New RVV, Toy, TensorExtLite, Template, Scalar, Offload, IME, or future
  extension behavior.
- New header/object/bundle features, target routes, artifact kinds, EmitC
  routes, descriptors, direct C semantic exporters, source-export paths,
  Python compiler-core logic, wrappers, aliases, compatibility overloads,
  legacy modes, or extension-specific branches in common/core/tool code.
- Runtime correctness, performance, or new `ssh rvv` evidence claims.

## Technical Notes

- Specs read for this round:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Previous task PRD read:
  `.trellis/tasks/archive/2026-05/05-17-legacy-builtin-plugin-registration-wrapper-erasure/prd.md`.
- Relevant files inspected:
  `include/TianChenRV/Target/BuiltinTargetArtifactExporters.h`,
  `include/TianChenRV/Target/BuiltinTargetTranslateRoutes.h`,
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`,
  `lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.
- Initial residue scan command:
  `rg -n "registerBuiltinTargetArtifactExporters|registerBuiltinTargetTranslateRoutes|TargetArtifactExporterRegistry|TargetTranslateRouteRegistry|ExtensionBundleRegistry|ExtensionPluginRegistry" include/TianChenRV/Target lib/Target tools/tcrv-translate test/Target test/Plugin .trellis/spec`.
