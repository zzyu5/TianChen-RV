# Extension Bundle Interface Target-Layer Extraction

## Goal

Move the generic extension bundle, bundle registry, and bundle configuration
contract out of target artifact export and into the plugin-facing common
extension interface. Target artifact export remains a consumer of plugin-owned
bundle and exporter registries; it no longer owns the extension-family
construction surface.

## Why Now

Commit `d557f3a` moved the concrete built-in bundle catalog out of
`lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` and into
`Plugin/BuiltinExtensionPlugins`. The remaining layering bottleneck is that the
common `ExtensionBundle`, `ExtensionBundleRegistry`, and
`ExtensionPluginRegistrationFn` abstractions still live in
`include/TianChenRV/Target/TargetArtifactExport.h`, which makes the built-in
plugin catalog depend on target artifact export types to describe
extension-family construction.

## Current Repository Facts

- `include/TianChenRV/Target/TargetArtifactExport.h` currently defines both
  target artifact exporter APIs and the generic extension bundle registry APIs.
- `include/TianChenRV/Plugin/BuiltinExtensionPlugins.h` forward-declares
  `target::ExtensionBundleRegistry`, so the plugin catalog still exposes a
  target-layer type in its public bundle registration API.
- `ExtensionPlugin::configureTargetSupportExtensionBundle` and concrete plugin
  overrides currently take `target::ExtensionBundle &`.
- `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp` owns the built-in bundle
  manifest, but includes `TargetArtifactExport.h` only to construct the generic
  bundle records.
- Target artifact export still owns exporter registries, artifact candidate
  validation, composite export, materialized EmitC artifact export, and bundle
  output behavior; those target responsibilities remain in target.
- RVV and TensorExtLite currently configure target artifact exporter bundle
  hooks through plugin-owned `configureTargetSupportExtensionBundle` hooks.

## Requirements

- Introduce a plugin/common header for the generic extension bundle surface.
- Move `ExtensionPluginRegistrationFn`, `ExtensionBundle`, and
  `ExtensionBundleRegistry` to the plugin/common namespace and implementation
  library.
- Rewire `ExtensionPlugin::configureTargetSupportExtensionBundle` and all
  concrete overrides to use the rehomed bundle type.
- Rewire `Plugin/BuiltinExtensionPlugins` to register built-in extension
  bundles without depending on `target::ExtensionBundleRegistry` from
  `TargetArtifactExport.h`.
- Keep target artifact exporter registries and target artifact export behavior
  in the target layer.
- Keep built-in target artifact exporter aggregation generic: it may consume
  plugin bundle registries and target exporter bundle registries, but it must
  not regain concrete plugin headers or hard-coded per-extension route lists.
- Preserve route IDs and artifact behavior for RVV object/header routes and
  TensorExtLite header routes.
- Preserve disabled-plugin and missing-plugin dependency fail-closed behavior.

## Acceptance Criteria

- [x] `include/TianChenRV/Target/TargetArtifactExport.h` no longer defines
      `ExtensionBundle`, `ExtensionBundleRegistry`, or
      `ExtensionPluginRegistrationFn`.
- [x] `Plugin/BuiltinExtensionPlugins` uses a plugin/common bundle registry type
      and no longer forward-declares or exposes `target::ExtensionBundleRegistry`.
- [x] `ExtensionPlugin::configureTargetSupportExtensionBundle` and concrete
      plugin overrides use the rehomed common bundle interface.
- [x] Built-in target artifact and target translate aggregators remain generic
      registry consumers and do not include concrete plugin headers or own
      per-extension catalog lists.
- [x] Existing RVV object/header and TensorExtLite header route behavior is
      unchanged.
- [x] Disabled-plugin and missing-plugin target exporter bundle dependencies
      continue to fail closed.
- [x] No descriptor-driven compute, direct C/source-export authority, Python
      compiler-core behavior, new artifact route, or extension-specific core
      semantic branch is introduced.

## Out Of Scope

- No new RVV, TensorExtLite, Offload, scalar, Toy, Template, or IME routes.
- No new artifact kinds, bundle formats, or route IDs.
- No descriptor adapters, direct C semantic exporters, source-export routes, or
  compatibility wrappers.
- No movement of real C/C++ emitter logic out of target artifact export.
- No broad repo cleanup beyond files needed for this interface extraction.

## Checks

- [x] Focused build for affected plugin, target, and tool targets.
- [x] `./build/bin/tianchenrv-target-artifact-export-test`.
- [x] RVV and TensorExtLite plugin tests if bundle hook signatures change.
- [x] Focused lit for target artifact/export/translate front doors if touched.
- [x] `cmake --build build --target check-tianchenrv -j2` when practical.
- [x] `git diff --check`.
- [x] Targeted scans over `lib/Target/Builtin`,
      `include/TianChenRV/Target/TargetArtifactExport.h`,
      `include/TianChenRV/Plugin`, and `lib/Plugin/Builtin` proving the target
      built-in aggregation does not own concrete plugin catalog lists and no
      descriptor/direct-C/source-export route authority was introduced.
- [x] Trellis task validation before archive and after archived context path
      refresh.

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Previous task context:
  `.trellis/tasks/archive/2026-05/05-17-built-in-extension-catalog-target-layer-decoupling/prd.md`.
- Intended boundary after this round:
  `Plugin/ExtensionBundle` owns the generic extension-family bundle registry;
  `TargetArtifactExport` owns target exporter and artifact export mechanics;
  built-in plugin and target front doors connect those registries without
  reintroducing target-layer catalog ownership.
