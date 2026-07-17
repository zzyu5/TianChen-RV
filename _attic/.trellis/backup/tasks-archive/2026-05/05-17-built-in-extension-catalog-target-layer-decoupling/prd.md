# Built-in Extension Catalog Target-Layer Decoupling

## Goal

Move built-in extension bundle/catalog ownership out of target artifact exporter
aggregation. Target artifact export should consume an already assembled
extension bundle/plugin registry through the common front door; it must not
include concrete extension plugin headers or own the per-family built-in bundle
manifest.

## Why Now

Commit `937e745` made plugin-owned target artifact exporter bundle registration
real through `ExtensionBundleRegistry::registerTargetArtifactExporterBundles`
and routed current RVV/TensorExtLite artifact routes through extension bundle
hooks. The remaining cross-layer ownership is that
`lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` still includes concrete
RVV/Offload/Toy/Template/TensorExtLite/Scalar plugin headers and owns the
built-in extension bundle list. Adding a built-in extension should require a
plugin/catalog edit point, not a target artifact exporter edit.

## Current Repository Facts

- `include/TianChenRV/Plugin/BuiltinExtensionPlugins.h` and
  `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp` already own built-in plugin
  registration for RVV, Offload, Toy, Template, TensorExtLite, and Scalar.
- `ExtensionBundle` can carry a plugin registration callback, required dialect
  metadata, lowering-boundary metadata, and a plugin-owned target artifact
  exporter bundle registration callback.
- `ExtensionBundleRegistry::registerTargetArtifactExportersForEnabledPlugins`
  already collects plugin-owned target artifact exporter bundles and registers
  exporters only for enabled plugins with enabled dependencies.
- RVV and TensorExtLite currently configure target artifact exporter bundle
  hooks through their plugin `configureTargetSupportExtensionBundle` overrides.
- Toy, Template, Offload, and Scalar currently have no target artifact route
  authority; Template/Offload/Toy only contribute required dialect metadata,
  and Scalar contributes no current target-support bundle metadata.
- `BuiltinTargetArtifactExporters.cpp` currently contains the concrete
  `kBuiltinExtensionBundles[]` list and directly includes concrete plugin
  headers, which is the ownership to remove in this round.

## Requirements

- Introduce or reuse a common built-in extension catalog/registry boundary owned
  outside target artifact export.
- Move concrete built-in extension bundle manifest ownership into the plugin
  catalog layer, reusing the existing built-in plugin registration surface where
  appropriate.
- Keep `ExtensionBundleRegistry` and the plugin-owned target artifact exporter
  bundle seam from `937e745`.
- Keep target artifact export responsible for artifact route validation,
  candidate coherence, generic front-door selection, and exporter invocation.
- Make target artifact export receive extension bundles/plugins through the
  common catalog/front door; it must not know which extension families exist.
- Preserve current route behavior: RVV object/header composite and
  TensorExtLite header routes remain available; Toy, Template, Offload, and
  Scalar do not gain target artifact route authority.
- Keep disabled or missing plugin dependencies fail-closed.
- Do not add a compatibility layer that restores descriptor/direct-C/source
  export authority.

## Acceptance Criteria

- [x] `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` no longer
      includes RVV, Offload, Toy, Template, TensorExtLite, or Scalar plugin
      headers.
- [x] `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` no longer owns a
      hard-coded per-extension bundle manifest list.
- [x] A future built-in extension insertion has one clear plugin/catalog
      registry edit point outside target artifact export.
- [x] Built-in registration still publishes the same current artifact routes
      through extension bundle hooks.
- [x] RVV and TensorExtLite positive route counts remain stable.
- [x] Toy, Template, Offload, and Scalar still do not gain current artifact
      route authority.
- [x] Disabled or missing plugin dependencies still fail closed through the
      existing registry path.
- [x] No descriptor-driven compute, direct C/source-export route, source
      authority API, Python compiler-core behavior, new artifact feature, or
      extension-specific semantic branch in common target code is added.

## Out Of Scope

- No new RVV or TensorExtLite artifact kinds.
- No new header/object/bundle routes.
- No new extension families.
- No generic RVV lowering, runtime execution, or performance claim.
- No descriptor adapters, compatibility wrappers, direct C semantic exporters,
  source-export routes, or Python compiler-core behavior.
- No broad test matrix beyond focused target/plugin/catalog checks unless
  required by build failures.

## Checks

- [x] Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-translate tcrv-opt -j2`.
- [x] Focused plugin build:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-tensorext-lite-extension-plugin-test -j2`.
- [x] Focused C++ tests:
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-rvv-extension-plugin-test`, and
  `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`.
- [x] Focused lit from `build/test`:
  `/usr/bin/python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'tensorext-lite-target-artifact|source-seed-target-artifact|emitc-to-cpp-handoff|emitc-to-cpp-non-materialized|toy-template-target-artifact|target-artifact-export-registry|plan-and-export-target-artifact-bundle-no-viable'`,
  8/8 passed.
- [x] Full check:
  `cmake --build build --target check-tianchenrv -j2`, 100/100 lit tests
  passed.
- [x] `git diff --check`.
- [x] Trellis validation:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-built-in-extension-catalog-target-layer-decoupling`.
- [x] Targeted scan:
  `rg -n "Plugin/(RVV|Offload|Toy|Template|TensorExtLite|Scalar)/.*ExtensionPlugin|kBuiltinExtensionBundles|BuiltinExtensionBundleSpec|registerRVVExtensionPlugin|registerOffloadExtensionPlugin|registerToyExtensionPlugin|registerTemplateExtensionPlugin|registerTensorExtLiteExtensionPlugin|registerScalarExtensionPlugin" lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`
  returned no matches.
- [x] Targeted diff scan:
  `git diff -U0 -- include lib tools test | rg -n "descriptor|source-export|source_authority|runtime-callable-c-source|direct C|raw C|C source|metadata-driven"`
  returned no matches.

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Previous task context:
  `.trellis/tasks/archive/2026-05/05-17-common-extension-target-artifact-bundle-registration-seam/prd.md`.
- Current intended boundary: plugin/catalog owns the built-in extension bundle
  manifest; target builtin artifact export delegates to that catalog and then
  consumes `ExtensionBundleRegistry` plus `ExtensionPluginRegistry`.
- Implemented boundary: `Plugin/BuiltinExtensionPlugins` owns the concrete
  built-in extension bundle manifest and `registerBuiltinExtensionBundles`;
  target builtin artifact export only delegates to that catalog API before
  invoking the existing `ExtensionBundleRegistry` front door.
