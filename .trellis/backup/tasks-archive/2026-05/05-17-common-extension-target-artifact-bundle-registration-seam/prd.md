# Common Extension Target Artifact Bundle Registration Seam

## Goal

Route built-in plugin-owned target artifact exporter bundles through one common
extension-bundle/front-door registry seam. The built-in target artifact
exporter surface may aggregate built-in extension bundle manifests, but it must
not directly include extension target-support headers or call
RVV/TensorExtLite/Toy-style target-support helpers as central route authority.

## Why Now

The previous TensorExtLite target artifact bridge made a second materialized
EmitC-derived artifact path real, but it still required direct
TensorExtLite-specific wiring in `lib/Target/Builtin`. That makes each future
extension-family artifact exporter a common target-code edit. This round
closes that registration seam so target support remains plugin/target-owned and
common target code consumes only generic extension bundle callbacks.

## Current Repository Facts

- `ExtensionBundle` already carries a
  `PluginTargetArtifactExporterBundleRegistrationFn`.
- `ExtensionBundleRegistry::registerTargetArtifactExportersForEnabledPlugins`
  already collects plugin-owned target exporter bundles and registers exporters
  only for enabled plugins.
- RVV and Toy already expose target-support metadata through
  `ExtensionPlugin::configureTargetSupportExtensionBundle`.
- TensorExtLite currently registers its target-support exporter from
  `BuiltinTargetArtifactExporters.cpp` by directly including
  `TensorExtLiteTargetSupportBundle.h`.
- Offload and Template currently receive required dialect metadata from
  built-in target code instead of their plugin manifest hook.
- Scalar intentionally contributes no target artifact exporter bundle in the
  current deletion/rebuild state.

## Requirements

- Keep target artifact route implementation in plugin/target-owned code.
- Make built-in extension bundle registration use a single generic list of
  built-in plugin registration callbacks plus bundle IDs.
- For manifest-owned bundles, instantiate the plugin once, call
  `configureTargetSupportExtensionBundle`, and register the resulting
  `ExtensionBundle`.
- Move TensorExtLite target-support bundle configuration behind the
  TensorExtLite plugin manifest hook.
- Move Offload and Template required dialect bundle metadata behind their
  plugin manifest hooks.
- Preserve existing RVV object/header bundle exporters and TensorExtLite header
  exporter behavior.
- Preserve Toy, Template, Offload, and Scalar absence of current target artifact
  route authority.
- Duplicate, missing, invalid, and malformed registrations must continue to
  fail closed with generic extension/target exporter diagnostics.

## Acceptance Criteria

- [x] `registerBuiltinTargetArtifactExporters` reaches RVV and TensorExtLite
  target artifact exporters through the common extension-bundle front door.
- [x] `BuiltinTargetArtifactExporters.cpp` no longer includes extension
  target-support bundle headers or directly calls RVV/TensorExtLite/Toy target
  support helpers.
- [x] Adding a future extension-family target exporter requires registering its
  plugin bundle/manifest hook, not hand-editing common built-in target exporter
  branch logic for the exporter.
- [x] Existing RVV object route, RVV header composite route, TensorExtLite
  header route, and Toy no-route behavior still pass through the front door.
- [x] Duplicate/missing/invalid registration behavior remains fail-closed with
  generic diagnostics.
- [x] No descriptor-driven compute, direct C/source-export route, source
  authority API, compatibility wrapper, or extension-specific semantic branch is
  restored.

## Out Of Scope

- No new plugin or new extension family.
- No new TensorExtLite/RVV compute route.
- No TensorExtLite object/bundle/runtime execution path.
- No RVV runtime or performance claim.
- No direct C semantic exporter, descriptor adapter, legacy mode, or Python
  compiler-core behavior.
- No broad smoke matrix beyond focused target/plugin checks unless needed for
  build confidence.

## Checks

- [x] Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-tensorext-lite-extension-plugin-test tianchenrv-rvv-extension-plugin-test tcrv-translate -j2`
- [x] Focused C++ tests:
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`, and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- [x] Focused lit:
  `/usr/bin/python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'tensorext-lite-target-artifact|source-seed-target-artifact|emitc-to-cpp-handoff|emitc-to-cpp-non-materialized|toy-template-target-artifact|target-artifact-export-registry|plan-and-export-target-artifact-bundle-no-viable'`,
  8/8 passed.
- [x] Full check:
  `cmake --build build --target check-tianchenrv -j2`, 100/100 lit tests
  passed.
- [x] `git diff --check`.
- [x] Trellis validation:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-common-extension-target-artifact-bundle-registration-seam`.
- [x] Targeted scans showing built-in target exporter code does not include
  extension target-support headers or direct helper calls, and no
  descriptor/direct-C/source-export authority is restored.
- [x] `clang-format` was not available in PATH; formatting was kept consistent
  by local style review and `git diff --check`.

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous task context:
  `.trellis/tasks/archive/2026-05/05-17-05-17-tensorextlite-materialized-emitc-target-artifact-bridge/prd.md`.
