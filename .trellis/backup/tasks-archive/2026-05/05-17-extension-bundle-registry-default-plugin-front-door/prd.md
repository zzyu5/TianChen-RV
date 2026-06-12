# Extension Bundle Registry Default Plugin Front Door

## Goal

Make `plugin::ExtensionBundleRegistry` the canonical built-in extension
construction front door for default tool and target registration paths. The
default route should be:

```text
built-in extension bundle catalog
  -> plugin::ExtensionBundleRegistry
  -> enabled plugin::ExtensionPluginRegistry
  -> dialect registration inputs
  -> target artifact exporter bundle consumers
  -> target translate route consumers
```

Target artifact exporter registries and target translate route registries remain
target-owned consumers. This round rewires default production paths; it does not
add a new extension family or emission route.

## Why Now

Commit `fca4211` extracted `ExtensionBundle` and
`ExtensionBundleRegistry` into plugin/common and kept target artifact export as a
consumer. The remaining bottleneck is that default production surfaces still
seed `ExtensionPluginRegistry` directly through `registerBuiltinExtensionPlugins`
in `tools/tcrv-opt`, `tools/tcrv-translate`,
`lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`, and
`lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp`. That keeps the new bundle
registry as a target-export helper instead of the canonical construction front
door.

## Current Repository Facts

- `.trellis/spec/index.md` requires C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck
  for compiler functionality and rejects descriptor-driven computation as
  current architecture.
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md` requires plugin
  registration and public source-seed pass visibility to flow through plugin
  interfaces/registries, not family-specific core branches.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` keeps current
  production emission on extension-family ops -> EmitC -> C/C++ emitter and
  rejects descriptor/direct-C source authority.
- `plugin::ExtensionBundleRegistry` already owns bundle validation,
  bundle-to-plugin registration, and plugin-owned target artifact exporter
  bundle collection.
- `registerBuiltinExtensionBundles` owns the concrete built-in bundle catalog.
- `registerBuiltinExtensionPlugins` still loops the built-in concrete catalog
  directly, so it is a second default construction authority.
- `tcrv-opt` and `tcrv-translate` default paths call
  `registerBuiltinExtensionPlugins` directly before dialect/pass/exporter setup.
- Built-in target artifact exporter no-arg registration calls
  `registerBuiltinExtensionPlugins` directly before consuming bundle metadata.
- Built-in target translate route registration calls
  `registerBuiltinExtensionPlugins` directly and then iterates plugins.
- Target artifact exporter registration can wrap bundle registration errors, but
  exporter callback failures after plugin-exporter bundle collection do not yet
  name the originating extension bundle.

## Requirements

- Add a built-in bundle front-door helper that populates an
  `ExtensionBundleRegistry` from the built-in catalog and obtains the enabled
  `ExtensionPluginRegistry` through `ExtensionBundleRegistry`.
- Make `registerBuiltinExtensionPlugins` delegate through the bundle front door
  so it remains only a compatibility/test escape hatch, not an independent
  production catalog loop.
- Rewire `tcrv-opt` default initialization to build plugins and target artifact
  exporters from the built-in `ExtensionBundleRegistry`.
- Rewire `tcrv-translate` default dialect, planning, emission manifest, target
  artifact, target artifact bundle, and target translate route initialization to
  use the bundle front door.
- Rewire built-in target artifact exporter default registration to consume a
  caller-provided `ExtensionBundleRegistry` plus the enabled
  `ExtensionPluginRegistry`, with the existing explicit-plugin overload kept as
  a low-level escape hatch.
- Rewire built-in target translate route registration to consume the same
  bundle-owned plugin registry path and to skip missing/disabled plugins without
  publishing routes.
- Preserve target-owned artifact and translate registry semantics; do not move
  route/export behavior into plugin/common.
- Improve error propagation so bundle registration, plugin registration, target
  artifact exporter registration, and target translate route registration name
  the responsible bundle and plugin where applicable.
- Preserve the `tcrv-opt --tcrv-disable-builtin-plugins` empty-registry escape
  hatch.
- Preserve current RVV, Toy, Template, TensorExtLite, Offload, and scalar
  behavior through the new front door.

## Acceptance Criteria

- [x] `tcrv-opt` default initialization obtains built-in plugin registries from
      `ExtensionBundleRegistry`, then registers plugin dialects, source-seed
      passes, pipelines, and target artifact exporters from that registry.
- [x] `tcrv-translate` default initialization obtains built-in plugin
      registries from `ExtensionBundleRegistry` for dialects, planning
      registries, emission manifest export, artifact front doors, bundle export,
      and target translate route registration.
- [x] Built-in target artifact exporter registration has a production overload
      that consumes an `ExtensionBundleRegistry` and the enabled
      `ExtensionPluginRegistry`; the no-arg overload builds both through the
      bundle front door.
- [x] Built-in target translate route registration consumes an
      `ExtensionBundleRegistry` and the enabled `ExtensionPluginRegistry`; the
      no-arg overload builds both through the bundle front door.
- [x] `registerBuiltinExtensionPlugins` delegates through the bundle registry and
      is not a second independent default catalog loop.
- [x] Errors identify the responsible bundle/plugin for bundle registration,
      plugin registration, target exporter registration, and target translate
      route registration failures.
- [x] Missing or disabled plugins remain fail-closed: no routes/exporters are
      exposed from disabled or absent plugins in bundle-driven default consumers.
- [x] Existing RVV/Toy/Template/TensorExtLite/Offload/scalar registration shape
      and current built-in artifact/translate route counts remain unchanged.
- [x] No descriptor-driven compute, direct C/source-export authority, Python
      compiler-core behavior, new artifact route, or extension-specific core
      semantic branch is introduced.

## Out Of Scope

- No new plugin, extension family, dialect, pass, route, artifact kind, or
  evidence matrix.
- No new RVV/IME/Offload/TensorExt lowering or EmitC emission.
- No descriptor compatibility path, direct C semantic exporter, source-export
  route, or helper wrapper that preserves two independent default registration
  authorities.
- No movement of target artifact exporter or target translate route semantics
  into plugin/common.
- No broad smoke matrix unless focused checks expose a build-system issue that
  requires it.

## Minimal Evidence

- Focused C++ coverage for bundle-registry-driven built-in plugin registration,
  target artifact exporter registration, target translate route registration,
  duplicate/invalid bundle diagnostics, disabled/missing-plugin fail-closed
  behavior, and target exporter/translate error context.
- Focused tool/lit evidence that `tcrv-opt` and `tcrv-translate` default paths
  still expose plugin dialects/source-seed passes, target artifact exporters,
  and target translate routes through the bundle front door.
- Run focused build/test targets for plugin bundle, target artifact export,
  RVV/TensorExtLite plugin behavior, and `tcrv-opt`/`tcrv-translate` front doors.
- Run `check-tianchenrv` if practical.
- Run targeted scans proving no concrete plugin catalog was reintroduced under
  `lib/Target/Builtin`, no `target::ExtensionBundle` resurrection occurred, no
  descriptor/direct-C/source-export authority was added, and no independent
  default `registerBuiltinExtensionPlugins` production loop remains outside the
  bundle front door.

## Completion Evidence

- `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-tensorext-lite-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- `./build/bin/tcrv-opt test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir --tcrv-source-seed-artifact-front-door-pipeline`
- `./build/bin/tcrv-opt test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir --tcrv-disable-builtin-plugins --tcrv-rvv-materialize-i32m1-selected-boundary-seed 2>&1 | rg "Unknown command line argument|tcrv-rvv-materialize-i32m1-selected-boundary-seed"`
- `./build/bin/tcrv-translate --help | rg "tcrv-rvv-emitc-to-cpp|tcrv-export-target-artifact|tcrv-export-target-header-artifact|tcrv-export-target-artifact-bundle"`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter 'source-seed-target-artifact-header|emitc-to-cpp-handoff|tensorext-lite-target-artifact-header|rvv-i32m1-selected-boundary-seed' .` from `build/test`
- `cmake --build build --target check-tianchenrv -j2`
- `git diff --check`
- `rg -n "registerBuiltinExtensionPlugins\(" tools lib/Target/Builtin lib/Plugin/Builtin include/TianChenRV/Plugin include/TianChenRV/Target -S`
- `rg -n "RVVExtensionPlugin|ToyExtensionPlugin|TensorExtLiteExtensionPlugin|OffloadExtensionPlugin|ScalarExtensionPlugin|TemplateExtensionPlugin|kBuiltinExtensionBundles" lib/Target/Builtin include/TianChenRV/Target -S`
- `git diff --unified=0 | rg '^\+.*(descriptor|direct-C|source-export|source export|direct C|Python)'`

## Technical Notes

- Read specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/implementation-stack/index.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Previous completed task:
  `.trellis/tasks/archive/2026-05/05-17-extension-bundle-interface-target-layer-extraction/prd.md`.
- Main implementation files:
  `include/TianChenRV/Plugin/BuiltinExtensionPlugins.h`,
  `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp`,
  `tools/tcrv-opt/tcrv-opt.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `include/TianChenRV/Target/BuiltinTargetArtifactExporters.h`,
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`,
  `include/TianChenRV/Target/BuiltinTargetTranslateRoutes.h`,
  `lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.
