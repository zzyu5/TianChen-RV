# TensorExtLite Materialized EmitC Target Artifact Bridge

## Goal

Rebuild one bounded TensorExtLite first-slice selected target artifact route from
the existing explicit TensorExtLite extension-family role sequence and common
materialized EmitC handoff. The route must prove:

```text
tcrv_tensorext_lite role ops
  -> plugin-owned TCRVEmitCLowerableRoute
  -> materialized MLIR EmitC module
  -> common target artifact exporter front door
  -> TensorExtLite declaration/header artifact
```

The route must not restore `metadata-diagnostic`, descriptor-driven compute,
direct C/source-export printers, independent-backend wording, or RVV-specific
target branches.

## Why Now

Commit `81eaa87` erased `metadata-diagnostic` as supported artifact-kind
authority and intentionally left Toy/TensorExtLite with unsupported diagnostics.
Current code inspection shows TensorExtLite still has a real in-memory EmitC
materialization path for the ordered first-slice body, but its emission plan
returns unsupported and there is no selected target artifact exporter. The next
rebuild bottleneck is to prove a second extension family can consume the same
selected emission-plan to materialized EmitC target-artifact handoff without
copying RVV object/runtime behavior.

## Current Code Facts

- `TensorExtLiteEmitCRouteProvider.cpp` already builds a
  `TCRVEmitCLowerableRoute` from exactly one selected
  `configure -> load_frag -> tile_mma -> store_frag` role sequence in the
  selected variant body.
- `TensorExtLiteExtensionPlugin.cpp` already reports emission readiness as
  supported once that route builds, but `buildVariantEmissionPlan` still returns
  an unsupported diagnostic.
- `TargetArtifactExport.cpp` already provides the common selected EmitC artifact
  front door: it selects one supported emission-plan candidate, rebuilds the
  plugin-owned `TCRVEmitCLowerableRoute`, materializes a verified MLIR EmitC
  module, and checks route/call provenance before target packaging.
- Current target artifact kinds accepted by the common exporter are
  `runtime-callable-c-header` and `riscv-elf-relocatable-object`.
- TensorExtLite has no real object compiler/runtime target in this slice, so
  the positive artifact boundary is a materialized-EmitC-derived declaration
  header only. The default object front door must remain negative.

## Requirements

- Change TensorExtLite first-slice emission planning from unsupported to one
  supported selected emission-plan route only when the explicit selected role
  sequence can build a valid plugin-owned EmitC lowerable route.
- Use artifact kind `runtime-callable-c-header` for the supported TensorExtLite
  route. Do not add a TensorExtLite object artifact route.
- Keep the TensorExtLite route ID, emission kind, runtime ABI ownership
  metadata, lowering-boundary metadata, and materialized EmitC provenance owned
  by TensorExtLite plugin/target-support code.
- Register a TensorExtLite target artifact exporter through the existing
  extension-bundle/target-exporter surfaces, not through core family-name
  branches.
- The TensorExtLite header exporter must use the common selected EmitC artifact
  front door to materialize and verify an MLIR EmitC module before printing a
  declaration/header artifact.
- The header artifact may declare the materialized EmitC function boundary and
  bounded provenance comments, but must not embed compute bodies, source-export
  C text, object bytes, runtime execution claims, hardware evidence, paths,
  logs, credentials, or performance/correctness claims.
- Positive target export must select the current selected path and must not
  infer the target by scanning for a single direct variant.
- Negative paths must fail closed for fallback/no-body inputs, missing selected
  candidate, missing materialized EmitC route provenance, stale
  `metadata-diagnostic`, unsupported direct C/source-export artifact kinds,
  ambiguous selected candidates, and metadata attempts to synthesize compute
  semantics.

## Acceptance Criteria

- [x] TensorExtLite supported emission-plan diagnostics contain the selected
  variant, role, origin, route ID, emission kind, artifact kind
  `runtime-callable-c-header`, runtime ABI metadata, lowering-boundary metadata,
  required capabilities, and bounded EmitC route provenance.
- [x] A positive TensorExtLite selected path reaches the common materialized
  EmitC front door and exports a TensorExtLite header artifact.
- [x] The same positive TensorExtLite input remains negative for default object
  export because no TensorExtLite object route is implemented.
- [x] TensorExtLite target support is registered through the built-in extension
  bundle/exporter registry without adding core RVV/TensorExtLite family
  branches.
- [x] C++ tests cover TensorExtLite emission-plan readiness/plan metadata,
  construction route validation, target exporter registration/validation, and
  missing route provenance failure.
- [x] lit tests cover positive selected header export and negative stale
  unsupported/object/source/metadata-diagnostic paths.
- [x] Focused scans show no production resurrection of `metadata-diagnostic`,
  descriptor-driven compute, direct C/source-export artifact authority, or
  TensorExtLite core special-case target branches.

## Verification

- [x] Focused build:
  `cmake --build build --target tianchenrv-tensorext-lite-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`.
- [x] Focused C++ tests:
  `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test` and
  `./build/bin/tianchenrv-target-artifact-export-test`.
- [x] Focused lit:
  `/usr/bin/python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'tensorext-lite-(first-slice-materialization|target-artifact-(unsupported|header))'`
  from `build/test`, 5/5 passed.
- [x] `git diff --check`.
- [x] Trellis validation:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-05-17-tensorextlite-materialized-emitc-target-artifact-bridge`.
- [x] Full `cmake --build build --target check-tianchenrv -j2`, 100/100 lit
  tests passed.
- [x] Targeted scans over touched plugin/target/test/spec surfaces for
  `metadata-diagnostic`, descriptor/source-export/direct-C route resurrection,
  and RVV branch leakage. Remaining matches are negative checks, guard strings,
  or deletion-rule spec text.

## Out Of Scope

- No TensorExtLite object compilation, linker path, runtime execution, hardware
  correctness, hardware performance, or `ssh rvv` evidence.
- No RVV intrinsic reuse, RVV route tables, RVV route IDs, RVV ABI metadata, or
  RVV-specific branches in TensorExtLite/common code.
- No descriptor routes, source-export printers, direct C semantic exporters,
  source artifact front doors, compatibility layers, legacy modes, Python
  compiler-core logic, new broad TensorExtLite feature families, or generic
  target artifact kind expansion.
- No task to rebuild Toy, Template, Scalar, Offload, IME, or arbitrary vendor
  plugins.

## Checks

- Focused build targets:
  `tianchenrv-tensorext-lite-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`.
- Focused C++ tests:
  `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test` and
  `./build/bin/tianchenrv-target-artifact-export-test`.
- Focused lit for TensorExtLite EmitC/target artifact files.
- `git diff --check`.
- Trellis validation for this task directory.
- `check-tianchenrv` if practical after focused checks pass.
- Targeted scans over touched plugin/target/test/spec surfaces for
  `metadata-diagnostic`, descriptor/source-export/direct-C route resurrection,
  and TensorExtLite/RVV branch leakage.
