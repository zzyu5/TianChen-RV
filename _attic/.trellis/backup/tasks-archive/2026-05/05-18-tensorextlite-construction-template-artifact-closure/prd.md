# TensorExtLite executable construction-template artifact closure

## Goal

Close one coherent TensorExtLite construction-template artifact path as the
next non-RVV executable construction-template consumer. The selected
TensorExtLite path must carry the same machine-readable construction protocol
through typed role materialization, EmitC route construction, emission-plan
metadata, target artifact validation, and object/header bundle export.

## Current HEAD Evidence

- Repo root is `/home/kingdom/phdworks/TianchenRV`; worktree was clean before
  this task.
- Current HEAD is `ef2287f plugin: close rvv construction template metadata`.
- Commit history already contains `bdb849d plugin: close tensorextlite
  construction template`; this round must repair remaining current-head gaps
  instead of adding a parallel TensorExtLite path.
- TensorExtLite source-front-door input already materializes a selected
  TensorExtLite variant, ordered role ops, a direct
  `tcrv_tensorext_lite.lowering_boundary`, emission-plan metadata, and target
  bundle coverage.
- The remaining default-path gap is that
  `TensorExtLiteExtensionPlugin::materializeSelectedLoweringBoundary` currently
  reports no-boundary even though the target artifact exporter requires a
  selected `tcrv_tensorext_lite.lowering_boundary` before C/C++ emission.

## Requirements

- Keep all compiler behavior in C++/MLIR/TableGen/CMake/lit/FileCheck.
- Do not add descriptor adapters, source-export routes, direct C semantic
  exporters, Python compiler-core behavior, legacy compatibility wrappers, RVV
  feature expansion, or TensorExtLite runtime correctness/performance claims.
- Use the existing TensorExtLite plugin, dialect, construction protocol, EmitC
  route provider, target support bundle, and focused tests.
- Make the default plugin-selected TensorExtLite path consume the existing
  construction protocol route instead of returning no-boundary for an active
  construction-template path.
- Preserve fail-closed behavior for stale or partial role materialization:
  missing/reordered/duplicated typed roles, stale route provenance, missing
  lowering-boundary metadata, stale source-front-door metadata, fallback-only
  selections, runtime ABI mismatch, and descriptor/direct-C/source-export
  residue must not become artifact authority.

## Acceptance Criteria

- A positive TensorExtLite selected path reaches an ordered
  `configure -> load_frag -> tile_mma -> store_frag` role sequence and one
  direct `tcrv_tensorext_lite.lowering_boundary` through plugin-owned
  construction metadata.
- TensorExtLite emission plan and artifact/header/bundle evidence carry:
  construction protocol, extension archetype, semantic role graph, common
  interface realization, typed role realization, EmitC route mapping, evidence
  profile, runtime ABI contract, object handoff, and bundle component group
  where applicable.
- Target artifact export remains selected-plan driven and materialized-EmitC
  derived; the header remains declaration-only and object-backed.
- Negative coverage proves fail-closed behavior for partial/mismatched typed
  role realization and missing selected boundary before target artifact output.
- Focused C++ and lit tests pass, plus targeted scans show no descriptor-driven
  computation, no direct C semantic exporter, no source-export route, and no
  extension-specific semantic branch added to common/core orchestration.

## Out Of Scope

- New RVV coverage, RVV dtype/SEW/LMUL expansion, new `ssh rvv` matrices, or
  hardware performance/correctness claims.
- Full TensorExt/IME hardware semantics beyond the existing TensorExtLite
  fragment-MMA proof slice.
- Broad common-interface rewrite or new extension family work.
- Documentation/status/report-only closure.

## Technical Notes

- Read specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Main code surfaces:
  `lib/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.cpp`,
  `lib/Plugin/TensorExtLite/TensorExtLiteSourceFrontDoor.cpp`,
  `lib/Plugin/TensorExtLite/EmitC/TensorExtLiteEmitCRouteProvider.cpp`,
  `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`.
- Main tests:
  `test/Plugin/TensorExtLiteExtensionPluginTest.cpp`,
  `test/Plugin/ConstructionProtocolCommonTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`, and focused TensorExtLite lit
  fixtures under `test/Transforms/TensorExtLite`,
  `test/Conversion/EmitC`, and `test/Target/TensorExtLite`.
