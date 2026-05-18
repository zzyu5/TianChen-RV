# RVV construction manifest route-authority demotion

## Goal

Make RVV construction metadata a mirror of provider-derived selected typed-body
route facts, not an independent RVV route authority. The bounded i32m1 labels
may remain as current specialization names only after
`describeRVVSelectedBodyEmitCRoute` has validated the explicit `tcrv_rvv` body,
runtime ABI, config, operation, memory form, and target artifact route.

This task keeps the Stage 1 path focused on:

```text
selected tcrv.exec RVV variant
  -> explicit typed tcrv_rvv body
  -> RVV provider route description
  -> construction/runtime/artifact metadata derived from that description
  -> target/export metadata mirror validation
  -> neutral common EmitC/artifact mechanics
```

## Background

- Commit `f075ec5` made RVV target artifact validation rebuild the provider
  selected-body route before consuming candidate metadata.
- The remaining upstream bottleneck is
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`: construction
  artifact metadata is still built by `emitCRouteID -> kSelectedBodyRoutes`
  lookup, and it still records `manifest.emitcRoute.routeID` and
  `manifest.emitcRoute.runtimeABI` as route-family metadata.
- `RVVExtensionPlugin::buildVariantEmissionPlan` currently derives the provider
  route description, but then asks construction for metadata using only
  `routeDescription.emitCRouteID`.
- Target/export already validates candidate metadata against the rebuilt
  provider route description; this task makes the metadata producer follow the
  same authority chain.

## Requirements

- Add a construction metadata path that consumes a provider-selected route
  description or an equivalent description-shaped facts object.
- Artifact metadata must mirror provider-derived fields including operation,
  typed compute op, EmitC route ID, target artifact route ID, target artifact
  kind, runtime ABI name, runtime ABI contract name, and ordered runtime ABI
  parameters.
- `RVVExtensionPlugin::buildVariantEmissionPlan` must build construction
  artifact metadata from the provider route description it already computed,
  not by route ID alone.
- Target/export validation must continue to fail closed on missing or mismatched
  selected-body facts, stale route ID, stale construction metadata, stale
  runtime ABI, stale config, descriptor/direct-C/source-export residue, and
  unsupported selected typed bodies.
- Manifest-level route IDs may remain only as derived target artifact labels or
  static exporter registration labels, not as the source of operation or route
  support.
- Retained `rvv-i32m1-*` strings are allowed only as ordinary specialization
  labels behind selected typed-body validation.

## Acceptance Criteria

- [x] Positive focused coverage proves an explicit typed i32m1 selected body
  produces construction metadata that mirrors the provider-selected route
  description.
- [x] Negative focused coverage proves stale construction metadata, stale route
  ID, mismatched runtime/config facts, or route-ID-only metadata fail before
  export.
- [x] `RVVExtensionPlugin::buildVariantEmissionPlan` no longer calls the
  construction artifact metadata builder with only an EmitC route ID.
- [x] Construction verification APIs reject metadata whose route ID/runtime
  facts disagree with the provider-derived selected-body description.
- [x] Target/export continues to rebuild the selected-body route and reject
  metadata mismatches before object/header/bundle output.
- [x] Bounded scans of `RVVConstructionProtocol.cpp`,
  `RVVEmitCRouteProvider.cpp`, and `RVVTargetSupportBundle.cpp` show no
  production path treating `manifest.emitcRoute.routeID`, route-family names,
  `getRVVI32M1*` helpers, descriptor/direct-C/source-export residue, or
  artifact metadata as independent RVV route authority.

## Non-Goals

- Do not add RVV coverage: no broadcast expansion beyond existing bounded
  coverage, compare/select expansion, reduction, conversion, dtype, LMUL,
  source-shape, or intrinsic case growth.
- Do not generalize Linalg, Vector, StableHLO, or frontend lowering.
- Do not move RVV semantic choice into common EmitC/export.
- Do not preserve old i32 route authority through compatibility wrappers.
- Do not rename every i32m1 string for aesthetics; rename or change APIs only
  when it removes route authority or clarifies derived specialization
  provenance.
- Do not work on Scalar, IME, Offload, TensorExt, autotuning, dashboards, broad
  smoke matrices, report/status machinery, or artifact-index-only evidence.
- Do not claim runtime, correctness, or performance without real `ssh rvv`
  evidence.

## Relevant Specs And Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/tasks/archive/2026-05/05-19-05-19-rvv-selected-body-target-artifact-export-gate/prd.md`

## Initial Code Surface

- `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- Focused C++/lit tests under `test/Plugin`, `test/Conversion/EmitC`, and
  `test/Target` that exercise construction metadata, provider route
  descriptions, and export metadata validation.

## Validation Plan

- Build and run focused C++ targets:
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-target-artifact-export-test`.
- Run focused lit checks for RVV selected-body EmitC materialization and RVV
  target artifact export negatives if touched expectations require it.
- Run bounded ref-scans over the three RVV files named in the brief for
  route-ID authority, route-family names, `getRVVI32M1*` authority,
  descriptor/direct-C/source-export residue, and metadata-as-authority.
- Run `git diff --check` and Trellis task validation.

## Implementation Notes

- Added `RVVSelectedBodyConstructionMetadataFacts` as the construction metadata
  bridge shape. It carries provider-selected operation, typed compute op, EmitC
  route ID, target artifact route/kind, runtime ABI name/contract, and ordered
  runtime ABI parameters.
- Rewired `RVVExtensionPlugin::buildVariantEmissionPlan` to build construction
  metadata from `getRVVSelectedBodyConstructionMetadataFacts(description)`
  after `describeRVVSelectedBodyEmitCRoute` succeeds.
- Removed the public route-ID-only construction metadata builder and replaced
  construction artifact verification with facts-aware verification.
- Added construction metadata entries for typed compute op, target artifact
  route, target artifact kind, and runtime ABI name. Runtime ABI contract now
  mirrors the provider-selected specialization contract instead of the manifest
  route-family contract.
- Updated RVV target/export validation to verify `rvv_*` construction metadata
  against provider-derived construction facts, while continuing to validate
  `tcrv_rvv.*` config/runtime-VL metadata against the provider route
  description.
- Renamed the provider-local fixed mapping from selected-body "descriptor" to
  selected-body "specialization" so bounded diagnostics do not imply
  descriptor-driven route authority.
- No `.trellis/spec/` update was needed: the existing RVV plugin and EmitC
  route specs already require target/export metadata to mirror the selected
  typed body after the RVV-owned route builder validates it.

## Checks Run

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `cmake --build build --target tianchenrv-construction-protocol-common-test -j2`
- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- Focused lit filter for RVV source-front-door metadata, RVV target artifact
  export, and RVV EmitC selected-body materialization/negative tests: 9/9
  passed.
- Bounded residue scan over `RVVConstructionProtocol.cpp`,
  `RVVEmitCRouteProvider.cpp`, and `RVVTargetSupportBundle.cpp`.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-rvv-construction-manifest-route-authority-demotion`
- `git diff --check`
