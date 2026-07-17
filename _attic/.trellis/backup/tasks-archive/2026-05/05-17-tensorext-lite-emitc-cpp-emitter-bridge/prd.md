# TensorExtLite Materialized EmitC C++ Emitter Bridge

## Goal

Add a bounded TensorExtLite-owned target translate bridge that consumes the
existing selected TensorExtLite role-sequence front door only after the route
has a selected emission-plan candidate and can materialize a verified MLIR
EmitC module. The bridge then emits C/C++ source through the upstream MLIR
EmitC C/C++ emitter.

The intended route is:

```text
TensorExtLite source marker
  -> selected TensorExtLite role ops
  -> TensorExtLite-owned EmitC route
  -> verified materialized MLIR EmitC module
  -> MLIR EmitC C/C++ emitter output
```

This task proves C/C++ source emission for the existing first-slice
TensorExtLite route without making source markers, selected metadata,
descriptors, direct-C exporters, or source-export artifacts into computation
authority.

## Current Repository Facts

- HEAD `8f965cb` added the TensorExtLite source-front-door template and left
  the repository clean.
- TensorExtLite already owns the bounded source-front-door pass, selected
  `tcrv.exec.variant`, ordered role ops, direct lowering-boundary marker,
  emission-plan metadata, runtime ABI metadata, EmitC route provider, and
  declaration-only materialized EmitC header artifact exporter.
- The common selected EmitC artifact helper can select a supported
  emission-plan candidate, build the plugin-owned
  `TCRVEmitCLowerableRoute`, materialize a fresh EmitC module, verify route
  and call provenance, verify the EmitC function boundary, and invoke
  `mlir::emitc::translateToCpp`.
- RVV currently has a target translate registration for a direct
  materialized-EmitC-to-C++ route. TensorExtLite has a target artifact header
  exporter but no plugin-owned target translate route that emits the selected
  first-slice C/C++ source.

## Requirements

- Add a TensorExtLite target translate route registered through the plugin's
  target-support translate hook.
- The public route must be plugin-owned and discoverable from `tcrv-translate
  --help`.
- The route must consume a selected TensorExtLite emission-plan candidate,
  validate origin plugin, selected variant, route id, artifact kind, emission
  kind, lowering boundary, runtime ABI kind/name, runtime glue role, runtime
  ABI parameters, construction protocol metadata, typed role metadata, source
  role sequence, and source-op interface provenance before emitting C/C++.
- The route must use the existing TensorExtLite
  `buildTensorExtLiteFragmentMmaEmitCLowerableRoute` builder and the common
  selected materialized EmitC helper. It must not add a direct C semantic
  exporter, descriptor adapter, source-export route, or common branch on
  TensorExtLite names for computation semantics.
- The route must require a materialized EmitC handoff with route source-op
  provenance, call source-op provenance, and exactly one EmitC function
  boundary before invoking the MLIR EmitC C/C++ emitter.
- The positive source-front-door fixture must reach emitted C/C++ source by
  running the source-artifact front-door pipeline first, then the TensorExtLite
  target translate route.
- Existing TensorExtLite declaration-only header artifact export must remain
  coherent with the same selected variant, route id, runtime ABI, and
  construction protocol evidence.

## Acceptance Criteria

- [ ] A positive TensorExtLite source-front-door fixture pipes through
      `--tcrv-source-artifact-front-door-pipeline` and then
      `tcrv-translate --tcrv-tensorext-lite-emitc-to-cpp`.
- [ ] The emitted C/C++ source contains the TensorExtLite first-slice function,
      the upstream EmitC-emitted opaque calls
      `tcrv_tensorext_lite_config`,
      `tcrv_tensorext_lite_load_frag`,
      `tcrv_tensorext_lite_tile_mma`, and
      `tcrv_tensorext_lite_store_frag`, plus route and call source-op
      provenance comments.
- [ ] `tcrv-translate --help` shows the TensorExtLite target translate route
      with MLIR EmitC C/C++ emitter wording.
- [ ] C++ target tests prove TensorExtLite target translate registration is
      plugin-owned, discoverable through built-in aggregation, idempotent, and
      does not disturb RVV's existing translate route.
- [ ] Existing TensorExtLite header artifact tests remain green and continue
      to export only declaration-only C headers.
- [ ] Negative lit/C++ coverage rejects non-materialized or unplanned input,
      missing route provenance, stale source-front-door metadata, missing
      lowering boundary, wrong origin plugin, fallback-only selection, and
      descriptor/direct-C/source-export residue before C/C++ output.
- [ ] Targeted scans over changed TensorExtLite plugin/target/tests show no
      descriptor route authority, no direct-C semantic exporter, no
      source-export route, and no tests preserving old-path behavior.

## Definition Of Done

- Focused build targets for `tcrv-opt`, `tcrv-translate`, and
  `tianchenrv-target-artifact-export-test` pass.
- Focused lit tests for TensorExtLite transform, EmitC materialization, target
  header export, target C++ emitter export, and negative cases pass.
- `git diff --check` passes.
- `check-tianchenrv` runs if practical after focused checks.
- Trellis task status truthfully records whether the task is finished and
  archived.
- One coherent commit is created when complete.

## Out Of Scope

- No TensorExtLite object export, bundle packaging, native compilation,
  runtime execution, correctness claim, or performance claim.
- No general TensorExt frontend, linalg lowering, high-level tensor/tile IR,
  new RVV work, IME/offload work, new common lowering pass, or new extension
  family.
- No descriptor-driven computation, direct C semantic exporter,
  source-export route, Python compiler-core behavior, compatibility wrapper,
  legacy mode, or core/common semantic branch on TensorExtLite/RVV/Toy names.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous PRD read:
  `.trellis/tasks/archive/2026-05/05-17-tensorext-lite-source-front-door-template/prd.md`.
- Initial code/test surfaces inspected:
  `include/TianChenRV/Plugin/TensorExtLite/TensorExtLiteSourceFrontDoor.h`,
  `lib/Plugin/TensorExtLite/TensorExtLiteSourceFrontDoor.cpp`,
  `lib/Plugin/TensorExtLite/Construction/TensorExtLiteConstructionProtocol.cpp`,
  `lib/Plugin/TensorExtLite/EmitC/TensorExtLiteEmitCRouteProvider.cpp`,
  `lib/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.cpp`,
  `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `lib/Target/TargetTranslateRegistration.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Transforms/TensorExtLite/tensorext-lite-fragment-mma-source-front-door.mlir`,
  `test/Conversion/EmitC/tensorext-lite-first-slice-materialization.mlir`,
  `test/Target/TensorExtLite/tensorext-lite-source-front-door-target-artifact-header.mlir`,
  and `test/Target/TargetArtifactExportTest.cpp`.
