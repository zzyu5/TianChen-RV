# Common Materialized EmitC Header Artifact Foundation

## Goal

Extract a common target-layer foundation for declaration-only runtime-callable
C header artifacts generated from selected materialized EmitC routes, then
rewire Toy and TensorExtLite to use that common foundation as their production
target artifact behavior.

This round makes the current route more reusable:

```text
plugin-owned construction route
  -> selected emission-plan candidate
  -> common materialized EmitC header artifact exporter
  -> default extension-bundle target artifact front door
```

## Current Repository Facts

- `TargetArtifactExport` already owns the selected emission-plan candidate
  collection, selected EmitC route materialization, route/call provenance
  checks, and generic target artifact front doors.
- `ToyTargetSupportBundle` and `TensorExtLiteTargetSupportBundle` each carry
  local copies of candidate field validation, forbidden metadata rejection,
  materialized EmitC function-boundary checks, and declaration-only header
  rendering.
- Toy already emits runtime ABI parameter metadata and construction-protocol
  evidence through its plugin-owned emission plan.
- TensorExtLite currently exports a positive declaration-only header, but its
  target header evidence is narrower than the desired common foundation surface.
- RVV has object/header/bundle behavior with RVV-specific runtime/object
  packaging and must be preserved, not forced into this common header slice.

## Requirements

- Add a common C++ API in the target layer for materialized EmitC header
  artifacts.
- The common API must validate extension-agnostic candidate shape from
  plugin-supplied configuration:
  route id, artifact kind, origin plugin, emission kind, lowering boundary,
  runtime ABI identity, runtime ABI kind/name, runtime glue role, ordered
  runtime ABI parameters, required artifact metadata, and forbidden
  descriptor/direct-C/source-export/compute-body residue.
- The common API must materialize the selected plugin-owned EmitC route, verify
  the existing materialized EmitC handoff, require exactly one `emitc.func`
  boundary, and require the function arity to match the selected ordered runtime
  ABI parameter signature.
- The common API must render only a non-semantic C header declaration:
  include guards, configured includes, evidence comments, and one function
  declaration. It must not render compute bodies, source exporters, descriptors,
  or extension-specific semantics.
- Toy must consume the new common header foundation for its production header
  route. Toy-local target code may keep only Toy configuration and Toy evidence
  metadata requirements.
- TensorExtLite must consume the new common header foundation for its production
  header route. TensorExtLite-local target code may keep only TensorExtLite
  configuration and TensorExtLite evidence metadata requirements.
- TensorExtLite header evidence must include origin plugin, selected variant,
  selected route, runtime ABI kind/name, source-op/interface provenance,
  construction protocol, semantic role graph, and typed role realization.
- Default extension-bundle front door registration must still publish RVV, Toy,
  and TensorExtLite target artifact exporters, with Toy and TensorExtLite using
  the common materialized EmitC header route.
- Negative behavior must fail closed for disabled/missing plugin registration,
  missing or stale route provenance, missing or mismatched runtime ABI fields,
  wrong origin/route/artifact kind, multiple selected candidates, forbidden
  descriptor/direct-C/source-export/compute-body metadata, and declaration
  output that would embed compute bodies.

## Acceptance Criteria

- [ ] `TargetArtifactExport` exposes a common materialized EmitC header
      artifact config, candidate validator, and export helper.
- [ ] Toy target support uses the common header helper; duplicated Toy-local
      validation, single-function check, runtime ABI parameter comment printing,
      and header rendering are removed or reduced to configuration.
- [ ] TensorExtLite target support uses the common header helper; duplicated
      TensorExtLite-local validation, single-function check, and header
      rendering are removed or reduced to configuration.
- [ ] Toy and TensorExtLite exporters are still registered through the
      extension-bundle target artifact front door.
- [ ] Toy and TensorExtLite positive header outputs include origin plugin,
      selected variant, selected route, runtime ABI kind/name, ordered ABI
      parameter evidence when present, source-op/interface provenance,
      construction protocol, role graph, and typed role evidence.
- [ ] Focused C++ tests prove both Toy and TensorExtLite are production
      consumers of the common header foundation and preserve fail-closed
      validation.
- [ ] Focused lit tests for Toy, TensorExtLite, target artifact front-door
      registry behavior, and RVV target artifact behavior pass.
- [ ] Targeted scans over changed target/test surfaces show no descriptor route
      authority, no direct C semantic exporter, no source-export route, and no
      tests protecting old paths.

## Out Of Scope

- No Toy native object/runtime execution.
- No new RVV family coverage, RVV behavior rewrite, or RVV hardware claim.
- No new TensorExtLite semantics or new extension family.
- No source-output routes, direct C semantic exporters, descriptor adapters,
  compatibility wrappers, legacy modes, or Python compiler-core behavior.
- No extension-specific semantic branches in common/core target code.
- No broad smoke matrix unless focused checks expose a build-system issue.

## Minimal Evidence

- Focused build for `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and
  `tcrv-translate`.
- Run `./build/bin/tianchenrv-target-artifact-export-test`.
- Focused lit for `toy-target-artifact-*`,
  `tensorext-lite-target-artifact-*`, `target-artifact-export-registry`, and
  RVV target artifact fixtures.
- Run RVV focused target artifact tests to prove object/header/bundle behavior
  is not regressed.
- Run `git diff --check`.
- Run targeted scans over `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/Toy`, `lib/Target/TensorExtLite`, `test/Target/Toy`, and
  `test/Target/TensorExtLite` for descriptor/direct-C/source-export residue.
- Run `check-tianchenrv` if practical after focused checks pass.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous PRDs read:
  `.trellis/tasks/archive/2026-05/05-17-toy-extension-bundle-target-artifact-bridge/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-17-extension-bundle-registry-default-plugin-front-door/prd.md`.
- Main implementation files:
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Target/Toy/ToyTargetSupportBundle.h`,
  `lib/Target/Toy/ToyTargetSupportBundle.cpp`,
  `include/TianChenRV/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.h`,
  `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`,
  `lib/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.cpp`,
  and `test/Target/TargetArtifactExportTest.cpp`.
