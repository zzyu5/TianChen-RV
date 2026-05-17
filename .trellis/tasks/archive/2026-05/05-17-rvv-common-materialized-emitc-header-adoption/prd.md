# RVV Common Materialized EmitC Header Adoption

## Goal

Migrate the selected RVV i32m1 materialized EmitC header artifact to the
common `MaterializedEmitCHeaderArtifact` validator/exporter while preserving
the existing RVV relocatable object route and coherent object+header bundle
composition.

This round makes the first real RVV executable plugin path consume the same
declaration-only header ABI foundation already used by other extension-family
target exporters:

```text
selected RVV extension-family ops
  -> RVV-owned EmitC lowerable route
  -> materialized EmitC module
  -> RVV object export
  -> common declaration-only materialized EmitC header export
  -> coherent RVV object+header bundle
```

## Current Repository Facts

- `TargetArtifactExport` already provides
  `MaterializedEmitCHeaderArtifactConfig`,
  `validateMaterializedEmitCHeaderArtifactCandidate`, and
  `exportMaterializedEmitCHeaderArtifact`.
- Toy and TensorExtLite already consume the common materialized EmitC header
  helper in production target-support code.
- RVV already has a selected i32m1 materialized EmitC object route, a header
  composite route, and a bundle path.
- RVV still keeps local header declaration rendering and local materialized
  EmitC function-boundary checks around
  `exportRVVI32M1ArithmeticHeaderArtifact`.
- RVV candidate validation must keep RVV-owned metadata checks for construction
  protocol readiness, arithmetic route/op consistency, runtime AVL/VL
  provenance, multi-VL materialized-loop evidence, and deleted descriptor/local
  element-count residue.
- The common header helper can express the route, artifact, origin, emission,
  lowering boundary, runtime ABI identity, ordered ABI parameter signature,
  required metadata evidence, materialized EmitC single-function check, and
  declaration-only header rendering.

## Requirements

- Add an RVV-local `MaterializedEmitCHeaderArtifactConfig` for the selected
  i32m1 arithmetic materialized EmitC header route.
- The RVV header exporter must call
  `exportMaterializedEmitCHeaderArtifact` as its production path.
- The common helper must validate the selected RVV header candidate for:
  route id, artifact kind, origin plugin, selected route, selected variant
  when configured by the selected candidate path, emission kind, lowering
  boundary, runtime ABI, runtime ABI kind/name, runtime glue role, and ordered
  ABI parameters.
- RVV-local code may keep plugin-owned candidate validation and bundle
  metadata where it checks RVV-specific provenance and object/header grouping.
- Remove or bypass the RVV-local declaration renderer and duplicated
  single-EmitC-function/arity validation from the production header path.
- The generated header must remain declaration-only: includes, bounded evidence
  comments, and one runtime-callable function declaration. It must not include
  RVV intrinsic compute bodies, source exporters, descriptors, `main`, return
  bodies, hardware logs, or correctness/performance claims.
- Header evidence must include origin plugin, selected variant, selected
  route, runtime ABI kind/name, ordered ABI parameters, and required RVV
  provenance metadata.
- The object route must continue to package MLIR EmitC C/C++ emitter output as
  a RISC-V relocatable object with the existing RVV object handoff.
- Bundle index records must preserve object/header component grouping, object
  route handoff, external ABI identity, runtime ABI kind/name, and matching
  runtime ABI parameters.

## Acceptance Criteria

- [ ] RVV header export delegates to
      `exportMaterializedEmitCHeaderArtifact` with an RVV-local config.
- [ ] The obsolete RVV-local header declaration renderer and duplicated
      materialized EmitC function-boundary/arity checks are removed from the
      production path.
- [ ] RVV object export continues to use the existing selected materialized
      EmitC route and RISC-V object packaging path.
- [ ] RVV bundle export still produces exactly one object component and one
      header component under the same selected-variant component group.
- [ ] Header output includes common helper evidence for origin plugin,
      selected variant, selected route, runtime ABI kind/name, and ordered ABI
      parameters.
- [ ] Header output includes required RVV provenance metadata:
      `tcrv_rvv.runtime_avl_source`,
      `tcrv_rvv.runtime_avl_abi_parameter`, `tcrv_rvv.vl_def`,
      `tcrv_rvv.vl_scope`, `tcrv_rvv.emitc_loop`,
      `tcrv_rvv.loop_induction`, `tcrv_rvv.loop_step`,
      `tcrv_rvv.remaining_avl`, `tcrv_rvv.pointer_advance`,
      `tcrv_rvv.bounded_slice`, and `tcrv_rvv.multi_vl`.
- [ ] Negative checks fail closed for missing materialized EmitC provenance,
      wrong route/artifact/origin/emission/lowering/runtime ABI fields,
      misordered ABI parameters, multiple selected supported candidates,
      forbidden descriptor/direct-C/source-export/compute-body metadata, and
      historical deleted route IDs.
- [ ] Toy and TensorExtLite common header coverage is not weakened.

## Out Of Scope

- No new RVV arithmetic families, SEW/LMUL coverage, source seeds, or generic
  RVV lowering.
- No executable plugin templates, scalar fallback compute, compatibility
  wrappers, descriptor adapters, direct C semantic exporters, source-export
  routes, or Python compiler-core behavior.
- No common/core branches that know RVV intrinsic semantics.
- No new hardware execution claim unless object bytes or bundle consumption
  change in a way that requires fresh `ssh rvv` evidence.

## Minimal Evidence

- Focused build for RVV target support, RVV plugin, target artifact export,
  `tcrv-opt`, and `tcrv-translate`.
- Run `./build/bin/tianchenrv-target-artifact-export-test`.
- Run relevant RVV/plugin tests affected by target artifact registration.
- Focused lit for RVV source-seed object/header/bundle target artifact
  front doors, including `CHECK-NOT` coverage for intrinsic compute bodies,
  descriptors, direct-C residue, source-export residue, and deleted route IDs.
- Run focused Toy/TensorExtLite target artifact checks if touched by common
  header expectations.
- Run targeted scans over RVV target/plugin/translate/tests and common target
  files showing no descriptor route authority, no direct C semantic exporter,
  no source-export route, and no stale local RVV header renderer left on the
  production path.
- Run `git diff --check`.
- Run `check-tianchenrv` if practical after focused checks pass.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/extension-plugins/rvv-plugin.md`.
- Previous task PRD read:
  `.trellis/tasks/archive/2026-05/05-17-common-materialized-emitc-header-foundation/prd.md`.
- Main implementation files:
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `test/Target/RVV/source-seed-target-artifact-object.mlir`,
  `test/Target/RVV/source-seed-target-artifact-header.mlir`, and
  `test/Target/TargetArtifactExportTest.cpp`.
