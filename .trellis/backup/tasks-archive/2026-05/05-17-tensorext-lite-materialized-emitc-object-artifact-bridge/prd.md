# TensorExtLite Materialized EmitC Object Artifact Bridge

## Goal

Carry the existing selected TensorExtLite source-front-door route through the
same plugin-owned materialized EmitC C/C++ emitter handoff into a real
relocatable object artifact exposed by the target artifact exporter surface.
The intended route is:

```text
TensorExtLite source-front-door materialization
  -> selected TensorExtLite extension-family role ops
  -> selected TensorExtLite emission-plan candidate
  -> materialized MLIR EmitC module
  -> upstream MLIR EmitC C/C++ emitter
  -> clang relocatable object packaging
```

This task proves object packaging only. It does not claim TensorExtLite
hardware runtime execution, correctness, performance, vendor intrinsic
semantics, or broader TensorExt behavior.

## Current Repository Facts

- Current HEAD is `6aa8e9f tensorext: add emitc cpp bridge`, with a clean
  worktree before this task.
- No `.trellis/.current-task` existed; this task was created from the Hermes
  Direction Brief as
  `05-17-tensorext-lite-materialized-emitc-object-artifact-bridge`.
- The brief references
  `.trellis/spec/extension-plugins/tensorext-lite-plugin.md`, but that file is
  absent in current HEAD. The relevant TensorExtLite rules currently live in
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md` and
  `.trellis/spec/lowering-runtime/{emitc-route.md,emission-runtime-contract.md}`.
- Existing specs still describe the previous first slice where TensorExtLite
  may publish only a declaration header and object routes remain unsupported.
  This task intentionally advances that durable contract; the specs must be
  updated after implementation to describe the new object route boundary.
- TensorExtLite already has source-front-door materialization, selected role
  ops, lowering-boundary validation, route metadata, runtime ABI metadata,
  common materialized EmitC construction, and a target translate route that
  emits C/C++ through the MLIR EmitC C/C++ emitter.
- The TensorExtLite emission plan currently advertises the shared
  `tensorext-lite-fragment-mma-emitc-route` as
  `runtime-callable-c-header`, so the generic default object front door finds
  no object candidate.
- RVV proves the current object packaging pattern: selected materialized EmitC
  source is generated through `emitSelectedEmitCArtifactCppSource`, then
  compiled by clang into a RISC-V relocatable object.

## Requirements

- Upgrade the TensorExtLite selected materialized EmitC handoff so the
  production emission-plan candidate is a bounded
  `riscv-elf-relocatable-object` artifact route.
- Add a TensorExtLite-owned object exporter registered through the existing
  target-support plugin bundle surface. The exporter must be discoverable by
  the built-in target artifact exporter registration path.
- The object exporter must consume the same selected materialized EmitC route
  as the TensorExtLite C++ bridge. It must validate selected candidate,
  origin plugin, selected variant, direct selected role, route id, artifact
  kind, emission kind, lowering boundary, runtime ABI kind/name, runtime glue
  role, ordered runtime ABI parameters, construction protocol metadata,
  source-op/source-role/interface provenance, and forbidden descriptor/direct-C
  residue before object output.
- The object exporter must materialize a verified EmitC module and invoke the
  upstream MLIR EmitC C/C++ emitter before clang object packaging. It must not
  synthesize TensorExtLite compute from source-front-door markers, metadata,
  descriptors, route records, or direct handwritten C/source-export printers.
- Keep the existing TensorExtLite header behavior available, but make it a
  declaration-only header derived from the same selected object candidate when
  necessary. It must not become a separate computation or object authority.
- Preserve the existing TensorExtLite C++ translate route for inspection of
  the materialized EmitC C/C++ source.
- Register routes idempotently and without disturbing existing RVV object/header
  and Toy header route registration.
- Update focused C++ and lit coverage so previous "object unsupported" tests
  become positive object/readobj evidence, while stale and malformed inputs
  still fail closed.
- Update durable specs that currently state TensorExtLite object export is
  unsupported, narrowing the new claim to object packaging from selected
  materialized EmitC only.

## Acceptance Criteria

- [x] A positive TensorExtLite source-front-door fixture pipes through
      `--tcrv-source-artifact-front-door-pipeline` and then
      `tcrv-translate --tcrv-export-target-artifact`, producing a non-empty
      object file.
- [x] `llvm-readobj -h` or equivalent proves the produced TensorExtLite object
      is an ELF relocatable object. The evidence is object packaging evidence
      only, not runtime/correctness/performance evidence.
- [x] The existing TensorExtLite header front door still exports a
      declaration-only header with origin, selected variant, selected route,
      runtime ABI, construction protocol, semantic role graph, typed role, and
      source-op interface evidence.
- [x] C++ tests prove TensorExtLite object exporter registration,
      route shape, idempotence, candidate validation, fail-closed metadata
      mismatches, and no regression to RVV/Toy route registration.
- [x] Negative coverage fails closed for stale source-front-door metadata,
      fallback-only selection, wrong origin, missing selected TensorExtLite
      lowering boundary, missing route provenance, missing or mismatched
      runtime ABI/handoff metadata, ambiguous supported candidates,
      direct-C/source-export/descriptor residue, and non-materialized inputs.
- [x] Targeted scans over TensorExtLite plugin/target/tests and
      `TargetArtifactExport` show no descriptor-driven authority, no direct C
      semantic exporter, no source-export route, and no object-as-runtime
      correctness/performance claim.
- [x] Focused build targets for `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test` pass.
- [x] Focused TensorExtLite/TARGET lit tests pass; `check-tianchenrv` runs if
      practical after focused validation.

## Completion Evidence

- Focused build passed:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-tensorext-lite-extension-plugin-test -j2`.
- C++ tests passed:
  `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test` and
  `./build/bin/tianchenrv-target-artifact-export-test`.
- Focused lit passed from `build/test` for TensorExtLite materialization,
  source-front-door C++ emission, header, object, and unsupported cases.
- Manual object evidence passed:
  source-front-door pipeline plus `--tcrv-export-target-artifact` emitted an
  `elf64-littleriscv` object with `Arch: riscv64`, `Type: Relocatable`, and
  symbol
  `tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice`.
- Full `check-tianchenrv` passed: 111/111 lit tests.
- Targeted scan over TensorExtLite target/plugin/tests and
  `TargetArtifactExport` found descriptor/direct-C/source-export terms only in
  fail-closed validation, negative tests, or spec boundary text.

## Out Of Scope

- No TensorExtLite runtime execution, hardware validation, correctness claim,
  performance claim, vendor intrinsic semantics, link/run harness, or `ssh rvv`
  evidence.
- No broader TensorExt families, IME/offload/scalar changes, new generic plugin
  templates, new high-level frontend lowering, linalg/tensor/tile IR work, or
  RVV behavior changes except narrow shared-route regression fixes forced by
  this task.
- No Python compiler-core behavior.
- No descriptor route, descriptor-driven computation, direct C semantic
  exporter, source-export route, compatibility wrapper, legacy route id, or
  common/core semantic branch on TensorExtLite.
- No prompt/report/helper-only round as the main achievement.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous PRD read:
  `.trellis/tasks/archive/2026-05/05-17-tensorext-lite-emitc-cpp-emitter-bridge/prd.md`.
- Initial code/test surfaces inspected:
  `include/TianChenRV/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.h`,
  `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`,
  `lib/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.cpp`,
  `lib/Plugin/TensorExtLite/Construction/TensorExtLiteConstructionProtocol.cpp`,
  `lib/Plugin/TensorExtLite/EmitC/TensorExtLiteEmitCRouteProvider.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Target/RVV/vector-source-target-artifact-object.mlir`,
  `test/Target/TensorExtLite/tensorext-lite-source-front-door-emitc-to-cpp.mlir`,
  `test/Target/TensorExtLite/tensorext-lite-target-artifact-header.mlir`,
  `test/Target/TensorExtLite/tensorext-lite-target-artifact-unsupported.mlir`,
  `test/Plugin/TensorExtLiteExtensionPluginTest.cpp`, and
  `test/Transforms/TensorExtLite/tensorext-lite-fragment-mma-source-front-door.mlir`.
