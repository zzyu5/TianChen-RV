# Toy target artifact export template

## Goal

Carry the already materialized Toy extension-family EmitC route across the
target artifact/export front door. A valid selected Toy template path should no
longer stop at an unknown target artifact exporter: the existing Toy
`tcrv_toy.compute_skeleton` boundary and plugin-owned
`TCRVEmitCLowerableRoute` must be discoverable, validated, exported, and
exposed through the existing target artifact and target translate abstractions.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting worktree was clean at HEAD `abb2b08`.
- No `.trellis/.current-task` existed when this task began.
- The prior Toy task completed construction-protocol validation, typed role
  boundary materialization, Toy-owned EmitC route construction, and common
  EmitC module materialization.
- `test/Transforms/ExecutionPlanning/execution-planning-pipeline-toy.mlir`
  currently fails at the target/export front door because
  `toy-template-compute-emitc-route` has no registered target artifact
  exporter.
- Existing target/export machinery already models selected emission-plan
  candidates, exporter registries, plugin target-support bundles, exact target
  artifact route export, generic target artifact export, target artifact bundle
  records, emission manifests, and translate routes.

## Requirements

- Add a Toy-owned or template-owned target support submodule that registers the
  Toy template target artifact route through existing target artifact exporter
  registries.
- The route must consume the selected Toy emission-plan candidate, the
  materialized `tcrv_toy.compute_skeleton` selected boundary, and the existing
  Toy `buildToyTemplateEmitCLowerableRoute` provider.
- The exporter may produce a bounded metadata/source artifact surface because
  Toy has no runtime, object compilation, correctness, performance, or hardware
  claim in this task.
- The route must not synthesize compute C from Toy metadata, descriptors, route
  ids, or direct string exporters; generated source evidence must come from the
  common EmitC materializer / MLIR C++ emitter path.
- Register a target translate route that points at the Toy target artifact
  route, so `tcrv-translate` can drive the same artifact-backed path.
- Keep common/core orchestration extension-neutral; do not add Toy/RVV semantic
  branches to generic passes or target artifact front-door logic.
- Update focused C++ and lit/FileCheck coverage for positive export and
  fail-closed cases.

## Acceptance Criteria

- [ ] A valid Toy selected path no longer fails because
      `toy-template-compute-emitc-route` is an unknown target artifact exporter.
- [ ] Target artifact collection/export can identify and validate the Toy
      template route against origin, emission kind, artifact kind, lowering
      boundary, runtime ABI metadata, and absence of unsupported runtime ABI
      parameters.
- [ ] The Toy artifact/export path proves materialized EmitC evidence by
      invoking the Toy route provider and common EmitC materializer before
      producing source/metadata artifact output.
- [ ] Stale route IDs, missing materialized Toy EmitC boundary evidence,
      unselected variants, missing emission-plan metadata, and legacy
      unsupported Toy lowering-boundary paths fail closed.
- [ ] Emission manifest or target artifact evidence shows the Toy path crossing
      the artifact/export boundary.
- [ ] C++ tests cover Toy target artifact route registration, candidate
      validation, target-support bundle registration, and translate route
      registration.
- [ ] lit/FileCheck tests cover the positive Toy target artifact/export path
      and focused negative cases.
- [ ] A changed-surface scan shows descriptor/direct-C/source-export/Python
      compiler-core legacy paths were not restored and common/core did not gain
      Toy/RVV semantic branches.

## Out of Scope

- No RVV dtype/LMUL/SEW/op family expansion.
- No TensorExt, IME, offload, scalar, high-level tensor lowering, MLIR vector
  lowering, Toy runtime ABI execution, object compilation, correctness,
  performance, or `ssh rvv` claim.
- No descriptor-driven computation, descriptor/binary-family registry, direct C
  semantic exporter, source skeleton generation, Python compiler-core logic,
  GCC-default route, checkpoint/state-machine ledger, broad target framework
  rewrite, docs-only/template-only work, or compatibility wrapper.
- No common/core Toy/RVV semantic branch.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`.
- Prior task context:
  `.trellis/tasks/archive/2026-05/05-16-toy-executable-plugin-template/prd.md`.
- Primary implementation surface:
  `include/TianChenRV/Plugin/Toy/*`,
  `lib/Plugin/Toy/*`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Target/TargetTranslateRegistration.h`,
  `lib/Target/TargetTranslateRegistration.cpp`,
  `include/TianChenRV/Target/EmissionManifest.h`,
  `lib/Target/EmissionManifest.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp` as a target/export pattern only.
- Focused test surface:
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Target/RVV/emitc-to-cpp-handoff.mlir`,
  `test/Target/EmissionManifest/*`,
  `test/Conversion/EmitC/toy-template-materialization*.mlir`,
  `test/Transforms/ExecutionPlanning/execution-planning-pipeline-toy.mlir`.

## Definition of Done

- Implement one coherent Toy target/export submodule.
- Update focused C++ and lit/FileCheck tests.
- Run focused build/tests and `git diff --check`; run broader
  `check-tianchenrv` if practical.
- Keep Trellis task state truthful, finish/archive if complete, and create one
  coherent commit.
