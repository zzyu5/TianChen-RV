# RVV Stage 2 elementwise/broadcast selected-body coverage

## Goal

Add one coherent RVV Stage 2 production coverage slice for vector-level
elementwise/broadcast selected bodies on the corrected `tcrv_rvv` authority
surface. This round targets bounded i32m1 vector-scalar/broadcast arithmetic:
a selected `tcrv.exec` RVV variant must contain an explicit typed
`tcrv_rvv` body, RVV plugin-local legality and route construction must consume
that body as the only semantic authority, and the route must reach common
EmitC materialization and target artifact validation/export.

## What I Already Know

- Current repo root is `/home/kingdom/phdworks/TianchenRV`; task start found a
  clean worktree at HEAD `f217419`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief and set as current.
- `.trellis/spec/index.md` keeps compiler implementation in
  C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck. Python is tooling only.
- `.trellis/spec/extension-plugins/rvv-plugin.md` defines Stage 2 as RVV
  completion work on the corrected vector-level surface. Stage 2 coverage must
  remain Vector-like and plugin-local, not high-level frontend lowering,
  one-intrinsic wrapper batches, dashboards, or dtype/LMUL clone work.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires EmitC routes to
  faithfully consume selected extension-family body structure. Common
  EmitC/export mechanics must not infer RVV semantics from route ids, artifact
  metadata, descriptor residue, source-front-door names, ABI parameter names,
  or test fixture names.
- The archived Stage 1 task closed target artifact authority for object,
  header, and bundle export: candidate metadata and route ids are mirrors only
  after a typed selected `tcrv_rvv` body rebuilds and validates the route.

## Requirements

- Inspect the current RVV typed-body surface, legality, construction protocol,
  route provider, target support bundle, and focused tests named in the
  Direction Brief.
- Add or extend typed `tcrv_rvv` IR for a bounded vector-scalar/broadcast i32m1
  arithmetic body shape. The scalar or broadcast input must be represented in
  typed `tcrv_rvv` IR or RVV plugin-owned construction state before emission.
- Ensure RVV legality accepts the new selected typed body only when the body
  explicitly carries the vector-scalar/broadcast semantics and required
  runtime ABI/VL boundary.
- Ensure the RVV route builder maps the selected typed body to a supported
  `TCRVEmitCLowerableRoute` payload and rejects unsupported or implied-only
  semantics before common EmitC/materialization or target export.
- Keep common EmitC/materialization/target export neutral. They may consume
  validated route payloads and selected-candidate mirrors only.
- Add focused positive coverage from selected `tcrv.exec` RVV variant through
  materialized EmitC or target artifact validation/export.
- Add focused negative coverage proving metadata-only, route-id-only,
  artifact-name-only, source-front-door-only, ABI-name-only, or fixture-name
  scalar/broadcast semantics fail closed.

## Acceptance Criteria

- [ ] A selected RVV variant containing an explicit typed vector-scalar or
      broadcast i32m1 `tcrv_rvv` body is accepted by RVV legality.
- [ ] The accepted typed body produces a supported RVV emission plan and a
      faithful `TCRVEmitCLowerableRoute`.
- [ ] The route materializes EmitC and reaches existing target artifact
      validation/export mechanics without common/export code learning RVV
      computation semantics.
- [ ] Route payload evidence records the selected typed RVV body as authority,
      not route id, artifact name, source provenance, ABI names, descriptor
      residue, or arithmetic metadata.
- [ ] Negative tests fail before emission/export when scalar/broadcast
      semantics are implied only by metadata, route ids, artifact names,
      source-front-door names, ABI parameter names, or fixtures.
- [ ] Existing Stage 1 typed-body authority positives and negatives remain
      green.

## Definition Of Done

- Focused C++ and/or lit coverage for the new typed-body op or body shape.
- Focused coverage for RVV legality, selected-body consumption, route builder
  payload, EmitC materialization, and target artifact candidate validation.
- Relevant RVV plugin, construction protocol, target artifact export, and
  focused lit tests pass.
- `git diff --check` passes.
- `check-tianchenrv` is run if practical; otherwise the exact blocker is
  recorded.
- Trellis task files and workspace journal are updated truthfully.
- If complete, the task is finished/archived and one coherent commit is
  created.

## Out Of Scope

- High-level Linalg/Vector/StableHLO frontend lowering.
- Per-Linalg-op lowerers, high-level kernel ops, one-op-per-intrinsic
  wrappers, dtype/LMUL clone batches, generic RVV maturity claims, scalar
  fallback/offload/IME/TensorExt work, global autotuning/readiness systems,
  reports, dashboards, compatibility wrappers for old i32 route authority,
  descriptor/direct-C/source-export routes, or common/core RVV semantic
  branches.
- Fresh `ssh rvv` runtime correctness/performance claims unless this round
  changes generated runtime behavior enough to require hardware validation.

## Technical Notes

- Primary read targets from the Direction Brief:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Plugin/ConstructionProtocolCommonTest.cpp`,
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir`, and
  `test/Target/TargetArtifactExportTest.cpp`.
- Previous Stage 1 PRD:
  `.trellis/tasks/archive/2026-05/05-18-rvv-stage1-typed-body-route-authority-closure/prd.md`.
- This task should produce production path behavior, not another closure-only
  or report-only round.
