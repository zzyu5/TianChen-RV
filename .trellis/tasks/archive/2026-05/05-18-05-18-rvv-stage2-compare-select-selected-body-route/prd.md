# RVV Stage 2 compare/select mask selected-body route

## Goal

Introduce one coherent RVV Stage 2 selected-body capability for bounded i32m1
compare/select dataflow on the corrected vector-level `tcrv_rvv` surface. A
selected `tcrv.exec` RVV variant must carry explicit typed compare and select
ops inside the selected `tcrv_rvv.with_vl` body; RVV legality and route
construction must consume that typed body as the semantic authority; common
EmitC and target artifact export may only consume the validated route payload
and selected-candidate mirrors.

## What I Already Know

- Current repo root is `/home/kingdom/phdworks/TianchenRV`; task start found a
  clean worktree at HEAD `172c6eb`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief and will be started before implementation.
- `.trellis/spec/index.md` keeps compiler implementation in
  C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck. Python is tooling only.
- `.trellis/spec/extension-plugins/rvv-plugin.md` defines Stage 2 as RVV
  completion work on the corrected vector-level surface. Compare/select is a
  listed Stage 2 low-level RVV coverage class.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires EmitC routes to
  faithfully consume selected extension-family body structure. Common
  EmitC/export mechanics must not infer RVV semantics from route ids, artifact
  metadata, descriptor residue, source-front-door names, ABI parameter names,
  or test fixture names.
- The archived broadcast selected-body task established the immediate pattern:
  explicit typed RVV op, verifier/config contract, plugin-owned route
  construction, EmitC materialization, and target artifact candidate validation.

## Requirements

- Inspect the current RVV typed-body surface, verifier/config contract,
  construction protocol, RVV EmitC route provider, target artifact validation,
  and focused tests named in the Direction Brief.
- Add or extend typed `tcrv_rvv` IR for a bounded i32m1 predicate-producing
  compare op and a predicate-consuming select op inside one selected
  `tcrv_rvv.with_vl` body.
- Define the typed mask/dataflow boundary: compare must produce a typed
  predicate/mask result, select must consume that exact predicate and compatible
  data operands under the same selected VL/config boundary.
- Ensure RVV legality accepts compare/select selected bodies only with explicit
  runtime ABI values, setvl/with_vl consistency, typed vector inputs, a typed
  mask result, and an explicit select consuming that predicate.
- Ensure the RVV route builder emits compare and select/merge EmitC call steps
  with source-op provenance and rejects unsupported or implied-only
  compare/select semantics before common EmitC/materialization or target
  export.
- Preserve existing vector-vector arithmetic and broadcast arithmetic selected
  body routes.
- Keep common EmitC/materialization/target export neutral. They may consume
  validated route payloads and selected-candidate mirrors only.

## Acceptance Criteria

- [x] A selected RVV variant containing explicit bounded i32m1 compare/select
      `tcrv_rvv` ops is accepted by RVV legality.
- [x] Compare produces a typed mask/predicate value and select consumes that
      mask under the same selected `with_vl` body and compatible config.
- [x] The accepted typed body produces a supported RVV emission plan and a
      faithful `TCRVEmitCLowerableRoute`.
- [x] Route payload evidence records compare and select source-op provenance in
      body order, not route id, artifact name, ABI names, descriptor residue,
      source-front-door metadata, or fixture names.
- [x] Materialized EmitC contains the RVV-owned compare and select/merge call
      steps.
- [x] Target artifact candidate validation/export accepts compare/select only
      through the materialized typed-body route and rejects metadata-only
      candidates.
- [x] Bad cases fail closed for missing selected typed body, masks not produced
      by the typed compare op, select using mismatched VL/config, unsupported
      predicate forms, wrong ABI roles/order, unsupported common/core RVV
      semantics, and descriptor/direct-C/source-front-door authority trying to
      choose compare/select behavior.
- [x] Existing vector-vector and broadcast arithmetic selected-body routes keep
      working.

## Completion Notes

- Added typed `tcrv_rvv.i32m1_mask`, `tcrv_rvv.i32_cmp_eq`, and
  `tcrv_rvv.i32_select` on the bounded i32m1 selected-body surface.
- RVV dialect verification now requires compare/select to live directly under
  the same `tcrv_rvv.with_vl`, use the same VL token, preserve i32m1 data
  types, and consume a mask produced by the typed compare op in the same body.
- RVV construction and route provider now expose a plugin-owned `cmp_select`
  selected-body route that emits compare then select/merge EmitC call steps
  with source-op provenance.
- Target artifact validation/export accepts the compare/select candidate
  through the materialized typed-body route. Metadata-only authority remains
  rejected by existing target artifact coverage.
- Focused build targets, focused C++ tests, focused lit filters,
  `git diff --check`, targeted residue scans, and full `check-tianchenrv`
  passed. One focused construction test initially failed because the new
  compare/select role graph has an extra compute role; the expectation was
  repaired and rerun.
- No `ssh rvv` runtime evidence was produced, so this round makes no hardware
  correctness or performance claim.

## Definition Of Done

- Focused dialect/verifier lit coverage for mask/dataflow syntax and
  fail-closed config/VL/ABI cases.
- Focused plugin tests for selected-body legality, emission plan metadata,
  route payload order, source-op provenance, and EmitC materialization.
- Focused target artifact tests proving compare/select selected bodies reach
  candidate validation/export through the materialized route and metadata-only
  candidates fail closed.
- Relevant focused build targets and lit filters pass.
- `git diff --check` passes.
- `check-tianchenrv` is run if practical; otherwise the exact blocker is
  recorded.
- Targeted scans over changed RVV/common surfaces show no descriptor route
  authority, direct-C semantic exporter, common/core RVV compute branch, or
  source-front-door compare/select authority.
- Trellis task files and workspace journal are updated truthfully.
- If complete, the task is finished/archived and one coherent commit is
  created.

## Out Of Scope

- High-level Linalg/Vector/StableHLO frontend lowering.
- Generic Vector dialect lowering.
- Per-Linalg-op lowerers, high-level kernel ops, one-op-per-intrinsic wrapper
  families, dtype/LMUL clone batches, scalar fallback, IME, Offload, TensorExt,
  global tuning databases, readiness dashboards, or broad benchmark matrices.
- Common/core RVV semantic branches, descriptor compatibility paths, direct C
  semantic exporters, source-front-door authority, or Python compiler-core
  behavior.
- `ssh rvv` runtime correctness or performance claims unless this round
  actually builds and runs hardware evidence.

## Technical Notes

- Primary read targets from the Direction Brief:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `test/Dialect/RVV/dataflow.mlir`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`, and
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`.
- Previous Stage 2 broadcast selected-body PRD:
  `.trellis/tasks/archive/2026-05/05-18-rvv-stage2-elementwise-broadcast-selected-body/prd.md`.
- This task should produce production path behavior, not another metadata-only,
  report-only, or helper-only round.
