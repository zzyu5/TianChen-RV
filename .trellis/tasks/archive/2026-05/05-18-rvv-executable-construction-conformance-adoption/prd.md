# RVV executable construction conformance adoption

## Goal

Migrate the existing bounded RVV i32m1 add/sub/mul executable construction path
onto the common executable construction conformance surface introduced by the
previous task. The RVV selected path must visibly consume the shared C++
contracts for selected executable role sequence, selected lowering-boundary
conformance, and ordered construction artifact metadata where those contracts
fit, while keeping RVV config/VL legality, runtime ABI roles, intrinsic callee
mapping, typed RVV op requirements, and materialized EmitC route construction
inside RVV ownership.

## What I already know

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Worktree was clean before this task.
- Current HEAD is `0d37c27 plugin: add executable construction conformance`.
- No `.trellis/.current-task` existed at session start, so this task was
  created from the Direction Brief before source edits.
- The previous archived task added common construction APIs for selected role
  sequence inspection/collection, selected lowering-boundary conformance, and
  artifact metadata validation, then demonstrated the surface on
  TensorExtLite.
- RVV already uses the common manifest and typed-role graph model, but still
  duplicates generic selected-role, selected-boundary, and construction
  artifact metadata checks in RVV-local code.
- RVV remains the current primary real hardware path. This round is adoption
  of common conformance around the existing i32m1 add/sub/mul route, not a new
  runtime or coverage claim.

## Requirements

- Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only be used for Trellis tooling, probes, scripts, and
  validation wrappers.
- RVV's construction manifest, typed role realization, selected boundary
  validation, EmitC route readiness, and target artifact candidate validation
  must consume common construction conformance APIs for generic role/order,
  selected boundary, interface/order, and artifact metadata checks where
  applicable.
- Common construction code may validate only generic role/interface/order,
  selected boundary, and artifact metadata contracts. It must not learn RVV
  SEW/LMUL/VL, runtime ABI roles, intrinsic names, typed RVV op dataflow, or
  i32m1 arithmetic semantics.
- RVV-local code must continue to own RVV config/VL legality,
  runtime ABI role binding, intrinsic callee mapping, typed RVV op shape,
  materialized EmitC route building, and RVV runtime/target packaging details.
- The production/default RVV selected path must remain the explicit typed RVV
  i32m1 add/sub/mul path. Do not add new SEW, LMUL, tail/mask, dtype, source
  seed, or generic RVV lowering behavior.
- Do not add descriptor-driven computation, direct C semantic export, source
  export authority, compatibility wrappers, legacy routes, Python compiler-core
  behavior, common/core RVV semantic branches, or standalone evidence/report
  work.
- If object bytes or target artifact emission changes, refresh `ssh rvv`
  evidence. If only conformance validation wiring changes, state why prior
  RVV object evidence remains valid.

## Acceptance Criteria

- RVV i32m1 selected role sequence validation visibly uses
  `plugin::construction` selected executable role sequence APIs, including
  fail-closed coverage for missing, duplicate, and out-of-order role ops.
- RVV selected lowering-boundary validation visibly uses the common selected
  lowering-boundary conformance API for generic boundary attributes and
  required capabilities, while RVV-local code still checks the unique
  `tcrv_rvv.with_vl` boundary and RVV i32m1 config/VL contract.
- RVV i32m1 EmitC route readiness uses the common role-sequence conformance
  surface before constructing the RVV-owned materialized EmitC route.
- RVV target artifact candidate validation uses common artifact metadata
  validation for construction metadata before applying RVV-local runtime
  AVL/VL metadata and ABI checks.
- Negative tests fail closed for missing/out-of-order RVV role ops, missing
  common interfaces, wrong selected boundary, mismatched runtime ABI or
  artifact metadata, stale descriptor/source metadata, and source/body-free
  RVV capability input.
- Existing positive RVV materialized EmitC object/header/bundle behavior
  remains available.
- Focused scans over touched common/RVV plugin/target/tests show no
  descriptor-driven route authority, direct C semantic exporter, source-export
  route, Python compiler-core behavior, or extension-specific semantic branch
  added to common construction code.

## Out of Scope

- New RVV SEW/LMUL/tail/mask families, new dtype coverage, or new arithmetic
  operations.
- New source seeds, generic RVV lowering, or new high-level frontend lowering.
- Descriptor or direct-C routes, source-export routes, compatibility wrappers,
  legacy route aliases, or metadata-generated compute bodies.
- Moving RVV config/VL, intrinsic, typed op, or runtime ABI semantics into
  common construction code.
- Fresh RVV runtime/correctness/performance claims unless emitted object bytes
  change.

## Definition of Done

- PRD, Trellis context, and task status accurately describe this round.
- Code changes are production-path changes, not helper-only or report-only
  work.
- Focused builds/tests pass for affected construction, RVV plugin, and target
  artifact export coverage.
- Focused lit for RVV materialized EmitC/source-seed artifact paths is run if
  touched.
- `git diff --check` passes.
- `check-tianchenrv` is run if practical; otherwise the blocker is recorded.
- Task is finished/archived if complete.
- One coherent commit is created if the task is complete.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Previous task context read:
  `.trellis/tasks/archive/2026-05/05-18-executable-construction-template-conformance-surface/prd.md`.
- Main source surfaces:
  `include/TianChenRV/Plugin/ConstructionProtocol.h`,
  `lib/Plugin/Construction/ConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`, and
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`.
- Main tests:
  `test/Plugin/ConstructionProtocolCommonTest.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.

## Completion Notes

- Common construction role-sequence conformance now accepts an extension-owned
  ordered role-operation list plus optional construction-order evidence, so RVV
  can validate selected executable role order without making common code aware
  of RVV semantics.
- RVV i32m1 add/sub/mul selected EmitC route readiness now verifies the
  bounded runtime ABI/configure/scope/load/compute/store role sequence through
  `verifyRVVI32M1ArithmeticSelectedRoleSequence`, backed by
  `plugin::construction` selected role-sequence inspection and completeness
  checks.
- RVV selected `tcrv_rvv.with_vl` boundary validation now uses common selected
  lowering-boundary conformance for generic selected-path attributes and
  required capability metadata, while RVV-local code still owns unique boundary
  selection, config/VL legality, typed RVV op shape, and intrinsic/runtime ABI
  semantics.
- RVV construction artifact metadata validation now delegates generic
  construction metadata checks to
  `plugin::construction::verifyConstructionArtifactMetadata` before RVV-local
  target artifact validation applies RVV runtime ABI and route-specific checks.
- No object emission or target artifact bytes were intentionally changed. Fresh
  `ssh rvv` evidence was not refreshed because this round changed conformance
  validation wiring and fail-closed checks, not generated RVV C/object payloads.

## Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-executable-construction-conformance-adoption`
- `cmake --build build --target tianchenrv-construction-protocol-common-test tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- Focused RVV lit for RVV materialized EmitC/source-seed/target artifact paths
  from `build/test` passed 14/14.
- Focused residue scans over touched common/RVV plugin/target/tests found no
  descriptor-driven route authority, direct-C semantic exporter, source-export
  route, Python compiler-core behavior, or extension-specific semantic branch
  added to common construction code.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` passed 122/122.
