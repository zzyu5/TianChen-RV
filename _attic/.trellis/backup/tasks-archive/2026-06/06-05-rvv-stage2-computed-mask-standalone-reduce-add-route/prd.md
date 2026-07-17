# Stage2 RVV computed-mask standalone reduce-add route executable closure

## Goal

Close one bounded Stage 2 computed-mask standalone reduction proof route:
a selected `tcrv.exec` RVV variant with an explicit typed
`tcrv_rvv.masked_standalone_reduce {kind = "add"}` body must flow through
RVV provider-derived mask/reduction/runtime facts, `TCRVEmitCLowerableRoute`,
common EmitC/export, target artifact validation, and executable `ssh rvv`
evidence only if the existing generated-bundle harness can be extended without
broad infrastructure.

This is one masked reduction capability. It is not a broad reduction matrix,
not a high-level frontend task, not a benchmark, and not a metadata-only
completion.

## What I Already Know

- No `.trellis/.current-task` existed when this round began, so this task was
  created from the Hermes Direction Brief.
- The worktree was clean at the start and the latest commit was
  `1ed9eac0 rvv: close standalone reduce-add executable abi`.
- The previous archived task
  `.trellis/tasks/archive/2026-06/06-05-rvv-stage2-standalone-reduce-add-executable-abi-closure/`
  proved the unmasked `standalone_reduce_add` generated artifact, external ABI
  harness, and real `ssh rvv` correctness for `n = 0, 1, 16, 17, 257`.
- The previous typed reduction task added provider-derived standalone
  `reduction_kind` mirrors and stale-mirror target validation for standalone
  reduction routes.
- This round's delta is computed-mask standalone reduce-add: mask producer,
  compare predicate/intrinsic, mask role/source/memory form, inactive-lane
  zeroing or neutralization, accumulator/result layout, runtime AVL/order, and
  statement plan must be real provider-derived facts.
- Target metadata remains a mirror only. Stale mirrors or externally invented
  mask/reduction/runtime facts must fail before artifact acceptance.

## Requirements

- Preserve the authority chain:
  typed selected `tcrv_rvv.masked_standalone_reduce` body -> RVV provider facts
  -> lowerable route -> common EmitC/export -> target artifact -> optional
  executable evidence.
- The typed body/provider route must structurally carry or derive:
  - `reduction_kind = add`;
  - source/result element type and vector type facts;
  - SEW, LMUL, tail policy, and mask policy;
  - runtime AVL source and one runtime-element-count ABI parameter;
  - runtime ABI order for compare/source/accumulator/result/count operands;
  - mask role, mask source, mask memory form, compare predicate, and compare
    intrinsic;
  - inactive-lane zeroing or neutralization contract;
  - accumulator layout, result layout, reduction store VL, and scalar-result
    boundary;
  - provider statement plan for compare, mask, source load, reduction, and
    result store.
- Common EmitC may only carry provider-supplied payloads. It must not infer
  mask semantics, reduction semantics, dtype, policy, runtime ABI roles, or
  RVV intrinsic spelling.
- Target artifact validation must compare provider-derived route facts against
  candidate mirrors before accepting the bundle.
- Unsupported or stale facts must fail closed with focused diagnostics:
  reduction kind, mask role/source/memory form, compare predicate/intrinsic,
  inactive-lane contract, accumulator/result layout, runtime AVL, runtime ABI
  order, route operand binding, header/type mapping, statement plan, and route
  string/artifact-name authority.
- If live production code already supports the route, finish by proving it with
  focused positive and negative evidence. Add source changes only for real gaps.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` describe this bounded
      computed-mask standalone reduce-add task and its spec basis.
- [x] Focused positive route/artifact evidence proves provider-derived
      computed-mask standalone reduce-add facts, including mask role/source,
      mask memory form, compare predicate/intrinsic, inactive-lane contract,
      reduction kind, accumulator/result layout, runtime AVL/order, operand
      binding, headers/types, and statement plan.
- [x] Negative evidence proves stale or missing reduction kind, mask
      role/source/memory form, compare predicate/intrinsic, inactive-lane
      contract, accumulator/result layout, runtime AVL, runtime ABI/order,
      operand binding, type/header mapping, statement plan, and route
      string/artifact-name authority fail closed before artifact acceptance.
- [x] Focused `tcrv-opt` / `tcrv-translate` / RVV plugin / target artifact
      tests for changed behavior pass.
- [x] If executable behavior is claimed, generated artifact plus real
      `ssh rvv` correctness passes with masked input cases covering false-mask
      lanes, tail, and `n = 0`.
- [x] Bounded old-authority/q-name scan over touched files and added diff
      lines shows no new positive legacy `i32m1`, descriptor,
      source-front-door, route-string/artifact-name/ABI-string/test-name,
      intrinsic-spelling, or common EmitC RVV semantic authority.
- [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are recorded.
- [x] The task is finished/archived and one coherent commit records the work if
      the task completes.

## Out of Scope

- Broad add/min/max/widening/masked reduction matrices.
- High-level Linalg/Vector/StableHLO frontend lowering.
- Per-Linalg route authority or one-intrinsic wrapper dialects.
- Common EmitC choice of mask or reduction semantics.
- Benchmarks or performance claims.
- Gearbox/autotuning work.
- Source-front-door positive routes.
- Route-string, artifact-name, ABI-string, test-name, descriptor,
  intrinsic-spelling, or metadata as route authority.
- New dtype-prefixed helper op families or legacy `i32m1` positive routes.

## Technical Notes

- Required specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/index.md`, and
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
- Previous task context:
  `.trellis/tasks/archive/2026-06/06-05-rvv-stage2-typed-reduction-accumulation-route/`
  and
  `.trellis/tasks/archive/2026-06/06-05-rvv-stage2-standalone-reduce-add-executable-abi-closure/`.
- Primary inspection surfaces from the brief:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `test/Target/RVV/explicit-selected-body-artifact-standalone-reduce-add.mlir`,
  `test/Dialect/RVV/standalone-reduction-dataflow.mlir`, and
  `scripts/rvv_generated_bundle_abi_e2e.py` if executable closure is in scope.

## Decision

Use one computed-mask standalone reduce-add proof route as the MVP. First prove
whether current production provider/target code already supports it. If not,
make the smallest production-path change that derives the missing facts from
typed body/config/runtime structure and fails closed on stale mirrors. Extend
the executable harness only if the generated ABI path is a bounded change.

## Completion Evidence

- Production code already supported the computed-mask standalone reduce-add
  route and generated-bundle executable path. This round closed focused
  evidence gaps instead of adding a new production route.
- Added route/header FileCheck coverage for `tcrv_rvv.reduction_kind = add` in
  both pre-realized and explicit computed-mask standalone reduce-add fixtures.
- Added target artifact negative coverage that rewrites the computed-mask
  standalone reduce-add `tcrv_rvv.reduction_kind` mirror to `min` and expects
  artifact validation to fail before acceptance.
- Added testing-spec evidence rules for computed-mask standalone reduction
  generated bundles, including provider-derived reduction kind, compare-mask
  facts, inactive-lane neutralization, runtime ABI order, and all-inactive
  seed-preservation oracle requirements.
- Focused checks passed:
  `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-target-artifact-export-test`,
  `build/bin/tianchenrv-target-artifact-export-test`,
  `build/bin/tcrv-opt
  test/Dialect/RVV/computed-mask-standalone-reduction-dataflow.mlir
  --split-input-file --verify-diagnostics`, and FileCheck route/header
  checks for both computed-mask standalone reduce-add fixtures.
- Generated-bundle dry-run passed at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/computed-mask-standalone-reduce-add-final-dryrun`
  with evidence for `reduction_kind=add`, mask role/source/memory form,
  inactive-lane zeroing, runtime ABI order `cmp_lhs,cmp_rhs,src,acc,out,n`,
  compare/merge/reduction/store intrinsics, and all-inactive-mask behavior.
- Real `ssh rvv` executable evidence passed at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/computed-mask-standalone-reduce-add-final-ssh`.
  The remote target reported `remote_arch=riscv64`,
  `/usr/bin/clang`, `Ubuntu clang version 18.1.3 (1ubuntu1)`, and
  `PASS op=computed_mask_standalone_reduce_add counts=0,1,16,17,257
  seeds=-11,17 patterns=0,1`.
- Staged-diff old-authority scan over code/test added lines produced no
  matches. The full staged-diff authority scan matched only negative guardrail
  text in specs/task notes, not positive route authority.
- `git diff --check` and `git diff --cached --check` passed. The task was
  archived as completed under
  `.trellis/tasks/archive/2026-06/06-05-rvv-stage2-computed-mask-standalone-reduce-add-route/`.
