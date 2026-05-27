# Stage2 RVV Strided-Input Widening Dot-Reduce Selected-Body Realization Migration

## Goal

Move the `strided_input_widening_dot_reduce_add` generated artifact path behind
the public RVV selected lowering-boundary realization producer. The executable
path must be:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv strided_input_widening_dot_reduce_add body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv setvl / with_vl / strided i16 lhs load /
     strided i16 rhs load / i32 scalar seed / widening_dot_reduce /
     i32 scalar store body
  -> existing direct contraction provider facts
  -> route materialization facts
  -> math operand-binding facts, including lhs/rhs stride ABI bindings
  -> route-control provider plan
  -> direct contraction provider plan and statement owner
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence
```

Direct pre-realized route-entry support for
`strided_input_widening_dot_reduce_add` must be demoted or deleted. A generated
artifact for this operation must not be accepted from a direct route-entry
shortcut, pre-realized fixture authority, route id, artifact name, script
option, ABI string, exact intrinsic spelling, common EmitC behavior,
descriptor residue, source-front-door metadata, or legacy i32 helper authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV strided-input widening dot-reduce
  selected-body realization migration`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `52a2a88b rvv: demote widening dot reduce route entry`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
- Serial worker constraint: one Codex worker, no subagents or parallel agent
  workflows.

## Current Repository Facts

- The archived
  `05-27-stage2-rvv-widening-dot-reduce-selected-realization-migration` task
  demoted direct pre-realized route-entry support for
  `widening_dot_reduce_add` while preserving the selected-boundary producer
  path and ssh RVV correctness evidence.
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` currently keeps the
  `contraction` owner selected-body capable for widening MAcc, plain widening
  dot-reduce, strided-input widening dot-reduce, and computed-mask widening
  dot variants. Direct route-entry eligibility already excludes
  `TypedWideningMAccPreRealizedBodyOp` and
  `TypedWideningDotReducePreRealizedBodyOp`, but still accepts
  `TypedStridedInputWideningDotReducePreRealizedBodyOp`.
- `scripts/rvv_generated_bundle_abi_e2e.py` still reports
  `strided_input_widening_dot_reduce_add` as
  `supports_direct_pre_realized_route_entry`.
- Existing selected-boundary fixture
  `test/Target/RVV/pre-realized-selected-body-artifact-strided-input-widening-dot-reduce-add.mlir`
  represents the desired public selected-boundary producer path.
- Existing planning/provider code already exposes strided-input widening
  dot-reduce contraction family plans, route-control provider plans, math
  operand-binding facts with `lhs_stride` / `rhs_stride`, direct contraction
  provider plans, statement plans, target ABI mirrors, and generated-bundle
  evidence fields. This task should reuse those paths rather than add a new
  route table or common EmitC semantic branch.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes are allowed only for evidence tooling and script
   guardrails.
2. Keep the selected-boundary producer path positive for
   `strided_input_widening_dot_reduce_add`. It must realize the typed
   pre-realized body into `setvl`, `with_vl`, strided i16 source loads, scalar
   i32 seed/reduction facts, `widening_dot_reduce`, and i32 scalar store before
   provider route facts are collected.
3. Demote or delete direct pre-realized route-entry support for
   `strided_input_widening_dot_reduce_add`. Direct shortcut requests must fail
   closed with targeted route-entry diagnostics before bundle generation.
4. Preserve typed facts for operation kind, i16 lhs/rhs source binding, which
   operands are strided, lhs/rhs stride runtime_param binding, i32
   accumulation/result binding, widening dtype relation, dot-reduction shape,
   source/result SEW and LMUL relation, memory roles, runtime `n`/AVL/VL
   values, setvl placement, loop relation, selected capability, provider route
   facts, and artifact ABI order.
5. Do not demote or rewrite adjacent computed-mask widening dot variants in
   this round:
   `computed_masked_widening_dot_reduce_add` and
   `computed_masked_strided_input_widening_dot_reduce_add` are non-goals unless
   a minimal predicate split is required to isolate
   `strided_input_widening_dot_reduce_add`.
6. Unsupported or inconsistent selected-boundary input must fail closed before
   provider/common route construction where the current hooks expose it:
   missing runtime_param, missing mem_window/runtime ABI value, wrong source
   dtype, wrong accumulator/result dtype, wrong widening relation, wrong
   reduction shape, wrong strided operand, missing or stale stride binding,
   wrong lhs/rhs binding, dtype/config/policy mismatch, wrong AVL/VL relation,
   wrong setvl placement, missing capability, stale route id or mirror
   metadata, direct-route-entry-only authority, artifact-name/script-derived
   authority, exact-intrinsic-as-authority, and common-EmitC semantic
   invention.
7. Do not start computed-mask widening dot variants, `widen_i16_to_i32`,
   segment2, compare/select, unrelated strided memory cleanup, standalone
   reduction cleanup, new dtype/LMUL clone batches, high-level Linalg/frontend
   lowering, one-intrinsic wrapper dialects, selected-body framework rewrites,
   dashboard/report work, broad smoke matrices, or proof-only tests for
   completed adjacent routes as the main achievement.

## Acceptance Criteria

- [x] Production code no longer treats
      `strided_input_widening_dot_reduce_add` as direct pre-realized
      route-entry eligible, while the `contraction` selected-body realization
      owner still realizes it through the public selected lowering-boundary
      producer path.
- [x] C++ tests prove a typed pre-realized
      `strided_input_widening_dot_reduce_add` body belongs to the
      `contraction` realization owner, is not a direct route-entry consumer,
      and fails closed when `realizePreRealizedRVVRouteEntrySelectedBody` is
      used as a direct route-entry shortcut.
- [x] C++ or lit tests prove the selected-boundary
      `strided_input_widening_dot_reduce_add` path consumes realized typed body
      facts, including strided i16 lhs/rhs source loads, `lhs_stride` and
      `rhs_stride` ABI bindings, i32 seed/result layout, widening dot-reduce
      relation, route-control provider plan, math operand-binding facts,
      direct contraction provider plan, ABI order, and provider-supported
      mirror.
- [x] The generated-bundle script rejects
      `--direct-pre-realized-route-entry --op-kind
      strided_input_widening_dot_reduce_add` before route-entry materialization
      or bundle generation.
- [x] Generated-bundle dry-run for
      `strided_input_widening_dot_reduce_add` passes through
      `--tcrv-materialize-selected-lowering-boundaries`, records
      `route_entry_realization: false`, and records no direct route-entry
      materializer.
- [x] Real `ssh rvv` generated-bundle execution covers counts `0`, `1`, exact,
      tail, and stress cases with at least two signed i16 input patterns,
      lhs/rhs stride patterns, and expected i32 scalar reduction results.
- [x] Focused non-regression covers adjacent selected-body realization paths
      risked by this demotion, including `widening_dot_reduce_add`,
      `computed_masked_widening_dot_reduce_add`,
      `computed_masked_strided_input_widening_dot_reduce_add`,
      `widening_macc_add`, non-widening MAcc, computed-mask MAcc,
      runtime-scalar MAcc, scalar_broadcast_add, runtime_i32_splat_store, and
      computed_mask_select as practical focused filters or exact blocker.
- [x] A bounded touched-file authority scan finds no new executable or route
      claim depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [x] `git diff --check` passes.
- [x] Focused build/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [x] Trellis status, journal/archive, and final commit are truthful.

## Validation Plan

1. Validate Trellis task context.
2. Narrow the RVV selected-body realization owner registry so the
   `contraction` owner remains selected-boundary capable, but
   `strided_input_widening_dot_reduce_add` is not direct route-entry eligible.
3. Update generated-bundle tooling and focused script coverage so
   `--direct-pre-realized-route-entry --op-kind
   strided_input_widening_dot_reduce_add` fails closed.
4. Keep or strengthen selected-boundary lit/script coverage for
   `pre-realized-selected-body-artifact-strided-input-widening-dot-reduce-add.mlir`.
5. Run focused C++ plugin tests and focused lit/script tests.
6. Run generated-bundle selected-boundary dry-runs for
   `strided_input_widening_dot_reduce_add`.
7. Run real `ssh rvv` generated-bundle evidence for counts `0,1,16,23,257`
   using the selected lowering-boundary producer path and strided input
   harness.
8. Run focused non-regression for selected-body producer paths listed in the
   acceptance criteria.
9. Run authority scan, `git diff --check`, task validation, and
   `check-tianchenrv`.

## Out Of Scope

- Computed-mask widening dot, computed-mask strided-input widening dot,
  `widen_i16_to_i32`, segment2 follow-ons, compare/select, unrelated strided
  memory cleanup, standalone reduction cleanup, new dtype/LMUL clone sets,
  high-level Linalg/Vector/StableHLO frontend work, source-front-door
  construction, descriptor-driven compute, direct C/source-export paths, and
  legacy `RVVI32M1` / `rvv-i32m1` compatibility.
- Dashboard/report-only/prompt-only work as the main achievement.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Relevant archived task read:
  `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-widening-dot-reduce-selected-realization-migration/`.
- Primary production files to inspect or modify:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and focused target/script tests.

## Implementation Result

- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` now excludes
  `TypedStridedInputWideningDotReducePreRealizedBodyOp` from the direct
  pre-realized route-entry predicate while keeping it in the `contraction`
  selected-body realization owner predicate.
- `scripts/rvv_generated_bundle_abi_e2e.py` no longer reports
  `strided_input_widening_dot_reduce_add` as
  `supports_direct_pre_realized_route_entry`. Its direct-entry allowlist and
  help text now describe only the remaining direct route-entry families.
- `test/Plugin/RVVExtensionPluginTest.cpp` now proves that
  `strided_input_widening_dot_reduce_add` is selected-body-realization capable,
  is not direct route-entry eligible, fails closed through
  `realizePreRealizedRVVRouteEntrySelectedBody`, and still realizes into typed
  strided RVV facts consumed by route analysis, math operand-binding facts, and
  the direct contraction provider plan.
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-strided-input-widening-dot-reduce-add-dry-run.test`
  now asserts `route_entry_realization: false` and rejects the route-entry
  materializer marker in selected-boundary generated-bundle evidence.
- `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-strided-input-widening-dot-reduce-add-fail-closed.test`
  covers the script-level fail-closed path for
  `--direct-pre-realized-route-entry --op-kind
  strided_input_widening_dot_reduce_add`.

## Validation Result

- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`
- [x] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Direct route-entry fail-closed probe for
      `--direct-pre-realized-route-entry --op-kind
      strided_input_widening_dot_reduce_add`.
- [x] Selected-boundary generated-bundle dry-run for
      `strided_input_widening_dot_reduce_add`, counts `0,1,16,23,257`, with
      evidence showing `materializer: tcrv-materialize-selected-lowering-boundaries`,
      `route_entry_realization: false`, and `pre_realized_body_consumed: true`.
- [x] Real `ssh rvv` generated-bundle run for
      `strided_input_widening_dot_reduce_add`, counts `0,1,16,23,257`, with
      signed horizontal dot products, i32 seed accumulation, `lhs_stride=2`,
      `rhs_stride=3`, scalar output, skipped source elements ignored, and tail
      preservation.
- [x] Adjacent widening dot family selected-boundary dry-run:
      `widening_macc_add`, `widening_dot_reduce_add`,
      `strided_input_widening_dot_reduce_add`,
      `computed_masked_widening_dot_reduce_add`, and
      `computed_masked_strided_input_widening_dot_reduce_add`.
- [x] Computed-mask widening dot direct route-entry non-regression dry-run:
      `computed_masked_widening_dot_reduce_add` and
      `computed_masked_strided_input_widening_dot_reduce_add`.
- [x] Full `cmake --build build --target check-tianchenrv -j2`: 395/395
      passed.
- [x] Production added-line authority scan found no new legacy i32,
      source-front-door, descriptor, direct-C, artifact-name, route-id, or
      exact-intrinsic authority in changed production files. Test-only hits are
      fail-closed messages or `implicit-check-not` assertions.
- [x] `git diff --check`
- [x] `python3 ./.trellis/scripts/task.py validate
      .trellis/tasks/05-27-05-27-stage2-rvv-strided-input-widening-dot-reduce-selected-realization-migration`
- [x] Spec update judgment: no `.trellis/spec/**` edit needed. Existing RVV
      specs already encode selected-body realization, direct contraction
      provider ownership, route-entry demotion, provider-built route, mirror
      metadata, neutral EmitC, and real RVV evidence contracts used here.

## Open Questions

- None blocking at task start. The user brief and current repository evidence
  agree on the module owner and non-goals.
