# Stage2 RVV standalone_reduce_add Selected-Body Realization Migration

## Goal

Move the generated artifact path for `standalone_reduce_add` behind the RVV
plugin-local selected lowering-boundary realization producer and remove its
active direct pre-realized route-entry shortcut authority.

The production path for this task must be:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv standalone reduction body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv body preserving lhs input memory role,
     accumulator seed/result semantics, scalar result binding, reduction
     operation kind, i32 result dtype, SEW32, LMUL m1, policy, runtime n/AVL/VL,
     setvl and with_vl placement, selected requires, and ABI order
  -> standalone-reduction route-family facts
  -> route materialization facts
  -> math operand-binding facts
  -> route-control provider plan
  -> migrated standalone-reduction statement-plan owner
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence
```

The generated artifact must not be accepted from a direct route-entry shortcut,
route id, artifact name, script option, ABI string, exact intrinsic spelling,
common EmitC behavior, descriptor residue, source-front-door metadata,
pre-realized fixture status, or legacy i32 helper authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV standalone reduce selected-body
  realization migration`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `d8643144 rvv: demote cmp select route entry`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint: one Codex worker; no subagents, spawned agents,
  parallel agents, or multi-agent workflows.

## Current Repository Facts

- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` has a standalone reduction
  selected-body realization owner. The owner consumes
  `TypedStandaloneReducePreRealizedBodyOp` and realizes it into `setvl`,
  `with_vl`, `load`, `standalone_reduce`, and `store`.
- The same owner currently marks `TypedStandaloneReducePreRealizedBodyOp` as
  direct route-entry eligible through
  `isPreRealizedRVVStandaloneReductionRouteEntryOp(...)` when `op_kind` and
  `memory_form` are supported.
- The route-entry registry still lists `standalone reduction` with
  `isPreRealizedRVVStandaloneReductionRouteEntryOp`, so route-entry requests
  can bypass the public selected lowering-boundary producer for
  `standalone_reduce_add`.
- `scripts/rvv_generated_bundle_abi_e2e.py` currently includes
  `self.is_standalone_reduce_add` in
  `supports_direct_pre_realized_route_entry`, and its direct-route diagnostic
  says `pre-realized standalone_reduce_add/... fixtures`.
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-standalone-reduce-add-dry-run.test`
  currently invokes `--direct-pre-realized-route-entry` and asserts
  `materializer: rvv-route-entry-selected-body-realization` and
  `route_entry_realization: true`.
- `test/Target/RVV/pre-realized-selected-body-artifact-standalone-reduce-add.mlir`
  already models the selected-boundary standalone reduction artifact and
  FileChecks realized standalone reduction/provider/target artifact facts.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
  `lib/Target/RVV/RVVTargetSupportBundle.cpp` already contain standalone
  reduction family plans, materialization facts, math operand-binding facts,
  route-control provider facts, migrated statement-plan ownership, provider
  route construction, and target artifact ABI validation. This task should
  reuse those boundaries unless focused checks expose stale direct-route
  assumptions.
- Archived task
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-cmp-select-selected-realization-migration/`
  completed the same migration pattern for plain compare/select: selected
  lowering-boundary path stayed positive, direct shortcut failed closed, the
  generated-bundle dry-run recorded `route_entry_realization: false`, real
  `ssh rvv` passed, and `check-tianchenrv` passed 400/400.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes are allowed only for evidence tooling, harnesses,
   generated-bundle guardrails, and artifact parsing.
2. Keep `TypedStandaloneReducePreRealizedBodyOp` under the RVV `standalone
   reduction` selected-body realization owner. The public selected
   lowering-boundary producer must realize the typed pre-realized body before
   provider route facts are collected.
3. Demote or delete direct pre-realized route-entry support for
   `standalone_reduce_add`. Direct shortcut requests must fail closed before
   route-entry materialization, bundle generation, or provider route
   construction.
4. Preserve typed facts for operation kind `standalone_reduce_add`, source input
   role `lhs`, accumulator seed role `acc`, output scalar result role `out`,
   runtime element count `n`, scalar reduction/result semantics, i32 element
   type, SEW32 LMUL m1, tail/mask policy, runtime `n`/AVL/VL values,
   `setvl`/`with_vl` placement, selected variant `requires`,
   standalone-reduction route-family plan, math operand-binding facts,
   route-control facts, migrated statement-plan facts, provider route facts,
   and artifact ABI order `lhs,acc,out,n`.
5. Fail closed for unsupported or inconsistent selected-boundary input,
   including missing runtime_param, missing mem_window/runtime ABI value, wrong
   input/output binding, wrong accumulator/result shape, wrong dtype/config/
   policy, wrong AVL/VL relation, wrong setvl placement, stale route id or
   mirror metadata, direct-route-entry-only authority, artifact-name or
   script-derived authority, exact-intrinsic-as-authority, and common-EmitC
   semantic invention.
6. Reuse existing standalone reduction route-family, materialization,
   route-control provider, math operand-binding, migrated statement-plan,
   target artifact, and generated-bundle boundaries where they already express
   the required facts. Do not add a new central route table, descriptor path,
   common EmitC semantic branch, or source-front-door path.
7. Do not start segment2, computed-masked segment2 load/store/update,
   segment2 interleave/deinterleave, unrelated memory movement cleanup,
   compare/select, widening dot, MAcc, conversion, new dtype/LMUL clone batches,
   high-level Linalg/frontend lowering, one-intrinsic wrapper dialects,
   selected-body realization framework rewrites, dashboard/report work, broad
   smoke matrices, or evidence-only tasks.

## Acceptance Criteria

- [x] Production code no longer treats `standalone_reduce_add` as direct
      pre-realized route-entry eligible, while the `standalone reduction`
      selected-body realization owner still realizes
      `TypedStandaloneReducePreRealizedBodyOp` through the public selected
      lowering-boundary producer.
- [x] C++ tests prove a typed pre-realized `standalone_reduce_add` body belongs
      to the `standalone reduction` realization owner, is not a direct
      route-entry consumer, and fails closed when
      `realizePreRealizedRVVRouteEntrySelectedBody` is used as a direct
      route-entry shortcut.
- [x] C++ or lit tests prove the selected-boundary path consumes realized typed
      facts, including `setvl`, `with_vl`, source `load`,
      `standalone_reduce`, scalar result `store`, ABI roles
      `lhs,acc,out,n`, runtime `n`/AVL/VL, standalone-reduction family plan,
      math operand-binding facts, route-control facts, migrated statement-plan
      facts, provider-supported mirrors, scalar-result runtime boundary, and
      artifact ABI order.
- [x] The generated-bundle script rejects
      `--direct-pre-realized-route-entry --op-kind standalone_reduce_add`
      before route-entry materialization or bundle generation.
- [x] Generated-bundle dry-run for pre-realized `standalone_reduce_add` passes
      through `--tcrv-materialize-selected-lowering-boundaries`, records
      `route_entry_realization: false`, records selected-body producer
      evidence, records realized standalone reduction/provider facts, and
      records no direct route-entry materializer.
- [x] Real `ssh rvv` generated-bundle execution covers counts including `0`,
      `1`, exact/full-chunk, tail, and stress cases, with signed input/seed
      patterns and expected scalar reduction results.
- [x] Focused non-regression covers adjacent selected-body realization paths
      risked by this demotion, including at minimum the RVV extension plugin
      test and a remaining direct route-entry family that should stay direct.
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

## Completion Results

- Production movement:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` no longer registers
  standalone reduction as a direct pre-realized route-entry owner. The
  `standalone reduction` owner still consumes
  `TypedStandaloneReducePreRealizedBodyOp` through the public selected
  lowering-boundary producer.
- Generated-bundle boundary:
  `scripts/rvv_generated_bundle_abi_e2e.py` no longer includes
  `standalone_reduce_add` in
  `supports_direct_pre_realized_route_entry`. Direct
  `--direct-pre-realized-route-entry --op-kind standalone_reduce_add` fails
  before bundle generation with the bounded segment2-only direct-route-entry
  allowlist diagnostic.
- Selected-boundary facts:
  focused C++ and lit coverage now checks producer-realized `setvl`,
  `with_vl`, source `load`, `standalone_reduce`, scalar result `store`, ABI
  order `lhs,acc,out,n`, runtime `n`,
  `rvv-standalone-reduction-route-family-plan.v1`, route operand binding,
  provider-supported mirrors, scalar-result
  runtime boundary, and `route_entry_realization: false`.
- Dry-run evidence:
  pre-realized selected-body generated-bundle dry-run passed for
  `standalone_reduce_add` with counts `0,1,7,16,23,257`, recording
  `materializer: tcrv-materialize-selected-lowering-boundaries`,
  `selected_body_realization_producer` set to
  `rvv-plugin-local-selected-body-realization-owner-registry`, and
  `route_entry_realization: false`.
- Real RVV evidence:
  `ssh rvv` generated-bundle execution passed for `standalone_reduce_add` over
  counts `0,1,7,16,23,257` with seeds `-11` and `17`. The run reported
  tail-preserved correctness for each count/seed case and
  `PASS op=standalone_reduce_add counts=0,1,7,16,23,257 seeds=-11,17`.
- Checks:
  `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test -j2`,
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`,
  generated-bundle self-test, manual direct fail-closed probe, manual selected
  dry-run, focused lit for standalone reduction and a remaining direct segment2
  route-entry family, bounded authority scans, `git diff --check`, Trellis
  task validation, and `cmake --build build --target check-tianchenrv -j2`
  passed with `401/401` tests.
- Self-repair:
  no source repair was needed after validation. One lit invocation was rerun
  from `build/test` after confirming the repo-root invocation fails only
  because `lit.site.cfg.py` resolves its source config relative to
  `build/test`.
- Spec-update judgment:
  `.trellis/spec/**` was updated to record the durable rule that standalone
  reduction belongs to the selected-body realization owner but is
  selected-boundary-only until a later explicit owner task adds direct
  route-entry support and matching evidence.

## Validation Plan

1. Validate Trellis task context.
2. Inspect current selected-body realization, route planning/provider,
   target-support bundle, generated-bundle script, focused plugin tests, and
   target/script tests for active direct `standalone_reduce_add` route-entry
   authority.
3. Narrow standalone reduction route-entry eligibility so
   `standalone_reduce_add` remains selected-boundary capable but is not direct
   route-entry eligible.
4. Update generated-bundle tooling/lit coverage so direct pre-realized
   route-entry mode fails closed for `standalone_reduce_add`.
5. Strengthen selected-boundary C++/lit/script coverage so evidence explicitly
   checks `route_entry_realization: false`, selected-body producer evidence,
   realized standalone reduction facts, scalar-result runtime boundary,
   provider mirrors, math operand bindings, and artifact ABI order.
6. Run focused C++ plugin tests and focused lit/script tests.
7. Run generated-bundle selected-boundary dry-run for
   `standalone_reduce_add`.
8. Run real `ssh rvv` generated-bundle evidence for counts
   `0,1,7,16,23,257` through the selected lowering-boundary producer path.
9. Run focused adjacent non-regression for explicit selected-body standalone
   reduction and remaining direct route-entry families that share the owner
   registry boundary.
10. Run authority scan, `git diff --check`, task validation, and
    `check-tianchenrv` or record the exact blocker.

## Out Of Scope

- Segment2 memory, computed-mask segment2 load/store/update,
  compare/select, computed-mask select, widening dot, MAcc, conversion,
  unrelated base-memory cleanup, additional dtype or LMUL clone batches,
  high-level Linalg/Vector/StableHLO frontend work, source-front-door
  construction, one-intrinsic wrapper dialects, selected-body framework
  rewrites, dashboard/report work, prompt-only work, and broad smoke matrices.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and shared guides for
  plugin locality, capability-first design, and compute-boundary review.
- Current journal facts read from `.trellis/workspace/codex/journal-16.md`:
  session 269 completed cmp_select/cmp_select_sle selected-realization
  migration at HEAD lineage, with ssh rvv counts `0/1/7/16/23/257` and
  `check-tianchenrv` 400/400.
- Existing direct route-entry positives for computed-mask segment2 families are
  non-goal non-regressions; this task should not demote or rewrite them.
