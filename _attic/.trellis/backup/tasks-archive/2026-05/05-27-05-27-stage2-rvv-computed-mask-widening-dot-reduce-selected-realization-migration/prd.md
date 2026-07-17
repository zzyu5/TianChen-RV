# Stage2 RVV Computed-Mask Widening Dot-Reduce Selected-Body Realization Migration

## Goal

Move the `computed_masked_widening_dot_reduce_add` generated artifact path
behind the public RVV selected lowering-boundary realization producer. The
production executable path must be:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv computed_masked_widening_dot_reduce_add body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv setvl / with_vl / compare i32 loads /
     i16 dot lhs/rhs loads / compare mask / masked widening dot-reduce /
     i32 scalar store body
  -> existing contraction route-family facts
  -> route materialization facts
  -> math operand-binding facts for compare, dot, accumulator, out, runtime n
  -> route-control provider plan
  -> direct contraction provider plan and statement owner
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence
```

Direct pre-realized route-entry support for
`computed_masked_widening_dot_reduce_add` must be demoted or deleted. A
generated artifact for this operation must not be accepted from a direct
route-entry shortcut, pre-realized fixture authority, route id, artifact name,
script option, ABI string, exact intrinsic spelling, common EmitC behavior,
descriptor residue, source-front-door metadata, or legacy i32 helper authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV computed-mask widening dot-reduce
  selected-body realization migration`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `57ef6746 rvv: demote strided widening dot route entry`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
- Serial worker constraint: one Codex worker, no subagents or parallel agent
  workflows.

## Current Repository Facts

- The archived
  `05-27-05-27-stage2-rvv-strided-input-widening-dot-reduce-selected-realization-migration`
  task demoted direct pre-realized route-entry support for
  `strided_input_widening_dot_reduce_add` while preserving selected-boundary
  producer evidence and real `ssh rvv` correctness.
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` keeps the `contraction`
  selected-body realization owner positive for widening MAcc, plain widening
  dot-reduce, strided-input widening dot-reduce, computed-mask widening
  dot-reduce, and computed-mask strided-input widening dot-reduce.
- The current contraction direct route-entry predicate already rejects
  `TypedWideningMAccPreRealizedBodyOp`,
  `TypedWideningDotReducePreRealizedBodyOp`, and
  `TypedStridedInputWideningDotReducePreRealizedBodyOp`, but still accepts
  `TypedComputedMaskWideningDotReducePreRealizedBodyOp` and
  `TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp`.
- `scripts/rvv_generated_bundle_abi_e2e.py` still reports both
  `computed_masked_widening_dot_reduce_add` and
  `computed_masked_strided_input_widening_dot_reduce_add` as
  `supports_direct_pre_realized_route_entry`.
- Existing selected-boundary fixture
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir`
  already represents the desired public producer path and checks typed realized
  facts: compare-produced mask, i16mf2 dot operands, i32 scalar seed/result,
  widening dot relation, mask metadata, route operand binding, route-control,
  and target ABI mirrors.
- Existing planning/provider code already exposes computed-mask widening
  dot-reduce contraction family facts, route-control provider plan, math
  operand-binding facts, direct contraction provider plan, statement plan,
  target ABI mirrors, and generated-bundle evidence fields. This task should
  reuse those paths rather than adding a new route table or common EmitC
  semantic branch.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes are allowed only for evidence tooling and script
   guardrails.
2. Keep the selected-boundary producer path positive for
   `computed_masked_widening_dot_reduce_add`. It must realize the typed
   pre-realized body into `setvl`, `with_vl`, i32 compare lhs/rhs loads, i16
   dot lhs/rhs loads, compare mask, `masked_widening_dot_reduce`, and i32
   scalar store before provider route facts are collected.
3. Demote or delete direct pre-realized route-entry support for
   `computed_masked_widening_dot_reduce_add`. Direct shortcut requests must
   fail closed with targeted route-entry diagnostics before bundle generation.
4. Preserve typed facts for operation kind, computed mask producer and mask use,
   inactive/pass-through behavior, i16 lhs/rhs dot source binding, i32
   accumulation/result binding, widening dtype relation, dot-reduction shape,
   source/result SEW and LMUL relation, memory roles, runtime `n`/AVL/VL
   values, setvl placement, loop relation, selected capability, provider route
   facts, and artifact ABI order.
5. Do not start or demote
   `computed_masked_strided_input_widening_dot_reduce_add` in this round unless
   a minimal predicate split is required to isolate the non-strided
   `computed_masked_widening_dot_reduce_add` route-entry demotion.
6. Unsupported or inconsistent selected-boundary input must fail closed before
   provider/common route construction where current hooks expose it: missing
   runtime_param, missing mem_window/runtime ABI value, missing or stale mask
   binding, wrong source dtype, wrong accumulator/result dtype, wrong widening
   relation, wrong reduction shape, wrong operand binding, dtype/config/policy
   mismatch, wrong AVL/VL relation, wrong setvl placement, missing capability,
   stale route id or mirror metadata, direct-route-entry-only authority,
   artifact-name/script-derived authority, exact-intrinsic-as-authority, and
   common-EmitC semantic invention.
7. Do not start widening conversion, segment2, compare/select, unrelated
   computed-mask cleanup, standalone reduction cleanup, new dtype/LMUL clone
   batches, high-level Linalg/frontend lowering, one-intrinsic wrapper
   dialects, selected-body framework rewrites, dashboard/report work, broad
   smoke matrices, or proof-only tests for completed adjacent routes as the
   main achievement.

## Acceptance Criteria

- [x] Production code no longer treats
      `computed_masked_widening_dot_reduce_add` as direct pre-realized
      route-entry eligible, while the `contraction` selected-body realization
      owner still realizes it through the public selected lowering-boundary
      producer path.
- [x] C++ tests prove a typed pre-realized
      `computed_masked_widening_dot_reduce_add` body belongs to the
      `contraction` realization owner, is not a direct route-entry consumer,
      and fails closed when `realizePreRealizedRVVRouteEntrySelectedBody` is
      used as a direct route-entry shortcut.
- [x] C++ or lit tests prove the selected-boundary
      `computed_masked_widening_dot_reduce_add` path consumes realized typed
      body facts, including i32 compare loads, compare mask, i16 dot source
      loads, masked widening dot-reduce, scalar i32 seed/result layout,
      widening relation, route-control provider plan, math operand-binding
      facts, direct contraction provider plan, ABI order, and
      provider-supported mirrors.
- [x] The generated-bundle script rejects
      `--direct-pre-realized-route-entry --op-kind
      computed_masked_widening_dot_reduce_add` before route-entry
      materialization or bundle generation.
- [x] Generated-bundle dry-run for
      `computed_masked_widening_dot_reduce_add` passes through
      `--tcrv-materialize-selected-lowering-boundaries`, records
      `route_entry_realization: false`, records selected-body producer
      evidence, and records no direct route-entry materializer.
- [x] Real `ssh rvv` generated-bundle execution covers counts `0`, `1`,
      exact-vector, tail, and stress cases with at least two signed i16 input
      patterns, computed-mask patterns, inactive-lane skip checks, and expected
      i32 scalar reduction results.
- [x] Focused non-regression covers adjacent selected-body realization paths
      risked by this demotion, including `widening_macc_add`,
      `widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
      `computed_masked_strided_input_widening_dot_reduce_add`,
      computed-mask MAcc, runtime-scalar MAcc, scalar_broadcast_add,
      runtime_i32_splat_store, computed-mask select, and at least one remaining
      direct route-entry fixture as practical focused filters or exact blocker.
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
2. Narrow the RVV selected-body realization owner registry so the `contraction`
   owner remains selected-boundary capable, but
   `computed_masked_widening_dot_reduce_add` is not direct route-entry eligible.
3. Update generated-bundle tooling and focused script coverage so
   `--direct-pre-realized-route-entry --op-kind
   computed_masked_widening_dot_reduce_add` fails closed.
4. Strengthen selected-boundary lit/script coverage for
   `pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir`
   so evidence explicitly checks `route_entry_realization: false`.
5. Run focused C++ plugin tests and focused lit/script tests.
6. Run generated-bundle selected-boundary dry-run for
   `computed_masked_widening_dot_reduce_add`.
7. Run real `ssh rvv` generated-bundle evidence for counts `0,1,16,23,257`
   using the selected lowering-boundary producer path and computed-mask
   widening dot harness.
8. Run focused non-regression for selected-body producer paths listed in the
   acceptance criteria.
9. Run authority scan, `git diff --check`, task validation, and
   `check-tianchenrv`.

## Out Of Scope

- `computed_masked_strided_input_widening_dot_reduce_add` migration,
  `widen_i16_to_i32`, segment2 follow-ons, compare/select, unrelated
  computed-mask cleanup, standalone reduction cleanup, new dtype/LMUL clone
  sets, high-level Linalg/Vector/StableHLO frontend work, source-front-door
  construction, descriptor-driven compute, direct C/source-export paths, and
  legacy `RVVI32M1` / `rvv-i32m1` compatibility.
- Dashboard/report-only/prompt-only work as the main achievement.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and shared guides under
  `.trellis/spec/guides/`.
- Relevant archived task read:
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-strided-input-widening-dot-reduce-selected-realization-migration/`.
- Primary production files to inspect or modify:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and focused target/script tests.

## Open Questions

- None blocking at task start. The user brief, specs, and current repository
  evidence agree on the module owner and non-goals.

## Implementation Result

- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` keeps
  `TypedComputedMaskWideningDotReducePreRealizedBodyOp` under the
  `contraction` selected-body realization owner, but rejects it from the
  contraction direct route-entry predicate. The strided computed-mask widening
  dot-reduce body remains route-entry eligible because that migration is outside
  this task.
- `scripts/rvv_generated_bundle_abi_e2e.py` removes
  `computed_masked_widening_dot_reduce_add` from
  `supports_direct_pre_realized_route_entry`, keeps the selected-boundary
  producer path positive, and narrows the direct route-entry error wording to
  the remaining allowed fixtures.
- `test/Plugin/RVVExtensionPluginTest.cpp` now proves the non-strided computed
  mask widening dot-reduce body is a `contraction` selected-body consumer, is
  rejected by direct route-entry realization, realizes into typed
  `setvl`/`with_vl`/loads/compare/`masked_widening_dot_reduce`/store IR, and
  carries computed-mask dot-reduction facts into route analysis, math binding,
  materialization, and provider planning.
- The focused generated-bundle dry-run lit test now requires
  `route_entry_realization: false` and the selected lowering-boundary
  materializer for `computed_masked_widening_dot_reduce_add`.
- A new fail-closed lit test rejects
  `--direct-pre-realized-route-entry --op-kind
  computed_masked_widening_dot_reduce_add` before route-entry materialization
  or bundle success.

## Validation Result

1. `python3 ./.trellis/scripts/task.py validate .../implement.jsonl` and
   `.../check.jsonl`: passed.
2. `cmake --build build --target tianchenrv-rvv-extension-plugin-test tcrv-opt
   tcrv-translate -j2`: passed.
3. `./build/bin/tianchenrv-rvv-extension-plugin-test`: passed.
4. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`: passed.
5. `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`: passed.
6. Focused build-tree lit filter for the selected-boundary dry-run and new
   direct route-entry fail-closed script tests: passed `2/2`. A source-tree lit
   invocation was rejected because it lacked the generated build-tree site
   config; the build-tree invocation superseded it.
7. Selected-boundary dry-run for
   `computed_masked_widening_dot_reduce_add` with counts
   `0,1,16,23,257`: passed; evidence records
   `materializer: tcrv-materialize-selected-lowering-boundaries` and
   `route_entry_realization: false`.
8. Direct route-entry request for
   `computed_masked_widening_dot_reduce_add` with counts `0,1,16,23,257`:
   fail-closed before bundle generation with the bounded direct route-entry
   diagnostic.
9. Adjacent contraction selected-boundary dry-run for `widening_macc_add`,
   `widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
   `computed_masked_widening_dot_reduce_add`, and
   `computed_masked_strided_input_widening_dot_reduce_add`: passed.
10. Remaining direct route-entry non-regression for
    `computed_masked_strided_input_widening_dot_reduce_add`: passed and records
    `route_entry_realization: true`.
11. Real `ssh rvv` generated-bundle run for
    `computed_masked_widening_dot_reduce_add` with counts `0,1,16,23,257`:
    passed, including computed mask, signed horizontal dot, seed addition,
    inactive-lane skip, scalar i32 output, and tail-preserved checks.
12. Bounded authority scan: production allowlist no longer contains the
    non-strided computed-mask widening dot-reduce direct route-entry case;
    touched-file diff contains descriptor/direct-C/source-export only as
    negative test checks or script-level denial text.
13. `git diff --check`: passed.
14. `cmake --build build --target check-tianchenrv -j2`: passed `396/396`.
