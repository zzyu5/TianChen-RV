# Stage2 RVV reduction selected-body owner cleanup

## Goal

Move production ownership for `TypedReducePreRealizedBodyOp` selected-body
realization out of `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` into a
dedicated RVV reduction selected-body realization owner module.

The production path must remain:

```text
selected tcrv.exec RVV variant
  -> typed reduction pre-realized tcrv_rvv body
  -> dedicated RVV reduction selected-body realization owner
  -> realized tcrv_rvv setvl/with_vl/load/load/reduce/store facts
  -> existing provider facts
  -> TCRVEmitCLowerableRoute
  -> neutral EmitC / artifact consumers
```

## Direction Source

- Direction title: `Switch: Stage2 RVV reduction selected-body owner-side cleanup`
- Module owner: RVV selected-body realization owner for
  `TypedReducePreRealizedBodyOp`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- `.trellis/.current-task` was absent at session start, so this task was
  created from the supplied Hermes Direction Brief.

## Current Repository Facts

- Session start HEAD was
  `2333cca7 rvv: move runtime-scalar memory selected-body owner-side`, with a
  clean worktree.
- The archived runtime-scalar memory task already moved its owner predicates
  and realization branches into
  `RVVRuntimeScalarMemorySelectedBodyRealizationOwner`.
- `RVVSelectedBodyRealization.cpp` still defines reduction-specific op-kind,
  memory-form, accumulator-role, accumulator-layout, result-layout predicates,
  `isPreRealizedRVVReductionOwnerOp`, `realizePreRealizedRVVReductionOwner`,
  `validatePreRealizedRVVSelectedReduceBody`, and concrete
  `TypedReducePreRealizedBodyOp` materialization.
- Existing `RVVStandaloneReductionSelectedBodyRealizationOwner` is a
  neighboring convention, not the right semantic home for this task: standalone
  reduction owns scalar accumulator/result channel behavior, while
  `TypedReducePreRealizedBodyOp` owns vector RHS seed/load, `tcrv_rvv.reduce`,
  and chunk-base output store behavior.
- The central owner registry already has a `reduction` entry; the entry should
  remain explicit but point to the dedicated reduction owner module.

## Requirements

1. Add `RVVReductionSelectedBodyRealizationOwner.{h,cpp}` and compile it into
   `TianChenRVRVVPlugin`.
2. Move the `TypedReducePreRealizedBodyOp` owner predicate, validation, and
   materialization out of `RVVSelectedBodyRealization.cpp`.
3. Keep the central owner registry / dispatch mechanics, but remove
   reduction-specific validation or production materialization branches from
   the central file.
4. Preserve accepted reduction facts:
   `op_kind = "reduce_add"`, `memory_form = "vector-rhs-load"`,
   `accumulator_role = "rhs-input-buffer"`,
   `accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk"`,
   `result_layout = "store-reduction-lane0-to-output-chunk-base"`,
   SEW32, LMUL m1, and tail/mask agnostic policy.
5. Preserve runtime ABI roles and selected-body materialization order:
   `lhs`, `rhs`, `out`, `n` -> `setvl` -> `with_vl` -> lhs load -> rhs load
   -> `tcrv_rvv.reduce` -> store.
6. Preserve fail-closed diagnostics for null/wrong-family owner calls, wrong
   typed reduction facts, mixed realized/pre-realized body, missing selected
   variant `requires`, and missing/invalid runtime ABI values.
7. Preserve provider route consumption and artifact mirror behavior for
   existing generated-bundle/lit reduction consumers. No runtime, ABI, route,
   common EmitC, or target artifact semantics should change.
8. Keep neighboring standalone reduction and runtime-scalar memory owner paths
   non-regressed.

## Acceptance Criteria

- [x] `RVVReductionSelectedBodyRealizationOwner.h` and `.cpp` exist and are
      compiled into the RVV plugin library.
- [x] `RVVSelectedBodyRealization.cpp` no longer defines reduction-specific
      predicates, validation, or concrete `TypedReducePreRealizedBodyOp`
      realization/materialization branches.
- [x] The central selected-body owner registry still contains exactly one
      explicit `reduction` entry and it uses the dedicated reduction owner
      predicate and realization hook.
- [x] C++ tests prove owner registry count/order/hooks remain valid and
      owner-local hooks are distinct.
- [x] C++ tests prove the reduction owner claims `TypedReducePreRealizedBodyOp`,
      rejects non-reduction bodies, fail-closes invalid typed facts, and
      consumes a valid pre-realized body into exactly one realized setvl/with_vl
      with load/load/reduce/store facts.
- [x] Focused route/provider coverage still proves realized reduction facts
      feed provider route construction.
- [x] Focused lit/generated-bundle evidence for reduction selected-boundary
      artifacts still shows the pre-realized body is consumed and provider
      route/artifact mirrors remain intact.
- [x] Existing standalone reduction and runtime-scalar memory owner tests still
      pass.
- [x] Bounded authority scans over touched production files show no new
      descriptor-, source-front-door-, artifact-name-, route-id-,
      exact-intrinsic-, legacy-i32-, ABI-string-, metadata-, script-, or
      common-EmitC-derived reduction authority.
- [x] `git diff --check` passes.
- [x] `cmake --build build --target check-tianchenrv -j2` passes, or the exact
      blocker is recorded.
- [x] Task status/context/journal are truthful; task is finished/archived and a
      coherent commit is created if acceptance passes.

## Out of Scope

- No widening conversion cleanup.
- No standalone reduction expansion or computed-mask standalone reduction
  changes.
- No new reduction op coverage, dtype/LMUL clone batches, compare/select,
  conversion, MAcc, contraction, memory, or segment2 work.
- No high-level Linalg/frontend lowering.
- No provider route rewrite, common EmitC semantic choices, target artifact ABI
  changes, dashboard/report work, broad smoke matrices, or evidence-only task.
- No hardware runtime/correctness/performance claim unless executable behavior
  changes; this task is owner-local refactoring and should not need `ssh rvv`
  evidence.

## Validation Plan

1. Run the focused RVV plugin C++ test target or binary for selected-body owner
   registry, reduction owner classification/materialization, provider
   consumption, standalone reduction non-regression, and runtime-scalar memory
   non-regression.
2. Run focused reduction lit/generated-bundle tests:
   `test/Target/RVV/pre-realized-selected-body-artifact-reduce-add.mlir` if
   present, or the exact reduction selected-boundary artifact equivalent.
3. Run bounded authority scans on touched production files.
4. Run `git diff --check`.
5. Run `cmake --build build --target check-tianchenrv -j2` when focused checks
   are green.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Prior pattern read:
  `.trellis/tasks/archive/2026-05/05-31-05-31-stage2-rvv-runtime-scalar-memory-selected-body-owner-cleanup/prd.md`.
- Neighboring code read:
  `RVVSelectedBodyRealization.cpp`,
  `RVVStandaloneReductionSelectedBodyRealizationOwner.{h,cpp}`,
  `RVVRuntimeScalarMemorySelectedBodyRealizationOwner.{h,cpp}`, and
  `RVVExtensionPluginTest.cpp`.
