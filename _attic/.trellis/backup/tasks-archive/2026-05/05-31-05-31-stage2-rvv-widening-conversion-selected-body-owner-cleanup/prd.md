# Stage2 RVV widening-conversion selected-body owner cleanup

## Goal

Move production ownership for
`TypedWideningConversionPreRealizedBodyOp` selected-body realization out of
`lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` into a dedicated RVV
widening-conversion selected-body realization owner module.

The production path must remain:

```text
selected tcrv.exec RVV variant
  -> typed widening-conversion pre-realized tcrv_rvv body
  -> dedicated RVV widening-conversion selected-body realization owner
  -> realized tcrv_rvv setvl/with_vl/load/widening_convert/store facts
  -> existing provider facts
  -> TCRVEmitCLowerableRoute
  -> neutral EmitC / artifact consumers
```

## Direction Source

- Direction title: `Continue same direction: Stage2 RVV widening-conversion selected-body owner cleanup`.
- Module owner: RVV selected-body realization owner for
  `TypedWideningConversionPreRealizedBodyOp`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- `.trellis/.current-task` was absent at session start, so this task was
  created from the supplied Hermes Direction Brief.

## Current Repository Facts

- Session start HEAD was
  `f02803c0 rvv: move reduction selected-body owner-side`, with a clean
  worktree.
- The archived reduction task moved `TypedReducePreRealizedBodyOp` predicate,
  validation, materialization, and fail-closed diagnostics into
  `RVVReductionSelectedBodyRealizationOwner.{h,cpp}` while leaving the central
  selected-body realization file as registry/dispatch for that family.
- `RVVSelectedBodyRealization.cpp` still defines widening-conversion-specific
  op-kind, memory-form, conversion-relation, signature helpers,
  `isPreRealizedRVVWideningConversionOwnerOp`,
  `realizePreRealizedRVVWideningConversionOwner`,
  `validatePreRealizedRVVSelectedWideningConversionBody`, and the concrete
  `TypedWideningConversionPreRealizedBodyOp` materialization branch.
- The central selected-body registry already has a `widening conversion`
  entry; the entry should remain explicit but point to the dedicated
  widening-conversion owner module.
- Existing widening-conversion route/provider/artifact tests already validate
  provider-owned route facts, conversion relation, runtime ABI order, statement
  plan steps, and artifact mirrors for `widen_i32_to_i64` and
  `sign_extend_widen_vf2`.

## Requirements

1. Add `RVVWideningConversionSelectedBodyRealizationOwner.{h,cpp}` and compile
   it into `TianChenRVRVVPlugin`.
2. Move the `TypedWideningConversionPreRealizedBodyOp` owner predicate,
   validation, fail-closed diagnostics, and materialization out of
   `RVVSelectedBodyRealization.cpp`.
3. Keep the central owner registry / dispatch mechanics, but remove
   widening-conversion-specific op-kind, memory-form, relation, SEW/LMUL,
   dtype, or production materialization branches from the central file.
4. Preserve accepted widening-conversion facts:
   `op_kind = "widen_i32_to_i64"` with source SEW32 LMUL m1, destination
   SEW64 LMUL m2, conversion relation `signed-i32m1-to-i64m2`; and
   `op_kind = "sign_extend_widen_vf2"` with source SEW16 LMUL mf2,
   destination SEW32 LMUL m1, conversion relation `signed-i16mf2-to-i32m1`;
   both with `memory_form = "unit-stride-conversion"` and tail/mask agnostic
   policy.
5. Preserve runtime ABI roles and selected-body materialization order:
   `lhs`, `out`, `n` -> `setvl` -> `with_vl` -> lhs load ->
   `tcrv_rvv.widening_convert` -> store.
6. Preserve fail-closed diagnostics for null/wrong-family owner calls, wrong
   typed conversion facts, mixed realized/pre-realized body, missing selected
   variant `requires`, and missing/invalid runtime ABI values.
7. Preserve provider route consumption and artifact/materialization behavior
   for existing widening-conversion selected-boundary consumers. No runtime,
   ABI, route, common EmitC, or target artifact semantics should change.
8. Keep neighboring reduction, standalone reduction, MAcc, contraction,
   memory, segment2, and runtime-scalar owner paths non-regressed.

## Acceptance Criteria

- [x] `RVVWideningConversionSelectedBodyRealizationOwner.h` and `.cpp` exist
      and are compiled into the RVV plugin library.
- [x] `RVVSelectedBodyRealization.cpp` no longer defines
      widening-conversion-specific predicates, validation, signature/relation
      helpers, or concrete `TypedWideningConversionPreRealizedBodyOp`
      realization/materialization branches.
- [x] The central selected-body owner registry still contains exactly one
      explicit `widening conversion` entry and it uses the dedicated
      widening-conversion owner predicate and realization hook.
- [x] C++ tests prove owner registry count/order/hooks remain valid and the
      widening-conversion owner-local hook is distinct.
- [x] C++ tests prove the widening-conversion owner claims
      `TypedWideningConversionPreRealizedBodyOp`, rejects non-conversion
      bodies, fail-closes invalid typed facts, and consumes a valid
      pre-realized body into exactly one realized setvl/with_vl with
      load/widening_convert/store facts.
- [x] Focused route/provider coverage still proves realized widening-conversion
      facts feed provider route construction for both supported conversion
      operations.
- [x] Focused lit/generated-bundle or artifact/materialization evidence for
      widening-conversion selected-boundary artifacts still shows provider
      route/artifact mirrors remain intact.
- [x] Existing neighboring owner tests still pass.
- [x] Bounded authority scans over touched production files show no new
      descriptor-, source-front-door-, artifact-name-, route-id-,
      exact-intrinsic-, legacy-i32-, ABI-string-, metadata-, script-, or
      common-EmitC-derived widening-conversion authority, and no
      widening-conversion semantic residue remains in the central file except
      the registry entry and include.
- [x] `git diff --check` passes.
- [x] `cmake --build build --target check-tianchenrv -j2` passes, or the exact
      blocker is recorded.
- [x] Task status/context/journal are truthful; task is finished/archived and a
      coherent commit is created if acceptance passes.

## Out of Scope

- No new widening conversion variants, dtype/LMUL clone batches, or new
  conversion coverage.
- No arithmetic, reduction, contraction, MAcc, memory, segment2, scalar,
  offload, IME, TensorExt, frontend/Linalg, or high-level source lowering work.
- No one-intrinsic wrappers, source-front-door positive routes, descriptor
  routes, dashboard/report work, broad smoke matrices, or evidence-only task.
- No rewrite of already extracted owners except minimal include/build/test
  integration required by this owner extraction.
- No hardware runtime/correctness/performance claim; this task is owner-local
  refactoring and should not require `ssh rvv` evidence.

## Validation Plan

1. Run the focused RVV plugin C++ test binary for selected-body owner registry,
   widening-conversion owner classification/materialization, provider
   consumption, and neighboring owner non-regression.
2. Run focused widening-conversion artifact/materialization tests if available,
   or the exact lit/C++ target artifact coverage currently used by
   check-tianchenrv.
3. Run bounded authority scans on touched production files and the central
   selected-body realization file.
4. Run `git diff --check`.
5. Run `cmake --build build --target check-tianchenrv -j2` when focused checks
   are green.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Prior pattern read:
  `.trellis/tasks/archive/2026-05/05-31-stage2-rvv-reduction-selected-body-owner-cleanup/prd.md`.
- Neighboring code read:
  `RVVSelectedBodyRealization.cpp`,
  `RVVReductionSelectedBodyRealizationOwner.{h,cpp}`,
  `lib/Plugin/RVV/CMakeLists.txt`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.
