# Stage2 RVV runtime-scalar memory selected-body owner cleanup

## Goal

Move runtime-scalar memory selected-body realization ownership out of
`lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` into a dedicated RVV
selected-body owner module. The central selected-body file should keep only
registry / dispatch / shared neutral mechanics for these families while the
new owner recognizes and realizes:

- `runtime_i32_splat_store`
- runtime-scalar computed-mask store
- runtime-scalar computed-mask load-store

The production behavior must remain the existing selected-boundary path:
selected typed pre-realized `tcrv_rvv` body -> RVV runtime-scalar memory owner
-> realized typed `tcrv_rvv` setvl/with_vl/load/splat/compare/masked memory
facts -> existing provider facts -> `TCRVEmitCLowerableRoute` -> neutral EmitC
-> target artifact consumers.

## Direction Source

- Direction title: `Switch: Stage2 RVV runtime-scalar memory selected-body owner-side cleanup`
- Module owner: RVV plugin-local selected-body realization owner for
  runtime-scalar memory bodies.
- Repository root: `/home/kingdom/phdworks/TianchenRV`
- `.trellis/.current-task` was absent at session start, so this task was
  created from the supplied Hermes Direction Brief.

## Current Repository Facts

- Current HEAD is `eeb22ba8 rvv: move segment2 memory selected-body owner-side`;
  the worktree was clean before task creation.
- The archived task
  `.trellis/tasks/archive/2026-05/05-31-stage2-rvv-segment2-memory-selected-body-owner-cleanup`
  moved segment2 selected-body realization into
  `RVVSegment2MemorySelectedBodyRealizationOwner`.
- `RVVSelectedBodyRealization.cpp` still defines runtime-scalar memory owner
  predicates, owner wrapper hooks, and concrete realization branches for
  splat-store, computed-mask store, and computed-mask load-store.
- Runtime-scalar computed-mask validation already lives in
  `RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners`; this task must preserve
  that validation behavior and reuse it from the owner.
- Base memory, computed-mask memory, segment2 memory, MAcc, contraction, and
  standalone reduction owner modules already provide the local extraction
  pattern for this cleanup.

## Requirements

1. Add a dedicated
   `RVVRuntimeScalarMemorySelectedBodyRealizationOwner.{h,cpp}` module.
2. Move the three runtime-scalar memory selected-body predicates out of
   `RVVSelectedBodyRealization.cpp` into the new owner module.
3. Move concrete realization for splat-store, computed-mask store, and
   computed-mask load-store out of `RVVSelectedBodyRealization.cpp`.
4. Update the selected-body owner registry so the three runtime-scalar memory
   entries use the new owner module predicates and realization hooks.
5. Preserve runtime ABI role binding and order:
   `rhs_scalar,out,n` for splat-store, and `lhs,rhs_scalar,src,dst,n` for
   computed-mask store/load-store.
6. Preserve runtime `n`/AVL/VL derivation through `RVVRuntimeAVLVLControl`.
7. Preserve SEW, LMUL, policy, scalar-source role, lhs/source/destination
   roles, predicate kind, mask construction/use, inactive-lane behavior,
   store/load-store semantics, provider facts, and artifact ABI order.
8. Fail closed for wrong op kind, memory form, runtime ABI role, missing
   runtime n/AVL binding, wrong policy/config, stale mask or scalar binding,
   mixed realized/pre-realized body, missing selected variant `requires`,
   ambiguous owner match, common EmitC semantic invention, or route/artifact/
   name-derived authority.
9. Do not change runtime-scalar semantics, provider route planning, common
   EmitC semantics, target artifact validation, route IDs, generated bundle ABI,
   or unrelated selected-body owner families except for include/CMake fallout
   caused by this owner split.

## Acceptance Criteria

- [ ] New runtime-scalar memory selected-body owner header/source exist and are
      compiled into the RVV plugin library.
- [ ] `RVVSelectedBodyRealization.cpp` no longer defines runtime-scalar memory
      predicates, owner wrapper hooks, or concrete runtime-scalar memory
      production realization branches.
- [ ] The central owner registry still contains the three explicit
      runtime-scalar memory entries, and each points at the new owner module.
- [ ] Owner registry uniqueness and distinct owner hooks remain covered by
      focused C++ tests.
- [ ] Focused C++ coverage proves the new owner claims only its three
      runtime-scalar memory pre-realized body classes and rejects cross-family
      bodies.
- [ ] Focused C++ coverage proves the selected-boundary producer still consumes
      representative splat-store and computed-mask store/load-store
      pre-realized bodies into realized typed `tcrv_rvv` facts.
- [ ] Existing generated-bundle or lit dry-runs for affected runtime-scalar
      paths still show selected-boundary materialization feeding provider facts.
- [ ] Previously extracted owner families, especially segment2, computed-mask
      memory, and base memory, have focused non-regression coverage.
- [ ] Bounded scans over touched files show no new name-, route-id-, metadata-,
      descriptor-, ABI-string-, script-, artifact-, common-EmitC-,
      source-front-door-, or legacy-i32-derived realization authority.
- [ ] `git diff --check` passes.
- [ ] `cmake --build build --target check-tianchenrv -j2` passes, or the exact
      blocker is recorded.
- [ ] The task is finished/archived and the round ends with one coherent commit,
      unless an exact blocker prevents completion.

## Out of Scope

- No runtime-scalar semantic changes.
- No new scalar ops or route coverage.
- No dtype/LMUL clone batches.
- No provider route planning rewrite.
- No common EmitC semantic changes.
- No segment2/base/computed-mask memory/contraction/reduction coverage
  expansion.
- No source-front-door positive route.
- No broad smoke/report/evidence-only round.
- No hardware runtime/correctness/performance claim unless executable behavior
  changes beyond owner-local refactoring.

## Validation Plan

1. Run focused RVV plugin C++ tests covering selected-body owner registry,
   runtime-scalar memory owner classification/negative cases, selected-boundary
   materialization, and non-regression for already extracted owner families.
2. Run representative generated-bundle or lit dry-runs for runtime-scalar
   splat-store and computed-mask memory paths if available.
3. Run bounded authority scans over touched owner / central / test files.
4. Run `git diff --check`.
5. Run `cmake --build build --target check-tianchenrv -j2` if focused checks
   pass.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous task reference:
  `.trellis/tasks/archive/2026-05/05-31-stage2-rvv-segment2-memory-selected-body-owner-cleanup/prd.md`.
- Runtime-scalar computed-mask validation remains in
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`.
- The new owner should follow the local extraction style used by
  `RVVBaseMemoryMovementSelectedBodyRealizationOwner` and
  `RVVSegment2MemorySelectedBodyRealizationOwner`.
