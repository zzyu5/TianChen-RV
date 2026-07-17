# Stage2 RVV non-computed MAcc selected-body owner cleanup

## Goal

Move production selected-body realization ownership for non-computed MAcc
pre-realized bodies out of the central `RVVSelectedBodyRealization.cpp` module
into an RVV plugin-local MAcc owner component. The owner covers
`TypedMAccPreRealizedBodyOp` for both `macc_add` and
`scalar_broadcast_macc_add`, while preserving the existing compiler path,
provider route facts, artifact ABI, and generated output behavior.

## What I already know

* The repository is clean on `main` at commit `06ebdeea`, after computed-mask
  MAcc selected-body realization was extracted into an owner-local component.
* There is no active Trellis task, so this task was created from the Hermes
  direction brief before source edits.
* The current central file still owns the non-computed MAcc body-family
  predicate, scalar-broadcast recognition, realization hook, setvl/with_vl
  construction, load/splat/MAcc/store materialization, and erase behavior.
* Existing owner-local extraction pattern is
  `RVVComputedMaskMAccSelectedBodyRealizationOwner`.
* The stable path remains selected `tcrv.exec` RVV variant -> typed
  pre-realized `tcrv_rvv` MAcc body -> RVV plugin-local owner realization ->
  realized typed `setvl`/`with_vl`/load or splat/MAcc/store facts -> existing
  MAcc provider route facts -> `TCRVEmitCLowerableRoute` -> neutral EmitC.

## Requirements

* Add an RVV plugin-local non-computed MAcc selected-body realization owner
  component with header/source and CMake wiring.
* The owner must expose a structural predicate for `TypedMAccPreRealizedBodyOp`
  and a realization hook for the selected-body owner registry.
* The owner must own validation and fail-closed behavior for body family,
  `op_kind`, `memory_form`, scalar-broadcast recognition, runtime `n`/AVL/VL
  binding, `lhs`/`rhs` or `rhs_scalar`/`acc`/`out` roles, SEW, LMUL, policy,
  accumulator/result layout, typed realized IR construction, and pre-realized
  body erase.
* `RVVSelectedBodyRealization.cpp` may retain only neutral shared helpers that
  are intentionally shared by remaining central owners, the owner include,
  registry entry, dispatch glue, and minimal fail-closed branch-helper
  exclusion needed for owner dispatch.
* Preserve computation semantics, dtype semantics, parameter roles, selected
  variant origin, required capabilities, dispatch/fallback behavior, runtime
  `n`/AVL values, provider route facts, target artifact ABI, and existing
  generated artifact behavior.
* Keep common EmitC/export neutral; no RVV semantic choice may move into
  common lowering, route ids, metadata, scripts, artifacts, descriptors, ABI
  strings, source-front-door paths, or exact intrinsic spelling.

## Acceptance Criteria

* [x] New `RVVMAccSelectedBodyRealizationOwner` or equivalent header/source is
  added and wired into `lib/Plugin/RVV/CMakeLists.txt`.
* [x] Central `RVVSelectedBodyRealization.cpp` no longer owns non-computed
  MAcc semantic helpers or materialization logic beyond neutral registry and
  dispatch glue.
* [x] Focused C++ tests prove the MAcc owner predicate/registry ownership,
  `macc_add` and `scalar_broadcast_macc_add` realization through the owner,
  and fail-closed behavior for wrong body, wrong `op_kind`, wrong
  `memory_form`, wrong ABI role, missing/stale runtime `n`, config/policy
  mismatch, stale route-id/metadata authority, direct-route-entry-only
  authority, and mixed realized/pre-realized bodies where existing fixtures
  cover the route path.
* [x] Focused generated-bundle or lit dry-runs for `macc_add` and
  `scalar_broadcast_macc_add` continue to pass as non-regression evidence.
* [x] Completed computed-mask MAcc owner behavior still passes focused checks.
* [x] Bounded authority scan over touched production files shows no legacy-i32,
  descriptor, source-front-door, route-id, direct-route-entry-only,
  exact-intrinsic, artifact-name, script-derived, metadata-derived,
  ABI-string-derived, or common-EmitC semantic authority.
* [x] `git diff --check` passes.
* [x] `check-tianchenrv` passes, or an exact blocker is recorded.
* [x] Worktree is clean after the finish/commit step, unless an exact
  continuation point is recorded.

## Out of Scope

* No new MAcc route coverage, dtype/LMUL clone, direct pre-realized route-entry
  support, source-front-door route, or runtime/correctness/performance claim.
* No computed-mask MAcc, widening MAcc, widening dot/reduce, contraction,
  memory, segment2, conversion, runtime-scalar store/load-store, elementwise,
  standalone reduction, frontend/Linalg, wrapper-only, dashboard/report, or
  broad owner-framework rewrite work.
* No movement of common EmitC or target artifact code into RVV semantic
  selection.
* No `ssh rvv` evidence is required unless this refactor changes emitted
  runtime/artifact behavior or makes a new executable/correctness/performance
  claim.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Prior task read:
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-computed-mask-macc-selected-body-owner-cleanup/prd.md`.
* Owner-local pattern:
  `include/TianChenRV/Plugin/RVV/RVVComputedMaskMAccSelectedBodyRealizationOwner.h`
  and `lib/Plugin/RVV/RVVComputedMaskMAccSelectedBodyRealizationOwner.cpp`.
* Primary implementation surface:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  the new MAcc owner header/source,
  `lib/Plugin/RVV/CMakeLists.txt`, and focused tests in
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Completion Evidence

* Added `RVVMAccSelectedBodyRealizationOwner` as the owner-local component for
  non-computed `TypedMAccPreRealizedBodyOp` realization.
* Rewired the selected-body owner registry so the `MAcc` family uses the
  owner-local predicate and realization hook.
* Removed non-computed MAcc predicate, scalar-broadcast recognition,
  `tcrv_rvv.macc` construction, runtime AVL/VL handling, load/splat/MAcc/store
  materialization, and erase logic from `RVVSelectedBodyRealization.cpp`.
* Preserved realized typed facts consumed by existing provider route planning:
  `setvl`, `with_vl`, `load`, scalar `splat` where applicable, `macc`, `store`,
  `lhs`/`rhs` or `rhs_scalar`/`acc`/`out`/`n` ABI roles, SEW32, LMUL m1, policy,
  accumulator/result layout, and selected variant `requires` metadata.
* Focused C++ validation passed:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2` and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
* Focused lit/generated-bundle validation passed for seven MAcc-related tests:
  pre-realized `macc_add`, scalar-broadcast MAcc generated-bundle,
  scalar-broadcast selected-body artifact, pre-realized computed-mask MAcc,
  runtime-scalar computed-mask MAcc, and direct pre-realized fail-closed checks
  for plain/scalar-broadcast MAcc.
* `git diff --check` passed.
* Full `cmake --build build --target check-tianchenrv -j2` passed 464/464.
* Bounded touched-production authority scan found only existing selected
  variant `requires metadata` diagnostic strings in the central file; no new
  legacy-i32, descriptor, source-front-door, route-id, exact-intrinsic,
  artifact-name, script-derived, ABI-string-derived, direct-route-entry-only,
  or common-EmitC semantic authority was introduced.
* No `.trellis/spec/**` update was required: the existing RVV plugin,
  `tcrv.exec`, EmitC route, testing, and locality specs already define the
  selected-body owner-local boundary implemented here.
* No new runtime, correctness, performance, route, artifact, or executable
  claim was made; `ssh rvv` was intentionally not required.
