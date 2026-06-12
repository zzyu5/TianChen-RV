# Stage2 RVV contraction selected-body owner cleanup

## Goal

Move production selected-body realization ownership for the active RVV
contraction family out of the central `RVVSelectedBodyRealization.cpp` module
into an RVV plugin-local contraction owner component. The owner covers
`widening_macc_add`, `widening_dot_reduce_add`,
`strided_input_widening_dot_reduce_add`,
`computed_masked_widening_dot_reduce_add`, and
`computed_masked_strided_input_widening_dot_reduce_add`, while preserving the
existing selected-boundary -> realized typed `tcrv_rvv` -> provider route facts
-> `TCRVEmitCLowerableRoute` path.

## What I already know

* The repository is clean on `main` at commit `7e5e1f37`, after
  non-computed MAcc selected-body realization was extracted into an owner-local
  component.
* There was no active Trellis task, so this task was created from the Hermes
  direction brief before source edits.
* The central selected-body file still owns the contraction body predicate,
  `RVVSelectedBodyContractionRealizationPlan`, widening MAcc/dot-reduction
  realization plan builders, contraction-specific realized compute builders,
  validation dispatch, materialization, and pre-realized body erasure.
* Existing owner-local extraction patterns are
  `RVVMAccSelectedBodyRealizationOwner` and
  `RVVComputedMaskMAccSelectedBodyRealizationOwner`.
* The existing contraction provider path and route-family plan owner already
  define provider-side facts for widening MAcc, widening dot reduction,
  strided-input widening dot reduction, computed-mask widening dot reduction,
  and computed-mask strided-input widening dot reduction.

## Requirements

* Add an RVV plugin-local contraction selected-body realization owner component
  with header/source and CMake wiring.
* The owner must expose a structural predicate for the five active contraction
  pre-realized body ops and a realization hook for the selected-body owner
  registry.
* The owner must own contraction validation dispatch, plan building,
  fail-closed diagnostics, `setvl`/`with_vl` creation, source/stride loads,
  compare/mask creation, accumulator/seed handling, widening MAcc or widening
  dot-reduction compute materialization, stores, and pre-realized body erasure.
* `RVVSelectedBodyRealization.cpp` may retain only neutral shared helpers that
  are genuinely still shared, plus the contraction owner include and registry
  entry. It must not keep contraction family predicate, plan structs/builders,
  contraction realization implementation, or family-specific dispatch logic.
* Preserve dtype/SEW/LMUL relation, policy, accumulator/result layout,
  mask role/source/form, predicate, stride ABI roles, runtime `n`/AVL/VL facts,
  selected variant `requires`, ABI order, provider materialization facts,
  route-control consumption, target artifact mirrors, generated-bundle
  evidence shape, and fail-closed behavior.
* Keep common EmitC/export neutral; no contraction semantic choice may move
  into common lowering, route ids, metadata, scripts, artifacts, descriptors,
  ABI strings, source-front-door paths, exact intrinsic spelling, or legacy i32
  helper residue.

## Acceptance Criteria

* [x] New `RVVContractionSelectedBodyRealizationOwner` or equivalent
  header/source is added and wired into `lib/Plugin/RVV/CMakeLists.txt`.
* [x] The selected-body owner registry uses the owner-local contraction
  predicate and realization hook.
* [x] Central `RVVSelectedBodyRealization.cpp` no longer owns contraction
  semantic helpers or materialization logic beyond neutral registry and shared
  helper use.
* [x] Focused C++ tests prove contraction owner predicate/registry ownership,
  exact unique-owner classification for all five active contraction bodies,
  realized typed facts for representative plain, strided, computed-mask, and
  computed-mask-strided paths, provider-plan consumption, and fail-closed
  behavior for outside-family bodies and invalid/stale typed facts.
* [x] Focused generated-bundle/lit dry-runs for representative contraction
  paths continue to pass without behavior drift, including direct pre-realized
  route-entry fail-closed probes for demoted variants.
* [x] Existing route-supported/executable claims for the contraction family are
  preserved; no new contraction op, dtype/LMUL clone, intrinsic, source-front
  door, high-level frontend, runtime, correctness, or performance claim is
  introduced.
* [x] Bounded authority scan over touched production/test files shows no
  central ad hoc, name-derived, metadata-derived, descriptor-derived,
  ABI-string-derived, script-derived, artifact-name-derived,
  common-EmitC-derived, source-front-door-derived, route-id-derived,
  exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only,
  or legacy-i32-derived route authority.
* [x] `git diff --check` passes.
* [x] `check-tianchenrv` passes, or an exact blocker is recorded.
* [ ] Worktree is clean after the finish/commit step, unless an exact
  continuation point is recorded.

## Out of Scope

* No new contraction operations, dtype/LMUL clones, intrinsics, source-front
  door positive routes, high-level Linalg/frontend lowering, widening
  conversion cleanup, segment2 cleanup, runtime-scalar cleanup, dashboard or
  report work, broad smoke matrices, or evidence-only task.
* No movement of common EmitC, target artifact export, or provider route
  construction authority into the selected-body realization owner.
* No direct pre-realized route-entry shortcut may be re-enabled for MAcc or
  widening dot variants already demoted.
* No `ssh rvv` evidence is required unless this refactor changes emitted
  runtime/artifact behavior or makes a new runtime/correctness/performance
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
  `.trellis/tasks/archive/2026-05/05-30-stage2-rvv-noncomputed-macc-owner-cleanup/prd.md`.
* Owner-local patterns:
  `include/TianChenRV/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVComputedMaskMAccSelectedBodyRealizationOwner.h`,
  and `lib/Plugin/RVV/RVVComputedMaskMAccSelectedBodyRealizationOwner.cpp`.
* Primary implementation surface:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  new contraction owner header/source,
  `lib/Plugin/RVV/CMakeLists.txt`, and focused coverage in
  `test/Plugin/RVVExtensionPluginTest.cpp`.
* Related provider-side owner:
  `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`
  and `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`.

## Completion Evidence

* Added `RVVContractionSelectedBodyRealizationOwner` as the owner-local
  selected-body realization component for the contraction family.
* Rewired the selected-body owner registry so the `contraction` family uses the
  owner-local predicate and realization hook.
* Removed contraction predicate, realization plan structs/builders,
  `tcrv_rvv.widening_macc` / `tcrv_rvv.widening_dot_reduce` /
  `tcrv_rvv.masked_widening_dot_reduce` construction, contraction validation
  dispatch, source/stride/compare/mask materialization, store materialization,
  and pre-realized body erasure from `RVVSelectedBodyRealization.cpp`.
* Preserved realized typed facts consumed by existing provider planning:
  `setvl`, `with_vl`, source loads or strided loads, compare-produced mask,
  accumulator load or scalar seed, contraction compute, store, ABI role order,
  source/result SEW/LMUL, policy, accumulator/result layout, mask facts,
  stride facts, selected `requires`, route operand binding facts, contraction
  family plan mirrors, route-control plan consumption, and direct contraction
  provider facts.
* Focused C++ validation passed:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2` and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
* Focused lit/generated-bundle validation passed for 10 selected contraction
  paths: pre-realized widening MAcc, widening dot, strided widening dot,
  computed-mask widening dot, computed-mask strided widening dot, plus direct
  pre-realized fail-closed probes for the corresponding demoted routes.
* `git diff --check` passed.
* Full `cmake --build build --target check-tianchenrv -j2` passed 464/464.
* Bounded central scan found no contraction typed-body op, contraction plan,
  contraction materialization, contraction compute op construction, or
  contraction validator include left in `RVVSelectedBodyRealization.cpp`.
* Bounded production authority scan over the new contraction owner, its header,
  and the central file found no legacy-i32, descriptor, source-front-door,
  route-id, exact-intrinsic, artifact-name, script-derived, ABI-string-derived,
  direct-route-entry-only, pre-realized-fixture-only, metadata-derived, or
  common-EmitC semantic authority.
* No `.trellis/spec/**` update was required: the existing RVV plugin,
  `tcrv.exec`, EmitC route, testing, and locality specs already define the
  owner-local selected-body contraction boundary implemented here.
* No new runtime, correctness, performance, route, artifact, or executable
  claim was made; `ssh rvv` was intentionally not required.
