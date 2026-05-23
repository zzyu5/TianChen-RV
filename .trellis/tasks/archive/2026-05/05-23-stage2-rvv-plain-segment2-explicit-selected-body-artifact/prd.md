# Stage2 RVV plain segment2 explicit selected-body artifact closure

## Goal

Close the missing positive explicit selected-body artifact path for the existing
plain segment2 RVV memory family. A selected `tcrv.exec` RVV variant with an
explicit typed `tcrv_rvv.segment2_load` / `tcrv_rvv.segment2_store` body should
flow through RVV-owned realization/provider planning into the common EmitC
artifact/header/export path, while incomplete or mismatched explicit bodies
remain fail-closed.

## What I already know

* The task source is the Hermes Direction Brief supplied for this session.
* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial real repository state: worktree clean; HEAD
  `5e06dd2c rvv: close plain segment2 memory provider binding`.
* No `.trellis/.current-task` existed before this task was created.
* The previous archived plain segment2 task closed provider-side runtime,
  route-family, ABI order, segment layout, and operand binding validation for
  the plain segment2 memory family.
* The previous task's PRD says pre-realized plain segment2 deinterleave and
  interleave generated-bundle and `ssh rvv` evidence existed, but this brief
  identifies the explicit selected-body artifact counterpart as the remaining
  blocker.
* Specs require RVV executable support to start from typed low-level
  `tcrv_rvv` body facts, keep RVV route semantics in the RVV plugin, and keep
  common EmitC/export neutral.
* Computed-mask segment2 already has an adjacent explicit selected-body
  artifact/evidence pattern; this task should follow that bounded pattern only
  where it applies to plain segment2.

## Requirements

* Inventory the existing plain segment2 explicit selected-body fixtures,
  selected-body realization path, provider planning path, EmitC route provider,
  and generated-bundle ABI script support before changing production code.
* Accept complete explicit selected bodies for:
  * segment2 deinterleave: one segment2 load feeding two plain stores;
  * segment2 interleave: two plain loads feeding one segment2 store.
* Require the accepted explicit body to structurally match the RVV provider
  facts for operation, segment count, field roles, memory forms, runtime
  `n`/AVL/VL use, typed vector/config facts, and ABI operand binding.
* Keep mismatched or incomplete explicit bodies fail-closed with focused
  diagnostics or FileCheck-visible failures.
* Prove positive explicit selected-body plan/header/export behavior for both
  plain segment2 op kinds.
* Use the narrow generated-bundle ABI script path for the two plain segment2
  op kinds only if runtime ABI evidence is claimed.
* Keep common EmitC/export neutral. Plain segment2 route semantics must remain
  in RVV planning/provider/realization/target support.

## Acceptance Criteria

* [x] Complete explicit selected-body deinterleave fixture is accepted and
  produces the expected segment2 plan/header/export artifacts.
* [x] Complete explicit selected-body interleave fixture is accepted and
  produces the expected segment2 plan/header/export artifacts.
* [x] Existing or new negative explicit selected-body segment2 fixture proves
  incomplete or mismatched structure remains fail-closed.
* [x] Focused FileCheck coverage verifies the selected-body plan/export surface
  without treating route ids, artifact names, descriptors, source-front-door
  metadata, or exact intrinsic spelling as route authority.
* [x] Production validation changes, if any, have targeted plugin/provider
  coverage.
* [x] Narrow generated-bundle ABI evidence is run only for the two plain
  segment2 op kinds if this task claims runtime/ABI executable behavior.
* [x] `git diff --check` passes.
* [x] Focused tests for the changed behavior pass.
* [x] `check-tianchenrv` or the repo's focused equivalent is run, or an exact
  toolchain blocker is documented.
* [x] Final `git status --short` is clean after the task commit.

## Definition of Done

* PRD and task context reflect the bounded module owner.
* Source changes, if needed, are limited to the RVV plugin-local selected-body,
  route/provider, target fixture, and focused script/test surfaces.
* Focused tests prove both positive explicit selected-body artifact paths and
  fail-closed negative behavior.
* Trellis task status and workspace journal are truthful.
* The task is finished/archived using the repo convention.
* One coherent commit records the completed task unless a blocker remains and
  the exact continuation point is documented.

## Out of Scope

* New route families or broad route-matrix expansion.
* Dtype or LMUL clone batches.
* Linalg/Vector/StableHLO frontend lowering.
* Source-front-door routes, source artifacts, descriptor-driven compute, or
  compatibility wrappers for old i32 authority.
* High-level kernel ops, one-op-per-intrinsic wrapper growth, dashboards,
  broad smoke matrices, or generic harness expansion unrelated to plain
  segment2 explicit selected-body execution.
* Moving RVV semantics into common EmitC/export code.

## Technical Approach

1. Read the relevant specs and previous plain segment2 PRD, then inspect the
   current fixture/code path for computed-mask segment2 and plain segment2.
2. Determine whether production code already accepts complete explicit plain
   segment2 bodies or whether realization/provider/export needs a bounded fix.
3. Add or repair positive explicit selected-body fixtures for both
   deinterleave and interleave artifact/header/export behavior.
4. Preserve or extend negative coverage for mismatched explicit bodies.
5. Run focused target/provider tests and the narrow generated-bundle ABI path
   if executable ABI evidence is claimed.
6. Finish/archive the task and commit the coherent result.

## Technical Notes

Read targets for this round:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-plain-segment2-memory-runtime-binding-closure/prd.md`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
* `test/Target/RVV/pre-realized-selected-body-artifact-segment2-*.mlir`
* `test/Target/RVV/explicit-selected-body-segment2-interleave-negative.mlir`
* `scripts/rvv_generated_bundle_abi_e2e.py`

## Completion Evidence

### What Changed

* Added positive explicit selected-body artifact fixtures for:
  * `segment2_deinterleave_unit_store`
  * `segment2_interleave_unit_load`
* Added explicit selected-body generated-bundle ABI expectations for the same
  two op kinds in `scripts/rvv_generated_bundle_abi_e2e.py`.
* Added a small testing-spec note that selected-body artifact fixture
  kernel/variant symbols should remain concise because target export derives a
  bounded C/EmitC function identifier from them.
* No production C++ route/provider changes were required. Current production
  RVV selected-body planning/provider/export already accepts the complete typed
  explicit plain segment2 bodies once fixtures and harness expectations exist.
* Kept existing `explicit-selected-body-segment2-interleave-negative.mlir` as
  the focused fail-closed explicit-body negative case.

### Evidence Run

* [OK] Manual explicit selected-body emission-plan materialization for both
  new plain segment2 fixtures with `./build/bin/tcrv-opt`.
* [OK] Manual target header export for both new plain segment2 fixtures with
  `./build/bin/tcrv-translate`.
* [OK] Existing negative explicit segment2 fixture still fails closed with:
  `bounded generic RVV segment2 interleave route requires exactly two
  unit-stride tcrv_rvv.load ops for field0 and field1`.
* [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [OK] Explicit generated-bundle dry-run for
  `segment2_deinterleave_unit_store` and
  `segment2_interleave_unit_load` at counts 7, 16, and 23.
* [OK] Explicit real `ssh rvv` evidence for
  `segment2_deinterleave_unit_store` and
  `segment2_interleave_unit_load` at counts 7, 16, and 23, including field
  order and tail preservation checks.
* [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
* [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
* [OK] `git diff --check`
* [OK] `cmake --build build --target check-tianchenrv -j2` (363/363)

### Notes

The positive route authority remains the typed `tcrv_rvv` body structure:
plain segment2 deinterleave uses explicit `segment2_load -> move -> move ->
store -> store`, and plain segment2 interleave uses explicit `load -> load ->
segment2_store`. Route ids, artifact names, emission diagnostics, and header
metadata are only checked as mirrors after RVV provider route construction.
