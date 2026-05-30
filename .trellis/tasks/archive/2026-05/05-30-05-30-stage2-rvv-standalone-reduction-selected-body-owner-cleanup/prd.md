# Stage2 RVV Standalone-Reduction Selected-Body Owner Cleanup

## Goal

Move standalone-reduction selected-body validation and realization authority out
of the central `RVVSelectedBodyRealization.cpp` family-specific branches and
into an RVV reduction-family owner-local production component. The central
selected-body file should remain responsible for owner registry, neutral
dispatch, shared mechanics, and fail-closed diagnostics only.

The intended production chain is:

```text
selected tcrv.exec RVV variant
  -> typed standalone-reduction pre-realized tcrv_rvv body
  -> reduction-family owner-local selected-body validation and realization
  -> realized typed tcrv_rvv setvl/with_vl/load/mask/reduce/store facts
  -> route analysis / provider facts
  -> TCRVEmitCLowerableRoute
  -> neutral common EmitC materializer
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv evidence when runtime/correctness/performance is claimed
```

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Before task creation, the worktree was clean and `HEAD` was
  `e12957a2 rvv: move elementwise selected-body realization owner-side`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  direction brief.
- Specs require RVV Stage2 selected-body realization to run before provider
  facts, route-control provider plans, statement plans, `TCRVEmitCLowerableRoute`,
  common EmitC, or target artifact export.
- `tcrv.exec` owns only execution envelope and ABI/runtime roles; standalone
  reduction compute semantics, dtype/config, mask/tail policy, accumulation
  shape, and route facts belong to `tcrv_rvv` plus RVV plugin-local owners.
- Common EmitC may materialize provider-built routes but must not infer RVV
  dtype, SEW, LMUL, policy, operation kind, intrinsic spelling, ABI order, or
  reduction shape.
- The previous archived elementwise/compare-select cleanup provides the
  owner-local API pattern through
  `RVVEmitCElementwiseRouteFamilyPlanOwners.h` and
  `RVVElementwiseSelectedBodyRealizationOwner.cpp`; it is not the semantic owner
  for reductions.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake /
   lit / FileCheck. Python remains limited to tooling, probes, runners,
   artifact parsing, and small support scripts.
2. Add a reduction-family owner-local selected-body realization component for
   the existing standalone-reduction selected-body family handled by the
   registry.
3. Move standalone-reduction semantic validation and materialization authority
   out of central `RVVSelectedBodyRealization.cpp`. Central code may keep shared
   helpers, owner registry entries, exact-one dispatch, and neutral diagnostics.
4. Cover the currently supported standalone-reduction variants, including plain,
   computed-mask, and runtime-scalar computed-mask bodies if current code routes
   them through the standalone-reduction owner.
5. Preserve operation kind, source/result dtype, source SEW/LMUL, scalar
   accumulator/result channel, policy, reduction/accumulation shape, mask
   producer/use, runtime scalar mask role, memory roles, runtime `n`/AVL/VL,
   `setvl` placement, required capabilities, provider route facts, target
   artifact ABI order, and fail-closed diagnostics.
6. Unsupported owner matches, wrong dtype/config, wrong mask binding, wrong
   runtime role, wrong reduction shape, stale metadata, direct-route-entry-only
   authority, source/artifact/script-derived authority, exact-intrinsic-as-
   authority, route-id-derived authority, legacy-i32-derived authority, or
   common-EmitC semantic invention must fail closed before provider facts or
   common materialization.
7. Do not modify completed elementwise/compare-select owner behavior except for
   narrow interface compatibility.

## Acceptance Criteria

- [ ] A production reduction-family selected-body realization owner exists
      outside `RVVSelectedBodyRealization.cpp`.
- [ ] The central selected-body realization file no longer contains
      standalone-reduction semantic validation/materialization branches beyond
      registry, neutral dispatch, shared mechanics, and diagnostics.
- [ ] Standalone reduction realization preserves typed source/work and scalar
      accumulator/result channels, mask/runtime-scalar facts, runtime AVL/VL,
      memory roles, and provider-consumed selected-body structure.
- [ ] Focused C++ or lit coverage proves standalone-reduction owner selection,
      fail-closed non-owner behavior, realized typed `tcrv_rvv` facts, provider
      consumption, and absence of central takeover.
- [ ] Generated-bundle dry-runs cover representative standalone reduction paths,
      including mask/runtime-scalar variants that remain supported.
- [ ] Focused `ssh rvv` generated-bundle subset covers representative reduction
      counts including 0, 1, exact, tail, and stress cases, or records an exact
      infrastructure blocker.
- [ ] Completed elementwise/compare-select owner-local paths have focused
      non-regression coverage.
- [ ] Bounded authority scan over touched code/tests shows no route or
      executable claim depends on central ad hoc, name-derived, metadata-
      derived, descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [ ] `git diff --check` passes, focused checks pass, `check-tianchenrv` passes
      or an exact blocker is recorded, and the task is finished/archived with one
      coherent commit if complete.

## Non-goals

- Do not extract all remaining selected-body owners in this round.
- Do not start MAcc, computed-mask MAcc, contraction, widening conversion, base
  memory movement, computed-mask memory, segment2 memory, high-level
  Linalg/frontend lowering, new reduction op coverage, dtype/LMUL clone batches,
  direct pre-realized route-entry restoration, broad smoke matrices,
  dashboard/report work, or evidence-only tasks.
- Do not treat prompt edits, reports, helper-only changes, or broad tests as
  the main achievement.
- Do not make common EmitC, artifact metadata, scripts, descriptors, ABI strings,
  route ids, exact intrinsic spellings, or legacy i32 helper names into route or
  realization authority.

## Validation Plan

1. Validate task context after PRD/context files are in place.
2. Inspect relevant specs, previous archived elementwise owner cleanup, central
   selected-body realization code, reduction route planning/provider consumers,
   and directly related tests.
3. Implement the reduction-family owner-local selected-body realization
   component and wire it into the selected-body owner registry/build system.
4. Shrink central standalone-reduction branches to registry/dispatch/shared
   helpers only.
5. Run focused C++/lit checks for selected-body realization and provider
   consumption, generated-bundle dry-runs for supported representative
   standalone reductions, focused non-regression for elementwise/compare-select,
   and `ssh rvv` representative reduction evidence if reachable.
6. Run bounded authority scans, `git diff --check`, and `check-tianchenrv` or
   record the exact blocker.
7. Finish/archive the Trellis task, update the workspace journal, and create one
   coherent commit if acceptance is satisfied.

## Files to Inspect First

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-elementwise-compare-select-selected-body-owner-cleanup/`
- `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h`
- `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- Reduction route planning/provider files and directly related tests as
  consumers and evidence.
