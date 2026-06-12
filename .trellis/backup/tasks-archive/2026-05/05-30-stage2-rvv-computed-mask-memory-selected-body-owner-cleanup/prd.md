# Stage2 RVV Computed-Mask Memory Selected-Body Owner Cleanup

## Goal

Move non-segment computed-mask memory selected-body validation and realization
authority out of the central `RVVSelectedBodyRealization.cpp` materialization
branch helper and into an RVV computed-mask memory owner-local production
component. The central selected-body file should remain responsible for owner
registry, neutral dispatch, shared construction mechanics, and fail-closed
diagnostics only.

The intended production chain is:

```text
selected tcrv.exec RVV variant
  -> typed computed-mask memory pre-realized tcrv_rvv body
  -> computed-mask memory owner-local validation and realization
  -> realized typed tcrv_rvv setvl/with_vl/load/compare/masked-load/
     masked-store/strided/indexed memory facts
  -> computed-mask memory route-family analysis/provider facts
  -> TCRVEmitCLowerableRoute
  -> neutral common EmitC materializer
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv evidence when runtime/correctness/performance is claimed
```

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Before task creation, the worktree was clean and `HEAD` was
  `2f01d9eb rvv: move standalone reduction selected-body realization owner-side`.
- No usable `.trellis/.current-task` existed, so this task was created from the
  Hermes direction brief.
- Specs require RVV Stage2 selected-body realization to run through the owner
  registry before route-family analysis, route-control provider plans,
  statement plans, `TCRVEmitCLowerableRoute`, common EmitC, or target artifact
  export.
- `tcrv.exec` owns only the execution envelope and ABI/runtime role
  declarations; computed-mask memory mask construction/use, passthrough,
  strided/indexed facts, runtime AVL/VL, SEW/LMUL, policy, and memory movement
  semantics belong to typed `tcrv_rvv` plus RVV plugin-local owners.
- Common EmitC may materialize provider-built routes but must not infer RVV
  dtype, SEW, LMUL, mask/tail policy, memory form, intrinsic spelling, ABI
  order, or statement shape.
- The previous archived standalone-reduction cleanup provides the owner-local
  API pattern through
  `RVVStandaloneReductionSelectedBodyRealizationOwner.h` and
  `RVVStandaloneReductionSelectedBodyRealizationOwner.cpp`.
- Current `RVVSelectedBodyRealization.cpp` still registers a computed-mask
  memory owner but routes it through the central owner-local branch helper,
  where typed computed-mask unit, strided-store, strided-load, indexed-gather,
  and indexed-scatter realization branches still live.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake /
   lit / FileCheck. Python remains limited to tooling, probes, runners,
   artifact parsing, and small support scripts.
2. Add an owner-local computed-mask memory selected-body realization component
   outside `RVVSelectedBodyRealization.cpp`.
3. Move computed-mask memory semantic predicates, validation dispatch,
   diagnostics, and materialization for these non-segment families into that
   owner component:
   - `TypedComputedMaskMemoryPreRealizedBodyOp`
   - `TypedComputedMaskStridedStorePreRealizedBodyOp`
   - `TypedComputedMaskStridedLoadPreRealizedBodyOp`
   - `TypedComputedMaskIndexedGatherPreRealizedBodyOp`
   - `TypedComputedMaskIndexedScatterPreRealizedBodyOp`
4. Preserve typed authority for mask producer/source, predicate kind,
   passthrough/inactive-lane policy, memory roles, source/destination stride,
   index runtime ABI facts, SEW/LMUL, AVL/VL, policy, selected requires, and
   ABI roles.
5. Keep computed-mask segment2 outside this owner; segment2 memory realization
   remains with the segment2 owner path.
6. Central `RVVSelectedBodyRealization.cpp` may keep shared construction
   helpers, owner registry entries, neutral exact-one owner dispatch, and
   generic fail-closed diagnostics, but must not retain the computed-mask
   memory realization branch bodies as central authority.
7. Unsupported owner matches, wrong config, wrong ABI role, wrong memory form,
   stale/missing mask/passthrough/stride/index facts, route-id-derived
   authority, metadata-derived authority, descriptor/source-front-door
   authority, exact-intrinsic-as-authority, direct-route-entry-only authority,
   legacy-i32-derived authority, or common-EmitC semantic invention must fail
   closed before provider facts or common materialization.
8. Do not modify completed standalone-reduction, elementwise/compare-select,
   base-memory, MAcc, contraction, widening-conversion, or segment2 behavior
   except for narrow interface/build compatibility.

## Acceptance Criteria

- [ ] A production computed-mask memory selected-body realization owner exists
      outside `RVVSelectedBodyRealization.cpp`.
- [ ] The computed-mask memory owner is wired into the owner registry and
      build system through a narrow header/source boundary.
- [ ] The central selected-body realization file no longer contains
      non-segment computed-mask memory semantic validation/materialization
      branches beyond registry, neutral dispatch, shared mechanics, and
      diagnostics.
- [ ] Realization preserves typed unit, strided-store, strided-load,
      indexed-gather, and indexed-scatter computed-mask memory facts, including
      mask construction, passthrough/inactive policy, runtime AVL/VL,
      source/destination stride, index, offset unit, index uniqueness, SEW/LMUL,
      policy, and selected variant requires.
- [ ] Focused C++ or lit coverage proves computed-mask memory owner selection,
      fail-closed out-of-family behavior, realized typed `tcrv_rvv` facts,
      provider consumption, and absence of central takeover.
- [ ] Focused generated-bundle dry-runs cover representative unit, strided, and
      indexed computed-mask memory cases, or record an exact blocker.
- [ ] At least one representative executable computed-mask memory path has
      `ssh rvv` evidence if runtime/correctness is claimed, or the final report
      avoids runtime/correctness claims and records the blocker.
- [ ] Completed standalone-reduction and existing elementwise owner paths have
      focused non-regression coverage.
- [ ] Bounded authority scan over touched code/tests shows no route or
      executable claim depends on central ad hoc, name-derived, metadata-
      derived, descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [ ] `git diff --check` passes, focused checks pass, `check-tianchenrv` passes
      or an exact blocker is recorded, and the task is finished/archived with
      one coherent commit if complete.

## Non-goals

- Do not rewrite the selected-body realization framework.
- Do not move neutral shared construction helpers unless a small header boundary
  is strictly needed for the owner extraction.
- Do not start new RVV coverage, dtype/LMUL clone batches, high-level
  Linalg/frontend work, Segment2 extraction, MAcc/contraction extraction,
  base-memory extraction, standalone-reduction rework, route-provider redesign,
  broad smoke matrices, dashboards, reports, or evidence-only tasks.
- Do not change computation semantics, dtype semantics, parameter roles,
  dispatch/fallback behavior, runtime `n`/AVL/VL, mask/tail policy, or common
  EmitC materialization semantics.

## Validation Plan

1. Validate task context after PRD/context files are in place.
2. Inspect relevant specs, previous archived standalone-reduction owner cleanup,
   central selected-body realization code, computed-mask memory route/provider
   consumers, and directly related tests.
3. Implement the computed-mask memory owner-local selected-body realization
   component and wire it into the selected-body owner registry/build system.
4. Shrink central computed-mask memory branches to registry/dispatch/shared
   mechanics only.
5. Add or update focused C++/lit tests for owner registry/selection,
   out-of-family rejection, realized typed facts, provider consumption, and
   owner-boundary evidence.
6. Run focused generated-bundle dry-runs for representative unit, strided, and
   indexed computed-mask memory cases; run focused non-regression for
   standalone reduction and existing elementwise owner paths.
7. Run bounded authority scans, `git diff --check`, and `check-tianchenrv` or
   record the exact blocker.
8. Finish/archive the Trellis task, update the workspace journal, and create one
   coherent commit if acceptance is satisfied.

## Files to Inspect First

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-standalone-reduction-selected-body-owner-cleanup/`
- `include/TianChenRV/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.h`
- `lib/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/CMakeLists.txt`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- Directly related computed-mask memory generated-bundle tests and provider
  consumers.
