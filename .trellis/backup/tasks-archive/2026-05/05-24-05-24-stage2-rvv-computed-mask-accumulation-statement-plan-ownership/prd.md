# Stage2 RVV computed-mask accumulation statement-plan ownership

## Goal

Introduce an RVV-owned computed-mask accumulation route statement-plan
boundary and rewire the selected-body RVV EmitC route provider to consume it
for one coherent production-active accumulation subcluster already represented
by verified RVV route-family, materialization, and operand-binding facts.

This round targets the masked accumulation side where mask production,
accumulator roles, VL/control, compute calls, and stores should be carried by
RVV-owned planning facts before `TCRVEmitCLowerableRoute` construction. The
provider remains the owner that instantiates the route object, adds neutral
headers, type mappings, ABI mappings, selected-boundary source provenance, and
attaches the RVV-owned statement plan. It must not locally recreate included
computed-mask accumulation statements from operation names, ABI strings,
intrinsic mirrors, mask-source mirrors, accumulator-role mirrors, route ids, or
artifact metadata.

## Direction Source

- Direction title: `Stage2 RVV computed-mask accumulation statement-plan ownership`.
- Module owner: RVV plugin-local selected-body route statement-plan boundary
  for production-active computed-mask accumulation routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `6cffc808 rvv: own segment2 memory statement plans`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- The stable RVV authority chain is selected `tcrv.exec` RVV variant -> typed
  low-level `tcrv_rvv` body -> RVV-owned legality/materialization/operand
  binding/route planning -> provider-built `TCRVEmitCLowerableRoute` -> common
  EmitC materialization.
- Recent completed tasks moved elementwise arithmetic, compare/select, base
  memory movement, computed-mask memory movement, and segment2 memory movement
  statement sequences behind RVV-owned statement-plan boundaries.
- The RVV spec already requires math operand-binding facts for reduction,
  plain MAcc, computed-mask MAcc, standalone reductions, widening MAcc,
  widening conversion, and widening dot-reduction shapes.
- The RVV spec explicitly names computed-mask accumulation as a
  materialization-facts dependency and requires fail-closed diagnostics when a
  computed-mask accumulation route lacks the required shared accumulation plan.
- This task is not Stage2 coverage expansion. It is ownership regularization
  for production-active computed-mask accumulation routes that already have
  verified typed-body/materialization/operand-binding facts.

## Requirements

1. Add or regularize an RVV planning API for a computed-mask accumulation
   route statement plan. The plan must be derived from verified
   `RVVSelectedBodyRouteAnalysis`,
   `RVVSelectedBodyRouteMaterializationFacts`, and
   `RVVSelectedBodyMathRouteOperandBindingFacts`.
2. The included subcluster must be production-active computed-mask
   accumulation only. If the full surface is too large, finish one coherent
   computed-mask accumulation subcluster and record exact continuation
   evidence rather than landing helper-only or test-only work.
3. The plan must carry source operation provenance, mask producer/source facts,
   accumulator role bindings, full-chunk `setvl`, loop/VL placement,
   load/seed/accumulator use, compute calls, store calls, operands, results,
   runtime element count, ABI order, intrinsic choices supplied by
   materialization facts, and fail-closed dependencies needed by the included
   route sequence.
4. `RVVEmitCRouteProvider` may still instantiate
   `TCRVEmitCLowerableRoute`, add neutral route-level metadata, and attach
   returned statements, but it must not locally recreate the included
   computed-mask accumulation statement sequence from operation names, ABI
   strings, route ids, artifacts, descriptor residue, intrinsic mirrors,
   mask-source mirrors, or accumulator-role mirrors.
5. Missing or stale statement-plan dependencies must fail closed before route
   statement construction and before common EmitC materialization with
   targeted diagnostics.
6. Preserve emitted semantics, ABI order, operand order, VL/control behavior,
   source provenance, intrinsic spelling, route ids, and generated artifacts
   for valid included selected-body routes.
7. Do not add route coverage, new reduction/contraction forms, dtype/LMUL clone
   batches, memory-route rewrites, high-level frontend lowering,
   source-front-door positive routes, legacy i32 authority,
   descriptor/direct-C/source-export paths, dashboards, broad smoke matrices,
   or common EmitC semantic logic.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` name the RVV plugin,
      EmitC route, testing specs, and recent statement-plan task context.
- [x] A production RVV planning API exists for the computed-mask accumulation
      statement plan and exposes an empty/default plan for unrelated route
      families.
- [x] Positive C++ plugin/provider tests prove statement-plan construction and
      provider consumption for at least one coherent production-active
      computed-mask accumulation subcluster already represented by verified
      route-family/materialization/operand-binding facts.
- [x] At least one focused C++ negative test proves a missing or stale
      statement-plan dependency fails closed before route statement
      construction.
- [x] `RVVEmitCRouteProvider.cpp` consumes the returned computed-mask
      accumulation statement plan for included routes before the older generic
      provider-local statement assembly path.
- [x] Bounded provider scan shows the included computed-mask accumulation
      statement sequence is no longer locally assembled by the provider beyond
      neutral route instantiation and attaching the RVV-owned plan statements.
- [x] Representative FileCheck coverage for affected existing computed-mask
      accumulation selected-body artifacts still passes.
- [x] Selected-boundary negative coverage still passes if touched by this
      route cluster.
- [x] Active-authority scan over touched RVV planning/provider/test/spec files
      finds no newly introduced legacy `RVVI32M1`, `rvv-i32m1`, finite
      positive `tcrv_rvv.i32_*`, descriptor/source-front-door/direct-C/
      source-export, or mirror-only route authority.
- [x] `git diff --check` passes.
- [x] Focused build/tests and `check-tianchenrv` pass, or an exact blocker is
      documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Non-Goals

- No new route coverage and no new accumulation/reduction/contraction forms.
- No memory-route statement-plan rewrites in this round.
- No scalar, IME, Offload, TensorExt, future plugin, source-front-door, or
  high-level frontend work.
- No changes to emitted target sequence, ABI, runtime correctness, or
  performance claims. `ssh rvv` evidence is not required unless those claims
  change.
- No movement of RVV semantics into common EmitC/export.
- No compatibility wrapper that preserves old i32m1 route authority.

## Validation Plan

1. Validate and start this Trellis task.
2. Build the focused C++ plugin test target.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build `tcrv-opt` and `tcrv-translate` if needed for focused lit coverage.
5. Run focused lit/FileCheck coverage for representative computed-mask
   accumulation selected-body artifacts and selected-boundary negative
   fixtures.
6. Run bounded provider scan for included computed-mask accumulation statement
   assembly residue in `RVVEmitCRouteProvider.cpp`.
7. Run active-authority scan over touched RVV planning/provider/test/spec
   files.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant previous tasks read:

- `.trellis/tasks/archive/2026-05/05-24-05-24-stage2-rvv-computed-mask-memory-statement-plan-ownership/prd.md`
- `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-segment2-memory-statement-plan-ownership/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- Representative existing computed-mask accumulation fixtures under
  `test/Target/RVV`

## Definition Of Done

The bounded computed-mask accumulation statement sequence is owned by RVV
route planning, consumed by the selected-body provider before common EmitC
route construction, covered by focused positive and fail-closed C++ tests plus
representative FileCheck coverage, checked for authority drift, archived using
repo convention, and committed as one coherent change.

## Implementation Result

- Added `RVVSelectedBodyComputedMaskAccumulationRouteStatementPlan` and
  `getRVVSelectedBodyComputedMaskAccumulationRouteStatementPlan(...)` as the
  RVV-owned statement-plan boundary for the production-active computed-mask
  MAcc subcluster: `ComputedMaskedMAccAdd` and
  `RuntimeScalarComputedMaskedMAccAdd`.
- The plan consumes verified selected-body route analysis, materialization
  facts, and math operand-binding facts, then carries full-chunk `setvl`, loop
  `setvl`, compare producer load/splat, payload loads, accumulator load,
  compare-mask creation, active MAcc compute, masked merge/passthrough, store,
  operands, results, and source operation provenance.
- `RVVEmitCRouteProvider` now attaches the returned plan before generic
  provider-local statement assembly and fails closed if an included
  computed-mask MAcc route lacks an RVV-owned statement plan.
- The provider-local computed-mask MAcc branches for local operand binding,
  payload load, accumulator load, active MAcc, and masked merge were removed
  from the generic provider path. Plain MAcc remains in the generic math path
  because it is outside this computed-mask accumulation statement-plan
  subcluster.
- Focused C++ tests now cover positive plan construction and provider
  consumption for vector-compare computed-mask MAcc and runtime-scalar
  computed-mask MAcc, an empty/default plan for an unrelated route, and a
  missing computed-mask accumulation plan failing closed before statement
  construction.
- `.trellis/spec/extension-plugins/rvv-plugin.md` now documents the durable
  computed-mask accumulation statement-plan boundary and wrong/correct
  ownership examples.

## Validation Result

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-05-24-stage2-rvv-computed-mask-accumulation-statement-plan-ownership`
  passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- First lit attempt from repo root failed because `build/test/lit.site.cfg.py`
  resolves `../../test/lit.cfg.py` relative to the current working directory.
  Rerunning from `build/test` with the same filter passed.
- Focused lit/FileCheck passed with
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter='(computed.*macc|runtime-scalar-cmp-masked-macc|selected-boundary)' .`
  from `build/test`: 14 selected tests passed.
- Bounded provider scan found only the computed-mask MAcc statement-plan guard
  in `RVVEmitCRouteProvider.cpp` and no local `macc_lhs_vec` /
  `active_macc_vec` computed-mask accumulation statement assembly residue.
- Active-authority scan over newly added C++ lines found no legacy
  `RVVI32M1`, `rvv-i32m1`, positive `tcrv_rvv.i32_*`,
  descriptor/source-front-door/direct-C/source-export, or mirror-only
  authority additions. The only broader diff match was the spec's negative
  prohibition list.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed, including all
  363 lit tests.
- No `ssh rvv` evidence was collected because this task did not change or
  claim emitted target sequence, ABI, runtime correctness, or performance.

## Continuation Notes

- Computed-mask standalone reduction routes remain outside this statement-plan
  boundary. They still use the existing generic math/reduction statement
  construction surface and should be handled by a later standalone reduction
  statement-plan owner if that surface becomes the next bottleneck.
