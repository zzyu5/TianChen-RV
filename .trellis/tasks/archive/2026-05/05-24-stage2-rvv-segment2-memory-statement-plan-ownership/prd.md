# Stage2 RVV segment2 memory statement-plan ownership

## Goal

Introduce an RVV-owned segment2 memory route statement-plan boundary and
rewire the selected-body RVV EmitC route provider to consume it for the
production-active segment2 memory movement routes already present in the typed
selected-body route surface: plain segment2 deinterleave/unit-store, plain
segment2 interleave/unit-load, computed-mask segment2 load/unit-store, and
computed-mask segment2 store/unit-load.

The provider remains the owner that instantiates `TCRVEmitCLowerableRoute`,
adds neutral headers, type mappings, ABI mappings, selected-boundary source
provenance, and attaches the RVV-owned statement plan. The provider must not
locally recreate the included segment2 statement sequence from operation
names, memory-form branches, ABI strings, intrinsic mirrors, mask-producer
mirrors, or field-role mirrors once the RVV-owned plan is available.

## Direction Source

- Direction title: `Stage2 RVV segment2 memory statement-plan ownership`.
- Module owner: RVV plugin-local selected-body route statement-plan boundary
  for production-active segment2 memory movement routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `80643544 rvv: own computed-mask memory statement plans`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- The stable RVV authority chain is selected `tcrv.exec` RVV variant -> typed
  low-level `tcrv_rvv` body -> RVV-owned legality/materialization/operand
  binding/route planning -> provider-built `TCRVEmitCLowerableRoute` -> common
  EmitC materialization.
- Recent completed tasks moved elementwise arithmetic, compare/select, base
  memory movement, and non-segment computed-mask memory statement sequences
  behind RVV-owned statement-plan boundaries.
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` already exposes
  `RVVSelectedBodySegment2MemoryRouteFamilyPlan`, segment2 materialization
  facts through `RVVSelectedBodyRouteMaterializationFacts`, and memory
  operand-binding facts carrying source/destination/field0/field1/runtime
  bindings.
- `RVVEmitCRouteProvider.cpp` still assembles the segment2 route statement
  sequence locally under the segmented-memory branch using `addLoopStep`,
  operation branches, field0/field1 binding choices, setvl/VL expressions,
  segment load/store calls, tuple construction/extraction, compare-mask
  creation, and address expressions.
- The previous computed-mask memory statement-plan task intentionally excluded
  computed-mask segment2 routes and left them for this segment2 owner.

## Requirements

1. Add an RVV planning API for a segment2 memory route statement plan. The
   plan must be derived from verified `RVVSelectedBodyRouteAnalysis`,
   `RVVSelectedBodyRouteMaterializationFacts`, and
   `RVVSelectedBodyMemoryRouteOperandBindingFacts`.
2. The statement plan must cover existing production-active segment2 memory
   routes: `Segment2DeinterleaveUnitStore`,
   `Segment2InterleaveUnitLoad`, `ComputedMaskSegment2LoadUnitStore`, and
   `ComputedMaskSegment2StoreUnitLoad`.
3. The plan must carry source operation provenance, field role bindings,
   full-chunk `setvl`, loop/VL placement, segment load/store or masked
   segment load/store calls, tuple create/extract calls, compare-mask
   construction for computed-mask segment2 routes, operands, results, address
   expressions, ABI order, runtime element count, field0/field1 input or
   output roles, inactive/passthrough handling where present, and fail-closed
   dependencies needed by the included route sequence.
4. `RVVEmitCRouteProvider` may still instantiate
   `TCRVEmitCLowerableRoute`, add neutral route-level metadata, and attach the
   returned statements, but it must not locally recreate the included segment2
   statement sequence from operation names, memory forms, ABI strings,
   intrinsic mirrors, mask-source mirrors, or field-role mirrors.
5. Missing or stale statement-plan dependencies must fail closed before common
   EmitC materialization with targeted diagnostics.
6. Preserve emitted semantics, ABI order, operand order, VL/control behavior,
   source provenance, intrinsic spelling, route ids, and generated artifacts
   for valid included segment2 selected-body routes.
7. Do not add route coverage, new memory forms, reductions, contractions,
   dtype/LMUL clone batches, source-front-door routes, high-level frontend
   lowering, legacy i32 authority, descriptor/direct-C/source-export paths,
   dashboards, broad smoke matrices, or common EmitC semantic logic.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` name the RVV plugin,
      EmitC route, testing specs, and previous computed-mask memory
      statement-plan task context.
- [x] A production RVV planning API exists for the segment2 memory statement
      plan and exposes an empty/default plan for unrelated route families.
- [x] Positive C++ plugin/provider tests prove statement-plan construction
      and provider consumption for plain segment2 deinterleave/unit-store,
      plain segment2 interleave/unit-load, computed-mask segment2
      load/unit-store, and computed-mask segment2 store/unit-load.
- [x] At least one focused C++ negative test proves a missing or stale
      statement-plan dependency fails closed before route statement
      construction.
- [x] `RVVEmitCRouteProvider.cpp` consumes the returned segment2 memory
      statement plan for the included routes before the older generic
      provider-local statement assembly path.
- [x] Bounded provider scan shows the included segment2 memory statement
      sequence is no longer locally assembled by the provider beyond neutral
      route instantiation and attaching the RVV-owned plan statements.
- [x] Representative FileCheck coverage for existing explicit/pre-realized
      segment2 selected-body artifacts still passes.
- [x] Selected-boundary negative coverage still passes.
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

- No route-family expansion and no new memory-form coverage.
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
5. Run focused lit/FileCheck coverage for representative segment2
   selected-body artifacts and selected-boundary negative fixtures.
6. Run bounded provider scan for included segment2 statement assembly residue
   in `RVVEmitCRouteProvider.cpp`.
7. Run active-authority scan over touched RVV planning/provider/test/spec
   files.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`.

## Implementation Result

- Added `RVVSelectedBodySegment2MemoryRouteStatementPlan` and
  `getRVVSelectedBodySegment2MemoryRouteStatementPlan` as the RVV-owned
  statement-plan boundary for the four production-active segment2 routes.
- The plan consumes verified selected-body route analysis, materialization
  facts, and memory operand-binding facts, then carries full-chunk `setvl`,
  loop `setvl`, address expressions, field0/field1 roles, compare-mask setup
  where needed, tuple create/extract calls, segment load/store calls, operands,
  results, and source operation provenance.
- `RVVEmitCRouteProvider` now attaches the returned plan before generic
  provider-local statement assembly and fails closed if an included segment2
  route lacks an RVV-owned statement plan.
- The older provider-local segment2 statement assembly and dead segment2
  operand-binding residue were removed from the generic provider path.
- Focused C++ tests now cover positive plan construction and provider
  consumption for plain deinterleave, plain interleave, computed-mask segment2
  load, and computed-mask segment2 store, plus a missing computed-mask route
  family plan failing closed before statement construction.
- `.trellis/spec/extension-plugins/rvv-plugin.md` now documents the durable
  segment2 memory statement-plan boundary and wrong/correct ownership examples.

## Validation Result

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-stage2-rvv-segment2-memory-statement-plan-ownership`
  passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- Focused lit/FileCheck passed with
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter='(segment2|selected-boundary)' /home/kingdom/phdworks/TianchenRV/build/test`:
  23 selected tests passed. The first bare `lit.py` attempt failed because the
  executable was not on `PATH`; rerunning through the configured lit path fixed
  the invocation issue.
- Bounded provider scan found the segment2 plan call and fail-closed guard in
  `RVVEmitCRouteProvider.cpp`, and found no remaining provider-local segment
  tuple, masked segment2, segment load/store intrinsic, or segment tuple
  create/extract assembly keywords in that file.
- Active-authority scan over newly added code lines in planning/provider/test
  files found no `RVVI32M1`, `rvv-i32m1`, positive `tcrv_rvv.i32_*`,
  descriptor/source-front-door/direct-C/source-export, or mirror-only authority
  additions. The only broader diff match was the spec's negative prohibition
  list.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed, including all
  363 lit tests.
- No `ssh rvv` evidence was collected because this task did not change or
  claim emitted target sequence, ABI, runtime correctness, or performance.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Relevant previous task read:

- `.trellis/tasks/archive/2026-05/05-24-05-24-stage2-rvv-computed-mask-memory-statement-plan-ownership/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- Representative segment2 memory selected-body fixtures under
  `test/Target/RVV`

## Definition Of Done

The bounded segment2 memory statement sequence is owned by RVV route planning,
consumed by the selected-body provider before common EmitC route construction,
covered by focused positive and fail-closed C++ tests plus representative
FileCheck coverage, checked for authority drift, archived using repo
convention, and committed as one coherent change.
