# Stage2 RVV base memory movement statement-plan ownership

## Goal

Introduce an RVV-owned base memory movement route statement-plan boundary and
rewire the selected-body RVV EmitC route provider to consume it for the bounded
production-active base memory movement routes in this round: strided load/unit
store, unit load/strided store, indexed gather/unit store, indexed
scatter/unit load, static-mask unit load/store, and static-mask unit store.

The provider remains the owner that instantiates `TCRVEmitCLowerableRoute`,
adds neutral route headers, type mappings, ABI mappings, and selected-boundary
source provenance. The provider must not locally recreate the included base
memory movement statement sequence from operation names, memory-form branches,
ABI strings, or intrinsic mirrors once the RVV-owned plan is available.

## Direction Source

- Direction title: `Stage2 RVV base memory movement statement-plan ownership`.
- Module owner: RVV plugin-local selected-body route statement-plan boundary
  for production-active base memory movement routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `a855f2e5 rvv: own compare select statement plans`.
- `.trellis/.current-task` was absent, so this task was created from the
  Direction Brief before source edits.

## What I Already Know

- The stable RVV authority chain is selected `tcrv.exec` RVV variant -> typed
  low-level `tcrv_rvv` body -> RVV-owned legality/materialization/operand
  binding/route planning -> provider-built `TCRVEmitCLowerableRoute` -> common
  EmitC materialization.
- Commit `c094bf2b` added an RVV-owned elementwise arithmetic statement-plan
  boundary, and commit `a855f2e5` added the matching compare/select
  statement-plan boundary.
- Current code already has base memory movement route-family provider plans,
  materialization facts, and memory operand-binding facts for
  `StridedLoadUnitStore`, `UnitLoadStridedStore`, `IndexedGatherUnitStore`,
  `IndexedScatterUnitLoad`, `MaskedUnitLoadStore`, and `MaskedUnitStore`.
- `RVVEmitCRouteProvider.cpp` still assembles the included base memory
  movement statement sequence locally through generic `addLoopStep`, operation
  and memory-form branches, direct callee selection, and hand-built address
  expressions.
- Computed-mask memory and segment2 memory are adjacent memory families, but
  the Direction Brief explicitly allows finishing one coherent production
  subcluster. This round owns the already route-supported base memory movement
  cluster and leaves computed-mask/segment2 statement plans out of scope.

## Requirements

1. Add an RVV planning API for a base memory movement route statement plan. The
   plan must be derived from verified `RVVSelectedBodyRouteAnalysis`,
   `RVVSelectedBodyRouteMaterializationFacts`, and
   `RVVSelectedBodyMemoryRouteOperandBindingFacts`.
2. The statement plan must cover existing production-active base memory
   movement operations:
   `StridedLoadUnitStore`, `UnitLoadStridedStore`,
   `IndexedGatherUnitStore`, `IndexedScatterUnitLoad`,
   `MaskedUnitLoadStore`, and `MaskedUnitStore`.
3. The plan must carry source operation provenance, callee/intrinsic names,
   operands, results, full-chunk `setvl`, loop/VL placement, strided load/store
   addressing, indexed load/index-scale/gather/scatter addressing, static-mask
   mask/passthrough handling where applicable, unit stores, runtime count, and
   fail-closed dependencies needed by the included route sequence.
4. `RVVEmitCRouteProvider` may still instantiate
   `TCRVEmitCLowerableRoute`, add neutral route-level metadata, and attach the
   returned statements, but it must not locally recreate the included base
   memory movement statement sequence from operation names, memory forms, ABI
   strings, or intrinsic mirrors.
5. Missing or stale statement-plan dependencies must fail closed before common
   EmitC materialization with targeted diagnostics.
6. Preserve emitted semantics, ABI order, operand order, VL/control behavior,
   source provenance, intrinsic spelling, route ids, and generated artifacts
   for valid included base memory movement selected-body routes.
7. Do not add route coverage, new memory forms, computed-mask memory expansion,
   segment2 rewrites, reductions, dtype/LMUL clone batches,
   source-front-door routes, high-level frontend lowering, legacy i32
   authority, descriptor/direct-C/source-export paths, dashboards, broad smoke
   matrices, or common EmitC semantic logic.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` name the RVV plugin,
      EmitC route, testing specs, and previous statement-plan task context.
- [x] A production RVV planning API exists for the base memory movement
      statement plan and exposes an empty/default plan for unrelated route
      families.
- [x] Positive C++ plugin/provider tests prove statement-plan construction and
      provider consumption for strided load/unit store, unit load/strided
      store, indexed gather/unit store, indexed scatter/unit load, static-mask
      unit load/store, and static-mask unit store.
- [x] At least one focused C++ negative test proves a missing or stale
      statement-plan dependency fails closed before route statement
      construction.
- [x] `RVVEmitCRouteProvider.cpp` consumes the returned base memory movement
      statement plan for the included routes before the older generic
      provider-local statement assembly path.
- [x] Bounded provider scan shows the included base memory movement statement
      sequence is no longer locally assembled by the provider beyond neutral
      route instantiation and attaching the RVV-owned plan statements.
- [x] Representative FileCheck coverage for existing explicit or pre-realized
      base memory selected-body artifacts still passes.
- [x] Selected-boundary negative coverage still passes.
- [x] Active-authority scan over touched RVV planning/provider/test/spec files
      finds no newly introduced legacy `RVVI32M1`, `rvv-i32m1`, finite positive
      `tcrv_rvv.i32_*`, descriptor/source-front-door/direct-C/source-export, or
      mirror-only route authority.
- [x] `git diff --check` passes.
- [x] Focused build/tests and `check-tianchenrv` pass, or an exact blocker is
      documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Non-Goals

- No route-family expansion and no new memory-form coverage.
- No computed-mask memory statement-plan ownership in this round.
- No segment2 memory statement-plan ownership in this round.
- No scalar, IME, Offload, TensorExt, future plugin, source-front-door, or
  high-level frontend work.
- No changes to emitted target sequence, ABI, runtime correctness, or
  performance claims. `ssh rvv` evidence is not required unless those claims
  change.
- No movement of RVV semantics into common EmitC/export.
- No compatibility wrapper that preserves old i32m1 route authority.

## Validation Plan

1. Validate this Trellis task.
2. Build the focused C++ plugin test target.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build `tcrv-opt` and `tcrv-translate` if needed for focused lit coverage.
5. Run focused lit/FileCheck coverage for representative base memory movement
   selected-body artifacts and selected-boundary negative fixtures.
6. Run bounded provider scan for included base memory movement statement
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

Relevant previous task read:

- `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-compare-select-statement-plan-ownership/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- Representative memory movement selected-body fixtures under `test/Target/RVV`

## Definition Of Done

The bounded base memory movement statement sequence is owned by RVV route
planning, consumed by the selected-body provider before common EmitC route
construction, covered by focused positive and fail-closed C++ tests plus
representative FileCheck coverage, checked for authority drift, archived using
repo convention, and committed as one coherent change.

## Completion Notes

- Added `RVVSelectedBodyBaseMemoryMovementRouteStatementPlan` and
  `getRVVSelectedBodyBaseMemoryMovementRouteStatementPlan(...)`.
- The provider now asks RVV route planning for the included base memory
  movement statement sequence before the older generic statement assembly
  branch and fails closed if an included base memory route reaches that branch.
- The older generic provider branches still exist later in the function for
  adjacent or future memory surfaces, but the included base memory operations
  are gated before those branches.
- Positive C++ coverage exercises statement-plan construction and provider
  consumption for all six in-scope base memory movement routes.
- Negative C++ coverage clears the verified base memory route-family plan and
  confirms route construction fails closed before statements are built.
- Focused lit/FileCheck coverage for affected selected-body base memory
  artifacts and selected-boundary negative cases passes.
