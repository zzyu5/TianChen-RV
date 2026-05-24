# Stage2 RVV computed-mask memory statement-plan ownership

## Goal

Introduce an RVV-owned computed-mask memory route statement-plan boundary and
rewire the selected-body RVV EmitC route provider to consume it for this
round's bounded production-active computed-mask memory movement subcluster:
runtime-scalar computed-mask store/load-store, computed-mask unit load/store,
computed-mask strided store, computed-mask strided load/unit-store,
computed-mask indexed gather/unit-store, and computed-mask indexed
scatter/unit-load.

The provider remains the owner that instantiates `TCRVEmitCLowerableRoute`,
adds neutral headers, type mappings, ABI mappings, selected-boundary source
provenance, and attaches the RVV-owned statement plan. The provider must not
locally recreate the included computed-mask memory statement sequence from
operation names, memory-form branches, ABI strings, intrinsic mirrors, or
mask producer mirrors once the RVV-owned plan is available.

## Direction Source

- Direction title: `Stage2 RVV computed-mask memory statement-plan ownership`.
- Module owner: RVV plugin-local selected-body route statement-plan boundary
  for production-active computed-mask memory movement routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `9a21c276 rvv: own base memory statement plans`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- The stable RVV authority chain is selected `tcrv.exec` RVV variant -> typed
  low-level `tcrv_rvv` body -> RVV-owned legality/materialization/operand
  binding/route planning -> provider-built `TCRVEmitCLowerableRoute` -> common
  EmitC materialization.
- Recent completed tasks moved elementwise arithmetic, compare/select, and
  base memory movement statement sequences behind RVV-owned statement-plan
  boundaries.
- Current code already has computed-mask memory route-family provider plans,
  materialization facts, and memory operand-binding facts for runtime-scalar
  computed-mask memory, computed-mask unit/strided/indexed memory, and
  computed-mask segment2 memory.
- `RVVEmitCRouteProvider.cpp` still assembles the included computed-mask
  memory route statement sequence locally through `addCallStepFromSource`,
  `addLoopStep`, operation/memory-form branches, hand-selected mask producer
  handling, direct intrinsic/callee choices, and address/result expressions.
- Computed-mask segment2 memory is adjacent but overlaps the segment2 memory
  route family. This round excludes computed-mask segment2 from the statement
  plan so the task remains one coherent non-segment computed-mask memory
  subcluster.

## Requirements

1. Add an RVV planning API for a computed-mask memory route statement plan. The
   plan must be derived from verified `RVVSelectedBodyRouteAnalysis`,
   `RVVSelectedBodyRouteMaterializationFacts`, and
   `RVVSelectedBodyMemoryRouteOperandBindingFacts`.
2. The statement plan must cover existing production-active non-segment
   computed-mask memory movement routes:
   `RuntimeScalarComputedMaskStore`,
   `RuntimeScalarComputedMaskLoadStore`, `ComputedMaskUnitLoadStore`,
   `ComputedMaskStridedStore`, `ComputedMaskStridedLoadUnitStore`,
   `ComputedMaskIndexedGatherLoadUnitStore`, and
   `ComputedMaskIndexedScatterStoreUnitLoad`.
3. The plan must carry source operation provenance, mask producer/source facts,
   compare mask creation, callee/intrinsic names, operands, results, full-chunk
   `setvl`, loop/VL placement, masked load/store calls, runtime-scalar
   store/load-store source handling, inactive-lane passthrough or output
   preservation where in scope, address expressions, stride/index expressions,
   runtime count, and fail-closed dependencies needed by the included route
   sequence.
4. `RVVEmitCRouteProvider` may still instantiate
   `TCRVEmitCLowerableRoute`, add neutral route-level metadata, and attach the
   returned statements, but it must not locally recreate the included
   computed-mask memory statement sequence from operation names, memory forms,
   ABI strings, intrinsic mirrors, or mask-source mirrors.
5. Missing or stale statement-plan dependencies must fail closed before common
   EmitC materialization with targeted diagnostics.
6. Preserve emitted semantics, ABI order, operand order, VL/control behavior,
   source provenance, intrinsic spelling, route ids, and generated artifacts
   for valid included computed-mask memory selected-body routes.
7. Do not add route coverage, new memory forms, computed-mask segment2
   rewrites, reductions, contractions, dtype/LMUL clone batches,
   source-front-door routes, high-level frontend lowering, legacy i32
   authority, descriptor/direct-C/source-export paths, dashboards, broad smoke
   matrices, or common EmitC semantic logic.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` name the RVV plugin,
      EmitC route, testing specs, and previous memory statement-plan task
      context.
- [x] A production RVV planning API exists for the computed-mask memory
      statement plan and exposes an empty/default plan for unrelated route
      families and excluded computed-mask segment2 routes.
- [x] Positive C++ plugin/provider tests prove statement-plan construction and
      provider consumption for runtime-scalar computed-mask store/load-store,
      computed-mask unit load/store, strided store, strided load/unit-store,
      indexed gather/unit-store, and indexed scatter/unit-load.
- [x] At least one focused C++ negative test proves a missing or stale
      statement-plan dependency fails closed before route statement
      construction.
- [x] `RVVEmitCRouteProvider.cpp` consumes the returned computed-mask memory
      statement plan for the included routes before the older generic
      provider-local statement assembly path.
- [x] Bounded provider scan shows the included computed-mask memory statement
      sequence is no longer locally assembled by the provider beyond neutral
      route instantiation and attaching the RVV-owned plan statements.
- [x] Representative FileCheck coverage for existing explicit/pre-realized
      computed-mask memory selected-body artifacts still passes.
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
- No computed-mask segment2 statement-plan ownership in this round.
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
5. Run focused lit/FileCheck coverage for representative computed-mask memory
   selected-body artifacts and selected-boundary negative fixtures.
6. Run bounded provider scan for included computed-mask memory statement
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

Relevant previous tasks read:

- `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-base-memory-statement-plan-ownership/prd.md`
- `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-memory-route-operand-binding-surface-ownership/prd.md`
- `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-compare-select-statement-plan-ownership/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- Representative computed-mask memory selected-body fixtures under
  `test/Target/RVV`

## Definition Of Done

The bounded non-segment computed-mask memory statement sequence is owned by
RVV route planning, consumed by the selected-body provider before common EmitC
route construction, covered by focused positive and fail-closed C++ tests plus
representative FileCheck coverage, checked for authority drift, archived using
repo convention, and committed as one coherent change.

## Implementation Result

This round added `RVVSelectedBodyComputedMaskMemoryRouteStatementPlan` and
`getRVVSelectedBodyComputedMaskMemoryRouteStatementPlan(...)` as an RVV-local
planning boundary. The API consumes verified route analysis, materialization
facts, and memory operand-binding facts to produce provider-ready full-chunk
`setvl` and loop statements for the bounded non-segment computed-mask memory
subcluster. It fails closed before statement construction when the verified
computed-mask memory family plan or required ABI/materialization/provenance
facts are missing.

`RVVEmitCRouteProvider` now asks for the computed-mask memory statement plan
after base memory planning and before the older generic provider-local route
assembly path. Included non-segment computed-mask memory routes return through
the RVV-owned plan. Computed-mask segment2 routes are explicitly excluded from
this boundary and continue to the segment2 memory path.

The RVV plugin spec now records the durable computed-mask memory
statement-plan boundary, including scope, API signature, validation matrix,
wrong/correct ownership examples, and required tests.

## Validation Evidence

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- Focused lit/FileCheck from `build/test` passed 22 selected tests using:
  `lit.py -sv --filter='(runtime-scalar-cmp-masked-(store|load-store)|computed-masked-(unit-load-store|strided-store|strided-load|indexed-gather-load|indexed-scatter-store)|selected-boundary)' .`
- Bounded provider scan confirmed the new provider gate at
  `getRVVSelectedBodyComputedMaskMemoryRouteStatementPlan(...)` and the
  fail-closed included-route guard before generic provider-local statement
  assembly. Residual computed-mask branches remain in the generic path only as
  unreachable fallback residue for the included non-segment routes and for
  out-of-scope segment2/adjacent logic.
- Active-authority scan over added lines found no new C++ legacy route
  authority. The only match was this PRD/spec wording that names forbidden
  patterns as red lines.
- `git diff --check` passed after removing trailing whitespace in the new
  test fixture text.
- `cmake --build build --target check-tianchenrv -j2` passed; all 363 lit
  tests passed.

No `ssh rvv` evidence was required because this round does not claim changed
runtime behavior, ABI behavior, correctness, or performance.
