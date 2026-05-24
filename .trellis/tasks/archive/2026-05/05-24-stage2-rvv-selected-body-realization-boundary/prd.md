# Stage2 RVV selected-body realization boundary

## Goal

Make selected-body realization an explicit RVV plugin-owned compiler boundary
for the statement-plan-backed elementwise arithmetic and compare/select
selected-body cluster. A selected `tcrv.exec` RVV variant containing a
pre-realized typed `tcrv_rvv` body must be realized into concrete typed
`tcrv_rvv` `setvl`/`with_vl`/dataflow structure before RVV route planning,
statement-plan selection, provider route construction, and common EmitC
materialization.

This is a boundary cleanup and production-path rewire, not new RVV coverage.
The provider and common EmitC must keep consuming already-realized body facts
and RVV-owned statement plans; they must not invent compute, dtype, policy,
schedule, operand shape, or runtime AVL/VL behavior.

## Direction Source

- Direction title: `Stage2 RVV selected-body realization boundary`.
- Module owner: RVV plugin-local selected-body realization boundary for the
  statement-plan-backed elementwise/compare-select cluster.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `3e15088c rvv: close migrated statement plan boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- The stable RVV authority chain is selected `tcrv.exec` RVV variant ->
  typed low-level `tcrv_rvv` body -> RVV plugin-owned selected-body
  realization/legality/route provider -> provider-built
  `TCRVEmitCLowerableRoute` -> common EmitC materializer.
- The previous archived task closed migrated statement-plan provider-neutral
  consumption for elementwise arithmetic, compare/select, base memory,
  computed-mask memory, segment2 memory, and computed-mask accumulation.
- Current `RVVSelectedBodyRealization.cpp` already performs realization inside
  RVV plugin code, but the production entry point is a single long dispatcher
  where elementwise/compare-select pre-realized body realization is interleaved
  with unrelated memory/math/contraction bodies.
- The affected statement-plan-backed elementwise/compare-select cluster
  includes production-active pre-realized bodies for plain binary arithmetic,
  scalar-broadcast arithmetic, strided arithmetic, masked arithmetic, plain
  compare-select, computed-mask select, runtime-scalar compare-select, and
  runtime-scalar dual compare-mask-and-select.
- `RVVEmitCRouteProvider.cpp` should continue to consume realized body facts,
  materialization facts, operand-binding facts, and migrated statement plans.
  It should not be changed to synthesize missing realization structure.
- Common EmitC/export remains neutral and must not choose RVV semantics,
  intrinsics, dtype/config, schedule, ABI roles, or route support.

## Requirements

1. Add an explicit RVV-owned realization boundary for the elementwise
   arithmetic and compare/select pre-realized selected-body cluster.
2. The production `realizePreRealizedRVVSelectedBody(...)` path must call this
   boundary before route planning/provider construction. Cluster bodies must be
   handled by the boundary, not by ad hoc provider/common logic.
3. The boundary must consume only pre-realized typed body structure,
   target/variant capability metadata, runtime ABI SSA values, policy/config,
   and existing RVV runtime AVL/VL control helpers. It must emit realized
   typed `tcrv_rvv` `setvl`/`with_vl`/load/splat/compare/mask/select/binary/
   masked-binary/store structure.
4. The boundary must fail closed with targeted diagnostics when an
   elementwise/compare-select pre-realized body is missing required runtime ABI
   roles, has unsupported config/policy/memory form, is mixed with an already
   realized body, or cannot derive a runtime AVL/VL plan where required.
5. Realized bodies must feed the existing RVV route analysis, materialization
   facts, operand-binding facts, migrated statement-plan boundary, provider
   route construction, and common EmitC path without changing emitted target
   sequence, ABI order, operand order, VL/control behavior, route ids,
   intrinsic spelling, or artifact metadata for valid fixtures.
6. Provider/common EmitC code must not gain semantic realization logic,
   dtype/config inference, operation selection, schedule construction, or
   pre-realized-body fallback behavior.
7. Do not add route coverage, arithmetic/select operation kinds, memory,
   reduction, contraction, dtype/LMUL clone batches, high-level frontend
   lowering, source-front-door positive routes, legacy i32 route authority,
   descriptor/direct-C/source-export paths, dashboards, broad smoke matrices,
   runtime ABI changes, dispatch/fallback changes, runtime `n`/AVL changes, or
   performance claims.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` name the RVV plugin,
      selected-body realization, EmitC route, testing specs, and previous
      statement-plan closure context.
- [x] A named RVV plugin-owned realization boundary exists for the
      elementwise/compare-select cluster and returns empty/not-applicable for
      unrelated pre-realized families without changing them.
- [x] `realizePreRealizedRVVSelectedBody(...)` consumes that boundary on the
      production path before unrelated realization fallbacks.
- [x] Focused C++ tests prove positive boundary realization for at least one
      elementwise arithmetic pre-realized body and one compare/select
      pre-realized body, and then prove the realized body continues into the
      existing route/provider statement-plan path.
- [x] Focused C++ negative coverage proves at least one missing or unsupported
      elementwise/compare-select realization fact fails closed before route
      construction.
- [x] Representative lit/FileCheck coverage for affected pre-realized and
      explicit selected-body artifacts remains passing.
- [x] A bounded scan over touched RVV realization/planning/provider files
      shows provider/common code does not perform semantic realization or infer
      compute/dtype/policy/schedule/body shape.
- [x] Active-authority scan over touched files finds no newly introduced
      legacy `RVVI32M1`, `rvv-i32m1`, positive finite `tcrv_rvv.i32_*`,
      descriptor/source-front-door/direct-C/source-export, or mirror-only
      route authority.
- [x] `git diff --check` passes.
- [x] Focused plugin/pass tests and `check-tianchenrv` pass, or an exact
      documented blocker is recorded.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Technical Approach

Introduce a small cluster boundary in `RVVSelectedBodyRealization` rather than
moving logic into route planning or provider code. The boundary should:

- classify elementwise arithmetic and compare/select pre-realized body ops;
- reuse the existing validation helpers and RVV runtime AVL/VL control helper;
- build the same realized `tcrv_rvv` structure that valid fixtures already
  depend on;
- erase the consumed pre-realized body only after the realized body has been
  built;
- return a not-applicable result for unrelated pre-realized families so memory,
  math, contraction, conversion, and residual bodies keep their current paths.

The provider should remain unchanged unless tests expose a required integration
hole. Validation should focus on the module behavior touched here: realization
boundary production use and downstream route/provider consumption for the
chosen cluster.

## Decision (ADR-lite)

Context: The previous task finished provider-neutral migrated statement-plan
consumption. The next bottleneck is not another statement-plan family, but the
pre-realized selected-body realization step itself being an explicit RVV-owned
compiler boundary before planning/provider construction.

Decision: Implement a bounded elementwise/compare-select realization boundary
inside `RVVSelectedBodyRealization`, wire the production dispatcher through it,
and test that realized bodies still feed existing route planning/provider
statement plans.

Consequences: This keeps semantics in RVV plugin code, preserves current
valid artifacts, and avoids broad memory/math/contraction churn. Unrelated
pre-realized families remain as continuation work unless the implementation
shows the boundary can include them safely without expanding scope.

## Out Of Scope

- No new RVV operation coverage or dtype/LMUL expansion.
- No high-level Linalg/Vector/StableHLO frontend work.
- No source-front-door positive routes, descriptor-driven computation, direct
  C/source-export route, or legacy i32 route authority.
- No runtime ABI, dispatch/fallback, emitted target sequence, correctness, or
  performance claim changes.
- No ssh rvv evidence unless emitted target sequence, runtime correctness, or
  performance is changed or claimed.

## Validation Plan

1. Validate and start this Trellis task.
2. Build the focused C++ plugin test target.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run representative affected lit/FileCheck tests for pre-realized and
   explicit elementwise/compare-select artifacts.
5. Run bounded provider/common scan over touched RVV realization/planning/
   provider files.
6. Run active-authority scan over touched files.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant previous task read:

- `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-statement-plan-provider-neutral-closure/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- representative `test/Target/RVV/pre-realized-selected-body-artifact-*.mlir`
- representative `test/Target/RVV/explicit-selected-body-artifact-*.mlir`

## Definition Of Done

The elementwise/compare-select pre-realized selected-body cluster is realized
through an explicit RVV plugin-owned boundary on the production path, realized
bodies continue into existing RVV planning/provider statement-plan route
construction, focused positive and fail-closed tests pass, authority scans show
no drift, Trellis task metadata is truthful, and the completed work is
committed as one coherent change.

## Implementation Result

- Added `RVVElementwiseCompareSelectRealizationResult` and
  `realizePreRealizedRVVElementwiseCompareSelectCluster(...)` as the named
  RVV plugin-owned realization boundary for the statement-plan-backed
  elementwise arithmetic and compare/select pre-realized selected-body cluster.
- The boundary handles plain/scalar-broadcast/strided binary arithmetic,
  masked binary arithmetic, plain compare-select, computed-mask select,
  runtime-scalar compare-select, and runtime-scalar dual compare-mask-and-
  select by consuming existing typed pre-realized body/config/runtime facts and
  creating realized `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, and typed dataflow
  ops.
- `realizePreRealizedRVVSelectedBody(...)` now calls the cluster boundary
  before the unrelated realization fallbacks. The provider and common EmitC
  code were not changed.
- Focused C++ coverage directly exercises the new boundary for an arithmetic
  body and a compare/select body, verifies unrelated reduce bodies return
  not-applicable, verifies mixed pre-realized/realized structure fails closed,
  and checks the realized bodies still reach the existing provider route path.

## Validation Result

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-stage2-rvv-selected-body-realization-boundary`
  passed, then the task was started.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed with
  `RVV extension plugin smoke test passed`.
- Focused lit/FileCheck command passed for four representative fixtures:
  `pre-realized-selected-body-artifact-add.mlir`,
  `pre-realized-selected-body-artifact-cmp-select.mlir`,
  `explicit-selected-body-artifact-add.mlir`, and
  `explicit-selected-body-artifact-cmp-select.mlir`.
- Provider/common semantic-realization scan:
  `rg -n "PreRealized|pre_realized|typed_.*pre_realized|createRealized|realizePreRealized|selected-body realization" lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h lib/Conversion include/TianChenRV/Conversion`
  found no provider/common pre-realized realization logic; the only match was
  the existing RVV planning runtime AVL/VL helper.
- Active-authority diff scan over touched files found no newly introduced
  legacy `RVVI32M1`, `rvv-i32m1`, positive finite `tcrv_rvv.i32_*`,
  descriptor/source-front-door/direct-C/source-export, or mirror-only route
  authority. The only matches were fixture `status` mirror attributes.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed: 363/363 lit
  tests passed.
- No `ssh rvv` evidence was collected because this task did not change or
  claim emitted target sequence, runtime correctness, ABI behavior, or
  performance.
