# Stage2 RVV compare/select statement-plan ownership

## Goal

Introduce an RVV-owned compare/select route statement-plan boundary and rewire
the selected-body RVV EmitC route provider to consume it for the bounded
production-active compare/select routes in this round: plain compare-select,
computed-mask select, runtime-scalar computed-mask select, and runtime-scalar
dual compare-mask-and-select.

The provider remains the owner that instantiates `TCRVEmitCLowerableRoute`,
adds neutral route headers, type mappings, ABI mappings, and selected-boundary
source provenance. The provider must not locally recreate the included
compare/select statement sequence from operation names, memory-form branches,
ABI strings, or intrinsic mirrors once the RVV-owned plan is available.

## Direction Source

- Direction title: `Stage2 RVV compare/select statement-plan ownership`.
- Module owner: RVV plugin-local selected-body route statement-plan boundary
  for compare/select routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `c094bf2b rvv: own elementwise arithmetic statement plans`.
- `.trellis/.current-task` was absent, so this task was created from the
  Direction Brief before source edits.

## What I Already Know

- The stable RVV authority chain is selected `tcrv.exec` RVV variant -> typed
  low-level `tcrv_rvv` body -> RVV-owned legality/materialization/operand
  binding/route planning -> provider-built `TCRVEmitCLowerableRoute` -> common
  EmitC materialization.
- Commit `c094bf2b` added an RVV-owned elementwise arithmetic statement-plan
  boundary and rewired the provider to attach that plan before the older
  generic provider-local statement assembly path.
- Current code already has compare/select route-family provider plans,
  materialization facts, and elementwise/select operand-binding facts for plain
  compare-select, computed-mask select, runtime-scalar computed-mask select,
  and runtime-scalar dual compare-mask-and-select.
- `RVVEmitCRouteProvider.cpp` still assembles the included compare/select
  route statement sequence locally via `addLoopStep`, operation/memory-form
  branches, direct callee selection, and hand-built operands/results.
- Representative selected-body fixtures already exist under `test/Target/RVV`
  for explicit and pre-realized compare/select artifacts, including computed
  mask and runtime-scalar dual routes.

## Requirements

1. Add an RVV planning API for a compare/select route statement plan. The plan
   must be derived from verified `RVVSelectedBodyRouteAnalysis`,
   `RVVSelectedBodyRouteMaterializationFacts`, and
   `RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts`.
2. The statement plan must cover existing production-active plain
   compare-select, computed-mask select, runtime-scalar computed-mask select,
   and runtime-scalar dual compare-mask-and-select routes.
3. The plan must carry source operation provenance, callee/intrinsic names,
   operands, results, full-chunk `setvl`, loop/VL placement, compare and
   optional secondary-compare/mask-and steps, select/merge steps, load/store
   steps, and fail-closed dependencies needed by the included route sequence.
4. `RVVEmitCRouteProvider` may still instantiate
   `TCRVEmitCLowerableRoute`, add neutral route-level metadata, and attach the
   returned statements, but it must not locally recreate the included
   compare/select statement sequence from operation names, memory forms, ABI
   strings, or intrinsic mirrors.
5. Missing or stale statement-plan dependencies must fail closed before common
   EmitC materialization with targeted diagnostics.
6. Preserve emitted semantics, ABI order, operand order, VL/control behavior,
   source provenance, intrinsic spelling, route ids, and generated artifacts
   for valid included compare/select selected-body routes.
7. Do not add route coverage, new operations, dtype/LMUL clone batches,
   reductions, memory-route rewrites, source-front-door routes, high-level
   frontend lowering, legacy i32 authority, descriptor/direct-C/source-export
   paths, dashboards, broad smoke matrices, or common EmitC semantic logic.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` name the RVV plugin,
      EmitC route, testing specs, and previous elementwise arithmetic
      statement-plan task context.
- [x] A production RVV planning API exists for the compare/select statement
      plan and exposes an empty/default plan for unrelated route families.
- [x] Positive C++ plugin/provider tests prove statement-plan construction and
      provider consumption for plain compare-select, computed-mask select,
      runtime-scalar computed-mask select, and runtime-scalar dual
      compare-mask-and-select.
- [x] At least one focused C++ negative test proves a missing or stale
      statement-plan dependency fails closed before route statement
      construction.
- [x] `RVVEmitCRouteProvider.cpp` consumes the returned compare/select
      statement plan for the included routes before the older generic
      provider-local statement assembly path.
- [x] Bounded provider scan shows the included compare/select statement
      sequence is no longer locally assembled by the provider beyond neutral
      route instantiation and attaching the RVV-owned plan statements.
- [x] Representative FileCheck coverage for existing explicit and pre-realized
      compare/select selected-body artifacts still passes.
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

- No route-family expansion and no new route operation coverage.
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
5. Run focused lit/FileCheck coverage for representative compare/select
   selected-body artifacts and selected-boundary negative fixtures.
6. Run bounded provider scan for included compare/select statement assembly
   residue in `RVVEmitCRouteProvider.cpp`.
7. Run active-authority scan over touched RVV planning/provider/test/spec
   files.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Relevant previous task read:

- `.trellis/tasks/archive/2026-05/05-24-05-24-stage2-rvv-elementwise-arithmetic-statement-plan-ownership/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- Representative compare/select selected-body fixtures under `test/Target/RVV`

## Definition Of Done

The bounded compare/select statement sequence is owned by RVV route planning,
consumed by the selected-body provider before common EmitC route construction,
covered by focused positive and fail-closed C++ tests plus representative
FileCheck coverage, checked for authority drift, archived using repo
convention, and committed as one coherent change.

## Implementation Summary

- Added `RVVSelectedBodyCompareSelectRouteStatementPlan` and
  `getRVVSelectedBodyCompareSelectRouteStatementPlan()` as the RVV-local
  statement-plan boundary for plain compare-select, computed-mask select,
  runtime-scalar computed-mask select, and runtime-scalar dual
  compare-mask-and-select.
- Implemented statement-plan construction from verified route-family plans,
  materialization facts, and elementwise/select operand-binding facts. The plan
  owns full-chunk `setvl`, loop `setvl`, load/splat steps, compare and optional
  secondary-compare/mask-and steps, select, and store.
- Rewired `RVVEmitCRouteProvider.cpp` to request the compare/select statement
  plan after provider verification, materialization facts, and
  elementwise/select operand-binding facts. When the plan is present, the
  provider attaches its pre-loop step and loop and returns before the older
  generic provider-local statement assembly path.
- Added a provider fail-closed guard so any compare/select route that is not
  consumed by the RVV-owned statement plan cannot fall through into generic
  provider-local statement assembly.
- Added focused C++ coverage for positive statement-plan construction and
  provider route consumption across all included compare/select routes;
  unrelated empty-plan behavior; and a stale computed-mask select family-plan
  dependency diagnostic before statement construction.
- Documented the durable compare/select statement-plan boundary in
  `.trellis/spec/extension-plugins/rvv-plugin.md`.

## Verification

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-stage2-rvv-compare-select-statement-plan-ownership`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- From `build/test`:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter='(selected-body-artifact-(cmp-select|computed-mask-select|runtime-scalar-(cmp-select|dual-cmp-mask-and-select))|selected-boundary)' .`
  passed 15/15 focused tests.
- Bounded provider scan:
  `rg -n "getRVVSelectedBodyCompareSelectRouteStatementPlan|plansCompareSelectRoute|compare/select provider requires|isRVVSelectedBodyCompareSelectRoute|RuntimeScalarDualCompareMaskAndSelect|ComputedMaskSelect|RuntimeScalarCompareSelect|addLoopStep\\(|compareIntrinsic|maskAndIntrinsic|vmerge" lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  showed the new statement-plan call and fail-closed guard before the older
  generic `addLoopStep` assembly. Remaining compare/select operation and
  memory-form references are in provider prelude/fallback code after the
  statement-plan gate and are no longer reachable for production-active
  included compare/select routes.
- Active-authority scan over touched RVV planning/provider/test diff:
  `git diff --unified=0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp test/Plugin/RVVExtensionPluginTest.cpp | rg -n '^\\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|descriptor|source-front-door|direct-C|source-export|provider_supported_mirror)'`
  found no matches.
- Broader added-line scan also found only expected spec/checklist text,
  MLIR fixture `status` attrs, and exact RVV intrinsic leaf assertions in C++
  tests; these are not route authority.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` (363/363)

## Self-Repair Notes

- Added an explicit provider fail-closed guard after the statement-plan query
  so compare/select routes cannot silently fall through to older provider-local
  statement assembly if the RVV-owned plan is missing or non-consuming.
- `clang-format` is not installed in this environment; formatting was checked
  by build plus `git diff --check`.

## Spec Update Judgment

`.trellis/spec/extension-plugins/rvv-plugin.md` was updated because this task
introduced a durable planning/provider API and boundary contract. The update
records scope, signature, contracts, fail-closed behavior, good/base/bad cases,
required tests, and wrong-vs-correct provider ownership shape.
