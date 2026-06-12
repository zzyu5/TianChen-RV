# Stage2 RVV elementwise/select operand-binding surface ownership

## Goal

Introduce an explicit RVV-owned operand-binding facts boundary for the mature
elementwise/select selected-body route cluster. `RVVEmitCRouteProvider` should
consume this boundary when mapping verified logical operands to materialized
runtime ABI uses before building `TCRVEmitCLowerableRoute` statements, instead
of locally enumerating elementwise/select logical operand names and
materialized-use strings in the central provider prelude.

## Direction Source

- Direction title: `Stage2 RVV elementwise/select operand-binding surface
  ownership`.
- Module owner: RVV plugin-local selected-body route operand-binding surface for
  ordinary elementwise arithmetic, scalar-broadcast elementwise, plain
  compare-select, computed-mask select, and runtime-scalar computed-mask select.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `b2991a5e rvv: own selected-body materialization facts`.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  Direction Brief before source edits.

## What I Already Know

- Specs require RVV route authority to flow from selected `tcrv.exec` variant
  through typed low-level `tcrv_rvv` body/config/runtime facts, RVV-owned
  legality/realization/provider, provider-built `TCRVEmitCLowerableRoute`,
  common EmitC materialization, and target artifact/evidence.
- Common EmitC/export must stay neutral. RVV ABI mapping, logical operands,
  materialized uses, route payload, legality, selected-body realization, and
  fail-closed diagnostics remain RVV-plugin-owned.
- The previous task added `RVVSelectedBodyRouteMaterializationFacts` and made
  selected-body route construction consume verified materialization facts for
  type/header/intrinsic/shape choices after the top-level family provider
  verifier.
- Current provider construction still owns a long operation-specific binding
  prelude with bound `RuntimeABIParameter` defaults, local `bindOperand` and
  `requireOperandUse` lambdas, and hard-coded logical operand/materialized-use
  strings for the elementwise/select cluster.
- The existing elementwise/select owner registry already groups ordinary
  elementwise arithmetic, scalar-broadcast elementwise, plain compare-select,
  and computed-mask select provider verifiers.

## Requirements

1. Add a named RVV planning/provider operand-binding facts boundary for the
   elementwise/select route-family cluster.
2. The boundary must consume the verified `RouteOperandBindingPlan` and expose
   bound runtime ABI parameters for the logical operands needed by:
   ordinary elementwise arithmetic, scalar-broadcast elementwise, plain
   compare-select, computed-mask select, and runtime-scalar computed-mask
   select.
3. The boundary must preserve plan id/order, runtime ABI roles, setvl/runtime
   count binding, load/compute/select/store materialized uses, header mirror
   uses, and the selected single/dual runtime-scalar computed-mask select shape.
4. The boundary must fail closed for missing or stale materialized uses and for
   missing/stale elementwise/select family plans that are required before
   binding.
5. Rewire the production `RVVEmitCRouteProvider.cpp` elementwise/select binding
   branches to consume the new facts instead of locally enumerating that
   cluster's logical operands and materialized uses.
6. Preserve emitted semantics, route ids, ABI order, target leaf/header facts,
   operand order, diagnostics, and generated artifacts. Do not add route
   coverage or new operations.
7. Keep the boundary RVV-local. Do not move RVV semantics into common
   EmitC/export or target metadata.

## Acceptance Criteria

- [x] PRD and context files reference the RVV plugin and EmitC route specs plus
      the previous materialization-facts task.
- [x] A named elementwise/select operand-binding facts boundary exists in the
      RVV planning API.
- [x] The production provider calls the top-level provider verifier, obtains
      materialization facts, obtains elementwise/select operand-binding facts,
      and consumes those facts for the elementwise/select binding branches.
- [x] The provider no longer locally enumerates ordinary elementwise,
      scalar-broadcast, plain compare-select, computed-mask select, or
      runtime-scalar computed-mask select logical operand/materialized-use
      binding strings.
- [x] Focused C++ plugin/provider tests cover representative ordinary
      elementwise, scalar-broadcast elementwise, plain compare-select,
      computed-mask select, and runtime-scalar computed-mask select binding
      facts, plus at least one missing/stale materialized-use diagnostic.
- [x] Representative FileCheck coverage for existing explicit/pre-realized
      elementwise/select artifact paths affected by the refactor still passes.
- [x] Selected-boundary negative coverage still passes.
- [x] Active-authority scan over touched RVV planning/provider/test files finds
      no newly introduced legacy `RVVI32M1`, `rvv-i32m1`, finite positive
      `tcrv_rvv.i32_*`, descriptor/source-front-door/direct-C/source-export, or
      mirror-only route authority.
- [x] Focused build/tests, `git diff --check`, and `check-tianchenrv` pass, or
      an exact blocker is documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Non-Goals

- No route coverage expansion, new operations, dtype/LMUL clone batches, memory
  or math operand-binding migration beyond direct integration needs, new
  verifier registries, source-front-door routes, high-level frontend lowering,
  legacy i32 authority, descriptor/direct-C/source-export paths, artifact
  dashboards, broad smoke matrices, or helper-only changes without production
  provider consumption.
- No emitted target sequence, runtime ABI, operand order, correctness, runtime,
  or performance claim changes.
- No migration of RVV semantics into common EmitC/export.
- No rewrite of unrelated statement emission.

## Validation Plan

1. Validate and start this Trellis task.
2. Build focused C++ plugin test target.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build `tcrv-opt` and `tcrv-translate` if not already current.
5. Run focused lit/FileCheck coverage for representative elementwise/select
   routes: explicit add, scalar-broadcast add, cmp-select, computed-mask
   select, runtime-scalar cmp-select, and selected-boundary negative.
6. Run active-authority scan over touched RVV planning/provider/test files.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant previous task read:

- `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-selected-body-route-materialization-facts-ownership/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- representative fixtures under `test/Target/RVV`

## Definition Of Done

The selected-body RVV provider consumes a named RVV-owned
elementwise/select operand-binding facts boundary after provider-plan
verification and materialization facts selection; central provider code no
longer owns that cluster's logical operand/materialized-use decision table;
representative route families and fail-closed diagnostics remain covered;
focused and full checks pass; the task is finished/archived using repo
convention; and one coherent commit records the work.

## Implementation Summary

- Added `RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts` and
  `getRVVSelectedBodyElementwiseSelectRouteOperandBindingFacts()` as the
  RVV-owned operand-binding facts boundary for the mature elementwise/select
  cluster.
- Rewired `RVVEmitCRouteProvider.cpp` so selected-body route construction
  consumes top-level provider-plan verification, route materialization facts,
  and then elementwise/select operand-binding facts before statement
  construction.
- Moved ordinary elementwise, scalar-broadcast elementwise, plain
  compare-select, computed-mask select, and runtime-scalar computed-mask select
  logical operand/materialized-use validation into the planning boundary.
- Added C++ plugin/provider coverage for representative binding facts and a
  stale `false_value` materialized-use diagnostic.
- Updated the RVV plugin spec with the durable elementwise/select
  operand-binding facts boundary contract.

## Verification

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-stage2-rvv-elementwise-select-operand-binding-surface-ownership`
- [OK] `git diff --check`
- [OK] Added-line active-authority scan over touched RVV planning/provider/test
  files found no new legacy i32/source-front-door/descriptor/direct-C/
  source-export or mirror-only authority terms.
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] Focused lit/FileCheck from `build/test`: 14/14 representative
  elementwise/select artifact and selected-boundary negative tests passed.
- [OK] `cmake --build build --target check-tianchenrv -j2` (363/363)

## Self-Repair Notes

- A direct lit invocation from repo root failed because this build's
  `lit.site.cfg.py` loads `../../test/lit.cfg.py` relative to the invocation
  directory. The same filter passed when rerun from `build/test`, matching the
  build tree configuration.
- No `ssh rvv` evidence was run because this task changed provider-local
  binding ownership only and did not claim emitted target sequence, ABI,
  operands, correctness, runtime, or performance changes.
