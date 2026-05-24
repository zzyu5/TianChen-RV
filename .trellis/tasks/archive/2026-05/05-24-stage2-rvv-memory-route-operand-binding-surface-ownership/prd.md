# Stage2 RVV memory route operand-binding surface ownership

## Goal

Introduce an explicit RVV-owned operand-binding facts boundary for the mature
selected-body memory route cluster. `RVVEmitCRouteProvider` should consume this
boundary when mapping verified memory logical operands to materialized runtime
ABI uses before building `TCRVEmitCLowerableRoute` statements, instead of
locally enumerating memory logical operand names and materialized-use strings in
the central provider prelude.

## Direction Source

- Direction title: `Stage2 RVV memory route operand-binding surface ownership`.
- Module owner: RVV plugin-local selected-body route operand-binding surface
  for base unit/strided/indexed memory movement, computed-mask memory movement,
  runtime-scalar computed-mask load/store, and segment2 memory movement.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `0b478072 rvv: own elementwise select operand binding facts`.
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
- The previous completed task added
  `RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts` and rewired the
  provider to consume it for ordinary elementwise, scalar-broadcast
  elementwise, plain compare-select, computed-mask select, and runtime-scalar
  computed-mask select routes.
- Current planning APIs already have memory route-family plans and owners for
  base memory movement, computed-mask memory, and plain segment2 memory.
- Current provider construction still owns a long memory binding prelude with
  local `bindOperand` and `requireOperandUse` calls for strided, indexed,
  masked, computed-mask, runtime-scalar computed-mask, and segment2 memory
  routes.

## Requirements

1. Add a named RVV planning/provider memory operand-binding facts boundary for
   the selected-body memory route-family cluster.
2. The boundary must consume the verified `RouteOperandBindingPlan` and expose
   bound runtime ABI parameters for source/destination windows, runtime AVL,
   strides, indices, masks, payloads, old-destination passthroughs, segment
   fields, and computed-mask compare operands required by the included memory
   route families.
3. The boundary must preserve route operand binding plan id/order, runtime ABI
   roles, logical memory roles, header mirror uses, load/store/address/mask
   materialized uses, and segment field roles.
4. The boundary must fail closed for missing memory family plans, stale
   materialized uses, stale segment or computed-mask markers, and missing
   runtime AVL/header closure before statement construction.
5. Rewire the production `RVVEmitCRouteProvider.cpp` memory binding branches to
   consume the new facts instead of locally enumerating memory logical
   operands and materialized uses.
6. Preserve emitted semantics, route ids, ABI order, target leaf/header facts,
   operand order, diagnostics, and generated artifacts. Do not add route
   coverage or new operations.
7. Keep the boundary RVV-local. Do not move RVV semantics into common
   EmitC/export or target metadata.

## Acceptance Criteria

- [x] PRD and context files reference the RVV plugin and EmitC route specs plus
      the previous elementwise/select operand-binding task.
- [x] A named memory operand-binding facts boundary exists in the RVV planning
      API.
- [x] The production provider calls the top-level provider verifier, obtains
      materialization facts, obtains elementwise/select operand-binding facts,
      obtains memory operand-binding facts, and consumes memory facts for the
      included memory binding branches.
- [x] The provider no longer locally enumerates base memory movement,
      computed-mask memory, runtime-scalar computed-mask memory, or segment2
      memory logical operand/materialized-use binding strings for the included
      cluster.
- [x] Focused C++ plugin/provider tests cover representative strided memory,
      indexed memory, masked/base memory, computed-mask memory, runtime-scalar
      computed-mask store/load-store, and segment2 memory binding facts, plus
      at least one missing/stale materialized-use diagnostic.
- [x] Representative FileCheck coverage for existing explicit/pre-realized
      memory artifact paths affected by the refactor still passes.
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

- No route coverage expansion, new operations, dtype/LMUL clone batches,
  elementwise/select remigration beyond direct integration needs,
  reduction/contraction/widening binding migration beyond direct integration
  needs, new verifier registries, source-front-door routes, high-level frontend
  lowering, legacy i32 authority, descriptor/direct-C/source-export paths,
  artifact dashboards, broad smoke matrices, or helper-only changes without
  production provider consumption.
- No emitted target sequence, runtime ABI, operand order, correctness, runtime,
  or performance claim changes.
- No migration of RVV semantics into common EmitC/export.
- No rewrite of unrelated statement emission.

## Validation Plan

1. Validate and start this Trellis task.
2. Build focused C++ plugin test target.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build `tcrv-opt` and `tcrv-translate` if not already current.
5. Run focused lit/FileCheck coverage for representative memory routes and
   selected-boundary negative coverage.
6. Run active-authority scan over touched RVV planning/provider/test files.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant previous task read:

- `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-elementwise-select-operand-binding-surface-ownership/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- representative fixtures under `test/Target/RVV`

## Implementation Summary

- Added `RVVSelectedBodyMemoryRouteOperandBindingFacts` and
  `getRVVSelectedBodyMemoryRouteOperandBindingFacts()` to the RVV planning API.
- Implemented the facts boundary in `RVVEmitCRoutePlanning.cpp` for the
  included memory cluster: base unit/strided/indexed movement, masked base
  movement, runtime-scalar computed-mask store/load-store, computed-mask
  strided/indexed/segment2 movement, and plain segment2 movement.
- Rewired `RVVEmitCRouteProvider.cpp` so the selected-body provider consumes
  memory operand-binding facts after provider-plan verification,
  materialization facts, and elementwise/select facts, instead of rebuilding
  the included memory logical operand/materialized-use table in the central
  provider prelude.
- Extended `test/Plugin/RVVExtensionPluginTest.cpp` with representative facts
  coverage and one stale materialized-use diagnostic.
- Added the durable memory operand-binding facts boundary contract to
  `.trellis/spec/extension-plugins/rvv-plugin.md`.

## Verification

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-stage2-rvv-memory-route-operand-binding-surface-ownership`
- [x] `git diff --check`
- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] Focused lit/FileCheck from `build/test`: 36/36 representative memory,
      computed-mask memory, segment2, and selected-boundary negative tests
      passed.
- [x] Added-line active-authority scan over touched RVV planning/provider/test
      files found no new legacy i32/source-front-door/descriptor/direct-C/
      source-export or mirror-only route authority terms.
- [x] `cmake --build build --target check-tianchenrv -j2` passed 363/363.
- [x] `ssh rvv` evidence was not run or required because this refactor did not
      claim emitted target sequence, runtime ABI, operands, correctness,
      runtime, or performance changes.

## Self-Repair Notes

- A direct focused lit invocation against source test paths initially failed
  because the source `lit.cfg.py` was loaded without the build-tree
  `tianchenrv_obj_root` site configuration. The corrected validation ran from
  `build/test` with the same memory-route filter and passed 36/36.
- After the first provider rewrite, several C++ `else if` chains had tab-based
  indentation noise. The indentation was repaired without changing binding
  semantics, then `git diff --check`, focused build/unit/lit, and full
  `check-tianchenrv` were rerun successfully.

## Definition Of Done

The selected-body RVV provider consumes a named RVV-owned memory
operand-binding facts boundary after provider-plan verification and
materialization facts selection; central provider code no longer owns the
included memory cluster's logical operand/materialized-use decision table;
representative route families and fail-closed diagnostics remain covered;
focused and full checks pass; the task is finished/archived using repo
convention; and one coherent commit records the work.
