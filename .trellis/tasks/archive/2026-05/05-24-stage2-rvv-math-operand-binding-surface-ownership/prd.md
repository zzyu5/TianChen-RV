# Stage2 RVV math operand-binding surface ownership

## Goal

Introduce an explicit RVV-owned operand-binding facts boundary for the remaining
selected-body math route cluster whose production provider branches still bind
logical operands and materialized uses locally. `RVVEmitCRouteProvider` should
consume this boundary for reduction, accumulation/contraction, computed-mask
accumulation producers, representative widening MAcc/dot-reduction, and
widening conversion route families before building `TCRVEmitCLowerableRoute`
statements.

## Direction Source

- Direction title: `Stage2 RVV reduction/contraction/widening operand-binding surface ownership`.
- Module owner: RVV plugin-local selected-body route operand-binding surface for
  reductions, accumulation/contraction, computed-mask accumulation producers,
  widening MAcc/dot-reduction, and widening conversion routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `f3212e4d rvv: own memory operand binding facts`.
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
- Commit `0b478072` completed elementwise/select operand-binding facts, and
  commit `f3212e4d` completed memory operand-binding facts.
- The previous memory task added
  `RVVSelectedBodyMemoryRouteOperandBindingFacts` and rewired provider memory
  route branches to consume it for base, strided, indexed, computed-mask,
  runtime-scalar computed-mask, and segment2 memory routes.
- The remaining production provider math cluster is expected to still contain
  local `bindOperand` and `requireOperandUse` decisions around route families
  such as `ReduceAdd`, `MAcc`, computed-mask MAcc, widening MAcc, widening
  conversion, and widening dot-reduction.

## Requirements

1. Add or regularize a named RVV planning/provider math operand-binding facts
   boundary for the included selected-body math route-family cluster.
2. The boundary must consume the verified `RouteOperandBindingPlan` and
   selected-body materialization facts before statement construction.
3. The boundary must expose bound runtime ABI parameters and materialized uses
   for source, accumulator, seed, result, runtime AVL/header closure, mask,
   compare-producer, stride, widened source/accumulator/result, and
   width/config mirror roles required by the included route families.
4. The boundary must preserve route operand binding plan id/order, ABI order,
   source/accumulator/result roles, mask and compare-producer roles, scalar seed
   roles, stride roles where applicable, materialized compute uses, and
   widening config/header mirror uses.
5. The boundary must fail closed for missing math family plans, missing logical
   operands, missing or stale materialized uses, stale computed-mask markers,
   stale widening markers, and missing runtime AVL/header closure before
   statement construction.
6. Rewire the production `RVVEmitCRouteProvider.cpp` math binding branches in
   the included cluster to consume the new facts instead of locally enumerating
   logical operands and materialized-use strings.
7. Preserve emitted semantics, route ids, ABI order, target leaf/header facts,
   operand order, diagnostics, and generated artifacts. Do not add route
   coverage or new operations.
8. Keep the boundary RVV-local. Do not move RVV semantics into common
   EmitC/export or target metadata.

## Acceptance Criteria

- [x] PRD and context files reference the RVV plugin and EmitC route specs plus
      the previous memory operand-binding task.
- [x] A named math operand-binding facts boundary exists in the RVV planning
      API.
- [x] The production provider calls the top-level provider verifier, obtains
      materialization facts, obtains elementwise/select operand-binding facts,
      obtains memory operand-binding facts, obtains math operand-binding facts,
      and consumes math facts for the included reduction/accumulation/widening
      binding branches.
- [x] The provider no longer locally enumerates the included math cluster's
      logical operands and materialized-use strings in the central provider
      prelude.
- [x] Focused C++ plugin/provider tests cover representative `ReduceAdd`,
      `MAcc`, computed-mask accumulation or computed-mask MAcc, widening MAcc
      or widening dot-reduction, and widening conversion where included.
- [x] Focused negative coverage includes at least one missing or stale
      materialized-use diagnostic for the new math binding facts boundary.
- [x] Representative FileCheck coverage for existing explicit/pre-realized math
      and widening artifact paths affected by the refactor still passes.
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
  elementwise/select or memory remigration beyond direct integration needs, new
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
5. Run focused lit/FileCheck coverage for representative reduction,
   accumulation/contraction, computed-mask accumulation, widening conversion,
   widening MAcc/dot-reduction, and selected-boundary negative routes included
   in the refactor.
6. Run active-authority scan over touched RVV planning/provider/test files.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant previous task read:

- `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-memory-route-operand-binding-surface-ownership/prd.md`

Initial implementation surface from the Direction Brief:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- representative fixtures under `test/Target/RVV`

## Definition Of Done

The selected-body RVV provider consumes a named RVV-owned math
operand-binding facts boundary after provider-plan verification,
materialization facts selection, elementwise/select facts, and memory facts;
central provider code no longer owns the included math cluster's logical
operand/materialized-use decision table; representative route families and
fail-closed diagnostics remain covered; focused and full checks pass; the task
is finished/archived using repo convention; and one coherent commit records the
work.

## Implementation Summary

- Added `RVVSelectedBodyMathRouteOperandBindingFacts` and
  `getRVVSelectedBodyMathRouteOperandBindingFacts(...)` to the RVV planning
  API.
- The new planning boundary verifies route operand binding closure, math
  family/provider plans, runtime AVL/header closure, source/accumulator/result
  roles, compare-producer roles, computed-mask accumulation producers,
  standalone-reduction roles, widening conversion relation/config mirrors,
  widening MAcc roles, widening dot-reduction payload roles, and strided
  widening dot address roles before statement construction.
- Rewired `RVVEmitCRouteProvider.cpp` so the included ReduceAdd, MAcc,
  computed-mask MAcc, runtime-scalar computed-mask MAcc, standalone reduction,
  computed-mask standalone reduction, widening MAcc, widening conversion,
  widening dot-reduction, strided widening dot-reduction, and computed-mask
  widening dot-reduction branches consume RVV-owned math operand-binding facts.
- Kept non-goal local binding paths outside this owner unchanged, including
  masked elementwise arithmetic, strided elementwise add, and runtime scalar
  splat-store handling.
- Added focused C++ coverage for the math facts boundary and one stale
  materialized-use diagnostic.
- Added a durable RVV spec section for the math operand-binding facts boundary.

## Verification

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-stage2-rvv-math-operand-binding-surface-ownership`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter='(reduce|macc|widen|selected-boundary)' /home/kingdom/phdworks/TianchenRV/build/test`
- Active-authority scan over touched RVV planning/provider/test files for
  newly introduced legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, exact `__riscv_*_i32m1`, source-front-door,
  source-artifact, descriptor, direct-C, source-export, mirror-only authority,
  and status-mirror authority strings.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`

All focused and full checks passed. No runtime, correctness, performance,
target sequence, or ABI-change claim was made, so `ssh rvv` evidence was not
required for this refactor.

## Self-Repair Notes

No check failure required self-repair after the completed implementation. The
provider refactor was verified with focused math/widening lit coverage and the
full `check-tianchenrv` target before task archival.

## Continuation Point

No continuation remains for this task's included math operand-binding cluster.
The next owner, if desired, should be a separate task for the still-local
non-goal binding paths outside this PRD, such as masked elementwise arithmetic,
strided elementwise add, or runtime scalar splat-store binding.
