# Stage2 RVV reduction and accumulation statement-plan owner boundary extraction

## Goal

Move provider-ready statement construction for the migrated RVV
reduction/accumulation families out of monolithic `RVVEmitCRoutePlanning.cpp`
and into the RVV EmitC statement-plan owner boundary. The target families for
this round are ordinary reduction, standalone reduction, plain/scalar-broadcast
MAcc, and computed-mask accumulation MAcc. Route planning remains responsible
for neutral route analysis, typed/materialization facts, family provider plans,
route-control provider plans, operand-binding facts, validation, diagnostics,
and shared statement-plan structs. `RVVEmitCRouteProvider` remains a neutral
`TCRVEmitCLowerableRoute` assembler.

## Direction Source

- Direction title: `Continue: Stage2 RVV reduction and accumulation
  statement-plan owner boundary extraction`.
- Module owner: RVV EmitC statement-plan owner boundary for migrated
  reduction/accumulation route families.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `28940b9f rvv: move memory statements to owners`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The archived predecessor
  `.trellis/tasks/archive/2026-05/05-30-stage2-rvv-memory-statement-plan-owner-boundary`
  moved base memory, computed-mask memory, and segment2 memory statement-plan
  construction into owner-local implementation while leaving planning with
  neutral facts and shared structs.
- `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp` is the current
  template for a large owner-local statement-plan implementation file.
- `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp` already owns the
  migrated statement-plan owner registry, exact-one owner selection, and
  migrated-family wrapper dispatch. The registry still calls reduction,
  standalone reduction, plain MAcc, and computed-mask accumulation builder
  symbols that are defined in central route planning.
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` still declares the
  reduction/accumulation statement-plan getters and migrated builder hooks:
  `getRVVSelectedBodyReductionRouteStatementPlan`,
  `getRVVSelectedBodyStandaloneReductionRouteStatementPlan`,
  `getRVVSelectedBodyPlainMAccRouteStatementPlan`,
  `getRVVSelectedBodyComputedMaskAccumulationRouteStatementPlan`, and their
  migrated builder hooks.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` still implements provider-ready
  statement sequences for those families, including `setvl`, loads, scalar
  splat, compare, reduction/MAcc compute, masked merge, accumulator/result
  movement, and stores.
- Existing C++ coverage in `test/Plugin/RVVExtensionPluginTest.cpp` already
  exercises migrated owner registry membership, ordinary reduction statement
  plans, standalone reduction statement plans, plain/scalar-broadcast MAcc
  statement plans, and computed-mask accumulation statement plans.

## Requirements

1. Move provider-ready statement construction for ordinary reduction,
   standalone reduction, plain/scalar-broadcast MAcc, and computed-mask
   accumulation MAcc into owner-local RVV EmitC statement-plan implementation.
2. Remove the reduction/accumulation migrated statement-plan builder
   declarations from `RVVEmitCRoutePlanning.h`; planning must no longer expose
   those builder hooks as provider-facing construction authority.
3. Move the family statement-plan getter declarations for the chosen
   reduction/accumulation families to the owner boundary header
   `RVVEmitCStatementPlanOwners.h`, matching the previous memory extraction
   pattern.
4. Remove or demote the reduction/accumulation statement-plan construction
   definitions from `RVVEmitCRoutePlanning.cpp`; planning may retain neutral
   route analysis, materialization facts, family plans, route-control provider
   plans, math operand-binding facts, diagnostics, and shared structs.
5. Preserve ordinary `reduce_add` semantics, including same-analysis math
   operand-binding facts, dtype/SEW/LMUL facts, reduction store VL, runtime
   `n`/AVL/VL control, source provenance, and load/reduce/store leaves.
6. Preserve standalone reduction semantics, including plain and computed-mask
   variants, seed/source/result channel split, scalar result layout, inactive
   lane neutralization for computed-mask reductions, route-control consumption,
   and accumulator/result ABI roles.
7. Preserve plain/scalar-broadcast MAcc semantics, including accumulator layout,
   scalar RHS splat for scalar-broadcast MAcc, route-control consumption where
   required, and correct loop-induction address advancement for stores.
8. Preserve computed-mask accumulation semantics for vector-compare and
   runtime-scalar computed-mask MAcc, including shared computed-mask
   accumulation family plan validation, route-control consumption, compare mask
   production, active MAcc, masked merge/passthrough, and store.
9. Keep `RVVEmitCRouteProvider.cpp` neutral: it may obtain provider facts,
   instantiate `TCRVEmitCLowerableRoute`, record neutral headers/type
   mappings/ABI mappings/source provenance, and attach owner-returned
   statements through the shared owner boundary. It must not rebuild
   reduction/accumulation statement sequences.
10. Do not add new route coverage, new RVV operations, new dtype or LMUL
    families, selected-body realization cases, source-front-door routes,
    high-level frontend lowering, performance tuning, dashboards, broad smoke
    matrices, or runtime/correctness/performance claims.

## Acceptance Criteria

- [ ] Owner-local RVV EmitC statement-plan code constructs the ordinary
      reduction, standalone reduction, plain/scalar-broadcast MAcc, and
      computed-mask accumulation MAcc statement plans.
- [ ] `RVVEmitCRoutePlanning.h` no longer declares the chosen
      reduction/accumulation migrated builder hooks as planning-owned
      provider-facing authority.
- [ ] `RVVEmitCRoutePlanning.cpp` no longer defines provider-ready statement
      construction for the extracted reduction/accumulation families.
- [ ] Planning retains only justified neutral facts, validation, family plans,
      route-control provider plans, math operand-binding facts, and shared
      statement-plan structs needed by the owner builders.
- [ ] The migrated owner registry still selects exactly one owner for each
      migrated family and fails closed for duplicate, missing, incomplete, or
      wrong-family owner hooks.
- [ ] Focused C++ coverage proves positive statement-plan construction and
      provider consumption for representative reduction/accumulation cases,
      plus fail-closed diagnostics for missing/stale dependencies.
- [ ] Representative generated-bundle dry-runs pass for at least one extracted
      reduction path and one extracted accumulation/MAcc path, with
      `route_entry_realization: false` when applicable.
- [ ] Bounded scans show `RVVEmitCRouteProvider.cpp` does not manually sequence
      reduction/accumulation family statement getters or rebuild their
      statement sequences.
- [ ] Bounded scans show touched RVV planning/provider/owner/test files do not
      introduce legacy-i32, descriptor, source-front-door/source-artifact,
      route-id, artifact-name, exact-intrinsic, common-EmitC,
      direct-route-entry-only, pre-realized-fixture-only, or status/supported
      mirror authority drift.
- [ ] `git diff --check` passes.
- [ ] Focused C++ plugin build/test and `tcrv-opt` / `tcrv-translate` build
      pass.
- [ ] `check-tianchenrv` passes, or an exact blocker is recorded.
- [ ] `ssh rvv` is not required unless this round claims new executable,
      runtime, correctness, or performance behavior.

## Technical Approach

The intended production flow after this task is:

```text
selected RVV variant
  -> typed/realized tcrv_rvv reduction or accumulation body
  -> route-family provider plans / materialization facts
  -> math operand-binding facts / route-control provider plan where required
  -> reduction/accumulation statement-plan owner-local implementation
  -> migrated statement-plan owner registry exact-one selection
  -> owner-produced provider-ready pre-loop and loop statements
  -> RVVEmitCRouteProvider neutral TCRVEmitCLowerableRoute assembly
  -> common EmitC materialization
```

Implementation will add an owner-local reduction/accumulation statement-plan
source file under `lib/Plugin/RVV/EmitC/`, move the selected getter and
migrated-builder definitions there, update the owner header and CMake list, and
remove those provider-ready statement builders from central planning. The owner
registry will continue to dispatch exactly once for each migrated family.

## Validation Plan

1. Build and run focused C++ plugin coverage:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
2. Build touched production tools:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
3. Run representative generated-bundle dry-runs for at least one reduction path
   and one accumulation/MAcc path covered by this task, checking
   `route_entry_realization: false` where the generated metadata reports it.
4. Run bounded planning/provider/owner scans proving statement construction
   moved and provider neutrality remained intact.
5. Run bounded authority drift scans over touched RVV production/test files.
6. Run `git diff --check`.
7. Run `cmake --build build --target check-tianchenrv -j2`, or record the exact
   blocker.

## Out of Scope

- Moving route analysis, route-family provider plans, materialization facts,
  route-control provider plans, math operand-binding facts, diagnostics, or
  shared data structures out of planning.
- Moving memory owners again.
- Starting elementwise/select/conversion/runtime-splat owner extraction.
- Changing direct contraction provider ownership unless required to keep this
  owner boundary compiling and neutral.
- Modifying common EmitC materialization, target artifact export semantics,
  selected-body realization, route-family provider-plan verification, or
  source-front-door behavior.
- New op coverage, dtype/SEW/LMUL/policy expansion, Linalg/Vector/StableHLO
  frontend work, IME, Offload, TensorExt, Toy, Template, future plugin work,
  dashboards, broad smoke matrices, or runtime performance claims.

## Completion Evidence

- Extracted the reduction/accumulation migrated statement-plan implementation
  into
  `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`.
- `RVVEmitCStatementPlanOwners.h` now declares the owner-boundary getters and
  migrated builder hooks for ordinary reduction, standalone reduction, plain
  MAcc, and computed-mask accumulation.
- `RVVEmitCRoutePlanning.h` no longer declares the selected
  reduction/accumulation statement-plan getters or migrated builder hooks.
- `RVVEmitCRoutePlanning.cpp` no longer defines provider-ready statement
  construction helpers/getters/builders for the extracted families. It retains
  neutral route facts, route-family provider plans, route-control provider
  plans, math operand-binding facts, metadata mirrors, diagnostics, and shared
  statement-plan structs.
- `RVVEmitCRouteProvider.cpp` was not changed; provider behavior remains
  neutral aggregate owner selection plus route assembly.
- Spec update review: no `.trellis/spec/**` edit was needed. The existing
  `.trellis/spec/extension-plugins/rvv-plugin.md` already contains the durable
  reduction, standalone reduction, plain/scalar-broadcast MAcc,
  computed-mask accumulation, migrated owner boundary, provider neutrality,
  fail-closed diagnostics, mirror metadata, and common EmitC neutrality rules
  implemented by this task.

Checks and evidence:

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- Representative generated-bundle dry-run passed for `reduce_add`,
  `standalone_reduce_add`, `macc_add`, and `computed_masked_macc_add` under
  `artifacts/tmp/stage2_rvv_reduction_accumulation_statement_plan_owner_boundary/owner-boundary-dry-run/reduction-accumulation-owner-boundary`.
- Generated-bundle evidence kept `route_entry_realization: false` for all four
  representative pre-realized selected-body paths.
- Bounded scans showed `RVVEmitCRoutePlanning.cpp` and
  `RVVEmitCRoutePlanning.h` no longer carry the extracted
  reduction/accumulation statement-plan getter/builder authority.
- Bounded provider scan showed `RVVEmitCRouteProvider.cpp` still only calls
  `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` and
  `attachRVVSelectedBodyRouteStatementPlanOwnerSelection(...)`; it does not
  rebuild reduction/MAcc/accumulation statement sequences.
- Added-line authority scan over the touched production diff showed no
  legacy-i32, source-front-door, source-artifact, descriptor,
  direct-route-entry, route-id, artifact-name, exact-intrinsic,
  `status`/bare `supported`, or `provider_supported` authority drift.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed with 464/464 lit
  tests.
- `clang-format` was not available in the local PATH, so no formatter run was
  possible; the moved code preserved existing formatting and `git diff --check`
  passed.
