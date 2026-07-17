# Stage2 RVV compare/select statement-plan owner boundary

## Goal

Move provider-ready statement construction for the existing RVV compare/select
route family out of monolithic `RVVEmitCRoutePlanning.cpp` and into an
owner-local RVV EmitC statement-plan implementation file. Route planning
remains responsible for neutral route analysis, typed/materialization facts,
plain and computed-mask compare/select route-family plans, route-control
provider plans, mask/tail policy provider plans, operand-binding facts,
validation, diagnostics, and shared statement-plan structs.
`RVVEmitCRouteProvider.cpp` remains a neutral `TCRVEmitCLowerableRoute`
assembler that consumes aggregate owner selection and attaches returned
statements.

## Direction Source

- Direction title: `Switch: Stage2 RVV compare/select statement-plan owner boundary`.
- Module owner: RVV EmitC compare/select statement-plan owner.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `c45a6b5d rvv: move arithmetic statements to owners`.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The immediate predecessor
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-elementwise-arithmetic-owner-boundary/prd.md`
  moved generic elementwise arithmetic statement sequencing into
  `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseArithmeticStatementPlanOwners.cpp`
  while preserving aggregate owner selection and provider attach.
- `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h` already declares
  migrated owner registry types and the compare/select consumer predicate, and
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp` already registers the
  compare/select owner in the aggregate owner registry.
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` still declares
  `getRVVSelectedBodyCompareSelectRouteStatementPlan` and
  `buildRVVSelectedBodyCompareSelectMigratedRouteStatementPlan`.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` still owns compare/select
  statement-plan helper/getter/builder sequencing, including plain
  `cmp_select`, computed-mask select, runtime-scalar compare-select, and
  runtime-scalar dual compare-mask-and-select statement construction.
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` currently reaches migrated
  statement plans through `getRVVSelectedBodyRouteStatementPlanOwnerSelection`
  and `attachRVVSelectedBodyRouteStatementPlanOwnerSelection`; this attach flow
  should remain aggregate-owner based.

## Requirements

1. Add an owner-local RVV EmitC source file that constructs existing
   compare/select provider-ready statement plans.
2. Move the compare/select statement-plan getter and migrated builder
   declarations from `RVVEmitCRoutePlanning.h` to
   `RVVEmitCStatementPlanOwners.h`.
3. Remove compare/select statement-plan helper/getter/builder authority from
   `RVVEmitCRoutePlanning.cpp`.
4. Keep central route planning as the owner only for neutral route analysis,
   route-family/provider facts, route-control provider plans, mask/tail policy
   provider plans, operand-binding facts, diagnostics, and shared data
   structures.
5. Preserve existing plain `cmp_select` statement order, source provenance,
   runtime `n`/AVL/VL control, lhs/rhs/out ABI operand binding,
   compare/select/load/store leaves, dtype/config/policy validation, and route
   facts.
6. Preserve existing computed-mask select behavior, including true/false value
   operand bindings, mask/tail policy provider-plan validation, compare mask
   construction, selected result construction, and store behavior.
7. Preserve existing runtime-scalar compare-select behavior, including
   runtime scalar RHS splat, computed-mask route-family facts, true/false
   value operand binding, mask/tail policy validation, and selected result
   construction.
8. Preserve existing runtime-scalar dual compare-mask-and-select behavior,
   including secondary compare lhs/RHS-scalar bindings, secondary compare,
   mask-and construction, true/false value loads, selected result construction,
   and store behavior.
9. Keep `RVVEmitCRouteProvider.cpp` neutral: it may obtain provider facts,
   instantiate `TCRVEmitCLowerableRoute`, record neutral headers/type
   mappings/ABI mappings/source provenance, and attach owner-returned
   statements through the shared owner boundary. It must not hand-sequence
   compare/select statements.
10. Do not add new compare/select route coverage, dtype/LMUL cases,
    source-front-door routes, high-level Linalg/frontend lowering,
    one-intrinsic wrappers, runtime performance claims, broad smoke matrices,
    dashboards, or unrelated owner migrations.

## Acceptance Criteria

- [ ] Owner-local RVV EmitC statement-plan code constructs existing
      compare/select plans for plain `cmp_select`, computed-mask select,
      runtime-scalar compare-select, and runtime-scalar dual
      compare-mask-and-select.
- [ ] `RVVEmitCRoutePlanning.h` no longer declares the compare/select
      statement-plan getter or migrated builder as planning-owned
      provider-facing authority.
- [ ] `RVVEmitCRoutePlanning.cpp` no longer defines compare/select
      statement-plan helper/getter/builder sequencing.
- [ ] Planning retains only justified neutral route analysis,
      route-family/provider facts, route-control provider plans,
      mask/tail policy provider plans, operand-binding facts, diagnostics, and
      shared statement-plan structs.
- [ ] The migrated owner registry still selects exactly one owner for
      compare/select and preserves aggregate owner selection/attach.
- [ ] Existing compare/select plugin tests and provider-consumption tests pass.
- [ ] At least one focused selected-body generated-bundle dry-run confirms
      `route_entry_realization` remains unchanged for the migrated
      compare/select path.
- [ ] Bounded scans show `RVVEmitCRouteProvider.cpp` still consumes the
      aggregate owner-selection/attach boundary and does not rebuild
      compare/select statements.
- [ ] Bounded scans show touched RVV planning/provider/owner/test files do not
      introduce legacy-i32, source-front-door/source-artifact,
      descriptor-derived, direct-route-entry-only, route-id, artifact-name,
      exact-intrinsic, bare status/supported, provider_supported, or
      common-EmitC semantic authority drift.
- [ ] `git diff --check` passes.
- [ ] Focused C++ plugin build/test and `tcrv-opt` / `tcrv-translate` build
      pass.
- [ ] `check-tianchenrv` passes, or an exact blocker is recorded.
- [ ] `ssh rvv` is not required unless this round claims new runtime,
      correctness, or performance behavior.

## Technical Approach

The intended production flow after this task is:

```text
selected RVV variant
  -> typed/realized tcrv_rvv compare/select body
  -> route-family provider plans / materialization facts
  -> elementwise/select operand-binding facts
  -> route-control provider plan
  -> mask/tail policy provider plan when required
  -> compare/select owner-local statement plan
  -> migrated statement-plan owner registry exact-one selection
  -> owner-produced provider-ready pre-loop and loop statements
  -> RVVEmitCRouteProvider neutral TCRVEmitCLowerableRoute assembly
  -> common EmitC materialization
```

Implementation will add a compare/select owner-local statement-plan source file
under `lib/Plugin/RVV/EmitC/`, move the selected getter and migrated-builder
definitions there, update the owner header and CMake list, and remove those
provider-ready statement builders from central planning. The owner registry
will continue to dispatch exactly once for the migrated compare/select family.

## Validation Plan

1. Build and run focused C++ plugin coverage:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
2. Build touched production tools:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
3. Run representative generated-bundle dry-runs for selected compare/select
   paths covered by this task, checking that `route_entry_realization` remains
   unchanged where generated metadata reports it.
4. Run bounded planning/provider/owner scans proving compare/select statement
   construction moved and provider neutrality remained intact.
5. Run bounded authority drift scans over touched RVV production/test files.
6. Run `git diff --check`.
7. Run `cmake --build build --target check-tianchenrv -j2`, or record the exact
   blocker.

## Out of Scope

- Moving route analysis, route-family provider plans, materialization facts,
  route-control provider plans, mask/tail policy provider plans,
  operand-binding facts, diagnostics, or shared data structures out of
  planning.
- Reworking completed elementwise arithmetic, memory, or
  reduction/accumulation owners.
- Migrating widening conversion, runtime scalar splat-store, reductions, MAcc,
  memory movement, segment2, computed-mask accumulation, direct contraction,
  or future statement-plan owners.
- Modifying common EmitC materialization, target artifact export semantics,
  selected-body realization, route-family provider-plan verification, or
  source-front-door behavior.
- New op coverage, dtype/SEW/LMUL/policy expansion, Linalg/Vector/StableHLO
  frontend work, IME, Offload, TensorExt, Toy, Template, future plugin work,
  dashboards, broad smoke matrices, or runtime performance claims.

## Completion Evidence

- Added owner-local compare/select statement-plan construction in
  `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`.
- `RVVEmitCStatementPlanOwners.h` now declares the compare/select
  owner-boundary getter and migrated builder hook.
- `RVVEmitCRoutePlanning.h` no longer declares the compare/select
  statement-plan getter or migrated builder hook.
- `RVVEmitCRoutePlanning.cpp` no longer defines the compare/select
  statement-plan helper/getter/builder authority. It retains neutral
  route-family plans, materialization facts, route-control provider plans,
  mask/tail policy provider plans, operand-binding facts, diagnostics, and
  shared structs.
- `RVVEmitCRouteProvider.cpp` was not changed; provider behavior remains
  neutral aggregate owner selection plus route assembly.
- Spec update review: no `.trellis/spec/**` edit was needed. The existing
  `.trellis/spec/extension-plugins/rvv-plugin.md` already contains the durable
  migrated statement-plan owner boundary, provider neutrality, compare/select
  selected-body behavior, fail-closed diagnostics, mirror metadata, and common
  EmitC neutrality rules implemented by this task.

Checks and evidence:

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-05-30-stage2-rvv-compare-select-owner-boundary`
  passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- Representative pre-realized selected-body generated-bundle dry-run passed
  for `cmp_select`, `computed_mask_select`, and `runtime_scalar_cmp_select`
  under
  `artifacts/tmp/stage2_rvv_compare_select_statement_plan_owner_boundary/owner-boundary-dry-run/compare-select-owner-boundary-pre-realized`.
- Generated-bundle evidence kept `route_entry_realization: false` for
  `cmp_select`, `computed_mask_select`, and `runtime_scalar_cmp_select`, with
  `selected_body_realization_producer:
  rvv-plugin-local-selected-body-realization-owner-registry`.
- First dry-run attempt failed before route execution because `llvm-readobj`
  was not on `PATH`; rerun with `/usr/lib/llvm-20/bin/llvm-readobj` passed.
- Bounded scans showed `RVVEmitCRoutePlanning.cpp` and
  `RVVEmitCRoutePlanning.h` no longer carry the extracted compare/select
  statement-plan getter/builder/helper authority.
- Bounded provider scan showed `RVVEmitCRouteProvider.cpp` still only calls
  `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` and
  `attachRVVSelectedBodyRouteStatementPlanOwnerSelection(...)`; it does not
  rebuild compare/select statement sequences.
- Added-line authority scan over the touched production diff showed no
  legacy-i32, source-front-door, source-artifact, descriptor,
  direct-route-entry, route-id, artifact-name, exact-intrinsic,
  `status`/bare `supported`, `provider_supported`, or common-EmitC authority
  drift.
- Generated-bundle evidence scan showed no descriptor, direct-C,
  source-export, source-front-door, `tcrv_rvv.i32_`, or direct pre-realized
  route-entry residue for the focused compare/select dry-run.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed with 464/464 lit
  tests.
- `ssh rvv` was not run because this task made no new runtime, correctness, or
  performance claim.
