# Stage2 RVV residual conversion/scalar-splat statement-plan owner boundary

## Goal

Move provider-ready statement-plan construction for the existing RVV
`WideningConversion` and `RuntimeScalarSplatStore` migrated route families out
of central `RVVEmitCRoutePlanning.cpp` and into owner-local RVV EmitC
statement-plan owner module code. Preserve the aggregate owner-selection and
attach boundary consumed by `RVVEmitCRouteProvider`, while keeping route
planning responsible for neutral route analysis, family/provider facts,
route-control provider plans, operand-binding facts, diagnostics, and shared
statement-plan data structures.

## Direction Source

- Direction title: `Switch: Stage2 RVV residual conversion/scalar-splat statement-plan owner boundary`.
- Module owner: RVV EmitC statement-plan owner boundary for
  `WideningConversion` and `RuntimeScalarSplatStore`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `125045d8 rvv: move compare select statements to owners`.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The immediate predecessor
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-compare-select-owner-boundary/prd.md`
  moved compare/select statement sequencing into
  `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`.
- Current owner-local files already cover elementwise arithmetic, memory,
  reduction/accumulation, and compare/select statement plans.
- `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h` declares
  consumer predicates for `WideningConversion` and
  `RuntimeScalarSplatStore`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp` already registers
  both families in `getRVVSelectedBodyMigratedRouteStatementPlanOwners()`.
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` still declares
  `getRVVSelectedBodyWideningConversionRouteStatementPlan`,
  `getRVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan`,
  `buildRVVSelectedBodyWideningConversionMigratedRouteStatementPlan`, and
  `buildRVVSelectedBodyRuntimeScalarSplatStoreMigratedRouteStatementPlan`.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` still defines the
  family-specific statement-plan helper/getter/builder authority for both
  residual singleton families.
- The central route planning file may still retain neutral shared data
  structures and analysis/fact builders needed by owner-local statement-plan
  modules.

## Requirements

1. Add owner-local RVV EmitC statement-plan construction code for the existing
   `WideningConversion` and `RuntimeScalarSplatStore` migrated route families.
2. Move the two statement-plan getter declarations and two migrated builder
   declarations from `RVVEmitCRoutePlanning.h` to
   `RVVEmitCStatementPlanOwners.h`.
3. Remove `WideningConversion` and `RuntimeScalarSplatStore`
   helper/getter/builder statement-plan sequencing from
   `RVVEmitCRoutePlanning.cpp`.
4. Keep central route planning as the owner only for neutral route analysis,
   selected-body route-family/provider facts, materialization facts,
   route-control provider plans, operand-binding facts, diagnostics, and shared
   structs.
5. Preserve existing widening conversion behavior, including statement order,
   source provenance, runtime `n`/AVL/VL control, lhs/out ABI binding,
   source load, widening conversion compute leaf, store leaf, source/result
   dtype/config facts, conversion relation, route-control facts, and emitted
   route fact spelling.
6. Preserve existing runtime scalar splat-store behavior, including statement
   order, source provenance, runtime `n`/AVL/VL control, `rhs_scalar`/out ABI
   binding, scalar splat leaf, store leaf, typed result vector facts,
   route-control facts, and emitted route fact spelling.
7. Preserve aggregate owner selection: `RVVEmitCRouteProvider` must continue
   to consume `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` and
   `attachRVVSelectedBodyRouteStatementPlanOwnerSelection(...)`; it must not
   grow hand-written sequencing for these families.
8. Do not add new RVV op coverage, dtype/LMUL clone batches, direct
   pre-realized route-entry paths, source-front-door positives, high-level
   frontend/Linalg lowering, selected-body realization semantic changes,
   provider route rewrites, common EmitC semantic choices, performance tuning,
   broad smoke matrices, or dashboards.

## Acceptance Criteria

- [ ] Owner-local RVV EmitC statement-plan code constructs existing
      `WideningConversion` plans for `WidenI32ToI64` and `WidenI16ToI32`.
- [ ] Owner-local RVV EmitC statement-plan code constructs existing
      `RuntimeScalarSplatStore` plans for `RuntimeI32SplatStore`.
- [ ] `RVVEmitCRoutePlanning.h` no longer declares the two residual singleton
      statement-plan getters or migrated builders as planning-owned
      provider-facing authority.
- [ ] `RVVEmitCRoutePlanning.cpp` no longer defines residual singleton
      statement-plan helper/getter/builder sequencing for
      `WideningConversion` or `RuntimeScalarSplatStore`.
- [ ] Planning retains only justified neutral route analysis,
      route-family/provider facts, materialization facts, route-control
      provider plans, operand-binding facts, diagnostics, and shared
      statement-plan structs.
- [ ] The migrated owner registry still registers and dispatches exactly one
      owner for each of `WideningConversion` and `RuntimeScalarSplatStore`.
- [ ] Focused C++ or lit coverage proves owner selection/attach consumes
      owner-produced statements for runtime scalar splat-store and existing
      widening conversion routes.
- [ ] Representative generated-bundle dry-runs for directly affected existing
      fixtures preserve selected-body materialization and
      `route_entry_realization` mirror evidence.
- [ ] Bounded scans show no `WideningConversion` or
      `RuntimeScalarSplatStore` statement-plan getter/builder/helper authority
      remains in `RVVEmitCRoutePlanning.h/cpp`, while aggregate owner
      selection still registers and dispatches those families.
- [ ] Bounded scans show `RVVEmitCRouteProvider.cpp` still consumes aggregate
      owner selection/attach and does not rebuild these statement sequences.
- [ ] Bounded touched-file authority scan shows no legacy-i32,
      source-front-door/source-artifact, descriptor-derived,
      direct-route-entry-only, route-id, artifact-name, exact-intrinsic,
      bare status/supported, provider_supported, script-derived, ABI-string,
      or common-EmitC semantic authority drift.
- [ ] `git diff --check` passes.
- [ ] `tianchenrv-rvv-extension-plugin-test`, `tcrv-opt`, and
      `tcrv-translate` build.
- [ ] The RVV extension plugin C++ test passes.
- [ ] `check-tianchenrv` passes, or the exact blocker is recorded.
- [ ] `ssh rvv` is not required unless this round changes runtime/executable
      behavior or makes a new runtime/correctness/performance claim.

## Technical Approach

The intended production flow after this task is:

```text
selected RVV variant
  -> typed/realized tcrv_rvv widening conversion or runtime scalar splat body
  -> verified route-family/provider facts
  -> route materialization facts
  -> math or residual operand-binding facts
  -> route-control provider plan
  -> residual singleton owner-local statement plan
  -> migrated statement-plan owner registry exact-one selection
  -> owner-produced provider-ready pre-loop and loop statements
  -> RVVEmitCRouteProvider neutral TCRVEmitCLowerableRoute assembly
  -> common EmitC materialization
```

Implementation will add an owner-local residual singleton statement-plan source
file under `lib/Plugin/RVV/EmitC/`, move the selected getter and migrated
builder definitions there, update `RVVEmitCStatementPlanOwners.h`, remove those
declarations from `RVVEmitCRoutePlanning.h`, wire CMake, and delete the
planning-owned helper/getter/builder definitions from
`RVVEmitCRoutePlanning.cpp`.

## Validation Plan

1. Validate the Trellis task context:
   `python3 ./.trellis/scripts/task.py validate <task-dir>`.
2. Build and run focused C++ plugin coverage:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
3. Build touched production tools:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
4. Run generated-bundle dry-runs for directly affected existing fixtures,
   including runtime scalar splat-store and widening conversion paths.
5. Run bounded planning/provider/owner scans proving residual singleton
   statement-plan construction moved and provider neutrality remained intact.
6. Run bounded authority drift scans over touched RVV production/test files.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Out of Scope

- Moving route analysis, route-family provider plans, materialization facts,
  route-control provider plans, operand-binding facts, diagnostics, or shared
  data structures out of planning.
- Reworking already-completed compare/select, elementwise arithmetic, memory,
  reduction/accumulation, MAcc, widening dot, direct-contraction, or segment2
  paths except for necessary non-regression consumers.
- Modifying common EmitC materialization, target artifact export semantics,
  selected-body realization, route-family provider-plan verification, or
  source-front-door behavior.
- New RVV op coverage, dtype/SEW/LMUL/policy expansion, Linalg/Vector/StableHLO
  frontend work, IME, Offload, TensorExt, Toy, Template, future plugin work,
  dashboards, broad smoke matrices, or runtime performance claims.

## Completion Evidence

- Added owner-local residual singleton statement-plan construction in
  `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`.
- Moved `WideningConversion` and `RuntimeScalarSplatStore` statement-plan
  getter and migrated builder declarations from `RVVEmitCRoutePlanning.h` into
  `RVVEmitCStatementPlanOwners.h`.
- Removed the `WideningConversion` and `RuntimeScalarSplatStore`
  statement-plan helper/getter/builder authority from
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
- Wired the owner-local file into `lib/Plugin/RVV/EmitC/CMakeLists.txt`.
- `RVVEmitCRoutePlanning.cpp` still retains neutral route analysis,
  route-family/provider facts, materialization facts, route-control provider
  plans, operand-binding facts, diagnostics, direct contraction provider-plan
  code, memory/segment provider-plan code, and shared data structures.
- `RVVEmitCRouteProvider.cpp` was not changed. It still consumes
  `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` and
  `attachRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` instead of
  hand-sequencing these families.
- Spec update review: no `.trellis/spec/**` edit was needed. Existing
  `.trellis/spec/extension-plugins/rvv-plugin.md` already contains the durable
  migrated statement-plan aggregate boundary, widening conversion ownership,
  runtime scalar splat-store residual boundary, route-control provider plan,
  provider neutrality, fail-closed diagnostics, mirror metadata, and common
  EmitC neutrality rules implemented by this task.

Checks and evidence:

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-05-30-stage2-rvv-residual-conversion-splat-owner-boundary`
  passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- Focused generated-bundle dry-run passed for
  `runtime_i32_splat_store`:
  `artifacts/tmp/stage2_rvv_residual_statement_plan_owner_boundary/runtime-splat-store-owner-boundary-pre-realized`.
- Focused generated-bundle dry-run passed for `widen_i32_to_i64` and
  `widen_i16_to_i32`:
  `artifacts/tmp/stage2_rvv_residual_statement_plan_owner_boundary/widening-conversion-owner-boundary-pre-realized`.
- Generated-bundle evidence preserved `route_entry_realization: false`,
  `selected_body_realization_producer:
  rvv-plugin-local-selected-body-realization-owner-registry`, and
  `pre_realized_body_consumed: true` for `runtime_i32_splat_store`,
  `widen_i32_to_i64`, and `widen_i16_to_i32`.
- Focused lit dry-run fixtures passed 3/3:
  `rvv-generated-bundle-abi-e2e-pre-realized-runtime-i32-splat-store-dry-run.test`,
  `rvv-generated-bundle-abi-e2e-pre-realized-widen-i32-to-i64-dry-run.test`,
  and `rvv-generated-bundle-abi-e2e-pre-realized-widen-i16-to-i32-dry-run.test`.
- Self-repair performed: the first direct lit invocation from the source root
  failed because `tianchenrv_obj_root` is only configured under `build/test`;
  rerunning from `build/test` passed 3/3.
- Bounded planning scan showed no residual singleton statement-plan
  getter/builder/helper authority remains in `RVVEmitCRoutePlanning.h` or
  `RVVEmitCRoutePlanning.cpp`.
- Bounded owner/provider scan showed the aggregate owner registry still
  registers `widening conversion` and `runtime scalar splat-store`, and
  `RVVEmitCRouteProvider.cpp` still reaches route statements through the
  owner-selection/attach boundary.
- Added-line production authority scan over touched RVV owner/planning/CMake
  files found no legacy-i32, source-front-door/source-artifact,
  descriptor-derived, direct-route-entry-only, route-id, artifact-name,
  exact-intrinsic, bare status/supported, provider_supported, script-derived,
  ABI-string, or common-EmitC semantic authority drift.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed with 464/464
  tests.
- `ssh rvv` was not run because this task moved provider-ready statement-plan
  ownership without changing runtime/executable behavior or making a new
  runtime, correctness, or performance claim.
