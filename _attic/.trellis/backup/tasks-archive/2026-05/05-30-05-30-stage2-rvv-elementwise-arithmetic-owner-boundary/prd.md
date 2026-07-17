# Stage2 RVV elementwise arithmetic statement-plan owner boundary

## Goal

Move provider-ready statement construction for the generic RVV elementwise
arithmetic route family out of monolithic `RVVEmitCRoutePlanning.cpp` and into
the RVV EmitC statement-plan owner boundary. Route planning remains responsible
for neutral route analysis, typed/materialization facts, elementwise arithmetic
and scalar-broadcast route-family plans, route-control provider plans,
operand-binding facts, validation, diagnostics, and shared statement-plan
structs. `RVVEmitCRouteProvider` remains a neutral
`TCRVEmitCLowerableRoute` assembler that consumes aggregate owner selection and
attaches returned statements.

## Direction Source

- Direction title: `Switch: Stage2 RVV elementwise arithmetic statement-plan
  owner boundary`.
- Module owner: RVV EmitC owner-local statement-plan construction for the
  generic elementwise arithmetic route family.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `9ed999d2 rvv: move reduction statements to owners`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The archived predecessor
  `.trellis/tasks/archive/2026-05/05-30-stage2-rvv-reduction-accumulation-owner-boundary`
  moved ordinary reduction, standalone reduction, plain/scalar-broadcast MAcc,
  and computed-mask accumulation statement-plan construction into
  `RVVEmitCReductionAccumulationStatementPlanOwners.cpp`.
- `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp` and
  `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`
  are the current templates for owner-local migrated statement-plan
  implementation files.
- `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp` already owns migrated
  statement-plan owner registry membership, exact-one owner selection, and
  aggregate owner selection/attach. The registry still selects the elementwise
  arithmetic owner but calls a builder defined in central route planning.
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` still declares
  `getRVVSelectedBodyElementwiseArithmeticRouteStatementPlan` and
  `buildRVVSelectedBodyElementwiseArithmeticMigratedRouteStatementPlan`.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` still constructs the
  provider-ready elementwise arithmetic statement sequence for ordinary
  add/sub/mul, RHS broadcast load, scalar-broadcast add/sub/mul, masked
  add/sub/mul, and `strided_add`.
- Existing C++ coverage in `test/Plugin/RVVExtensionPluginTest.cpp` covers the
  migrated owner registry and representative elementwise arithmetic statement
  plans/provider paths.

## Requirements

1. Move provider-ready elementwise arithmetic statement construction into an
   owner-local RVV EmitC statement-plan implementation file.
2. Remove the elementwise arithmetic statement-plan getter and migrated builder
   declarations from `RVVEmitCRoutePlanning.h`; planning must no longer expose
   these APIs as planning-owned provider-facing statement authority.
3. Declare the elementwise arithmetic statement-plan getter and migrated builder
   on the owner boundary header `RVVEmitCStatementPlanOwners.h`, matching the
   memory and reduction/accumulation extraction pattern.
4. Remove or demote the elementwise arithmetic provider-ready statement
   construction definitions and helper authority from `RVVEmitCRoutePlanning.cpp`.
   Planning may retain neutral elementwise route-family plan derivation,
   materialization facts, route-control provider plans, operand-binding facts,
   diagnostics, and shared statement-plan structs.
5. Preserve existing ordinary elementwise semantics, including unit-load
   add/sub/mul, RHS broadcast-load behavior, same-analysis
   elementwise/select operand-binding facts, dtype/SEW/LMUL facts, runtime
   `n`/AVL/VL control, and load/compute/store leaves.
6. Preserve existing scalar-broadcast elementwise semantics, including runtime
   scalar ABI binding, RHS scalar splat leaf, route-control consumption,
   result dtype/config/policy facts, and store behavior.
7. Preserve existing masked elementwise semantics, including compare mask
   production, active arithmetic, masked merge/passthrough behavior, mask/tail
   policy facts already validated by route-family planning, and store behavior.
8. Preserve existing `strided_add` semantics, including residual
   operand-binding facts, lhs/rhs/out stride ABI roles, strided load/store
   leaves, and loop-induction address advancement.
9. Keep `RVVEmitCRouteProvider.cpp` neutral: it may obtain provider facts,
   instantiate `TCRVEmitCLowerableRoute`, record neutral headers/type
   mappings/ABI mappings/source provenance, and attach owner-returned
   statements through the shared owner boundary. It must not hand-sequence
   elementwise arithmetic statements.
10. Do not add new operation coverage, dtype/LMUL families, compare/select,
    widening conversion, runtime scalar splat-store, memory movement,
    segment, selected-body realization, source-front-door routes, high-level
    frontend lowering, dashboards, broad smoke matrices, or runtime/
    correctness/performance claims.

## Acceptance Criteria

- [ ] Owner-local RVV EmitC statement-plan code constructs existing
      elementwise arithmetic plans for ordinary unit-load/broadcast-load,
      scalar-broadcast, masked, and strided-add routes.
- [ ] `RVVEmitCRoutePlanning.h` no longer declares the elementwise arithmetic
      statement-plan getter or migrated builder as planning-owned
      provider-facing authority.
- [ ] `RVVEmitCRoutePlanning.cpp` no longer defines provider-ready
      elementwise arithmetic statement construction helpers/getters/builders.
- [ ] Planning retains only justified neutral route analysis,
      route-family/provider facts, route-control provider plans,
      operand-binding facts, diagnostics, and shared statement-plan structs.
- [ ] The migrated owner registry still selects exactly one owner for
      elementwise arithmetic and preserves aggregate owner selection/attach.
- [ ] Focused C++ coverage proves representative elementwise arithmetic
      owner-local statement-plan construction and provider consumption, with
      existing fail-closed diagnostics preserved.
- [ ] Representative generated-bundle dry-runs pass for existing selected
      arithmetic paths, with `route_entry_realization: false` where generated
      metadata reports it.
- [ ] Bounded scans show `RVVEmitCRouteProvider.cpp` does not manually sequence
      arithmetic statement getters or rebuild arithmetic statements.
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
  -> typed/realized tcrv_rvv elementwise arithmetic body
  -> route-family provider plans / materialization facts
  -> elementwise/select or residual operand-binding facts
  -> route-control provider plan where required
  -> elementwise arithmetic owner-local statement plan
  -> migrated statement-plan owner registry exact-one selection
  -> owner-produced provider-ready pre-loop and loop statements
  -> RVVEmitCRouteProvider neutral TCRVEmitCLowerableRoute assembly
  -> common EmitC materialization
```

Implementation will add an owner-local elementwise arithmetic statement-plan
source file under `lib/Plugin/RVV/EmitC/`, move the selected getter and
migrated-builder definitions there, update the owner header and CMake list,
and remove those provider-ready statement builders from central planning. The
owner registry will continue to dispatch exactly once for the migrated
elementwise arithmetic family.

## Validation Plan

1. Build and run focused C++ plugin coverage:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
2. Build touched production tools:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
3. Run representative generated-bundle dry-runs for selected arithmetic paths
   covered by this task, checking `route_entry_realization: false` where the
   generated metadata reports it.
4. Run bounded planning/provider/owner scans proving arithmetic statement
   construction moved and provider neutrality remained intact.
5. Run bounded authority drift scans over touched RVV production/test files.
6. Run `git diff --check`.
7. Run `cmake --build build --target check-tianchenrv -j2`, or record the exact
   blocker.

## Out of Scope

- Moving route analysis, route-family provider plans, materialization facts,
  route-control provider plans, operand-binding facts, diagnostics, or shared
  data structures out of planning.
- Reworking completed memory or reduction/accumulation owners.
- Starting compare/select, widening conversion, runtime scalar splat-store,
  direct contraction, or future migrated-family owner extraction.
- Modifying common EmitC materialization, target artifact export semantics,
  selected-body realization, route-family provider-plan verification, or
  source-front-door behavior.
- New op coverage, dtype/SEW/LMUL/policy expansion, Linalg/Vector/StableHLO
  frontend work, IME, Offload, TensorExt, Toy, Template, future plugin work,
  dashboards, broad smoke matrices, or runtime performance claims.

## Completion Evidence

- Added owner-local elementwise arithmetic statement-plan construction in
  `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseArithmeticStatementPlanOwners.cpp`.
- `RVVEmitCStatementPlanOwners.h` now declares the elementwise arithmetic
  owner-boundary getter and migrated builder hook.
- `RVVEmitCRoutePlanning.h` no longer declares the elementwise arithmetic
  statement-plan getter or migrated builder hook.
- `RVVEmitCRoutePlanning.cpp` no longer defines the elementwise arithmetic
  statement-plan helper/getter/builder authority. It retains neutral
  route-family plans, materialization facts, route-control provider plans,
  operand-binding facts, diagnostics, and shared structs.
- `RVVEmitCRouteProvider.cpp` was not changed; provider behavior remains
  neutral aggregate owner selection plus route assembly.
- Spec update review: no `.trellis/spec/**` edit was needed. The existing
  `.trellis/spec/extension-plugins/rvv-plugin.md` already contains the durable
  migrated statement-plan owner boundary, provider neutrality, fail-closed
  diagnostics, mirror metadata, and common EmitC neutrality rules implemented
  by this task.

Checks and evidence:

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-05-30-stage2-rvv-elementwise-arithmetic-owner-boundary`
  passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- Representative explicit selected-body generated-bundle dry-run passed for
  `add`, `scalar_broadcast_add`, `masked_add`, and `strided_add` under
  `artifacts/tmp/stage2_rvv_elementwise_arithmetic_statement_plan_owner_boundary/owner-boundary-dry-run/elementwise-arithmetic-owner-boundary`.
- Representative pre-realized selected-body generated-bundle dry-run passed for
  `add`, `masked_add`, and `strided_add` under
  `artifacts/tmp/stage2_rvv_elementwise_arithmetic_statement_plan_owner_boundary/owner-boundary-dry-run/elementwise-arithmetic-owner-boundary-pre-realized`.
- Generated-bundle evidence kept `route_entry_realization: false` for the
  pre-realized `add`, `masked_add`, and `strided_add` selected-body paths.
- Bounded scans showed `RVVEmitCRoutePlanning.cpp` and
  `RVVEmitCRoutePlanning.h` no longer carry the extracted elementwise
  arithmetic statement-plan getter/builder authority.
- Bounded provider scan showed `RVVEmitCRouteProvider.cpp` still only calls
  `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` and
  `attachRVVSelectedBodyRouteStatementPlanOwnerSelection(...)`; it does not
  rebuild arithmetic statement sequences.
- Added-line authority scan over the touched production diff showed no
  legacy-i32, source-front-door, source-artifact, descriptor,
  direct-route-entry, route-id, artifact-name, exact-intrinsic,
  `status`/bare `supported`, or `provider_supported` authority drift.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed with 464/464 lit
  tests.
- `ssh rvv` was not run because this task made no new runtime, correctness, or
  performance claim.
