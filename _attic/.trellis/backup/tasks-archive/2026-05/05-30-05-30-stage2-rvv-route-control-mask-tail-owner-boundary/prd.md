# Stage2 RVV route-control and mask/tail policy owner boundary

## Goal

Move active selected-body route-control and mask/tail policy provider-plan
construction out of central `RVVEmitCRoutePlanning.cpp` into an explicit
RVV-owned control/policy owner module. Preserve current selected-body
statement owner attach/selection behavior, while keeping central route
planning limited to selected-body analysis, route-family facts,
materialization facts, operand-binding facts, diagnostics, and shared provider
data structures.

## Direction Source

- Direction title: `Continue: Stage2 RVV route-control and mask/tail policy owner boundary`.
- Module owner: RVV EmitC route-control and mask/tail policy provider-plan
  ownership boundary.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `8a88113c rvv: move residual statements to owners`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The immediate predecessor
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-residual-conversion-splat-owner-boundary/prd.md`
  completed residual `WideningConversion` and
  `RuntimeScalarSplatStore` statement-plan movement into
  `RVVEmitCResidualStatementPlanOwners.cpp`; `check-tianchenrv` passed 464/464
  and no runtime claim required `ssh rvv`.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires route-control
  provider plans to join typed config facts, runtime AVL/VL control facts,
  selected capability facts, route materialization facts, tail policy, mask
  policy, and runtime ABI order before statement construction.
- The same spec says adopted route-control families must be selected through an
  RVV plugin-local registry exactly once; no owner match returns an empty plan,
  multiple matches fail closed, and common EmitC/artifacts/scripts/descriptors/
  ABI strings/route ids/legacy i32 names must not supply control or policy
  authority.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires
  `TCRVEmitCLowerableRoute` to be provider-built and common EmitC to remain
  neutral; common EmitC must not infer dtype, SEW, LMUL, mask/tail policy,
  operation kind, ABI, or RVV schedule.
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` currently holds the
  route-control and mask/tail plan data structures, owner structs, registry
  declarations, and plan getter declarations.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` currently still owns the
  route-control provider-plan construction authority: registry construction,
  exact-once owner selection, typed config/runtime AVL/VL/capability checks,
  and family-specific route-control builders.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` also owns the mask/tail
  policy provider-plan construction authority for computed-mask select and
  computed-mask memory.
- Existing statement owner modules consume the provider plans:
  `RVVEmitCElementwiseArithmeticStatementPlanOwners.cpp`,
  `RVVEmitCCompareSelectStatementPlanOwners.cpp`,
  `RVVEmitCMemoryStatementPlanOwners.cpp`,
  `RVVEmitCReductionAccumulationStatementPlanOwners.cpp`, and
  `RVVEmitCResidualStatementPlanOwners.cpp`.
- `test/Plugin/RVVExtensionPluginTest.cpp` already contains focused tests for
  route-control and mask/tail policy owner registry behavior and many family
  consumer paths.

## Requirements

1. Add an explicit RVV-owned route-control/mask-tail policy provider-plan owner
   header and implementation module.
2. Move route-control provider owner structs, registry declarations, registry
   definitions, exact-once selector, family-specific route-control provider
   builders, and final typed-config/runtime-control/capability plan validation
   out of `RVVEmitCRoutePlanning.cpp`.
3. Move mask/tail policy provider owner structs, registry declarations,
   registry definitions, exact-once selector, and computed-mask select/memory
   policy provider builders out of `RVVEmitCRoutePlanning.cpp`.
4. Keep shared plan data structures in `RVVEmitCRoutePlanning.h` where needed
   by existing analysis, statement-plan, direct-contraction, and provider data
   structures; do not churn shared structs only to reduce line count.
5. Keep central `RVVEmitCRoutePlanning.cpp` responsible for neutral route
   analysis, selected-body route-family/provider facts, materialization facts,
   operand-binding facts, diagnostics, and route description metadata mirrors.
6. Preserve existing route-control behavior for ordinary elementwise, masked
   elementwise, scalar-broadcast elementwise, plain compare/select,
   computed-mask select, widening conversion, computed-mask memory, segment2
   memory, base memory movement, standalone reduction, plain MAcc,
   scalar-broadcast MAcc, runtime scalar splat-store, computed-mask
   accumulation, and contraction.
7. Preserve existing mask/tail policy behavior for computed-mask select and
   computed-mask memory, including owner mirror labels, policy mirrors,
   inactive/pass-through mirrors, selected capability mirrors, and
   binding-plan checks.
8. Preserve all consumer/attach paths: statement owners must still call the
   route-control and mask/tail owner boundary before building statements, and
   `RVVEmitCRouteProvider` must still consume aggregate statement-owner attach
   instead of rebuilding family statements.
9. Unsupported or inconsistent typed facts, stale materialization facts,
   missing runtime AVL/VL facts, policy mismatches, stale selected capability
   facts, stale binding plans, mismatched mask/tail requirements, and multiple
   owner matches must fail closed with targeted diagnostics.
10. Do not add new RVV operation coverage, dtype/LMUL clone batches,
    high-level Linalg/Vector frontend lowering, IME/Offload/TensorExt work,
    source-front-door positive routes, one-intrinsic wrappers, broad smoke
    matrices, dashboards, reports, artifact indexes, or common EmitC semantic
    choices.

## Acceptance Criteria

- [x] New RVV-owned control/policy owner module owns route-control and
      mask/tail policy provider-plan construction.
- [x] `RVVEmitCRoutePlanning.cpp` no longer defines
      `getRVVSelectedBodyRouteControlProviderPlan`,
      `getRVVSelectedBodyRouteControlProviderOwners`,
      `getRVVSelectedBodyMaskTailPolicyProviderPlan`,
      `getRVVSelectedBodyMaskTailPolicyProviderOwners`, or the moved
      family-specific route-control/mask-tail builder functions.
- [x] `RVVEmitCRoutePlanning.h` no longer exposes route-control or mask/tail
      owner registry structs/functions as planning-owned APIs; consumer code
      includes the explicit owner-boundary header.
- [x] Shared route analysis, route materialization facts, operand-binding
      facts, route-family provider plans, plan data objects, and diagnostics
      remain available without moving unrelated planning code.
- [x] Existing statement owner consumers still compile and call
      `getRVVSelectedBodyRouteControlProviderPlan(...)` and
      `getRVVSelectedBodyMaskTailPolicyProviderPlan(...)` through the explicit
      owner-boundary header.
- [x] Focused C++ tests cover route-control owner selection/attach and
      mask/tail policy fail-closed diagnostics.
- [x] Representative generated-bundle dry-runs pass for ordinary elementwise,
      compare/select with mask policy, base memory movement, contraction or
      reduction, and residual conversion/splat paths, preserving
      `route_entry_realization` and `selected_body_realization` evidence.
- [x] Bounded scans prove central RoutePlanning no longer contains the moved
      construction authority and provider/statement consumers go through the
      owner boundary.
- [x] Bounded authority scans over touched files show no legacy-i32,
      source-front-door/source-artifact, descriptor-derived,
      direct-route-entry-only, route-id, artifact-name, exact-intrinsic,
      bare status/supported, script-derived, ABI-string, or common-EmitC
      semantic authority drift.
- [x] `git diff --check` passes.
- [x] `tianchenrv-rvv-extension-plugin-test`, `tcrv-opt`, and
      `tcrv-translate` build.
- [x] The RVV extension plugin C++ test passes.
- [x] `check-tianchenrv` passes, or the exact blocker is recorded.
- [x] `ssh rvv` is not required unless this round changes executable behavior
      or makes a new runtime/correctness/performance claim.

## Technical Approach

The intended production flow after this task is:

```text
selected RVV variant
  -> typed/realized tcrv_rvv body/config/runtime facts
  -> verified route-family provider plans
  -> RVVSelectedBodyRouteMaterializationFacts
  -> RVV-owned route-control/mask-tail policy owner boundary
  -> RVV-owned statement/provider plan consumers
  -> aggregate statement owner selection/attach
  -> RVVEmitCRouteProvider builds TCRVEmitCLowerableRoute
  -> common EmitC materialization
```

Implementation will add an owner-local control/policy source file under
`lib/Plugin/RVV/EmitC/` and a matching public RVV plugin header under
`include/TianChenRV/Plugin/RVV/`. The new header will expose the owner
registry structs and plan getter APIs; the source file will contain the moved
route-control and mask/tail registry/build/validation code. Existing statement
owner headers/tests will include this boundary header. CMake will compile the
new source into `TianChenRVRVVEmitCRouteProvider`.

## Validation Plan

1. Validate Trellis task context:
   `python3 ./.trellis/scripts/task.py validate <task-dir>`.
2. Build focused plugin test:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`.
3. Run focused C++ plugin coverage:
   `./build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build touched production tools:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
5. Run representative generated-bundle dry-runs for ordinary elementwise,
   compare/select with mask policy, base memory movement, contraction or
   reduction, and residual conversion/splat paths.
6. Run bounded scans for moved owner authority and consumer path evidence.
7. Run bounded authority drift scans over touched RVV production/test files.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Out of Scope

- New route coverage, dtype/SEW/LMUL/policy expansion, or high-level frontend
  integration.
- Moving all route-family provider-plan verification or materialization facts
  out of route planning.
- Rewriting statement-plan owner files for style churn.
- Modifying common EmitC materialization, target artifact export semantics,
  selected-body realization semantics, or runtime executable behavior.
- IME, Offload, TensorExt, Toy, Template, future plugin work, dashboards,
  artifact indexes, broad smoke matrices, or runtime performance claims.

## Completion Evidence

- Added `include/TianChenRV/Plugin/RVV/RVVEmitCControlPolicyPlanOwners.h` and
  `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp` as the explicit
  RVV-owned route-control and mask/tail policy provider-plan owner boundary.
- Moved route-control owner registries, family-specific route-control builders,
  exact-once owner selection, final plan validation, mask/tail owner registries,
  computed-mask select policy builders, and computed-mask memory policy
  builders out of central `RVVEmitCRoutePlanning.cpp`.
- Kept shared plan data structures and neutral facts in
  `RVVEmitCRoutePlanning.h`/`.cpp`: route analysis, materialization facts,
  operand-binding facts, provider-plan data, diagnostics, and mirror metadata
  remain central because existing route/provider/statement data structures
  share them.
- Updated `RVVEmitCStatementPlanOwners.h` and
  `test/Plugin/RVVExtensionPluginTest.cpp` to include the owner-boundary header;
  CMake now compiles the new owner source into the RVV EmitC provider library.
- Self-repair during implementation: the owner source now carries owner-local
  helper validation for typed config facts and computed-mask memory producer
  source classification because the equivalent helpers were file-local to
  central route planning.
- Spec update review found no `.trellis/spec/**` edit was needed: the existing
  RVV plugin and EmitC route specs already require this route-control and
  mask/tail policy owner boundary, fail-closed diagnostics, typed-fact
  authority, and common EmitC neutrality.
- Checks run:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-05-30-stage2-rvv-route-control-mask-tail-owner-boundary`;
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`;
  `./build/bin/tianchenrv-rvv-extension-plugin-test`;
  `cmake --build build --target tcrv-opt tcrv-translate -j2`;
  representative generated-bundle lit dry-runs for ordinary elementwise,
  computed-mask select, base memory movement, contraction, runtime splat, and
  widening conversion paths; bounded owner/consumer/authority scans;
  `git diff --check`; and
  `cmake --build build --target check-tianchenrv -j2`.
- Final `check-tianchenrv` result: 464/464 passed.
- `clang-format` was not available in this environment, but `git diff --check`
  passed.
- `ssh rvv` was not run because this task changed owner boundaries only and did
  not make a new runtime, correctness, or performance claim.
