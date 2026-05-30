# Stage2 RVV MAcc operand-binding owner completion

## Goal

Complete the MAcc route-family ownership split by moving MAcc-specific
route-operand binding authority out of central `RVVEmitCRoutePlanning.cpp` and
into `RVVEmitCMAccRouteFamilyPlanOwners`. The owner boundary must own the
plain MAcc, scalar-broadcast MAcc, computed-mask MAcc, and runtime-scalar
computed-mask MAcc operand-binding plan identity and logical operand to runtime
ABI role validation while preserving the existing typed selected-body route
path and generated-bundle behavior.

## Direction Source

- Direction title: `Continue: Stage2 RVV MAcc route-family operand-binding owner completion`.
- Module owner: `RVVEmitCMAccRouteFamilyPlanOwners`, with central
  `RVVEmitCRoutePlanning` retaining only neutral route analysis, typed-fact
  collection, and aggregate orchestration.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `81c1e118 rvv: move macc route plans to owner`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.
- Serial worker constraint: one Codex worker; no subagents, spawned agents,
  parallel agents, or multi-agent workflows.

## What I Already Know

- The predecessor archived task
  `.trellis/tasks/archive/2026-05/05-30-stage2-rvv-macc-route-family-provider-plan-owner-boundary/`
  introduced `RVVEmitCMAccRouteFamilyPlanOwners.h/cpp`, moved the MAcc
  provider-plan owner registry and provider-plan verifiers out of central
  route planning, and passed focused generated-bundle dry-runs plus
  `check-tianchenrv` 464/464.
- `.trellis/spec/index.md` and `.trellis/spec/extension-plugins/rvv-plugin.md`
  require typed `tcrv_rvv` body/config/runtime facts and plugin-owned route
  provider authority. Route ids, ABI strings, metadata, descriptors,
  source-front-door markers, exact intrinsic spellings, and legacy i32 helpers
  are not authority.
- `.trellis/spec/extension-plugins/rvv-plugin.md` says central
  `RVVEmitCRoutePlanning` may keep shared route-analysis structs, route-family
  plan structs, route descriptions, materialization facts, operand-binding
  containers, and top-level aggregate orchestration, but MAcc consumer
  selection and provider-plan verification must stay owner-local.
- `.trellis/spec/core-dialect/tcrv-exec-contract.md` keeps ABI/runtime role
  declaration in `tcrv.exec`; selected typed `tcrv_rvv` body and RVV plugin
  code must consume those values without making `tcrv.exec` compute authority.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires provider-built
  `TCRVEmitCLowerableRoute` and neutral common EmitC materialization.
- Current code still keeps MAcc-specific operand-binding plan IDs and
  MAcc-specific logical operand to `RuntimeABIParameterRole` mapping inside
  central `RVVEmitCRoutePlanning.cpp`.
- Current code derives MAcc route-operand binding plans inside the central
  `deriveRVVRouteOperandBindingPlan(...)` switch instead of through the MAcc
  owner boundary.
- `RVVEmitCMAccRouteFamilyPlanOwners.cpp` already owns MAcc family consumer
  predicates and provider-plan verifiers, so this round should extend that
  existing boundary rather than add a second MAcc owner surface.

## Requirements

1. Extend the explicit RVV-owned MAcc owner boundary with an owner-owned
   interface for MAcc route-operand binding plan identity and logical operand
   to runtime ABI role validation.
2. Move the plain MAcc, scalar-broadcast MAcc, computed-mask MAcc, and
   runtime-scalar computed-mask MAcc operand-binding plan IDs out of central
   `RVVEmitCRoutePlanning.cpp` and into `RVVEmitCMAccRouteFamilyPlanOwners.cpp`.
3. Move MAcc-specific logical operand to `RuntimeABIParameterRole` validation
   out of central `getExpectedRVVRouteOperandBindingRole(...)`; central
   validation may call the MAcc owner neutrally.
4. Move or delegate MAcc-specific route-operand binding plan construction so
   central `deriveRVVRouteOperandBindingPlan(...)` no longer locally assembles
   MAcc logical operands from operation names.
5. The owner-owned MAcc binding logic must validate accumulator/result binding,
   plain lhs/rhs/acc/out/n roles, scalar-broadcast `rhs_scalar`, computed-mask
   compare lhs/rhs plus dot lhs/rhs, runtime-scalar compare RHS scalar, runtime
   `n`/AVL/VL use, and stale/mismatched plan identity before provider
   materialization.
6. Preserve existing MAcc provider-plan verification, statement-plan
   construction, route-control consumers, materialization facts, and generated
   artifact behavior. No new route coverage or runtime claim is allowed.
7. Keep central route planning limited to shared containers, neutral helper
   validation, typed-fact collection, and aggregate dispatch into owner
   boundaries. It must not choose MAcc semantics from operation names, route
   IDs, ABI strings, helper names, descriptors, artifacts, scripts, exact
   intrinsic spellings, or common EmitC code.
8. Add or update focused C++ tests proving the MAcc owner API supplies plan IDs
   and role validation for plain, scalar-broadcast, computed-mask, and
   runtime-scalar computed-mask MAcc, including fail-closed wrong-role cases.
9. Run focused generated-bundle dry-runs for the existing MAcc selected-body
   paths and non-regression for control-policy consumers that depend on MAcc
   predicates.

## Acceptance Criteria

- [x] `RVVEmitCMAccRouteFamilyPlanOwners.h/cpp` expose and implement the
      MAcc-owned operand-binding plan API.
- [x] `RVVEmitCRoutePlanning.cpp` no longer defines MAcc operand-binding plan
      ID constants or MAcc logical-operand-to-runtime-role mapping bodies.
- [x] Central route planning delegates MAcc binding plan identity, binding plan
      construction, and role validation through the MAcc owner boundary while
      preserving neutral route analysis and closure checks.
- [x] MAcc provider-plan verifiers use the owner-owned expected binding plan ID
      rather than central generic MAcc constants.
- [x] Focused C++ tests cover owner plan-ID selection and role validation for
      `macc_add`, `scalar_broadcast_macc_add`, `computed_masked_macc_add`, and
      `runtime_scalar_cmp_masked_macc_add`, including wrong-role diagnostics.
- [x] Existing plain, scalar-broadcast, computed-mask, and runtime-scalar
      computed-mask MAcc generated-bundle dry-runs pass.
- [x] Control-policy consumers that depend on MAcc owner predicates continue to
      pass focused C++ coverage.
- [x] Bounded symbol scan shows moved MAcc operand-binding owner symbols and
      MAcc-specific operand-role logic concentrated in the MAcc owner module.
- [x] Authority scan over touched production/test files finds no new
      legacy-i32, route-id-as-authority, ABI-string-as-authority,
      descriptor/source-front-door/artifact-name/exact-intrinsic/common-EmitC,
      direct-route-entry-only, script-derived, or mirror-only authority drift.
- [x] `git diff --check` passes.
- [x] Focused RVV plugin C++ build/test passes.
- [x] `tcrv-opt` and `tcrv-translate` build.
- [x] `check-tianchenrv` passes, or the exact blocker is recorded.
- [x] `ssh rvv` is not required unless this round changes emitted runtime
      semantics or claims new runtime/correctness/performance behavior.

## Technical Approach

Add a MAcc-owner API beside the existing provider-plan owner API:

```text
selected typed MAcc route analysis
  -> central neutral route analysis/fact collection
  -> MAcc owner derives/validates route-operand binding plan identity and roles
  -> central generic closure comparison against route description mirrors
  -> MAcc owner provider-plan verifier
  -> materialization facts / statement plans
  -> RVVEmitCRouteProvider builds TCRVEmitCLowerableRoute
```

The owner module will carry the four bounded MAcc plan IDs, expose an
operation-to-plan-ID query, expose a plan-ID/logical-operand-to-runtime-role
query, and build the MAcc route-operand binding plan from the typed selected
route analysis. Central route planning will keep the shared
`RVVRouteOperandBindingPlan` struct and shared closure validation, but it will
delegate MAcc-specific cases to the owner before falling through to other
families.

## Out of Scope

- New MAcc variants, widening dot-reduce expansion, standalone reduction
  changes, conversion/dtype/LMUL clone batches, segment2 changes, memory
  movement changes, compare/select changes, high-level frontend/Linalg
  lowering, selected-body realization rewrites, dashboards, tuning databases,
  source-front-door routes, one-intrinsic wrappers, or runtime performance
  claims.
- Moving neutral shared route-analysis structs, common closure helpers,
  provider-built route mechanics, or common EmitC materialization into the
  MAcc owner.
- Reclassifying widening MAcc/direct contraction ownership in this round.

## Validation Plan

1. Validate task context with
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-05-30-stage2-rvv-macc-operand-binding-owner-completion`.
2. Build focused plugin test:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`.
3. Run focused plugin C++ coverage:
   `./build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build touched tools:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
5. Run representative generated-bundle dry-runs for `macc_add`,
   `scalar_broadcast_macc_add`, `computed_masked_macc_add`, and
   `runtime_scalar_cmp_masked_macc_add`.
6. Run bounded symbol and authority scans over touched files.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Completion Evidence

- Added MAcc owner APIs in
  `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h` and
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp` for
  owner-owned operand-binding plan ID lookup, logical operand to runtime ABI
  role lookup, and MAcc binding-plan derivation from selected route analysis.
- Moved the four MAcc operand-binding plan ID constants and MAcc logical
  operand role mapping bodies out of central `RVVEmitCRoutePlanning.cpp`.
- Rewired central `deriveRVVRouteOperandBindingPlan(...)` to dispatch MAcc
  sub-families to the owner boundary while keeping shared closure validation
  in central route planning.
- Rewired plain/scalar/computed-mask MAcc provider-plan verifiers to compare
  binding plan identity against the owner-owned plan-id API. The shared
  computed-mask accumulation verifier still lets standalone accumulation use
  its non-MAcc shared plan-id path.
- Added focused C++ coverage in `test/Plugin/RVVExtensionPluginTest.cpp` for
  MAcc owner plan-id selection, MAcc owner role mapping, and wrong-role
  fail-closed diagnostics for scalar-broadcast, computed-mask, and
  runtime-scalar computed-mask MAcc binding plans.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to record that MAcc
  plan ids, MAcc logical operands, and MAcc logical-operand-to-runtime-ABI-role
  mapping are owner-boundary authority.
- Checks run:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-05-30-stage2-rvv-macc-operand-binding-owner-completion`;
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`;
  `./build/bin/tianchenrv-rvv-extension-plugin-test`;
  `cmake --build build --target tcrv-opt tcrv-translate -j2`;
  direct generated-bundle dry-runs for `macc_add`,
  `scalar_broadcast_macc_add`, `computed_masked_macc_add`, and
  `runtime_scalar_cmp_masked_macc_add`; bounded central/owner symbol scans;
  bounded authority scan; `git diff --check`; and
  `cmake --build build --target check-tianchenrv -j2`.
- Focused C++ result: `RVV extension plugin smoke test passed`.
- Generated-bundle dry-run result: all four selected-body MAcc paths emitted
  plan and header artifacts with the expected owner-derived route operand
  binding plan IDs and logical operand summaries.
- Final `check-tianchenrv` result: 464/464 passed.
- `ssh rvv` was not run because this task changed owner placement and
  validation authority only, with no new runtime, correctness, or performance
  claim.
