# Stage2 RVV contraction and widening dot-reduce route-family ownership

## Goal

Repair and complete the RVV plugin-local contraction route-family ownership
boundary for the active widening multiply-accumulate and widening dot-reduce
routes. Provider materialization for `widening_macc_add`,
`widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
`computed_masked_widening_dot_reduce_add`, and
`computed_masked_strided_input_widening_dot_reduce_add` must depend on a
validated contraction family plan derived from typed `tcrv_rvv`
body/config/runtime facts, not on route ids, helper names, description mirrors,
target metadata, artifact names, scripts, or common EmitC/export logic.

## Direction Source

- Direction title: `Stage2 RVV contraction and widening dot-reduce route-family ownership`.
- Module owner: RVV plugin-local contraction family boundary for active
  widening MAcc and widening dot-reduce routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `bb51b594 rvv: own computed mask macc accumulation family`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes/Codex Direction Brief before source edits.

## What I Already Know

- Long-term specs require RVV executable authority to flow through selected
  `tcrv.exec` variant, typed low-level `tcrv_rvv` body, RVV plugin-owned
  legality/realization/provider, provider-built `TCRVEmitCLowerableRoute`,
  common EmitC materialization, target artifact packaging, and real `ssh rvv`
  evidence for runtime/correctness claims.
- Common EmitC/export may materialize provider output and serialize mirrors,
  but it must not choose contraction semantics, widening product semantics,
  mask policy, dot operand roles, accumulator/result contracts, dtype/config,
  intrinsic spelling, ABI order, support state, or evidence authority.
- Current code already contains
  `RVVSelectedBodyContractionRouteFamilyPlan` and derives it for the five active
  contraction routes. This task should repair that existing plan's ownership
  boundary and evidence rather than adding duplicate wrappers or metadata-only
  mirrors.
- Current active contraction routes are:
  - `widening_macc_add`;
  - `widening_dot_reduce_add`;
  - `strided_input_widening_dot_reduce_add`;
  - `computed_masked_widening_dot_reduce_add`;
  - `computed_masked_strided_input_widening_dot_reduce_add`.
- Adjacent prior tasks already own computed-mask MAcc accumulation,
  standalone reductions, memory movement, and computed-mask select families.
  This task must preserve those boundaries and avoid expanding them.

## Scope

1. Inventory only active widening MAcc and widening dot-reduce production routes
   and directly adjacent provider predicates.
2. Repair the existing contraction family plan boundary so provider
   materialization requires a validated plan for the five active routes and
   rejects stale/missing plans for contraction and non-contraction cases.
3. Carry or verify, through planning-owned facts:
   - family plan id;
   - operation kind;
   - memory form;
   - widening MAcc versus dot-reduction distinction;
   - unit-stride versus strided input distinction;
   - optional computed-mask facts;
   - dot LHS/RHS roles;
   - accumulator role and result layout;
   - widening product or widening MAcc leaf facts;
   - runtime AVL/VL and runtime ABI order;
   - dtype/config and target leaf/header facts;
   - `RouteOperandBindingPlan` closure.
4. Preserve current route ids, ABI names/order, parameter roles, runtime
   `n`/AVL behavior, selected-body realization semantics, generated artifact
   shape, target header contract, and common EmitC/export neutrality.

## Requirements

1. Add or expose planning-owned contraction family consumer predicates for:
   - plain widening MAcc;
   - plain widening dot-reduce;
   - strided-input widening dot-reduce;
   - computed-mask widening dot-reduce;
   - computed-mask strided-input widening dot-reduce.
2. Provider materialization must call a planning-owned verifier before route
   construction. The verifier must require `RVVSelectedBodyContractionRouteFamilyPlan`
   for contraction consumers and reject stale contraction plans on non-consumers.
3. The plan must contain an explicit family plan id mirror, and the route
   description/target metadata/generated-bundle evidence may mirror it only
   after planning/provider validation.
4. The verifier must check operation/memory-form consistency, widening MAcc
   versus dot-reduction booleans, computed-mask and strided-input booleans,
   runtime ABI order, target leaf/header/type facts, source/result vector
   facts, contraction leaf choices, mask facts where applicable, strided-input
   source facts where applicable, accumulator/result contracts, and runtime ABI
   parameter closure.
5. Provider route construction must use the validated plan for materialized
   booleans and emitted leaf/type/header facts. Description mirrors may be used
   only as consistency checks or final serialized mirrors.
6. `RouteOperandBindingPlan` closure must remain required and visible for all
   five routes, including dot LHS/RHS, accumulator, output, runtime count, and
   stride operands where active.
7. Selected-body realization tests must prove the family plan survives
   pre-realized selected bodies for representative active routes without
   changing computation semantics, parameter roles, dispatch/fallback, ABI
   order, runtime AVL, route ids, or target artifact contracts.
8. No positive legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
   `!tcrv_rvv.i32m*`, descriptor/direct-C/source-front-door, helper-string,
   artifact-name, or mirror-only authority may be introduced.

## Acceptance Criteria

- [ ] PRD and task context reference the RVV plugin, EmitC route, plugin
      protocol, testing, variant-pipeline specs, previous computed-mask MAcc
      task, and focused contraction source/test surfaces.
- [ ] Active contraction routes are inventoried without unrelated route
      expansion.
- [ ] Provider materialization for all five active contraction routes requires
      a validated contraction route-family plan through planning-owned
      consumer predicates and provider-plan verifier.
- [ ] The plan has an explicit family plan id mirrored through description,
      target metadata, and generated-bundle evidence after route construction.
- [ ] Focused lit/FileCheck coverage proves family plan id, operation/memory
      form facts, widening MAcc versus dot-reduction facts, dot operand roles,
      accumulator/result contracts, computed-mask and strided-input
      distinctions, route operand binding closure, header/mirror stability, and
      selected-body realization behavior.
- [ ] Generated-bundle dry-runs pass for representative active explicit and
      pre-realized routes across counts `7,16,23`.
- [ ] Real `ssh rvv` evidence passes for `widening_macc_add` and at least one
      widening dot-reduce route, plus a computed-mask or strided-input
      representative if active, proving widened product/accumulation
      correctness, dot reduction correctness, accumulator behavior,
      active/inactive lane accounting, tail preservation, and runtime `n`
      variation.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive legacy or mirror-only route authority.
- [ ] Focused checks, `git diff --check`, `check-tianchenrv`, task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No standalone reductions, memory routes, plain/computed-mask MAcc coverage,
  computed-mask select coverage, dtype/LMUL clone batches, source-front-door
  routes, high-level Linalg routes, dashboards, broad test matrices,
  helper-only cleanup, or report-only evidence packaging.
- No computation semantics change, parameter role change, runtime n/AVL change,
  dispatch/fallback change, ABI order change, route id change, or target
  artifact contract change except to preserve existing behavior through the
  ownership boundary.
- No common EmitC/export ownership of contraction semantics, dtype/config,
  mask policy, intrinsic spelling, support state, runtime ABI, or evidence
  authority.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test`.
3. Run focused lit/FileCheck tests for explicit and pre-realized contraction
   route target/header/script fixtures.
4. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if route planning,
   provider helpers, binding helpers, or target metadata helpers are changed.
5. Run generated-bundle dry-runs for representative explicit and pre-realized
   contraction routes with counts `7,16,23`.
6. Run real `ssh rvv` generated-bundle evidence for `widening_macc_add`, one
   widening dot-reduce route, and one computed-mask or strided-input route if
   active.
7. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`

Relevant prior task read:

- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-macc-accumulation-route-family-ownership/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-macc-accumulation-route-family-ownership/implement.jsonl`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-macc-accumulation-route-family-ownership/check.jsonl`

Initial source surfaces inspected:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Focused widening MAcc and widening dot-reduce tests under
  `test/Target/RVV` and `test/Scripts`.

## Definition Of Done

Every active widening MAcc and widening dot-reduce route consumes an explicit
validated plugin-local contraction route-family plan before provider
materialization, target/header/artifact fields mirror the already validated
plan, common EmitC/export remains neutral, focused tests and generated-bundle
evidence pass, real `ssh rvv` evidence is collected for representative active
routes, the task is finished/archived, and one coherent commit records the
work.

## Completion Evidence

Production changes:

- Added `rvv-contraction-route-family-plan.v1` as the explicit contraction
  family plan mirror for active widening MAcc and widening dot-reduce routes.
- Added planning-owned contraction family consumer predicates and provider-plan
  verification.
- Made the RVV EmitC provider require the validated contraction plan before
  materializing contraction routes.
- Strengthened contraction plan validation for operation/memory-form
  classification, runtime ABI order, target leaf/header/type facts,
  strided-input facts, computed-mask facts, accumulator/result contracts, and
  route operand binding closure.
- Mirrored the validated family plan through route description, target metadata,
  target header export, and generated-bundle evidence.
- Preserved existing route ids, ABI names/order, parameter roles, runtime
  `n`/AVL behavior, selected-body realization semantics, generated artifact
  contracts, and common EmitC/export neutrality.

Active routes covered:

- `widening_macc_add`
- `widening_dot_reduce_add`
- `strided_input_widening_dot_reduce_add`
- `computed_masked_widening_dot_reduce_add`
- `computed_masked_strided_input_widening_dot_reduce_add`

Routes intentionally out of scope:

- Standalone reductions, memory route families, computed-mask select, and
  computed-mask MAcc accumulation remain owned by prior family tasks.
- Explicit-selected-body generated-bundle mode for these contraction op kinds
  is currently unsupported by `scripts/rvv_generated_bundle_abi_e2e.py`; the
  active generated-bundle surface for this family is pre-realized selected-body.
- Dtype/LMUL clone batches, source-front-door routes, high-level Linalg routes,
  dashboards, and report-only evidence were not touched.

Selected-body realization and binding closure evidence:

- Focused manual FileCheck for all five pre-realized target fixtures passed
  REALIZED, PLAN, and HEADER checks.
- PLAN/HEADER checks now assert
  `tcrv_rvv.contraction_route_family_plan =
  rvv-contraction-route-family-plan.v1`.
- Route operand binding mirrors remain operation-specific:
  `widening_macc_add`, `widening_dot_reduce`, `strided_widening_dot_reduce`,
  `masked_widening_dot_reduce`, and `masked_strided_wdot`.

Generated-bundle evidence:

- Pre-realized dry-runs passed for all five active op kinds with runtime counts
  `7,16,23`.
- Evidence JSON and generated target headers contain
  `tcrv_rvv.contraction_route_family_plan:
  rvv-contraction-route-family-plan.v1`.
- Attempted explicit-selected-body dry-run for `widening_macc_add` failed
  before bundle generation with the script diagnostic:
  `--op-kind values ['widening_macc_add'] are not supported in
  explicit-selected-body mode`.

Real `ssh rvv` evidence:

- `widening_macc_add` passed counts `7,16,23` with
  `signed_products accumulation tail_preserved`.
- `widening_dot_reduce_add` passed counts `7,16,23` with
  `signed_horizontal_dot seed_added scalar_output tail_preserved`.
- `computed_masked_strided_input_widening_dot_reduce_add` passed counts
  `7,16,23` with
  `compare_masked_strided_signed_horizontal_dot seed_added
  inactive_lanes_skipped source_strides=2,3 skipped_source_elements_ignored
  scalar_output tail_preserved`.

Checks:

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- Focused target FileCheck commands for five pre-realized contraction fixtures
  passed.
- Focused generated-bundle evidence FileCheck commands passed for the four
  script fixtures updated in this task.
- Added-line active-authority scan found no new positive legacy or mirror-only
  route authority.
- Full touched-file scan only found existing negative/guardrail mentions of
  descriptor/direct-C/source-export and stale `tcrv_rvv.i32_` rejection.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed 351/351.

Spec update judgment:

- No `.trellis/spec/**` update was needed. This task implemented the existing
  RVV plugin ownership, common EmitC neutrality, and testing contracts without
  creating a new long-term rule.
