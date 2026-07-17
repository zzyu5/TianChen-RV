# Stage2 RVV computed-mask MAcc accumulation route-family ownership

## Goal

Repair and complete the RVV plugin-local computed-mask accumulation
route-family ownership boundary for the active vector multiply-accumulate
routes `computed_masked_macc_add` and
`runtime_scalar_cmp_masked_macc_add`. This is a production-path ownership and
evidence task: provider materialization for those routes must depend on a
validated computed-mask accumulation family plan derived from typed
`tcrv_rvv` body/config/runtime facts, not on provider-local route predicates,
route ids, helper strings, target metadata, scripts, or common EmitC/export
logic.

## Direction Source

- Direction title: `Stage2 RVV computed-mask MAcc accumulation route-family ownership`.
- Module owner: RVV plugin-local computed-mask accumulation family boundary for
  active vector MAcc routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `e4457608 rvv: own standalone reduction route family`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes/Codex Direction Brief before source edits.

## What I Already Know

- Long-term specs require the active RVV route authority chain to flow through
  selected `tcrv.exec` variant, typed low-level `tcrv_rvv` body, RVV
  plugin-owned legality/realization/provider, provider-built
  `TCRVEmitCLowerableRoute`, common EmitC materialization, target artifact, and
  real `ssh rvv` evidence for runtime/correctness claims.
- Common EmitC/export may materialize provider output and serialize mirrors,
  but it must not choose MAcc semantics, mask producer facts, accumulator/result
  contracts, inactive-lane behavior, dtype/config, ABI order, support state, or
  artifact authority.
- Current code already contains
  `RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan` and derives it for
  `computed_masked_macc_add`,
  `runtime_scalar_cmp_masked_macc_add`, and the shared computed-mask standalone
  reduction producer path. This task should repair ownership and evidence
  around that existing plan rather than add a duplicate wrapper.
- The previous standalone-reduction task made standalone reductions require a
  validated standalone reduction family plan before provider materialization
  and retained the shared computed-mask accumulation producer plan for
  runtime-scalar masked standalone reduction. This task keeps that shared
  producer boundary but focuses on MAcc suffix ownership.

## Scope

1. Inventory only the active computed-mask MAcc production routes:
   - `computed_masked_macc_add`;
   - `runtime_scalar_cmp_masked_macc_add`.
2. Touch adjacent plain `macc_add` only to keep separation explicit: plain MAcc
   must remain outside the computed-mask accumulation family.
3. Repair or complete planning/provider ownership for:
   - computed-mask accumulation family consumer predicates;
   - provider-facing plan verifier;
   - vector compare versus runtime-scalar mask producer facts;
   - vector MAcc suffix facts;
   - accumulator/result/inactive-lane contracts;
   - runtime ABI order and runtime AVL/VL control;
   - `RouteOperandBindingPlan` closure;
   - target leaf/header mirrors and generated artifact behavior.
4. Preserve existing route names, ABI names, ABI order, typed body semantics,
   mask producer sources, accumulator input/output roles, inactive-lane
   behavior, result contract, generated artifact shape, target header mirrors,
   and common EmitC/export neutrality.

## Requirements

1. Provider materialization for `computed_masked_macc_add` and
   `runtime_scalar_cmp_masked_macc_add` must require a validated
   `RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan`.
2. The provider-facing boundary must be planning-owned and callable from the
   provider layer. If provider-local route predicates still act as authority,
   expose and use planning-owned predicates/verifiers instead.
3. The plan must validate:
   - operation kind and memory form;
   - `usesVectorMAccSuffix == true`;
   - `usesScalarHorizontalReductionSuffix == false`;
   - vector-compare producer for `computed_masked_macc_add`;
   - runtime-scalar producer for `runtime_scalar_cmp_masked_macc_add`;
   - compute suffix `vector-masked-macc-add`;
   - accumulator/result/inactive-lane/passthrough contracts;
   - runtime ABI order and runtime AVL/VL control;
   - SEW32/LMUL m1 typed config, mask type, vector type, header/type mapping,
     compare/store/load/setvl leaves;
   - runtime ABI parameter closure and route operand binding closure.
4. Plain `macc_add` must not require or carry the computed-mask accumulation
   plan. It remains a separate plain MAcc route with its existing route operand
   binding, accumulator layout, result layout, ABI, and artifact behavior.
5. Computed-mask standalone reduction routes may continue to consume the shared
   accumulation producer plan where already required, but this task must not
   add standalone reduction coverage or change standalone reduction semantics.
6. Description fields and target metadata may mirror plan facts only after the
   plan/provider decision. Mirror fields, route ids, scripts, helper names,
   artifact names, or common/export code must not choose support state or
   semantics.
7. No positive legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
   `!tcrv_rvv.i32m*`, descriptor/direct-C/source-front-door, helper-string,
   route-id, artifact-name, or mirror-only route authority may be introduced.

## Acceptance Criteria

- [ ] PRD and task context reference the RVV plugin, EmitC route, plugin
      interface, testing specs, previous standalone-reduction task, and focused
      MAcc source/test surfaces.
- [ ] Active computed-mask MAcc routes are inventoried without broad route
      rewrites or unrelated coverage expansion.
- [ ] Provider materialization for `computed_masked_macc_add` and
      `runtime_scalar_cmp_masked_macc_add` requires a validated
      computed-mask accumulation family plan through planning-owned
      predicates/verifier.
- [ ] `computed_masked_macc_add` proves vector compare producer source,
      vector MAcc suffix, accumulator/result contracts, inactive-lane
      passthrough, route operand binding closure, and target/header mirrors.
- [ ] `runtime_scalar_cmp_masked_macc_add` proves runtime-scalar producer
      source, vector MAcc suffix, accumulator/result contracts, inactive-lane
      passthrough, route operand binding closure, and target/header mirrors.
- [ ] Plain `macc_add` remains separate and does not gain computed-mask
      accumulation family metadata or provider-plan dependency.
- [ ] Focused lit/FileCheck coverage for explicit and pre-realized
      computed-mask MAcc routes proves family plan id, mask producer source,
      vector/runtime-scalar producer distinction, accumulator/result contracts,
      route operand binding closure, header/mirror stability, and selected-body
      realization behavior.
- [ ] Generated-bundle dry-runs pass for representative explicit and
      pre-realized computed-mask MAcc routes across counts `7,16,23`.
- [ ] Real `ssh rvv` evidence passes for `computed_masked_macc_add` and
      `runtime_scalar_cmp_masked_macc_add` across counts `7,16,23` and at
      least two thresholds/seeds or payload patterns where supported by the
      runner, proving active-lane MAcc correctness, accumulator
      preservation/overwrite contract, inactive/tail preservation, and runtime
      `n` variation.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive legacy or mirror-only route authority.
- [ ] Focused checks, `git diff --check`, `check-tianchenrv`, task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No standalone reduction expansion or rewrite beyond preserving the existing
  shared computed-mask producer boundary.
- No memory routes, dot-product/contraction routes, widening MAcc, dtype/LMUL
  clone batches, source-front-door routes, high-level Linalg routes,
  dashboards, broad test matrices, helper-only cleanup, or report-only
  evidence packaging.
- No computation semantics change, accumulator role change, runtime n/AVL
  change, mask semantics change, route id change, ABI name change, or target
  artifact contract change.
- No common EmitC/export ownership of MAcc operation kind, mask producer facts,
  accumulator/result contracts, inactive-lane policy, dtype/config, runtime
  AVL/VL, route support, or acceptance state.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test`.
3. Run focused lit/FileCheck tests for explicit/pre-realized
   `computed_masked_macc_add`, `runtime_scalar_cmp_masked_macc_add`, and
   adjacent `macc_add` separation where touched.
4. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if route planning,
   provider helpers, binding helpers, or target metadata helpers are changed.
5. Run generated-bundle dry-runs for representative explicit and pre-realized
   computed-mask MAcc routes with counts `7,16,23`.
6. Run real `ssh rvv` generated-bundle evidence for
   `computed_masked_macc_add` and `runtime_scalar_cmp_masked_macc_add` with
   counts `7,16,23` and threshold/seed variation where supported.
7. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`

Relevant prior task read:

- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-standalone-reduction-route-family-ownership/prd.md`

Initial source surfaces inspected:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Focused MAcc tests under `test/Target/RVV` and `test/Scripts`.

## Definition Of Done

Every active computed-mask MAcc route consumes an explicit validated
plugin-local computed-mask accumulation family plan before provider
materialization, plain `macc_add` remains separate, common EmitC/export remains
neutral, focused tests and generated-bundle evidence pass, real `ssh rvv`
evidence is collected for representative explicit and pre-realized
computed-mask MAcc routes, the task is finished/archived, and one coherent
commit records the work.

## Completion Evidence

Production changes:

- Added planning-owned computed-mask accumulation family consumer predicates:
  `isRVVSelectedBodyComputedMaskMAccAccumulationRouteFamilyConsumer` for the
  active MAcc routes and
  `isRVVSelectedBodyComputedMaskAccumulationRouteFamilyConsumer` for the
  shared accumulation family boundary.
- Added
  `verifyRVVSelectedBodyComputedMaskAccumulationRouteFamilyProviderPlans` to
  require a plan before provider materialization, reject stale plans on
  non-consumers, require selected operation/description mirror consistency,
  require vector MAcc suffix facts for MAcc consumers, preserve the shared
  standalone-reduction suffix for the existing standalone add producer path,
  and validate vector-compare versus runtime-scalar mask producer facts.
- Made the RVV EmitC provider call the planning-owned computed-mask
  accumulation verifier before route materialization.
- Replaced the provider-local computed-mask MAcc route predicate used in the
  materialization path with the planning-owned MAcc accumulation consumer
  predicate.
- Preserved existing route ids, ABI names/order, typed body semantics,
  accumulator/result layouts, inactive-lane behavior, generated artifacts, and
  common EmitC/export neutrality.

Active routes covered:

- `computed_masked_macc_add`
- `runtime_scalar_cmp_masked_macc_add`

Adjacent route intentionally checked but not moved into this family:

- `macc_add`: remains a plain MAcc route without computed-mask accumulation
  route-family plan metadata or dependency.

Routes intentionally out of scope:

- Standalone reductions: previous task owns the standalone reduction family
  plan; this task only preserves the existing shared computed-mask accumulation
  producer boundary for the standalone add/runtime-scalar add cases.
- Memory routes: already owned by previous memory family extraction tasks.
- Dot-product/contraction/widening MAcc: separate route families and not part
  of the active computed-mask MAcc boundary in this PRD.
- Dtype/LMUL clone batches, source-front-door routes, high-level Linalg
  routes, dashboards, or report-only evidence.

Selected-body realization and binding closure evidence:

- Focused lit/FileCheck passed 15/15 selected tests for explicit and
  pre-realized MAcc target artifacts, dialect dataflow, and generated-bundle
  dry-run fixtures.
- Explicit and pre-realized target/header fixtures now prove
  `rvv-computed-mask-accumulation-route-family-plan.v1`, vector versus
  runtime-scalar mask producer source, vector MAcc suffix, accumulator/result
  contracts, inactive-lane passthrough, route operand binding closure,
  provider-supported mirror, required headers, and C type mapping.
- New generated-bundle dry-run lit coverage covers
  `runtime_scalar_cmp_masked_macc_add` in explicit and pre-realized selected
  body mode, counts `7,16,23`, and `rhs_scalar` thresholds `-37,91`.
- Existing generated-bundle dry-run lit coverage for `computed_masked_macc_add`
  now asserts the accumulation family plan, suffix, mask producer source, and
  provider-supported mirror.

Generated-bundle dry-run evidence:

- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/codex-macc-validation --run-id explicit-macc-dry --overwrite --op-kind computed_masked_macc_add --op-kind runtime_scalar_cmp_masked_macc_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/bin/llvm-readobj-20`
  passed with `dry_run_success`.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/codex-macc-validation --run-id pre-realized-macc-dry --overwrite --op-kind computed_masked_macc_add --op-kind runtime_scalar_cmp_masked_macc_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/bin/llvm-readobj-20`
  passed with `dry_run_success`.

Runtime evidence:

- Explicit `ssh rvv` generated-bundle run passed for
  `computed_masked_macc_add` and `runtime_scalar_cmp_masked_macc_add`.
- Pre-realized `ssh rvv` generated-bundle run passed for
  `computed_masked_macc_add` and `runtime_scalar_cmp_masked_macc_add`.
- Runtime evidence covered counts `7,16,23`; runtime-scalar MAcc covered
  `rhs_scalar=-37,91`. Output proved active-lane MAcc correctness,
  inactive-lane accumulator preservation, add-only and mul-only distinguishing
  cases, runtime `n` variation, and tail preservation.

Checks run:

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'computed-masked-macc|runtime-scalar.*macc|macc-add'`
  from `build/test`: 15/15 selected tests passed after one FileCheck ordering
  repair.
- Generated-bundle dry-runs listed above.
- Explicit and pre-realized real `ssh rvv` generated-bundle runs listed above.
- Added-line active-authority scan over touched RVV/plugin/export/script/test
  paths: no positive legacy or mirror-only route authority added.
- Full touched-file active-authority scan: matches were existing negative
  FileCheck `implicit-check-not` lines and an existing `tcrv_rvv.i32_`
  rejection guard, not new positive authority.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` passed 351/351.

Self-repair:

- Fixed the new runtime-scalar generated-bundle harness FileCheck ordering by
  using `HARNESS-DAG` for checks that intentionally appear in different helper
  scopes.

Spec-update judgment:

- No `.trellis/spec/**` update was made. This task applies existing durable
  RVV plugin-owned route-family, typed-body authority, provider-built route,
  mirror-only metadata, selected-body realization, and common EmitC neutrality
  rules; it did not introduce a new architecture rule or cross-layer contract
  beyond the concrete computed-mask accumulation provider verifier.
