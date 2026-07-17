# Stage2 RVV standalone reduction route-family ownership and runtime evidence

## Goal

Repair and complete the RVV plugin-local standalone reduction route-family
ownership boundary for the active plain, computed-mask, and runtime-scalar
computed-mask standalone reduction routes. This is a production-path migration
and evidence task, not new coverage: provider support must depend on a
validated standalone reduction family plan derived from typed `tcrv_rvv`
body/config/runtime facts, and the generated artifacts must keep existing
semantics.

## Direction Source

- Direction title: `Stage2 RVV standalone reduction route-family ownership and runtime evidence`.
- Module owner: RVV plugin-local standalone reduction route-family path for
  active plain and computed-mask standalone reduction routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `a846a226 rvv: extract memory route family ownership`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes/Codex Direction Brief before source edits.

## What I Already Know

- Long-term specs require the RVV route authority chain to flow from selected
  `tcrv.exec` variant, through typed `tcrv_rvv` body/config/runtime facts, RVV
  plugin-owned legality/realization/route provider, provider-built
  `TCRVEmitCLowerableRoute`, common EmitC materialization, target artifact, and
  real `ssh rvv` evidence for runtime/correctness claims.
- Common EmitC/export may consume provider output and mirrors. It must not
  infer standalone reduction operation kind, accumulator/result layout, mask
  producer facts, dtype/config, runtime AVL/VL, support state, or ABI order
  from route ids, helper strings, artifact names, scripts, manifests, status,
  or mirror fields.
- Current code already has `RVVSelectedBodyStandaloneReductionRouteFamilyPlan`
  plus derivation/validation/application helpers. This task must repair the
  ownership boundary if it is partial; it must not add a duplicate wrapper.
- Recent completed memory-family ownership extraction moved provider
  materialization behind planning-owned family predicates/verifiers and kept
  common EmitC/export neutral. This task should apply the same discipline to
  standalone reductions where needed.
- The active standalone reduction operation inventory from current code/tests
  is:
  - `standalone_reduce_add`;
  - `standalone_reduce_min`;
  - `standalone_reduce_max`;
  - `computed_mask_standalone_reduce_add`;
  - `computed_mask_standalone_reduce_min`;
  - `computed_mask_standalone_reduce_max`;
  - `runtime_scalar_cmp_masked_standalone_reduce_add`.
- The runtime-scalar route also participates in the existing computed-mask
  accumulation family for shared threshold-compare/mask producer mechanics.
  This task must preserve that shared producer boundary while making
  standalone-reduction suffix ownership explicit and provider-required.

## Scope

1. Inventory only the active standalone reduction production routes listed
   above and their directly related tests/scripts.
2. Repair or complete explicit RVV plugin-local ownership for:
   - standalone reduction family consumer predicates;
   - family plan derivation and validation;
   - description application and metadata mirror production;
   - provider-side plan requirements before materialization;
   - route operand binding closure;
   - target leaf/header facts;
   - generated-bundle expectations and runtime evidence.
3. Preserve existing route names, operation semantics, selected-body
   realization behavior, runtime ABI order, scalar seed/result behavior,
   inactive/tail behavior, target artifact behavior, and common EmitC/export
   neutrality.
4. Cover plain and computed-mask `add/min/max`; include the active
   runtime-scalar computed-mask standalone reduction add route as the
   repository-equivalent runtime-scalar masked standalone reduction path.

## Requirements

1. Provider support must require a validated standalone reduction route-family
   plan before any active standalone reduction route is materialized.
2. The family plan must carry or validate:
   - operation kind and add/min/max relation;
   - memory form: plain, computed-mask, or runtime-scalar computed-mask
     unit-stride standalone reduction;
   - accumulator seed layout;
   - scalar result layout and reduction store VL;
   - optional computed-mask producer facts;
   - runtime ABI order and runtime ABI parameter closure;
   - `RouteOperandBindingPlan` closure and materialized operand uses;
   - SEW/LMUL/policy-derived type/header/intrinsic facts;
   - target leaf/profile facts and provider-supported mirror;
   - fail-closed diagnostics for missing/unsupported typed-body facts.
3. Plain standalone reductions must continue to bind `lhs`, scalar seed
   `acc`, scalar result `out`, and runtime `n`, then store the scalar result
   with the standalone reduction store VL.
4. Computed-mask standalone reductions must continue to bind compare `cmp_lhs`
   and `cmp_rhs`, payload `src`, scalar seed `acc`, scalar result `out`, and
   runtime `n`; `add` must zero inactive lanes before reduction, while
   `min/max` must use the operation-appropriate neutral value.
5. Runtime-scalar computed-mask standalone reduction must continue to bind
   `cmp_lhs`, scalar threshold `rhs_scalar`, payload `src`, scalar seed `acc`,
   scalar result `out`, and runtime `n`; it must preserve the existing shared
   computed-mask accumulation producer plan where applicable.
6. Description fields and target metadata may mirror plan facts only after
   planning/provider decisions. Mirror-only fields, route ids, scripts, helper
   names, artifact names, or common/export code must not choose reduction
   semantics or support state.
7. No positive legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
   `!tcrv_rvv.i32m*`, descriptor/direct-C/source-front-door, helper-string,
   route-id, artifact-name, or mirror-only route authority may be introduced.

## Acceptance Criteria

- [x] PRD and task context reference the RVV plugin, EmitC route, plugin
      interface, testing specs, previous memory-family extraction task, and
      directly relevant accumulation/standalone-reduction prior tasks.
- [x] Current active standalone reduction routes are inventoried without
      broad all-route rewrites.
- [x] Provider materialization for every active standalone reduction route
      requires a validated standalone reduction family plan rather than local
      route predicates, route ids, helper names, mirrors, scripts, or common
      exporter inference.
- [x] Plain and computed-mask `add/min/max` route semantics, accumulator seed
      layout, scalar result layout, reduction store VL, runtime ABI order, and
      `RouteOperandBindingPlan` closure remain preserved.
- [x] Runtime-scalar computed-mask standalone reduction remains supported only
      through the active typed body/config/runtime facts and shared
      computed-mask producer mechanics, with standalone-reduction suffix facts
      still owned by the standalone reduction family plan.
- [x] Target/header mirrors explicitly expose family-plan/provider-support
      facts where useful and remain mirrors only.
- [x] Focused lit/FileCheck coverage for representative explicit and
      pre-realized plain, computed-mask, and runtime-scalar standalone
      reductions proves family plan presence, materialized operands, binding
      closure, accumulator/result layout, optional mask facts, headers, and
      fail-closed unsupported combinations.
- [x] Generated-bundle dry-runs pass for representative explicit and
      pre-realized standalone reduction routes, including add/min/max coverage
      where current fixtures exist.
- [x] Real `ssh rvv` evidence passes for at least one plain standalone
      reduction and one computed-mask or runtime-scalar masked standalone
      reduction across counts `7,16,23`, proving scalar result correctness,
      inactive/tail behavior, accumulator seed behavior, and runtime `n`
      variation.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive legacy or mirror-only route authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No memory routes, contraction/dot-product routes, broad accumulation
  rewrites, conversion/dtype clone batches, new LMUL families, frontend/source
  front-door routes, high-level Linalg routes, dashboards, helper-only cleanup,
  or report-only evidence packaging.
- No new standalone reduction operation coverage beyond the active
  `add/min/max` and runtime-scalar masked add routes already present.
- No collapse of plain, computed-mask, and runtime-scalar computed-mask
  semantics into one ambiguous compute route.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`,
  descriptor, direct-C, source-front-door, helper-string, route-id,
  artifact-name, or mirror-only route authority.
- No common EmitC/export ownership of reduction operation kind, mask producer,
  accumulator seed, scalar result layout, dtype/config, runtime AVL/VL, route
  support, or acceptance state.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test`.
3. Run focused lit/FileCheck tests for standalone reduction target artifacts,
   dataflow/fail-closed fixtures, and generated-bundle dry-run fixtures.
4. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if route planning,
   provider helpers, binding helpers, or target metadata helpers are changed.
5. Run generated-bundle dry-runs for representative explicit and pre-realized
   standalone reduction routes with counts `7,16,23`.
6. Run real `ssh rvv` generated-bundle evidence for at least one plain
   standalone reduction and one computed-mask or runtime-scalar masked
   standalone reduction with counts `7,16,23`.
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

Relevant prior tasks/journal entries read:

- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-memory-route-family-ownership-extraction/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-accumulation-route-family-interface/prd.md`
- `.trellis/workspace/codex/journal-13.md` sessions 162 and 166.

Initial source surfaces inspected:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- focused standalone-reduce tests under `test/Target/RVV`,
  `test/Dialect/RVV`, and `test/Scripts`.

## Definition Of Done

Every active standalone reduction route consumes an explicit validated
plugin-local standalone reduction family plan before provider materialization,
common EmitC/export remains neutral, focused tests and generated-bundle
evidence pass, real `ssh rvv` evidence is collected for representative plain
and masked standalone reductions, the task is finished/archived, and one
coherent commit records the work.

## Completion Evidence

Production changes:

- Added explicit `standaloneReductionRouteFamilyPlanID` mirror plumbing to the
  RVV route description and target metadata as
  `tcrv_rvv.standalone_reduction_route_family_plan`.
- Added `familyPlanID` to `RVVSelectedBodyStandaloneReductionRouteFamilyPlan`
  and validate it as `rvv-standalone-reduction-route-family-plan.v1`.
- Added planning-owned standalone-reduction family consumer predicates and
  `verifyRVVSelectedBodyStandaloneReductionRouteFamilyProviderPlans`.
- Made RVV EmitC provider materialization require the validated standalone
  reduction family plan before handling plain, computed-mask, or
  runtime-scalar computed-mask standalone reduction routes.
- Replaced provider-local plain/computed-mask standalone reduction route
  predicates with plan-derived booleans for materialization branches.
- Updated target support metadata mapping and generated-bundle metadata
  expectations for the standalone reduction route-family plan.
- Added focused explicit and pre-realized FileCheck assertions for plain,
  computed-mask, and runtime-scalar standalone reduction family plan mirrors.

Active routes covered:

- `standalone_reduce_add`
- `standalone_reduce_min`
- `standalone_reduce_max`
- `computed_mask_standalone_reduce_add`
- `computed_mask_standalone_reduce_min`
- `computed_mask_standalone_reduce_max`
- `runtime_scalar_cmp_masked_standalone_reduce_add`

Routes intentionally out of scope:

- Memory route families: already handled by the previous ownership extraction.
- Contraction/dot-product/widening accumulation routes: separate family/suffix
  ownership and not standalone scalar-output reductions.
- New dtype/LMUL/frontend/source-front-door routes: non-goals for this PRD.

Selected-body realization and binding closure evidence:

- Focused lit/FileCheck passed 21/21 selected tests for standalone reduction
  target artifacts, dialect dataflow, and script dry-run fixtures.
- Explicit and pre-realized FileCheck fixtures now prove
  `tcrv_rvv.standalone_reduction_route_family_plan`, route operand binding
  closure, accumulator/result layout, runtime control, and target/header
  mirrors for representative plain, computed-mask, and runtime-scalar routes.
- Generated-bundle dry-runs passed for explicit and pre-realized selected
  bodies covering all seven active standalone reduction op kinds, counts
  `7,16,23`, and runtime scalar thresholds `-37,91` where applicable.

Runtime evidence:

- Explicit `ssh rvv` generated-bundle run passed for:
  - `standalone_reduce_add`;
  - `computed_mask_standalone_reduce_add`;
  - `runtime_scalar_cmp_masked_standalone_reduce_add`.
- Pre-realized `ssh rvv` generated-bundle run passed for:
  - `standalone_reduce_add`;
  - `runtime_scalar_cmp_masked_standalone_reduce_add`.
- Runtime evidence covered counts `7,16,23`, seeds `-11,17`, and
  runtime-scalar thresholds `-37,91`, proving scalar result correctness,
  accumulator seed behavior, inactive-lane accounting, runtime `n` variation,
  and tail preservation.

Checks run:

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'standalone-reduce|standalone-reduction|runtime-scalar-cmp-masked-standalone|computed-mask-standalone'` from `build/test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run ... --run-id explicit_dry ...`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body ... --run-id pre_realized_dry ...`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py ... --run-id ssh_explicit_representative ...`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body ... --run-id ssh_pre_realized_representative ...`
- Added-line active-authority scan over touched RVV/plugin/export/script/test
  paths.
- Full touched-file authority scan; matches were existing descriptor guards,
  source-front-door self-tests, and existing intrinsic leaf spellings, not new
  added positive legacy route authority.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` passed 349/349.

Self-repair:

- Fixed a compile failure caused by defining provider-facing standalone
  family predicates/verifier inside the anonymous implementation namespace
  instead of the public planning boundary.
- Fixed FileCheck order failures by aligning PLAN assertions to diagnostic
  metadata order and HEADER assertions to target metadata mapping order.

Spec-update judgment:

- No `.trellis/spec/**` update was made. This task applies existing durable
  RVV plugin-owned route-family, typed-body authority, provider-built route,
  mirror-only metadata, and common EmitC neutrality rules; it did not introduce
  a new architecture rule or cross-layer contract beyond the concrete
  standalone reduction family plan mirror implemented here.
