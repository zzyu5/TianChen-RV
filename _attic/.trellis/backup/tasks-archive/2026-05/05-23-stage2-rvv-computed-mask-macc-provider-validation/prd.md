# Stage2 RVV computed-mask MAcc accumulation provider validation

## Goal

Deepen the RVV plugin-local computed-mask MAcc accumulation route-family
provider validation so active `ComputedMaskedMAccAdd` and
`RuntimeScalarComputedMaskedMAccAdd` routes are materialized only after the
provider has checked the real typed RVV mask, accumulator, body, config,
runtime, ABI, and binding facts carried by the computed-mask accumulation
family plan.

This is a bounded provider-boundary repair. It does not add new RVV coverage.
It tightens the existing computed-mask accumulation family boundary and checks
only the directly shared computed-mask standalone-reduction consumers needed to
prove that the accumulation and standalone-reduction provider boundaries remain
isolated.

## Current-head facts

- `.trellis/.current-task` was absent when this task was created.
- The worktree was clean before the task was created.
- Recent HEAD is `1cfd990e rvv: deepen standalone reduction provider
  validation`.
- The archived `05-23-stage2-rvv-standalone-reduction-provider-validation`
  task repaired standalone reduction provider validation and left computed-mask
  accumulation as adjacent work.
- Current planning has an
  `RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan` for computed-mask
  accumulation routes.
- The current computed-mask accumulation verifier is known to validate a
  narrower subset of available plan facts than the route-family plan carries:
  family id, compute suffix, mask producer, accumulator/result/scalar-carry
  contracts, MAcc-vs-standalone suffix selection, and binding plan id.
- The plan also carries runtime control, runtime ABI parameters, target leaf,
  header and type mirrors, vector/mask/VL types, setvl/load/rhs-splat/compare
  and store intrinsic mirrors, mask/source/destination memory forms,
  inactive-lane and passthrough contracts, and binding closure facts.

## Requirements

- Inventory active `ComputedMaskedMAccAdd` and
  `RuntimeScalarComputedMaskedMAccAdd` production routes and the directly
  shared computed-mask standalone-reduction consumers.
- Require computed-mask MAcc accumulation consumers to carry a computed-mask
  accumulation family plan before provider materialization.
- Reject stale computed-mask accumulation family plans on non-consumer routes.
- Validate operation identity, family plan id, compute suffix, vector-MAcc
  versus standalone-reduction suffix selection, and vector-compare versus
  runtime-scalar mask producer selection.
- Validate runtime control facts: VL/AVL source, runtime `n`/count use, and
  relevant runtime-control mirror ids.
- Validate runtime ABI order and parameter mirrors, including source,
  destination, accumulator, scalar/runtime threshold, and count parameters as
  applicable.
- Validate target/profile mirrors: target leaf profile, provider-supported
  mirror, required header list, header declaration mirror, C scalar/vector/mask
  type mirrors, and VL C type.
- Validate intrinsic mirrors for setvl, vector load, accumulator load when
  present, rhs scalar splat, compare/mask construction, MAcc compute, and store.
- Validate mask role/source/memory form, source memory form, destination memory
  form, accumulator/result contracts, inactive-lane contract, passthrough
  contract, and scalar-carry contract.
- Validate RouteOperandBindingPlan closure using the selected route
  description's binding mirrors before provider materialization.
- Preserve standalone reduction provider validation as a separate family
  boundary; include only the directly shared computed-mask standalone-reduction
  cases as isolation checks.
- Preserve plain/widening MAcc contraction boundaries; do not re-open or
  broaden them in this task.
- Keep mirror fields mirror-only: they may be checked after RVV planning builds
  the provider family plan, but they must not become common EmitC/export route
  authority.

## Out of scope

- No redo of standalone reduction, plain compare-select, elementwise
  arithmetic, base memory movement, scalar broadcast, runtime splat-store,
  widening conversion, plain/widening MAcc contraction, segment2, high-level
  frontend/Linalg, dtype/LMUL expansion, or broad evidence dashboards.
- No new route family or operation coverage beyond the existing active
  computed-mask MAcc accumulation provider boundary.
- No movement of mask, accumulation, accumulator, runtime scalar, or intrinsic
  semantics into common EmitC/export, artifact names, route ids, helper strings,
  descriptors, or mirror fields.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-artifact, descriptor, or exact
  intrinsic spelling as route authority.

## Acceptance Criteria

- [ ] `verifyRVVSelectedBodyComputedMaskAccumulationRouteFamilyProviderPlans`
      performs deep computed-mask accumulation plan validation comparable in
      depth to adjacent repaired family verifiers.
- [ ] Provider materialization rejects missing accumulation plans for active
      computed-mask MAcc consumers and rejects stale accumulation plans on
      non-consumers.
- [ ] Provider validation rejects mirror mismatches for operation, memory form,
      runtime control, ABI order/parameters, target/header/type facts,
      vector/mask/VL type facts, setvl/load/splat/compare/MAcc/store
      intrinsics, mask/source/destination memory forms, accumulator/result
      contracts, inactive-lane/passthrough/scalar-carry contracts, suffix
      selection, mask producer selection, and RouteOperandBindingPlan closure.
- [ ] Active `ComputedMaskedMAccAdd` and
      `RuntimeScalarComputedMaskedMAccAdd` routes remain positive and coherent.
- [ ] Directly shared computed-mask standalone-reduction consumers are covered
      only as provider-boundary isolation checks.
- [ ] Plain/widening MAcc contraction routes remain excluded with focused
      checks or code evidence showing no accidental coupling.
- [ ] Focused C++ and/or lit/FileCheck coverage proves deep validation,
      missing-plan and stale-plan fail-closed cases, mirror mismatch failures,
      runtime ABI/binding closure, mask producer selection, vector-MAcc suffix
      selection, accumulator/result/inactive-lane contracts, and isolation from
      standalone reduction and plain/widening MAcc routes.
- [ ] Generated-bundle dry-runs cover representative explicit and pre-realized
      `computed_masked_macc_add` and
      `runtime_scalar_cmp_masked_macc_add` cases at counts 7, 16, and 23.
- [ ] Real `ssh rvv` evidence covers representative explicit and pre-realized
      computed-mask MAcc cases, including masked accumulation values,
      false-lane accumulator preservation, runtime threshold variation, tail
      preservation, and runtime `n` variation.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      shows no new legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
      source-front-door/source-artifact, descriptor, or common/export route
      authority.
- [ ] `check-tianchenrv`, focused changed-behavior checks, `git diff --check`,
      and clean git status pass before finish.

## Relevant specs and evidence inputs

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- Archived task:
  `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-standalone-reduction-provider-validation/`
- Current code around
  `RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan`
- Current verifier
  `verifyRVVSelectedBodyComputedMaskAccumulationRouteFamilyProviderPlans`
- Current provider/realization/target-support paths:
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`, and
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- Generated-bundle runner `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing computed-mask MAcc lit fixtures under `test/Target/RVV/`

## Completion report requirements

The final report must state task id/title, current phase, module behavior
completed, changed files, provider validation fields now checked, active
computed-mask MAcc routes covered, adjacent standalone-reduction and
plain/widening MAcc boundaries included or excluded with reason,
selected-body realization evidence, binding closure evidence, generated-bundle
and `ssh rvv` commands/results, checks, active-authority scan, self-repair,
finish/archive status, commit hash, and the exact continuation point if
unfinished.
