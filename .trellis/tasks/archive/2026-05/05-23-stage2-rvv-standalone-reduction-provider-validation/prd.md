# Stage2 RVV standalone reduction provider validation repair

## Goal

Repair the current-head RVV plugin-local standalone reduction route-family
provider validation so standalone horizontal reductions are materialized only
after the provider has checked the real typed RVV family facts carried by the
validated family plan and RouteOperandBindingPlan closure.

This task is a bounded validation repair. It does not add new RVV coverage. It
deepens the existing standalone reduction family boundary for active
StandaloneReduceAdd, StandaloneReduceMin, StandaloneReduceMax, and the directly
coupled computed-mask/runtime-scalar standalone-reduction cases where current
planning already shares the standalone reduction family boundary.

## Current-head facts

- `.trellis/.current-task` was absent when this task was created.
- The worktree was clean before the task was created.
- Recent HEAD history includes route-family ownership commits for compare-select,
  elementwise arithmetic, base memory movement, widening conversion, and scalar
  broadcast/splat-store.
- Current `RVVSelectedBodyStandaloneReductionRouteFamilyPlan` already carries
  operation, memory form, runtime control, runtime ABI order/parameters, target
  leaf/header/type/intrinsic mirrors, seed/reduction/store fields, accumulator
  and result layout, inactive-lane/mask contracts, result name, and
  computed-mask/runtime-scalar flags.
- Current `verifyRVVSelectedBodyStandaloneReductionRouteFamilyProviderPlans`
  only checks missing consumer plan, stale non-consumer plan, and operation
  match. This is shallower than newer family verifiers.
- Existing standalone reduction lit fixtures already assert many output mirrors,
  but provider-facing validation does not currently prove that provider
  materialization depends on those family-plan facts.
- Prior archived standalone reduction work and journal entries are useful
  history, but they are not current-head proof that this verifier is deep.

## Requirements

- Require standalone reduction consumers to carry a standalone reduction
  route-family plan before provider materialization.
- Reject standalone reduction family plans on non-consumer routes.
- Validate standalone reduction plan operation and memory form against the
  selected route description.
- Validate standalone reduction family-plan mirrors against route description:
  family plan id, runtime control plan, runtime ABI order, target leaf profile,
  provider-supported mirror, required headers/header declaration mirror, C type
  mapping, VL C type, vector type/name, setvl intrinsic, vector-load intrinsic,
  scalar seed/splat intrinsic, reduction intrinsic, compare and masked-merge
  intrinsics where applicable, store intrinsic, accumulator layout, result
  layout, reduction-store VL, inactive-lane/mask contracts, mask role/source/form,
  result name, and runtime ABI parameter list.
- Validate that computed-mask standalone reductions carry computed-mask fields
  only when the operation is a computed-mask/runtime-scalar standalone reduction,
  and that plain standalone reductions do not carry stale mask or runtime-scalar
  mirrors.
- Validate RouteOperandBindingPlan closure for standalone reduction routes using
  the selected description's route operand binding mirrors.
- Keep computed-mask accumulation plan validation separate, while preserving the
  existing shared accumulation boundary for computed-mask standalone add and
  runtime-scalar computed-mask standalone add.
- Keep mirror fields mirror-only: they may be checked after the provider builds
  the plan, but they must not become route authority in common EmitC/export.

## Out of scope

- No new route family or operation coverage beyond the existing standalone
  reduction family boundary.
- No redo of plain compare-select, elementwise arithmetic, base memory movement,
  scalar broadcast, runtime splat-store, widening conversion, contraction,
  general computed-mask accumulation/MAcc, segment2, or high-level frontend
  work.
- No dtype/LMUL matrix expansion, source-front-door positive path, descriptor
  compute path, dashboards, readiness state machines, or helper-only/report-only
  achievement.
- No legacy RVVI32M1, rvv-i32m1, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-artifact, or exact-intrinsic route authority.

## Acceptance Criteria

- [ ] `verifyRVVSelectedBodyStandaloneReductionRouteFamilyProviderPlans`
      performs deep standalone reduction plan validation comparable in depth to
      adjacent family verifiers.
- [ ] Provider materialization rejects missing standalone plans for all active
      standalone consumers and rejects stale standalone plans on non-consumers.
- [ ] Provider validation rejects mirror mismatches for runtime ABI order,
      target/profile/header/type/intrinsic facts, accumulator/result/seed fields,
      inactive-lane/mask contracts, result name, and RouteOperandBindingPlan
      closure.
- [ ] Plain standalone add/min/max remain coherent and positive.
- [ ] Computed-mask/runtime-scalar standalone reductions are included only where
      current planning already couples them to this family boundary.
- [ ] Focused C++ and/or lit/FileCheck coverage proves deep validation,
      missing-plan/stale-plan failures, mirror mismatch failures, ABI/binding
      closure, accumulator/result/seed contracts, and adjacent computed-mask
      isolation.
- [ ] Generated-bundle dry-runs cover representative explicit and pre-realized
      standalone add/min/max with counts 7, 16, and 23.
- [ ] Real `ssh rvv` evidence covers representative explicit and pre-realized
      standalone add/min/max horizontal values, min/max edge cases, seed use,
      tail preservation, and runtime n variation when runtime correctness is
      claimed.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      shows no new legacy RVVI32M1, rvv-i32m1, finite `tcrv_rvv.i32_*`,
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
- Current code around
  `RVVSelectedBodyStandaloneReductionRouteFamilyPlan` and
  `verifyRVVSelectedBodyStandaloneReductionRouteFamilyProviderPlans`
- Existing standalone reduction lit fixtures under `test/Target/RVV/`
- Existing RVV C++ plugin tests in `test/Plugin/RVVExtensionPluginTest.cpp`

## Completion report requirements

The final report must state task id/title, current phase, module behavior
completed, changed files, stale-brief repair status, provider validation fields
now checked, active standalone routes covered, adjacent reduction boundaries
included/excluded with reason, selected-body realization evidence, binding
closure evidence, generated-bundle and `ssh rvv` commands/results, checks,
active-authority scan, self-repair, finish/archive status, commit hash, and the
exact continuation point if unfinished.
