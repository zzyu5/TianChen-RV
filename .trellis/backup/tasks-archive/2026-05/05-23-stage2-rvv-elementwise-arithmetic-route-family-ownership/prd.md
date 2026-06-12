# Stage2 RVV elementwise arithmetic route-family ownership

## Goal

Own the active RVV elementwise arithmetic route family through RVV
plugin-local planning and provider validation. The production provider must
materialize active plain vector `add` / `sub` / `mul`, static masked
`masked_add` / `masked_sub` / `masked_mul`, and `strided_add` only from a
validated elementwise-arithmetic family plan, and stale or missing plans must
fail closed before `TCRVEmitCLowerableRoute` construction.

## Direction Source

Hermes direction brief supplied on 2026-05-23:

- module owner: RVV plugin-local elementwise arithmetic family boundary for
  active vector `Add` / `Sub` / `Mul`, `MaskedAdd` / `MaskedSub` /
  `MaskedMul`, and `StridedAdd` routes
- previous baseline: `2ab8f6c4 rvv: own base memory movement route family`
- required follow-up: carry typed `tcrv_rvv` body/config/runtime facts,
  operation kind, memory form, vector operand roles, mask role for masked
  arithmetic, strided input role for `StridedAdd`, runtime AVL/VL, target
  leaf/header facts, arithmetic intrinsic leaves, and
  `RouteOperandBindingPlan` closure through RVV-owned planning/provider
  validation into routes, generated artifacts, and `ssh rvv` evidence

## Current Facts Verified

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Worktree started clean at `2ab8f6c4`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
- The previous archived base memory movement task added the current public
  family-plan pattern: consumer predicate, analysis plan, provider verifier,
  description mirror population, focused C++/lit coverage, generated-bundle
  dry-runs, `ssh rvv` evidence, `check-tianchenrv`, finish/archive, and one
  coherent commit.
- Current planning/provider code already has explicit family-plan boundaries
  for scalar broadcast elementwise, runtime scalar splat-store, widening
  conversion, base memory movement, computed-mask select, computed-mask
  memory, segment2 memory, standalone reduction, accumulation, and
  contraction.
- Current active plain vector elementwise arithmetic operations exist as
  `Add`, `Sub`, and `Mul`.
- Current active static masked elementwise arithmetic operations exist as
  `MaskedAdd`, `MaskedSub`, and `MaskedMul`.
- Current active strided elementwise arithmetic exists as `StridedAdd`.
- These routes already carry route operand binding plan ids and typed route
  description mirrors, but they do not yet have a dedicated
  elementwise-arithmetic family-plan boundary that provider materialization
  must validate.
- Focused explicit and pre-realized fixtures exist under `test/Target/RVV`,
  and generated-bundle dry-run tests exist under `test/Scripts`, for plain,
  masked, and strided add; sub/mul route fixtures also exist for explicit and
  pre-realized target artifact coverage.

## Requirements

- Inventory only active plain vector elementwise arithmetic, static masked
  elementwise arithmetic, and strided-add production routes plus directly
  adjacent provider predicates:
  `Add`, `Sub`, `Mul`, `MaskedAdd`, `MaskedSub`, `MaskedMul`, and
  `StridedAdd`.
- Introduce an elementwise arithmetic route-family plan for those consumers,
  separate from scalar-broadcast elementwise, computed-mask select,
  accumulation, contraction, and memory movement plans.
- Add a planning-owned elementwise arithmetic consumer predicate covering
  exactly the seven active consumers listed above.
- Populate the plan from typed body/config/runtime analysis facts, not from
  route ids, artifact names, helper strings, or common EmitC/export code.
- The plan must preserve operation kind, memory form, runtime ABI order,
  runtime AVL/VL control plan, target leaf/profile mirrors,
  provider-supported mirror, required headers, C type mapping, VL/vector/mask
  type facts, setvl/load/store/arithmetic intrinsic leaves, result and mask
  names, source/destination memory forms, masked inactive-lane and passthrough
  contracts where applicable, strided input/output stride roles where
  applicable, and runtime ABI parameters.
- Populate route description mirror fields from the validated elementwise
  arithmetic family plan during analysis.
- Make provider materialization for elementwise arithmetic consumers depend on
  an elementwise arithmetic provider verifier before
  `TCRVEmitCLowerableRoute` construction.
- Reject missing elementwise arithmetic family plans for elementwise consumers.
- Reject stale elementwise arithmetic family plans on non-consumers.
- Preserve existing route ids, ABI order, arithmetic semantics, runtime
  `n`/AVL behavior, dispatch/fallback behavior, target artifact contracts, and
  common EmitC/export neutrality.
- Keep scalar broadcast, memory movement, select, conversion, reduction,
  accumulation, and contraction verifier boundaries isolated.

## Non-goals

- No scalar-broadcast, base memory movement, computed-mask memory, segment2
  memory, computed-mask select, widening conversion, reduction, accumulation,
  or contraction ownership rewrite beyond keeping their verifier boundaries
  isolated.
- No frontend/Linalg, source-front-door, new dtype/LMUL clone batches, new
  arithmetic operations, dashboard, report-only, or script-only work.
- No route id, ABI order, arithmetic semantics, runtime `n`/AVL behavior,
  dispatch/fallback behavior, target artifact contract, or common EmitC/export
  neutrality changes.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, descriptor
  residue, helper string, artifact name, or mirror-only acceptance authority.

## Acceptance Criteria

- [x] Active elementwise arithmetic routes and directly adjacent provider
      predicates are inventoried from current source and tests.
- [x] Provider materialization for `Add`, `Sub`, `Mul`, `MaskedAdd`,
      `MaskedSub`, `MaskedMul`, and `StridedAdd` requires a validated
      elementwise arithmetic family plan.
- [x] Missing elementwise arithmetic family plans on elementwise consumers fail
      closed with targeted diagnostics.
- [x] Stale elementwise arithmetic family plans on non-consumers fail closed
      with targeted diagnostics.
- [x] Adjacent family plans remain isolated and are not accidentally required
      for elementwise arithmetic routes or vice versa.
- [x] Focused C++ or lit/FileCheck coverage proves consumer classification,
      family plan id, operation kind, memory form, vector operand roles,
      mask/strided roles, runtime ABI order, intrinsic leaf/type/header
      mirrors, binding closure, and missing/stale-plan rejection.
- [x] Generated-bundle dry-runs cover counts `7`, `16`, and `23` for
      representative explicit and pre-realized active elementwise routes.
- [x] Real `ssh rvv` evidence covers representative plain vector, masked, and
      strided elementwise routes where executable, including arithmetic
      correctness, mask behavior, strided addressing, tail preservation, and
      runtime `n` variation.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      shows no legacy route authority regression.
- [x] `check-tianchenrv`, `git diff --check`, and clean git status pass.

## Completion Evidence

- Implemented `RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan` with
  operation/memory-form classification for active plain `Add` / `Sub` / `Mul`,
  static masked `MaskedAdd` / `MaskedSub` / `MaskedMul`, and `StridedAdd`.
- Provider construction now calls
  `verifyRVVSelectedBodyElementwiseArithmeticRouteFamilyProviderPlans` before
  `TCRVEmitCLowerableRoute` materialization for those consumers.
- Route descriptions and generated artifacts now mirror the validated
  elementwise plan id, target leaf profile, provider-supported mirror,
  required header declarations, C type mapping, runtime control plan, runtime
  ABI order, route operand binding closure, source/destination memory forms,
  mask facts, stride facts, and typed intrinsic/type facts.
- Focused C++ coverage:
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Focused lit coverage:
  `lit --filter='(explicit-selected-body-artifact-(add|masked-add|strided-add)|pre-realized-selected-body-artifact-(add|masked-add|strided-add))'`
  passed 6/6, and
  `lit --filter='rvv-generated-bundle-abi-e2e-(selected-body|pre-realized|masked-add|pre-realized-masked-add|strided-add|pre-realized-strided-add)-dry-run'`
  passed 6/6.
- Generated-bundle dry-runs:
  `scripts/rvv_generated_bundle_abi_e2e.py --dry-run --op-kind add --op-kind masked_add --op-kind strided_add --runtime-count 7 --runtime-count 16 --runtime-count 23`
  passed for explicit and pre-realized selected bodies under
  `artifacts/tmp/stage2_elementwise_arithmetic_route_family/`.
- Real RVV evidence:
  explicit and pre-realized selected bodies both passed on `ssh rvv` for
  `add`, `masked_add`, and `strided_add` at counts `7`, `16`, and `23`;
  masked runs reported true/false mask lanes and passthrough preservation, and
  strided runs checked strided addressing.
- Full gate:
  `cmake --build build --target check-tianchenrv -j2` passed 361/361.
- Hygiene:
  `git diff --check` passed. Active-authority scan over added RVV/plugin/test
  lines found no `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor/source-front-door/source-export/direct-C, or
  exact `__riscv_*_i32m1` authority additions.

## Definition Of Done

- Implementation is bounded to the PRD module behavior.
- Focused tests and runtime evidence are recorded in task notes or journal.
- The Trellis task is finished and archived when complete.
- One coherent git commit records the completed task.

## Technical Starting Points

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-base-memory-movement-route-family-ownership/`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- focused add/sub/mul, masked add/sub/mul, and strided-add fixtures under
  `test/Target/RVV` and `test/Scripts`
