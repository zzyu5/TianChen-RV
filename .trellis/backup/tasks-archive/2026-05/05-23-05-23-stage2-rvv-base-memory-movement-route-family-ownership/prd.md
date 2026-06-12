# Stage2 RVV base memory movement route-family ownership

## Goal

Own the active non-computed-mask, non-segment2 RVV base memory movement route
families through RVV plugin-local planning and provider validation. The
production provider must materialize `strided_load_unit_store`,
`unit_load_strided_store`, `indexed_gather_unit_store`,
`indexed_scatter_unit_load`, `masked_unit_load_store`, and
`masked_unit_store` only from validated base memory movement family plans, and
stale or missing plans must fail closed before `TCRVEmitCLowerableRoute`
construction.

## Direction Source

Hermes direction brief supplied on 2026-05-23:

- module owner: RVV plugin-local memory movement family boundary for active
  non-computed-mask, non-segment2 memory routes
- previous baseline: `9fdee42d rvv: own widening conversion route family`
- required follow-up: make base memory movement carry typed `tcrv_rvv`
  body/config/runtime facts, memory form, source/destination roles,
  stride/index/mask roles, runtime AVL/VL, target leaf/header facts,
  load/store/index intrinsic leaves, and `RouteOperandBindingPlan` closure
  through RVV-owned planning/provider validation

## Current Facts Verified

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Worktree started clean at `9fdee42d`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
- The previous archived widening conversion task added a public family-plan
  pattern: consumer predicate, optional analysis plan, provider verifier,
  description mirror population, focused tests, generated-bundle dry-runs,
  `ssh rvv` evidence, and `check-tianchenrv`.
- Current planning/provider code already has explicit family-plan boundaries
  for computed-mask memory and plain segment2 memory consumers.
- Active base memory operations exist in production route enums and provider
  code: `StridedLoadUnitStore`, `UnitLoadStridedStore`,
  `IndexedGatherUnitStore`, `IndexedScatterUnitLoad`,
  `MaskedUnitLoadStore`, and `MaskedUnitStore`.
- Focused explicit and pre-realized fixtures exist under `test/Target/RVV`,
  and generated-bundle dry-run tests exist under `test/Scripts`, for strided,
  indexed, and static masked unit-load/store routes.
- Base memory route description fields already carry adjacent facts such as
  strided/indexed/masked memory layout mirrors, stride/index sources, source
  and destination memory forms, mask role/source/form, inactive-lane contract,
  intrinsic leaves, and operand binding summaries; this task must move route
  authority for these consumers behind an explicit base memory movement
  planning/provider boundary.

## Requirements

- Inventory only active non-computed-mask, non-segment2 memory movement
  production routes plus directly adjacent provider predicates:
  `StridedLoadUnitStore`, `UnitLoadStridedStore`, `IndexedGatherUnitStore`,
  `IndexedScatterUnitLoad`, `MaskedUnitLoadStore`, and `MaskedUnitStore`.
- Introduce a base memory movement route-family plan for those consumers,
  separate from computed-mask memory and plain segment2 memory plans.
- Add a planning-owned base memory movement consumer predicate covering exactly
  the six active base memory movement consumers listed above.
- Populate the plan from typed body/config/runtime analysis facts, not from
  route ids, artifact names, helper strings, or common EmitC/export code.
- The plan must preserve operation kind, memory form, runtime ABI order,
  runtime AVL/VL control plan, target leaf/profile mirrors,
  provider-supported mirror, required headers, C type mapping, VL type,
  vector/index/mask type facts where applicable, setvl/load/store/index
  intrinsic leaves, source/destination memory forms, stride source,
  index source/EEW/offset/uniqueness where applicable, mask role/source/form
  and inactive-lane/passthrough contracts where applicable, and result names.
- Populate route description mirror fields from the validated base memory
  movement family plan during analysis.
- Make provider materialization for base memory movement consumers depend on a
  base memory movement provider verifier before `TCRVEmitCLowerableRoute`
  construction.
- Reject missing base memory movement family plans for base memory consumers.
- Reject stale base memory movement family plans on non-consumers.
- Keep computed-mask memory and plain segment2 verifier boundaries isolated.
- Preserve existing route ids, ABI order, memory semantics, runtime `n`/AVL
  behavior, dispatch/fallback behavior, target artifact contracts, and common
  EmitC/export neutrality.
- If the full six-route set proves too large for one coherent round, complete
  strided plus indexed base memory movement first and leave static masked-unit
  memory as the exact continuation point.

## Non-goals

- No computed-mask memory or segment2 ownership rewrite beyond keeping their
  verifier boundaries isolated.
- No arithmetic, broadcast, select, conversion, reduction, accumulation,
  contraction, frontend/Linalg, source-front-door, dtype/LMUL clone, dashboard,
  report-only, or script-only work.
- No route id, ABI order, memory semantics, runtime `n`/AVL behavior,
  dispatch/fallback behavior, target artifact contract, or common EmitC/export
  neutrality changes.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, descriptor
  residue, helper string, artifact name, or mirror-only acceptance authority.

## Acceptance Criteria

- [x] Active base memory movement routes and directly adjacent provider
      predicates are inventoried from current source and tests.
- [x] Provider materialization for base memory consumers requires a validated
      base memory movement family plan.
- [x] Missing base memory movement family plans on base memory consumers fail
      closed with targeted diagnostics.
- [x] Stale base memory movement family plans on non-consumers fail closed with
      targeted diagnostics.
- [x] Computed-mask memory and segment2 memory family plans remain isolated and
      are not accidentally required for base memory routes.
- [x] Focused C++ or lit/FileCheck coverage proves consumer classification,
      family plan id, memory form, strided/indexed/masked roles, runtime ABI
      order, intrinsic leaf/type/header mirrors, binding closure, and
      missing/stale-plan rejection.
- [x] Generated-bundle dry-runs cover counts `7`, `16`, and `23` for
      representative explicit and pre-realized active base memory routes.
- [x] Real `ssh rvv` evidence covers representative strided and indexed routes,
      and static masked-unit routes where executable, including addressing,
      mask behavior, tail preservation, and runtime `n` variation.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      shows no legacy route authority regression.
- [x] `check-tianchenrv`, `git diff --check`, and clean git status pass.

## Completion Evidence

- Added `rvv-base-memory-movement-route-family-plan.v1` as the explicit
  family-plan boundary for active `strided_load_unit_store`,
  `unit_load_strided_store`, `indexed_gather_unit_store`,
  `indexed_scatter_unit_load`, `masked_unit_load_store`, and
  `masked_unit_store`.
- Added planning-owned base-memory consumer classification and provider-plan
  verification. `StridedLoadUnitStore`, `UnitLoadStridedStore`,
  `IndexedGatherUnitStore`, `IndexedScatterUnitLoad`, `MaskedUnitLoadStore`,
  and `MaskedUnitStore` now require a validated base memory movement plan
  before provider materialization; stale plans on non-consumers fail closed.
- Kept computed-mask memory and plain segment2 ownership isolated: their
  consumer predicates and verifier boundaries remain separate from base memory
  movement consumers.
- The validated plan carries runtime AVL/VL control, runtime ABI order, target
  leaf profile, provider-supported mirror, required headers, C type mapping,
  vector/index/mask types, intrinsic leaves, source/destination memory forms,
  strided/indexed/masked role facts, inactive-lane contracts, and
  `RouteOperandBindingPlan` closure into the route description and target
  artifact mirrors.
- Focused C++ coverage in `tianchenrv-rvv-extension-plugin-test` proves the
  six-route consumer set, aggregate memory-family inclusion, computed-mask and
  segment2 isolation, missing-plan rejection for strided/indexed/masked
  consumers, and stale-plan rejection on `add`.
- Focused lit/FileCheck coverage passed for explicit
  `strided_load_unit_store`, `unit_load_strided_store`,
  `indexed_gather_unit_store`, `indexed_scatter_unit_load`,
  `masked_unit_load_store`, and pre-realized `masked_unit_store`, proving
  family plan id, runtime control plan, route operand binding, memory roles,
  target leaf profile, provider mirror, required headers, and C type mapping.
- Generated-bundle dry-runs passed for counts `7`, `16`, and `23`:
  explicit strided/indexed/masked-load routes and pre-realized
  strided/indexed/static masked routes.
- Real `ssh rvv` evidence passed for counts `7`, `16`, and `23`:
  explicit and pre-realized strided load/store, indexed gather/scatter,
  `masked_unit_load_store`, and pre-realized `masked_unit_store`. The harnesses
  reported stride `4,8,12`, non-monotonic indexed gather, unique non-monotonic
  indexed scatter, active/inactive mask lane checks, inactive-lane preservation,
  and tail preservation.
- Self-repair performed during continuation: corrected indexed gather/scatter
  leaf expectations to match ordered indexed RVV intrinsics and corrected the
  stale source-memory-form mirror for indexed gather before rerunning focused
  lit.
- Added-line active-authority scan found no new `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor, source-front-door,
  source-artifact, source-export, or direct-C authority. Added exact
  `__riscv_*_i32m1` strings are provider-owned intrinsic leaf mirrors derived
  from typed e32m1 body/config facts, not route ids or legacy route authority.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`,
  focused build, focused C++/lit/script checks, `git diff --check`, and
  `cmake --build build --target check-tianchenrv -j2` passed; full
  `check-tianchenrv` passed `361/361`.

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
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-widening-conversion-route-family-ownership/`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- focused strided/indexed/masked unit-load-store fixtures under
  `test/Target/RVV` and `test/Scripts`
