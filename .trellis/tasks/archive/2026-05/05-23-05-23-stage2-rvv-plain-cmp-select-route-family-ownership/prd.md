# Stage2 RVV plain compare-select route-family ownership

## Goal

Own the active RVV plain `CmpSelect` compare/select route family through an
RVV plugin-local planning and provider-validation boundary. The production
provider must materialize active plain compare-select routes only from typed
`tcrv_rvv` compare/select body facts captured in a validated family plan, while
keeping this path distinct from computed-mask select and runtime-scalar select
consumers.

## Direction Source

Hermes direction brief supplied on 2026-05-23:

- module owner: RVV plugin-local plain compare-select family boundary for
  active `CmpSelect` routes, aligned with but not merged into the existing
  computed-mask/runtime-scalar select family
- previous baseline: `7fcdbb6d rvv: own elementwise arithmetic route family`
- required follow-up: carry typed `tcrv_rvv` compare/select body facts through
  RVV-owned planning/provider validation into route materialization, generated
  C artifacts, and `ssh rvv` correctness evidence

## Current Facts Verified

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Worktree started clean at `7fcdbb6d`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.
- The previous archived elementwise arithmetic task added the current
  family-plan pattern: planning consumer predicates, plan storage, route
  description mirrors, provider-side verification, focused C++/lit coverage,
  generated-bundle dry-runs, real `ssh rvv` evidence, `check-tianchenrv`, task
  archive, and one coherent commit.
- Specs require typed `tcrv_rvv` body/config/runtime facts to be route
  authority, with common EmitC/export remaining a neutral materializer of
  provider-built `TCRVEmitCLowerableRoute` objects.

## Requirements

- Inventory only active plain `CmpSelect` production routes and directly
  adjacent computed-mask/runtime-scalar select boundaries.
- Introduce or complete a validated plain compare-select family plan, or an
  equivalent typed select-family branch that keeps plain `CmpSelect` distinct
  from computed-mask select consumers.
- Make provider materialization for `CmpSelect` depend on the validated plain
  compare-select family boundary before `TCRVEmitCLowerableRoute`
  construction.
- Fail closed when an active `CmpSelect` consumer is missing the plain
  compare-select family plan.
- Fail closed when a stale plain compare-select family plan appears on
  non-consumers.
- Preserve operation kind, memory form, compare predicate/profile, compare
  mask type/value, true/false vector operand roles, selected result role,
  runtime ABI order, AVL/VL control, target leaf/header mirrors, compare/select
  intrinsic leaves, `RouteOperandBindingPlan` closure, and mirror-only metadata
  semantics.
- Keep computed-mask select and runtime-scalar select isolation truthful. Do
  not rewrite those families except for adjacent classification/diagnostic
  checks needed to prevent plan mixing.
- Preserve existing route ids, runtime `n` behavior, ABI order,
  dispatch/fallback behavior, generated artifact contracts, and common
  EmitC/export neutrality.

## Non-goals

- No broadcast, conversion, reduction, accumulation, contraction, memory
  movement, computed-mask memory, segment2, scalar frontend/Linalg, new
  dtype/LMUL clone batches, or new compare predicate expansion beyond the
  active plain `CmpSelect`/SLE-style route fixtures needed for this owner.
- No computed-mask select or runtime-scalar select rewrite beyond isolation
  checks.
- No route authority from common EmitC/export, artifact names, route ids,
  helper strings, mirror fields, descriptor residue, source-front-door paths,
  or exact intrinsic spelling.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, or
  `!tcrv_rvv.i32m*` compatibility route preservation or introduction.

## Acceptance Criteria

- [x] Active plain `CmpSelect` routes and directly adjacent computed-mask /
      runtime-scalar select boundaries are inventoried from current source and
      tests.
- [x] Provider materialization for active plain `CmpSelect` requires a
      validated plain compare-select family plan.
- [x] Missing plain compare-select family plans on `CmpSelect` consumers fail
      closed with targeted diagnostics.
- [x] Stale plain compare-select family plans on non-consumers fail closed with
      targeted diagnostics.
- [x] Computed-mask select and runtime-scalar select boundaries remain isolated
      from the plain `CmpSelect` family plan.
- [x] Focused C++ or lit/FileCheck coverage proves consumer classification,
      family plan id, operation kind, memory form, compare predicate/profile,
      mask role/source, true/false operand binding, runtime ABI order,
      intrinsic/type/header mirrors, route operand binding closure, and
      missing/stale-plan rejection.
- [x] Generated-bundle dry-runs cover representative explicit and pre-realized
      `CmpSelect` artifacts at runtime counts `7`, `16`, and `23`.
- [x] Real `ssh rvv` evidence covers representative explicit and pre-realized
      `CmpSelect`/SLE artifacts, including compare truth table, selected lanes,
      tail preservation, and runtime `n` variation.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      shows no legacy route authority regression.
- [x] `check-tianchenrv`, `git diff --check`, and clean git status pass.

## Completion Evidence

- Implemented a plugin-local plain compare-select family plan in RVV route
  planning, with provider validation before `TCRVEmitCLowerableRoute`
  materialization.
- Active plain consumers covered: `CmpSelect` explicit and pre-realized
  fixtures, including `eq` and `sle` compare predicates.
- Adjacent non-consumers checked: `ComputedMaskSelect`,
  `RuntimeScalarCompareSelect`, `RuntimeScalarDualCompareMaskAndSelect`, and
  `Add` do not accept stale plain compare-select family plans.
- Focused checks passed:
  - `./build/bin/tianchenrv-rvv-extension-plugin-test`
  - `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter=cmp-select .`
    from `build/test`, 10/10 selected tests passed
  - generated-bundle dry-runs for explicit and pre-realized
    `cmp_select`/`cmp_select_sle`, runtime counts `7,16,23`
  - real `ssh rvv` generated-bundle runs for explicit and pre-realized
    `cmp_select`/`cmp_select_sle`, runtime counts `7,16,23`
  - added-line active-authority scan over touched RVV/plugin/export/script/test
    paths found no legacy authority matches
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2`, 361/361 passed
- Self-repair performed: after focused lit exposed a missing inactive-lane
  contract mirror for plain `CmpSelect`, the family plan was extended to carry
  inactive-lane and passthrough layout facts into the route description instead
  of relaxing provider validation.

## Definition Of Done

- Implementation is bounded to this module behavior.
- Focused checks and runtime evidence are recorded in task notes or journal.
- Trellis task status is truthful and archived when complete.
- One coherent git commit records the completed task if the task is complete.

## Technical Approach

Follow the existing RVV route-family ownership pattern from base memory
movement and elementwise arithmetic: add a plugin-local family plan for the
plain `CmpSelect` consumer, derive it during route planning from typed body
facts, mirror the validated plan into route descriptions/artifacts, and require
provider-side validation before route construction. Adjacent computed-mask and
runtime-scalar select plans remain separate consumers with stale-plan checks.

## Technical Starting Points

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-elementwise-arithmetic-route-family-ownership/`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `test/Target/RVV/*cmp-select*.mlir`
- focused generated-bundle script tests for selected-body artifacts
