# Stage2 RVV computed-mask compare/select route-family ownership

## Goal

Tighten the active Stage2 RVV compare/select compiler path so
`computed_mask_select`, `runtime_scalar_cmp_select`, and
`runtime_scalar_dual_cmp_mask_and_select` are materialized only through a
validated RVV-owned computed-mask select route-family plan and its
operation-specific `RouteOperandBindingPlan`. The family boundary must carry
typed `tcrv_rvv` body/config/runtime facts into provider materialization,
target/header mirrors, generated bundle artifacts, and focused runtime
evidence without expanding compare/select coverage.

## Direction Source

- Direction title: `Stage2 RVV computed-mask compare/select route-family ownership`.
- Module owner: RVV plugin-local computed-mask compare/select family boundary
  for active select routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `aa165fca rvv: enable explicit strided masked contraction bodies`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes/Codex Direction Brief before source edits.

## What I Already Know

- The current long-term RVV authority chain is:
  `tcrv.exec` envelope, selected RVV variant, typed low-level `tcrv_rvv` body,
  RVV plugin-owned legality/selected-body realization/provider,
  provider-built `TCRVEmitCLowerableRoute`, common EmitC materialization,
  target artifact, and real `ssh rvv` evidence for runtime/correctness claims.
- The previous archived contraction task
  `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-explicit-strided-masked-selected-body-realization/`
  finished the remaining explicit contraction variants and left no contraction
  continuation point.
- Current code already has active compare/select routes and fixtures for:
  `cmp_select`, `computed_mask_select`, `computed_mask_select_sle`,
  `runtime_scalar_cmp_select`, and
  `runtime_scalar_dual_cmp_mask_and_select`.
- This task's positive family boundary is bounded to
  `computed_mask_select`, `runtime_scalar_cmp_select`, and
  `runtime_scalar_dual_cmp_mask_and_select`. Plain `cmp_select` is adjacent
  only for ownership separation and must not be forced into the computed-mask
  select family plan.
- Existing source surfaces show `RVVSelectedBodyComputedMaskSelectRouteFamilyPlan`
  and plan derivation/validation are present, but the provider path does not yet
  expose a public `isRVVSelectedBodyComputedMaskSelectRouteFamilyConsumer`
  predicate or a shared provider-plan verifier analogous to memory,
  contraction, standalone-reduction, and accumulation family owners.

## Scope

1. Inventory only active computed-mask and runtime-scalar compare/select
   production routes plus directly adjacent provider predicates.
2. Add a planning-owned consumer predicate for the computed-mask select family:
   - consumer routes:
     `ComputedMaskSelect`, `RuntimeScalarCompareSelect`,
     `RuntimeScalarDualCompareMaskAndSelect`;
   - non-consumer adjacent route: `CmpSelect`.
3. Add a shared provider-plan verifier that:
   - requires the computed-mask select family plan for consumers before
     provider materialization;
   - rejects a computed-mask select family plan attached to non-consumers;
   - revalidates the plan;
   - checks plan operation, family-plan mirror, mask-producer mirror,
     runtime/control/ABI, target/header/C-type mirrors, mask/select layout,
     true/false/source/destination facts, and runtime ABI parameter closure.
4. Rewire provider materialization to use that verifier instead of local
   provider-only boolean authority.
5. Preserve existing explicit and pre-realized selected-body compare/select
   fixtures, generated bundle expectations, runtime ABI order, route ids,
   target/header metadata, and common EmitC/export neutrality.

## Requirements

1. Provider materialization for `computed_mask_select`,
   `runtime_scalar_cmp_select`, and
   `runtime_scalar_dual_cmp_mask_and_select` must depend on a validated
   `rvv-computed-mask-select-route-family-plan.v1`.
2. The provider must fail closed if any non-consumer route, including plain
   `cmp_select`, carries a computed-mask select family plan.
3. The computed-mask select provider boundary must preserve:
   - operation kind;
   - mask producer source;
   - vector-compare vs runtime-scalar vs dual-mask-and distinctions;
   - true/false operand roles;
   - select/store layout;
   - runtime ABI order;
   - AVL/VL control;
   - dtype/config;
   - target leaf/header mirrors;
   - `RouteOperandBindingPlan` closure.
4. `cmp_select` must remain a plain compare/select route with its own operand
   binding plan, and must not acquire computed-mask select family-plan mirrors.
5. Common EmitC/export must continue to consume provider-built routes and must
   not choose compare/select semantics, dtype/config, mask policy, intrinsic
   spelling, or support state.
6. No new contraction, memory, reduction, MAcc, dtype/LMUL,
   frontend/Linalg/source-front-door coverage is allowed.
7. No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, descriptor,
   artifact-name, helper-string, or mirror-only acceptance authority may be
   introduced.

## Acceptance Criteria

- [ ] The task context references relevant RVV plugin, EmitC route, testing
      specs, and the previous contraction task.
- [ ] Active route inventory is bounded to `computed_mask_select`,
      `runtime_scalar_cmp_select`, and
      `runtime_scalar_dual_cmp_mask_and_select`; plain `cmp_select` is
      documented as adjacent/out-of-family.
- [ ] A public planning-owned computed-mask select family consumer predicate is
      available to provider code.
- [ ] A shared provider-plan verifier requires a family plan for consumers and
      rejects stale family plans for non-consumers.
- [ ] Provider materialization calls that verifier before building the
      `TCRVEmitCLowerableRoute`.
- [ ] Focused C++/lit/FileCheck coverage proves consumer predicate behavior,
      stale/non-consumer rejection, family plan id, mask producer source,
      runtime-scalar and dual-mask distinctions, true/false operand binding,
      select layout, target/header mirror stability, and binding closure.
- [ ] Generated-bundle dry-runs pass across counts `7,16,23` for representative
      active select routes:
      `computed_mask_select`, `runtime_scalar_cmp_select`, and
      `runtime_scalar_dual_cmp_mask_and_select`.
- [ ] Real `ssh rvv` evidence passes for vector computed-mask select and at
      least one runtime-scalar select route; dual-mask-and is also run if the
      current generated bundle path is executable on `ssh rvv`.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive legacy or mirror-only route authority.
- [ ] Focused checks, `git diff --check`, `check-tianchenrv`, task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No new compare/select operation variants beyond active production routes.
- No new contraction, memory, reduction, MAcc, dtype/LMUL, high-level frontend,
  Linalg, source-front-door, dashboard, report-only, helper-only, or broad smoke
  matrix work.
- No route id, ABI order, computation semantics, mask semantics, true/false
  value semantics, runtime n/AVL behavior, dispatch/fallback behavior, or
  target artifact contract changes except to preserve them through the family
  boundary.
- No shift of compare/select semantics into common EmitC/export or target
  artifact metadata.

## Validation Plan

1. Start the Trellis task after this PRD and context repair.
2. Add/verify focused code in:
   - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
   - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
   - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
   - `test/Plugin/RVVExtensionPluginTest.cpp`
   - focused compare/select lit/script fixtures only as needed.
3. Build or run focused test target:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run focused lit/FileCheck tests for explicit and pre-realized select routes
   and runtime-scalar negative coverage.
5. Run generated-bundle dry-runs for active representative select routes with
   counts `7,16,23`.
6. Run focused real `ssh rvv` generated-bundle evidence for active executable
   select routes.
7. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant prior task read:

- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-explicit-strided-masked-selected-body-realization/task.json`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-explicit-strided-masked-selected-body-realization/prd.md`

Initial source surfaces inspected:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- focused compare/select fixtures under `test/Target/RVV`, `test/Scripts`,
  and `test/Plugin/RVVExtensionPluginTest.cpp`.

## Definition Of Done

The active computed-mask select family has the same kind of planning/provider
ownership boundary as the recently tightened RVV family owners: consumers are
declared by planning-owned predicates, provider materialization requires and
revalidates the family plan, stale plans are rejected for non-consumers, focused
tests and generated-bundle evidence cover the active routes, real `ssh rvv`
evidence backs runtime/correctness claims, the task is finished/archived
truthfully, and one coherent commit records the change.

## Completion Evidence

Module behavior completed:

- Added public planning-owned computed-mask select route-family consumer
  predicate for `computed_mask_select`, `runtime_scalar_cmp_select`, and
  `runtime_scalar_dual_cmp_mask_and_select`.
- Added shared provider-plan verifier requiring
  `rvv-computed-mask-select-route-family-plan.v1` for family consumers and
  rejecting stale family plans on non-consumers such as plain `cmp_select`.
- Rewired RVV provider materialization to call the shared computed-mask select
  family verifier before building the `TCRVEmitCLowerableRoute`.
- Revalidated family-plan mirrors for operation, mask producer source,
  runtime/control/ABI, target/header/C-type facts, true/false select layout,
  source/destination forms, runtime ABI parameters, and
  `RouteOperandBindingPlan` ID before provider materialization.
- Carried `selectLayout` from the validated family plan into route description
  metadata instead of emitting the select-layout mirror as a hardcoded metadata
  constant.
- Preserved plain `cmp_select` as adjacent/out-of-family; focused check confirms
  it emits no computed-mask select family metadata.

Active routes covered:

- `computed_mask_select_sle` explicit selected-body generated bundle.
- `computed_mask_select` pre-realized selected-body generated bundle.
- `runtime_scalar_cmp_select`.
- `runtime_scalar_dual_cmp_mask_and_select`.

Variants intentionally out of scope:

- Plain `cmp_select`: adjacent route only; preserved as a separate plain
  compare/select route with no computed-mask select family plan.
- Bare explicit `computed_mask_select`: the current generated-bundle script does
  not support this op kind in explicit-selected-body mode; explicit vector
  computed-mask select evidence is represented by the active
  `computed_mask_select_sle` fixture, while bare `computed_mask_select` remains
  covered by the pre-realized fixture.

Files changed:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- This Trellis task record.

Family plan/provider boundary evidence:

- `build/bin/tianchenrv-rvv-extension-plugin-test` passed, including focused
  consumer-predicate checks and missing/stale plan rejection checks.
- `tcrv-opt` checks over explicit selected-body fixtures proved family plan id,
  mask producer source, runtime ABI order, route operand binding plan,
  provider-supported mirror, target leaf profile, and `select_layout` for:
  `computed_mask_select_sle`, `runtime_scalar_cmp_select`, and
  `runtime_scalar_dual_cmp_mask_and_select`.
- `tcrv-opt` checks over pre-realized fixtures proved selected-body realization
  still reaches family-plan metadata for `computed_mask_select` and
  `runtime_scalar_dual_cmp_mask_and_select`.
- Plain `cmp_select` metadata scan reported:
  `plain cmp_select has no computed-mask select family metadata`.

Generated-bundle evidence:

- Explicit generated-bundle dry-run passed for `computed_mask_select_sle` with
  runtime counts `7,16,23`.
- Pre-realized generated-bundle dry-run passed for `computed_mask_select` with
  runtime counts `7,16,23`.
- Explicit generated-bundle dry-run passed for `runtime_scalar_cmp_select` with
  runtime counts `7,16,23`.
- Explicit generated-bundle dry-run passed for
  `runtime_scalar_dual_cmp_mask_and_select` with runtime counts `7,16,23`.

Real RVV evidence:

- Real `ssh rvv` explicit generated-bundle run passed for
  `computed_mask_select_sle` with counts `7,16,23`, checking true/false selected
  lanes and tail preservation.
- Real `ssh rvv` explicit generated-bundle run passed for
  `runtime_scalar_cmp_select` with counts `7,16,23` and RHS scalars `5,11`,
  checking runtime scalar threshold behavior, true/false selected lanes, and
  tail preservation.
- Real `ssh rvv` explicit generated-bundle run passed for
  `runtime_scalar_dual_cmp_mask_and_select` with counts `7,16,23`,
  `rhs_scalar_a` values `5,11`, and `rhs_scalar_b` values `-37,91`, checking
  both masks, composed-mask lanes, single-mask-only lanes, and tail
  preservation.

Checks:

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- Focused `tcrv-opt`/`tcrv-translate` metadata checks passed for explicit and
  pre-realized select fixtures. `llvm-lit` and `FileCheck` were not available
  in this shell, so these were run directly with built local tools and `rg`.
- `git diff --check` passed.
- Added-line active-authority scan over touched RVV/plugin/test paths found no
  new `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  descriptor, direct-C, source-front-door, artifact-name, or mirror-only
  authority.
- `cmake --build build --target check-tianchenrv -j2` passed 361/361.

Self-repair:

- Retried generated-bundle dry-runs with `/usr/lib/llvm-20/bin/llvm-readobj`
  after the default `llvm-readobj` was absent from `PATH`.
- Switched explicit vector computed-mask generated-bundle evidence from
  unsupported explicit `computed_mask_select` to active explicit
  `computed_mask_select_sle`, while keeping bare `computed_mask_select` covered
  through pre-realized mode.
- Attempted `clang-format`, but the binary was not present in this shell; code
  formatting was checked manually and by compilation.

Spec update judgment:

- No `.trellis/spec/**` update was needed. This round implements existing RVV
  plugin ownership, provider-plan, common EmitC neutrality, and testing
  contracts without adding a new long-term rule.

Continuation point:

- None. The bounded computed-mask compare/select ownership boundary is complete
  for the active routes named in this task.
