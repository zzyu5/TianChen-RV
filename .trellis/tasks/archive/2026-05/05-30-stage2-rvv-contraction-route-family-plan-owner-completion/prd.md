# Stage2 RVV contraction route-family plan owner completion

## Goal

Complete the Stage2 RVV widening-contraction route-family plan ownership split
for the existing selected-body contraction family. Move contraction-specific
route-family plan construction and validation authority behind
`RVVEmitCContractionRouteFamilyPlanOwners`, leaving
`RVVEmitCRoutePlanning.cpp` responsible for shared route analysis, typed/config
fact collection, neutral owner dispatch, route description mirroring, and
generic closure checks.

## Direction Source

- Direction title: `Expand: Stage2 RVV contraction route-family plan construction owner completion`.
- Module owner: `RVVEmitCContractionRouteFamilyPlanOwners` as the single owner
  for widening-contraction route-family plan construction and verification.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `4a4722a0 rvv: repair dirty segment2 owner state`.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  Direction Brief before source edits.
- Serial worker constraint: one Codex worker; no subagents, spawned agents,
  parallel agents, or multi-agent workflows.

## What I Already Know

- The prior widening-contraction operand-binding task moved direct contraction
  operand-binding plan IDs, logical operand role lookup, and binding-plan
  derivation into `RVVEmitCContractionRouteFamilyPlanOwners`.
- The repair task committed the contraction owner files as active production
  state because CMake, route planning, and C++ tests already consume them.
- `.trellis/spec/index.md` requires the RVV authority chain to run through a
  selected typed `tcrv_rvv` body, RVV plugin-owned legality/realization/route
  provider, provider-built `TCRVEmitCLowerableRoute`, and neutral common EmitC.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires direct-provider
  contraction facts for `widening_macc_add`,
  `widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add` to be owner-local
  before provider route construction.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC to
  remain neutral. It must not choose RVV intrinsics, infer dtype/SEW/LMUL,
  synthesize ABI roles, or use route ids, status, manifests, descriptors, or
  metadata as route authority.
- Current code inspection found `RVVEmitCContractionRouteFamilyPlanOwners`
  owning contraction provider-plan verification and operand-binding plan
  derivation, while `RVVEmitCRoutePlanning.cpp` still directly owns
  contraction route-family classification helpers, runtime ABI order,
  family plan ID/profile/header/type mapping constants, intrinsic leaves,
  mask/stride facts, dtype/source/result relation fields, and the
  `deriveRVVSelectedBodyContractionRouteFamilyPlan(...)` /
  `validateRVVSelectedBodyContractionRouteFamilyPlan(...)` /
  `applyRVVSelectedBodyContractionRouteFamilyPlan(...)` implementation.

## Requirements

1. Extend `RVVEmitCContractionRouteFamilyPlanOwners` so it builds or derives the
   active contraction route-family plan facts it verifies.
2. Cover all current direct-provider contraction operations:
   `WideningMAccAdd`, `WideningDotReduceAdd`,
   `StridedInputWideningDotReduceAdd`,
   `ComputedMaskWideningDotReduceAdd`, and
   `ComputedMaskStridedInputWideningDotReduceAdd`.
3. Move contraction-specific plan constants and derivation choices out of
   `RVVEmitCRoutePlanning.cpp`, including operation family identity, runtime
   ABI order, target leaf/profile facts, source/result dtype relation,
   widening MAcc vs widening dot relation, mask facts, stride facts, required
   headers, and derived intrinsic mirrors.
4. Keep central route planning as shared route analysis, typed/config/
   capability collection, neutral owner dispatch, description mirror
   propagation, and generic verification only.
5. Provider validation must fail closed for missing or stale contraction family
   plan facts, wrong operation classification, wrong runtime ABI order, stale
   provider-plan mirrors, wrong source/result dtype relation, wrong widening
   relation, wrong mask facts, wrong stride facts, wrong target leaf/profile,
   wrong required headers, wrong operand binding, or stale route/name/metadata
   authority.
6. Preserve the existing selected-body contraction route behavior and existing
   provider-built route path. This task does not add new operation coverage or
   claim new runtime/correctness/performance behavior.
7. Preserve MAcc, segment2, computed-mask accumulation, memory, compare/select,
   reduction, conversion, and residual owner behavior except where shared
   declarations need to stay compiling.
8. Add focused C++ coverage proving owner-owned contraction plan construction,
   verification, and fail-closed behavior across the five active contraction
   subfamilies.

## Acceptance Criteria

- [ ] `RVVEmitCContractionRouteFamilyPlanOwners` exposes the owner API needed to
      derive, validate, apply, and verify contraction route-family plans for
      the five existing direct-provider contraction routes.
- [ ] `RVVEmitCRoutePlanning.cpp` no longer owns contraction-specific plan IDs,
      runtime ABI orders, target leaf/profile/header/type constants,
      mask/stride/widening relation choices, source/result dtype relation
      choices, or exact intrinsic leaf authority for this family except via
      neutral owner calls or shared containers.
- [ ] Central route analysis still collects typed body/config/capability facts,
      dispatches to the contraction owner, propagates owner-produced mirrors
      into `RVVSelectedBodyEmitCRouteDescription`, and runs generic route
      description verification.
- [ ] Positive C++ tests prove owner plan construction and verification for
      `widening_macc_add`, `widening_dot_reduce_add`,
      `strided_input_widening_dot_reduce_add`,
      `computed_masked_widening_dot_reduce_add`, and
      `computed_masked_strided_input_widening_dot_reduce_add`.
- [ ] Fail-closed C++ tests cover stale mirror, wrong operand binding, wrong
      runtime ABI order, wrong dtype/source/result relation, wrong mask fact,
      wrong stride fact, wrong target leaf/profile, missing provider plan, and
      missing materialized leaf cases within the contraction boundary.
- [ ] Non-regression coverage for MAcc and segment2 owner behavior still passes.
- [ ] Bounded scan shows central `RVVEmitCRoutePlanning.cpp` no longer acts as
      authority for contraction plan IDs, ABI orders, mask/stride/widening
      relation choices, or exact contraction intrinsic leaves.
- [ ] Authority scan over touched production/test files finds no new central
      ad hoc, name-derived, metadata-derived, descriptor-derived,
      ABI-string-derived, script-derived, artifact-name-derived,
      common-EmitC-derived, source-front-door-derived, route-id-derived,
      exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [ ] `git diff --check` passes.
- [ ] Focused RVV plugin C++ build and binary pass.
- [ ] `check-tianchenrv` passes, or the exact blocker is recorded.
- [ ] Final `git status --short` is clean after finish/archive/commit.

## Technical Approach

Use the owner split already established for MAcc and segment2:

```text
selected typed contraction tcrv_rvv body
  -> central shared route analysis and typed/config/capability facts
  -> contraction owner derives route-family plan facts
  -> central neutral mirror propagation from owner plan
  -> owner verifies same-analysis provider mirrors and operand binding
  -> route materialization facts / route-control provider plan / math binding facts
  -> direct contraction provider plan
  -> TCRVEmitCLowerableRoute provider
  -> neutral common EmitC materialization
```

Implementation should first move the existing contraction plan derivation,
validation, runtime ABI order lookup, constants, and apply-to-description logic
into `RVVEmitCContractionRouteFamilyPlanOwners`. Central route planning should
call the owner API at the same production points where it currently derives and
applies the plan. Tests should be tightened around the owner API rather than
only around final route description fields.

## Out of Scope

- New RVV operation coverage, dtype/LMUL clone batches, Linalg/frontend
  lowering, new artifact generators, new direct pre-realized route-entry
  paths, one-intrinsic wrapper dialects, broad smoke matrices, dashboards, or
  report-only work.
- Reopening completed segment2 owner work except for shared declarations or
  tests needed to keep the current build coherent.
- Moving common EmitC into RVV semantic ownership.
- Runtime, correctness, or performance claims requiring `ssh rvv`; no emitted
  executable semantics are intended to change in this owner-boundary task.

## Validation Plan

1. Validate task context with
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-stage2-rvv-contraction-route-family-plan-owner-completion`.
2. Build focused plugin test:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`.
3. Run focused plugin C++ coverage:
   `./build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build touched tools if affected:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
5. Run bounded owner/authority scans over touched RVV planning/provider/test
   files.
6. Run `git diff --check`.
7. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Definition Of Done

The contraction owner is the production owner of existing widening-contraction
route-family plan construction and verification, central planning retains only
shared route analysis/dispatch/mirror propagation, focused checks pass, the
task is finished and archived truthfully, one coherent commit is created, and
the final worktree is clean.
