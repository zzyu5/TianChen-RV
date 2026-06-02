# Stage2 RVV runtime-scalar standalone reduction route-fact canonicalization

## Goal

Canonicalize the RVV plugin-owned route fact boundary for the existing
runtime-scalar computed-mask standalone reduction family:
`runtime_scalar_cmp_masked_standalone_reduce_add`,
`runtime_scalar_cmp_masked_standalone_reduce_min`, and the already-supported
`runtime_scalar_cmp_masked_standalone_reduce_max` fixture path. Provider
planning, `TCRVEmitCLowerableRoute` diagnostics/mirrors, target artifact
validation, and generated-bundle evidence must consume one typed
body/config/runtime-derived fact surface rather than rebuilding literal route
summaries in multiple places.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV runtime-scalar standalone reduction route-fact canonicalization`

## What I Already Know

* The repository started clean from commit
  `825c5f9d rvv: prove runtime scalar masked standalone reduce min abi`.
* There was no active `.trellis/.current-task`, so this task was created from
  the Hermes direction brief before source edits.
* The immediately previous add and min tasks proved runnable generated bundles
  and real `ssh rvv` correctness for add/min, but they mostly strengthened
  fixtures, scripts, and Trellis metadata because production support already
  existed.
* Live code already has `TypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOp`,
  `tcrv_rvv.masked_standalone_reduce`, selected-body realization, route-family
  planning, provider fact verification, target artifact validation, and
  generated-bundle dry-run coverage for add/min/max variants.
* The fragile surface is duplicated route-fact text: runtime ABI order,
  route operand binding plan/summary, `rhs_scalar` ABI+splat ownership,
  compare-produced mask role/source/form, op-kind, inactive-lane
  zero-vs-neutral policy, scalar seed/result layout, SEW/LMUL/policy,
  C type/header facts, and `provider_supported_mirror`.
* Specs require these facts to be derived by RVV plugin owners from typed
  `tcrv_rvv` body/config/runtime facts. Common EmitC/export may carry mirrors
  only after provider route construction.

## Requirements

* Introduce or expose a bounded canonical RVV plugin route-fact surface for
  runtime-scalar computed-mask standalone reductions.
* The canonical surface must cover add, min, and max operation-specific facts:
  route operation kind, runtime ABI order, route operand binding plan ID,
  route operand binding summary, inactive-lane token/contract, target leaf
  profile, provider-supported mirror, header declarations, C type mapping,
  runtime scalar RHS splat participation, compare-mask source/role/form,
  scalar accumulator/result layout, runtime VL carry, SEW/LMUL/policy, and
  source/scalar-result channel facts.
* Provider planning and provider fact verification must use the canonical
  surface before creating `TCRVEmitCLowerableRoute`.
* Target artifact validation must consume the same canonical surface, or fail
  closed when candidate/provider facts are missing, stale, cross-op, or
  inconsistent.
* Existing add/min generated-bundle evidence must keep passing. Max may be
  touched only as an existing supported instance affected by shared
  canonicalization, not as a new evidence milestone.

## Acceptance Criteria

* [x] Runtime-scalar standalone reduction route operand binding plan ID and
      operand summary are generated from one production C++ canonical fact
      helper for add/min/max, including `zero-inactive` for add and
      `neutral-inactive` for min/max.
* [x] Provider route construction verifies canonical facts against the selected
      typed body/config/runtime facts before `TCRVEmitCLowerableRoute`
      creation.
* [x] Target artifact validation calls the same canonical helper and rejects
      stale/missing binding plan, operand summary, op-kind, inactive-lane,
      header/type mapping, runtime ABI order, or provider mirror facts.
* [x] Focused fixtures/scripts continue to show provider-derived facts for add
      and min; max shared fixtures remain consistent if touched.
* [x] Generated-bundle dry-run evidence still records provider-derived route
      facts and does not become route authority.
* [x] If generated artifact behavior changes, rerun at least one existing
      real `ssh rvv` runtime-scalar standalone reduction evidence path. If
      only validation/fact plumbing changes, record why prior runtime evidence
      remains sufficient.
* [x] Bounded old-authority scan over touched files finds no new positive
      reliance on legacy `i32m1`, descriptors, source-front-door routes,
      direct-C/source-export compute, exact-intrinsic authority, or common
      EmitC RVV semantic inference.
* [x] Focused build/test/script checks pass, Trellis task is finished and
      archived, one coherent commit is created, and `git status --short` is
      clean after archive/commit.

## Definition Of Done

* PRD, implement/check context, journal, archive status, and commit are
  truthful.
* Production/default RVV provider and target validation paths changed; this
  task is not closed by helper-only, report-only, or evidence-only changes.
* No new reduction op kinds, dtypes, LMUL clone batches, frontend lowering,
  source-front-door positive routes, broad smoke matrix, common EmitC RVV
  semantic inference, or max-only evidence milestone is introduced.
* Runtime/correctness claims are limited to actual `ssh rvv` evidence already
  present or rerun in this task.

## Technical Approach

1. Add a small C++ canonical route-fact helper in RVV plugin code for
   runtime-scalar computed-mask standalone reduction add/min/max.
2. Rewire provider-side expected binding/fact checks and target artifact
   validation to consume that helper instead of separately reconstructing the
   same strings.
3. Add or strengthen focused target-artifact regressions that mutate stale
   add/min/max binding, inactive-lane, provider mirror, header/type, and
   runtime ABI facts.
4. Update focused MLIR/FileCheck or script expectations only where the
   canonical wording changes.
5. Run focused build/test/script checks, bounded old-authority scan,
   `git diff --check`, Trellis validation, finish/archive, and commit.

## Out Of Scope

* Adding max as a standalone evidence owner.
* Adding new reduction op kinds, new dtypes, new LMUL batches, frontend
  lowering, source-front-door routes, tuning/performance databases, broad
  smoke matrices, report/index work, or harness-only commits.
* Moving RVV semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, scripts, test names, C
  strings, descriptors, exact intrinsic spellings, mirror metadata, or runtime
  harness constants as `rhs_scalar`, mask, reduction, scalar seed/output,
  dtype/config, memory form, policy, or route authority.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`

Prior tasks read:

* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-min-artifact-abi/prd.md`

Likely implementation/test files inspected:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/TargetArtifactExportTest.cpp`
* focused runtime-scalar standalone reduction target fixtures and script tests

## Current Phase

Check / finish.

## Implementation Summary

* Added `RVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts` and
  `getRVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts` as the RVV
  plugin-owned canonical route-fact surface for existing runtime-scalar
  computed-mask standalone reduction add/min/max.
* Rewired provider planning and validation to read runtime ABI order, memory
  form, target leaf profile, provider-supported mirror, required header
  declarations, C type mapping, route operand binding plan/summary, and
  inactive-lane policy from that helper for the runtime-scalar family.
* Rewired target artifact validation to consume the same helper and fail closed
  on stale route binding, provider mirror, header/type mapping, runtime ABI,
  and inactive-lane facts.
* Strengthened `TargetArtifactExportTest` with direct canonical-fact assertions
  for add/min/max; existing stale binding/header/op-kind/inactive-lane/provider
  mirror mutations continue to run through the focused C++ test.

## Evidence

* `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `rtk ./build/bin/tianchenrv-target-artifact-export-test`
* Manual FileCheck equivalent of the three RUN lines for:
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-min.mlir`,
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-max.mlir`.
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body ... --op-kind runtime_scalar_cmp_masked_standalone_reduce_add --op-kind runtime_scalar_cmp_masked_standalone_reduce_min --op-kind runtime_scalar_cmp_masked_standalone_reduce_max ...`
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* `rtk git diff --check`
* Bounded changed-line scan for legacy `i32m1`, descriptor, source-front-door,
  direct-C/source-export, and exact-intrinsic authority markers returned no
  matches.

No `ssh rvv` path was rerun in this round because the generated runtime
artifact behavior did not change: the diff canonicalizes provider/target
route-fact ownership and validation, and the dry-run regenerated the same
add/min/max provider-derived runtime ABI, route operand binding, provider
mirror, header declaration, and inactive-lane evidence. Previous add/min real
`ssh rvv` evidence remains the runtime correctness basis for this fact-plumbing
round.
