# Stage2 RVV runtime-scalar cmp masked standalone reduce-min artifact ABI boundary

## Goal

Make one bounded Stage 2 RVV compiler path real and fail-closed:

```text
selected tcrv.exec RVV variant
  -> typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body
  -> RVV plugin-local realization into setvl / with_vl / source load /
     rhs_scalar load+splat / compare-produced mask / accumulator seed /
     neutral inactive lanes / masked standalone reduce-min / scalar lane0 store
  -> provider-built TCRVEmitCLowerableRoute
  -> target artifact validation
  -> generated bundle/header/object
  -> ssh rvv correctness evidence
```

Route authority must remain the typed/realized `tcrv_rvv` body and RVV
plugin/provider facts. Route ids, artifact names, metadata mirrors, exact
intrinsic spellings, descriptors, C strings, scripts, runtime counts, test
names, or source-front-door markers are mirrors/evidence only.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV runtime-scalar cmp masked standalone reduce-min artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no short-status entries through `rtk`
  filtering.
* Initial `git log --oneline -8` started at
  `7e8fc882 rvv: validate computed masked segment2 update artifact facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* The immediately archived segment2-update task shows the current successful
  pattern: tighten provider/target fail-closed checks, tighten generated-bundle
  dry-run facts, run real `ssh rvv` evidence, finish/archive, then commit.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` agree that the current RVV
  route authority chain is selected typed body -> RVV plugin realization and
  provider facts -> common EmitC -> target artifact -> `ssh rvv` evidence for
  runtime/correctness claims.
* The standalone reduction scalar-channel spec requires a distinct
  source/work channel and scalar accumulator/result channel; inactive-lane
  neutralization for computed-mask standalone reductions belongs to the
  source/work channel.
* The math route operand-binding spec requires math routes, including
  runtime-scalar computed-mask standalone reductions, to consume RVV-owned
  operand-binding facts after route-family provider-plan verification.

## Current Repository Evidence To Inspect

This round must inspect and update only the bounded route family:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` around
  `TypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOp`, `splat`,
  `compare`, `masked_standalone_reduce`, `load`, and `store`.
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`.
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` and adjacent RVV planning /
  statement-plan owners for standalone reduction and computed-mask math.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
* `scripts/rvv_generated_bundle_abi_e2e.py`.
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-min.mlir`.
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-min-dry-run.test`.
* The direct pre-realized runtime-scalar reduce-min fail-closed script/test.

## Requirements

* Keep support rooted in selected typed/pre-realized `tcrv_rvv` body,
  RVV plugin-local realization, standalone-reduction/math route-family plans,
  provider-built route facts, and target validator consumption.
* Preserve and validate runtime ABI order
  `cmp_lhs,rhs_scalar,src,acc,out,n`.
* Validate `rhs_scalar` as runtime ABI input, compare RHS, and scalar splat
  source in the same-VL scope.
* Validate compare predicate `sle`, compare-produced mask role/source/form,
  unit-stride source load, accumulator seed role/layout, reduction kind `min`,
  neutral inactive-lane contract before reduction, scalar lane0 result store,
  SEW/LMUL/policy, runtime AVL/VL, and source/scalar-result channel facts.
* Provider route description must carry route operand binding plan/summary,
  target leaf/profile facts, required headers, C type mapping, reduction
  intrinsic/header/type facts as mirrors only, and an explicit
  `provider_supported_mirror` or equivalent provider-derived mirror.
* Target artifact validation must consume provider facts and reject stale or
  missing candidate/provider mirrors before bundle acceptance.
* Generated-bundle dry-run evidence must record provider-derived reduce-min
  facts, not script/test-name-derived facts.
* Runtime correctness must use real `ssh rvv` generated-bundle evidence over
  counts `0,1,16,23,257`, at least two `rhs_scalar` values such as `-37` and
  `91`, at least two seed values, and at least two source/mask patterns proving
  active lanes contribute to the min, all-inactive lanes preserve the seed,
  source and tail sentinels are preserved, and runtime `n`/AVL is honored.

## Acceptance Criteria

* [x] Focused production diff strengthens runtime-scalar computed-mask
      standalone reduce-min provider/target validation; no metadata-only
      closeout.
* [x] Selected-body realization/FileCheck proves pre-realized body consumption
      into setvl/with_vl/load/splat/compare/masked_standalone_reduce/store.
* [x] Target artifact tests fail closed for stale or missing provider mirror,
      binding plan/order, rhs_scalar splat role, compare predicate/mask
      source/form, source memory form, accumulator seed layout, reduction kind,
      neutral inactive-lane contract, scalar result boundary, header/type facts,
      reduction mirror facts, route-family facts, descriptor/direct-C/
      source-front-door residue, stale unmasked reduction facts, wrong ABI
      order, wrong predicate, wrong neutral contract, wrong seed/result layout,
      and arithmetic/intrinsic residue facts.
* [x] Existing or tightened REALIZED/PLAN/HEADER checks for the pre-realized
      fixture pass.
* [x] Generated-bundle dry-run records provider-derived reduce-min facts,
      route operand binding, target leaf/profile, headers/types,
      `provider_supported_mirror`, runtime-scalar splat/compare facts,
      source/scalar-result channel facts, neutral inactive-lane facts, and
      scalar lane0 result facts.
* [x] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,16,23,257`, at least two `rhs_scalar` values, at least two seeds,
      and at least two source/mask patterns.
* [x] Smallest relevant build/test/script commands, direct FileCheck
      equivalents if lit is unavailable, `git diff --check`, and a bounded
      old-authority scan over touched files pass.
* [x] Trellis task is finished/archived and one coherent commit is created if
      the task completes.

## Completion Notes

Implemented on 2026-06-02.

* Production owner changes:
  * Extended
    `RVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts` with canonical
    runtime-scalar compare predicate, mask role/source/form, computed-mask
    accumulation plan/suffix/contracts, neutral result layout/store VL, and
    scalar-result runtime boundary facts.
  * Tightened
    `validateRVVRuntimeScalarComputedMaskStandaloneReductionRoutePayloadFacts`
    so target artifact export requires the provider description to match those
    canonical facts instead of accepting non-empty stale mirrors.
* Regression coverage:
  * Added reduce-min provider fail-closed cases for stale scalar-result
    boundary, scalar seed/result layout, compare predicate, mask source/role/
    form, accumulation plan/suffix/accumulator contract/scalar-carry contract.
  * Added reduce-min candidate mirror fail-closed cases for compare predicate,
    required headers, C type mapping, and reduction intrinsic mirrors.
  * Tightened generated-bundle dry-run checks for target leaf/profile,
    headers/types, provider-supported mirror, compare/mask facts, source and
    scalar-result channel types, intrinsic mirrors, accumulation contracts, and
    neutral inactive-lane/scalar lane0 boundary facts.
* Evidence:
  * `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
  * `rtk ./build/bin/tianchenrv-target-artifact-export-test`
  * `rtk cmake --build build --target tcrv-opt tcrv-translate -j2`
  * `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-runtime-scalar-cmp-masked-standalone-reduce-min`
    from `build/test` passed 2 tests.
  * Direct dry-run generated
    `artifacts/tmp/stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-min/dry-run`
    with `dry_run_success`.
  * Real `ssh rvv` generated
    `artifacts/tmp/stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-min/ssh-rvv`
    and passed counts `0,1,16,23,257`, rhs scalars `-37,91`, seeds
    `-11,17`, patterns `0,1`, source preservation, and tail preservation.
  * `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  * `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
  * `rtk git diff --check`
  * Bounded scan over touched files found only pre-existing negative legacy
    tests or dry-run `implicit-check-not` guardrails for old authority markers;
    the changed diff adds no positive legacy i32m1/descriptor/source-front-door
    route authority.

## Out Of Scope

* Broad reduction matrix, reduce-max/add clone batches, dtype/LMUL/i64 clone
  batches, MAcc/contraction expansion, segment memory redo, indexed-memory
  redo, frontend/source-front-door positive route, high-level Linalg lowering,
  global tuning databases, dashboards, descriptor routes, direct-C/source
  exporters, or common EmitC RVV semantic inference.
* Treating exact intrinsic spellings, route ids, artifact names, metadata
  mirrors, generated C strings, scripts, runtime count values, test names, or
  harness constants as the source of reduction kind, mask behavior, neutral
  inactive lanes, accumulator/result layout, dtype/config, runtime ABI, policy,
  or route support.

## Technical Notes

Specs and context read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/index.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/guides/index.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-segment2-update-artifact-abi-boundary/prd.md`
