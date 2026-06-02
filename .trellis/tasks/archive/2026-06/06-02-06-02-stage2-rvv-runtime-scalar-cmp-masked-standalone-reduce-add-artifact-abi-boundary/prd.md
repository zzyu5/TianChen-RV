# Stage2 RVV runtime-scalar cmp masked standalone reduce-add artifact ABI boundary

## Goal

Make one bounded Stage 2 RVV compiler path real and fail-closed:

```text
selected tcrv.exec RVV variant
  -> typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body
  -> RVV plugin-local realization into setvl / with_vl / cmp-lhs load /
     runtime rhs_scalar splat / compare-produced mask / source load /
     masked standalone reduce-add / scalar lane0 result store
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

`Stage2 RVV runtime-scalar cmp masked standalone reduce-add artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no short-status entries through `rtk`.
* Initial `git log --oneline -8` started at
  `e443a8eb rvv: validate runtime scalar reduce min artifact facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* The immediately archived reduce-min task converged the same runtime-scalar
  computed-mask standalone reduction class for `min`, including provider fact
  expansion, target artifact fail-closed checks, generated-bundle dry-run, real
  `ssh rvv` correctness, archive, and commit.
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
* `tcrv_rvv.masked_standalone_reduce` explicitly distinguishes inactive
  neutral values by reduction kind: add uses `0`, min uses `INT32_MAX`, and max
  uses `INT32_MIN` before reducing active source lanes with the scalar
  accumulator seed.
* Existing reduce-add fixture/script coverage is present, including an m1
  pre-realized runtime-scalar computed-mask standalone reduce-add path and an
  m2 sibling. This task is m1-centered; m2 may remain passive/negative unless
  an existing check requires it.

## Current Repository Evidence To Inspect

This round must inspect and update only the bounded route family:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` around
  `TypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOp`, `splat`,
  `compare`, `masked_standalone_reduce`, `load`, and `store`.
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`.
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
* `scripts/rvv_generated_bundle_abi_e2e.py`.
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add.mlir`.
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-dry-run.test`.
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-fail-closed.test`.
* The archived reduce-min task under
  `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-min-artifact-abi-boundary/`.

## Requirements

* Keep support rooted in selected typed/pre-realized `tcrv_rvv` body,
  RVV plugin-local realization, standalone-reduction/math route-family plans,
  provider-built route facts, and target validator consumption.
* Preserve and validate runtime ABI order
  `cmp_lhs,rhs_scalar,src,acc,out,n`.
* Validate `rhs_scalar` as runtime ABI input, compare RHS, and scalar splat
  source in the same-VL scope.
* Validate compare predicate `sle`, compare-produced mask role/source/form,
  unit-stride source input, accumulator seed role/layout, reduction kind `add`,
  zero inactive-lane handling before reduction, scalar carry across runtime VL
  chunks, scalar lane0 result store, SEW/LMUL/policy, runtime AVL/VL, and
  source/scalar-result channel facts.
* Provider route description must carry route operand binding plan/summary,
  target leaf/profile facts, required headers, C type mapping, reduction
  intrinsic/header/type facts as mirrors only, and an explicit
  `provider_supported_mirror` or equivalent provider-derived mirror.
* Target artifact validation must consume provider facts and reject stale or
  missing candidate/provider mirrors before bundle acceptance.
* Fail closed for stale min/max facts, wrong inactive neutral contract, wrong
  seed/result layout, wrong runtime ABI order, stale descriptor/direct-C/
  source-front-door residue, stale arithmetic/intrinsic residue, missing
  provider-supported mirror, and candidate artifact mirrors that disagree with
  provider facts.
* Generated-bundle dry-run evidence must record provider-derived reduce-add
  facts, not script/test-name-derived facts.
* Runtime correctness must use real `ssh rvv` generated-bundle evidence over
  counts `0,1,16,23,257`, at least two `rhs_scalar` values such as `-37` and
  `91`, at least two seed values, and at least two source/mask patterns proving
  active lanes add into the seed, all-inactive lanes preserve the seed through
  zero inactive lanes, source and tail sentinels are preserved, and runtime
  `n`/AVL is honored.

## Acceptance Criteria

* [x] Focused production diff strengthens runtime-scalar computed-mask
      standalone reduce-add provider/target validation; no metadata-only
      closeout.
* [x] Selected-body realization/FileCheck proves pre-realized body consumption
      into setvl/with_vl/load/splat/compare/load/
      masked_standalone_reduce/store.
* [x] Target artifact tests fail closed for stale or missing provider mirror,
      binding plan/order, rhs_scalar splat role, compare predicate/mask
      source/form, source memory form, accumulator seed layout, reduction kind
      `add`, zero inactive-lane requirement, scalar carry/result boundary,
      header/type facts, reduction mirror facts, route-family facts,
      descriptor/direct-C/source-front-door residue, stale min/max facts, wrong
      neutral contract, wrong seed/result layout, and arithmetic/intrinsic
      residue facts.
* [x] Existing or tightened REALIZED/PLAN/HEADER checks for the pre-realized
      reduce-add fixture pass.
* [x] Generated-bundle dry-run records provider-derived reduce-add facts,
      route operand binding, target leaf/profile, headers/types,
      `provider_supported_mirror`, runtime-scalar splat/compare facts,
      source and scalar-result channel facts, zero inactive-lane facts, scalar
      carry facts, and scalar lane0 result boundary facts.
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

Production owner changes:

* Added
  `getRVVSelectedBodyStandaloneReductionInactiveNeutralLiteral(...)` to the RVV
  provider API so add/min/max inactive neutral literals are provider-owned
  facts.
* Updated target artifact validation to consume the RVV provider helper when
  checking computed-mask standalone reduction inactive neutral splats, removing
  the target-local neutral literal table.
* Recorded the new provider API contract in
  `.trellis/spec/extension-plugins/rvv-plugin.md`.

Regression coverage:

* Extended target artifact C++ tests so runtime-scalar computed-mask
  standalone reduce-add rejects stale scalar-result runtime boundary, seed and
  result layouts, compare predicate, mask source/role/form, accumulation
  plan/suffix/contracts, wrong inactive neutral literal, stale required header
  mirror, stale C type mapping mirror, and stale reduction intrinsic mirror.
* Tightened reduce-add generated-bundle dry-run FileCheck for target leaf,
  headers/types, provider-supported mirror, compare/mask facts,
  source/scalar-result channel facts, intrinsics, accumulation contracts, zero
  inactive-lane facts, and scalar lane0 boundary facts.

Evidence:

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* `rtk ./build/bin/tianchenrv-target-artifact-export-test`
* `rtk cmake --build build --target tcrv-opt tcrv-translate -j2`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add`
  from `build/test` passed 2 tests after correcting FileCheck wording to match
  the generated evidence.
* `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* Dry-run generated
  `artifacts/tmp/stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add/dry-run`
  with `dry_run_success`.
* Real `ssh rvv` generated
  `artifacts/tmp/stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add/ssh-rvv`
  and passed counts `0,1,16,23,257`, rhs scalars `-37,91`, seeds `-11,17`,
  patterns `0,1`, source preservation, and tail preservation.
* `rtk git diff --check`
* Bounded old-authority scan over touched files found only pre-existing
  negative legacy fixtures or dry-run `implicit-check-not` guardrails; the
  changed diff adds no positive legacy i32m1/descriptor/source-front-door route
  authority.

## Out Of Scope

* Broad reduction matrix, reduce-min/max redo, dtype/LMUL/i64 clone batches,
  MAcc/contraction expansion, segment memory redo, indexed-memory redo,
  frontend/source-front-door positive route, high-level Linalg lowering,
  global tuning databases, dashboards, descriptor routes, direct-C/source
  exporters, or common EmitC RVV semantic inference.
* Treating exact intrinsic spellings, route ids, artifact names, metadata
  mirrors, generated C strings, scripts, runtime count values, test names, or
  harness constants as the source of reduction kind, mask behavior, zero
  inactive lanes, accumulator/result layout, dtype/config, runtime ABI, policy,
  or route support.

## Technical Notes

Specs and context read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-min-artifact-abi-boundary/prd.md`
