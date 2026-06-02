# Stage2 RVV runtime-scalar cmp masked standalone reduce-max artifact ABI boundary

## Goal

Make one bounded Stage 2 RVV compiler path real and fail-closed:

```text
selected tcrv.exec RVV variant
  -> typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body
  -> RVV plugin-local realization into setvl / with_vl / cmp-lhs load /
     runtime rhs_scalar splat / compare-produced mask / source load /
     masked standalone reduce-max / scalar lane0 result store
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

`Stage2 RVV runtime-scalar cmp masked standalone reduce-max artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no short-status entries through `rtk`.
* Initial `git log --oneline -8` started at
  `47f3af6c rvv: validate runtime scalar reduce add artifact facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* The immediately archived reduce-add task converged the same runtime-scalar
  computed-mask standalone reduction class for `add`, including provider-owned
  inactive neutral literal authority, target validator consumption, generated
  bundle checks, real `ssh rvv` correctness, archive, and commit.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` agree that the current RVV
  route authority chain is selected typed body -> RVV plugin realization and
  provider facts -> common EmitC -> target artifact -> `ssh rvv` evidence for
  runtime/correctness claims.
* The standalone reduction scalar-channel spec requires a distinct
  source/work channel and scalar accumulator/result channel when the route
  needs that split; inactive-lane neutralization for computed-mask standalone
  reductions belongs to the source/work channel.
* `tcrv_rvv.masked_standalone_reduce` distinguishes inactive neutral values by
  reduction kind: add uses `0`, min uses `INT32_MAX`, and max must use
  `INT32_MIN` before reducing active source lanes with the scalar accumulator
  seed.
* Existing reduce-max selected pre-realized fixture coverage is present in the
  target artifact tests. This task is m1-centered; an m2 sibling may remain
  passive/negative unless an existing check already constrains it.

## Current Repository Evidence To Inspect

This round must inspect and update only the bounded route family:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` around
  `TypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOp`, `splat`,
  `compare`, `masked_standalone_reduce`, `load`, and `store`.
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`.
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
* `scripts/rvv_generated_bundle_abi_e2e.py`.
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-max.mlir`.
* The archived reduce-add task under
  `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-artifact-abi-boundary/`.

## Requirements

* Keep support rooted in selected typed/pre-realized `tcrv_rvv` body,
  RVV plugin-local realization, standalone-reduction/math route-family plans,
  provider-built route facts, and target validator consumption.
* Preserve and validate runtime ABI order
  `cmp_lhs,rhs_scalar,src,acc,out,n`.
* Validate `rhs_scalar` as runtime ABI input, compare RHS, and scalar splat
  source in the same-VL scope.
* Validate compare predicate `sle`, compare-produced mask role/source/form,
  unit-stride source input, accumulator seed role/layout, reduction kind `max`,
  `INT32_MIN` inactive-lane handling before reduction, scalar carry across
  runtime VL chunks, scalar lane0 result store, SEW/LMUL/policy, runtime
  AVL/VL, and source/scalar-result channel facts.
* Provider route description must carry route operand binding plan/summary,
  target leaf/profile facts, required headers, C type mapping, reduction
  intrinsic/header/type facts as mirrors only, and an explicit
  `provider_supported_mirror` or equivalent provider-derived mirror.
* Target artifact validation must consume provider facts and reject stale or
  missing candidate/provider mirrors before bundle acceptance.
* Fail closed for stale add/min facts, wrong inactive neutral contract, wrong
  seed/result layout, wrong runtime ABI order, stale descriptor/direct-C/
  source-front-door residue, stale arithmetic/intrinsic residue, missing
  provider-supported mirror, and candidate artifact mirrors that disagree with
  provider facts.
* Generated-bundle dry-run evidence must record provider-derived reduce-max
  facts, not script/test-name-derived facts.
* Runtime correctness must use real `ssh rvv` generated-bundle evidence over
  counts `0,1,16,23,257`, at least two `rhs_scalar` values such as `-37` and
  `91`, at least two seed values, and at least two source/mask patterns proving
  active lanes max into the seed, all-inactive lanes preserve the seed through
  `INT32_MIN` inactive lanes, source and tail sentinels are preserved, and
  runtime `n`/AVL is honored.

## Acceptance Criteria

* [x] Focused production diff strengthens runtime-scalar computed-mask
      standalone reduce-max provider/target validation; no metadata-only
      closeout.
* [x] Selected-body realization/FileCheck proves pre-realized body consumption
      into setvl/with_vl/load/splat/compare/load/
      masked_standalone_reduce/store.
* [x] Target artifact tests fail closed for stale or missing provider mirror,
      binding plan/order, rhs_scalar splat role, compare predicate/mask
      source/form, source memory form, accumulator seed layout, reduction kind
      `max`, `INT32_MIN` inactive-lane requirement, scalar carry/result
      boundary, header/type facts, reduction mirror facts, route-family facts,
      descriptor/direct-C/source-front-door residue, stale add/min facts, wrong
      neutral contract, wrong seed/result layout, and arithmetic/intrinsic
      residue facts.
* [x] Existing or tightened REALIZED/PLAN/HEADER checks for the pre-realized
      reduce-max fixture pass.
* [x] Generated-bundle dry-run records provider-derived reduce-max facts,
      route operand binding, target leaf/profile, headers/types,
      `provider_supported_mirror`, runtime-scalar splat/compare facts,
      source and scalar-result channel facts, `INT32_MIN` inactive-lane facts,
      scalar carry facts, and scalar lane0 result boundary facts.
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

* Added `inactiveNeutralLiteralSEW32` and `inactiveNeutralLiteralSEW64` to
  `RVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts`, deriving those
  values from the RVV provider-owned inactive neutral helper.
* Updated runtime-scalar computed-mask standalone reduction target statement
  validation to consume those canonical route facts for inactive neutral
  splats, so reduce-max rejects stale add zero or reduce-min `INT32_MAX`
  literals.
* Extended generated-bundle evidence to record the concrete inactive neutral
  literal, the source/work vector-channel splat, and neutralization-vs-zeroing
  facts for runtime-scalar standalone reductions.
* Updated the RVV plugin spec to document the new per-SEW runtime-scalar
  canonical fact fields and target-validation consumption boundary.

Regression coverage:

* Extended `TargetArtifactExportTest.cpp` so runtime-scalar reduce-max m1
  rebuilds provider route validation inputs and rejects stale add binding
  facts, stale min/add inactive neutral literals, and stale min reduction
  mirror metadata.
* Added focused generated-bundle dry-run coverage for m1
  `runtime_scalar_cmp_masked_standalone_reduce_max`.
* Added direct pre-realized route-entry fail-closed coverage for m1 reduce-max.
* Kept the existing min/max aggregate dry-run test passive for m2 and repaired
  its harness assertions to be presence-based rather than order-sensitive.

Evidence:

* `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* `rtk ./build/bin/tianchenrv-target-artifact-export-test`
* `rtk cmake --build build --target tcrv-opt tcrv-translate -j2`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter runtime-scalar-cmp-masked-standalone-reduce-max`
  from `build/test` passed 4 tests.
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter runtime-scalar-cmp-masked-standalone`
  from `build/test` passed 17 tests after repairing the aggregate min/max
  harness FileCheck ordering.
* Final dry-run evidence:
  `artifacts/tmp/stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-max/dry-run/pre-realized-runtime-scalar-cmp-masked-standalone-reduce-max`
  with `dry_run_success`.
* Final `ssh rvv` evidence:
  `artifacts/tmp/stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-max/ssh-rvv/pre-realized-runtime-scalar-cmp-masked-standalone-reduce-max`
  with `PASS op=runtime_scalar_cmp_masked_standalone_reduce_max
  counts=0,1,16,23,257 rhs_scalars=-37,91 seeds=-11,17 patterns=0,1
  runtime_scalar_computed_mask_standalone_reduce source_preserved
  tail_preserved`.
* `rtk git diff --check`
* Bounded old-authority scan over the diff added no positive legacy
  `RVVI32M1` / `rvv-i32m1` / `tcrv_rvv.i32_*` / descriptor / direct-C /
  source-front-door route authority. New occurrences are FileCheck negative
  guardrails only.

## Out Of Scope

* Broad reduction matrix, add/min redo, dtype/LMUL/i64 clone batches,
  MAcc/contraction expansion, segment memory redo, indexed-memory redo,
  frontend/source-front-door positive route, high-level Linalg lowering,
  global tuning databases, dashboards, descriptor routes, direct-C/source
  exporters, or common EmitC RVV semantic inference.
* Treating exact intrinsic spellings, route ids, artifact names, metadata
  mirrors, generated C strings, scripts, runtime count values, test names, or
  harness constants as the source of reduction kind, mask behavior,
  `INT32_MIN` inactive lanes, accumulator/result layout, dtype/config, runtime
  ABI, policy, or route support.

## Technical Notes

Specs and context read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-artifact-abi-boundary/prd.md`
