# Stage2 RVV runtime-scalar cmp masked macc-add artifact ABI boundary

## Goal

Make one bounded Stage 2 RVV compiler path fail-closed and executable:

```text
selected tcrv.exec RVV variant
  -> typed_runtime_scalar_computed_mask_macc_pre_realized_body
  -> RVV plugin-local realization into setvl / with_vl / cmp-lhs load /
     runtime rhs_scalar splat / payload lhs-rhs loads / accumulator load /
     compare-produced mask / masked_macc add / inactive-lane accumulator
     passthrough / result store
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

`Stage2 RVV runtime-scalar cmp masked macc-add artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no short-status entries through `rtk`.
* Initial `git log --oneline -8` started at
  `c64f43d1 chore: record journal`, followed by
  `297e075e rvv: validate runtime scalar reduce max artifact facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` agree that current RVV
  runtime/correctness claims require selected typed body -> RVV
  realization/provider facts -> common EmitC -> target artifact -> real
  `ssh rvv` evidence.
* The immediately archived reduce-max task provides the closest finish pattern:
  provider-owned facts, target validator stale-mirror rejection, generated
  bundle dry-run, `ssh rvv` correctness evidence, archive, and commit.
* The current runtime-scalar macc-add path already has a typed pre-realized
  ODS op, selected-body realization fixture, MAcc owner/provider facts, target
  validator hooks, generated-bundle dry-run tests, a direct pre-realized
  fail-closed script, and harness support.
* Repository inspection found a bounded production gap rather than a missing
  route: target/provider validation and fixtures must be tightened so
  runtime-scalar macc-add rejects stale or missing provider facts for compare
  predicate `sle`, rhs scalar splat/compare-RHS binding, mask source/form,
  accumulator passthrough, result layout, header/type facts, route-family
  facts, and candidate mirrors before artifact acceptance.
* This task is m1-centered. Existing m2 dry-run coverage may remain passive or
  negative unless it is already constrained by touched code.

## Requirements

* Keep support rooted in the selected typed/pre-realized `tcrv_rvv` body, RVV
  plugin-local realization, MAcc/computed-mask accumulation route-family plans,
  provider-built route facts, and target validator consumption.
* Preserve and validate runtime ABI order
  `cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n`.
* Validate `rhs_scalar` as runtime ABI input, scalar splat source, and compare
  RHS in the same-VL mask producer path.
* Validate compare predicate `sle`, compare-produced mask role/source/form,
  unit-stride `lhs`/`rhs` payload loads, accumulator input role/layout, typed
  compute op `tcrv_rvv.masked_macc`, macc kind `add`, inactive-lane contract
  `masked-macc-false-lanes-preserve-accumulator`, passthrough layout
  `accumulator-vector-preserves-inactive-lanes`, result layout
  `store-multiply-accumulate-result-to-output-buffer`, SEW/LMUL/policy, and
  runtime AVL/VL.
* Provider route description must carry route operand binding plan/summary,
  target leaf/profile facts, required headers, C type mapping, MAcc
  intrinsic/header/type facts as mirrors only, and an explicit
  `provider_supported_mirror`.
* Target artifact validation must consume provider facts and reject stale or
  missing candidate/provider mirrors before bundle acceptance.
* Fail closed for stale standalone-reduction facts, wrong ABI order, wrong
  compare predicate, wrong compare/mask source, wrong accumulator/result
  layout, missing passthrough contract, stale binding summary, stale
  descriptor/direct-C/source-front-door residue, stale arithmetic/intrinsic
  residue, or missing provider-supported mirror.
* Generated-bundle dry-run evidence must record provider-derived macc-add
  facts, route operand binding, target leaf/profile, headers/types,
  `provider_supported_mirror`, runtime-scalar splat/compare facts, compare
  predicate, mask facts, accumulator passthrough, result layout, statement-plan
  operand order, and runtime AVL/VL facts.
* Runtime correctness must use real `ssh rvv` generated-bundle evidence over
  counts `0,1,16,17,257`, rhs scalars `-37,91`, and at least two source/mask
  patterns proving active lanes compute `acc + lhs * rhs`, inactive lanes
  preserve accumulator in the stored result, source/acc/tail sentinels are
  preserved, and runtime `n`/AVL is honored.

## Acceptance Criteria

* [x] Focused production diff strengthens runtime-scalar computed-mask macc-add
      provider/target validation; no metadata-only closeout.
* [x] Selected-body realization/FileCheck proves pre-realized body consumption
      into setvl/with_vl/load/splat/load/load/load/compare/masked_macc/store.
* [x] Target artifact tests fail closed for stale or missing provider mirror,
      binding plan/order/summary, rhs_scalar splat role, compare predicate,
      mask role/source/form, lhs/rhs/acc memory roles, macc kind add,
      accumulator passthrough, result layout, header/type facts,
      route-family facts, descriptor/direct-C/source-front-door residue, stale
      standalone-reduction facts, and arithmetic/intrinsic residue facts.
* [x] Existing or tightened REALIZED/PLAN/HEADER checks for the pre-realized
      runtime-scalar macc-add fixture pass.
* [x] Generated-bundle dry-run records provider-derived runtime-scalar macc-add
      facts, route operand binding, target leaf/profile, headers/types,
      `provider_supported_mirror`, runtime-scalar splat/compare facts, compare
      predicate, mask facts, accumulator passthrough, result layout, statement
      operand order, and runtime AVL/VL facts.
* [x] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,16,17,257`, rhs scalars `-37,91`, and at least two source/mask
      patterns.
* [x] Smallest relevant build/test/script commands, direct FileCheck
      equivalents if lit is unavailable, `git diff --check`, and a bounded
      old-authority scan over touched files pass.
* [x] Trellis task is finished/archived and one coherent commit is created if
      the task completes.

## Out Of Scope

* Broad macc/reduction matrices, reduce add/min/max redo, dtype/LMUL clone
  batches, m2 promotion beyond existing passive coverage, contraction
  expansion, segment/indexed-memory redo, frontend/source-front-door positive
  route, high-level Linalg lowering, matmul kernel frontend, global tuning
  databases, dashboards, descriptor routes, direct-C/source exporters, or
  common EmitC RVV semantic inference.
* Treating exact intrinsic spellings, route ids, artifact names, metadata
  mirrors, generated C strings, scripts, runtime count values, test names, or
  harness constants as the source of macc semantics, mask behavior,
  accumulator passthrough, dtype/config, runtime ABI, policy, or route
  support.

## Technical Notes

Specs and context read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-max-artifact-abi-boundary/prd.md`

Repository files inspected while deriving scope:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-macc-add-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-macc-add-fail-closed.test`

## Completion Notes

* Added provider-owned runtime-scalar computed-mask MAcc route facts for
  `typedComputeOpName = tcrv_rvv.masked_macc` and
  `comparePredicateKind = sle`.
* Tightened target artifact validation so runtime-scalar computed-mask MAcc
  consumes typed compute op, compare predicate, unit-stride source/destination
  memory form, indexed memory layout, accumulator/result passthrough contracts,
  route-family plan facts, and explicit provider mirror facts.
* Added fail-closed target tests and MLIR fixture mutations for stale
  standalone-reduction residue, scalar-carry residue, stale typed compute op,
  wrong predicate, wrong binding summary/order, wrong mask source/form, wrong
  source memory form, wrong accumulator layout, missing passthrough, and stale
  provider/candidate mirrors.
* Tightened generated-bundle dry-run checks for provider-derived MAcc facts,
  route operand binding, compare predicate, rhs scalar splat/compare role,
  source/destination/indexed memory layout, statement operand order, header/type
  facts, target leaf/profile facts, and `provider_supported_mirror`.
* Verified real `ssh rvv` generated-bundle execution for counts
  `0,1,16,17,257`, rhs scalars `-37,91`, and patterns `0,1`; active lanes
  computed `acc + lhs * rhs`, inactive lanes preserved accumulator, and tails
  were preserved.
* No spec update was needed: existing RVV plugin, EmitC route, and testing specs
  already require provider-owned typed-body facts, Common EmitC neutrality, and
  real RVV evidence for runtime/correctness claims.
