# Stage2 RVV computed-masked macc-add artifact ABI boundary

## Goal

Make one bounded Stage 2 RVV compiler path fail-closed and executable:

```text
selected tcrv.exec RVV variant
  -> typed_computed_mask_macc_pre_realized_body
  -> RVV plugin-local realization into setvl / with_vl / cmp_lhs load /
     cmp_rhs load / compare-produced mask / lhs load / rhs load / acc load /
     masked_macc add / false-lane accumulator passthrough / result store
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

`Stage2 RVV computed-masked macc-add artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no entries after re-running through
  `rtk bash -lc`.
* Initial `git log --oneline -8` started at
  `b7f0910e rvv: validate runtime scalar masked macc artifact facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` require selected typed body
  -> RVV realization/provider facts -> common EmitC -> target artifact -> real
  `ssh rvv` evidence for runtime/correctness claims.
* The immediately archived runtime-scalar cmp masked macc-add task provides the
  closest finish pattern: provider-owned facts, target validator stale-mirror
  rejection, generated-bundle dry-run checks, real `ssh rvv` correctness
  evidence, archive, and commit.
* The current computed-masked macc-add path already has a typed pre-realized
  ODS op, selected-body realization fixture, MAcc owner/provider facts, target
  validator hooks, generated-bundle dry-run tests, a direct pre-realized
  fail-closed script, and harness support.
* The bounded production gap is to tighten the vector/vector computed-mask
  MAcc ABI and fact surface so the provider and target validator consume
  explicit `cmp_lhs`, `cmp_rhs`, `lhs`, `rhs`, `acc`, `out`, and `n` facts,
  rather than accepting stale runtime-scalar, standalone-reduction,
  descriptor/direct-C/source-front-door, or route-id/intrinsic residue.
* This task is m1-centered. Existing m2 checks may remain passive/negative only
  if they are already present and do not define this route.

## Requirements

* Keep support rooted in the selected typed/pre-realized `tcrv_rvv` body, RVV
  plugin-local realization, computed-mask MAcc route-family plan, provider-built
  route facts, and target validator consumption.
* Preserve and validate runtime ABI order
  `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`.
* Validate vector compare operand roles `cmp_lhs` and `cmp_rhs`, compare
  predicate `slt`, compare-produced mask role/source/form, and same-VL mask
  producer scope.
* Validate unit-stride payload roles for `lhs` and `rhs`, accumulator input
  role/layout, typed compute op `tcrv_rvv.masked_macc`, MAcc kind `add`,
  inactive-lane contract `masked-macc-false-lanes-preserve-accumulator`,
  passthrough layout `accumulator-vector-preserves-inactive-lanes`, result
  layout `store-multiply-accumulate-result-to-output-buffer`, SEW/LMUL/policy,
  and runtime AVL/VL.
* Provider route description must carry route operand binding plan/summary,
  target capability/profile facts, required headers, C type mapping, MAcc
  intrinsic/header/type facts as mirrors only, and an explicit
  `provider_supported_mirror`.
* Target artifact validation must consume provider facts and reject stale or
  missing candidate/provider mirrors before bundle acceptance.
* Fail closed for stale runtime-scalar MAcc facts, stale standalone-reduction
  facts, wrong ABI order, wrong compare predicate, wrong compare/mask source,
  wrong accumulator/result layout, missing passthrough contract, stale binding
  summary, stale descriptor/direct-C/source-front-door residue, stale
  arithmetic/intrinsic residue, or missing provider-supported mirror.
* Generated-bundle dry-run evidence must record provider-derived
  computed-masked macc-add facts, route operand binding, target leaf/profile,
  headers/types, `provider_supported_mirror`, vector compare operand roles,
  compare predicate, mask facts, accumulator passthrough, result layout,
  statement operand order, and runtime AVL/VL facts.
* Runtime correctness must use real `ssh rvv` generated-bundle evidence over
  counts `0,1,16,17,257` and at least two source/mask patterns proving active
  lanes compute `acc + lhs * rhs`, inactive lanes preserve accumulator in the
  stored result, source/acc/tail sentinels are preserved, and runtime `n`/AVL
  is honored.

## Acceptance Criteria

* [x] Focused production diff strengthens computed-masked macc-add
      provider/target validation; no metadata-only closeout.
* [x] Selected-body realization/FileCheck proves pre-realized body consumption
      into setvl/with_vl/load/load/load/load/load/compare/masked_macc/store.
* [x] Target artifact tests fail closed for stale or missing provider mirror,
      binding plan/order/summary, vector compare roles, compare predicate,
      mask role/source/form, lhs/rhs/acc memory roles, MAcc kind add,
      accumulator passthrough, result layout, header/type facts,
      route-family facts, descriptor/direct-C/source-front-door residue, stale
      runtime-scalar MAcc facts, stale standalone-reduction facts, and
      arithmetic/intrinsic residue facts.
* [x] Existing or tightened REALIZED/PLAN/HEADER checks for the pre-realized
      computed-masked macc-add fixture pass.
* [x] Generated-bundle dry-run records provider-derived computed-masked
      macc-add facts, route operand binding, target leaf/profile,
      headers/types, `provider_supported_mirror`, vector compare roles,
      compare predicate, mask facts, accumulator passthrough, result layout,
      statement operand order, and runtime AVL/VL facts.
* [x] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,16,17,257` and at least two source/mask patterns.
* [x] Smallest relevant build/test/script commands, direct FileCheck
      equivalents if lit is unavailable, `git diff --check`, and a bounded
      old-authority scan over touched files pass.
* [x] Trellis task is finished/archived and one coherent commit is created if
      the task completes.

## Out Of Scope

* Runtime-scalar macc redo, broad macc/reduction matrices, reduce add/min/max
  redo, dtype/LMUL clone batches, m2 promotion beyond existing passive
  coverage, contraction expansion, segment/indexed-memory redo,
  frontend/source-front-door positive route, high-level Linalg lowering,
  matmul kernel frontend, global tuning databases, dashboards, descriptor
  routes, direct-C/source exporters, or common EmitC RVV semantic inference.
* Treating exact intrinsic spellings, route ids, artifact names, metadata
  mirrors, generated C strings, scripts, runtime count values, test names, or
  harness constants as the source of MAcc semantics, mask behavior,
  accumulator passthrough, dtype/config, runtime ABI, policy, or route support.

## Technical Notes

Specs and context read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-runtime-scalar-cmp-masked-macc-add-artifact-abi-boundary/prd.md`

Repository files inspected while deriving scope:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-macc-add.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-macc-add-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-macc-add-fail-closed.test`

## Completion Notes

* Added `RVVComputedMaskMAccRouteFacts` and
  `getRVVComputedMaskMAccRouteFacts(...)` so vector/vector computed-mask
  MAcc-add has the same provider-owned canonical fact surface as the
  runtime-scalar MAcc path.
* Rewired computed-mask accumulation provider plan validation, derivation, and
  route operand binding plan selection to consume the canonical vector MAcc
  facts and fail closed if they are missing.
* Rewired target artifact MAcc validation so vector computed-mask MAcc ABI
  order, route binding summary, typed compute op, predicate, target leaf,
  provider mirror, header/type facts, mask source/form, accumulator
  passthrough, and result layout are consumed from the provider fact surface
  instead of duplicated target-side constants.
* Tightened the pre-realized computed-masked macc-add fixture with stale
  runtime-scalar binding, predicate, mask source, source memory, passthrough,
  and result-layout fail-closed mutations.
* Tightened generated-bundle dry-run checks for vector compare roles,
  source/destination memory facts, MAcc kind, and statement operand order.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the RVV
  route-family provider fact surface contract because this task added a
  cross-layer provider/target API contract.
* Verified real `ssh rvv` generated-bundle execution for counts
  `0,1,16,17,257` and patterns `0,1`; active lanes computed
  `acc + lhs * rhs`, inactive lanes preserved accumulator, and source/tail
  sentinels were preserved.
