# Stage2 RVV runtime-scalar masked MACC accumulation artifact ABI boundary

## Goal

Make `runtime_scalar_cmp_masked_macc_add` a bounded production-owned Stage 2
RVV accumulation route boundary. The selected `tcrv.exec` RVV variant must
realize the typed runtime-scalar computed-mask MAcc pre-realized body into
explicit `tcrv_rvv` structure, and the RVV plugin/provider plus target artifact
validator must consume one provider-owned route-fact surface for the runtime ABI
order, scalar RHS splat/compare source, masked MAcc accumulation semantics,
accumulator passthrough, result layout, headers/types, binding plan, and
provider mirror before generated artifacts are accepted.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV runtime-scalar masked MACC accumulation artifact ABI boundary`

## What I Already Know

* Repository started clean from commit
  `4e2d08b2 rvv: canonicalize runtime scalar reduction route facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes direction brief before source edits.
* The previous archived reduction canonicalization task introduced a provider
  header-visible canonical fact helper consumed by provider planning and target
  validation. This task should use the same production-boundary pattern for the
  neighboring runtime-scalar computed-mask MAcc family.
* Current code already has `TypedRuntimeScalarComputedMaskMaccPreRealizedBodyOp`,
  selected-body realization into `setvl` / `with_vl` / compare-lhs load /
  `tcrv_rvv.splat` / payload loads / accumulator load / compare /
  `masked_macc` / store, and focused pre-realized PLAN/HEADER/fail-closed
  checks.
* Current target artifact validation still locally reconstructs
  runtime-scalar computed-mask MAcc facts such as runtime ABI order, route
  operand binding summary, target leaf profile, provider-supported mirror,
  required headers, C type mapping, mask producer source, accumulator layout,
  inactive-lane contract, and result layout.
* Generated-bundle dry-run already exercises counts `0,1,16,17,257`, scalar RHS
  values `-37,91`, and patterns `0,1`, but evidence JSON should be tightened so
  runtime-scalar computed-mask MAcc exposes the spec-required
  `computed_masked_macc_boundary` facts instead of relying only on generic
  metadata checks.

## Requirements

* Add or expose a bounded RVV plugin-owned canonical route-fact surface for
  `runtime_scalar_cmp_masked_macc_add`.
* The canonical surface must cover:
  runtime ABI order `cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n`;
  `rhs_scalar` as typed scalar ABI, splat source, and compare RHS;
  compare-produced mask role/source/form;
  masked MAcc add operation kind and compute suffix;
  accumulator input/pass-through layout;
  inactive-lane contract
  `masked-macc-false-lanes-preserve-accumulator`;
  result store layout;
  runtime AVL/VL carry;
  SEW/LMUL/policy-derived type and intrinsic leaves;
  required C headers and C type mapping;
  provider route operand binding plan/summary;
  target leaf profile; and
  `provider_supported_mirror`.
* Provider planning and provider fact verification must use that surface before
  `TCRVEmitCLowerableRoute` construction.
* Target artifact route-family validation must consume the same provider-owned
  surface for runtime-scalar computed-mask MAcc and fail closed on stale or
  missing runtime ABI order, route binding plan/summary, scalar RHS splat facts,
  mask producer facts, accumulator layout, inactive-lane passthrough, result
  layout, header/type facts, target leaf profile, or provider mirror.
* Generated-bundle dry-run evidence must record provider-derived
  runtime-scalar computed-mask MAcc facts, including a populated
  `computed_masked_macc_boundary` for the runtime-scalar pre-realized path.
* The generated harness must keep proving active lanes compute
  `acc + lhs * rhs`, inactive lanes preserve `acc`, output/tail sentinels are
  preserved, runtime `n`/AVL is honored, both scalar RHS values are passed
  through the generated ABI, and both data patterns are exercised.

## Acceptance Criteria

* [x] Runtime-scalar computed-mask MAcc route facts are represented by one
      provider-owned C++ helper or equivalent provider fact surface, not by
      duplicated stale local target-validator assembly.
* [x] Provider route construction validates the canonical runtime ABI order,
      route operand binding plan/summary, scalar RHS splat/compare role,
      accumulation/mask contracts, accumulator/result layout, headers/types,
      target leaf profile, and provider-supported mirror.
* [x] Target artifact validation consumes the same fact surface and rejects
      stale/missing provider mirror, binding plan/order, binding summary,
      `rhs_scalar`/splat compare source, accumulator layout, inactive-lane
      passthrough, result layout, header/type facts, and route-family facts.
* [x] Existing REALIZED/PLAN/HEADER checks for the pre-realized macc fixture
      pass or are tightened to check provider-derived facts.
* [x] Generated-bundle dry-run records runtime-scalar computed-mask MAcc
      `computed_masked_macc_boundary` facts, `rhs_scalar` scalar-loop evidence,
      route operand binding `abi|splat|cmp-rhs|hdr`, and harness pattern loops.
* [x] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,16,17,257`, scalar RHS values `-37,91`, and patterns `0,1`, proving
      active-lane MAcc, inactive accumulator passthrough, tail sentinel
      preservation, runtime `n`/AVL, and output layout.
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
  task is not closed by helper-only, script-only, or evidence-only changes.
* Common EmitC/export remains neutral and does not infer RVV MAcc semantics.
* No reduction/max evidence task, broad MAcc matrix, LMUL m2 acceptance owner,
  widening-dot/matmul, new dtype, frontend/source-front-door positive route,
  high-level Linalg lowering, report/dashboard/index work, or harness-only
  commit is introduced.

## Technical Approach

1. Add a small C++ canonical route-fact helper for
   `RuntimeScalarComputedMaskedMAccAdd` in the RVV plugin/provider surface.
2. Rewire provider description verification and target artifact MAcc validation
   to consume that helper for runtime-scalar computed-mask MAcc facts.
3. Strengthen `TargetArtifactExportTest` with direct canonical-fact assertions
   and stale runtime-scalar MAcc mutations.
4. Tighten generated-bundle evidence extraction and dry-run FileCheck for
   runtime-scalar computed-mask MAcc boundary fields.
5. Run focused C++ build/tests, direct MLIR/FileCheck equivalents, dry-run
   script tests, real `ssh rvv` generated-bundle evidence, bounded
   old-authority scan, `git diff --check`, Trellis validation, archive, and
   commit.

## Out Of Scope

* Reduction/max evidence work.
* Broad MAcc or LMUL m2 matrix ownership.
* Widening-dot, matmul, new dtype, frontend/source-front-door positive routes,
  high-level Linalg lowering, dashboards, or report/index work.
* Moving RVV accumulation, mask, ABI, dtype/config, or route authority into
  common EmitC/export, scripts, route ids, artifact names, test names,
  descriptors, exact intrinsic spellings, or mirror metadata.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`

Prior task read:

* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-runtime-scalar-standalone-reduction-route-fact-canonicalization/prd.md`

Likely implementation/test files inspected:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-macc-add-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-macc-add-fail-closed.test`
* `test/Target/TargetArtifactExportTest.cpp`

## Evidence Completed

Production changes:

* Added `RVVRuntimeScalarComputedMaskMAccRouteFacts` and
  `getRVVRuntimeScalarComputedMaskMAccRouteFacts()` to the RVV provider surface.
* Rewired computed-mask MAcc provider planning/description verification to use
  that helper for runtime-scalar MAcc ABI, binding, mask, accumulator/result,
  header/type, target profile, and provider mirror facts.
* Rewired RVV target artifact MAcc validation to consume the same helper for
  runtime-scalar MAcc instead of maintaining duplicate target-local fact
  assembly.
* Tightened C++ artifact-export tests, script evidence extraction, explicit and
  pre-realized dry-run FileCheck coverage, and real `ssh rvv` evidence.

Checks run:

* `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `rtk ./build/bin/tianchenrv-target-artifact-export-test`
* `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* Manual `tcrv-opt` / `tcrv-translate` / `FileCheck-20` REALIZED, PLAN,
  HEADER, and stale provider/binding/ABI/header/type/scalar/layout checks for
  `pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir`
* Explicit and pre-realized runtime-scalar MAcc generated-bundle dry-runs plus
  ROOT/MACC/HARNESS `FileCheck-20` checks
* Real `ssh rvv` generated-bundle correctness:
  `artifacts/tmp/06-02-runtime-scalar-macc-ssh-rvv/pre-realized-runtime-scalar-cmp-masked-macc-add/runtime_scalar_cmp_masked_macc_add/evidence.json`
* Diff-only bounded old-authority scan over touched files found no new legacy
  i32m1, descriptor, source-front-door, source-artifact, or exact-intrinsic
  route-authority additions.
* `rtk git diff --check`

## Current Phase

Finish/archive.
