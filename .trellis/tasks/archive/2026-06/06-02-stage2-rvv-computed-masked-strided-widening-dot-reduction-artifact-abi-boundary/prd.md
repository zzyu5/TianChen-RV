# Stage2 RVV computed-masked strided widening-dot reduction artifact ABI boundary

## Goal

Make `computed_masked_strided_input_widening_dot_reduce_add` a bounded
production-owned Stage 2 RVV contraction route boundary. The selected
`tcrv.exec` RVV variant must realize the typed computed-mask strided-input
widening dot-product reduction pre-realized body into explicit `tcrv_rvv`
structure, and the RVV plugin/provider plus target artifact validator must
consume one provider-owned route-fact surface for runtime ABI order,
compare-produced mask, i16mf2 strided source loads, i32 scalar seed/result,
stride ABI, widening-dot relation, headers/types, binding plan, target leaf
profile, and provider mirror before generated artifacts are accepted.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-masked strided widening-dot reduction artifact ABI boundary`

## What I Already Know

* Repository started clean from commit
  `090f1602 rvv: canonicalize runtime scalar masked macc facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes direction brief before source edits.
* The previous archived runtime-scalar masked MAcc task introduced
  `RVVRuntimeScalarComputedMaskMAccRouteFacts` and rewired provider planning
  plus target validation to consume a canonical RVV provider fact surface.
* Current code already has
  `TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp`,
  realization into `setvl` / `with_vl` / compare loads / strided dot source
  loads / `compare` / `masked_widening_dot_reduce` / scalar lane0 store, and
  focused pre-realized REALIZED/PLAN/HEADER checks.
* Current generated-bundle dry-run already records
  `computed_masked_widening_dot_reduce_boundary`, provider/target fact markers,
  counts `0,1,16,17,257`, stride pairs `2:3,3:2`, mask patterns, input
  patterns, strided source checks, inactive-lane skipping, scalar output, and
  tail preservation.
* Provider and target validation already contain substantial widening-dot
  route checks, but the computed-mask strided-input widening-dot facts are not
  yet exposed as a small provider-owned canonical helper equivalent to the
  runtime-scalar MAcc/reduction fact helpers.

## Requirements

* Add or expose a bounded RVV plugin-owned canonical route-fact surface for
  `ComputedMaskStridedInputWideningDotReduceAdd`.
* The canonical surface must cover runtime ABI order
  `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride`;
  compare predicate `slt`; mask role/source/form;
  i16mf2 source SEW/LMUL and strided load memory form;
  i32 accumulator/result SEW/LMUL and scalar seed/result lane0 layouts;
  runtime stride sources; widening-dot relation; masked widening product and
  strided load leaf facts; route operand binding plan/summary; contraction
  family plan; target leaf profile; required C headers and C type mapping; and
  `provider_supported_mirror`.
* Provider planning and provider fact verification must use that surface before
  `TCRVEmitCLowerableRoute` construction or target acceptance.
* Target artifact route-family validation must consume the same provider-owned
  surface for the computed-mask strided-input widening-dot route and fail
  closed on stale or missing runtime ABI order, binding plan/summary, compare
  mask facts, source stride facts, source/accumulator/result config,
  widening-dot relation, header/type facts, target leaf profile, or provider
  mirror.
* Existing REALIZED/PLAN/HEADER checks for the pre-realized fixture must pass
  or be tightened to check provider-derived facts.
* Generated-bundle dry-run evidence must continue to record provider-derived
  computed-mask strided-input widening-dot facts and must not be a script-only
  substitute for provider/target validation.
* Real `ssh rvv` generated-bundle evidence must prove counts `0,1,16,17,257`,
  at least stride pairs `2:3` and `3:2`, compare-produced mask patterns, signed
  i16 products accumulated into the i32 scalar seed, inactive lanes skipped,
  strided source elements used, skipped source elements ignored, lane0 scalar
  output, tail sentinel preservation, and runtime n/AVL behavior.

## Acceptance Criteria

* [x] Computed-mask strided-input widening-dot route facts are represented by
      one provider-owned C++ helper or equivalent provider fact surface, not by
      duplicated target-local assembly.
* [x] Provider route construction or route description verification validates
      the canonical runtime ABI order, binding plan/summary, compare/mask
      source facts, strided source facts, source/accumulator/result config,
      scalar seed/result layout, widening-dot relation, headers/types, target
      leaf profile, and provider-supported mirror.
* [x] Target artifact validation consumes the same fact surface and rejects
      stale/missing provider mirror, binding plan/order, compare mask source,
      source stride facts, i16mf2 source config, i32 accumulator/result config,
      scalar seed/result layout, widening-dot relation, header/type facts, and
      contraction route-family facts.
* [x] Existing pre-realized REALIZED/PLAN/HEADER checks pass or are tightened
      to prove provider-derived computed-mask strided widening-dot facts.
* [x] Generated-bundle dry-run records provider-derived widening-dot boundary
      facts, route operand binding with all exported ABI entries carrying
      `abi` and `hdr`, strided source facts, compare-produced mask facts, and
      harness pattern/stride loops.
* [x] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,16,17,257`, stride pairs `2:3,3:2`, mask patterns, and input
      patterns, proving active signed i16 dot contributions, inactive-lane
      skipping, strided source addressing, skipped-source isolation, scalar
      lane0 output, tail preservation, and runtime n/AVL.
* [x] Bounded old-authority scan over touched files finds no new positive
      reliance on legacy `i32m1`, descriptors, source-front-door routes,
      direct-C/source-export compute, exact-intrinsic authority, or common
      EmitC RVV semantic inference.
* [x] Focused build/test/script checks pass, Trellis task is finished and
      archived if complete, one coherent commit is created, and
      `git status --short` is clean after archive/commit.

## Definition Of Done

* PRD, implement/check context, journal, archive status, and commit are
  truthful.
* Production/default RVV provider and target validation paths changed; this
  task is not closed by helper-only, script-only, or evidence-only changes.
* Common EmitC/export remains neutral and does not infer RVV contraction, mask,
  stride, ABI, dtype/config, or route semantics.
* No LMUL/dtype clone batch, matmul, high-level Linalg/frontend lowering,
  source-front-door positive route, tuning database, report/index work, or
  harness-only commit is introduced.

## Technical Approach

1. Add a small C++ canonical route-fact helper for the computed-mask
   strided-input widening-dot reduction route in the RVV provider surface.
2. Rewire provider description verification and/or route-family plan
   validation to use that helper for computed-mask strided widening-dot ABI,
   binding, mask, stride, dtype/config, layout, relation, header/type, target
   profile, and provider mirror facts.
3. Rewire RVV target artifact widening-dot validation to consume the same
   helper for the computed-mask strided-input widening-dot route.
4. Tighten `TargetArtifactExportTest` and the pre-realized/generated-bundle
   checks where needed so stale computed-mask strided widening-dot facts fail
   closed through the production validators.
5. Run focused C++ build/tests, direct MLIR/FileCheck equivalents, generated
   bundle dry-run, real `ssh rvv` generated-bundle evidence, bounded
   old-authority scan, `git diff --check`, Trellis validation, archive, and
   commit.

## Out Of Scope

* LMUL/dtype clone batches or broad widening-dot matrices.
* Matmul, high-level Linalg/frontend lowering, source-front-door positives,
  generic tuning databases, dashboards, or report/index work.
* Moving RVV contraction, mask, stride, ABI, dtype/config, or route authority
  into common EmitC/export, scripts, route ids, artifact names, test names,
  descriptors, exact intrinsic spellings, or mirror metadata.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task read:

* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-runtime-scalar-masked-macc-accumulation-artifact-abi-boundary/prd.md`

Likely implementation/test files inspected:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-strided-input-widening-dot-reduce-add-fail-closed.test`
* `test/Target/TargetArtifactExportTest.cpp`

## Completion Evidence

Completed 2026-06-02T17:12:26+08:00.

* Added `RVVComputedMaskStridedInputWideningDotReduceRouteFacts` and
  `getRVVComputedMaskStridedInputWideningDotReduceRouteFacts` as the
  provider-owned canonical fact surface for
  `ComputedMaskStridedInputWideningDotReduceAdd`.
* Provider contraction route-family verification now compares the computed-mask
  strided widening-dot route description against that canonical surface before
  materialization.
* RVV target artifact validation now consumes the same canonical surface and
  fails closed on stale provider mirror, target leaf, header/type, binding,
  mask, stride, config, layout, relation, intrinsic, and route-family facts.
* `TargetArtifactExportTest` covers positive canonical fact equality and stale
  target leaf/header fact rejection.
* REALIZED/PLAN/HEADER fixture checks and generated-bundle dry-run checks were
  tightened for target leaf/profile facts, provider mirror, header/type facts,
  inactive-lane zeroing, binding summary, and widening-dot facts.
* `ssh rvv` generated-bundle correctness passed for counts `0,1,16,17,257`,
  stride pairs `2:3` and `3:2`, two mask patterns, and two input patterns.

## Current Phase

Finish/archive.
