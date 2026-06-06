# Stage2 RVV strided-input widening dot-reduce-add resource path

## Goal

Make the base unmasked `strided_input_widening_dot_reduce_add` selected-body
resource path truthful and executable through the current production compiler
chain:

```text
selected/pre-realized typed tcrv_rvv body
  -> low-precision contraction resource selection facts
  -> RVV provider-owned route-family validation and direct-contraction plan
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact bundle
  -> generated-bundle ABI and ssh rvv correctness evidence
```

If the current path cannot support this representative, it must fail closed
with a targeted diagnostic naming the missing typed low-precision source,
product, accumulator/result, stride, resource, ABI/header, or runtime AVL/VL
fact. The task must not close as evidence-only if production code currently
accepts the route without the resource seam required by the Direction Brief.

## What I Already Know

- No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief before source edits.
- Initial repo state was `/home/kingdom/phdworks/TianchenRV`, branch `main`,
  clean `git status --short`, with HEAD
  `c5dcaf0b rvv: prove gearbox resource artifact boundary`.
- The previous archived task
  `.trellis/tasks/archive/2026-06/06-06-rvv-gearbox-resource-fact-artifact-boundary/`
  fixed the generated-bundle pass order for pre-realized
  `widening_product_reduce_dequantize_f32` so Gearbox/resource facts are
  available before selected-body realization/provider planning.
- The existing strided-input widening dot-reduce path already has explicit and
  pre-realized target fixtures, generated-bundle dry-run tests, stride-pair
  harness checks, and a fail-closed direct-pre-realized route-entry test.
- Current production planning has a structural
  `RVVLowPrecisionContractionResourceSelection` and carries it through
  contraction family plans, direct-contraction provider plans, route metadata,
  and target validation.
- Current `verifyRVVLowPrecisionContractionResourceSelection` only expects
  resource selection on product-reduction dequantization representatives, and
  rejects resource selection on non-dequant contraction descriptions. That is
  the likely production seam to repair for this task's base strided-input
  widening dot-reduce representative.
- Memory-derived context reinforces that this round must avoid evidence-only
  drift and should advance the real low-precision/quantized RVV Stage 2
  primitive seam.

## Requirements

- The base unmasked `strided_input_widening_dot_reduce_add` route must have a
  provider-consumed low-precision resource selection before route construction,
  or must fail closed with a targeted missing-resource diagnostic.
- Resource facts must derive from typed selected-body/config/runtime facts and
  selected target capability facts. They must not be inferred from route ids,
  artifact names, helper names, test names, ABI strings, common EmitC branches,
  descriptor residue, or emitted intrinsic spellings.
- The resource selection must validate narrow source dtype/SEW/LMUL, widening
  product dtype/SEW/LMUL/EMUL, accumulator/result dtype/SEW/LMUL, strided input
  memory form and stride sources, tail/mask policy, runtime AVL source, runtime
  ABI order, vector register budget, and legality/rejection state.
- The RVV provider must consume the validated resource selection into the
  direct-contraction provider plan before `TCRVEmitCLowerableRoute`
  construction.
- Target artifact validation and generated-bundle evidence must expose resource
  facts only as provider-derived mirrors after route construction, while still
  validating stride ABI/header/type/runtime payloads through provider-owned
  contracts.
- Explicit and pre-realized generated-bundle paths must both cover the base
  strided-input dot-reduce route. When executable correctness is claimed, the
  generated bundle must run on `ssh rvv`.
- At least one focused stale/missing fact must fail closed. Acceptable examples
  include missing low-precision resource selection, stale resource memory form,
  stale runtime ABI order, missing stride binding, or stale source/product/
  accumulator/result dtype facts.

## Acceptance Criteria

- [x] PRD and task context identify the bounded module owner, read-first files,
      non-goals, and continuation point.
- [x] Production code requires or derives a provider-consumed resource selection
      for base `strided_input_widening_dot_reduce_add` before route
      construction.
- [x] The resource selection validates strided memory facts, runtime ABI order,
      runtime AVL source, and widening product/reduction/accumulator/result
      facts without using metadata or route-name authority.
- [x] Explicit selected-body generated-bundle evidence shows materialized
      selected boundary, provider route facts, target artifact export, generated
      bundle compile path, strided harness coverage, and resource mirrors.
- [x] Pre-realized selected-body generated-bundle evidence shows selected-body
      realization/provider consumption before artifact export, with matching
      strided/resource facts.
- [x] A focused negative test fails closed for a missing or stale executable
      boundary fact.
- [x] If runtime correctness is claimed, non-dry-run generated-bundle evidence
      runs on `ssh rvv` for runtime counts `0,1,16,17,257`, stride pairs
      `2:3,3:2`, and two data patterns.
- [x] Focused checks pass:
      `build/bin/tianchenrv-rvv-extension-plugin-test`,
      `build/bin/tianchenrv-target-artifact-export-test`, relevant generated
      bundle dry-run tests, and `scripts/rvv_generated_bundle_abi_e2e.py
      --self-test`.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] A bounded old-authority scan over touched files/added lines shows no new
      positive legacy `i32m1`, descriptor, source-front-door, direct-C,
      route-id, artifact-name, bare status/readiness, or common EmitC RVV
      semantic authority.
- [x] Task status, journal/context notes, archive state, and final commit are
      truthful.

## Outcome

- Repaired the production strided-input widening dot-reduce-add seam by making
  this direct-contraction representative derive and validate a
  provider-consumed `RVVLowPrecisionContractionResourceSelection` before
  route construction.
- Extended direct-contraction route planning so the route provider requires
  the resource selection for base `strided_input_widening_dot_reduce_add`, not
  only product-reduction dequantization representatives.
- Moved target artifact resource mirror validation to any provider contract
  carrying a low-precision resource selection, then added a stale selected
  candidate negative test for the strided-dot target artifact path.
- Updated generated-bundle ABI evidence tooling and FileCheck tests so
  explicit and pre-realized strided-dot paths expose resource mirrors and
  provider route facts in the boundary summary.
- Proved executable correctness on `ssh rvv` for explicit and pre-realized
  selected bodies with runtime counts `0,1,16,17,257`, stride pairs `2:3` and
  `3:2`, and two data patterns.
- No spec update was required: the existing RVV plugin and EmitC route specs
  already required provider-owned resource facts and common EmitC neutrality;
  this round made the base strided-dot owner conform to those contracts.

## Out Of Scope

- No computed-masked strided-input widening dot-reduce expansion in this round.
- No broad strided-memory matrix.
- No dtype/LMUL clone batch.
- No q8/q4 performance parity claim, benchmark database, or autotuning cache.
- No product-dequant or dequant-clamp rework except as reference.
- No high-level Linalg/Vector/StableHLO frontend.
- No per-Linalg route authority.
- No source-front-door positive route.
- No common EmitC invention of RVV stride, dtype, resource, or contraction
  semantics.
- No dashboard/index/report-only closeout.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and shared guides under
  `.trellis/spec/guides/`.
- Previous task read:
  `.trellis/tasks/archive/2026-06/06-06-rvv-gearbox-resource-fact-artifact-boundary/`.
- Implementation files inspected or expected:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Focused fixtures/tests from the brief:
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-strided-input-widening-dot-reduce-add-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-strided-input-widening-dot-reduce-add-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-strided-input-widening-dot-reduce-add-fail-closed.test`,
  `test/Target/RVV/explicit-selected-body-artifact-strided-input-widening-dot-reduce-add.mlir`, and
  `test/Target/RVV/pre-realized-selected-body-artifact-strided-input-widening-dot-reduce-add.mlir`.

## Continuation Point If Larger Than One Round

If base unmasked strided-input widening dot-reduce requires broader resource
surface work than this round can safely complete, leave the task open at the
exact missing owner: resource candidate derivation, selected-body realization
copying, provider-plan consumption, target validation, generated-bundle ABI, or
`ssh rvv` correctness evidence. Computed-mask strided variants remain a later
bounded owner.
