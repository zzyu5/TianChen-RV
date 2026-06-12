# Stage2 RVV low-precision product-plus-reduction contraction chain

## Goal

Implement one production route-supported RVV low-precision contraction chain:
selected typed `tcrv_rvv` signed i8 unit-load operands feed a typed
`signed_widening_product` into an i16 product/intermediate value, then that i16
value feeds a typed i16-to-i32 standalone widening reduction with an i32
accumulator/result boundary. The route must be derived by the RVV
plugin/provider from explicit selected-body/config/runtime facts and validated
by target artifact logic. This is not a q8/q4/llama benchmark route and not a
standalone evidence-recording task.

## What I Already Know

- The repository had no current Trellis task at session start; this task was
  created from the Hermes Direction Brief.
- Commit `02ac1d54` added the route-supported signed i8 widening product
  primitive to an i16 result.
- Commit `ce34f7d6` added the route-supported standalone i16-to-i32 widening
  reduction primitive with i32 accumulator/result boundary.
- Current RVV mainline is selected typed `tcrv_rvv` body -> RVV plugin
  legality/realization/provider route -> common EmitC materializer -> target
  artifact.
- Specs require dtype, signedness, SEW/LMUL, policy, memory form, operation
  kind, runtime AVL/VL, intermediate dtype, accumulator/result layout, and
  result boundary to come from typed body/config/runtime facts, not route ids,
  artifact names, q-like benchmark names, metadata mirrors, descriptors, or
  common EmitC.

## Scope

- Add or repair the production selected-body route-supported chain owner for
  signed i8 unit-load sources -> i16 widening product -> i32 widening reduction
  result boundary.
- Keep the surface at low-level typed RVV execution-body level; do not add
  high-level Linalg/frontend lowering, q-like benchmark-specific authority, or
  descriptor/source-front-door computation.
- Derive provider route facts from explicit typed body/config/runtime facts:
  source signedness, source element type i8, product element type i16,
  accumulator/result element type i32, source/product/reduction SEW and LMUL,
  policy, unit-load memory form, runtime AVL/VL, operand bindings, product
  relation, reduction kind, scalar seed/result layout, and result store/boundary.
- Validate target artifact mirrors only after provider route construction and
  fail closed on stale or mismatched product/reduction chain facts.

## Requirements

- Production code changes are required before closure.
- A selected typed RVV body can structurally express and validate the bounded
  low-precision chain with source signedness, i8 source operands, i16 product,
  i16-to-i32 widening reduction, i32 accumulator/result boundary, runtime AVL/VL,
  and ABI operand bindings.
- RVV provider route planning composes or sequences the existing product and
  reduction facts without letting common EmitC, route strings, artifact names,
  test names, q-like names, or metadata mirrors choose semantics.
- Common EmitC/export may carry provider-built payloads but must not choose RVV
  dtype, vector type, intrinsic spelling, reduction semantics, product relation,
  accumulator layout, or ABI roles.
- Target validation rejects mismatched product/reduction dtype chains, stale
  provider descriptions, stale route payload fields, unsupported LMUL/SEW
  combinations, missing accumulator/result boundaries, or stale candidate
  metadata mirrors fail-closed.
- Focused tests prove the positive route-supported chain and at least one
  structural negative mismatch.

## Acceptance Criteria

- [ ] A production RVV dialect/plugin/target route or validation surface is
  changed for the bounded product-plus-reduction chain.
- [ ] Positive focused test proves route-supported selected-body behavior for
  signed i8 unit-load sources, i16 widening product, and i16-to-i32 widening
  reduction into an i32 result boundary.
- [ ] Negative focused tests prove at least one product/reduction dtype-chain,
  accumulator/result-boundary, or stale metadata mismatch fails closed.
- [ ] `tianchenrv-rvv-extension-plugin-test` passes if provider/plugin code
  changes.
- [ ] `tianchenrv-target-artifact-export-test` passes if target validation code
  changes.
- [ ] Focused lit/unit tests for the changed chain pass.
- [ ] Generated artifact dry-run or target fixture exposes the chain facts as
  provider-derived route/mirror data.
- [ ] `git diff --check` passes.
- [ ] Bounded scan over touched files finds no new q8/q4/llama/name-authority
  or legacy i32m1 positive route authority.
- [ ] Task status, PRD, and final report truthfully distinguish
  route-supported from executable/runtime claims.
- [ ] Worktree is clean after the final commit if the task is complete.

## Definition of Done

- The bounded chain is route-supported and target-validated, or the PRD is
  updated to name the exact remaining production blocker.
- No `ssh rvv` runtime correctness or performance claim is made unless a real
  RVV host run is executed and recorded.
- A coherent commit records production, test, task, and journal changes when the
  task is complete.

## Out Of Scope

- q8/q4/llama benchmark-specific route names, benchmark authority, or harnesses.
- Dequantization or full q-like model-kernel closure in this same round unless
  the product-plus-reduction route chain is already complete and the extra work
  is demonstrably trivial.
- Runtime/correctness/performance claims without real `ssh rvv` evidence.
- High-level Linalg/frontend generalization.
- Descriptor-driven computation, source-front-door positive routes, or
  compatibility wrappers preserving legacy i32m1 authority.
- Broad smoke matrices, dashboards, report-only tasks, or Trellis archive-only
  updates.

## Technical Approach

- Start from the existing route-supported primitive surfaces added by the two
  preceding tasks rather than creating a separate benchmark route.
- Prefer a small production chain fact/validation surface that composes the
  already-canonical low-precision product facts and widening standalone
  reduction facts.
- Keep provider-owned facts as the shared authority consumed by route planning,
  route description validation, target validation, and artifact metadata mirror
  checks.
- Add focused lit and C++ target/provider tests for the positive chain and a
  representative fail-closed mismatch.

## Technical Notes

- Required specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Prior task references:
  `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-low-precision-widening-product-route/`,
  `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-i16-i32-widening-reduction-route/`.
- First code inspection targets:
  `test/Dialect/RVV/generic-widening-product-dataflow.mlir`,
  `test/Dialect/RVV/generic-widening-standalone-reduction-dataflow.mlir`,
  `test/Target/RVV/explicit-selected-body-artifact-widening-standalone-reduce-add.mlir`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`.
