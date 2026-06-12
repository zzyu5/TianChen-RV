# Stage2 RVV standalone reduction selected-body realization boundary

## Goal

Harden and prove the RVV plugin-local selected-body realization boundary for
the existing standalone reduction cluster. A selected
`tcrv_rvv.typed_standalone_reduce_pre_realized_body` must be consumed by the
RVV standalone reduction realization owner into explicit generic vector-level
`tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, source `load`,
`standalone_reduce`, and scalar-result `store` structure before route facts or
`TCRVEmitCLowerableRoute` construction. The boundary must preserve runtime
`n`/AVL, source/accumulator/output ABI roles, reduction kind, scalar seed and
result-layout facts, element/config facts, policy, and target capability
constraints.

## What I Already Know

* The Hermes Direction Brief is the task source for this round.
* The repository started clean on `main` at
  `1b276a58 rvv: prove reduction realization boundary`.
* No current Trellis task existed. This task was created as
  `.trellis/tasks/06-01-06-01-stage2-rvv-standalone-reduction-realization-boundary`.
* The RVV plugin spec requires selected pre-realized RVV bodies to be consumed
  through the selected-body owner registry before route analysis, statement
  plans, `TCRVEmitCLowerableRoute`, common EmitC, or target artifact export.
* Standalone reduction is listed as a selected-boundary-only RVV realization
  owner family.
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` defines
  `typed_standalone_reduce_pre_realized_body` as a pre-realized selected body
  that must realize into explicit `setvl` / `with_vl` / `load` /
  `standalone_reduce` / `store` structure before provider route construction.
* `lib/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.cpp`
  already validates supported plain standalone reduction bodies and realizes
  them through the standalone reduction owner.
* Existing broad selected-body tests exercise standalone reduction through the
  shared producer path, and route-family/provider tests cover already-realized
  standalone reduction facts. The missing evidence is a focused owner-boundary
  test analogous to the previous ordinary reduction test.

## Requirements

* Preserve the authority chain:
  selected `tcrv.exec` RVV variant -> typed pre-realized standalone reduction
  body -> RVV owner realization -> realized `tcrv_rvv` body -> route-family
  analysis/provider facts -> `TCRVEmitCLowerableRoute`.
* Route description and route construction must fail closed when a direct
  pre-realized standalone reduction body reaches provider code before selected
  lowering-boundary materialization.
* Supported standalone `add` selected bodies must realize through the public
  selected lowering-boundary producer into one `setvl`, one `with_vl`, one
  source `load`, one `standalone_reduce`, and one scalar-result `store`.
* The realized standalone reduction op must preserve operation kind,
  accumulator seed layout, scalar result layout, source vector type/config,
  scalar result vector type/config, runtime ABI order, and runtime AVL/VL
  control facts.
* Negative owner checks must cover unsupported operation/config/policy,
  missing or wrong runtime AVL role, missing or wrong scalar output role, and
  stale/missing scalar seed or result-layout facts before provider route
  construction.
* Provider-side route facts, materialization facts, math operand-binding facts,
  route-control provider plan, standalone reduction statement plan, and
  provider-built route must succeed only after owner realization.
* Legacy helper-body or metadata bypasses must remain fail-closed. No new
  positive path may use `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
  `!tcrv_rvv.i32m`, exact intrinsic spelling, route id, artifact name,
  source-front-door/source-artifact metadata, descriptor residue, or common
  EmitC inference as standalone reduction authority.

## Acceptance Criteria

* [x] The standalone reduction selected-body owner is found by registry and
  rejects null/wrong-family bodies.
* [x] Direct pre-realized standalone reduction route description fails closed
  with a selected-body realization diagnostic.
* [x] Direct pre-realized standalone reduction route construction fails closed
  before provider route construction.
* [x] Negative owner checks reject unsupported op kind, unsupported LMUL/config,
  non-agnostic policy, wrong runtime `n` role, wrong scalar output role, wrong
  accumulator seed role/layout, and wrong scalar result layout.
* [x] The public selected lowering-boundary producer erases the pre-realized op
  and materializes explicit `setvl` / `with_vl` / `load` /
  `standalone_reduce` / `store` structure.
* [x] Realized route description records standalone operation kind, memory
  form, SEW/LMUL, source vector type, scalar result vector type, runtime ABI
  order, operand-binding plan, accumulator/result layouts, scalar result
  runtime boundary, store VL, and RVV provider-derived leaves.
* [x] Route-family provider verification, materialization facts, math
  operand-binding facts, route-control provider plan, standalone statement
  plan, and provider-built route consume realized facts only.
* [x] Focused RVV plugin test target passes; the broad `check-tianchenrv`
  target passes if practical for this round.
* [x] A bounded old-authority scan over touched files and relevant tests/specs
  classifies remaining hits.
* [x] `git diff --check` passes.

## Definition of Done

* `implement.jsonl` and `check.jsonl` contain the relevant spec/task context and
  validate through Trellis.
* No runtime, correctness, or performance claim is made without real `ssh rvv`
  evidence.
* Trellis task status, workspace journal, archive state, and final commit
  truthfully record the completed bounded change or the exact continuation
  point.

## Out of Scope

* Expanding computed-mask standalone reduction, runtime-scalar computed-mask
  standalone reduction, MAcc, computed-mask MAcc, contractions, new reduction
  families, dtype/LMUL clone batches, source-front-door routes, common EmitC
  semantic inference, high-level frontend lowering, dashboards, broad smoke
  matrices, or runtime/hardware correctness claims.
* Moving standalone reduction operation kind, dtype, SEW/LMUL, accumulator seed
  layout, scalar result layout, policy, memory form, intrinsic, or route
  authority into common EmitC/export code.
* Adding new `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`, or
  `rvv-i32m1` compatibility surfaces.

## Technical Approach

Add a focused C++ test in `test/Plugin/RVVExtensionPluginTest.cpp` mirroring
the previous ordinary reduction owner-boundary evidence for the standalone
reduction owner. If the focused test exposes a production gap, fix only the
directly required RVV plugin-local owner/provider path.

The expected primary implementation is test hardening: direct pre-realized
provider failures, owner-local negative validation, public selected-boundary
materialization, realized-route description/facts checks, statement-plan checks,
and provider route checks for the existing supported plain standalone
`standalone_reduce_add` case.

## Completion Notes

* The existing production standalone reduction owner/provider path already
  implemented the required selected-body realization boundary; no production
  owner/provider code change was needed.
* Added focused C++ evidence in `runStandaloneReductionSelectedBodyRealizationOwnerTest`
  that the standalone reduction owner is found by registry, rejects null and
  wrong-family bodies, and fails closed for unsupported op kind, unsupported
  LMUL/config, non-agnostic policy, wrong runtime `n` role, wrong scalar output
  role, wrong accumulator seed role/layout, and wrong scalar result layout.
* Added direct pre-realized route-description and provider-route construction
  rejection before selected-body materialization.
* Verified that the public selected lowering-boundary producer erases
  `tcrv_rvv.typed_standalone_reduce_pre_realized_body` and materializes one
  `setvl`, one `with_vl`, one source `load`, one `standalone_reduce`, and one
  scalar-result `store`.
* Verified that realized route description, route-family provider plans,
  materialization facts, math operand-binding facts, route-control provider
  plan, standalone statement plan, and provider-built route consume realized
  facts only.
* Bounded old-authority scan found the new exact `__riscv_*_i32m1` mentions
  only as provider-derived focused evidence in the C++ test. Remaining hits in
  scanned specs/tests are pre-existing spec text, fail-closed legacy inventory,
  stale-negative tests, or provider-derived leaf checks; no new legacy helper,
  route-id, source-front-door, descriptor, artifact-name, or common EmitC
  authority was introduced.
* No `ssh rvv` evidence was collected and no runtime, correctness, or
  performance claim is made.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: `.trellis/spec/guides/plugin-locality-review-guide.md`.
* Read: `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-reduction-accumulation-realization-boundary/prd.md`.
* Inspected:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.
