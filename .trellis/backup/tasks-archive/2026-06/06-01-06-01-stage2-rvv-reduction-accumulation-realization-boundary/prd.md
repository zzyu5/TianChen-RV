# Stage2 RVV reduction-accumulation selected-body realization boundary

## Goal

Harden and prove the RVV plugin-local selected-body realization boundary for
the existing ordinary reduction/accumulation route family. A selected
pre-realized `tcrv_rvv.typed_reduce_pre_realized_body` must be consumed by the
RVV plugin into explicit generic vector-level `tcrv_rvv.setvl`,
`tcrv_rvv.with_vl`, input `load`, accumulator `load`, `reduce`, and `store`
structure before route construction. Route construction must fail closed when
the pre-realized body reaches route facts or `TCRVEmitCLowerableRoute`
construction directly, or when accumulator/result/runtime facts are stale,
missing, or unsupported.

## What I Already Know

* The Hermes Direction Brief is the task source for this round.
* The repository started this round on `main` with a clean worktree at
  `267aae0f rvv: prove widening conversion realization boundary`.
* No current Trellis task existed. This task was created as
  `.trellis/tasks/06-01-06-01-stage2-rvv-reduction-accumulation-realization-boundary`.
* The global spec and RVV plugin spec require the current authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level pre-realized
  `tcrv_rvv` body -> RVV plugin-owned realization -> realized typed
  `tcrv_rvv` body -> RVV route/provider facts ->
  `TCRVEmitCLowerableRoute` -> common EmitC.
* The RVV plugin spec lists `reduction` as a selected-boundary-only
  realization owner family. Pre-realized selected RVV bodies must be consumed
  by the public selected lowering-boundary materialization producer before
  route facts, route-control provider plans, statement plans,
  `TCRVEmitCLowerableRoute`, common EmitC, or target artifact export.
* The existing ordinary reduction owner already recognizes
  `tcrv_rvv.typed_reduce_pre_realized_body` and realizes it into explicit
  `setvl` / `with_vl` / `load` / accumulator `load` / `reduce` / `store`
  structure.
* The current C++ owner test proves owner lookup, null/wrong-family rejection,
  unsupported `op_kind` rejection, selected-boundary materialization, and final
  provider route construction after realization.
* Compared with the immediately preceding widening conversion task, the
  ordinary reduction owner test is missing a focused same-function proof that
  direct pre-realized route description and direct route construction fail
  closed, and that the realized body exposes reduction operation, ABI/runtime,
  accumulator/result layout, route-control, statement-plan, and route payload
  facts before provider-built route construction.

## Requirements

* Preserve the production authority chain for ordinary reduction:
  selected `tcrv.exec` RVV variant -> typed pre-realized reduction body -> RVV
  owner realization -> realized `tcrv_rvv` body -> route-family/provider facts
  -> `TCRVEmitCLowerableRoute`.
* Supported `reduce_add` pre-realized bodies must realize through the RVV
  selected-body owner registry and public selected lowering-boundary producer.
* Realization must preserve runtime `n`/AVL, input ABI role, accumulator seed
  ABI role, output ABI role, reduction operation kind, memory form,
  accumulator role/layout, result layout, SEW, LMUL, policy, selected variant
  `requires`, and target capability constraints.
* The realized structure must include one `setvl`, one `with_vl`, source
  `load`, accumulator `load`, `reduce {kind = "add"}`, and `store` before
  route facts are collected.
* Direct pre-realized reduction route description and direct
  `TCRVEmitCLowerableRoute` construction must fail closed before route
  construction with selected-body realization diagnostics.
* Missing or wrong runtime AVL/VL, input/accumulator/output ABI roles,
  unsupported op kind, memory form, accumulator role/layout, result layout,
  SEW/LMUL, policy, or mixed already-realized body ops must fail closed before
  provider route construction.
* Route-family/provider checks must continue to consume realized typed body
  facts and reject stale statement-plan dependencies; no common EmitC logic may
  infer reduction semantics.
* Legacy helper-body bypasses must remain negative/fail-closed and must not be
  accepted as route support through `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  `RVVI32M1*`, `rvv-i32m1`, exact intrinsic spelling, route ids, artifact
  names, source-front-door/source-artifact metadata, emission-plan/status
  mirrors, descriptors, scripts, or common EmitC inference.
* If the current production owner already implements the required boundary,
  add the focused missing evidence rather than broad test inventory.

## Acceptance Criteria

* [x] The public selected lowering-boundary path realizes supported
  `reduce_add` bodies into explicit `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`,
  source `load`, accumulator `load`, `reduce`, and destination `store`
  structure before route facts are collected.
* [x] Direct pre-realized reduction route description fails closed with a
  targeted selected-body realization diagnostic.
* [x] Direct pre-realized reduction `TCRVEmitCLowerableRoute` construction
  fails closed before provider route construction.
* [x] The realized reduction route description records operation kind,
  vector type, SEW/LMUL, runtime ABI order, accumulator layout, result layout,
  store VL, route operand-binding plan, and RVV provider-derived leaf facts
  from realized typed body structure.
* [x] Route-family provider verification, materialization facts, math
  operand-binding facts, ordinary reduction's empty route-control non-consumer
  result, reduction statement plan, and provider-built route construction
  succeed only after realization.
* [x] Focused negative checks cover unsupported reduction operation/config and
  runtime/ABI role failures that would otherwise bypass the owner boundary.
* [x] Focused route/emission or target artifact evidence for existing
  supported reduction cases remains passing.
* [x] A bounded old-authority scan over touched realization/provider/planning
  files, relevant tests, and specs classifies remaining hits for `RVVI32M1`,
  `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, and selected route wording as spec, negative/stale test,
  provider-derived mirror, or otherwise non-authoritative.
* [x] Focused build/test commands for touched RVV plugin paths pass and
  `git diff --check` passes.

## Definition of Done

* `implement.jsonl` and `check.jsonl` contain relevant spec/task context and
  validate through the Trellis workflow.
* No runtime, correctness, or performance claim is made without real `ssh rvv`
  evidence.
* Trellis task status, evidence, workspace journal, and archive state
  truthfully record the completed bounded change or the exact continuation
  point.
* One coherent commit is created if the bounded task is complete.

## Out of Scope

* Expanding standalone reduction clusters, computed-mask standalone reduction,
  MAcc, computed-mask MAcc, contractions, widening dot reductions, new
  reduction families, dtype/LMUL clone batches, source-front-door routes,
  common EmitC semantic inference, high-level frontend lowering, dashboards,
  broad smoke matrices, or runtime/hardware correctness claims.
* Moving reduction operation kind, dtype, SEW/LMUL, accumulator layout, result
  layout, policy, memory form, intrinsic, or route authority into common
  EmitC/export code.
* Reintroducing exact-intrinsic, route-id, artifact-name, descriptor, script,
  source-front-door, or metadata authority outside RVV plugin-derived route
  facts.
* Adding new `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`, or
  `rvv-i32m1` compatibility surfaces.

## Technical Approach

Inspect the ordinary reduction selected-body realization owner, the typed
ordinary reduction pre-realized op, generic `reduce` op, reduction
route-family statement plan, route-control provider plan, and focused C++/lit
tests. Current evidence indicates the production path already realizes
supported ordinary reduction bodies, so the planned change is a focused C++
selected-boundary test that proves:

* the `reduction` owner claims `typed_reduce_pre_realized_body`;
* route description and direct route construction fail before public
  selected-boundary materialization;
* owner-local validation rejects invalid operation/config/runtime facts;
* `materializeSelectedLoweringBoundary(...)` erases the pre-realized op and
  materializes the exact generic reduction sequence;
* the realized body feeds route description, route-family provider
  verification, materialization facts, math operand-binding facts,
  ordinary reduction's empty route-control non-consumer result, reduction
  statement-plan construction, and provider-built route creation.

If implementation inspection finds a production gap while writing this test,
fix only the directly required RVV plugin-local owner/provider path and keep
the same focused evidence target.

## Completion Notes

* The existing production owner path was sufficient; this round hardened the
  focused same-function C++ evidence in `runReductionSelectedBodyRealizationOwnerTest`.
* Added negative evidence that unsupported `op_kind`, invalid runtime `n` ABI
  role, and invalid accumulator layout fail inside the reduction realization
  owner before provider route construction.
* Added direct pre-realized route-description and
  `TCRVEmitCLowerableRoute` construction rejection checks before public
  selected-body materialization.
* Added post-realization evidence that the pre-realized op is erased and the
  realized body exposes `setvl`, `with_vl`, two `load`s, `reduce {kind =
  "add"}`, and `store`; route description, route-family verification,
  materialization facts, math operand-binding facts, ordinary reduction's empty
  route-control non-consumer result, reduction statement plan, and provider
  route construction all consume realized facts only.
* Bounded old-authority scan found remaining exact `__riscv_*_i32m1` mentions
  only as provider-derived focused evidence in the touched C++ test or as
  pre-existing/spec/negative classification hits. No new `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, `RVVI32M1*`, `rvv-i32m1`, source-front-door,
  source-artifact, descriptor, or common EmitC authority was introduced.
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
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-widening-conversion-realization-boundary/prd.md`.
* Inspected:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Plugin/RVV/RVVReductionSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVReductionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.
