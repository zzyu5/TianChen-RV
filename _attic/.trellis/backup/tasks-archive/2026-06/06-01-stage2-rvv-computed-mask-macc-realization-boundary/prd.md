# Stage2 RVV computed-mask MAcc selected-body realization boundary

## Goal

Harden and prove the RVV plugin-local selected-body realization boundary for
the existing computed-mask MAcc path. A selected
`tcrv_rvv.typed_computed_mask_macc_pre_realized_body` or
`tcrv_rvv.typed_runtime_scalar_computed_mask_macc_pre_realized_body` must be
consumed by the RVV computed-mask MAcc selected-body realization owner into
explicit generic vector-level `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, compare
input materialization, mask construction, MAcc operand/accumulator
materialization, masked MAcc update, and store structure before route
analysis, route-family provider facts, statement plans, or
`TCRVEmitCLowerableRoute` construction.

The boundary must preserve runtime `n`/AVL, mask-computation inputs,
comparison kind, lhs/rhs/accumulator/output ABI roles, masked MAcc operation
kind, accumulator passthrough/update layout, element/result types, SEW, LMUL,
policy, runtime-control facts, and target capability facts as structural RVV
plugin facts. Route ids, artifact names, helper names, intrinsic spellings,
diagnostics, or common EmitC must not become computed-mask MAcc authority.

## What I Already Know

* The Hermes Direction Brief is the task source for this round.
* The repository started clean on `main` at
  `b38b1c00 rvv: prove macc realization boundary`.
* No current Trellis task existed. This task was created as
  `.trellis/tasks/06-01-stage2-rvv-computed-mask-macc-realization-boundary`.
* `.trellis/spec/index.md` requires the RVV-first chain:
  selected RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned selected-body realization / route provider -> common EmitC
  route.
* `.trellis/spec/extension-plugins/rvv-plugin.md` lists computed-mask MAcc as
  a selected-boundary-only realization-owner family and requires provider route
  analysis to fail closed if a pre-realized body reaches route construction
  before the public selected lowering-boundary producer.
* `.trellis/spec/lowering-runtime/emitc-route.md` keeps common EmitC neutral:
  it consumes provider-built routes and must not choose RVV semantics,
  intrinsic names, dtype, SEW/LMUL, policy, mask facts, MAcc operation, or body
  shape.
* The archived base MAcc task
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-macc-realization-boundary/`
  completed the directly preceding boundary by adding focused C++ evidence
  that direct pre-realized route consumption fails closed, then public
  selected-boundary materialization erases the pre-realized body and feeds
  realized provider facts.
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` defines computed-mask MAcc
  pre-realized bodies and generic mask/MAcc ops that should carry the
  operation, dtype/config, memory, mask, accumulator, and runtime facts
  structurally.
* `lib/Plugin/RVV/RVVComputedMaskMAccSelectedBodyRealizationOwner.cpp` is the
  owner-local implementation point for computed-mask and runtime-scalar
  computed-mask MAcc realization.
* `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp` and
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` are the provider-side
  facts and route-construction boundary that must consume realized structure
  only.

## Requirements

* Preserve the authority chain:
  selected `tcrv.exec` RVV variant -> typed pre-realized computed-mask MAcc
  body -> RVV computed-mask MAcc owner realization -> realized `tcrv_rvv`
  body -> route-family analysis and provider facts ->
  `TCRVEmitCLowerableRoute`.
* Direct pre-realized computed-mask MAcc route description and route
  construction must fail closed before provider route construction.
* Supported vector computed-mask MAcc selected bodies must realize into
  explicit `setvl`, `with_vl`, compare lhs/rhs loads, mask construction,
  MAcc lhs/rhs loads, accumulator load, masked MAcc update, and store
  structure.
* Supported runtime-scalar computed-mask MAcc selected bodies must realize into
  explicit `setvl`, `with_vl`, compare lhs load, runtime scalar RHS splat or
  equivalent mask input materialization, mask construction, MAcc lhs/rhs
  loads, accumulator load, masked MAcc update, and store structure.
* The realized route description must preserve operation kind, memory form,
  SEW/LMUL, policy, comparison predicate, mask role/source facts, typed compute
  op name, runtime ABI order, route operand-binding plan, computed-mask MAcc
  route-family plan mirror, accumulator/result layouts, runtime-control plan,
  and provider-derived leaves.
* Route-family provider verification, route materialization facts, math
  operand-binding facts, computed-mask MAcc statement-plan construction, and
  provider-built route must consume realized facts only.
* Negative checks must cover wrong-family/null-style owner inputs where
  applicable, unsupported MAcc kind/config/policy, missing or wrong runtime
  AVL role, missing or wrong mask/compare/lhs/rhs/accumulator/output ABI facts,
  wrong scalar or mask ABI role, and stale route metadata or helper-body bypass
  attempts.
* Do not expand contractions, new MAcc families, reductions, dtype/LMUL clone
  batches, source-front-door routes, common EmitC semantic inference,
  high-level frontend lowering, dashboards, or broad smoke matrices.

## Acceptance Criteria

* [x] `implement.jsonl` and `check.jsonl` contain the relevant spec/task
  context and no code-path entries.
* [x] The computed-mask MAcc selected-body owner is found by registry and
  rejects wrong-family inputs through owner-local diagnostics.
* [x] Direct pre-realized vector and runtime-scalar computed-mask MAcc
  route-entry attempts fail closed with the retired selected-body route-entry
  diagnostic before provider construction.
* [x] Direct pre-realized vector and runtime-scalar computed-mask MAcc provider
  route construction fails closed unless the public selected lowering-boundary
  producer has materialized the body.
* [x] Vector computed-mask MAcc realization erases
  `typed_computed_mask_macc_pre_realized_body` and materializes explicit
  generic `tcrv_rvv` setvl/with_vl/load/compare/mask/MAcc/store structure.
* [x] Runtime-scalar computed-mask MAcc realization erases
  `typed_runtime_scalar_computed_mask_macc_pre_realized_body` and materializes
  explicit generic `tcrv_rvv` setvl/with_vl/load/splat/compare/mask/MAcc/store
  structure.
* [x] Realized computed-mask MAcc route description and route analysis record
  the expected operation kind, memory form, comparison predicate, mask facts,
  MAcc route-family plan, runtime ABI order, route operand-binding plan,
  accumulator/result layouts, and typed provider-derived leaf facts.
* [x] Computed-mask MAcc route-family provider verification, materialization
  facts, math operand-binding facts, statement-plan boundary, and
  provider-built route succeed only after selected-body realization.
* [x] Focused RVV plugin test target passes.
* [x] `git diff --check` passes.
* [x] A bounded old-authority scan over touched realization/provider/planning
  files, relevant tests, and specs classifies remaining hits for `RVVI32M1`,
  `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, `descriptor`, and `selected route`.

## Definition of Done

* The bounded computed-mask MAcc selected-body realization behavior is either
  implemented or proven already implemented with focused tests.
* No runtime, correctness, or performance claim is made without real
  `ssh rvv` evidence.
* Trellis task status, workspace journal, archive state, and final commit
  truthfully record the completed bounded change or the exact continuation
  point.

## Out of Scope

* New contraction, widening MAcc, dot reduction, standalone reduction, memory,
  conversion, or future plugin work.
* Dtype/LMUL clone batches, source-front-door routes, common EmitC semantic
  inference, high-level frontend lowering, dashboards, broad smoke matrices,
  and runtime/hardware correctness claims.
* Adding new `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`, or
  `rvv-i32m1` compatibility surfaces.

## Technical Approach

Inspect the existing computed-mask MAcc owner, route-family facts, provider
preflight, and focused plugin tests. Prefer focused C++ evidence in
`test/Plugin/RVVExtensionPluginTest.cpp`, analogous to the archived base MAcc
task, over broad test inventory. If the focused test exposes a production gap,
fix only the directly required RVV plugin-local owner/provider path.

## Completion Notes

* Added `runComputedMaskMAccSelectedBodyRealizationOwnerTest` to
  `test/Plugin/RVVExtensionPluginTest.cpp` and wired it into the RVV plugin
  smoke sequence immediately after the base MAcc owner test.
* Verified the computed-mask MAcc owner is selected by the registry, rejects
  null/wrong-family inputs, and fails closed for stale helper-route metadata
  before provider route construction.
* Added owner-local negative checks for invalid predicate/mask facts, invalid
  LMUL/config, non-agnostic policy, wrong compare lhs/rhs roles, wrong payload
  lhs/rhs roles, wrong accumulator/output/runtime `n` roles, and wrong
  runtime-scalar RHS role.
* Verified vector computed-mask MAcc pre-realized bodies fail before direct
  route facts, then public selected-boundary materialization erases the
  pre-realized body and produces one `setvl`, one `with_vl`, five loads, one
  compare, one `masked_macc`, and one store.
* Verified runtime-scalar computed-mask MAcc pre-realized bodies fail before
  direct route facts, then materialize into one `setvl`, one `with_vl`, four
  loads, one scalar splat, one compare, one `masked_macc`, and one store.
* Verified realized route description, route analysis, route-family provider
  plans, materialization facts, math operand-binding facts, route-control plan,
  computed-mask accumulation statement plan, and provider-built route consume
  realized computed-mask MAcc facts only.
* No production owner/provider change was needed; the existing implementation
  already enforced the boundary. This round adds the missing focused evidence.
* `ctest -R tianchenrv-rvv-extension-plugin-test` reported no registered tests
  in the current build tree, so validation used the direct build target and
  direct test binary.
* Bounded old-authority scan over touched files and requested specs found the
  new exact `__riscv_*_i32m1` mentions only as provider-derived expected leaf
  checks in the focused C++ test. Existing spec/test hits remain migration
  rules, negative/fail-closed inventory, or provider-derived route evidence;
  no new legacy helper, route-id, source-front-door, descriptor, artifact-name,
  or common EmitC authority was introduced.
* No `ssh rvv` evidence was collected and no runtime, correctness, or
  performance claim is made.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/index.md`.
* Read: `.trellis/spec/guides/plugin-locality-review-guide.md`.
* Read: `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-macc-realization-boundary/prd.md`.
