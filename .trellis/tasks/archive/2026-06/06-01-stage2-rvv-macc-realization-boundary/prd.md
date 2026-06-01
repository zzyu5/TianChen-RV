# Stage2 RVV MAcc selected-body realization boundary

## Goal

Harden and prove the RVV plugin-local selected-body realization boundary for
the existing plain and scalar-broadcast MAcc path. A selected
`tcrv_rvv.typed_macc_pre_realized_body` must be consumed by the RVV MAcc
selected-body realization owner into explicit generic vector-level
`tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, operand loads or scalar splat,
accumulator load, `tcrv_rvv.macc`, and `tcrv_rvv.store` structure before route
analysis, route-family provider facts, statement plans, or
`TCRVEmitCLowerableRoute` construction.

The boundary must preserve runtime `n`/AVL, lhs/rhs or rhs-scalar,
accumulator/output ABI roles, MAcc operation kind, accumulator/result layout,
element type, SEW, LMUL, policy, runtime-control facts, and target capability
facts as structural RVV plugin facts. Route ids, artifact names, helper names,
intrinsic spellings, diagnostics, or common EmitC must not become MAcc
authority.

## What I Already Know

* The Hermes Direction Brief is the task source for this round.
* The repository started clean on `main` at
  `b0015214 rvv: prove standalone reduction realization boundary`.
* No current Trellis task existed. This task was created as
  `.trellis/tasks/06-01-stage2-rvv-macc-realization-boundary`.
* `.trellis/spec/index.md` requires the RVV-first chain:
  selected RVV variant -> typed low-level `tcrv_rvv` body -> RVV plugin-owned
  selected-body realization / route provider -> common EmitC route.
* `.trellis/spec/extension-plugins/rvv-plugin.md` lists MAcc as a
  selected-boundary-only realization-owner family and requires provider route
  analysis to fail closed if a pre-realized body reaches route construction
  before the public selected lowering-boundary producer.
* `.trellis/spec/lowering-runtime/emitc-route.md` keeps common EmitC neutral:
  it consumes provider-built routes and must not choose RVV semantics,
  intrinsic names, dtype, SEW/LMUL, policy, or body shape.
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` defines
  `typed_macc_pre_realized_body` as a plugin-local pre-realized body that must
  become explicit `setvl` / `with_vl` / load or splat / accumulator-load /
  `macc` / store structure before provider route construction.
* `lib/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.cpp` already contains an
  owner-local realization path for `macc_add` and
  `scalar_broadcast_macc_add`.
* `test/Plugin/RVVExtensionPluginTest.cpp` already has broad pre-realized
  selected-body production tests and MAcc route-family / statement-plan tests.
  The missing bounded evidence is a focused MAcc selected-body realization
  owner test analogous to the standalone reduction task: direct pre-realized
  route construction must fail closed, then public selected-boundary
  materialization must erase the pre-realized body, produce the explicit MAcc
  op sequence, and feed realized route/provider facts.

## Requirements

* Preserve the authority chain:
  selected `tcrv.exec` RVV variant -> typed pre-realized MAcc body -> RVV
  MAcc owner realization -> realized `tcrv_rvv` body -> route-family analysis
  and provider facts -> `TCRVEmitCLowerableRoute`.
* Direct pre-realized MAcc route description and route construction must fail
  closed before provider route construction.
* Supported plain `macc_add` selected bodies must realize into one `setvl`,
  one `with_vl`, lhs load, rhs load, accumulator load, one `macc`, and one
  store.
* Supported `scalar_broadcast_macc_add` selected bodies must realize into one
  `setvl`, one `with_vl`, lhs load, RHS scalar splat, accumulator load, one
  `macc`, and one store.
* The realized route description must preserve operation kind, memory form,
  SEW/LMUL, policy, typed compute op name, runtime ABI order, route
  operand-binding plan, MAcc route-family plan mirror, accumulator/result
  layouts, runtime-control plan where required, and provider-derived leaves.
* Route-family provider verification, route materialization facts, math
  operand-binding facts, plain MAcc statement-plan construction, and
  provider-built route must consume realized facts only.
* Negative checks must cover wrong-family/null-style owner inputs where
  applicable, unsupported MAcc kind/config/policy, missing or wrong runtime
  AVL role, missing or wrong lhs/rhs or rhs-scalar/accumulator/output ABI
  facts, and stale route metadata or helper-body bypass attempts.
* Do not expand computed-mask MAcc, contraction/widening MAcc, dtype/LMUL clone
  batches, source-front-door routes, common EmitC semantic inference, high-level
  frontend lowering, dashboards, or broad smoke matrices.

## Acceptance Criteria

* [x] `implement.jsonl` and `check.jsonl` contain the relevant spec/task
  context and no code-path entries.
* [x] The MAcc selected-body owner is found by registry and rejects wrong-family
  inputs through owner-local diagnostics.
* [x] Direct pre-realized plain and scalar-broadcast MAcc route-entry attempts
  fail closed with the retired selected-body route-entry diagnostic before
  provider construction.
* [x] Direct pre-realized plain and scalar-broadcast MAcc provider route
  construction fails closed unless the public selected lowering-boundary
  producer has materialized the body.
* [x] Plain MAcc realization erases `typed_macc_pre_realized_body` and
  materializes exactly one `setvl`, one `with_vl`, three loads, one `macc`, and
  one store.
* [x] Scalar-broadcast MAcc realization erases `typed_macc_pre_realized_body`
  and materializes exactly one `setvl`, one `with_vl`, two loads, one `splat`,
  one `macc`, and one store.
* [x] Realized MAcc route description and route analysis record the expected
  operation kind, memory form, MAcc route-family plan, runtime ABI order,
  route operand-binding plan, accumulator/result layouts, and typed
  provider-derived leaf facts.
* [x] MAcc route-family provider verification, materialization facts, math
  operand-binding facts, statement-plan boundary, and provider-built route
  succeed only after selected-body realization.
* [x] Focused RVV plugin test target passes.
* [x] `git diff --check` passes.
* [x] A bounded old-authority scan over touched realization/provider/planning
  files, relevant tests, and specs classifies remaining hits for `RVVI32M1`,
  `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, and `selected route`.

## Definition of Done

* The bounded MAcc selected-body realization behavior is either implemented or
  proven already implemented with focused tests.
* No runtime, correctness, or performance claim is made without real
  `ssh rvv` evidence.
* Trellis task status, workspace journal, archive state, and final commit
  truthfully record the completed bounded change or the exact continuation
  point.

## Out of Scope

* Computed-mask MAcc expansion beyond directly required existing-path
  non-regression.
* Contractions, widening MAcc, dot reductions, standalone reductions, new MAcc
  families, dtype/LMUL clone batches, source-front-door routes, common EmitC
  semantic inference, high-level frontend lowering, dashboards, broad smoke
  matrices, and runtime/hardware correctness claims.
* Adding new `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`, or
  `rvv-i32m1` compatibility surfaces.

## Technical Approach

Add focused C++ coverage in `test/Plugin/RVVExtensionPluginTest.cpp`, reusing
the existing pre-realized selected-body fixture and MAcc owner/provider helper
APIs. Prefer a new focused MAcc selected-body realization owner test rather
than broad fixture inventory. If the focused test exposes a production gap,
fix only the directly required RVV plugin-local owner/provider path.

## Completion Notes

* Added `runMAccSelectedBodyRealizationOwnerTest` to
  `test/Plugin/RVVExtensionPluginTest.cpp` and wired it into the RVV plugin
  smoke test sequence after standalone reduction owner evidence.
* Verified that plain and scalar-broadcast MAcc pre-realized selected bodies
  fail closed before route facts and provider route construction, then
  materialize through the public selected lowering-boundary producer.
* Verified plain MAcc realization produces one `setvl`, one `with_vl`, lhs/rhs
  loads, accumulator load, one `macc`, and one store.
* Verified scalar-broadcast MAcc realization produces one `setvl`, one
  `with_vl`, lhs load, RHS scalar splat, accumulator load, one `macc`, and one
  store.
* Verified realized route description, route analysis, route-family provider
  plans, materialization facts, math operand-binding facts, route-control
  provider plan, plain MAcc statement plan, and provider-built route consume
  realized MAcc facts only.
* Added owner-local negative checks for null/wrong-family inputs, unsupported
  operation/memory-form pairing, unsupported LMUL/config, non-agnostic policy,
  wrong lhs/rhs/acc/out/n ABI roles, wrong scalar RHS role, and stale
  `route_id` metadata bypass attempts.
* No production owner/provider code change was needed; the existing MAcc owner
  and provider path already implemented the required boundary.
* Spec update judgment: no `.trellis/spec/` update was needed because the RVV
  plugin, EmitC route, and testing specs already described this MAcc selected
  body boundary and required evidence shape.
* Bounded old-authority scan over touched files and requested specs found the
  new exact `__riscv_*_i32m1` mentions only as provider-derived expected leaf
  checks in the focused C++ test. PRD hits are acceptance/out-of-scope text.
  Existing spec/test hits remain pre-existing fail-closed inventory,
  migration rules, or provider-derived route evidence; no new legacy helper,
  route-id, source-front-door, descriptor, artifact-name, or common EmitC
  authority was introduced.
* No `ssh rvv` evidence was collected and no runtime, correctness, or
  performance claim is made.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/index.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
* Read: `.trellis/spec/variant-pipeline/index.md`.
* Read: `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
* Read: `.trellis/spec/testing/index.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: `.trellis/spec/guides/index.md`.
* Read: `.trellis/spec/guides/plugin-locality-review-guide.md`.
* Read: `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-standalone-reduction-realization-boundary/prd.md`.
* Inspected:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.
