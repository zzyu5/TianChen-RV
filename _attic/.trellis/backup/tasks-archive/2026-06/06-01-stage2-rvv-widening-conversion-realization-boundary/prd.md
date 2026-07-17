# Stage2 RVV widening conversion selected-body realization boundary

## Goal

Harden and prove the RVV plugin-local selected-body realization boundary for
the existing widening conversion route family. A selected pre-realized
widening conversion body must be consumed by the RVV plugin into explicit
generic vector-level `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, source `load`,
`widening_convert`, and `store` structure before route construction. Route
construction must fail closed when a pre-realized widening conversion body is
consumed directly or when stale helper-body, metadata, route mirror, dtype, or
SEW facts try to bypass realization.

## What I Already Know

* The Hermes Direction Brief is the task source for this round.
* The repository started this round on `main` with a clean worktree at
  `5629fc60 rvv: prove segment2 memory realization boundary`.
* No current Trellis task existed; this task was created as
  `.trellis/tasks/06-01-stage2-rvv-widening-conversion-realization-boundary`.
* The global spec says Stage 2 includes RVV plugin-local selected-body
  realization from selected pre-realized typed body to realized `tcrv_rvv`
  structure before faithful EmitC/intrinsic lowering.
* The RVV plugin spec lists `widening conversion` as a selected-boundary-only
  realization owner family. It also requires production route construction to
  fail closed if any pre-realized selected RVV body reaches route facts,
  route-control provider plans, statement plans, or `TCRVEmitCLowerableRoute`
  construction directly.
* The conversion/SEW policy testing contract says conversion kind, relation,
  source/result dtype, SEW, LMUL, materialized body, emitted C++, and metadata
  mirrors must come from typed `tcrv_rvv` body/config/runtime facts after
  provider route construction, not from route ids, artifact names, ABI strings,
  scripts, or intrinsic spelling.
* Current production code already has a dedicated widening conversion
  selected-body realization owner, route-family provider facts, route-control
  provider plan, widening conversion statement plan, and generated artifact
  coverage for `widen_i32_to_i64` and `widen_i16_to_i32`.
* The immediately preceding archived segment2 memory task added focused C++
  selected-boundary evidence that pre-realized bodies fail route construction,
  realize through the public selected lowering-boundary producer, and then feed
  provider/statement-plan route construction. Widening conversion should get
  the same targeted proof instead of relying on broad route-path tests alone.

## Requirements

* Preserve the production authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level pre-realized widening
  conversion `tcrv_rvv` body -> RVV plugin-owned realization -> realized typed
  `tcrv_rvv` body -> RVV route/provider facts ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
* Supported `widen_i32_to_i64` and `widen_i16_to_i32` pre-realized bodies must
  realize through the RVV selected-body owner registry and public selected
  lowering-boundary producer.
* Realization must preserve runtime `n`/AVL, source/result element type,
  source/result SEW, source/result LMUL, conversion relation, operation kind,
  memory form, policy, input/output ABI roles, selected variant `requires`,
  and target capability constraints.
* Realized `widen_i32_to_i64` structure must include explicit destination
  `setvl`/`with_vl` using SEW64 LMUL m2, source load using i32/m1,
  `widening_convert {kind = "widen_i32_to_i64"}`, and store using i64/m2
  before provider route facts are collected.
* Realized `widen_i16_to_i32` structure must include explicit destination
  `setvl`/`with_vl` using SEW32 LMUL m1, source load using i16/mf2,
  `widening_convert {kind = "sign_extend_widen_vf2"}`, and store using i32/m1
  before provider route facts are collected.
* Direct pre-realized widening conversion route consumption must fail closed
  before route description, route planning, `TCRVEmitCLowerableRoute`
  construction, common EmitC, or target artifact export.
* Missing runtime AVL/VL, missing ABI imports, unsupported source/result dtype
  or SEW transition, unsupported LMUL, unsupported policy, stale route
  materialization facts, stale route-control facts, stale operand-binding
  facts, or stale widening conversion statement leaves must fail closed before
  provider route construction.
* Stale legacy helper-body bypasses must remain negative/fail-closed and must
  not be accepted as ordinary route support through `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, `RVVI32M1*`, `rvv-i32m1`, exact intrinsic spellings,
  route ids, artifact names, source-front-door/source-artifact metadata,
  emission-plan/status mirrors, descriptors, scripts, or common EmitC
  inference.
* Keep selected-body owner APIs in
  `RVVWideningConversionSelectedBodyRealizationOwner.h`. EmitC route-family
  planning headers remain route-plan/provider interfaces only.
* If the current owner already implements the production path, add the single
  focused missing evidence rather than a broad test inventory. If the boundary
  is incomplete, update only the RVV plugin-local owner, directly required
  registry path, route planning/provider checks, and focused tests.

## Acceptance Criteria

* [x] The public selected lowering-boundary path realizes supported
  `widen_i32_to_i64` bodies into explicit `tcrv_rvv.setvl`,
  `tcrv_rvv.with_vl`, source `load`, `widening_convert`, and destination
  `store` structure before route facts are collected.
* [x] The public selected lowering-boundary path realizes supported
  `widen_i16_to_i32` bodies into explicit `setvl`, `with_vl`, source `load`,
  `widening_convert`, and destination `store` structure before route facts
  are collected.
* [x] Route construction for both supported widening conversion cases consumes
  the realized body and produces focused provider route evidence.
* [x] Direct pre-realized widening conversion route construction fails closed
  with a targeted selected-body realization diagnostic.
* [x] Missing runtime AVL/VL or required ABI binding fails closed before route
  construction.
* [x] Unsupported source/result dtype, SEW/LMUL transition, conversion
  relation, memory form, or policy fails closed before route construction.
* [x] Stale materialization, route-control, operand-binding, statement-plan,
  or provider-plan facts fail closed before statement-plan construction or
  provider route construction.
* [x] Legacy i32 helper-body bypasses remain negative/fail-closed and are not
  positive executable route support.
* [x] Focused tests cover realization sequence, route-supported emission for
  already-supported cases, and the negative paths above.
* [x] A bounded Stage 1/2 gate scan over touched realization/provider/planning
  files, relevant tests, and specs classifies any remaining hits for
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, and selected route wording as spec, parseable-only,
  negative test, provider-derived mirror, or otherwise non-authoritative.
* [x] Focused build/test commands for touched RVV plugin paths pass and
  `git diff --check` passes.

## Completion Notes

* Existing production code already realized
  `typed_widening_conversion_pre_realized_body` through the RVV selected-body
  owner registry and public selected lowering-boundary producer. This round
  added focused C++ selected-boundary evidence rather than changing production
  lowering logic.
* Strengthened `runWideningConversionSelectedBodyRealizationOwnerTest` in the
  RVV plugin C++ test. It now proves both supported pre-realized widening
  conversion forms are owned by the `widening conversion` selected-body
  realization owner, fail route description and direct route construction
  before realization, and are consumed by
  `materializeSelectedLoweringBoundary(...)` into explicit `setvl` /
  `with_vl` / `load` / `widening_convert` / `store` structure.
* The C++ test checks both realized structures:
  `widen_i16_to_i32` materializes SEW32 LMUL m1 destination control, i16/mf2
  source load, `sign_extend_widen_vf2`, and i32/m1 store;
  `widen_i32_to_i64` materializes SEW64 LMUL m2 destination control, i32/m1
  source load, `widen_i32_to_i64`, and i64/m2 store.
* The realized body then feeds widening conversion route description,
  route-family provider verification, materialization facts, math
  operand-binding facts, route-control provider plan, widening conversion
  statement-plan construction, provider preflight, and provider-built
  `TCRVEmitCLowerableRoute` construction for both supported cases.
* Existing lit/generated-bundle coverage already proves selected-boundary
  materialization, target-header metadata, dry-run evidence, direct
  pre-realized fail-closed behavior, unsupported dtype/SEW/relation/policy
  rejection, missing runtime ABI role rejection, and stale target artifact
  mirror rejection for the supported widening conversion cases. The focused
  lit filter and full `check-tianchenrv` run verified this evidence.
* Bounded old-authority scan classification: this diff introduced exact
  `__riscv_*_i32m1` strings only as expected provider-derived statement-plan
  callees in focused C++ evidence after route-family/provider validation.
  Remaining scanned hits are spec/PRD prohibitions, negative/fail-closed
  tests, provider-derived intrinsic mirrors after route construction,
  existing legacy fail-closed inventory, or explicit `emission_plan` mirrors.
  No new `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  source-front-door/source-artifact, descriptor, route-id, artifact-name, or
  common EmitC authority was added to production/default RVV paths.
* No `.trellis/spec/**` update is needed because this round implements and
  proves an existing Stage 2 widening conversion selected-body realization
  contract without changing durable architecture.

## Checks Run

* [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-stage2-rvv-widening-conversion-realization-boundary`
* [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`
* [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* [OK] Focused lit filter for generic widening conversion, pre-realized
  widening conversion negative tests, pre-realized selected-body target
  artifacts, direct pre-realized fail-closed tests, and generated-bundle dry
  runs: 8/8 passed.
* [OK] Bounded old-authority scan over touched realization/provider/planning
  files, relevant tests, specs, and this PRD; remaining hits classified above.
* [OK] `rtk git diff --check`
* [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 465/465.

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

* Adding new conversion families beyond the existing `widen_i32_to_i64` and
  `widen_i16_to_i32` boundary.
* Expanding dtype/LMUL clone batches, reductions, MAcc, contractions,
  segment/indexed/gather/scatter memory, source-front-door routes, common
  EmitC semantic inference, high-level frontend lowering, dashboards, broad
  smoke matrices, or runtime/hardware correctness claims.
* Moving conversion relation, dtype, SEW/LMUL, policy, memory-form,
  intrinsic, or route authority into common EmitC/export code.
* Reintroducing exact-intrinsic, route-id, artifact-name, descriptor, script,
  source-front-door, or metadata authority outside RVV plugin-derived route
  facts.
* Adding new `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`, or
  `rvv-i32m1` compatibility surfaces.

## Technical Approach

Inspect the widening conversion selected-body realization owner, the typed
widening conversion pre-realized op, generic `widening_convert` op, widening
conversion route-family planning/provider checks, and focused tests. Current
evidence indicates the production path already realizes supported widening
conversion bodies, so the planned change is a focused C++ selected-boundary
test that proves:

* the `widening conversion` owner claims both supported pre-realized forms;
* route description and direct route construction fail before public
  selected-boundary materialization;
* owner-local validation rejects bad dtype/SEW/relation facts;
* `materializeSelectedLoweringBoundary(...)` erases the pre-realized op and
  materializes the exact generic `tcrv_rvv` sequence for both cases;
* the realized body feeds widening conversion route-family planning,
  materialization facts, math operand-binding facts, route-control provider
  plan, statement-plan construction, provider preflight, and provider-built
  route creation.

If implementation inspection finds a production gap while writing this test,
fix only the directly required RVV plugin-local owner/provider path and keep
the same focused evidence target.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: `.trellis/spec/guides/plugin-locality-review-guide.md`.
* Read: `.trellis/spec/guides/capability-first-design-guide.md`.
* Read: `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-segment2-memory-realization-boundary/prd.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-conversion-sew-policy-route-closure/prd.md`.
* Read: `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` widening conversion
  pre-realized and generic conversion ops.
* Read: `include/TianChenRV/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.h`.
* Read: `lib/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.cpp`.
* Read: `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`.
* Read: `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
* Read: focused C++ and lit widening conversion tests in `test/Plugin`,
  `test/Dialect/RVV`, `test/Transforms/LoweringBoundary`, `test/Target/RVV`,
  and `test/Scripts`.
