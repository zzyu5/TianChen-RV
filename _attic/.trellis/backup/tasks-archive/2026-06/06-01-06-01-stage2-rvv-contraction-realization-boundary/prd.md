# Stage2 RVV contraction selected-body realization boundary

## Goal

Harden and prove the RVV plugin-local selected-body realization boundary for
the existing contraction owner path. Supported pre-realized contraction bodies
must be consumed by the RVV contraction selected-body realization owner into
explicit generic vector-level `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`,
load/strided-load or movement, widening MAcc or widening dot-reduction,
optional compare-produced mask structure, accumulator/result movement, and
store structure before route analysis, provider planning, statement planning,
or `TCRVEmitCLowerableRoute` construction.

The boundary must preserve runtime `n`/AVL and VL, lhs/rhs or dot-lhs/dot-rhs
operand roles, accumulator and output roles, optional stride roles, optional
computed-mask roles, contraction layout/relation facts, accumulation/update
kind, element/result types, SEW, LMUL, memory form, policy, target capability
facts, and provider-derived route leaves as RVV plugin structural facts. Route
ids, artifact names, helper names, intrinsic spellings, diagnostics, and common
EmitC must not become contraction authority.

## What I Already Know

* The Hermes Direction Brief is the task source for this round.
* The repository started clean on `main` at
  `9f887121 rvv: prove computed-mask macc realization boundary`.
* No `.trellis/.current-task` existed. This task was created as
  `.trellis/tasks/06-01-06-01-stage2-rvv-contraction-realization-boundary`.
* `.trellis/spec/index.md` requires the RVV-first chain:
  selected RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned selected-body realization / route provider -> common EmitC
  route.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires the RVV plugin to
  own selected-body realization, route support, intrinsic mapping, ABI mapping,
  legality, and fail-closed diagnostics.
* `.trellis/spec/lowering-runtime/emitc-route.md` keeps common EmitC neutral:
  it consumes provider-built routes and must not choose RVV semantics,
  intrinsic names, dtype, SEW/LMUL, policy, operand layout, or body shape.
* The immediately preceding computed-mask MAcc task proved the same kind of
  selected-boundary handoff for computed-mask MAcc; this contraction task must
  follow that pattern without expanding into new high-level frontend or source
  artifact authority.
* The existing contraction owner files are
  `include/TianChenRV/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.h`
  and `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`.
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` already defines bounded
  contraction pre-realized bodies and generic realized ops:
  `typed_widening_macc_pre_realized_body`,
  `typed_widening_dot_reduce_pre_realized_body`,
  `typed_strided_input_widening_dot_reduce_pre_realized_body`,
  `typed_computed_mask_widening_dot_reduce_pre_realized_body`,
  `typed_computed_mask_strided_input_widening_dot_reduce_pre_realized_body`,
  `widening_macc`, `widening_dot_reduce`, and
  `masked_widening_dot_reduce`.
* Initial inspection found an existing
  `runPreRealizedContractionRouteEntryOwnerTest` in
  `test/Plugin/RVVExtensionPluginTest.cpp`. This task should not duplicate
  broad inventory; it should close any missing focused evidence or directly
  fix a production boundary gap if the current test exposes one.

## Requirements

* Preserve the authority chain:
  selected `tcrv.exec` RVV variant -> typed pre-realized contraction body ->
  RVV contraction owner realization -> realized `tcrv_rvv` body ->
  route-family analysis and provider facts -> `TCRVEmitCLowerableRoute`.
* Direct route-entry use of pre-realized contraction bodies must fail closed
  before provider route construction.
* Route analysis and provider construction must reject any pre-realized
  contraction body that reaches route collection without public selected
  lowering-boundary materialization.
* Supported widening MAcc bodies must realize into explicit
  `setvl`/`with_vl`/source loads/accumulator load/`widening_macc`/store
  structure.
* Supported widening dot-reduction bodies must realize into explicit
  `setvl`/`with_vl`/source loads or strided source loads/
  `widening_dot_reduce`/store structure.
* Supported computed-mask dot-reduction bodies must realize compare loads,
  compare-produced mask structure, source loads or strided source loads,
  `masked_widening_dot_reduce`, and store structure.
* Route-family provider verification, materialization facts, math
  operand-binding facts, route-control facts, direct contraction statement
  planning, and provider-built route construction must consume realized facts
  only.
* Negative checks must cover missing or wrong runtime AVL/VL facts, unsupported
  config/policy/layout/relation/kind facts, missing operand/accumulator/output
  facts, wrong ABI role, stale route-family/operand-binding/provider mirrors,
  and legacy helper-body bypass attempts where applicable.
* Do not add high-level Linalg/matmul frontend authority, new contraction
  families beyond the current coherent boundary, dtype/LMUL clone batches,
  source-front-door routes, common EmitC semantic inference, dashboards, or
  broad smoke matrices.

## Acceptance Criteria

* [x] `implement.jsonl` and `check.jsonl` contain relevant spec/task context
  and no source-code paths.
* [x] Contraction pre-realized bodies dispatch through the RVV contraction
  realization owner registry and wrong-family inputs fail closed.
* [x] Direct pre-realized contraction route-entry attempts fail closed with the
  retired selected-body route-entry diagnostic before provider construction.
* [x] Route analysis/provider construction rejects direct pre-realized
  contraction consumption until public selected-boundary realization erases the
  pre-realized body.
* [x] Widening MAcc, widening dot-reduction, strided-input dot-reduction,
  computed-mask dot-reduction, and computed-mask strided-input dot-reduction
  bodies materialize explicit generic `tcrv_rvv` structure before route facts
  are collected, or any unsupported member fails closed with targeted
  diagnostics.
* [x] Realized contraction route descriptions and route-family plans preserve
  operation kind, memory form, source/result SEW/LMUL, policy, accumulator and
  result layouts, relation, runtime ABI order, target leaf profile, provider
  support mirror, operand-binding plan, runtime-control facts, and
  provider-derived leaves.
* [x] Focused negative coverage rejects stale provider facts for runtime
  AVL/VL, unsupported policy/config, missing operand/accumulator/output/stride
  facts, wrong ABI roles, stale layout/relation/mask facts, and stale
  helper-route bypass attempts.
* [x] A focused RVV plugin test target passes.
* [x] `git diff --check` passes.
* [x] A bounded old-authority scan over touched realization/provider/planning
  files, relevant tests, and specs classifies remaining hits for `RVVI32M1`,
  `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, `descriptor`, and `selected route`.

## Definition of Done

* The bounded contraction selected-body realization behavior is either
  implemented or proven already implemented with focused tests.
* No runtime, correctness, or performance claim is made without real
  `ssh rvv` evidence.
* Trellis task status, workspace journal, archive state, and final commit
  truthfully record the completed bounded change or the exact continuation
  point.

## Out of Scope

* New high-level Linalg, matmul, or tensor frontend lowering.
* New contraction families beyond the existing widening MAcc / widening
  dot-reduction / strided-input / computed-mask contraction boundary.
* Dtype/LMUL clone batches, source-front-door routes, common EmitC semantic
  inference, dashboards, broad smoke matrices, and runtime/hardware
  correctness or performance claims.
* Adding new `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`, or
  `rvv-i32m1` compatibility surfaces.

## Technical Approach

Inspect the existing contraction selected-body realization owner, contraction
route-family plan owners, route planning/provider preflight, direct
contraction statement-plan owner, and focused RVV plugin tests. Prefer one
focused production-boundary fix or one focused missing evidence addition over
broad inventory. If the existing implementation and test already enforce most
of the requested behavior, add the smallest missing coverage that makes the
contraction selected-body realization boundary trustworthy.

## Completion Notes

* The existing contraction realization owner already materialized all five
  bounded contraction pre-realized families through explicit generic
  `tcrv_rvv` structure before route analysis.
* Added focused owner-local negative evidence in
  `test/Plugin/RVVExtensionPluginTest.cpp` for computed-mask strided
  contraction rejecting non-agnostic policy and wrong ABI roles on compare
  lhs/rhs, dot lhs/rhs, accumulator seed, output, runtime `n`/AVL, lhs stride,
  and rhs stride.
* Reused the existing `runPreRealizedContractionRouteEntryOwnerTest` instead
  of adding broad inventory. The test already proves direct pre-realized
  route-entry demotion, selected-boundary materialization, route-family facts,
  operand-binding facts, materialization facts, direct provider plan facts,
  and provider-built route/emission after realization.
* No production owner/provider change was required; this round adds the
  missing focused evidence that the existing production boundary rejects stale
  contraction role/config authority before route construction.
* Bounded old-authority scan found no requested legacy-authority strings in the
  new diff hunk. Existing hits in related code are selected-route diagnostic
  wording and a fail-closed `tcrv_rvv.i32_` legacy guard. Existing test hits
  are fail-closed source-front-door/legacy-body tests, stale `rvv-i32m1`
  metadata negative checks, provider-derived exact-intrinsic leaf assertions,
  and selected-route analysis wording. Existing spec hits are guardrails,
  negative examples, or historical/deprecated inventory.
* No `ssh rvv` evidence was collected and no runtime, correctness, or
  performance claim is made.

## Checks

* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-06-01-stage2-rvv-contraction-realization-boundary`
* [x] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* [x] `rtk git diff --check`
* [x] `rtk cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 465/465.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/index.md`.
* Read: `.trellis/spec/guides/plugin-locality-review-guide.md`.
* Read: `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-computed-mask-macc-realization-boundary/prd.md`.
