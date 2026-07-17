# Stage2 RVV segment2 memory selected-body realization boundary

## Goal

Harden and prove the RVV plugin-local selected-body realization boundary for
the existing plain segment2 interleave/deinterleave memory movement path. A
selected pre-realized segment2 memory body must be consumed by the RVV plugin
into explicit generic vector-level `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`,
segment2 load-or-store, move, load, and store structure before route
construction. Route construction must fail closed when a pre-realized segment2
body is consumed directly or when stale helper-body, metadata, or route mirror
facts try to bypass realization.

## What I Already Know

* The Hermes Direction Brief is the task source for this round.
* The repository started this round on `main` with a clean worktree at
  `09091eff rvv: prove computed-mask memory realization boundary`.
* No current Trellis task existed; this task was created as
  `.trellis/tasks/06-01-06-01-stage2-rvv-segment2-memory-realization-boundary`.
* The global spec says Stage 2 includes RVV plugin-local selected-body
  realization from selected pre-realized typed body to realized `tcrv_rvv`
  structure before faithful EmitC/intrinsic lowering.
* The RVV plugin spec lists `segment2 memory` as a selected-boundary-only
  realization owner family. It also requires production route construction to
  fail closed if any pre-realized selected RVV body reaches route facts,
  route-control provider plans, statement plans, or `TCRVEmitCLowerableRoute`
  construction directly.
* The segment2 route-family planning spec says segment2 planning is an
  owner-built provider boundary and must not be rediscovered from operation
  names, ABI strings, route ids, artifact names, intrinsic mirrors, common
  EmitC, source-front-door markers, or legacy helper names.
* The segment2 memory statement-plan spec says provider statements must consume
  verified typed body/config/runtime facts, route materialization facts,
  memory operand-binding facts, route-control facts, and the owner-built
  segment2 provider plan.
* `TypedSegment2DeinterleaveMemoryPreRealizedBodyOp` and
  `TypedSegment2InterleaveMemoryPreRealizedBodyOp` already describe the
  intended explicit realization into segment2 load/move/store or
  load/load/segment2 store structure.
* Current production code already has a `segment2 memory` selected-body owner
  and the plain segment2 route-family provider/statement-plan path. This round
  should first harden the missing production boundary if one exists; otherwise
  add focused evidence that does not broaden Stage 2 coverage.
* The immediately preceding archived runtime computed-mask memory task added a
  focused C++ selected-boundary evidence pattern. Segment2 should get the same
  targeted proof instead of relying only on broad route-path tests.

## Requirements

* Preserve the production authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level pre-realized segment2
  memory `tcrv_rvv` body -> RVV plugin-owned realization -> realized typed
  `tcrv_rvv` body -> RVV route/provider facts -> `TCRVEmitCLowerableRoute`
  -> common EmitC -> target artifact.
* Supported `segment2_deinterleave_unit_store` and
  `segment2_interleave_unit_load` pre-realized bodies must realize through the
  RVV selected-body owner registry and public selected lowering-boundary
  producer.
* Realization must preserve runtime `n`/AVL, segment count 2, field pairing and
  field roles, source/destination memory forms, element type, SEW, LMUL,
  policy, input/output ABI roles, selected variant `requires`, and target
  capability constraints.
* Realized `segment2_deinterleave_unit_store` structure must include explicit
  `setvl`, `with_vl`, `segment2_load`, two field `move` operations, and two
  field `store` operations before provider route facts are collected.
* Realized `segment2_interleave_unit_load` structure must include explicit
  `setvl`, `with_vl`, two field `load` operations, and one `segment2_store`
  before provider route facts are collected.
* Direct pre-realized segment2 route consumption must fail closed before route
  description, route planning, `TCRVEmitCLowerableRoute` construction, common
  EmitC, or target artifact export.
* Missing runtime AVL/VL, missing ABI imports, unsupported segment count,
  swapped or duplicate field roles, unsupported memory forms, unsupported typed
  config, unsupported policy, stale route materialization facts, stale
  route-control facts, stale operand-binding facts, or mismatched owner family
  mirrors must fail closed before provider route construction.
* Stale legacy helper-body bypasses must remain negative/fail-closed and must
  not be accepted as ordinary route support through `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, `RVVI32M1*`, `rvv-i32m1`, exact intrinsic spellings,
  route ids, artifact names, source-front-door/source-artifact metadata,
  emission-plan/status mirrors, descriptors, scripts, or common EmitC
  inference.
* Keep selected-body owner APIs in
  `RVVSegment2MemorySelectedBodyRealizationOwner.h`. EmitC segment2
  route-family planning headers remain route-plan/provider interfaces only.
* If the current owner already implements the production path, add the single
  focused missing evidence rather than a broad test inventory. If the boundary
  is incomplete, update only the RVV plugin-local owner, directly required
  registry path, route planning/provider checks, and focused tests.

## Acceptance Criteria

* [x] The public selected lowering-boundary path realizes supported
  `segment2_deinterleave_unit_store` bodies into explicit `tcrv_rvv.setvl`,
  `tcrv_rvv.with_vl`, `segment2_load`, field `move`, and field `store`
  structure before route facts are collected.
* [x] The public selected lowering-boundary path realizes supported
  `segment2_interleave_unit_load` bodies into explicit `setvl`, `with_vl`,
  field `load`, and `segment2_store` structure before route facts are
  collected.
* [x] Route construction for both supported plain segment2 cases consumes the
  realized body and produces focused provider route evidence.
* [x] Direct pre-realized plain segment2 route construction fails closed with a
  targeted selected-body realization diagnostic.
* [x] Missing runtime AVL/VL or required ABI binding fails closed before route
  construction.
* [x] Unsupported segment count, segment layout, memory form, field role,
  typed config, or policy fails closed before route construction.
* [x] Stale materialization, route-control, operand-binding, or provider-plan
  facts fail closed before statement-plan construction or provider route
  construction.
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
  `typed_segment2_deinterleave_memory_pre_realized_body` and
  `typed_segment2_interleave_memory_pre_realized_body` through the RVV
  selected-body owner registry and public selected lowering-boundary producer.
  This round added focused C++ selected-boundary evidence rather than changing
  production lowering logic.
* Added `runSegment2MemoryRealizationBoundaryTest` to the RVV plugin C++ test.
  It proves both plain segment2 pre-realized forms are owned by the
  `segment2 memory` selected-body realization owner, fail route description
  and direct route construction before realization, pass owner-local
  validation, and are consumed by
  `materializeSelectedLoweringBoundary(...)` into explicit `setvl` /
  `with_vl` / segment memory structure.
* The C++ test checks realized structures:
  `segment2_deinterleave_unit_store` materializes `segment2_load`, two field
  `move` operations, and two stores; `segment2_interleave_unit_load`
  materializes two field loads and one `segment2_store`. Both then feed
  segment2 route-family provider planning, segment2 statement-plan
  construction, provider preflight, and `TCRVEmitCLowerableRoute`
  construction.
* Existing lit/generated-bundle coverage already proves positive pre-realized
  selected-boundary materialization, target-header metadata, dry-run evidence,
  and direct pre-realized fail-closed behavior for both plain segment2 cases.
  The full `check-tianchenrv` run verified this evidence.
* Existing dialect and provider negative tests cover missing runtime/ABI,
  unsupported segment count/layout/config/policy, stale materialization,
  stale route-control, stale operand-binding, stale provider-plan mirrors, and
  legacy helper/body bypass behavior for this route family.
* Bounded old-authority scan classification: this diff introduced exact
  `__riscv_*_i32m1` strings only as expected provider-derived statement-plan
  callees in the focused C++ evidence. Remaining scanned hits are spec/PRD
  prohibitions, parseable-only dialect debt, existing negative/fail-closed
  tests, provider-derived intrinsic mirrors after route construction, or
  explicit `emission_plan` mirrors. No new `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`, source-front-door/source-artifact, or
  descriptor authority was added to production/default RVV paths.
* No `.trellis/spec/**` update is needed because this round implements and
  proves an existing Stage 2 plain segment2 selected-body realization contract
  without changing durable architecture.

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

* Expanding indexed/gather/scatter, computed-masked segment2 update,
  reductions, MAcc, contractions, conversion/dtype clone batches, segment
  widths beyond segment2, source-front-door routes, common EmitC semantic
  inference, high-level frontend lowering, dashboards, broad smoke matrices, or
  runtime/hardware correctness claims.
* Moving segment layout, dtype, SEW/LMUL, policy, memory-form, intrinsic, or
  route authority into common EmitC/export code.
* Reintroducing exact-intrinsic, route-id, artifact-name, descriptor, script,
  source-front-door, or metadata authority outside RVV plugin-derived route
  facts.
* Adding new `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`, or
  `rvv-i32m1` compatibility surfaces.

## Technical Approach

Inspect the segment2 selected-body realization owner, the typed segment2
pre-realized ops, generic segment2 load/store ops, segment2 route-family
planning/provider checks, and focused tests. Current evidence indicates the
production path already realizes plain segment2 bodies, so the planned change
is a focused C++ selected-boundary test that proves:

* the `segment2 memory` owner claims both plain segment2 pre-realized forms;
* route description and direct route construction fail before public
  selected-boundary materialization;
* owner-local validation accepts the supported typed body facts;
* `materializeSelectedLoweringBoundary(...)` erases the pre-realized op and
  materializes the exact generic `tcrv_rvv` sequence;
* the realized body feeds segment2 route-family planning, statement-plan
  construction, and provider-built route creation.

If implementation inspection finds a production gap while writing this test,
fix only the directly required RVV plugin-local owner/provider path and keep
the same focused evidence target.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: `.trellis/spec/guides/plugin-locality-review-guide.md`.
* Read: `.trellis/spec/guides/capability-first-design-guide.md`.
* Read: `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-runtime-computed-mask-memory-realization-boundary/prd.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-base-memory-movement-selected-body-realization-boundary/prd.md`.
* Read: `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` segment2 pre-realized
  and generic segment memory ops.
* Read: `include/TianChenRV/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h`.
* Read: `lib/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.cpp`.
* Read: `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`.
* Read: `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`.
* Read: focused C++ and lit segment2 tests in `test/Plugin`,
  `test/Dialect/RVV`, `test/Target/RVV`, and `test/Scripts`.
