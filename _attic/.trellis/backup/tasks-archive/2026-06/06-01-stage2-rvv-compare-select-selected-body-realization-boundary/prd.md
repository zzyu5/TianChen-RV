# Stage2 RVV compare/select selected-body realization boundary

## Goal

Implement and harden the RVV plugin-local selected-body realization boundary
for the currently supported generic typed compare/select selected bodies. A
selected pre-realized typed `tcrv_rvv` compare/select body must be consumed by
the RVV plugin into explicit generic vector-level `tcrv_rvv.setvl` /
`with_vl` / load / compare / select / store or equivalent mask-carrying
structure before route construction. Route construction must fail closed if a
pre-realized compare/select body bypasses the selected lowering-boundary
producer or if stale legacy helper-body authority is presented as executable
route support.

## What I Already Know

* The Hermes Direction Brief is the task source for this round.
* The repository started this round on `main` with a clean worktree at
  `9512a6f8 rvv: harden elementwise selected-body realization boundary`.
* No current Trellis task existed; this task was created as
  `.trellis/tasks/06-01-stage2-rvv-compare-select-selected-body-realization-boundary`.
* The global spec says Stage 2 includes RVV plugin-local selected-body
  realization from selected pre-realized typed body to realized `tcrv_rvv`
  structure before faithful EmitC/intrinsic lowering.
* The RVV plugin spec defines one elementwise/compare-select selected-body
  realization boundary, and the previous archived task hardened the
  elementwise binary slice of that boundary.
* This round is bounded to compare/select realization and directly required
  registry, route planning/provider integration, and focused tests.

## Requirements

* Preserve the production authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level pre-realized
  `tcrv_rvv` compare/select body -> RVV plugin-owned realization -> realized
  typed `tcrv_rvv` body -> RVV route/provider facts ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
* The selected-body owner must validate and consume typed body/config/runtime
  facts, including element type, SEW, LMUL, policy, predicate kind, memory
  form, mask/source facts where applicable, and runtime AVL/VL/ABI roles.
* Supported plain compare/select pre-realized bodies must realize into
  explicit `setvl`, `with_vl`, compare-input loads, value/passthrough loads,
  compare, select, and store structure before provider route facts are
  collected.
* Supported computed-mask and runtime-scalar compare/select bodies must realize
  into explicit generic mask-carrying `tcrv_rvv` structure before provider
  route facts are collected, if already supported by the current codebase.
* Route construction must reject direct pre-realized compare/select body
  consumption. A pre-realized compare/select body that reaches emission-plan or
  EmitC route construction without selected-boundary realization is a
  fail-closed error.
* Route construction and planning must reject stale legacy helper-body bypasses
  instead of treating `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`,
  `rvv-i32m1`, exact intrinsic spelling, route ids, artifact names,
  source-front-door/source-artifact metadata, or emission-plan/status mirrors
  as route authority.
* Keep selected-body owner APIs in the dedicated owner header. Route-family
  planning headers must remain route-plan/provider interfaces only.
* If the full compare/select cluster is larger than one coherent round, finish
  the plain compare/select submodule and leave the exact computed-mask or
  runtime-scalar continuation point.

## Acceptance Criteria

* [x] The public selected lowering-boundary path realizes at least one
  supported plain typed compare/select pre-realized body into explicit
  `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, load, compare, select, and store ops
  before route facts are collected.
* [x] Already-supported computed-mask/runtime-scalar compare/select cases are
  either realized through the same selected boundary or explicitly identified
  as the bounded continuation point without weakening fail-closed behavior.
* [x] Route construction for already-supported compare/select cases consumes
  the realized body and still produces route-supported EmitC/target dry-run
  evidence.
* [x] A direct pre-realized compare/select body at route construction fails
  closed with a targeted selected-body realization diagnostic.
* [x] Missing runtime AVL/VL or required ABI binding fails closed before route
  construction.
* [x] Unsupported typed config or predicate fails closed before route
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
  negative test, or derived output rather than active authority.
* [x] Focused build/test commands for touched RVV plugin paths pass and
  `git diff --check` passes.

## Completion Notes

* Extended the C++ RVV plugin realization-boundary test so the same
  elementwise/compare-select owner cluster directly verifies plain
  compare/select, computed-mask select, runtime-scalar compare/select, and
  runtime-scalar dual compare-mask-and-select pre-realized bodies.
* The new assertions prove the owner materializes the expected explicit
  `setvl` / `with_vl` / load / splat / compare / mask_and / select / store
  shape before route facts and `TCRVEmitCLowerableRoute` construction.
* Existing production code already owned the compare/select realization and
  provider preflight. This round added the missing focused C++ boundary
  evidence without adding new predicates, new route families, common EmitC
  inference, or legacy helper compatibility.
* Bounded scan classification: remaining old-authority strings in touched
  production/test/spec paths are spec prohibitions, fail-closed guards,
  negative/stale tests, provider-derived target leaves, or emission-plan mirror
  checks. No active compare/select route authority was derived from legacy i32
  helper names, route ids, source-front-door/source-artifact metadata, or
  common EmitC inference.
* No `.trellis/spec/**` update is needed because this round implements existing
  Stage 2 selected-body owner/provider-boundary contracts without changing the
  durable architecture.

## Definition of Done

* `implement.jsonl` and `check.jsonl` contain only relevant spec/context
  entries and validate through the Trellis workflow.
* No runtime, correctness, or performance claim is made without real `ssh rvv`
  evidence.
* Trellis task status, evidence, workspace journal, and archive state truthfully
  record the completed bounded change or the exact continuation point.
* One coherent commit is created if the bounded task is complete.

## Out of Scope

* New predicates, reductions, MAcc, contractions, memory-movement coverage,
  dtype/LMUL clone batches, source-front-door routes, common EmitC semantic
  inference, artifact/report-only work, high-level frontend lowering, broad
  Stage 2 coverage expansion, or runtime/hardware correctness claims.
* Reintroducing exact-intrinsic, route-id, artifact-name, descriptor, or
  metadata authority outside RVV plugin-derived route facts.
* Adding new `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`, or
  `rvv-i32m1` compatibility surfaces.

## Technical Approach

Inspect the current RVV selected-body realization owner, typed
compare/select pre-realized ops, generic compare/select ops, compare/select
route planning/provider checks, and focused tests. If compare/select
realization already exists, harden missing boundary checks and add focused
evidence. If materialization is incomplete, update only the RVV plugin-local
owner path and directly required provider preflight so route construction
continues to consume realized typed structure rather than pre-realized bodies
or metadata.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: `.trellis/spec/guides/plugin-locality-review-guide.md`.
* Read: `.trellis/spec/guides/capability-first-design-guide.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-elementwise-selected-body-realization-boundary/`.
