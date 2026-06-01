# Stage2 RVV elementwise selected-body realization boundary

## Goal

Implement and harden the RVV plugin-local selected-body realization boundary
for generic typed elementwise binary selected bodies. A selected pre-realized
typed `tcrv_rvv` elementwise binary body must be consumed by the RVV plugin
into explicit generic vector-level `tcrv_rvv.setvl` / `with_vl` / load /
binary / store structure before route construction. Route construction must
fail closed if a pre-realized body bypasses that selected lowering-boundary
producer or if stale legacy helper-body authority is presented as executable
route support.

## What I Already Know

* The Hermes Direction Brief is the task source.
* The repository started this round on `main` with a clean worktree at
  `b0135205 rvv: make target validation consume provider leaves`.
* No current Trellis task existed; this task was created as
  `.trellis/tasks/06-01-06-01-stage2-rvv-elementwise-selected-body-realization-boundary`.
* The global spec says Stage 2 includes RVV plugin-local selected-body
  realization from selected pre-realized typed body to realized `tcrv_rvv`
  structure before faithful EmitC/intrinsic lowering.
* The RVV plugin spec already defines the selected-body owner registry and an
  Elementwise/Compare-Select realization boundary. This round is bounded to the
  generic typed elementwise binary slice needed by the current Direction Brief.
* The previous archived task closed the target/export consume-only boundary so
  target artifact validation mirrors provider-derived leaves instead of
  choosing RVV intrinsic semantics locally.

## Requirements

* Preserve the production authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level pre-realized
  `tcrv_rvv` body -> RVV plugin-owned realization -> realized typed
  `tcrv_rvv` body -> RVV route/provider facts -> `TCRVEmitCLowerableRoute` ->
  common EmitC -> target artifact.
* The elementwise selected-body owner must validate and consume typed
  body/config/runtime facts, including element type, SEW, LMUL, policy, memory
  form, operation kind, and runtime AVL/VL/ABI roles.
* The realized body for supported generic typed elementwise binary cases must
  contain explicit `setvl`, `with_vl`, typed loads, typed binary op, and typed
  store before provider route construction.
* Route construction must reject direct pre-realized body consumption. A
  pre-realized elementwise body that reaches emission-plan or EmitC route
  construction without selected-boundary realization is a fail-closed error.
* Route construction and planning must reject stale legacy helper-body bypasses
  instead of treating `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`,
  `rvv-i32m1`, exact intrinsic spelling, route ids, artifact names,
  source-front-door/source-artifact metadata, or emission-plan/status mirrors
  as route authority.
* If existing code already implements part of this path, harden the boundary
  and add focused evidence rather than widening Stage 2 coverage.

## Acceptance Criteria

* [x] The public selected lowering-boundary path realizes a generic typed
  elementwise binary pre-realized body into explicit `tcrv_rvv.setvl`,
  `tcrv_rvv.with_vl`, load, binary, and store ops before route facts are
  collected.
* [x] Route construction for the already-supported elementwise binary cases
  consumes the realized body and still produces a route-supported EmitC/target
  path.
* [x] A direct pre-realized elementwise body at route construction fails closed
  with a targeted selected-body realization diagnostic.
* [x] Missing runtime AVL/VL or required ABI binding fails closed before route
  construction.
* [x] Unsupported typed config fails closed before route construction.
* [x] Legacy i32 helper-body bypasses remain negative/fail-closed and are not
  positive executable route support.
* [x] Focused tests cover realization sequence, route-supported emission for
  already-supported cases, and the negative paths above.
* [x] A bounded Stage 1 gate scan over touched realization/provider/planning
  files, relevant tests, and specs classifies any remaining hits for
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, and selected route wording as spec, parseable-only,
  negative test, or derived output rather than active authority.
* [x] Focused build/test commands for touched RVV plugin paths pass and
  `git diff --check` passes.

## Definition of Done

* `implement.jsonl` and `check.jsonl` contain only relevant spec/context
  entries and validate through the Trellis workflow.
* No runtime, correctness, or performance claim is made without real `ssh rvv`
  evidence.
* Trellis task status, evidence, workspace journal, and archive state truthfully
  record the completed bounded change or the exact continuation point.
* One coherent commit is created if the bounded task is complete.

## Out of Scope

* New elementwise operation kinds, dtype/LMUL clone batches, reductions,
  contractions, source-front-door routes, common EmitC semantic inference,
  artifact/report-only work, high-level frontend lowering, broad Stage 2
  coverage expansion, or runtime/hardware correctness claims.
* Reintroducing exact-intrinsic, route-id, artifact-name, descriptor, or
  metadata authority outside RVV plugin-derived route facts.
* Adding new `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`, or
  `rvv-i32m1` compatibility surfaces.

## Technical Approach

Inspect the current selected-body realization owner, RVV pre-realized/generic
ops, elementwise owner tests, and route provider checks. If the owner already
materializes the required structure, add the missing fail-closed provider
boundary and focused evidence. If the materialization is incomplete, update the
RVV plugin-local owner implementation only, preserving provider ownership of
route construction and common EmitC neutrality. Keep route-family planning
headers free of selected-body owner APIs.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: `.trellis/spec/guides/plugin-locality-review-guide.md`.
* Read: `.trellis/spec/guides/capability-first-design-guide.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-stage1-rvv-target-artifact-validation-consume-only-leaf-facts/prd.md`.
* Relevant journal context: `.trellis/workspace/codex/journal-19.md` entries
  for selected-body owner cleanup and recent Stage 1 typed RVV leaf derivation.
