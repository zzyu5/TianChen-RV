# Stage2 RVV base memory-movement selected-body realization boundary

## Goal

Harden the RVV plugin-local selected-body realization boundary for the
currently supported generic typed base memory-movement selected bodies. A
selected pre-realized unit or strided memory movement body must be consumed by
the RVV plugin into explicit generic vector-level `tcrv_rvv.setvl` /
`tcrv_rvv.with_vl` / load-or-store or strided-load-or-strided-store structure
before route construction. Route construction must fail closed if a
pre-realized base-memory body bypasses the selected lowering-boundary producer
or if stale legacy helper-body authority is presented as executable support.

## What I Already Know

* The Hermes Direction Brief is the task source for this round.
* The repository started this round on `main` with a clean worktree at
  `ae78f435 rvv: prove compare-select realization boundary`.
* No current Trellis task existed; this task was created as
  `.trellis/tasks/06-01-stage2-rvv-base-memory-movement-selected-body-realization-boundary`.
* The global spec says Stage 2 includes RVV plugin-local selected-body
  realization from selected pre-realized typed body to realized `tcrv_rvv`
  structure before faithful EmitC/intrinsic lowering.
* The RVV plugin spec lists base memory movement as a selected-body
  realization owner family and requires direct route-entry requests for
  pre-realized selected RVV bodies to fail closed before provider route
  construction.
* The lowering/runtime spec says common EmitC materializes only
  provider-built `TCRVEmitCLowerableRoute` payloads; it must not infer RVV
  memory form, dtype, SEW/LMUL, policy, intrinsic spelling, or body shape.
* The archived compare/select task proved the adjacent selected-body boundary
  with focused C++ evidence and no production-code changes; this round should
  first inspect whether base memory already has comparable production behavior
  and then harden the missing boundary or evidence.

## Requirements

* Preserve the production authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level pre-realized
  `tcrv_rvv` base memory body -> RVV plugin-owned realization -> realized
  typed `tcrv_rvv` body -> RVV route/provider facts ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
* The selected-body owner must validate and consume typed body/config/runtime
  facts, including element type, SEW, LMUL, policy, runtime AVL/VL, ABI role
  imports, memory form, source/destination roles, and stride operands where
  applicable.
* Supported unit and strided base memory pre-realized bodies must realize into
  explicit `setvl`, `with_vl`, `load`, `store`, `strided_load`, or
  `strided_store` structure before provider route facts are collected.
* Route construction must reject direct pre-realized base-memory body
  consumption. A pre-realized base-memory body that reaches route planning,
  emission-plan description, or EmitC route construction without selected-body
  realization is a fail-closed error.
* Missing runtime AVL/VL, missing required ABI imports, missing or stale stride
  operands, unsupported memory form, and unsupported typed config must fail
  closed before route construction.
* Stale legacy helper-body bypasses must remain negative/fail-closed and must
  not be accepted as ordinary route support through `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, `RVVI32M1*`, `rvv-i32m1`, exact intrinsic spellings,
  route ids, artifact names, source-front-door/source-artifact metadata,
  emission-plan/status mirrors, descriptors, or common EmitC inference.
* Keep selected-body owner APIs in
  `RVVBaseMemoryMovementSelectedBodyRealizationOwner.h`. Route-family
  planning headers must remain route-plan/provider interfaces only.
* If the current owner already implements the production path, add the missing
  focused evidence instead of broad test inventory. If implementation is
  incomplete, update only the RVV plugin-local owner, directly required
  registry path, provider preflight/planning path, and focused tests.

## Acceptance Criteria

* [x] The public selected lowering-boundary path realizes at least one
  supported unit base memory pre-realized body into explicit
  `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, and typed load/store structure before
  route facts are collected.
* [x] The public selected lowering-boundary path realizes supported strided
  base memory pre-realized bodies into explicit `setvl`, `with_vl`, stride
  operand consumption, and typed `strided_load` or `strided_store` structure
  before route facts are collected.
* [x] Route construction for already-supported unit/strided base memory cases
  consumes the realized body and still produces focused route/emission or
  target-artifact dry-run evidence.
* [x] A direct pre-realized base-memory body at route construction fails closed
  with a targeted selected-body realization diagnostic.
* [x] Missing runtime AVL/VL or required ABI binding fails closed before route
  construction.
* [x] Missing or stale stride facts for strided memory movement fail closed
  before route construction.
* [x] Unsupported typed config or unsupported memory form fails closed before
  route construction.
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

* Existing production code already had the RVV plugin-local base memory owner
  implementation for `strided_load_unit_store` and `unit_load_strided_store`.
  This round added the missing focused selected-boundary evidence rather than
  widening production behavior.
* Added `runBaseMemoryMovementRealizationBoundaryTest` to the RVV plugin C++
  test. It proves both supported base memory pre-realized forms are registry
  owned by `base memory movement`, fail route description and direct route
  construction before realization, pass owner-local validation, and are
  consumed by the public selected lowering-boundary producer into explicit
  `setvl` / `with_vl` / memory ops before provider route construction.
* The C++ test checks the realized structures:
  `strided_load_unit_store` materializes `strided_load` / `move` / `store`,
  and `unit_load_strided_store` materializes `load` / `move` /
  `strided_store`, preserving source/destination memory-form and runtime
  stride facts into the base memory provider plan.
* Added a lit generated-bundle dry-run fail-closed test for
  `--direct-pre-realized-route-entry --op-kind unit_load_strided_store`, which
  mirrors the existing strided-load fail-closed coverage.
* Existing dialect and EmitC negative tests already cover missing or stale
  runtime/stride facts, unsupported memory form/config, and legacy authority
  metadata for these base-memory forms; the focused lit filter and full
  `check-tianchenrv` run verified them.
* Bounded old-authority scan classification: no new old-authority strings were
  introduced in the diff. Remaining matches in scanned files are spec
  prohibitions, existing fail-closed/negative tests, provider-derived target
  leaf or intrinsic mirror checks after route construction, explicit
  `emission_plan` mirrors, and legacy selected-body fail-closed inventory.
* No `.trellis/spec/**` update is needed because this round implements and
  proves existing Stage 2 base-memory selected-body realization contracts
  without changing the durable architecture.

## Definition of Done

* `implement.jsonl` and `check.jsonl` contain relevant context entries and
  validate through the Trellis workflow.
* No runtime, correctness, or performance claim is made without real `ssh rvv`
  evidence.
* Trellis task status, evidence, workspace journal, and archive state
  truthfully record the completed bounded change or the exact continuation
  point.
* One coherent commit is created if the bounded task is complete.

## Out of Scope

* Expanding indexed, static-mask, computed-mask, segment2, reductions, MAcc,
  contractions, dtype/LMUL clone batches, source-front-door routes, common
  EmitC semantic inference, artifact/report-only work, high-level frontend
  lowering, broad Stage 2 coverage expansion, or runtime/hardware correctness
  claims.
* Reintroducing exact-intrinsic, route-id, artifact-name, descriptor, or
  metadata authority outside RVV plugin-derived route facts.
* Adding new `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`, or
  `rvv-i32m1` compatibility surfaces.

## Technical Approach

Inspect the current base memory selected-body realization owner, the typed
base-memory pre-realized ops, generic memory ops, base memory route-family
planning/provider checks, and focused tests. If unit/strided memory
realization already exists, add focused C++ and lit evidence proving the
boundary and the fail-closed route-construction behavior. If materialization
is incomplete, implement only the RVV plugin-local owner path and directly
required provider checks so route construction consumes realized typed memory
structure rather than pre-realized bodies, metadata, descriptors, or common
EmitC inference.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: `.trellis/spec/guides/plugin-locality-review-guide.md`.
* Read: `.trellis/spec/guides/capability-first-design-guide.md`.
* Read: `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-compare-select-selected-body-realization-boundary/prd.md`.
* Read: `.trellis/workspace/codex/journal-19.md` recent sessions 357-361.
