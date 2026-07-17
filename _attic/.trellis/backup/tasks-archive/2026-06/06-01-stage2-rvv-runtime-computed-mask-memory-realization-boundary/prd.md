# Stage2 RVV runtime computed-mask memory selected-body realization boundary

## Goal

Harden and prove the RVV plugin-local selected-body realization boundary for
runtime-scalar computed-mask memory store and load-store bodies. A selected
pre-realized runtime-scalar computed-mask memory body must be consumed by the
RVV plugin into explicit generic vector-level `tcrv_rvv.setvl`,
`tcrv_rvv.with_vl`, load-or-splat, compare, and masked memory structure before
route construction. Route construction must fail closed when a pre-realized
computed-mask memory body is consumed directly or when stale i32/helper-body
authority tries to bypass realization.

## What I Already Know

* The Hermes Direction Brief is the task source for this round.
* The repository started this round on `main` with a clean worktree at
  `c3d73e59 rvv: prove base memory realization boundary`.
* No current Trellis task existed; this task was created as
  `.trellis/tasks/06-01-stage2-rvv-runtime-computed-mask-memory-realization-boundary`.
* The global spec says Stage 2 includes RVV plugin-local selected-body
  realization from selected pre-realized typed body to realized `tcrv_rvv`
  structure before faithful EmitC/intrinsic lowering.
* The RVV plugin spec lists `runtime scalar computed-mask store`,
  `runtime scalar computed-mask load-store`, and `computed-mask memory` as
  selected-boundary-only owner families. A pre-realized body may belong to an
  owner family, but production route construction must consume it through the
  public selected lowering-boundary producer before provider facts are
  collected.
* The RVV plugin spec requires non-segment computed-mask memory realization to
  preserve compare input loads before payload/passthrough/index materialization,
  then create the compare mask before masked load/store/strided/indexed
  operations.
* The computed-mask memory statement-plan spec requires provider statement
  construction to consume verified typed body/config/runtime facts, materialized
  route facts, memory operand-binding facts, and route-control facts. It must
  not rebuild the sequence from operation names, ABI strings, memory-form
  branches, mask-producer mirrors, or intrinsic mirrors.
* The lowering/runtime spec says common EmitC materializes only provider-built
  `TCRVEmitCLowerableRoute` payloads; it must not infer RVV memory form, dtype,
  SEW/LMUL, mask/tail policy, intrinsic spelling, or body shape.
* The immediately preceding archived base-memory task proved the adjacent
  selected-body boundary with focused C++ evidence and a direct pre-realized
  generated-bundle fail-closed test. This round should follow that evidence
  pattern without broadening Stage 2 coverage.

## Requirements

* Preserve the production authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level pre-realized
  runtime-scalar computed-mask memory `tcrv_rvv` body -> RVV plugin-owned
  realization -> realized typed `tcrv_rvv` body -> RVV route/provider facts ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
* Supported runtime-scalar computed-mask store and load-store pre-realized
  bodies must realize through the RVV selected-body owner registry and public
  selected lowering-boundary producer.
* Realization must preserve runtime `n`/AVL, scalar compare operands, produced
  mask facts, inactive-lane/tail policy, memory form, element type, SEW, LMUL,
  input/output ABI roles, and selected target capability constraints.
* Realized structure for supported runtime-scalar computed-mask memory bodies
  must include explicit `setvl`, `with_vl`, compare-LHS load,
  runtime-scalar RHS load-or-splat, compare, masked store or masked load/store,
  and required passthrough/payload memory operations before provider route
  facts are collected.
* Direct pre-realized computed-mask memory route consumption must fail closed
  before route description, route planning, `TCRVEmitCLowerableRoute`
  construction, common EmitC, or target artifact export.
* Missing runtime AVL/VL, missing scalar or mask facts, missing ABI imports,
  unsupported typed config, unsupported policy, unsupported memory form, stale
  materialization facts, or stale memory operand-binding facts must fail closed
  before route statement construction or provider route construction.
* Stale legacy helper-body bypasses must remain negative/fail-closed and must
  not be accepted as ordinary route support through `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, `RVVI32M1*`, `rvv-i32m1`, exact intrinsic spellings,
  route ids, artifact names, source-front-door/source-artifact metadata,
  emission-plan/status mirrors, descriptors, or common EmitC inference.
* Keep selected-body owner APIs in
  `RVVComputedMaskMemorySelectedBodyRealizationOwner.h`. EmitC route-family
  planning headers remain route-plan/provider interfaces only.
* If current production code already implements the required boundary, add the
  single focused missing evidence rather than a broad test inventory. If the
  boundary is incomplete, update only the RVV plugin-local owner, directly
  required registry path, route planning/provider checks, and focused tests.

## Acceptance Criteria

* [x] The public selected lowering-boundary path realizes supported
  `runtime_scalar_cmp_masked_store` bodies into explicit `tcrv_rvv.setvl`,
  `tcrv_rvv.with_vl`, compare-LHS load, runtime-scalar load-or-splat,
  compare, and typed `masked_store` structure before route facts are collected.
* [x] The public selected lowering-boundary path realizes supported
  `runtime_scalar_cmp_masked_load_store` bodies into explicit `setvl`,
  `with_vl`, compare-LHS load, runtime-scalar load-or-splat, compare,
  passthrough/source load as required, masked memory operation, and store
  structure before route facts are collected.
* [x] Route construction for already-supported runtime-scalar computed-mask
  memory cases consumes the realized body and still produces focused
  route/emission or target-artifact dry-run evidence.
* [x] A direct pre-realized runtime-scalar computed-mask memory body at route
  construction fails closed with a targeted selected-body realization
  diagnostic.
* [x] Missing runtime AVL/VL or required ABI binding fails closed before route
  construction.
* [x] Missing scalar compare, mask producer, passthrough/source, memory form,
  or stale materialization/operand-binding facts fail closed before route
  statement construction.
* [x] Unsupported typed config or unsupported policy fails closed before route
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

* Existing production code already realizes
  `typed_runtime_scalar_computed_mask_store_pre_realized_body` and
  `typed_runtime_scalar_computed_mask_load_store_pre_realized_body` through
  the RVV selected-body owner registry and public selected lowering-boundary
  producer. This round added the missing focused C++ selected-boundary
  evidence rather than widening production behavior.
* Added `runRuntimeScalarComputedMaskMemoryRealizationBoundaryTest` to the RVV
  plugin C++ test. It proves both runtime-scalar computed-mask memory
  pre-realized forms are registry owned before route planning, fail route
  description and direct route construction before realization, pass
  owner-local validation, and are consumed by
  `materializeSelectedLoweringBoundary(...)` into explicit `setvl` /
  `with_vl` / load-or-splat / compare / masked memory structure.
* The C++ test checks realized structures:
  `runtime_scalar_cmp_masked_store` materializes lhs load, scalar splat,
  payload load, compare, and `masked_store`; `runtime_scalar_cmp_masked_load_store`
  materializes lhs load, scalar splat, old-destination passthrough load,
  compare, `masked_load`, and store. Both then feed the computed-mask memory
  provider plan and build one provider route loop.
* Existing lit/generated-bundle coverage already proves positive
  selected-boundary materialization, target-header metadata, dry-run evidence,
  and direct pre-realized fail-closed behavior for both runtime-scalar
  computed-mask memory cases. Focused lit filters and the full
  `check-tianchenrv` run verified this evidence.
* Existing dialect/route planning negative tests cover missing runtime/ABI,
  unsupported config/policy, stale materialization or operand-binding facts,
  unsupported memory forms, and legacy helper-body bypass behavior for this
  route family.
* Bounded old-authority scan classification: this diff introduced no new
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`, or
  `emission_plan` hits. Remaining matches in scanned files are spec/PRD
  prohibitions, existing fail-closed/negative tests, provider-derived intrinsic
  mirrors after route construction, or non-authoritative selected-route wording.
* No `.trellis/spec/**` update is needed because this round implements and
  proves an existing Stage 2 runtime-scalar computed-mask memory realization
  contract without changing durable architecture.

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

* Expanding segment2, indexed/gather/scatter, reductions, MAcc, contractions,
  conversion/dtype clone batches, new predicates, source-front-door routes,
  common EmitC semantic inference, high-level frontend lowering, dashboards,
  broad smoke matrices, or runtime/hardware correctness claims.
* Moving mask, dtype, SEW/LMUL, policy, memory-form, or route authority into
  common EmitC/export code.
* Reintroducing exact-intrinsic, route-id, artifact-name, descriptor, or
  metadata authority outside RVV plugin-derived route facts.
* Adding new `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`, or
  `rvv-i32m1` compatibility surfaces.

## Technical Approach

Inspect the current computed-mask memory selected-body realization owner, the
runtime-scalar computed-mask memory pre-realized ops, generic masked memory
ops, computed-mask memory route-family planning/provider checks, and focused
tests. If runtime-scalar computed-mask memory realization already exists, add
focused C++ and lit evidence proving the public selected-boundary realization
and direct pre-realized fail-closed behavior. If implementation is incomplete,
update only the RVV plugin-local owner path and directly required provider
checks so route construction consumes realized typed memory structure rather
than pre-realized bodies, metadata, descriptors, or common EmitC inference.

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
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-base-memory-movement-selected-body-realization-boundary/prd.md`.
* Read: `.trellis/workspace/codex/journal-19.md` recent session 362.
