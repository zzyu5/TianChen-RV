# Stage2 RVV computed-mask segment2 load/store runtime ABI closure

## Goal

Close the selected-boundary-to-artifact runtime ABI path for
`computed_masked_segment2_load_unit_store` and
`computed_masked_segment2_store_unit_load`. Both routes must execute from
selected typed `tcrv.exec` RVV variants through explicit typed `tcrv_rvv`
segment2 body facts, RVV plugin-owned realization/provider facts, the common
EmitC route, segment2 target artifact validation, generated RVV C artifacts,
and real `ssh rvv` correctness evidence.

This task is not an evidence-only follow-up. If production code already mostly
supports the path, this round must still verify the production boundary and fix
stale route/provider/script/harness assumptions discovered while closing the
runtime ABI.

## What I already know

- The repo began this round at `/home/kingdom/phdworks/TianchenRV`, branch
  `main`, clean worktree, HEAD `05e28f57 rvv: factor segment2 artifact
  validator boundary`.
- No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief and started as the current Trellis task.
- The previous archived update task closed
  `computed_masked_segment2_update_unit_load` runtime ABI evidence and kept
  load/store as dry-run non-regressions.
- The immediately preceding validator-boundary task moved segment2 target
  artifact acceptance behind a target-owned route-family validator consuming
  rebuilt provider route facts for all five segment2 consumers.
- Specs require computed-mask segment2 load/store to be selected-boundary-only:
  direct pre-realized route-entry must remain fail-closed even though the same
  typed bodies are accepted by the segment2 selected-body realization owner.
- `tcrv.exec` owns ABI/runtime envelope roles only. RVV operation kind, mask
  provenance, tuple field roles, memory direction/form, SEW/LMUL/policy,
  runtime AVL/VL use, route materialization, and intrinsic/header/type mapping
  must come from typed `tcrv_rvv` body facts and RVV plugin-owned provider
  facts.

## Scope

- Production owners, if stale assumptions are found:
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- Direct evidence and consumer files:
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Target/TargetArtifactExportTest.cpp`
  - directly related computed-mask segment2 load/store target, generated
    bundle, and selected-boundary tests.
- Non-regression only:
  - `computed_masked_segment2_update_unit_load`
  - plain segment2 deinterleave/interleave selected-boundary paths.

## Requirements

- The selected-boundary artifact path for both load/store consumers must
  preserve and validate:
  - operation kind;
  - compare-produced computed mask and mask use;
  - inactive/pass-through behavior;
  - segment tuple type/count;
  - field0 and field1 roles;
  - load-vs-store direction;
  - source and destination memory forms;
  - runtime ABI parameter order;
  - runtime `n`/AVL/VL relation;
  - mask/tail policy;
  - provider route-family plan;
  - route-control provider facts;
  - memory operand-binding facts;
  - target artifact validator consumption;
  - generated harness expectations.
- Unsupported or inconsistent selected bodies must fail closed with targeted
  diagnostics, including wrong field roles, missing mask facts, wrong ABI
  order, wrong memory form, stale `provider_supported_mirror`, missing
  segment2 family plan, direct route-entry-only authority, script-derived
  acceptance, exact-intrinsic-as-authority, or common-EmitC semantic choice.
- Common EmitC/export must remain neutral and may only materialize the
  provider-built `TCRVEmitCLowerableRoute`.
- Artifact metadata, route ids, generated file names, script expectations,
  C ABI strings, intrinsic spellings, descriptors, source-front-door markers,
  pre-realized fixtures alone, and legacy i32 terms are mirror/negative
  surfaces only; they must not authorize executable claims.
- Direct pre-realized route-entry probes for plain and computed-mask segment2
  must remain fail-closed.

## Acceptance Criteria

- [ ] Selected-boundary dry-runs pass for
  `computed_masked_segment2_load_unit_store` and
  `computed_masked_segment2_store_unit_load` with
  `route_entry_realization` false.
- [ ] Generated RVV C artifact/harness execution on `ssh rvv` passes for both
  load/store routes over counts including `0`, `1`, exact-VL, tail, and stress
  cases with varied masks and tuple field values.
- [ ] Runtime evidence checks active-lane behavior, inactive/pass-through
  preservation, ABI order, runtime `n`/AVL/VL relation, mask use, field roles,
  and tail/sentinel preservation.
- [ ] Target artifact tests prove the segment2 validator consumes
  provider-derived load/store facts and rejects stale/missing mirrors for
  operation kind, mask facts, memory direction/form, field roles, ABI order,
  provider mirror, route operand binding, header/type mapping, and runtime
  AVL/VL facts.
- [ ] Direct pre-realized route-entry probes remain fail-closed for segment2.
- [ ] Non-regression coverage passes for
  `computed_masked_segment2_update_unit_load` and plain segment2
  deinterleave/interleave selected-boundary paths.
- [ ] Bounded touched-file authority scan finds no central ad hoc,
  name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
  script-derived, artifact-name-derived, common-EmitC-derived,
  source-front-door-derived, route-id-derived, exact-intrinsic-derived,
  direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
  executable authority.
- [ ] `git diff --check` passes.
- [ ] Focused checks pass; `check-tianchenrv` passes or an exact blocker is
  recorded.
- [ ] Trellis task status, context, and notes are truthful; the task is
  finished/archived and one coherent commit is created if all acceptance
  criteria pass.

## Non-Goals

- Do not redo `computed_masked_segment2_update_unit_load` beyond
  non-regression.
- Do not redo plain segment2 deinterleave/interleave, widening dot, MAcc,
  standalone reduction, conversion, compare/select, or base memory movement.
- Do not add high-level frontend/Linalg work, dtype/LMUL clone batches,
  one-intrinsic wrapper dialects, broad validator rewrites, dashboards,
  reports, or broad smoke matrices.
- Do not resurrect direct pre-realized segment2 route-entry support.
- Do not move RVV segment2 semantics into common EmitC/export.

## Technical Notes

- Required specs read for this PRD:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/index.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant archived tasks read:
  - `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-segment2-memory-target-artifact-validator-boundary/`
  - `.trellis/tasks/archive/2026-05/05-29-05-29-stage2-rvv-computed-mask-segment2-update-runtime-abi-closure/`
- Key spec facts:
  - Computed-mask segment2 load/store/update are segment2 selected-body
    realization-owner families but selected-boundary-only.
  - Segment2 target artifact export must rebuild the provider route and
    validate metadata only as mirrors of provider facts.
  - Computed-mask segment2 families require the computed-mask memory family
    plan and mask source/role/memory-form mirrors, and must not require the
    plain segment2 family plan mirror.

## Definition of Done

- Both computed-mask segment2 load/store runtime ABI paths are executable from
  selected typed bodies through provider-derived target artifact validation and
  generated RVV C artifacts.
- Focused dry-runs, target tests, direct route-entry fail-closed checks,
  `ssh rvv` evidence, authority scan, and final repository checks are recorded.
- The task is finished/archived and committed when complete.
