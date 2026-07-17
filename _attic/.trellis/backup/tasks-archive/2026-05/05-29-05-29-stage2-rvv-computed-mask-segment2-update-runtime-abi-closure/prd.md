# Stage2 RVV computed-mask segment2 update runtime ABI closure

## Goal

Close the `computed_masked_segment2_update_unit_load` runtime ABI path from a selected `tcrv.exec` RVV variant through provider-derived RVV route facts, target-owned artifact route-family validation, neutral EmitC/export materialization, generated RVV C artifact output, and real `ssh rvv` correctness evidence.

This task is a bounded Stage2 RVV runtime ABI closure. It must move production behavior in the target-owned RVV route-family validator and only the directly necessary generated-bundle artifact plumbing. Tests and evidence alone are not sufficient.

## What I already know

- The current repo has no active Trellis task; this task is created from the Hermes Direction Brief.
- HEAD is `eb0b62bf rvv: close computed-mask widening dot runtime abi`, and the worktree was clean before task creation.
- The previous completed task closed computed-mask widening dot runtime ABI evidence and archived cleanly.
- The adjacent open gap is `computed_masked_segment2_update_unit_load`.
- Segment2 update already has selected/pre-realized coverage and direct route-entry fail-closed behavior, but explicit selected artifact coverage exists for computed-mask segment2 load/store and not update.
- Direct pre-realized route-entry support must remain globally fail-closed and must not become a positive executable shortcut.
- `tcrv.exec` binds ABI/runtime roles; `tcrv_rvv` selected bodies and the RVV plugin own RVV compute, memory movement, mask/tail, and route construction facts.

## Scope

- Production owner: `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Directly necessary generated-bundle owner: `scripts/rvv_generated_bundle_abi_e2e.py`.
- Directly necessary tests/evidence around target artifact export and script-generated RVV bundles.
- Explicit selected-body and pre-realized selected-boundary paths for `computed_masked_segment2_update_unit_load` if both are supported by current producer/provider facts.
- Load/store and computed-mask widening-dot may be touched only for shared factoring or non-regression coverage.

## Requirements

- Target-owned artifact validation must consume provider-derived facts for:
  - operation kind;
  - computed-mask producer and mask use;
  - mask/tail policy and inactive-lane behavior;
  - segment2 tuple field roles;
  - update/pass-through semantics;
  - source and destination memory roles;
  - runtime ABI order;
  - route operand binding plan and summary;
  - `provider_supported_mirror`;
  - required headers and type mapping;
  - VL/AVL/runtime `n` relation;
  - stale non-family rejection.
- Common EmitC/export mechanics must remain neutral and must not choose RVV semantics.
- Generated-bundle ABI evidence must cover the update path without relying on route ids, artifact names, script expectation names, metadata, descriptors, exact intrinsic spellings, common EmitC branches, source-front-door state, direct route-entry support, pre-realized fixtures alone, or legacy i32-derived authority.
- Direct pre-realized route-entry probes must remain fail-closed.

## Acceptance Criteria

- [ ] `computed_masked_segment2_update_unit_load` explicit selected-body generated artifact path validates through target-owned provider-derived route-family facts.
- [ ] Pre-realized selected-boundary generated artifact path is represented if current producer/provider support exists; otherwise the blocker is precise and documented.
- [ ] Direct pre-realized route-entry remains uniformly fail-closed.
- [ ] Target tests reject stale route family, wrong operation, wrong mask binding, wrong tuple field roles, wrong update/pass-through role, wrong runtime ABI order, wrong provider mirror, wrong header/type mapping, wrong VL/AVL/runtime relation, and metadata/route-id/script-derived authority.
- [ ] Generated-bundle correctness runs on `ssh rvv` for counts including 0, 1, exact, tail, and stress cases with mask patterns and segment field update expectations.
- [ ] Non-regression coverage remains for computed-mask segment2 load/store and completed computed-mask widening dot paths.
- [ ] Bounded touched-file authority scan finds no central ad hoc, name-derived, metadata-derived, descriptor-derived, ABI-string-derived, script-derived, artifact-name-derived, common-EmitC-derived, source-front-door-derived, route-id-derived, exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived executable authority.
- [ ] `git diff --check` passes.
- [ ] Focused tests pass; `check-tianchenrv` passes or an exact blocker is recorded.
- [ ] Final git status is clean after a coherent commit, or the task remains open with the exact next continuation point.

## Non-Goals

- Do not reopen computed-mask widening dot, widening MAcc/dot, standalone reduction, base memory movement, compare/select, conversion, high-level frontend, or Linalg work.
- Do not restore direct pre-realized route-entry support.
- Do not add route-id, artifact-name, script-name, metadata, descriptor, exact-intrinsic, common-EmitC, source-front-door, or legacy i32 authority.
- Do not turn this into a broad segment2 audit, dashboard/report task, smoke matrix, or evidence-only PRD.
- Do not treat helper-only, guardrail-only, prompt-only, or report-only changes as completion.

## Technical Notes

- Read first:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - archived task `05-29-stage2-rvv-computed-mask-widening-dot-runtime-abi-closure`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Target/TargetArtifactExportTest.cpp`
  - relevant `test/Target/RVV` and `test/Scripts` computed-mask segment2 files
  - RVV selected-body realization and route provider files only as needed to verify producer/provider facts.

## Definition of Done

- Production route-family validation and necessary generated-bundle plumbing are updated.
- Focused tests and real RVV runtime evidence are captured.
- Trellis task status and journal are updated truthfully.
- Task is finished/archived if all acceptance criteria pass.
- One coherent commit records the completed round.
