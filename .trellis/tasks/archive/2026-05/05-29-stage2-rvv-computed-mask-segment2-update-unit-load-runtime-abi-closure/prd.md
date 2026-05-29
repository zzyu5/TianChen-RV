# Stage2 RVV computed-mask segment2 update-unit-load runtime ABI closure

## Goal

Close the Hermes-requested `computed_masked_segment2_update_unit_load` selected-boundary/runtime-ABI-to-artifact round against current HEAD truth. The selected RVV path must be proven from `tcrv.exec` ABI/runtime roles through typed or pre-realized `tcrv_rvv` update body facts, RVV provider-derived route facts, neutral EmitC materialization, target artifact validation, generated RVV C harnesses, and `ssh rvv` correctness evidence.

Current inspection shows the production selected-body/provider/target path already contains the update route closure from the archived `05-29-05-29-stage2-rvv-computed-mask-segment2-update-runtime-abi-closure` task. This round is therefore not allowed to invent a new architecture or duplicate production wiring. The bounded implementation blocker is focused update-specific target validator evidence for stale facts that Hermes called out after the later load/store closure, especially update source/destination memory form and route operand binding mirrors.

## What I already know

- The repo began this round at `/home/kingdom/phdworks/TianchenRV`, branch `main`, clean worktree, HEAD `84a2b472 rvv: close computed-mask segment2 load store runtime abi`.
- `.trellis/.current-task` did not exist; this task was created from the Hermes Direction Brief.
- Archived task `05-29-05-29-stage2-rvv-computed-mask-segment2-update-runtime-abi-closure` already added production update arithmetic/provider facts, explicit and pre-realized target artifact fixtures, generated-bundle dry-runs, direct route-entry fail-closed checks, and `ssh rvv` correctness evidence.
- Archived task `05-29-05-29-stage2-rvv-computed-mask-segment2-load-store-runtime-abi-closure` subsequently closed adjacent load/store runtime ABI evidence and left update as a non-regression path.
- Current production code already carries `computed_masked_segment2_update_unit_load` route operand binding with `add-lhs` and `add-rhs`, provider mirror `provider_supported_mirror:rvv-computed-mask-segment2-update-add-plan-validated`, runtime ABI order `cmp_lhs,cmp_rhs,src0,src1,dst,n`, and target leaf profile `rvv-v1-e32m1-computed-mask-segment2-update-add-leaf-profile.v1`.
- Current target/export lit tests already reject stale update route id, provider mirror, route operand binding plan, ABI order, headers, type mapping, mask source, inactive lane contract, field role, arithmetic kind, computed-mask family plan, segment count, mask role, and stale non-family residue.
- Current C++ target artifact tests already reject update wrong operation, route-id-derived support, stale plain segment2 family facts, wrong mask binding, wrong tuple field role, wrong inactive/passthrough contract, wrong runtime ABI order, metadata-only provider mirror, wrong headers, wrong type mapping, wrong VL/AVL runtime relation, stale mask mirror, stale field mirror, stale update arithmetic mirror, stale ABI mirror, stale provider mirror, stale header mirror, stale type mirror, stale runtime plan mirror, and stale plain segment2 family mirror.

## Scope

- Production code may change only if inspection or checks prove update selected-body/provider/target facts are stale or incomplete:
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- Expected implementation target for this round:
  - `test/Target/TargetArtifactExportTest.cpp`, adding focused update-specific validator stale-fact negatives for source/destination memory form and route operand binding mirrors if production already validates them.
- Direct verification consumers:
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-update.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-update.mlir`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-*-computed-masked-segment2-update-*.test`

## Requirements

- The update route must remain selected-boundary-only; direct pre-realized route-entry must fail closed.
- `tcrv.exec` must remain only the execution/ABI envelope. RVV compute, mask/tail, segment2 update semantics, route construction, headers, intrinsic mapping, and ABI consumption must remain RVV-provider-owned.
- The update path must preserve and validate:
  - operation kind `computed_masked_segment2_update_unit_load`;
  - compare-produced computed mask inputs and same-VL mask source;
  - segment2 source field roles and interleaved destination role;
  - `add(field0, field1)` update binding for field0 and original field1 binding for field1;
  - source memory form `unit-stride-load`;
  - destination memory form `segment2-interleaved-unit-stride-store`;
  - runtime ABI order `cmp_lhs,cmp_rhs,src0,src1,dst,n`;
  - runtime `n`/AVL/VL loop relation;
  - mask/tail policy and inactive/tail preservation;
  - `provider_supported_mirror` and route operand binding mirror facts;
  - target artifact ABI/header/type mapping facts.
- Any new tests must fail closed on stale provider or artifact mirrors rather than accepting names, route ids, artifact names, ABI strings, script expectations, exact intrinsic spelling, descriptors, source-front-door state, common EmitC inference, direct-route-entry-only paths, pre-realized fixtures alone, or legacy i32-derived authority.

## Acceptance Criteria

- [ ] Current production selected-body/provider/target path for `computed_masked_segment2_update_unit_load` is inspected and either changed if stale, or explicitly recorded as already correct.
- [ ] Target artifact validator C++ tests include update-specific stale negatives for source memory form, destination memory form, and route operand binding mirror facts.
- [ ] Existing explicit selected-body update artifact lit coverage still validates provider-supported mirrors, route operand binding, ABI order, headers, type mapping, mask facts, field roles, inactive lane policy, and update arithmetic facts.
- [ ] Existing pre-realized selected-boundary update artifact lit coverage still validates realization into typed compare/load/binary/masked segment2 store structure and rejects stale mirrors.
- [ ] Generated-bundle dry-runs pass for explicit and pre-realized selected-boundary update over counts `0`, `1`, exact-VL, tail, and stress cases.
- [ ] Direct pre-realized route-entry dry-run remains fail-closed for update.
- [ ] `ssh rvv` explicit and pre-realized update runs pass over counts `0`, `1`, exact-VL, tail, and stress cases, proving active update values, inactive preservation, tail preservation, runtime ABI order, and source preservation.
- [ ] Load/store closure remains covered as non-regression.
- [ ] Bounded touched-file authority scan finds no central ad hoc, name-derived, metadata-derived, descriptor-derived, ABI-string-derived, script-derived, artifact-name-derived, common-EmitC-derived, source-front-door-derived, route-id-derived, exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived executable authority.
- [ ] `git diff --check` passes.
- [ ] Focused checks pass; `check-tianchenrv` passes or an exact blocker is recorded.
- [ ] Trellis task status, check notes, archive state, journal, and commit are truthful.

## Non-Goals

- Do not reopen plain segment2 deinterleave/interleave, computed-mask segment2 load/store implementation, widening dot, compare/select, reduction, conversion, dtype/LMUL clone batches, high-level Linalg/frontend lowering, dashboards, reports, or broad smoke matrices.
- Do not restore direct pre-realized segment2 route-entry support.
- Do not move RVV update semantics into common EmitC/export.
- Do not treat prompt edits, helper-only changes, broad smoke tests, or report text as the main achievement.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
- Archived tasks read:
  - `.trellis/tasks/archive/2026-05/05-29-05-29-stage2-rvv-computed-mask-segment2-update-runtime-abi-closure/`
  - `.trellis/tasks/archive/2026-05/05-29-05-29-stage2-rvv-computed-mask-segment2-load-store-runtime-abi-closure/`
- Current inspection points:
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - update explicit/pre-realized selected-body artifact fixtures and generated-bundle script tests

## Definition of Done

- The update-unit-load selected-boundary/runtime ABI closure is either production-fixed or verified current, with the missing focused validator evidence added.
- Focused dry-runs, target tests, direct route-entry fail-closed check, `ssh rvv` evidence, authority scan, and final repository checks are recorded.
- The task is finished/archived and one coherent commit is created if all acceptance criteria pass.
