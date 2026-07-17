# Stage2 RVV computed-masked widening-dot selected-body realization

## Goal

Close the bounded Stage 2 selected-body realization task for
`computed_masked_widening_dot_reduce_add`: a pre-realized selected
`tcrv.exec` RVV variant with compare operands, i16 dot sources, i32 scalar
seed/output, runtime `n`/VL, and policy facts must be consumed by the RVV
plugin-local contraction selected-body realization owner, materialized into a
typed mask-aware `tcrv_rvv` body, and then consumed by the existing
provider-derived route through target artifact and generated-bundle evidence.
The realization boundary must preserve compare-produced mask provenance,
inactive-lane zeroing facts, masked product/reduction operand order, scalar
seed/result layout, runtime ABI order, and provider-owned route authority.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-masked widening-dot selected-body realization`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `3d5ce564 rvv: close widening dot reduce selected-body realization`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- The archived plain widening-dot selected-body realization task closed
  `widening_dot_reduce_add`, including pre-realized materialization, stale
  route-id rejection, generated-bundle dry-run, and real `ssh rvv` evidence.
- Current repository evidence shows the computed-mask widening-dot production
  owner already exists in
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp` for
  `tcrv_rvv.typed_computed_mask_widening_dot_reduce_pre_realized_body`.
- Focused baseline before edits showed the pre-realized computed-mask target
  fixture already passes `REALIZED`, `PLAN`, and `HEADER` checks when run with
  `build/bin/tcrv-opt`, `build/bin/tcrv-translate`, and LLVM 20 `FileCheck`.
- Focused generated-bundle dry-run before edits already passed for the
  pre-realized computed-mask path with runtime counts `0,1,16,17,257`.
- The current dialect negative test already rejects stale `route_id` authority
  and missing/stale mask provenance for the typed computed-mask widening-dot
  pre-realized body.
- The remaining bounded gap is target-fixture evidence: the pre-realized
  target fixture should explicitly lock the provider-derived inactive-lane
  zeroing fact and fail closed when the pre-realized selected body carries
  stale route authority or stale mask provenance.

## Requirements

- Start from
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir`.
- Preserve
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir`
  as the explicit selected-body regression.
- Preserve the generated-bundle pre-realized dry-run path and runtime counts
  `0,1,16,17,257`.
- Keep the production route authority in the realized typed `tcrv_rvv` body and
  RVV provider facts, not in route ids, artifact names, script options,
  metadata mirrors, descriptors, direct pre-realized route-entry shortcuts, or
  common EmitC semantics.
- Ensure the pre-realized target fixture proves selected-body realization
  happens before emission planning/header export and that provider-derived
  computed-mask facts include compare predicate, mask role/source/form,
  inactive-lane zeroing, route operand binding, scalar seed/result layout, and
  runtime ABI order `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`.
- Add or strengthen a fail-closed target-level negative for stale route-id
  authority or stale/missing mask provenance on the pre-realized selected body.
- Run real `ssh rvv` generated-bundle execution for the pre-realized path if
  reachable, using counts `0,1,16,17,257`, and record the exact external
  blocker if unreachable.

## Acceptance Criteria

- [x] The pre-realized target fixture proves realization into typed
      compare-mask plus `tcrv_rvv.masked_widening_dot_reduce` structure before
      route planning.
- [x] The same fixture checks provider-derived inactive-lane zeroing for
      computed-mask widening dot-reduce.
- [x] A target-level negative proves stale route-id authority or stale/missing
      mask provenance fails closed before route/provider facts can authorize
      the path.
- [x] Existing explicit selected-body computed-mask target/header fixture
      remains passing.
- [x] Pre-realized computed-mask generated-bundle dry-run succeeds for counts
      `0,1,16,17,257`.
- [x] Real `ssh rvv` generated-bundle execution for the pre-realized
      computed-mask path passes if reachable, or the exact external blocker is
      recorded.
- [x] Provider/target facts remain provider-derived mirrors after realization:
      runtime ABI order, operand binding, contraction route-family plan, typed
      compute op, compare predicate, mask role/source/form, inactive-lane
      zeroing, masked product/reduction/store facts, header/type mapping,
      target leaf profile, and provider-supported mirror.
- [x] Bounded old-authority scan over touched files and added diff lines passes.
- [x] `git diff --check`, `git diff --cached --check`, and Trellis context
      validation pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/TableGen/CMake/lit/FileCheck.
- Python remains limited to generated-bundle support tooling; no script change
  is expected unless focused validation exposes a real runner blocker.
- No computed-masked strided-input batch, plain widening-dot rework,
  widening-MAcc follow-up, dtype/LMUL clone batch, high-level frontend,
  performance/autotuning, source-front-door positive route, direct
  pre-realized route-entry shortcut, descriptor-driven compute, or common
  EmitC semantic branch is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when the
  acceptance criteria are met or a genuine external execution blocker is
  proven.

## Technical Approach

Use the current production contraction selected-body realization owner as the
implementation surface. Do not rewrite the already-existing computed-mask
owner or common EmitC path unless focused evidence shows a production behavior
gap. Strengthen the pre-realized computed-mask target fixture so it proves the
provider-derived inactive-lane zeroing fact and fails closed when the
pre-realized selected body attempts to carry stale route-id authority or stale
mask provenance. Then run the focused materialization, emission plan, header,
explicit regression, generated-bundle dry-run, real `ssh rvv` execution if
reachable, and final diff/context checks.

## Decision (ADR-lite)

Context: The Hermes brief asks for computed-masked widening-dot selected-body
realization, but current repository evidence already contains a production RVV
contraction realization owner, provider facts, target artifact contract, and
generated-bundle pre-realized dry-run support for this operation.

Decision: Treat the live production owner as the behavior owner and close the
remaining bounded gap with target-level fail-closed/evidence guards rather than
duplicating or moving semantics into common EmitC/export.

Consequences: The change remains small and evidence-focused, but it is not
report-only: it tightens the executable target fixture around provider-derived
mask/zeroing authority and stale-authority rejection for the exact
pre-realized selected-boundary path.

## Out Of Scope

- Computed-mask strided-input widening-dot batch.
- Plain widening-dot realization rework.
- Widening-MAcc follow-up.
- New dtype/LMUL clone routes.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Performance/autotuning.
- Common EmitC invention of mask, dtype, reduction, or schedule semantics.
- Direct pre-realized route-entry shortcut bypassing typed-body realization.
- Source-front-door positive route.
- Broad smoke/report/status-only completion.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  and `.trellis/spec/testing/index.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-widening-dot-reduce-selected-body-realization/prd.md`.
- Fixtures read:
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir`,
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir`,
  and
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-widening-dot-reduce-add-dry-run.test`.
- Implementation files inspected:
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  and target artifact tests under `test/Target/TargetArtifactExportTest.cpp`.

## Completion Evidence

- Focused baseline before edits passed for the pre-realized computed-mask
  target fixture:
  `REALIZED`, `PLAN`, and `HEADER`.
- The source change strengthened
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir`
  only.
- The pre-realized fixture now checks provider-derived
  `tcrv_rvv.inactive_lane_zeroing_requirement =
  masked-widening-products-zero-inactive-lanes-before-reduction` in both
  emission-plan and target-header evidence.
- The same fixture now has target-level fail-closed RUNs for:
  stale `route_id = "rvv-i32m1"` authority and stale mask provenance
  `mask_source = "runtime_abi:mask"`.
- Focused pre-realized target fixture passed after edits for:
  `REALIZED`, `PLAN`, `HEADER`, `STALE-AUTH`, and
  `MISSING-MASK-PROVENANCE`.
- Existing explicit computed-mask target fixture passed for `PLAN` and
  `HEADER`.
- Pre-realized computed-mask generated-bundle dry-run passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-computed-masked-widening-dot-reduce-add-stage2-realization-dry-run`.
- Direct FileCheck of the pre-realized dry-run evidence passed for `ROOT`,
  `MDOT`, and `HARNESS`.
- Explicit computed-mask generated-bundle dry-run regression passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-computed-masked-widening-dot-reduce-add-stage2-realization-regression-dry-run`.
- Direct pre-realized route-entry fail-closed generated-bundle negative passed
  for `computed_masked_widening_dot_reduce_add`.
- Real `ssh rvv` generated-bundle execution passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-computed-masked-widening-dot-reduce-add-stage2-realization-ssh`.
- Generated bundle path:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-computed-masked-widening-dot-reduce-add-stage2-realization-ssh/computed_masked_widening_dot_reduce_add/generated_bundle`.
- Remote PASS marker:
  `PASS op=computed_masked_widening_dot_reduce_add counts=0,1,16,17,257`.
- Runtime evidence covered counts `0,1,16,17,257` and reported
  `compare_masked_signed_horizontal_dot`, `seed_added`,
  `inactive_lanes_skipped`, `scalar_output`, and `tail_preserved`.
- Existing dialect negative passed:
  `build/bin/tcrv-opt test/Dialect/RVV/computed-mask-widening-dot-reduction-negative.mlir --split-input-file --verify-diagnostics`.
- Focused C++ tests passed:
  `build/bin/tianchenrv-target-artifact-export-test` and
  `build/bin/tianchenrv-rvv-extension-plugin-test`.
- Bounded added-line old-authority scan found only the intentional negative
  `route_id = "rvv-i32m1"` RUN and `STALE-AUTH` FileCheck label; no positive
  executable route-authority growth was introduced.
- `git diff --check` passed.
- `git diff --cached --check` passed after staging.
- Trellis context validation passed with five implement entries and five check
  entries.

## Self-Repair Performed

- Initial `PLAN` check placement for
  `tcrv_rvv.inactive_lane_zeroing_requirement` was after the mask metadata,
  but the actual provider metadata emits it before `source_sew`; moved the
  FileCheck line to match the real provider order.
- Initial `HEADER` check placement put inactive-lane zeroing before the
  widening-dot relation scan point, but the header emits it after the
  contraction route-family/type-mapping metadata; moved the FileCheck line to
  match the actual header order.

## Spec Update Judgment

- No `.trellis/spec/` update was needed. This round did not introduce a new
  selected-body owner API, route contract, target artifact contract, generated
  bundle harness convention, or runtime evidence rule. It applied the existing
  RVV plugin-local selected-body realization owner registry, provider-owned
  computed-mask widening dot-reduce route validation contract, and RVV testing
  evidence requirements to the pre-realized computed-mask fixture.
