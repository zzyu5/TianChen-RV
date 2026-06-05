# Stage2 RVV computed-masked strided-input widening-dot executable selected-body closure

## Goal

Close the bounded Stage 2 executable selected-body path for
`computed_masked_strided_input_widening_dot_reduce_add`: a pre-realized
selected `tcrv.exec` RVV variant with compare operands, i16 strided dot
sources, runtime `lhs_stride`/`rhs_stride`, i32 scalar seed/output, runtime
`n`/VL, mask/policy facts, and selected RVV capability must be consumed by the
RVV plugin-local contraction realization owner, materialized into typed
`tcrv_rvv` compare, `strided_load`, `masked_widening_dot_reduce`, and store
structure, then consumed by provider-derived route facts through
`TCRVEmitCLowerableRoute`, target bundle generation, and real `ssh rvv`
evidence when reachable.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-masked strided-input widening-dot executable selected-body closure`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `ed144ced rvv: close computed masked widening dot realization`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- The previous archived task closed the non-strided
  `computed_masked_widening_dot_reduce_add` selected-body path with target
  guards, generated-bundle dry-run, and real `ssh rvv` evidence.
- The current pre-realized strided fixture already names the required ABI order:
  `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride`.
- The current pre-realized strided target fixture expects realization to remove
  `tcrv_rvv.typed_computed_mask_strided_input_widening_dot_reduce_pre_realized_body`
  and produce typed `setvl`, compare loads, i16mf2 `strided_load`s,
  compare-produced mask, `tcrv_rvv.masked_widening_dot_reduce`, and scalar
  result store.
- The current dry-run script has an op expectation and harness path for
  `computed_masked_strided_input_widening_dot_reduce_add`, including runtime
  counts `0,1,16,17,257`, stride pairs `2:3` and `3:2`, mask/input pattern
  coverage, inactive-lane skipping, skipped source element checks, scalar
  output, seed-added behavior, and tail preservation.

## Requirements

- Start from
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`.
- Preserve
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`
  as the explicit selected-body regression.
- Prove realization happens before route planning and target/header export for
  the pre-realized selected-body path.
- Preserve route authority in realized typed `tcrv_rvv` structure and RVV
  provider facts: compare predicate, strided source memory form, runtime stride
  ABI binding, compare-produced mask provenance, inactive-lane zeroing,
  masked widening dot relation, scalar seed/result layout, header/type mapping,
  and operand binding.
- Generated-bundle dry-run must show `route_entry_realization=false`,
  `pre_realized_body_consumed=true`, provider-derived target facts, and the
  strided/mask/reduction ABI bridge.
- Run real `ssh rvv` generated-bundle compile/run for the pre-realized path if
  reachable; if unreachable, record the exact external blocker.
- If execution fails because the harness, runtime stride coverage, preservation
  checks, or ABI bridge is incomplete, repair only the minimal production/script
  blocker for this route while keeping RVV plugin ownership of semantics.
- Add or strengthen a focused fail-closed guard only if existing coverage does
  not already reject stale route-id, stale/missing mask provenance, stale stride
  facts, or stale target/provider mirrors for this exact path.

## Acceptance Criteria

- [x] Pre-realized target fixture proves realization into typed compare,
      `strided_load`, `masked_widening_dot_reduce`, and store before route
      planning.
- [x] Pre-realized target fixture or target export negative coverage rejects
      stale route-id, stale/missing mask provenance, or stale/missing runtime
      stride facts if existing coverage is incomplete.
- [x] Generated-bundle dry-run for the pre-realized path succeeds for counts
      `0,1,16,17,257`, carries `route_entry_realization=false`, and records
      `pre_realized_body_consumed=true`.
- [x] Explicit strided computed-mask selected-body target and generated-bundle
      dry-run regressions remain passing when shared code changes.
- [x] Real `ssh rvv` generated-bundle execution passes for the pre-realized
      path if reachable, covering counts `0,1,16,17,257`, stride pairs `2:3`
      and `3:2`, mask/input patterns, inactive-lane skipping, skipped source
      elements, seed-added behavior, add/mul distinguishing, scalar `out[0]`,
      source/accumulator preservation, and tail preservation.
- [x] Provider/target facts are derived from the realized typed body and mirror
      runtime ABI order, operand binding, contraction route-family plan, typed
      compute op, compare predicate, mask role/source/form, strided input facts,
      inactive-lane zeroing, masked product/reduction/store facts, header/type
      mapping, target leaf profile, and provider-supported mirror.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no positive legacy `i32m1`, descriptor, source-front-door, direct-C, or
      route-id authority drift.
- [x] Focused checks, `git diff --check`, `git diff --cached --check`, and
      Trellis context validation pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/TableGen/CMake/lit/FileCheck.
- Python remains limited to generated-bundle support tooling; script changes are
  allowed only for the focused executable harness/ABI validation blocker.
- No broad widening-dot matrix, plain/non-strided rework, dtype/LMUL clone
  batch, new reduction family, high-level frontend, per-Linalg route authority,
  performance/autotuning, common EmitC semantic invention, direct pre-realized
  route-entry shortcut, source-front-door positive route, descriptor-driven
  compute, or report/status-only closure is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when
  acceptance criteria are met or a genuine external execution blocker is
  proven.

## Technical Approach

First run focused baseline evidence for the existing pre-realized strided
fixture, explicit regression, generated-bundle dry-run, and real generated
bundle execution. If production realization/provider/target behavior is already
present, close by hardware evidence plus any minimal missing guard needed to
prevent stale metadata, route-id, mask, or stride authority. If a real failure
appears, fix the smallest production or script blocker on the strided
computed-mask widening-dot route while preserving the selected-boundary
realization chain:

```text
pre-realized selected tcrv.exec RVV variant
  -> RVV contraction selected-body realization owner
  -> realized typed tcrv_rvv compare + strided_load + masked reduction body
  -> provider-derived route/control/operand facts
  -> TCRVEmitCLowerableRoute
  -> generated bundle
  -> ssh rvv evidence
```

## Decision (ADR-lite)

Context: The current repository already has strong fixtures and script support
for this operation, but the Hermes brief requires executable closure, not a
report-only pass over existing text.

Decision: Treat live focused evidence as the authority. Implement only if the
focused evidence exposes a concrete blocker or missing fail-closed guard.
Otherwise, strengthen the bounded guard/evidence surface and close with real
`ssh rvv` execution for the pre-realized strided path.

Consequences: The work stays scoped to the strided computed-mask selected-body
boundary and does not move RVV semantics into common EmitC/export or route
metadata.

## Out Of Scope

- Broad widening-dot matrix.
- Plain or non-strided rework.
- Dtype/LMUL clone batches.
- New reduction family.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Performance/autotuning.
- Common EmitC invention of stride, mask, or reduction semantics.
- Direct pre-realized route-entry shortcut bypassing typed-body realization.
- Source-front-door positive route.
- Report/status-only completion.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  and `.trellis/spec/testing/index.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-computed-masked-widening-dot-selected-body-realization/prd.md`.
- Fixtures read:
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`,
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test`,
  and
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test`.
- Script inspected:
  `scripts/rvv_generated_bundle_abi_e2e.py`.

## Completion Evidence

- The source change strengthened only
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`.
- Added pre-realized selected-boundary target guards for:
  stale `route_id = "rvv-i32m1"` authority on the pre-realized body,
  stale `mask_source = "runtime_abi:mask"` provenance, and stale target
  artifact `tcrv_rvv.lhs_stride_source = metadata-derived-stride` mirror.
- Pre-realized target fixture passed after edits for:
  `REALIZED`, `PLAN`, `HEADER`, `STALE-AUTH`,
  `MISSING-MASK-PROVENANCE`, and `STALE-STRIDE-SOURCE`.
- Existing dialect negative passed:
  `build/bin/tcrv-opt test/Dialect/RVV/computed-mask-strided-input-widening-dot-reduction-negative.mlir --split-input-file --verify-diagnostics`.
- Explicit selected-body target fixture passed for `PLAN` and `HEADER`.
- Pre-realized generated-bundle dry-run passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-computed-masked-strided-input-widening-dot-reduce-add-stage2-closure-dry-run`.
- Direct FileCheck of the pre-realized dry-run evidence passed for `ROOT`,
  `MSDOT`, and `HARNESS`.
- Explicit generated-bundle dry-run regression passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-computed-masked-strided-input-widening-dot-reduce-add-stage2-closure-regression-dry-run`.
- Direct FileCheck of the explicit dry-run evidence passed for `ROOT`,
  `MSDOT`, and `HARNESS`.
- Real `ssh rvv` generated-bundle compile/run passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-computed-masked-strided-input-widening-dot-reduce-add-stage2-closure-ssh`.
- Generated bundle path:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-computed-masked-strided-input-widening-dot-reduce-add-stage2-closure-ssh/computed_masked_strided_input_widening_dot_reduce_add/generated_bundle`.
- Remote compile evidence: `ssh rvv` on `riscv64` used
  `/usr/bin/clang`, `Ubuntu clang version 18.1.3`, with
  `-march=rv64gcv -mabi=lp64d`.
- Remote PASS marker:
  `PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,1,16,17,257 stride_pairs=2:3,3:2 mask_patterns=2 input_patterns=2`.
- Runtime output covered counts `0,1,16,17,257`, stride pairs `2:3` and
  `3:2`, both mask/input patterns, `seed_added`,
  `inactive_lanes_skipped`, `skipped_source_elements_ignored`,
  `scalar_output`, and `tail_preserved`.
- Evidence JSON recorded `ssh_evidence = true`, `status = success`,
  `input_mode = pre-realized-selected-body`, `pre_realized_selected_body = true`,
  provider-derived `runtime_abi_order`,
  `route_operand_binding_operands`, `strided_input_facts`,
  `mask_source = compare-produced-mask-same-vl-scope`,
  `inactive_lane_zeroing_requirement`,
  `masked_widening_product_intrinsic`, `strided_load_intrinsic`, and generated
  header/object paths.
- Focused C++ tests passed:
  `build/bin/tianchenrv-target-artifact-export-test` and
  `build/bin/tianchenrv-rvv-extension-plugin-test`.
- Bounded added-line old-authority scan found only intentional negative guard
  strings for `route_id = "rvv-i32m1"`, `runtime_abi:mask`, and
  `metadata-derived-stride`; no positive executable authority growth was
  introduced.
- `git diff --check` passed.
- Trellis context validation passed with eight implement entries and eight
  check entries.

## Self-Repair Performed

- Existing production realization/provider/target/script paths already
  produced correct executable evidence, so no compiler or script behavior
  change was needed.
- The remaining non-report-only gap was pre-realized target-level guard
  coverage. Added the three focused fail-closed RUNs listed above and verified
  their real diagnostics manually because the local environment has no
  standalone `not` helper.

## Spec Update Judgment

- No `.trellis/spec/` update was needed. This round did not introduce a new
  selected-body owner API, route contract, target artifact contract, generated
  bundle convention, or runtime evidence rule. It applied the existing RVV
  plugin-local selected-body realization, direct contraction provider-plan,
  provider-owned operand binding, target mirror validation, and `ssh rvv`
  evidence requirements to the pre-realized strided computed-mask widening-dot
  executable boundary.
