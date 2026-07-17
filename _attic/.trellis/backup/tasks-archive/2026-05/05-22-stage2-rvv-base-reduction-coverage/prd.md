# Stage2 RVV closure-gated base reduction coverage expansion

## Goal

Normalize and complete one bounded Stage2 base reduction/accumulation slice
after the completed computed-mask standalone reduction task: ordinary
`standalone_reduce_add` plus directly related chunk-wise `reduce_add`.

This task must keep reduction authority in the selected typed `tcrv_rvv` body,
RVV plugin legality/realization/planning/provider code, and
`RVVRouteOperandBindingPlan` closure. It must not add new frontend lowering,
new reduction kinds, dtype/LMUL clone batches, descriptor/direct-C/source-front
door routes, or legacy `RVVI32M1` / `rvv-i32m1` / finite `tcrv_rvv.i32_*`
authority.

## Direction Source

- Direction title: `Stage2 RVV closure-gated base reduction coverage expansion`.
- Module owner: RVV plugin-owned route-supported expansion/normalization for
  base standalone reduction and reduce-add accumulation routes on the corrected
  `tcrv_rvv` surface.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `d57f5c6c rvv: add closure-gated computed-mask reduction`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Active Inventory

Inventory comes from current code, tests, and archived task evidence, not from
status words alone.

- `computed_mask_standalone_reduce_add` is complete from the archived task
  `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-closure-gated-reduction-accumulation/`.
  It remains only a regression anchor for this task.
- `standalone_reduce_add` already has:
  - `tcrv_rvv.typed_standalone_reduce_pre_realized_body`;
  - realized `setvl/with_vl/load/standalone_reduce/store` structure;
  - `RVVRouteOperandBindingPlan` ID
    `rvv-route-operand-binding:standalone_reduce_add.v1`;
  - generated-bundle pre-realized dry-run coverage.
- `standalone_reduce_add` does not yet have a dedicated explicit selected-body
  generated-bundle fixture/evidence path parallel to computed-mask standalone
  reduction.
- `reduce_add` already has:
  - explicit and pre-realized selected-body fixtures;
  - route ID `rvv-route-operand-binding:reduce_add.v1`;
  - route support through generic `tcrv_rvv.reduce`;
  - generated-bundle script support.
- `reduce_add` route operand binding summaries still use older abbreviated
  materialized-use labels such as `abi`, `load-base`, `store-base`, and
  `header`, while newer closure-gated routes use explicit mirror labels such
  as `runtime-abi-mirror`, `materialized-load-base`, `materialized-store-base`,
  and `header-mirror`.

## Requirements

1. Keep the bounded route family to base ordinary reduction:
   - chunk-wise `reduce_add`;
   - scalar-output `standalone_reduce_add`;
   - computed-mask standalone reduction only as a regression anchor.
2. Normalize `reduce_add` route operand binding summaries so source vector,
   accumulator vector, output, runtime `n`/AVL, materialized uses, and header
   mirrors are explicit and checked through `RVVRouteOperandBindingPlan`.
3. Keep provider materialization closure-gated: provider code must obtain
   materialized operands from the binding plan and fail closed on stale plan
   IDs, missing bindings, role mismatches, or materialized-use mismatches.
4. Add explicit selected-body evidence for `standalone_reduce_add`:
   - selected `tcrv.exec` RVV variant;
   - explicit `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, load,
     `tcrv_rvv.standalone_reduce`, scalar-output store;
   - structural reduction kind, source vector role, accumulator seed role,
     scalar output role, runtime `n`/AVL, dtype/config facts;
   - route/header mirrors derived from the binding plan.
5. Extend generated-bundle script/test coverage so representative explicit and
   pre-realized `standalone_reduce_add` runs distinguish source values,
   accumulator seed, runtime `n`/AVL, scalar result, and tail sentinel
   preservation.
6. Do not change `computed_mask_standalone_reduce_add` semantics; use it as a
   regression check when practical.
7. Unsupported or malformed cases must fail closed through existing or updated
   tests for unsupported reduction kind, missing source vector, missing
   accumulator/seed, missing scalar/result role, vector/scalar role swaps,
   dtype/config mismatch, stale route ID, materialized-use mismatch,
   descriptor/direct-C/source-front-door authority, and common/export semantic
   inference where current test surfaces can express those failures.

## Acceptance Criteria

- [x] PRD records current base reduction inventory and explains why this task
      normalizes `standalone_reduce_add` / `reduce_add` instead of redoing
      computed-mask standalone reduction.
- [x] Production RVV planning/provider code derives normalized `reduce_add`
      materialized operands and mirrors from `RVVRouteOperandBindingPlan`.
- [x] Positive structural tests prove explicit and pre-realized `reduce_add`
      carry normalized binding plan summaries and header mirrors.
- [x] A positive explicit selected-body `standalone_reduce_add` fixture proves
      typed body facts, route support, materialized operands, header mirrors,
      accumulator seed handling, scalar output layout, runtime `n`/AVL, and no
      route-ID/helper-string authority.
- [x] Generated-bundle dry-runs pass for explicit and pre-realized
      `standalone_reduce_add` at representative counts `7,16,23`.
- [x] Real `ssh rvv` generated-bundle evidence passes for representative
      explicit and pre-realized `standalone_reduce_add`, proving source sum,
      accumulator seed, runtime `n`/AVL, scalar output, and tail sentinel
      preservation.
- [x] Focused negative tests or existing verifier tests cover stale route ID,
      invalid reduction kind/layout/config/policy, role mismatches,
      materialized-use mismatch, descriptor/direct-C/source-front-door
      authority, and common/export semantic inference for touched surfaces.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, public exact intrinsic route authority, artifact-name
      authority, or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, task finish or
      truthful open status, clean git status, and one coherent commit complete
      if the task finishes.

## Non-Goals

- No new contractions/dot-product variants.
- No new compare/select predicates.
- No conversions, segmented/indexed movement, masked memory movement, or new
  memory movement routes.
- No new dtype/LMUL clone batches.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive RVV routes.
- No dashboards, report-only work, helper-only cleanup, or broad smoke
  matrices as the main achievement.
- No compatibility wrapper preserving legacy i32 route authority.

## Validation Plan

1. Start this Trellis task after PRD/context validation.
2. Run focused target FileCheck checks for explicit/pre-realized `reduce_add`
   and explicit/pre-realized `standalone_reduce_add`.
3. Run focused dialect verifier tests for standalone reduction negative cases.
4. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   script self-test if available.
5. Run generated-bundle dry-runs for explicit and pre-realized
   `standalone_reduce_add` at counts `7,16,23`.
6. Run real `ssh rvv` generated-bundle checks for explicit and pre-realized
   `standalone_reduce_add` at counts `7,16,23`.
7. Run focused active-authority scans over touched RVV include/lib/script/test
   paths.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Code Surface

- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/explicit-selected-body-artifact-reduce-add.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-reduce-add.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-standalone-reduce-add.mlir`
- `test/Dialect/RVV/standalone-reduction-dataflow.mlir`
- `test/Scripts/`

## Definition Of Done

- Base `reduce_add` binding mirrors are normalized and provider-checked through
  the closure gate.
- `standalone_reduce_add` has explicit and pre-realized selected-body
  generated-bundle evidence.
- Runtime/correctness claims are backed by real `ssh rvv` evidence.
- No legacy/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished/archived and committed when complete;
  otherwise it remains open with an exact continuation point.

## Completion Notes

- Normalized `reduce_add` binding summary from abbreviated
  `abi|load-base|store-base|header` labels to explicit
  `runtime-abi-mirror`, `materialized-load-base`,
  `materialized-accumulator-load-base`, `materialized-store-base`, and
  `header-mirror` labels.
- Rewired `reduce_add` provider checks to require those normalized
  materialized-use labels through `RVVRouteOperandBindingPlan`.
- Added explicit selected-body `standalone_reduce_add` target fixture and
  generated-bundle dry-run lit coverage.
- Added explicit `standalone_reduce_add` expectation to
  `scripts/rvv_generated_bundle_abi_e2e.py`; pre-realized standalone coverage
  now reuses the same expectation and only swaps the fixture/function.
- Focused checks passed:
  - `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
  - focused `tcrv-opt` / `tcrv-translate` / `FileCheck` for explicit and
    pre-realized `reduce_add`, explicit and pre-realized
    `standalone_reduce_add`
  - `build/bin/tcrv-opt test/Dialect/RVV/standalone-reduction-dataflow.mlir --split-input-file --verify-diagnostics`
  - `build/bin/tianchenrv-rvv-extension-plugin-test`
  - `build/bin/tianchenrv-target-artifact-export-test`
  - generated-bundle dry-runs for explicit/pre-realized
    `standalone_reduce_add` and explicit/pre-realized `reduce_add`
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2` with 284/284 tests
    passed.
- Real `ssh rvv` evidence passed:
  - explicit `standalone_reduce_add`, counts `7,16,23`, seeds `-11,17`,
    result `PASS op=standalone_reduce_add counts=7,16,23 seeds=-11,17`
  - pre-realized `standalone_reduce_add`, counts `7,16,23`, seeds `-11,17`,
    result `PASS op=standalone_reduce_add counts=7,16,23 seeds=-11,17`
- Diff-level active-authority scan found no new positive legacy/source
  authority additions. Existing exact intrinsic spellings in touched source are
  pre-existing provider-owned mapping data, not new route-authority additions
  in this task.
