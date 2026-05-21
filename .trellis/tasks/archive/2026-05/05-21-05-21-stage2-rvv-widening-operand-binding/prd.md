# Stage2 RVV widening operand binding adoption

## Goal

Adopt the RVV plugin-owned `RVVRouteOperandBindingPlan` contract for the
current active `widening_macc_add` widening accumulate route, so its narrow
i16 source operands, wide i32 accumulator/result operands, runtime `n`/AVL,
provider materialized operands, emission-plan mirrors, generated headers, and
generated-bundle expectations all come from one binding authority.

This is one bounded operand-binding hardening round. It does not add new RVV
operation coverage.

## Direction Source

- Direction title: `Stage2 RVV widening operand binding adoption`.
- Module owner: RVV plugin-owned `RouteOperandBindingPlan` adoption for the
  current active widening RVV routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `8a8e2baa rvv: adopt segmented movement operand binding plans`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.

## What I Already Know

- Recent completed operand-binding tasks converted ordinary arithmetic/select,
  indexed movement, and segmented movement routes to
  `RVVRouteOperandBindingPlan`.
- Current active widening-like routes split into three groups:
  - Widening accumulate subcluster: `widening_macc_add`.
  - Contraction dot-reduction family:
    `widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
    `computed_masked_widening_dot_reduce_add`, and
    `computed_masked_strided_input_widening_dot_reduce_add`.
  - Conversion family: `widen_i32_to_i64` and `widen_i16_to_i32`.
- The brief explicitly excludes contraction/conversion for this round, and
  allows one coherent documented subcluster if the full widening surface is too
  large. This PRD therefore converts `widening_macc_add` only and records the
  other active routes as intentionally deferred.
- `widening_macc_add` already has typed/pre-realized body coverage, route
  planning/provider support, generated-bundle dry-run support, and prior
  `ssh rvv` evidence from the widening macc executable-slice task.
- Today `widening_macc_add` is handled by the contraction route-family plan for
  target leaf/profile validation, but it does not yet carry
  `tcrv_rvv.route_operand_binding_plan` or
  `tcrv_rvv.route_operand_binding_operands`.
- Common EmitC/export must remain neutral and must not infer RVV widening
  source/result width, accumulator/result role, operation semantics, intrinsic
  choice, or route support from route IDs, artifact names, ABI strings,
  descriptors, source-front-door metadata, or mirrors.

## Scope

Convert this active route-supported/executable subcluster:

- `widening_macc_add`: `out_i32[i] = acc_i32[i] + i16(lhs[i]) * i16(rhs[i])`
  with source `SEW16 LMUL mf2`, accumulator/result `SEW32 LMUL m1`, unit-stride
  source/accumulator/output memory, runtime `n`, and signed widening macc
  relation.

For the converted route, record logical operand, typed body value or runtime
role, source/result width role, C parameter, materialized expression/use, and
mirror/header field in `RVVRouteOperandBindingPlan`, then make provider
emission consume the contract for actual emitted operands.

## Requirements

1. Inventory active widening-like routes and classify converted, deferred
   contraction, or deferred conversion.
2. Add a binding-plan ID and logical-operand role validation for
   `widening_macc_add`.
3. Bind `widening_macc_add` operands through one contract:
   - `lhs`: `lhs-input-buffer`, i16mf2 source load base, widening lhs compute
     operand, source-width mirror/header role.
   - `rhs`: `rhs-input-buffer`, i16mf2 source load base, widening rhs compute
     operand, source-width mirror/header role.
   - `acc`: `accumulator-input-buffer`, i32m1 accumulator load base, widening
     accumulator compute operand, accumulator-width mirror/header role.
   - `out`: `output-buffer`, i32m1 result store base, result-width
     mirror/header role.
   - `n`: `runtime-element-count`, setvl AVL, loop control, and header mirror.
4. Rewire provider emission so `widening_macc_add` materialized source loads,
   accumulator load, compute operands, store, setvl, and mirrors are checked
   through the binding contract.
5. Require route description mirrors and generated artifact/header metadata for
   converted `widening_macc_add` to carry the same binding plan ID and summary.
6. Add or update positive structural checks proving `widening_macc_add` carries
   binding plan IDs, binding summaries, header mirrors, and materialized
   operands from the contract.
7. Add or update negative fail-closed checks for expressible widening macc
   operand errors: narrow source role swaps, accumulator/output swaps, missing
   or duplicate ABI roles, missing materialized uses, mirror/header mismatch,
   stale route-id authority, descriptor/direct-C/source-front-door authority,
   and common/export semantic inference.
8. Keep dot-reduction contraction routes, conversion routes, segmented/indexed
   movement, new operation coverage, source-front-door positive routes, and
   dtype/LMUL clone batches out of scope.

## Acceptance Criteria

- [x] Current active widening-like route inventory is recorded in completion
      notes.
- [x] `widening_macc_add` derives `lhs/rhs/acc/out/n` materialized operands and
      route/header mirrors from `RVVRouteOperandBindingPlan`.
- [x] Provider emission fails closed when `widening_macc_add` bindings are
      missing, duplicated, role-swapped, materialized-use mismatched, source or
      result width mismatched, accumulator/result swapped, or inconsistent with
      runtime ABI mirrors.
- [x] Positive structural lit/FileCheck coverage proves the converted route
      carries binding plan IDs and summaries in emission-plan metadata and
      generated headers.
- [x] C++ or lit negative coverage checks role swaps, missing/duplicate roles,
      missing materialized uses, and mirror/materialized operand mismatch for
      the converted route.
- [x] Generated-bundle dry-runs pass for `widening_macc_add` with counts
      `7,16,23`, signed width-distinguishing values, and tail/sentinel cases.
- [x] Real `ssh rvv` PASS evidence exists for `widening_macc_add` when
      runtime/correctness is claimed.
- [x] Active-authority scan shows no new positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public exact
      intrinsic authority, stale route-id authority, artifact-name authority,
      or common/export RVV semantic authority.
- [x] Focused checks, `check-tianchenrv`, `git diff --check`, Trellis finish/
      archive, clean git status, and one coherent commit are completed if the
      task finishes.

## Non-Goals

- No conversion-route binding adoption for `widen_i32_to_i64` or
  `widen_i16_to_i32`.
- No contraction dot-reduction binding adoption for
  `widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, or
  `computed_masked_strided_input_widening_dot_reduce_add`.
- No new Stage2 operation families or operation coverage.
- No segmented/indexed/masked movement changes.
- No dtype/LMUL clone batches, unsigned variants, high-level
  Linalg/Vector/StableHLO/frontend lowering, or source-front-door positive
  routes.
- No dashboards, report-only work, helper-only cleanup, or broad smoke tests as
  the main achievement.
- No descriptor-driven compute, direct-C/source-export route restoration, or
  compatibility wrapper preserving old route authority.
- No movement of RVV operand semantics into common EmitC/export, target
  metadata, artifact names, route IDs, descriptors, ABI strings alone, test
  names, or exact intrinsic spelling.

## Validation Plan

1. Validate and start this Trellis task.
2. Run focused C++ coverage for route operand binding plan validation.
3. Run focused lit/FileCheck checks for `widening_macc_add` route plan mirrors,
   generated headers, target artifact metadata, and provider materialization.
4. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes.
5. Run generated-bundle dry-run for pre-realized `widening_macc_add` at counts
   `7,16,23`.
6. Run real `ssh rvv` evidence for `widening_macc_add` after dry-run passes.
7. Run active-authority scans over active RVV include/lib/script/test paths.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Code Surface

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/pre-realized-selected-body-artifact-widening-macc-add.mlir`
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-macc-add-dry-run.test`
- Relevant plugin C++ tests for `RVVRouteOperandBindingPlan`.

## Definition Of Done

- `widening_macc_add` consumes `RVVRouteOperandBindingPlan` for real provider
  materialization and mirrors.
- Deferred active widening-like routes are named with exact reasons.
- Focused checks, hardware evidence, active-authority scan, and
  `check-tianchenrv` are complete or any skip is justified by concrete
  environment failure.
- Trellis task status is truthful, archived when complete, and one coherent
  commit records the work.

## Completion Notes

### Active Widening-Like Route Inventory

- Converted in this round:
  - `widening_macc_add`: active route-supported/executable widening accumulate
    route with i16mf2 source operands, i32m1 accumulator/result operands,
    runtime `n`, generated-bundle support, and real `ssh rvv` evidence.
- Intentionally not converted in this round:
  - `widening_dot_reduce_add`
  - `strided_input_widening_dot_reduce_add`
  - `computed_masked_widening_dot_reduce_add`
  - `computed_masked_strided_input_widening_dot_reduce_add`

  These are contraction/dot-reduction family routes. The Direction Brief
  explicitly excluded contraction for this bounded owner.
- Intentionally not converted in this round:
  - `widen_i32_to_i64`
  - `widen_i16_to_i32`

  These are conversion family routes. The Direction Brief explicitly excluded
  conversion for this bounded owner.

### Module Behavior Completed

- Added `rvv-route-operand-binding:widening_macc_add.v1` as the
  `widening_macc_add` binding authority.
- Required logical operands:
  - `lhs`: `lhs-input-buffer`, C ABI `lhs`, source load, widening-macc lhs,
    source-width `i16mf2`, header mirror.
  - `rhs`: `rhs-input-buffer`, C ABI `rhs`, source load, widening-macc rhs,
    source-width `i16mf2`, header mirror.
  - `acc`: `accumulator-input-buffer`, C ABI `acc`, accumulator load,
    widening-macc accumulator, accumulator-width `i32m1`, header mirror.
  - `out`: `output-buffer`, C ABI `out`, result store, result-width `i32m1`,
    header mirror.
  - `n`: `runtime-element-count`, C ABI `n`, setvl AVL, loop control, header
    mirror.
- Rewired the provider so `widening_macc_add` binds actual materialized source
  loads, accumulator load, result store, compute roles, width mirrors, and
  header mirrors through `RVVRouteOperandBindingPlan`.
- Required emission-plan metadata, target header metadata, and
  generated-bundle evidence to carry the plan ID and compact operand binding
  summary.
- Added C++ fail-closed coverage for accumulator/output role swaps and
  source/result-width materialized-use mismatch on the converted route.

### Self-Repair

- The first focused lit pass failed because the initial full-length
  `route_operand_binding_operands` mirror exceeded the existing
  `artifact metadata value must be bounded single-line text` coherence limit.
- Fixed by compacting widening-macc materialized-use mirror tokens
  (`src-load`, `wmacc-lhs`, `src-i16mf2`, `acc-load`, `acc-i32m1`,
  `res-store`, `res-i32m1`, `hdr`) while keeping provider-side required-use
  checks intact.

### Validation Evidence

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- Focused lit passed:
  - `Target/RVV/pre-realized-selected-body-artifact-widening-macc-add.mlir`
  - `Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-macc-add-dry-run.test`
- Explicit generated-bundle dry-run passed:
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_widening_macc_operand_binding --run-id widening-macc-binding-dry-run --overwrite --op-kind widening_macc_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- Real RVV hardware evidence passed:
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_widening_macc_operand_binding --run-id widening-macc-binding-rvv --overwrite --op-kind widening_macc_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 120`
  - Result: `PASS op=widening_macc_add counts=7,16,23`.
  - Per-count evidence reported signed product accumulation and tail
    preservation for `n=7`, `n=16`, and `n=23`.
- Diff-added active-authority scan found no newly introduced positive
  `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor/direct-C/source-front-door, or public exact
  intrinsic authority.
- Full changed-file scan still contains pre-existing fail-closed negative-test
  and route-leaf mirror strings, but no new diff-added positive authority.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed: 267/267.
