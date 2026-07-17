# Stage2 RVV conversion operand binding adoption

## Goal

Adopt the RVV plugin-owned `RVVRouteOperandBindingPlan` contract for the
current active widening conversion route cluster, so conversion source/result
ABI roles, typed body values, source/result element width/config facts,
conversion direction, runtime `n`/AVL, materialized EmitC operands, generated
headers, and route mirrors come from one binding authority.

This is one bounded operand-origin hardening round for existing conversion
routes. It does not add new RVV conversion coverage.

## Direction Source

- Direction title: `Stage2 RVV conversion operand binding adoption`.
- Module owner: RVV plugin-owned `RouteOperandBindingPlan` adoption for the
  current active RVV conversion routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `adb2bffd rvv: adopt contraction dot operand binding plans`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.

## Current Repository Facts

- The prior completed task converted the active contraction dot-reduction
  cluster to `RVVRouteOperandBindingPlan`; it intentionally left widening
  conversion as the separate remaining owner.
- Current route planning/provider code already carries binding plans for
  arithmetic/select, reduction, macc, widening macc, dot-reduction,
  movement/layout, segmented movement, masked store, and computed-mask strided
  store clusters.
- Current conversion operation inventory found two active route-supported and
  executable conversion routes:
  - `widen_i32_to_i64`: source `i32/m1`, result `i64/m2`, signed widening
    conversion, runtime ABI order `lhs,out,n`.
  - `widen_i16_to_i32`: source `i16/mf2`, result `i32/m1`, signed
    `sign_extend_widen_vf2`, runtime ABI order `lhs,out,n`.
- The active production conversion path uses:
  - `tcrv_rvv.typed_widening_conversion_pre_realized_body`;
  - `tcrv_rvv.runtime_abi_value` for `lhs`, `out`, and `n`;
  - selected-body realization into `setvl/with_vl/load/widening_convert/store`;
  - RVV route planning/provider and generated-bundle support;
  - target artifact tests for both conversion routes.
- Conversion route descriptions currently carry typed source/result facts and
  conversion relation mirrors, but no conversion-specific
  `routeOperandBindingPlanID` / summary.
- Provider emission currently materializes conversion load, compute, store, and
  runtime control through slice ABI fields; this task must require those uses
  through a conversion binding contract.

## Scope

Convert both active conversion routes:

- `widen_i32_to_i64`
  - logical source operand `lhs`;
  - logical destination operand `out`;
  - runtime element count operand `n`;
  - source element/config facts: i32, SEW32, LMUL m1;
  - result element/config facts: i64, SEW64, LMUL m2;
  - conversion direction: `signed-i32m1-to-i64m2` / `widen_i32_to_i64`.
- `widen_i16_to_i32`
  - logical source operand `lhs`;
  - logical destination operand `out`;
  - runtime element count operand `n`;
  - source element/config facts: i16, SEW16, LMUL mf2;
  - result element/config facts: i32, SEW32, LMUL m1;
  - conversion direction: `signed-i16mf2-to-i32m1` /
    `sign_extend_widen_vf2`.

For each route, the binding plan must record logical operand, runtime ABI
parameter, materialized load/convert/store/control use, source/result
width/config mirror use, conversion-direction mirror use, and header mirror
use.

## Requirements

1. Add conversion-specific binding plan IDs in RVV route planning/provider
   code, not in common EmitC/export.
2. Bind `widen_i32_to_i64` through one contract:
   - `lhs`: `lhs-input-buffer`, materialized source load base, conversion
     source operand, source i32/m1 mirror, source header mirror.
   - `out`: `output-buffer`, materialized destination store base, conversion
     result store, result i64/m2 mirror, destination header mirror.
   - `n`: `runtime-element-count`, setvl AVL, loop/control, runtime header
     mirror.
3. Bind `widen_i16_to_i32` through the same logical contract while recording
   the i16/mf2 source and i32/m1 result/config facts.
4. Provider emission must use `RVVRouteOperandBindingPlan` lookups before
   materializing conversion source load, conversion compute source, destination
   store, runtime loop control, source/result config mirrors, conversion
   relation mirror, and generated header mirrors.
5. Route description verification must require the expected conversion binding
   plan ID and non-empty summary for both converted routes.
6. Fail closed on missing roles, duplicate roles, source/destination swaps,
   source/result width/config mismatch, conversion-direction mismatch, stale
   route-id/helper-string authority, materialized operand mismatch,
   mirror/header mismatch, descriptor/direct-C/source-front-door authority, or
   common/export RVV semantic inference.
7. Keep conversion coverage unchanged; do not add new conversion variants,
   narrowing/unsigned/floating conversion routes, dtype/LMUL clone batches, or
   frontend lowering.

## Acceptance Criteria

- [x] Conversion inventory is recorded in completion notes with exact
      active/inactive/deferred status.
- [x] `widen_i32_to_i64` derives source load, conversion source, destination
      store, runtime `n`, source/result config mirrors, conversion relation,
      and header mirrors from `RVVRouteOperandBindingPlan`.
- [x] `widen_i16_to_i32` derives source load, conversion source, destination
      store, runtime `n`, source/result config mirrors, conversion relation,
      and header mirrors from `RVVRouteOperandBindingPlan`.
- [x] Provider emission fails closed when converted conversion bindings are
      missing, duplicated, role-swapped, materialized-use mismatched,
      source/result-width mismatched, conversion-direction mismatched, or
      inconsistent with runtime ABI mirrors.
- [x] Positive structural FileCheck coverage proves both conversion routes
      carry binding plan IDs and summaries in emission-plan metadata and
      generated headers.
- [x] C++ or lit negative coverage checks source/destination role swaps,
      missing/duplicate roles, missing materialized uses, conversion mismatch,
      and mirror/materialized operand mismatch for the converted cluster.
- [x] Generated-bundle dry-runs pass for `widen_i32_to_i64` and
      `widen_i16_to_i32` at counts `7,16,23`, with width-distinguishing,
      sign/value-distinguishing, and tail/sentinel cases covered by existing
      script expectations.
- [x] Real `ssh rvv` PASS evidence exists for representative converted
      conversion routes when runtime/correctness is claimed.
- [x] Active-authority scan shows no new positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public exact
      intrinsic authority, stale route-id authority, artifact-name authority,
      or common/export RVV semantic authority.
- [x] Focused checks, `check-tianchenrv`, `git diff --check`, Trellis finish/
      archive, clean git status, and one coherent commit are completed if the
      task finishes.

## Non-Goals

- No new Stage2 operation families or new conversion variants.
- No contraction, movement, arithmetic/select, reduction, macc, segmented, or
  indexed cluster changes except shared helper adjustments required by the
  conversion binding plan.
- No high-level Linalg/Vector/StableHLO/frontend lowering.
- No source-front-door positive routes, descriptor/direct-C/source-export
  restoration, route-id authority, helper-string authority, or artifact-name
  authority.
- No dashboards, report-only work, helper-only cleanup, broad smoke tests, or
  prompt/spec-only work as the main achievement.
- No Python compiler-core implementation. Python changes, if any, are limited
  to generated-bundle tooling/evidence checks.

## Validation Plan

1. Validate and start this Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-extension-plugin-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused C++ route operand binding plan validation.
4. Run focused lit/FileCheck checks for conversion route plan mirrors,
   generated headers, target artifact metadata, provider materialization, and
   fail-closed diagnostics.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
6. Run generated-bundle dry-runs for `widen_i32_to_i64` and
   `widen_i16_to_i32` at counts `7,16,23`.
7. Run real `ssh rvv` evidence for representative converted conversion routes
   after dry-runs pass.
8. Run active-authority scans over active RVV include/lib/script/test paths.
9. Run `git diff --check`.
10. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Code Surface

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing conversion fixtures/tests for:
  - `widen_i32_to_i64`
  - `widen_i16_to_i32`

## Definition Of Done

- Both active conversion routes consume `RVVRouteOperandBindingPlan` for real
  provider materialization and mirrors, or any inactive route is documented
  with exact evidence.
- Focused positive/negative checks, generated-bundle evidence,
  representative hardware evidence, active-authority scan, and
  `check-tianchenrv` are complete or any skip is justified by concrete
  environment failure.

## Completion Evidence

### Conversion Inventory

- Converted active routes:
  - `widen_i32_to_i64`: active route-supported and executable conversion route,
    source `i32/m1`, result `i64/m2`, signed widening relation
    `signed-i32m1-to-i64m2`, ABI order `lhs,out,n`.
  - `widen_i16_to_i32`: active route-supported and executable conversion route,
    source `i16/mf2`, result `i32/m1`, signed widening relation
    `signed-i16mf2-to-i32m1`, ABI order `lhs,out,n`.
- Intentionally not converted:
  - No additional active conversion route was found in the current production
    conversion cluster. This task did not invent positive route support for
    parser-only, inactive, narrowing, unsigned, floating, frontend, or clone
    variants.

### Implementation Evidence

- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - Added conversion binding plan IDs
    `rvv-route-operand-binding:widen_i32_to_i64.v1` and
    `rvv-route-operand-binding:widen_i16_to_i32.v1`.
  - Added expected roles for `lhs`, `out`, and `n`.
  - Added summaries that bind ABI operands, materialized load/convert/store
    uses, source/result config mirrors, conversion relation mirrors, runtime
    loop/AVL use, and header mirrors.
  - Route description verification now requires the expected conversion plan ID.
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - Conversion emission now consumes the binding plan for source load,
    conversion source, destination store, runtime loop/control, source/result
    config mirrors, conversion relation mirrors, and header mirrors before
    building materialized artifacts.
- `test/Plugin/RVVExtensionPluginTest.cpp`
  - Added positive structural checks for both converted conversion plans.
  - Added fail-closed negative checks for destination/source role swap and
    conversion-direction/materialized-use mismatch.
- Target artifact fixtures and generated-bundle script expectations now require
  the route binding plan and operand summary mirrors for both conversion routes.

### Checks

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] `build/bin/tianchenrv-target-artifact-export-test`.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Focused FileCheck for `widen_i32_to_i64` and `widen_i16_to_i32` REALIZED,
  PLAN, and HEADER outputs.
- [OK] `build/bin/tcrv-opt test/Dialect/RVV/generic-widening-conversion-dataflow.mlir --split-input-file --verify-diagnostics | /usr/lib/llvm-20/bin/FileCheck test/Dialect/RVV/generic-widening-conversion-dataflow.mlir`.
- [OK] Generated-bundle dry-run for `widen_i32_to_i64` at counts `7,16,23`:
  `rvv_generated_bundle_abi_e2e: dry_run_success`, artifact root
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-conversion-binding-widen-i32-to-i64-dry`.
- [OK] Generated-bundle dry-run for `widen_i16_to_i32` at counts `7,16,23`:
  `rvv_generated_bundle_abi_e2e: dry_run_success`, artifact root
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-conversion-binding-widen-i16-to-i32-dry`.
- [OK] Real `ssh rvv` generated-bundle run for both converted routes at counts
  `7,16,23`: PASS for `widen_i32_to_i64` and `widen_i16_to_i32`, including
  sign/value-distinguishing correctness and tail preservation.
- [OK] Diff-level active-authority scan over `include`, `lib`, `scripts`, and
  `test`: no newly added positive `RVVI32M1`, `rvv-i32m1`, finite
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
  source-front-door, public exact intrinsic authority, or common/export RVV
  semantic authority.
- [OK] Generated ssh artifact scan: no forbidden legacy/source-front-door/common
  semantic authority markers.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 267/267 passed.

### Self-Repair

- Raw source-tree `llvm-lit` was not usable directly because `/usr/lib/llvm-20/bin/llvm-lit`
  was absent and the source lit config lacked `tianchenrv_obj_root`; this was
  replaced with focused direct `tcrv-opt`/`tcrv-translate`/FileCheck validation
  plus the full build-tree `check-tianchenrv` target.

### Status

[OK] Completed and ready to archive.

### Next Steps

- None for the bounded active conversion operand binding cluster.
