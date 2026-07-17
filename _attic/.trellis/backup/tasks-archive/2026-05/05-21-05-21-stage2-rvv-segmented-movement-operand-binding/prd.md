# Stage2 RVV segmented movement operand binding adoption

## Goal

Adopt the RVV plugin-owned `RVVRouteOperandBindingPlan` contract for the
current active segmented RVV memory-movement routes, so segment source or
destination buffers, field roles, runtime `n`/AVL, provider materialized
operands, emission-plan mirrors, generated headers, and generated-bundle
expectations all come from one route binding authority.

This task continues the operand-binding hardening line from:

- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-route-operand-abi-binding/`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-operand-binding-coverage-closure/`
- `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-arithmetic-select-operand-binding/`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-indexed-movement-operand-binding/`

## Direction Source

- Direction title: `Stage2 RVV segmented movement operand binding adoption`.
- Module owner: RVV plugin-owned `RouteOperandBindingPlan` adoption for the
  current active segmented RVV movement routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `c0aec151 rvv: adopt indexed movement operand binding plans`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.

## What I Already Know

- Current active segmented route-supported/executable routes are:
  - `segment2_deinterleave_unit_store`: `out0[i] = src[2 * i]`,
    `out1[i] = src[2 * i + 1]`.
  - `segment2_interleave_unit_load`: `dst[2 * i] = src0[i]`,
    `dst[2 * i + 1] = src1[i]`.
- These two routes already have typed/pre-realized body coverage, route
  planning/provider support, target artifact fixtures, generated-bundle
  dry-run tests, and prior `ssh rvv` correctness evidence from their archived
  executable-slice tasks.
- At current HEAD, `RVVRouteOperandBindingPlan` has plan IDs for macc,
  strided load/store, scalar broadcast, standalone reduction, masked store,
  computed-mask strided store, ordinary arithmetic/select/reduction/strided
  arithmetic, and indexed gather/scatter routes.
- At current HEAD, segmented routes still carry segment metadata mirrors such
  as `segment_memory_layout`, `segment_count`, `segment_tuple_c_type`,
  `segment_load_intrinsic`, `segment_store_intrinsic`, field roles, field
  names, and source/destination memory forms, but do not yet carry
  `tcrv_rvv.route_operand_binding_plan` or
  `tcrv_rvv.route_operand_binding_operands`.
- Provider materialization for segment routes still reads the segmented source
  and field destination/source ABI parameters directly from the route slice
  (`boundLHSABI`, `boundRHSABI`, `boundOutABI`, `slice.field0ABI`,
  `slice.field1ABI`) rather than requiring segmented binding-plan materialized
  uses.
- Specs require route authority to remain:
  `tcrv.exec` envelope -> selected RVV variant -> typed low-level
  `tcrv_rvv` body -> RVV legality/realization/provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
- Common EmitC/export must remain neutral and must not infer RVV segmented
  memory semantics, segment count, field order, source/destination roles,
  dtype, SEW/LMUL, tail policy, intrinsic choices, route support, or
  correctness from route IDs, artifact names, metadata, descriptors, or
  source-front-door markers.

## Scope

Convert this current segmented movement cluster:

- `segment2_deinterleave_unit_store`: interleaved `src` segment2 load,
  field0/field1 extraction, unit-stride stores to `out0` and `out1`, runtime
  `n`.
- `segment2_interleave_unit_load`: unit-stride loads from `src0` and `src1`,
  segment2 tuple construction, interleaved `dst` segment2 store, runtime `n`.

For every converted route, record logical operand, typed body/runtime role,
segment/field role, C parameter, materialized expression/use, and
mirror/header field in `RVVRouteOperandBindingPlan`, then make provider
emission consume the contract for real emitted operands.

## Requirements

1. Inventory segmented RVV movement routes that are active route-supported or
   executable today and classify them as converted, inactive/parser-only, or
   out of scope.
2. Add binding-plan IDs and logical-operand role validation for:
   - `segment2_deinterleave_unit_store`
   - `segment2_interleave_unit_load`
3. For `segment2_deinterleave_unit_store`, bind:
   - `src`: existing `lhs-input-buffer` runtime role for the interleaved
     segment source, materialized segment2 load base, segment-load source
     memory form, and header mirror.
   - `out0`: `segment-field0-output-buffer`, materialized field0 store base,
     field0 destination role, destination memory form, and header mirror.
   - `out1`: `segment-field1-output-buffer`, materialized field1 store base,
     field1 destination role, destination memory form, and header mirror.
   - `n`: `runtime-element-count`, setvl AVL, loop control, and header mirror.
4. For `segment2_interleave_unit_load`, bind:
   - `src0`: `segment-field0-input-buffer`, materialized field0 load base,
     field0 source role, source memory form, tuple field0 input, and header
     mirror.
   - `src1`: `segment-field1-input-buffer`, materialized field1 load base,
     field1 source role, source memory form, tuple field1 input, and header
     mirror.
   - `dst`: `segment-interleaved-output-buffer`, materialized segment2 store
     base, interleaved destination memory form, and header mirror.
   - `n`: `runtime-element-count`, setvl AVL, loop control, and header mirror.
5. Rewire provider emission so segmented load/store materialized operands use
   the binding contract rather than direct segmented route slice ABI lookup.
6. Require route description mirrors and generated artifact/header metadata for
   converted segmented routes to carry the same binding plan ID and summary.
7. Add or update positive structural checks proving both segmented routes
   carry binding plan IDs, binding summaries, header mirrors, and materialized
   operands from the contract.
8. Add or update negative fail-closed tests for expressible segmented movement
   errors: segment field swaps, base/source/destination swaps, missing or
   duplicate ABI roles, missing materialized uses, mirror/header mismatch,
   stale route-id authority, descriptor/direct-C/source-front-door authority,
   and common/export semantic inference.
9. Keep indexed movement, widening, contraction, conversion, masked matrix
   expansion, source-front-door positive routes, and new operation families
   out of scope.

## Acceptance Criteria

- [x] Current active segmented route inventory is recorded in completion notes.
- [x] `segment2_deinterleave_unit_store` derives `src/out0/out1/n`
      materialized operands and route/header mirrors from
      `RVVRouteOperandBindingPlan`.
- [x] `segment2_interleave_unit_load` derives `src0/src1/dst/n`
      materialized operands and route/header mirrors from
      `RVVRouteOperandBindingPlan`.
- [x] Provider emission fails closed when segmented route bindings are missing,
      duplicated, role-swapped, materialized-use mismatched, or inconsistent
      with runtime ABI mirrors.
- [x] Positive structural lit/FileCheck coverage proves converted segmented
      routes carry binding plan IDs and summaries in emission-plan metadata and
      generated headers.
- [x] Negative fail-closed coverage exists for segmented operand role swaps,
      segment-field swaps, and binding-plan/mirror mismatch surfaces that the
      current textual IR can express.
- [x] Generated-bundle dry-runs pass for representative pre-realized segmented
      deinterleave/interleave routes with counts `7,16,23` and tail/sentinel
      cases.
- [x] Real `ssh rvv` PASS evidence exists for representative converted
      segmented routes when runtime/correctness is claimed.
- [x] Active-authority scan shows no new positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public exact
      intrinsic authority, stale route-id authority, artifact-name authority,
      or common/export RVV semantic authority.
- [x] Focused checks, `check-tianchenrv`, `git diff --check`, Trellis finish/
      archive, clean git status, and one coherent commit are completed if the
      task finishes.

## Non-Goals

- No indexed movement, widening, contraction, conversion, masked movement
  matrix expansion, dtype/LMUL clone batches, or new Stage2 operation
  families.
- No broad segment3/segment4 segmented-memory framework or new segmented
  operation coverage.
- No high-level Linalg/Vector/StableHLO/frontend lowering.
- No source-front-door positive RVV routes.
- No dashboards, report-only work, helper-only cleanup, or broad smoke tests
  as the main achievement.
- No descriptor-driven compute, direct-C/source-export route restoration, or
  compatibility wrapper preserving old route authority.
- No movement of RVV segmented semantics into common EmitC/export, target
  metadata, artifact names, route IDs, descriptors, ABI strings alone, test
  names, or exact intrinsic spelling.

## Validation Plan

1. Validate and maintain this Trellis task.
2. Run focused C++ coverage for route operand binding plan validation.
3. Run focused lit/FileCheck checks for segmented route plan mirrors, target
   artifacts, generated headers, EmitC materialization, and negative
   fail-closed cases.
4. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes.
5. Run generated-bundle dry-runs for pre-realized segmented deinterleave and
   interleave routes at counts `7,16,23`.
6. Run real `ssh rvv` evidence for representative converted segmented routes
   after dry-runs pass.
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
- `test/Target/RVV/*segment2*`
- `test/Scripts/*segment2*`
- Existing segmented movement dialect/materialization fixtures.

## Definition Of Done

- The active segment2 deinterleave/interleave movement cluster consumes
  `RVVRouteOperandBindingPlan` for provider materialization and mirrors.
- Inactive or out-of-scope segmented-like routes, if found, are named with
  status and reason.
- Focused checks, hardware evidence, active-authority scan, and
  `check-tianchenrv` are complete or any skip is justified by concrete
  environment failure.
- Trellis task status is truthful, archived when complete, and one coherent
  commit records the work.

## Completion Notes

### Segmented Route Inventory

- Converted active executable route: `segment2_deinterleave_unit_store`
  (`src[2*i] -> out0[i]`, `src[2*i+1] -> out1[i]`).
- Converted active executable route: `segment2_interleave_unit_load`
  (`src0[i] -> dst[2*i]`, `src1[i] -> dst[2*i+1]`).
- No inactive/parser-only segmented movement route was promoted or rewritten
  in this round.
- Indexed movement, widening, contraction, conversion, masked-memory matrix
  expansion, source-front-door positive routes, and new operation coverage were
  intentionally left out of scope.

### Production Behavior Completed

- `RVVRouteOperandBindingPlan` now has segmented movement plan IDs:
  - `rvv-route-operand-binding:segment2_deinterleave_unit_store.v1`
  - `rvv-route-operand-binding:segment2_interleave_unit_load.v1`
- `segment2_deinterleave_unit_store` now binds `src`, `out0`, `out1`, and `n`
  through the shared route operand binding contract, including materialized
  segment-load base, field0/field1 store bases, segment/field memory-role
  mirrors, setvl AVL, route metadata, and generated header mirrors.
- `segment2_interleave_unit_load` now binds `src0`, `src1`, `dst`, and `n`
  through the same contract, including materialized field0/field1 load bases,
  tuple field inputs, segment-store base, field/source/destination mirrors,
  setvl AVL, route metadata, and generated header mirrors.
- RVV provider emission now consumes the segmented binding contract for
  materialized operands instead of direct segmented route slice ABI lookups.
- Route description verification requires segmented binding-plan mirrors for
  the two converted memory forms and fails closed on missing, mismatched,
  duplicate, role-swapped, stale, or unsupported binding summaries.

### Self-Repair

- Negative interleave field-swap coverage exposed that
  `assignRVVGenericSegment2StoreBinding` overwrote the field values previously
  bound from field0/field1 source loads. That made the existing segment2-store
  field-order check tautological.
- The planner now keeps field source authority from the load bindings and uses
  `segment2_store` only as the segment destination/store operation, so swapped
  field source consumption fails closed before materialization.

### Evidence

- Focused script checks:
  - `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Focused build/checks:
  - `cmake --build build --target tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`
  - `build/bin/tianchenrv-rvv-extension-plugin-test`
- Structural checks:
  - `pre-realized-selected-body-artifact-segment2-deinterleave-unit-store.mlir`
    PLAN and HEADER FileCheck.
  - `pre-realized-selected-body-artifact-segment2-interleave-unit-load.mlir`
    PLAN and HEADER FileCheck.
  - Generated-bundle metadata now checks `tcrv_rvv.route_operand_binding_plan`
    and `tcrv_rvv.route_operand_binding_operands` for both segmented routes.
- Negative fail-closed checks:
  - `rvv-generic-stage2-segment2-deinterleave-operand-binding-negative.mlir`
    rejects source/output ABI name swaps.
  - `rvv-generic-stage2-segment2-interleave-operand-binding-negative.mlir`
    rejects swapped field source consumption.
  - C++ provider tests validate valid segmented binding plans and reject
    `out0/out1` and `src0/src1` role swaps.
- Generated-bundle dry-runs:
  - pre-realized `segment2_deinterleave_unit_store`, counts `7,16,23`
  - pre-realized `segment2_interleave_unit_load`, counts `7,16,23`
- Real hardware evidence:
  - `ssh rvv` PASS for pre-realized `segment2_deinterleave_unit_store`,
    counts `7,16,23`, field-order distinguishing lanes and tail preservation.
  - `ssh rvv` PASS for pre-realized `segment2_interleave_unit_load`, counts
    `7,16,23`, field-order distinguishing lanes and tail preservation.
- Active-authority scan:
  - Added-line scan over the diff found no new positive `RVVI32M1`,
    `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
    descriptor/direct-C/source-export/source-front-door, public exact
    intrinsic route authority, route-id authority, artifact-name authority, or
    common/export RVV semantic authority.
  - Full touched-path scan only found existing guardrail/self-test text and
    pre-existing provider/fixture intrinsic constants.
- Final gates:
  - `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-21-05-21-stage2-rvv-segmented-movement-operand-binding`
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2`
    (`267/267` lit tests passed)
