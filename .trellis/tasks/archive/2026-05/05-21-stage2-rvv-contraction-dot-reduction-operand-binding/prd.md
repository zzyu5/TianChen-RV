# Stage2 RVV contraction dot-reduction operand binding adoption

## Goal

Adopt the RVV plugin-owned `RVVRouteOperandBindingPlan` contract for the
current active contraction dot-reduction route cluster, so logical operands,
typed body values or runtime roles, widening source/result-width facts,
accumulator/reduction roles, optional mask/stride roles, C ABI parameters,
provider materialized operands, emission-plan mirrors, generated headers, and
generated-bundle expectations all come from one binding authority.

This is one bounded operand-origin hardening round. It does not add new RVV
operation coverage.

## Direction Source

- Direction title: `Stage2 RVV contraction dot-reduction operand binding
  adoption`.
- Module owner: RVV plugin-owned `RouteOperandBindingPlan` adoption for the
  current active contraction/dot-reduction RVV routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `4db6013d rvv: adopt widening macc operand binding plan`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.

## What I Already Know

- The prior completed task adopted `RVVRouteOperandBindingPlan` for
  `widening_macc_add` and intentionally left dot-reduction contraction routes
  for this follow-up.
- Recent completed operand-binding tasks converted ordinary arithmetic/select,
  indexed movement, segmented movement, and widening macc routes to
  `RVVRouteOperandBindingPlan`.
- Archived contraction tasks already established a shared contraction route
  family plan, target leaf/profile validation, and selected-body realization
  boundary for:
  - `widening_macc_add`
  - `widening_dot_reduce_add`
  - `strided_input_widening_dot_reduce_add`
  - `computed_masked_widening_dot_reduce_add`
  - `computed_masked_strided_input_widening_dot_reduce_add`
- `widening_macc_add` is already converted by commit `4db6013d`; this task
  treats it as a predecessor pattern, not as a route to reconvert.
- The active dot-reduction cluster to inventory and convert is:
  - `widening_dot_reduce_add`
  - `strided_input_widening_dot_reduce_add`
  - `computed_masked_widening_dot_reduce_add`
  - `computed_masked_strided_input_widening_dot_reduce_add`
- If repository evidence shows one of these routes is inactive/parser-only, the
  exact status must be recorded and positive support must not be invented.
- Common EmitC/export must remain neutral and must not infer RVV contraction
  semantics, dtype/config, source/result width, mask/stride provenance,
  accumulator/reduction layout, intrinsic choice, route support, or ABI roles
  from route IDs, artifact names, ABI strings alone, descriptors,
  source-front-door metadata, mirrors, or exact intrinsic spelling.

## Scope

Convert the current active dot-reduction contraction routes if repository
evidence confirms they are route-supported or executable:

- `widening_dot_reduce_add`: unit-stride i16 source buffers, scalar i32 seed,
  scalar i32 output, runtime `n`, signed widening dot reduction.
- `strided_input_widening_dot_reduce_add`: i16 source buffers with runtime
  `lhs_stride` and `rhs_stride`, scalar i32 seed/output, runtime `n`, signed
  widening dot reduction.
- `computed_masked_widening_dot_reduce_add`: compare-produced mask,
  unit-stride i16 dot source buffers, scalar i32 seed/output, runtime `n`,
  signed masked widening dot reduction with inactive lanes excluded/zeroed by
  the RVV plugin-owned plan.
- `computed_masked_strided_input_widening_dot_reduce_add`: compare-produced
  mask, runtime-strided i16 dot source buffers, scalar i32 seed/output,
  runtime `n`, signed masked widening dot reduction with stride and inactive
  lane facts owned by the RVV plugin plan.

For each converted route, record logical operand, typed body value or runtime
role, source/result width role, reduction/accumulator role, mask role where
applicable, stride role where applicable, C parameter, materialized expression
or use, and mirror/header field in `RVVRouteOperandBindingPlan`, then make
provider emission consume that contract for actual emitted operands.

## Requirements

1. Inventory contraction routes and classify them as converted, already
   converted predecessor, inactive/parser-only, or intentionally deferred with
   exact evidence.
2. Add dot-reduction binding-plan IDs and logical-operand role validation in
   the RVV route planning/provider owner, not in common EmitC/export.
3. Bind `widening_dot_reduce_add` operands through one contract:
   - `lhs`: `lhs-input-buffer`, i16 source load base, dot lhs compute operand,
     source-width mirror/header role.
   - `rhs`: `rhs-input-buffer`, i16 source load base, dot rhs compute operand,
     source-width mirror/header role.
   - `acc`: `scalar-seed-input`, scalar seed splat or reduction seed role,
     result-width/accumulator mirror/header role.
   - `out`: `scalar-output`, scalar reduction store, result-width
     mirror/header role.
   - `n`: `runtime-element-count`, setvl AVL, loop/reduction control, header
     mirror.
4. Bind `strided_input_widening_dot_reduce_add` with the same operands plus:
   - `lhs_stride`: lhs element-stride runtime role, strided load byte/element
     stride use, stride mirror/header role.
   - `rhs_stride`: rhs element-stride runtime role, strided load byte/element
     stride use, stride mirror/header role.
5. Bind `computed_masked_widening_dot_reduce_add` with the unit-stride dot
   operands plus mask-producing compare inputs or mask materialized value,
   inactive-lane zeroing/exclusion role, and mask mirror/header fields as
   required by the existing route structure.
6. Bind `computed_masked_strided_input_widening_dot_reduce_add` with dot,
   mask, and stride roles in the same plugin-owned contract.
7. Rewire provider emission so dot source loads, optional strided loads,
   optional compare/mask values, scalar seed/accumulator use, widening
   product/reduction compute operands, scalar result store, setvl/AVL, and
   mirrors are checked through `RVVRouteOperandBindingPlan`.
8. Require route description mirrors and generated artifact/header metadata for
   converted routes to carry the same binding plan ID and compact summary.
9. Add or update positive structural checks proving converted dot-reduction
   routes carry binding plan IDs, binding summaries, header mirrors, and
   materialized operands from the contract.
10. Add or update negative fail-closed checks for expressible dot-reduction
    operand errors: lhs/rhs swaps, accumulator/output swaps, mask/vector
    mismatches, stride role swaps, source/result-width mismatches, missing or
    duplicate ABI roles, missing materialized uses, mirror/header mismatch,
    stale route-id/helper-string authority, descriptor/direct-C/source-front
    door authority, and common/export semantic inference.
11. Keep conversion routes, segmented/indexed movement, new operation coverage,
    source-front-door positive routes, dtype/LMUL clone batches, frontend
    lowering, dashboards, and helper-only cleanup out of scope.

## Acceptance Criteria

- [ ] Current contraction/dot-reduction route inventory is recorded in
      completion notes with exact active/inactive/deferred status.
- [ ] `widening_dot_reduce_add` derives `lhs/rhs/acc/out/n` materialized
      operands and route/header mirrors from `RVVRouteOperandBindingPlan`.
- [ ] `strided_input_widening_dot_reduce_add` derives
      `lhs/rhs/acc/out/n/lhs_stride/rhs_stride` materialized operands and
      route/header mirrors from `RVVRouteOperandBindingPlan`.
- [ ] `computed_masked_widening_dot_reduce_add` derives dot source operands,
      scalar seed/result operands, mask role/value facts, runtime `n`, and
      route/header mirrors from `RVVRouteOperandBindingPlan`.
- [ ] `computed_masked_strided_input_widening_dot_reduce_add` derives dot
      source operands, scalar seed/result operands, mask role/value facts,
      runtime stride operands, runtime `n`, and route/header mirrors from
      `RVVRouteOperandBindingPlan`.
- [ ] Provider emission fails closed when converted dot-reduction bindings are
      missing, duplicated, role-swapped, materialized-use mismatched,
      source/result-width mismatched, accumulator/result swapped, mask/vector
      mismatched, stride-role swapped, or inconsistent with runtime ABI
      mirrors.
- [ ] Positive structural lit/FileCheck coverage proves converted routes carry
      binding plan IDs and summaries in emission-plan metadata and generated
      headers.
- [ ] C++ or lit negative coverage checks role swaps, missing/duplicate roles,
      missing materialized uses, mirror/materialized operand mismatch,
      source/result-width mismatch, mask/vector mismatch, and stride swaps for
      the converted cluster.
- [ ] Generated-bundle dry-runs pass for representative converted
      dot-reduction routes with counts `7,16,23`, signed width-distinguishing
      values, mask/stride cases where applicable, and tail/sentinel cases.
- [ ] Real `ssh rvv` PASS evidence exists for representative converted routes
      when runtime/correctness is claimed, including dot-reduction correctness,
      mask/stride correctness where applicable, and tail/sentinel preservation.
- [ ] Active-authority scan shows no new positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public exact
      intrinsic authority, stale route-id authority, artifact-name authority,
      or common/export RVV semantic authority.
- [ ] Focused checks, `check-tianchenrv`, `git diff --check`, Trellis finish/
      archive, clean git status, and one coherent commit are completed if the
      task finishes.

## Non-Goals

- No reconversion of `widening_macc_add`; it is already converted and is only a
  predecessor pattern unless a shared helper must be adjusted without changing
  its behavior.
- No conversion-route binding adoption for `widen_i32_to_i64` or
  `widen_i16_to_i32`.
- No new Stage2 operation families or operation coverage.
- No segmented/indexed/masked movement changes outside direct impact on shared
  binding helpers.
- No dtype/LMUL clone batches, unsigned variants, high-level
  Linalg/Vector/StableHLO/frontend lowering, or source-front-door positive
  routes.
- No dashboards, report-only work, helper-only cleanup, broad smoke tests, or
  prompt/spec-only work as the main achievement.
- No descriptor-driven compute, direct-C/source-export route restoration, or
  compatibility wrapper preserving old route authority.
- No movement of RVV operand semantics into common EmitC/export, target
  metadata, artifact names, route IDs, descriptors, ABI strings alone, test
  names, mirrors, or exact intrinsic spelling.

## Validation Plan

1. Validate and start this Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-extension-plugin-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused C++ coverage for route operand binding plan validation.
4. Run focused lit/FileCheck checks for dot-reduction route plan mirrors,
   generated headers, target artifact metadata, provider materialization, and
   fail-closed diagnostics.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
6. Run generated-bundle dry-runs for converted dot-reduction routes at counts
   `7,16,23`.
7. Run real `ssh rvv` evidence for representative converted routes after
   dry-runs pass.
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
- Existing positive/negative tests for:
  - `widening_dot_reduce_add`
  - `strided_input_widening_dot_reduce_add`
  - `computed_masked_widening_dot_reduce_add`
  - `computed_masked_strided_input_widening_dot_reduce_add`

## Definition Of Done

- Active dot-reduction contraction routes consume `RVVRouteOperandBindingPlan`
  for real provider materialization and mirrors, or any inactive route is
  documented with exact evidence.
- Focused positive/negative checks, generated-bundle evidence, representative
  hardware evidence, active-authority scan, and `check-tianchenrv` are complete
  or any skip is justified by concrete environment failure.
- Trellis task status is truthful, archived when complete, and one coherent
  commit records the work.

## Completion Notes

### Route Inventory

- Already converted predecessor, not reconverted here:
  `widening_macc_add` from commit `4db6013d`.
- Converted in this task:
  `widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`.
- Intentionally not converted:
  none in the current active dot-reduction contraction cluster.
- Inactive/parser-only suspected routes:
  none found for the four scoped dot-reduction routes; all four have active
  pre-realized selected-body fixtures, route planning/provider support,
  generated-bundle dry-runs, and real `ssh rvv` executable evidence.

### Implemented Behavior

- Added dot-reduction `RVVRouteOperandBindingPlan` IDs and logical role
  validation for the four scoped routes.
- Rewired RVV route provider materialization so lhs/rhs dot inputs, optional
  compare/mask inputs, scalar accumulator seed, scalar output, runtime `n`,
  and optional lhs/rhs stride roles are consumed through the binding plan.
- Updated emission-plan metadata, target header mirrors, generated-bundle
  expectations, and structural plugin tests to use the binding contract.
- Added fail-closed coverage for accumulator/output role swaps, stride role
  swaps, dot source role swaps, mask/materialized-use mismatch, and
  source/result width materialized-use mismatch.

### Evidence

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`: passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test`: passed.
- Focused target artifact lit for the four scoped dot-reduction MLIR tests:
  4/4 passed.
- Focused generated-bundle dry-run lit for existing dot-reduction script
  tests: 3/3 passed.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`: passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`: passed.
- Generated-bundle dry-run for all four scoped routes at counts `7,16,23`:
  `rvv_generated_bundle_abi_e2e: dry_run_success`, artifact root
  `artifacts/tmp/stage2_contraction_operand_binding/dry-run-all`.
- Real `ssh rvv` generated-bundle run for all four scoped routes at counts
  `7,16,23`: PASS for ordinary dot, strided dot, computed-mask dot, and
  computed-mask strided dot with signed horizontal dot correctness,
  accumulator seed addition, mask/stride checks where applicable, and tail
  preservation. Artifact root:
  `artifacts/tmp/stage2_contraction_operand_binding/ssh-rvv-all`.
- Diff-level active-authority scan found no newly added positive `RVVI32M1`,
  `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  descriptor/direct-C/source-export, source-front-door, public exact intrinsic,
  stale route-id, artifact-name, or common/export RVV semantic authority.
- `git diff --check`: passed.
- `cmake --build build --target check-tianchenrv -j2`: 267/267 passed.

### Self-Repair

- The first focused target artifact lit run exposed two issues:
  generated header FileCheck expected the new binding mirrors before existing
  provider-supported contraction metadata, while the actual header writer emits
  binding mirrors after relation/profile/provider mirrors; the checks were
  reordered to match the production writer.
- The computed-mask strided dot route's full binding summary exceeded the
  bounded single-line target header metadata limit. The plan ID and
  provider-checked materialized-use tokens were compacted while preserving the
  operand ABI roles, dot/mask/stride/source-width/result-width facts, and
  runtime/header mirror requirements that the provider consumes.
