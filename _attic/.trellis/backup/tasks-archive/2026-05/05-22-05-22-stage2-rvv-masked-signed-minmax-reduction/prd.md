# Stage2 RVV masked signed minmax reduction closure

## Goal

Add one bounded Stage2 RVV computed-mask signed standalone min/max reduction
subcluster on the corrected typed `tcrv_rvv` surface.

The new routes must combine the existing computed-mask standalone reduction
closure with the completed signed standalone min/max reduction mechanics:

```text
out_scalar_i32[0] = min(seed_i32, src_i32[i] where cmp_lhs_i32[i] <= cmp_rhs_i32[i])
out_scalar_i32[0] = max(seed_i32, src_i32[i] where cmp_lhs_i32[i] <= cmp_rhs_i32[i])
```

The selected RVV bodies must structurally carry compare-produced mask facts,
signed min/max reduction kind, source vector role, accumulator seed role,
scalar output role, runtime `n`/AVL, inactive-lane neutralization policy, and
SEW32/LMUL m1 config facts. RVV plugin legality, selected-body realization,
route planning/provider, target mirrors, and generated-bundle evidence must
derive from those facts and `RVVRouteOperandBindingPlan` closure.

## Direction Source

- Direction title: `Stage2 RVV closure-gated masked signed min/max reduction coverage`.
- Module owner: RVV plugin-owned route-supported expansion for masked or
  computed-mask signed standalone min/max reductions.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `ad627f05 rvv: add signed standalone minmax reductions`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Active Inventory

Inventory comes from current RVV dialect, construction, selected-body
realization, route planning/provider, target artifact, and generated-bundle
script owners.

- `standalone_reduce_add`, `standalone_reduce_min`, and
  `standalone_reduce_max` are route-supported on the generic typed
  `tcrv_rvv.standalone_reduce` surface.
- Signed standalone min/max already have explicit and pre-realized selected
  body fixtures, per-kind route IDs, binding-plan IDs, materialized operand
  mirrors, header metadata, generated-bundle expectations, dry-run coverage,
  and real `ssh rvv` evidence from the archived signed min/max task.
- `computed_mask_standalone_reduce_add` is route-supported on the generic typed
  `tcrv_rvv.masked_standalone_reduce` surface with compare-produced `sle`
  mask, source vector, accumulator seed, scalar output, runtime `n`/AVL,
  binding plan `rvv-route-operand-binding:computed_mask_standalone_reduce_add.v1`,
  materialized compare/source/accumulator/output operands, and generated-bundle
  coverage.
- Current computed-mask standalone reduction support is add-only:
  - RVV dialect verifier accepts only
    `op_kind = "computed_mask_standalone_reduce_add"` for the pre-realized
    computed-mask body and only `kind = "add"` for
    `tcrv_rvv.masked_standalone_reduce`.
  - selected-body realization maps only computed-mask add to realized
    `tcrv_rvv.masked_standalone_reduce kind = "add"`.
  - route planning recognizes only
    `RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd` for
    computed-mask standalone reduction.
  - construction protocol has only the add computed-mask standalone reduction
    route.
  - provider emission currently zeroes inactive lanes before reduction, which
    is correct for add but not for signed min/max.
  - generated-bundle script has only
    `computed_mask_standalone_reduce_add`.
- There is no current positive computed-mask signed min or max standalone
  reduction route support.

## Requirements

1. Keep this task bounded to the pair
   `computed_mask_standalone_reduce_min` and
   `computed_mask_standalone_reduce_max`.
2. Do not redo `standalone_reduce_add/min/max`,
   `computed_mask_standalone_reduce_add`, compare/select, masked elementwise,
   or reduction matrix work except as regression anchors.
3. Add masked signed min/max as ordinary instances of the corrected generic
   typed `tcrv_rvv.masked_standalone_reduce` surface:
   - no `tcrv_rvv.i32_*` helper op growth;
   - no `RVVI32M1*` or `rvv-i32m1` route authority;
   - no descriptor/direct-C/source-front-door positive route;
   - no common EmitC/export semantic inference.
4. Selected RVV bodies must structurally carry:
   - compare lhs/rhs runtime roles and `sle` predicate;
   - compare-produced mask role/source/form;
   - source vector runtime role;
   - signed min/max reduction kind;
   - scalar accumulator seed role;
   - scalar output role;
   - runtime `n`/AVL role;
   - inactive-lane neutral element policy for min/max;
   - SEW32, LMUL m1, and agnostic policy.
5. RVV selected-body realization must consume pre-realized min/max bodies into
   explicit `setvl/with_vl/load/load/load/compare/masked_standalone_reduce/store`
   structure with `kind = "min"` or `kind = "max"`.
6. RVV route planning/provider/construction must derive route support,
   reduction intrinsic, materialized operands, ABI order, headers, route
   mirrors, neutral inactive-lane materialization, and target artifact metadata
   from typed body/config/runtime facts and `RVVRouteOperandBindingPlan`.
7. Unsupported or malformed cases must fail closed where current surfaces can
   express them: unsupported kind, missing mask, missing source vector, missing
   accumulator seed, missing scalar output, runtime role mismatch,
   compare/source role swap, dtype/config mismatch, stale or wrong plan ID,
   materialized-use mismatch, route-id/helper-string fallback,
   descriptor/direct-C/source-front-door authority, and common/export semantic
   inference.

## Acceptance Criteria

- [x] PRD records current masked/computed-mask reduction inventory and bounds
      the task to computed-mask signed standalone min/max.
- [x] Production RVV dialect, construction, selected-body realization,
      planning/provider, and target/script owners accept
      `computed_mask_standalone_reduce_min/max` only through typed body facts
      and fail closed for unsupported variants.
- [x] Positive structural tests prove explicit and pre-realized computed-mask
      min/max routes carry typed reduction kind, compare predicate/mask, source,
      accumulator seed, scalar output, runtime n/AVL, config/policy, binding
      plan materialized operands, inactive-lane neutral policy, and header
      mirrors.
- [x] Negative tests cover unsupported kind, stale route ID or authority
      metadata, missing or swapped source/accumulator/output/runtime roles,
      invalid mask source, materialized-use mismatch, and no
      descriptor/direct-C/source-front-door/common-export authority for touched
      surfaces.
- [x] Generated-bundle dry-runs pass for representative explicit and
      pre-realized computed-mask signed min/max routes at counts `7,16,23`.
- [x] Real `ssh rvv` generated-bundle evidence passes for representative new
      routes, proving signed positive/negative value distinction, accumulator
      seed handling, runtime `n`/AVL handling, inactive-lane neutralization, and
      scalar output tail/sentinel preservation.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, public exact intrinsic route authority, artifact-name
      authority, or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, task finish or
      truthful open status, clean git status, and one coherent commit complete
      if the task finishes.

## Non-Goals

- No unsigned or floating masked min/max reductions.
- No new compare/select predicates beyond reusing supported `sle`.
- No chunk-wise reductions, contractions/dot-product variants, conversions,
  segmented/indexed movement, masked memory movement, dtype or LMUL clone
  batches, Linalg/Vector/StableHLO frontend lowering, dashboards, or report-only
  work.
- No source-front-door positive RVV route and no descriptor/direct-C export
  route.
- No compatibility wrapper preserving legacy i32 route authority.

## Validation Plan

1. Start this Trellis task after PRD/context validation.
2. Build focused targets needed for RVV dialect/plugin/target/script checks.
3. Run focused dialect verifier tests for computed-mask standalone reduction
   positive and negative cases.
4. Run focused target FileCheck tests for explicit/pre-realized computed-mask
   signed min/max realization, route plan, generated header, and artifact
   metadata.
5. Run focused RVV plugin/target tests where relevant for route operand binding
   plan validation and provider fail-closed materialized-use checks.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
7. Run generated-bundle dry-runs for explicit and pre-realized
   `computed_mask_standalone_reduce_min/max` at counts `7,16,23`.
8. Run real `ssh rvv` generated-bundle correctness for representative new
   routes.
9. Run active-authority scans over touched RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Completion Evidence

- Production owners updated: RVV dialect verifier, selected-body realization,
  construction protocol, route planning/provider, and generated-bundle evidence
  tooling.
- Added explicit and pre-realized selected-body artifact tests for
  `computed_mask_standalone_reduce_min` and
  `computed_mask_standalone_reduce_max`.
- Added generated-bundle dry-run lit coverage for explicit and pre-realized
  computed-mask signed min/max routes at counts `7,16,23`.
- Real `ssh rvv` evidence passed for explicit min/max at counts `7,16,23` and
  seeds `-11,17`, reporting active/inactive lanes, scalar output, and
  `tail_preserved`.
- Real `ssh rvv` evidence passed for pre-realized min/max at counts `7,16,23`
  and seeds `-11,17`, reporting active/inactive lanes, scalar output, and
  `tail_preserved`.
- `cmake --build build --target check-tianchenrv -j2` passed `294/294`.
- `git diff --check` passed.
- Added-line authority scan over touched production/script/test paths found no
  new positive `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export, source-front-door, or
  exact `__riscv_*_i32m1` route authority.

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Dialect/RVV/`
- `test/Target/RVV/`
- `test/Scripts/`

## Definition Of Done

- Computed-mask signed standalone min/max reductions are route-supported
  through typed `tcrv_rvv` body facts, plugin-owned realization/planning/provider
  logic, and closure-gated operand binding.
- Runtime/correctness claims are backed by real `ssh rvv` evidence.
- No legacy/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished/archived and committed when complete;
  otherwise it remains open with an exact continuation point.
