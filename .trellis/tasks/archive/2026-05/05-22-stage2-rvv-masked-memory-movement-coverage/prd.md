# Stage2 RVV closure-gated masked memory movement coverage

## Goal

Repair the bounded Stage2 RVV contiguous masked memory movement subcluster so
masked unit-load/store movement is represented by typed RVV memory facts and
validated through `RVVRouteOperandBindingPlan` closure.

Current `masked_unit_load_store` and `computed_masked_unit_load_store` routes
already have selected-body, route planning/provider, generated-bundle, and
`ssh rvv` history. The remaining architectural gap is that their positive
route shape still materializes an unmasked source load followed by a masked
merge:

```text
source load + old-destination load + mask -> masked_move -> unit store
```

This round rewires that production/default route shape to a contiguous masked
data load with explicit passthrough:

```text
mask + source mem_window + old-destination passthrough -> masked_load -> unit store
```

The selected RVV body must carry source/destination memory roles, mask operand
or compare-produced mask, passthrough/inactive-lane policy, runtime n/AVL,
SEW/LMUL/policy, materialized operands, header mirrors, and route support from
typed `tcrv_rvv` structure through RVV plugin-owned realization/planning/provider
logic. `masked_unit_store` stays in scope as the contiguous masked-store sibling
and regression anchor, but this is not a broad memory matrix.

## Direction Source

- Direction title: `Stage2 RVV closure-gated masked memory movement coverage`.
- Module owner: RVV plugin-owned route-supported masked contiguous load/store
  movement on the corrected `tcrv_rvv` surface, enforced by
  `RouteOperandBindingPlan` closure.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `d4a9b79f rvv: add masked signed minmax reductions`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Inventory

Inventory is from current RVV dialect, construction, selected-body realization,
route planning/provider, target artifact, and generated-bundle owners.

- `masked_unit_load_store` is positive for explicit and pre-realized selected
  bodies. It binds `src,mask,dst,n`, uses
  `rvv-route-operand-binding:masked_unit_load_store.v1`, emits header and
  evidence mirrors, and has generated-bundle dry-run/ssh history.
- `computed_masked_unit_load_store` is positive for pre-realized selected
  bodies. It binds `cmp_lhs,cmp_rhs,src,dst,n`, produces the mask by compare,
  uses `rvv-route-operand-binding:computed_masked_unit_load_store.v1`, and has
  generated-bundle dry-run/ssh history.
- `masked_unit_store` is positive for pre-realized selected bodies. It binds
  `src,mask,dst,n`, uses
  `rvv-route-operand-binding:masked_unit_store.v1`, emits masked-store target
  leaves, and has generated-bundle dry-run/ssh history.
- `RVVRouteOperandBindingPlan` closure is present and provider-side construction
  now calls `verifyRVVRouteOperandBindingClosure`, including the masked memory
  operation kinds.
- The old route shape for `masked_unit_load_store` and
  `computed_masked_unit_load_store` still uses `tcrv_rvv.masked_move` and
  provider-side masked merge as the selected compute op. That proves active
  lane update but not a typed contiguous masked data-load movement surface.
- There is no `tcrv_rvv.masked_load` data op. `tcrv_rvv.mask_load` is only a
  predicate mask-buffer load and must not be confused with data masked load.

## Requirements

1. Keep scope to contiguous masked memory movement:
   - runtime-mask `masked_unit_load_store`;
   - compare-produced-mask `computed_masked_unit_load_store`;
   - `masked_unit_store` as the sibling route and regression anchor.
2. Add or repair a typed generic `tcrv_rvv.masked_load` dataflow op, or an
   equivalent explicit typed body fact, for masked contiguous source loads with
   an inactive-lane passthrough vector.
3. Rewire production/default selected-body realization for
   `masked_unit_load_store` and `computed_masked_unit_load_store` to materialize
   masked data load plus unit store instead of unmasked source load plus
   `masked_move`.
4. Route planning/provider must derive route support, ABI order, materialized
   operands, masked-load leaf, headers, mirrors, and diagnostics from typed
   body/config/runtime facts and the binding plan closure.
5. `masked_unit_store` must continue to derive masked-store route support from
   typed mask/payload/destination/runtime facts and the binding plan closure.
6. Unsupported indexed, segmented, strided, source-front-door, descriptor,
   direct-C/source-export, stale route-id, helper-string, artifact-name, and
   common/export semantic authority must fail closed or stay absent.
7. Missing mask, missing source mem window, missing destination mem window,
   missing passthrough where masked load requires one, missing runtime n/AVL,
   vector/scalar role swaps, dtype/config mismatches, wrong memory form, wrong
   plan id, mirror/header mismatch, and materialized-use mismatch must be
   covered where current surfaces can express them.

## Acceptance Criteria

- [x] PRD records current load/store/masked movement inventory and bounds the
      task to contiguous masked data load + unit store and masked store sibling
      coverage.
- [x] Production RVV dialect/ODS/verifier adds or repairs typed masked data-load
      body authority without adding dtype-prefixed helper ops or legacy route
      authority.
- [x] RVV selected-body realization for runtime-mask and computed-mask unit
      load-store materializes `setvl`, mask producer, old-destination
      passthrough load, typed masked data load, and unit store.
- [x] RVV route planning/provider recognizes typed masked data load as the
      selected compute/movement op and derives masked-load intrinsic/header,
      materialized operands, ABI bindings, mirrors, and diagnostics from typed
      body facts and `RVVRouteOperandBindingPlan`.
- [x] Positive structural tests prove explicit and pre-realized runtime-mask
      unit load-store routes and pre-realized computed-mask unit load-store
      routes carry mask, source, passthrough/destination, runtime n/AVL,
      dtype/config/policy, binding-plan operands, and header mirrors.
- [x] Positive masked-store regression tests still prove `masked_unit_store`
      materialized operands and mirrors are closure-gated.
- [x] Negative fail-closed tests cover missing mask, unsupported memory form,
      missing or wrong source/destination role, missing passthrough, missing
      runtime role, vector/scalar role swap or dtype/config mismatch, stale or
      wrong plan id/mirror, materialized-use mismatch, route-id/helper-string
      fallback, descriptor/direct-C/source-front-door authority, and
      common/export semantic inference where expressible.
- [x] Generated-bundle dry-runs pass for representative explicit and
      pre-realized runtime-mask unit load-store routes and pre-realized
      computed-mask unit load-store routes at counts `7,16,23`; masked-store
      dry-run remains passing.
- [x] Real `ssh rvv` evidence passes for representative new/rewired routes,
      proving active masked loads/stores, inactive-lane passthrough or store
      suppression, runtime n/AVL handling, and tail/sentinel preservation.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, public exact intrinsic route authority,
      artifact-name authority, or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, truthful task
      finish/archive state, clean git status, and one coherent commit complete
      if the task finishes.

## Non-Goals

- No segmented, indexed, strided, gather/scatter, or atomic masked movement.
- No contractions/dot-product variants, conversions, unsigned/floating
  reduction work, dtype/LMUL clone batches, or high-level Linalg/Vector/StableHLO
  frontend lowering.
- No source-front-door positive route, descriptor-driven computation,
  direct-C/source-export route, dashboard, report-only work, or helper-only
  cleanup as the main result.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, or
  dtype-prefixed helper route authority.
- No common EmitC/export logic choosing RVV masked-load/store semantics.

## Validation Plan

1. Start the Trellis task after PRD/context validation.
2. Build focused targets needed for RVV dialect/plugin/target/script checks.
3. Run focused RVV dialect verifier tests for masked load/store positive and
   negative dataflow.
4. Run focused target FileCheck tests for explicit/pre-realized runtime-mask
   load-store, pre-realized computed-mask load-store, and masked-store
   regression mirrors.
5. Run relevant plugin/target C++ tests covering route operand binding closure
   and target artifact mirror validation.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-runs for `masked_unit_load_store`,
   `computed_masked_unit_load_store`, and `masked_unit_store` at counts
   `7,16,23`.
8. Run real `ssh rvv` generated-bundle correctness for representative rewired
   routes at counts `7,16,23`.
9. Run active-authority scans over touched RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
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

- Runtime-mask and computed-mask contiguous masked unit load-store routes use
  typed masked data-load body facts and closure-gated operand bindings, not
  unmasked source load plus masked-merge route authority.
- `masked_unit_store` remains route-supported through plugin-owned typed
  mask/payload/store facts and closure-gated operands.
- Runtime/correctness claims are backed by real `ssh rvv` evidence.
- No legacy/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished/archived and committed when complete;
  otherwise it remains open with an exact continuation point.
