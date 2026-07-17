# Stage2 RVV signed minmax standalone reduction coverage

## Goal

Expand one bounded Stage2 RVV reduction family from add-only standalone scalar
reduction to signed standalone min/max scalar reductions on the corrected typed
`tcrv_rvv` surface.

This task is RVV plugin-owned and closure-gated. The selected body must carry
operation kind, source vector role, scalar accumulator or identity/seed role,
scalar result role, runtime `n`/AVL, dtype/config, and policy facts. The RVV
plugin must derive selected-body realization, route planning, materialized
operands, headers, mirrors, and generated-bundle evidence from those facts and
`RVVRouteOperandBindingPlan`.

## Direction Source

- Direction title: `Stage2 RVV closure-gated signed min/max reduction coverage expansion`.
- Module owner: RVV plugin-owned route-supported expansion for signed
  standalone min/max reduction routes on the corrected `tcrv_rvv` surface.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: only
  `.trellis/tasks/05-22-05-22-stage2-rvv-signed-minmax-reduction/` from the
  failed prior attempt.
- Initial HEAD: `75ed4dd5 rvv: normalize base reduction binding coverage`.
- Previous run failed before PRD/source implementation with a transient
  Codex/API timeout; live repo evidence shows only this task scaffold was
  created.

## Current Active Inventory

Inventory comes from current code and tests, not status or report wording.

- `standalone_reduce_add` is complete from the archived base-reduction task:
  explicit and pre-realized selected-body fixtures, `ssh rvv` evidence,
  generated-bundle script support, and binding plan
  `rvv-route-operand-binding:standalone_reduce_add.v1`.
- `reduce_add` is complete as an add-only chunk-wise reduction regression
  anchor with normalized binding summaries.
- `computed_mask_standalone_reduce_add` is complete as a computed-mask add-only
  regression anchor.
- Current standalone reduction support is add-only in these production owners:
  - RVV dialect verifier accepts only `standalone_reduce_add` /
    `tcrv_rvv.standalone_reduce kind = "add"`;
  - selected-body realization maps only `standalone_reduce_add` to realized
    `kind = "add"`;
  - route planning records only `StandaloneReduceAdd` for standalone reduction;
  - construction protocol has only the add standalone reduction route;
  - generated-bundle evidence has only `standalone_reduce_add`.
- There is no current positive signed standalone min or max route support.

## Requirements

1. Keep this task bounded to signed standalone scalar min/max reductions:
   - preferably implement `standalone_reduce_min` and `standalone_reduce_max`
     together because they share the same body, ABI, route, and evidence
     mechanics;
   - if the pair proves too large, complete one coherent signed min/max
     submodule and leave a truthful continuation point.
2. Do not redo `reduce_add`, `standalone_reduce_add`, or
   `computed_mask_standalone_reduce_add` except as regression anchors.
3. Add signed standalone min/max as ordinary instances of the corrected generic
   typed `tcrv_rvv.standalone_reduce` surface:
   - no `tcrv_rvv.i32_*` helper op growth;
   - no `RVVI32M1*` or `rvv-i32m1` route authority;
   - no descriptor/direct-C/source-front-door positive route.
4. Selected RVV bodies must structurally carry:
   - reduction kind (`min` / `max` through typed body facts);
   - source vector ABI role;
   - scalar accumulator/identity seed role;
   - scalar output role;
   - runtime `n`/AVL role;
   - SEW32 LMUL m1 config and agnostic policy.
5. RVV plugin selected-body realization must consume
   `standalone_reduce_min` / `standalone_reduce_max` into explicit
   `setvl/with_vl/load/standalone_reduce/store` body structure with
   `tcrv_rvv.standalone_reduce kind = "min"` / `"max"`.
6. RVV route planning/provider/construction must derive route support,
   reduction intrinsic, materialized operands, ABI order, headers, route
   mirrors, and target artifact metadata from typed body/config facts and
   `RVVRouteOperandBindingPlan`.
7. Unsupported reduction kinds, unsigned/floating variants, missing source,
   missing accumulator/seed, missing output, vector/scalar role swaps, missing
   runtime role, stale plan IDs, materialized-use mismatch, route-id/helper
   fallback, descriptor/direct-C/source-front-door authority, and common/export
   semantic inference must fail closed through focused tests where the current
   surfaces can express them.

## Acceptance Criteria

- [ ] PRD records current signed standalone min/max inventory and bounds the
      task to the standalone route family.
- [ ] Production RVV dialect/config/construction/selected-body realization/
      planning/provider code accepts signed standalone min/max only through
      typed body facts and fails closed for unsupported variants.
- [ ] Positive structural tests prove explicit and pre-realized
      `standalone_reduce_min` / `standalone_reduce_max` routes carry typed
      reduction kind, source, accumulator seed, scalar result, runtime n/AVL,
      config/policy, binding-plan materialized operands, and header mirrors.
- [ ] Negative tests cover unsupported kind or unsigned/floating kind, stale
      route ID, missing or swapped source/accumulator/output/runtime roles,
      materialized-use mismatch, and no descriptor/direct-C/source-front-door
      authority for touched surfaces.
- [ ] Generated-bundle dry-runs pass for representative explicit and
      pre-realized signed min/max routes at counts `7,16,23`.
- [ ] Real `ssh rvv` generated-bundle evidence passes for representative new
      routes, proving signed positive/negative value distinction,
      accumulator/identity seed handling, runtime `n`/AVL handling, scalar-only
      output, and tail/sentinel preservation.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, public exact intrinsic route authority,
      artifact-name authority, or common/export RVV semantic authority.
- [ ] Focused checks, `git diff --check`, `check-tianchenrv`, task finish or
      truthful open status, clean git status, and one coherent commit complete
      if the task finishes.

## Non-Goals

- No unsigned or floating min/max reductions.
- No masked standalone min/max reductions.
- No chunk-wise `reduce_min` / `reduce_max` unless strictly required by shared
  helper mechanics; this task targets standalone scalar-output routes.
- No contractions/dot-product variants.
- No compare/select predicate expansion.
- No conversion, segmented/indexed movement, masked memory movement, dtype or
  LMUL clone batches, Linalg/Vector/StableHLO frontend lowering, dashboards, or
  report-only work.
- No source-front-door positive RVV route and no descriptor/direct-C export
  route.
- No compatibility wrapper preserving legacy i32 route authority.

## Validation Plan

1. Start this Trellis task after PRD/context validation.
2. Run focused build for touched binaries when needed:
   `cmake --build build --target tcrv-opt tcrv-translate`.
3. Run focused dialect and target FileCheck coverage for explicit/pre-realized
   signed standalone min/max routes and negative fail-closed cases.
4. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
5. Run generated-bundle dry-runs for explicit and pre-realized
   `standalone_reduce_min` / `standalone_reduce_max`, counts `7,16,23`.
6. Run real `ssh rvv` generated-bundle checks for representative new routes,
   counts `7,16,23`, with seed variation.
7. Run focused active-authority scans over touched RVV include/lib/script/test
   paths.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`.

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
- `test/Target/RVV/`
- `test/Dialect/RVV/`
- `test/Transforms/LoweringBoundary/`
- `test/Scripts/`

## Definition Of Done

- Signed standalone min/max reductions are route-supported through typed
  `tcrv_rvv` body facts, plugin-owned realization/planning/provider logic, and
  closure-gated operand binding.
- Runtime/correctness claims are backed by real `ssh rvv` evidence.
- No legacy/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished/archived and committed when complete;
  otherwise it remains open with an exact continuation point.
