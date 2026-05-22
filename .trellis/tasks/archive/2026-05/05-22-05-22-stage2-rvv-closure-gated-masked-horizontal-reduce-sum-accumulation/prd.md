# Stage2 RVV Closure-Gated Masked Horizontal Reduce-Sum Accumulation

## Goal

Resolve the Hermes direction brief for masked horizontal reduce-sum
accumulation against current HEAD truth.

The requested production behavior is:

```text
out_scalar_i32[0] = seed_i32 + sum(src_i32[i] where mask[i] is true)
```

on the corrected typed `tcrv_rvv` surface, with RVV plugin-owned selected-body
realization, route planning/provider derivation, common EmitC neutrality, and
`RVVRouteOperandBindingPlan` closure.

Current repository evidence shows that the exact bounded compare-produced mask
route already exists as `computed_mask_standalone_reduce_add`, committed by
`d57f5c6c rvv: add closure-gated computed-mask reduction` and still present at
HEAD. This round must therefore not duplicate the route or create a second
spelling. The bounded work is to validate the existing owner against the new
post-masked-macc brief, repair only true gaps, and add focused fail-closed
coverage if current evidence shows a missing guard.

## Direction Source

- Direction title: `Stage2 RVV closure-gated masked horizontal reduction
  accumulation`.
- Module owner requested: RVV plugin-owned route-supported masked horizontal
  reduce-sum accumulation on the corrected `tcrv_rvv` surface, enforced by
  `RouteOperandBindingPlan` closure.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `6630ae79 rvv: add closure-gated computed masked macc`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Inventory

Inventory is from current code, specs, archived PRDs, and current tests rather
than status fields.

- `computed_mask_standalone_reduce_add` already exists in production route
  planning/provider/script surfaces.
- The typed pre-realized surface is
  `tcrv_rvv.typed_computed_mask_standalone_reduce_pre_realized_body`.
- The realized/explicit typed compute op is
  `tcrv_rvv.masked_standalone_reduce`.
- The route structurally carries compare lhs/rhs inputs, source payload, scalar
  seed accumulator, scalar output, runtime `n`/AVL, SEW32, LMUL m1, policy,
  compare-produced mask facts, reduction kind, accumulator layout, and result
  layout.
- The closure plan ID is
  `rvv-route-operand-binding:computed_mask_standalone_reduce_add.v1`.
- Positive explicit and pre-realized target fixtures exist:
  `test/Target/RVV/explicit-selected-body-artifact-computed-mask-standalone-reduce-add.mlir`
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-standalone-reduce-add.mlir`.
- Negative dialect coverage exists for unsupported kind, stale route id,
  unsupported predicate, wrong source role, wrong accumulator role, wrong mask
  source, and wrong runtime role in
  `test/Dialect/RVV/computed-mask-standalone-reduction-dataflow.mlir`.
- The latest masked macc task added `computed_masked_macc_add` after the
  computed-mask reduction route. The new risk introduced by task ordering is
  accidental post-macc confusion: masked macc must not be accepted or reported
  as horizontal reduction accumulation.

## Requirements

1. Do not add a duplicate production route family if
   `computed_mask_standalone_reduce_add` already satisfies the requested add
   reduction semantics.
2. Verify that the existing route derives route support, materialized operands,
   header mirrors, target leaf, ABI order, and diagnostics from typed
   body/config/runtime facts plus `RVVRouteOperandBindingPlan` closure.
3. Verify that common EmitC/export remains neutral and does not choose
   reduction semantics, dtype/config, mask policy, ABI roles, or intrinsic
   spelling.
4. If live tests reveal missing production behavior, repair the production
   owner in RVV dialect/config/construction/realization/planning/provider only.
5. If the only discovered gap is post-macc stale-brief confusion, add focused
   fail-closed coverage proving that `tcrv_rvv.macc` or
   `tcrv_rvv.masked_macc` cannot be claimed as the masked standalone reduction
   route.
6. Unsupported or malformed missing payload, missing seed accumulator, missing
   scalar result, missing mask for the masked route, missing runtime role,
   dtype/config mismatch, stale route id, wrong binding plan, materialized-use
   mismatch, mirror/header mismatch, elementwise add fallback, macc fallback,
   route-id/helper-string fallback, descriptor/direct-C/source-front-door
   authority, and common/export semantic inference must fail closed where the
   current test surfaces can express them.
7. Runtime/correctness claims require current `ssh rvv` evidence for the
   existing representative explicit and pre-realized add-reduction routes.

## Acceptance Criteria

- [x] PRD and task context truthfully record that the requested production add
      route exists at current HEAD and identify any remaining real gap.
- [x] Positive structural checks prove explicit and pre-realized
      `computed_mask_standalone_reduce_add` carry compare-produced mask,
      source payload, scalar seed accumulator, scalar result, runtime `n`/AVL,
      dtype/config facts, binding plan ID, binding summary, and header mirrors.
- [x] Post-macc regression coverage proves masked macc is not horizontal
      reduce-sum accumulation and cannot claim the standalone reduction binding
      plan or operation mirror.
- [x] Generated-bundle dry-runs pass for representative explicit and
      pre-realized `computed_mask_standalone_reduce_add` counts `7,16,23`.
- [x] Real `ssh rvv` generated-bundle runs pass for representative explicit
      and pre-realized `computed_mask_standalone_reduce_add`, proving seed use,
      active-lane accumulation, inactive-lane exclusion, runtime `n`/AVL
      variation, scalar output correctness, and tail/sentinel preservation.
- [x] Active-authority scan shows no new positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public exact
      intrinsic route authority, artifact-name authority, or common/export RVV
      semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, truthful task
      finish/archive state, clean git status, and one coherent commit complete
      if this task finishes.

## Non-Goals

- No redo of masked macc, segmented/indexed/strided/contiguous masked
  movement, compare/select, signed min/max reductions, widening dot reductions,
  or existing `computed_mask_standalone_reduce_add` production logic unless a
  real current-head defect is found.
- No high-level matmul/GEMM, Linalg/Vector/StableHLO frontend lowering, tensor
  contraction ops, widening dot variants, full reduction matrices, dtype/LMUL
  clone batches, source-front-door positive routes, dashboards, report-only
  work, helper-only cleanup, or future plugin work.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, exact intrinsic
  spelling authority, descriptor-driven direct C, or common EmitC/export
  semantic inference.
- No duplicate `computed_masked_reduce_sum_accum` spelling beside the current
  repository route if it would represent the same production behavior.

## Validation Plan

1. Validate and start the Trellis task.
2. Run focused dialect/target checks for explicit and pre-realized
   `computed_mask_standalone_reduce_add`.
3. Add and run focused post-macc negative coverage only if missing.
4. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
5. Run generated-bundle dry-runs for explicit and pre-realized
   `computed_mask_standalone_reduce_add`, counts `7,16,23`.
6. Run real `ssh rvv` correctness for explicit and pre-realized
   `computed_mask_standalone_reduce_add`, counts `7,16,23`.
7. Run active-authority scans over touched RVV include/lib/script/test paths.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing computed-mask standalone reduction and computed masked macc tests as
  directly relevant anchors.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-closure-gated-masked-macc-accumulation/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-closure-gated-reduction-accumulation/prd.md`

## Completion Evidence

- Current-head production route inventory:
  `computed_mask_standalone_reduce_add` is already implemented by the existing
  typed `tcrv_rvv.masked_standalone_reduce` route family from commit
  `d57f5c6c rvv: add closure-gated computed-mask reduction`.
- Production owners changed in this round: none. The brief was stale for the
  production add route, so this round did not duplicate the route or introduce
  a second route spelling.
- Test owner changed:
  `test/Dialect/RVV/computed-mask-standalone-reduction-dataflow.mlir`.
- Added post-macc fail-closed regressions:
  - `computed_mask_standalone_reduce_rejects_elementwise_add_fallback_claim`
    rejects an elementwise `tcrv_rvv.masked_binary` route-id claim as
    metadata authority.
  - `computed_mask_standalone_reduce_rejects_masked_macc_fallback_claim`
    rejects `tcrv_rvv.masked_macc` with a standalone-reduction result layout.
- Focused structural checks passed:
  - `build/bin/tcrv-opt test/Dialect/RVV/computed-mask-standalone-reduction-dataflow.mlir --split-input-file --verify-diagnostics`
  - explicit selected-body PLAN and HEADER FileCheck for
    `computed_mask_standalone_reduce_add`
  - pre-realized selected-body REALIZED, PLAN, and HEADER FileCheck for
    `computed_mask_standalone_reduce_add`
- Script checks passed:
  - `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Generated-bundle dry-runs passed:
  - explicit route, counts `7,16,23`, artifact
    `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260522-masked-horizontal-reduce-explicit`
  - pre-realized route, counts `7,16,23`, artifact
    `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260522-masked-horizontal-reduce-pre-realized`
- Real `ssh rvv` runs passed:
  - explicit route artifact
    `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260522-masked-horizontal-reduce-explicit-ssh`
  - pre-realized route artifact
    `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260522-masked-horizontal-reduce-pre-realized-ssh`
  - both runs passed counts `7,16,23` with seeds `-11,17`, active-lane counts
    `5,10,14`, inactive-lane counts `2,6,9`, scalar outputs
    `1,-33,-26,29,-5,2`, and tail preservation.
- Active-authority scan on the touched RVV test diff found only the intended
  negative `route_id`, `masked_binary`, and `masked_macc` fail-closed cases.
  No new positive legacy i32/source-front-door/descriptor/direct-C/common-export
  route authority was introduced.
- Final checks passed:
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2` with 322/322 tests.

## Definition Of Done

- The brief is resolved against current HEAD without duplicating an existing
  route.
- Any true gap discovered by focused validation is repaired.
- Current explicit/pre-realized generated-bundle and `ssh rvv` evidence exists
  for masked horizontal reduce-sum accumulation.
- No legacy/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.
