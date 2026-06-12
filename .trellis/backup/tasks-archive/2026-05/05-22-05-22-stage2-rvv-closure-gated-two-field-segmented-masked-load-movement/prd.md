# Stage2 RVV Closure-Gated Two-Field Segmented Masked Load Movement

## Goal

Implement one bounded Stage2 RVV memory-movement owner: two-field segmented
masked load movement on the corrected typed `tcrv_rvv` surface, enforced by
RVV plugin-owned legality, selected-body realization, route planning/provider
logic, and `RVVRouteOperandBindingPlan` closure.

The concrete bounded behavior is signed i32 / SEW32 / LMUL m1 segment2 masked
deinterleave load from one interleaved source buffer into two unit-stride
field result buffers, preserving per-field old-vector passthrough values when
the mask lane is false:

```text
if cmp_lhs[i] < cmp_rhs[i]:
  out0[i] = src[2 * i]
  out1[i] = src[2 * i + 1]
else:
  out0[i] = old0[i]
  out1[i] = old1[i]
tail and sentinel lanes remain preserved
```

This route must be selected from `tcrv.exec` RVV variant structure and typed
low-level `tcrv_rvv` facts. It must not be inferred from route ids, helper
strings, artifact names, metadata mirrors, descriptors, source-front-door
markers, direct-C/source exports, or common EmitC/export semantics.

## Direction Source

- Direction title: `Stage2 RVV closure-gated two-field segmented masked load
  movement`.
- Module owner: RVV plugin-owned route-supported two-field segmented masked
  load movement on the corrected `tcrv_rvv` surface, enforced by
  `RouteOperandBindingPlan` closure.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `c2505495 rvv: add closure-gated masked indexed scatter
  store`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Inventory

Inventory is from current RVV dialect/config/construction/selected-body
realization/route planning/provider/generated-bundle surfaces and closest
archived tasks.

- `segment2_deinterleave_unit_store` is an active executable segmented-load
  to two unit-store route. It already carries interleaved source memory,
  field0/field1 result order, two destination buffers, runtime `n`/AVL,
  segment count, dtype/config, selected-body realization, route provider
  output, generated-bundle evidence, and prior real `ssh rvv` evidence.
- `segment2_interleave_unit_load` is the sibling segmented-store route for
  two unit-load sources into one interleaved destination. It is regression
  context only for this owner.
- The segmented movement operand-binding task adopted
  `RVVRouteOperandBindingPlan` for active segment2 deinterleave/interleave
  routes, including materialized segmented source or destination operands,
  field roles, runtime `n`, header mirrors, and generated-bundle expectations.
- `computed_masked_unit_load_store`, `computed_masked_strided_load_unit_store`,
  and `computed_masked_indexed_gather_load_unit_store` are masked load
  siblings. They provide the compare-produced mask, old-vector passthrough,
  runtime `n`/AVL, closure-gated binding, target mirror, and generated-bundle
  evidence patterns.
- `computed_masked_indexed_scatter_store_unit_load` is the immediate closure
  gate predecessor. It proved the provider can carry compare lhs/rhs, source
  payload, index vector, destination memory, runtime, dtype/config, materialized
  operands, mirrors, and provider-owned target leaf through typed RVV facts.
- The current gap is a route-supported two-field segmented masked load:
  interleaved source memory plus compare-produced mask or mask operand,
  per-field old-vector passthrough inputs, per-field loaded vector results,
  per-field unit-store outputs, runtime `n`/AVL, segment count, dtype/config,
  and closure-gated materialized operands on one production path.

## Requirements

1. Keep scope to one coherent two-field segmented masked load route family. Use
   repository naming conventions; this PRD uses
   `computed_masked_segment2_load_unit_store` unless implementation evidence
   requires a more local name.
2. Selected RVV bodies must structurally carry:
   - compare-produced mask or an explicit typed mask operand;
   - interleaved source memory base/window role;
   - segment field count `2` or equivalent segmented memory form;
   - field0 and field1 loaded vector result roles;
   - field0 and field1 old-vector passthrough roles for inactive lanes;
   - field0 and field1 unit-store output memory roles;
   - field order and segmented source memory form;
   - runtime `n` / AVL;
   - dtype/config facts including SEW, LMUL, tail policy, and mask policy.
3. If a pre-realized body is used, RVV selected-body realization must consume
   the pre-realized facts into explicit typed `tcrv_rvv` body structure before
   route construction. It must not change computation semantics, dtype
   semantics, parameter roles, selected variant origin, required capabilities,
   dispatch/fallback behavior, field order, inactive-lane policy, or runtime
   `n` / AVL.
4. Route planning/provider must derive route support, ABI order, materialized
   operands, target leaf/profile, headers, mirrors, diagnostics, and runtime
   behavior from typed body/config/capability/runtime facts plus the
   closure-gated `RVVRouteOperandBindingPlan`.
5. Unsupported segment factors beyond two fields, segmented stores,
   indexed/strided fallback, missing mask, missing source memory window,
   missing field result or passthrough, missing field output memory, missing
   runtime role, dtype/config mismatch, stale route id, old masked_move
   fallback, mirror-only authority, descriptor/direct-C/source-front-door
   authority, and common/export semantic inference must fail closed with
   targeted diagnostics.
6. Generated-bundle evidence must use value-distinguishing interleaved source
   data, mixed true/false mask lanes, distinct field old-vector passthrough
   values, per-field inactive-lane preservation, tail/sentinel preservation,
   and runtime `n`/AVL variation.
7. Runtime/correctness claims require real `ssh rvv` PASS evidence.

## Acceptance Criteria

- [ ] PRD, implement/check context, and task metadata describe this bounded
      two-field segmented masked load movement owner.
- [ ] RVV dialect/ODS/verifier and selected-body realization represent the
      route with typed segmented masked load facts, not positive
      route-id/helper/metadata authority.
- [ ] RVV route planning/provider recognizes the typed structure and derives
      ABI order, interleaved source base, compare mask, old field
      passthroughs, field outputs, runtime `n`/AVL, segment count, field
      order, vector/mask/tuple C types, target leaf/profile, headers,
      materialized operands, and mirrors from body/config/runtime facts.
- [ ] `RVVRouteOperandBindingPlan` closure includes compare lhs/rhs or mask
      operand, segmented source memory, field0/field1 old passthroughs,
      field0/field1 outputs, and runtime `n`/AVL with materialized uses that
      prove segmented masked-load operands rather than mirror-only authority.
- [ ] Positive structural/FileCheck tests prove explicit selected-body and, if
      feasible in this bounded round, pre-realized computed-mask routes carry
      compare-produced mask, interleaved source, segment count, field order,
      field results, old-vector passthrough, field output stores, tail policy,
      runtime `n`/AVL, dtype/config, binding operands, header mirrors, and
      provider-supported mirrors.
- [ ] Negative fail-closed tests cover unsupported segment field count,
      segmented-store form, indexed/strided fallback, missing mask, missing
      source memory window, missing field result, missing old passthrough,
      missing output field role, missing runtime role, dtype/config mismatch,
      stale or wrong plan id, mirror/header mismatch, materialized-use
      mismatch, old masked_move fallback, route-id/helper-string fallback,
      descriptor/direct-C/source-front-door authority, and common/export
      semantic inference where expressible.
- [ ] Generated-bundle dry-runs pass for representative explicit and
      pre-realized two-field segmented masked load routes at runtime counts
      `7,16,23` with value-distinguishing interleaved field data, mixed masks,
      inactive-lane passthrough preservation, and tail/sentinel checks.
- [ ] Real `ssh rvv` PASS evidence covers representative new routes and proves
      segmented masked load correctness, field ordering, runtime `n`/AVL
      handling, inactive-lane passthrough, and tail/sentinel preservation.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, public exact intrinsic route authority,
      artifact-name authority, or common/export RVV semantic authority.
- [ ] Focused checks, `git diff --check`, `check-tianchenrv`, truthful finish/
      archive state, clean git status, and one coherent commit complete if
      this task finishes.

## Non-Goals

- No redo of indexed gather-load/scatter-store, strided load/store, contiguous
  masked load/store, reductions, compare/select expansion, masked elementwise
  routes, or existing segment2 deinterleave/interleave routes except as
  regression anchors.
- No segmented stores, segment3/segment4 matrices, scatter/gather expansion,
  conversions, contractions, dtype/LMUL clone batches, high-level
  Linalg/Vector/StableHLO frontend lowering, source-front-door positive
  routes, dashboards, report-only work, helper-only cleanup, or future plugin
  work.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, exact intrinsic
  spelling authority, descriptor-driven direct C, or common EmitC/export
  semantic inference.
- No ad-hoc provider fallback around the closure gate.

## Validation Plan

1. Validate and maintain the Trellis task.
2. Build focused targets needed for `tcrv-opt`, `tcrv-translate`, RVV dialect,
   RVV plugin, construction, route provider, and target artifact checks.
3. Run focused RVV dialect/verifier tests for positive and negative segmented
   masked load structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-runs for representative explicit and pre-realized
   segmented masked load routes at counts `7,16,23`.
8. Run real `ssh rvv` correctness for representative new routes.
9. Run active-authority scans over touched RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing segment2 deinterleave/interleave, masked load, masked strided
  load, masked indexed gather-load, indexed masked scatter-store, and
  generated-bundle tests as directly relevant anchors.

## Definition Of Done

- One bounded two-field segmented masked load route family is represented,
  verified, route-supported, materialized, dry-run validated, and
  runtime-validated on `ssh rvv` if correctness is claimed.
- Existing segment2 deinterleave/interleave, masked unit load/store, strided
  masked load/store, indexed masked gather/scatter, reduction, macc,
  conversion, broadcast, compare/select, and tail/mask Stage2 routes remain
  intact.
- The final report distinguishes route-supported, generated-bundle dry-run,
  and executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; if incomplete, the exact continuation point is recorded.
