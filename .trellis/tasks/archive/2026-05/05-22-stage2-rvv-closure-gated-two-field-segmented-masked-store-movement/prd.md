# Stage2 RVV Closure-Gated Two-Field Segmented Masked Store Movement

## Goal

Implement one bounded Stage2 RVV memory-movement owner: two-field segmented
masked store movement on the corrected typed `tcrv_rvv` surface, enforced by
RVV plugin-owned legality, selected-body realization, route planning/provider
logic, and `RVVRouteOperandBindingPlan` closure.

The concrete bounded behavior is signed i32 / SEW32 / LMUL m1 segment2 masked
interleaved store from two unit-stride field payload vectors into one
interleaved destination buffer, preserving inactive lanes and tail/sentinel
destination slots:

```text
if cmp_lhs[i] < cmp_rhs[i]:
  dst[2 * i]     = field0[i]
  dst[2 * i + 1] = field1[i]
else:
  dst[2 * i] and dst[2 * i + 1] remain unchanged
tail and sentinel lanes remain preserved
```

This route must be selected from `tcrv.exec` RVV variant structure and typed
low-level `tcrv_rvv` facts. It must not be inferred from route ids, helper
strings, artifact names, metadata mirrors, descriptors, source-front-door
markers, direct-C/source exports, or common EmitC/export semantics.

## Direction Source

- Direction title: `Stage2 RVV closure-gated two-field segmented masked store
  movement`.
- Module owner: RVV plugin-owned route-supported two-field segmented masked
  store movement on the corrected `tcrv_rvv` surface, enforced by
  `RouteOperandBindingPlan` closure.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `bc272368 rvv: add closure-gated segment2 masked load`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Inventory

Inventory is from current RVV dialect/config/construction/selected-body
realization/route planning/provider/generated-bundle surfaces and closest
archived tasks.

- `segment2_interleave_unit_load` is the active executable unmasked segmented
  store sibling. It already carries field0/field1 unit-load sources,
  interleaved destination memory, segment count 2, runtime `n`/AVL,
  dtype/config, target artifact fixtures, generated-bundle evidence, and prior
  real `ssh rvv` evidence.
- `segment2_deinterleave_unit_store` and
  `computed_masked_segment2_load_unit_store` are segmented-load siblings. The
  latest masked-load task introduced `tcrv_rvv.masked_segment2_load`,
  explicit and pre-realized computed-mask paths, closure-gated field roles,
  per-field inactive passthrough, generated-bundle dry-runs, and real
  `ssh rvv` evidence. They are regression anchors only for this owner.
- `computed_masked_strided_store` and
  `computed_masked_indexed_scatter_store_unit_load` are masked-store siblings.
  They provide the no-write inactive-lane policy, compare-produced mask,
  payload/destination/runtime binding closure, provider-owned target leaf,
  generated-bundle evidence, and real `ssh rvv` patterns.
- The current gap is the sibling segment2 masked-store path: no active route
  family such as `computed_masked_segment2_store_unit_load`, no typed
  `tcrv_rvv.masked_segment2_store` surface, no pre-realized computed-mask
  segment2 store body, and no generated-bundle evidence for false-lane
  no-write interleaved stores.

## Requirements

1. Keep scope to one coherent two-field segmented masked store route family.
   Use repository naming conventions; this PRD uses
   `computed_masked_segment2_store_unit_load` unless implementation evidence
   requires a more local name.
2. Selected RVV bodies must structurally carry:
   - compare-produced mask or an explicit typed mask operand;
   - interleaved destination memory base/window role;
   - segment field count `2` or equivalent segmented memory form;
   - field0 and field1 vector payload source roles;
   - inactive-lane no-write / destination-preservation policy;
   - field order and segmented destination memory form;
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
5. Unsupported segment factors beyond two fields, segmented loads beyond
   regression, indexed/strided fallback, missing mask, missing destination
   memory window, missing field payload, missing runtime role,
   dtype/config mismatch, stale route id, old `masked_move` fallback,
   mirror-only authority, descriptor/direct-C/source-front-door authority, and
   common/export semantic inference must fail closed with targeted
   diagnostics.
6. Generated-bundle evidence must use value-distinguishing per-field payload
   data, an interleaved destination layout, mixed true/false mask lanes,
   false-lane no-write preservation, unrelated destination preservation,
   tail/sentinel preservation, and runtime `n`/AVL variation.
7. Runtime/correctness claims require real `ssh rvv` PASS evidence.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe this bounded
      two-field segmented masked store movement owner.
- [x] RVV dialect/ODS/verifier and selected-body realization represent the
      route with typed segmented masked store facts, not positive
      route-id/helper/metadata authority.
- [x] RVV route planning/provider recognizes the typed structure and derives
      ABI order, field0/field1 payloads, compare mask, interleaved
      destination base, runtime `n`/AVL, segment count, field order,
      vector/mask/tuple C types, target leaf/profile, headers, materialized
      operands, and mirrors from body/config/runtime facts.
- [x] `RVVRouteOperandBindingPlan` closure includes compare lhs/rhs or mask
      operand, field0/field1 payload sources, interleaved destination store,
      and runtime `n`/AVL with materialized uses that prove segmented
      masked-store operands rather than mirror-only authority.
- [x] Positive structural/FileCheck tests prove explicit selected-body and, if
      feasible in this bounded round, pre-realized computed-mask routes carry
      compare-produced mask, field payloads, interleaved destination, segment
      count, field order, inactive-lane no-write policy, tail policy,
      runtime `n`/AVL, dtype/config, binding operands, header mirrors, and
      provider-supported mirrors.
- [x] Negative fail-closed tests cover unsupported segment field count,
      segmented-load fallback, indexed/strided fallback, missing mask, missing
      destination memory window, missing field payload, missing runtime role,
      dtype/config mismatch, stale or wrong plan id, mirror/header mismatch,
      materialized-use mismatch, old `masked_move` fallback,
      route-id/helper-string fallback, descriptor/direct-C/source-front-door
      authority, and common/export semantic inference where expressible.
- [x] Generated-bundle dry-runs pass for representative explicit and
      pre-realized two-field segmented masked store routes at runtime counts
      `7,16,23` with value-distinguishing field payload data, mixed masks,
      false-lane no-write preservation, unrelated destination preservation,
      and tail/sentinel checks.
- [x] Real `ssh rvv` PASS evidence covers representative new routes and proves
      segmented masked store correctness, field ordering, runtime `n`/AVL
      handling, inactive-lane no-write, unrelated destination preservation,
      and tail/sentinel preservation.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, public exact intrinsic route authority,
      artifact-name authority, or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, truthful finish/
      archive state, clean git status, and one coherent commit complete if
      this task finishes.

## Non-Goals

- No redo of segment2 masked load, indexed gather/scatter, strided load/store,
  contiguous masked load/store, reductions, compare/select expansion, masked
  elementwise routes, or existing segment2 deinterleave/interleave routes
  except as regression anchors.
- No segment3/segment4, broad segmented load/store matrix, gather/scatter
  expansion, conversions, contractions, dtype/LMUL clone batches, high-level
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
   masked store structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-runs for representative explicit and pre-realized
   segmented masked store routes at counts `7,16,23`.
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
- Existing segment2 interleave/deinterleave, masked segment2 load, masked
  strided store, masked indexed scatter-store, and generated-bundle tests as
  directly relevant anchors.

## Completion Evidence

- Production owners changed:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  target export test ownership, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Added route family:
  `computed_masked_segment2_store_unit_load`, memory form
  `computed-mask-unit-load-segment2-store`, runtime ABI order
  `cmp_lhs,cmp_rhs,src0,src1,dst,n`, typed op
  `tcrv_rvv.masked_segment2_store`, and pre-realized op
  `tcrv_rvv.typed_computed_mask_segment2_store_pre_realized_body`.
- Selected-body realization evidence:
  pre-realized segment2 masked store is rewritten into explicit `setvl`,
  four unit-stride typed loads, compare-produced mask, and
  `tcrv_rvv.masked_segment2_store`; FileCheck verifies the pre-realized op is
  gone and no indexed/strided/segment2 unmasked/masked_move/mask_load/binary
  fallback is used.
- Binding closure evidence:
  `rvv-route-operand-binding:computed_masked_segment2_store_unit_load.v1`
  binds compare lhs/rhs, field0/field1 payload sources, interleaved
  destination store, and runtime `n`/AVL. Plan/header mirrors expose the same
  compact binding operands string that provider materialization consumes.
- Runtime evidence:
  explicit and pre-realized generated bundles both passed real `ssh rvv` for
  counts `7,16,23`; outputs proved active store lanes, false-lane no-write
  preservation, value-distinguishing field order, unrelated source
  preservation, runtime n/AVL variation, and tail/sentinel preservation.
- Focused self-repair:
  repaired target header metadata ordering, kept computed segment2 load
  source/destination metadata as a regression anchor after de-duplicating
  generic masked-memory metadata, and added the new store ABI/role sequence to
  construction protocol tests.
- Checks passed:
  focused dialect/target FileCheck for explicit and pre-realized routes,
  generated-bundle dry-runs for explicit and pre-realized store routes,
  segment2 masked-load regression dry-runs, script py_compile and self-test,
  real `ssh rvv` explicit/pre-realized store runs, `git diff --check`,
  active-authority scans, and
  `cmake --build build --target check-tianchenrv -j2` with 316/316 tests.
- Routes intentionally not converted:
  segment3/4, broad segmented matrices, indexed/strided/contiguous masked
  movement, segment2 masked load redo, reductions, compare/select expansion,
  dtype/LMUL clone batches, frontend lowering, source-front-door routes, and
  future plugin work were left untouched by design.

## Definition Of Done

- One bounded two-field segmented masked store route family is represented,
  verified, route-supported, materialized, dry-run validated, and
  runtime-validated on `ssh rvv` if correctness is claimed.
- Existing segment2 deinterleave/interleave, masked segment2 load, masked unit
  load/store, strided masked load/store, indexed masked gather/scatter,
  reduction, macc, conversion, broadcast, compare/select, and tail/mask Stage2
  routes remain intact.
- The final report distinguishes route-supported, generated-bundle dry-run,
  and executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; if incomplete, the exact continuation point is recorded.
