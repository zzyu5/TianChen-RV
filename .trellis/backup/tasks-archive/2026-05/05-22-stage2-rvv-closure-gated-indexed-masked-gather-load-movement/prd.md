# Stage2 RVV Closure-Gated Indexed Masked Gather-Load Movement

## Goal

Implement one bounded Stage2 RVV memory-movement owner: indexed masked
gather-load movement on the corrected typed `tcrv_rvv` surface, enforced by
RVV plugin-owned legality, selected-body realization, route planning/provider
logic, and `RVVRouteOperandBindingPlan` closure.

The concrete bounded behavior is signed i32 / SEW32 / LMUL m1:

```text
loaded[i] = mask[i] ? data[index[i]] : old[i]
out[i] = loaded[i]
tail and inactive lanes preserve sentinels / old destination values
```

This route must be selected from `tcrv.exec` RVV variant structure and typed
low-level `tcrv_rvv` facts. It must not be inferred from route ids, helper
strings, artifact names, metadata mirrors, descriptors, source-front-door
markers, direct-C/source exports, or common EmitC/export semantics.

## Direction Source

- Direction title: `Stage2 RVV closure-gated indexed masked gather-load movement`.
- Module owner: RVV plugin-owned route-supported indexed/gather masked load
  movement on the corrected `tcrv_rvv` surface, enforced by
  `RouteOperandBindingPlan` closure.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `8dc50487 rvv: add closure-gated masked strided load`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Inventory

Inventory is from the current RVV dialect/config/construction/selected-body
realization/route planning/provider/generated-bundle surfaces and closest
archived tasks.

- `indexed_gather_unit_store` is an active executable indexed-load to
  unit-store route. It already carries `data`, `index`, `out`, and `n` through
  `RVVRouteOperandBindingPlan` with indexed data/source mirrors, index EEW,
  offset unit, and generated-bundle evidence.
- `indexed_scatter_unit_load` is an active executable indexed-store sibling and
  is a regression anchor only for indexed operand binding behavior.
- `masked_unit_load_store` and `computed_masked_unit_load_store` are active
  masked load routes that carry compare-produced or explicit mask, source
  buffer, old-destination passthrough, destination store, runtime n/AVL,
  dtype/config, materialized operands, mirrors, generated-bundle evidence, and
  real RVV evidence.
- `computed_masked_strided_load_unit_store` is the immediate predecessor. It
  uses typed `tcrv_rvv.masked_strided_load` facts and closure-gated planning
  for compare-produced mask, source memory window, source byte stride,
  old-destination passthrough, loaded vector result, runtime n/AVL, SEW/LMUL/
  policy, materialized operands, headers, mirrors, and provider-owned target
  leaf derivation.
- The current gap is the combination of indexed/gather memory form with masked
  load semantics and inactive-lane old-vector passthrough. This task must not
  broaden into scatter, indexed stores, segmented movement, or a gather/scatter
  matrix.

## Requirements

1. Keep scope to one coherent indexed masked gather-load route family. Use the
   repository naming convention if it differs from
   `computed_masked_indexed_gather_load_unit_store`.
2. Selected RVV bodies must structurally carry:
   - compare-produced mask or an explicit typed mask operand;
   - source data memory base/window role;
   - index or byte-offset vector role with supported index EEW and offset unit;
   - old-vector passthrough role for inactive lanes;
   - loaded vector result role;
   - optional unit-store artifact sink for executable evidence;
   - tail preservation and inactive-lane passthrough policy;
   - runtime `n` / AVL;
   - dtype/config facts including SEW, LMUL, tail policy, and mask policy.
3. If a pre-realized body is used, RVV selected-body realization must consume
   the pre-realized facts into explicit typed `tcrv_rvv` body structure before
   route construction. It must not change computation semantics, dtype
   semantics, parameter roles, selected variant origin, required capabilities,
   dispatch/fallback behavior, or runtime `n` / AVL values.
4. Route planning/provider must derive route support, ABI order, materialized
   operands, target leaf/profile, headers, mirrors, diagnostics, and runtime
   behavior from typed body/config/capability/runtime facts plus the
   closure-gated `RVVRouteOperandBindingPlan`.
5. Unsupported segmented/scatter/indexed-store variants, missing mask, missing
   index or offset vector, missing source memory window, missing old-vector
   passthrough where required, missing runtime role, dtype/config/index-width
   mismatch, stale route id, strided fallback, old `masked_move` fallback,
   mirror-only authority, descriptor/direct-C/source-front-door authority, and
   common/export semantic inference must fail closed with targeted diagnostics.
6. Generated-bundle evidence must use value-distinguishing source buffers,
   noncontiguous and permuted index patterns, true/false mask lanes,
   inactive-lane passthrough preservation, untouched unrelated lanes,
   tail/sentinel preservation, and runtime n/AVL variation.
7. Runtime/correctness claims require real `ssh rvv` PASS evidence.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe this bounded
      indexed masked gather-load movement owner.
- [x] RVV dialect/config and selected-body realization represent the route with
      typed indexed masked load facts, not positive route-id/helper/metadata
      authority.
- [x] RVV route planning/provider recognizes the typed structure and derives
      ABI order, indexed source base, index vector, mask, old-vector
      passthrough, loaded value, output sink, runtime n/AVL, index EEW, offset
      unit, vector/mask C types, target leaf/profile, headers, materialized
      operands, and mirrors from body/config/runtime facts.
- [x] `RVVRouteOperandBindingPlan` closure includes source data, index vector,
      compare/mask inputs, old passthrough/destination, output sink when
      present, and runtime n/AVL with materialized uses that prove indexed
      masked-load operands rather than mirror-only authority.
- [x] Positive structural/FileCheck tests prove explicit selected-body and, if
      feasible in this bounded round, pre-realized computed-mask routes carry
      compare-produced mask, source data, index vector, old passthrough,
      loaded result, destination/unit-store sink, tail policy, runtime n/AVL,
      dtype/config, binding operands, header mirrors, and provider-supported
      mirrors.
- [x] Negative fail-closed tests cover unsupported memory form, missing mask,
      missing index/offset vector, missing source memory window, missing old
      passthrough where required, missing runtime role, index/dtype/config
      mismatch, stale/wrong plan id, mirror/header mismatch,
      materialized-use mismatch, strided fallback, old `masked_move` fallback,
      route-id/helper-string fallback, descriptor/direct-C/source-front-door
      authority, and common/export semantic inference where expressible.
- [x] Generated-bundle dry-runs pass for representative explicit and
      pre-realized routes at runtime counts `7,16,23` with noncontiguous and
      permuted indices, mixed masks, inactive-lane passthrough, unrelated-lane
      preservation, and tail/sentinel checks.
- [x] Real `ssh rvv` PASS evidence covers representative new routes and proves
      indexed load correctness, runtime n/AVL handling, inactive-lane
      passthrough, noncontiguous/permuted index behavior, and tail/sentinel
      preservation.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, public exact intrinsic route authority,
      artifact-name authority, or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, truthful finish/
      archive state, clean git status, and one coherent commit complete if
      this task finishes.

## Non-Goals

- No redo of strided masked load/store, contiguous masked load/store,
  reductions, compare/select expansion, masked elementwise routes, or existing
  indexed gather/scatter routes except as regression anchors.
- No scatter/indexed stores, segmented movement, gather/scatter matrix,
  conversions, contractions, dtype/LMUL clone batches, high-level
  Linalg/Vector/StableHLO frontend lowering, source-front-door positive
  routes, dashboards, report-only work, helper-only cleanup, or future plugin
  work.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, exact intrinsic
  spelling authority, descriptor-driven direct C, or common EmitC/export
  semantic inference.
- No ad-hoc provider fallback around the closure gate.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets needed for `tcrv-opt`, `tcrv-translate`, RVV dialect,
   RVV plugin, construction, route provider, and target artifact checks.
3. Run focused RVV dialect/verifier tests for positive and negative indexed
   masked gather-load structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-runs for representative explicit and pre-realized
   indexed masked gather-load routes at counts `7,16,23`.
8. Run real `ssh rvv` correctness for representative new routes.
9. Run active-authority scans over touched RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Completion Evidence

- Implemented bounded `computed_masked_indexed_gather_load_unit_store` support
  across typed RVV op surface, config ABI contract, selected-body realization,
  construction protocol metadata, route planning, route provider, and generated
  bundle harness support.
- Added explicit and pre-realized target artifact fixtures, a dialect dataflow
  fixture, generated-bundle dry-run tests, and regression updates for the
  generic RVV op allowlist/fail-closed diagnostics.
- Focused positive FileCheck tests passed for dialect and explicit/pre-realized
  target artifact routes.
- Script validation passed: `python3 -m py_compile
  scripts/rvv_generated_bundle_abi_e2e.py` and
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- Generated-bundle dry-runs passed for explicit and pre-realized
  computed-mask indexed gather-load routes at runtime counts `7,16,23`.
- Real `ssh rvv` generated-bundle runs passed for explicit and pre-realized
  computed-mask indexed gather-load routes at runtime counts `7,16,23`,
  proving active indexed loads, inactive passthrough, noncontiguous/permuted
  indices, source/tail preservation, and runtime n/AVL behavior.
- Self-repair was performed for FileCheck metadata ordering and route allowlist
  diagnostic growth; reruns passed.
- `cmake --build build --target check-tianchenrv -j2` passed with
  `305/305` tests.
- `git diff --check` passed.
- Diff-level authority scan found no new positive legacy/source/descriptor or
  common-export route authority. The exact intrinsic hit is provider-owned
  target leaf spelling after typed closure, not public route authority.

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
- Existing indexed gather, masked load, computed masked unit load,
  computed masked strided load, indexed operand binding, and generated-bundle
  tests as directly relevant anchors.

## Definition Of Done

- One bounded indexed masked gather-load route family is represented,
  verified, route-supported, materialized, dry-run validated, and
  runtime-validated on `ssh rvv` if correctness is claimed.
- Existing indexed gather/scatter, masked unit load/store, and strided masked
  load/store slices remain intact.
- The final report distinguishes route-supported, generated-bundle dry-run,
  and executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; if incomplete, the exact continuation point is recorded.
