# Stage2 RVV Closure-Gated Indexed Masked Scatter-Store Movement

## Goal

Implement one bounded Stage2 RVV memory-movement owner: computed-mask indexed
scatter-store movement on the corrected typed `tcrv_rvv` surface, enforced by
RVV plugin-owned legality, selected-body realization, route planning/provider
logic, and `RVVRouteOperandBindingPlan` closure.

The concrete bounded behavior is signed i32 / SEW32 / LMUL m1 with unique
indices:

```text
if cmp_lhs[i] < cmp_rhs[i]:
  dst[index[i]] = src[i]
else:
  dst[index[i]] and all unrelated lanes remain unchanged
tail and sentinel lanes remain preserved
```

This route must be selected from `tcrv.exec` RVV variant structure and typed
low-level `tcrv_rvv` facts. It must not be inferred from route ids, helper
strings, artifact names, metadata mirrors, descriptors, source-front-door
markers, direct-C/source exports, or common EmitC/export semantics.

## Direction Source

- Direction title: `Stage2 RVV closure-gated indexed masked scatter-store movement`.
- Module owner: RVV plugin-owned route-supported indexed/scatter masked store
  movement on the corrected `tcrv_rvv` surface, enforced by
  `RouteOperandBindingPlan` closure.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `3610b2d2 rvv: add closure-gated masked indexed gather load`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Inventory

Inventory is from current RVV dialect/config/construction/selected-body
realization/route planning/provider/generated-bundle surfaces and closest
archived tasks.

- `indexed_scatter_unit_load` is an active executable unit-load to
  indexed-store route. It already carries `src`, `index`, `dst`, and `n`
  through `RVVRouteOperandBindingPlan`, plus index EEW, offset unit,
  unique-index policy, target artifact fixtures, generated-bundle dry-runs,
  and prior real `ssh rvv` evidence.
- `computed_masked_strided_store` is the closest masked-store sibling. It uses
  typed `tcrv_rvv.masked_strided_store` facts and closure-gated planning for
  compare-produced mask, source payload, destination byte-strided no-write
  store, runtime n/AVL, dtype/config, materialized operands, headers, mirrors,
  and provider-owned target leaf derivation.
- `computed_masked_indexed_gather_load_unit_store` is the immediate indexed
  masked predecessor. It uses typed `tcrv_rvv.masked_indexed_load` facts and
  closure-gated planning for compare-produced mask, source memory, index
  vector, old-destination passthrough, unit-store sink, runtime n/AVL,
  dtype/config, materialized operands, headers, mirrors, generated-bundle
  evidence, and real `ssh rvv` evidence.
- The current gap is the sibling indexed/scatter masked-store path: no active
  `tcrv_rvv.masked_indexed_store`, no pre-realized computed-mask indexed
  scatter-store body, no route kind such as
  `computed_masked_indexed_scatter_store_unit_load`, and no generated-bundle
  evidence for false-lane no-write indexed stores.

## Requirements

1. Keep scope to one coherent indexed masked scatter-store route family. Use
   repository naming conventions; this PRD uses
   `computed_masked_indexed_scatter_store_unit_load` unless implementation
   evidence requires a more local name.
2. Selected RVV bodies must structurally carry:
   - compare-produced mask or an explicit typed mask operand;
   - destination memory base/window role;
   - index or byte-offset vector role with supported index EEW and offset unit;
   - vector payload source role;
   - inactive-lane no-write / untouched-destination policy;
   - explicit unique-index policy or fail-closed duplicate-index diagnostic;
   - tail preservation policy;
   - runtime `n` / AVL;
   - dtype/config facts including SEW, LMUL, tail policy, and mask policy.
3. If a pre-realized body is used, RVV selected-body realization must consume
   the pre-realized facts into explicit typed `tcrv_rvv` body structure before
   route construction. It must not change computation semantics, dtype
   semantics, parameter roles, selected variant origin, required capabilities,
   dispatch/fallback behavior, duplicate-index policy, or runtime `n` / AVL.
4. Route planning/provider must derive route support, ABI order, materialized
   operands, target leaf/profile, headers, mirrors, diagnostics, and runtime
   behavior from typed body/config/capability/runtime facts plus the
   closure-gated `RVVRouteOperandBindingPlan`.
5. Unsupported segmented variants, gather-load variants beyond regression,
   missing mask, missing index or offset vector, missing destination memory
   window, missing vector payload, missing runtime role, index/dtype/config
   mismatch, stale route id, strided fallback, old `masked_move` fallback,
   mirror-only authority, descriptor/direct-C/source-front-door authority, and
   common/export semantic inference must fail closed with targeted diagnostics.
6. Generated-bundle evidence must use value-distinguishing destination buffers,
   noncontiguous/permuted unique index patterns, mixed true/false mask lanes,
   false-lane no-write preservation, untouched unrelated lanes, duplicate-index
   behavior documented or fail-closed, tail/sentinel preservation, and runtime
   n/AVL variation.
7. Runtime/correctness claims require real `ssh rvv` PASS evidence.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe this bounded
      indexed masked scatter-store movement owner.
- [x] RVV dialect/ODS/verifier and selected-body realization represent the
      route with typed indexed masked store facts, not positive
      route-id/helper/metadata authority.
- [x] RVV route planning/provider recognizes the typed structure and derives
      ABI order, indexed destination base, index vector, compare mask, payload
      source, runtime n/AVL, index EEW, offset unit, vector/mask C types,
      target leaf/profile, headers, materialized operands, and mirrors from
      body/config/runtime facts.
- [x] `RVVRouteOperandBindingPlan` closure includes compare lhs/rhs, source
      payload, index vector, destination store, and runtime n/AVL with
      materialized uses that prove indexed masked-store operands rather than
      mirror-only authority.
- [x] Positive structural/FileCheck tests prove explicit selected-body and, if
      feasible in this bounded round, pre-realized computed-mask routes carry
      compare-produced mask, source payload, destination, index vector,
      inactive-lane no-write policy, unique-index policy, tail policy, runtime
      n/AVL, dtype/config, binding operands, header mirrors, and
      provider-supported mirrors.
- [x] Negative fail-closed tests cover unsupported memory form, missing mask,
      missing index/offset vector, missing destination memory window, missing
      vector payload, missing runtime role, index/dtype/config mismatch,
      duplicate-index ambiguity, stale/wrong plan id, mirror/header mismatch,
      materialized-use mismatch, strided or masked_move fallback,
      route-id/helper-string fallback, descriptor/direct-C/source-front-door
      authority, and common/export semantic inference where expressible.
- [x] Generated-bundle dry-runs pass for representative explicit and
      pre-realized indexed masked scatter-store routes at runtime counts
      `7,16,23` with noncontiguous/permuted unique indices, mixed masks,
      false-lane no-write preservation, unrelated-lane preservation, and
      tail/sentinel checks.
- [x] Real `ssh rvv` PASS evidence covers representative new routes and proves
      indexed store correctness, runtime n/AVL handling, false-lane no-write,
      noncontiguous/permuted index behavior, and tail/sentinel preservation.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, public exact intrinsic route authority,
      artifact-name authority, or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, truthful finish/
      archive state, clean git status, and one coherent commit complete if
      this task finishes.

## Non-Goals

- No redo of indexed masked gather-load, strided masked load/store, contiguous
  masked load/store, reductions, compare/select expansion, masked elementwise
  routes, or existing indexed gather/scatter routes except as regression
  anchors.
- No segmented movement, gather/scatter matrix, conversions, contractions,
  dtype/LMUL clone batches, high-level Linalg/Vector/StableHLO frontend
  lowering, source-front-door positive routes, dashboards, report-only work,
  helper-only cleanup, or future plugin work.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, exact intrinsic
  spelling authority, descriptor-driven direct C, or common EmitC/export
  semantic inference.
- No ad-hoc provider fallback around the closure gate.

## Validation Plan

1. Validate and maintain the Trellis task.
2. Build focused targets needed for `tcrv-opt`, `tcrv-translate`, RVV dialect,
   RVV plugin, construction, route provider, and target artifact checks.
3. Run focused RVV dialect/verifier tests for positive and negative indexed
   masked scatter-store structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-runs for representative explicit and pre-realized
   indexed masked scatter-store routes at counts `7,16,23`.
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
- Existing indexed scatter, indexed movement binding, masked strided store,
  computed masked indexed gather-load, and generated-bundle tests as directly
  relevant anchors.

## Definition Of Done

- One bounded indexed masked scatter-store route family is represented,
  verified, route-supported, materialized, dry-run validated, and
  runtime-validated on `ssh rvv` if correctness is claimed.
- Existing indexed gather/scatter, masked unit load/store, strided masked
  load/store, and indexed masked gather-load slices remain intact.
- The final report distinguishes route-supported, generated-bundle dry-run,
  and executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; if incomplete, the exact continuation point is recorded.
