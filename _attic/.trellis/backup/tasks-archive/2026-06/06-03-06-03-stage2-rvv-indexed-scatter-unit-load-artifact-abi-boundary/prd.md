# Stage2 RVV indexed scatter unit-load artifact ABI boundary

## Goal

Make one bounded Stage 2 RVV production compiler path real for
`indexed_scatter_unit_load`: a selected `tcrv.exec` RVV variant carries a typed
pre-realized indexed scatter body, the RVV plugin realizes it into typed
`tcrv_rvv` memory/control/move/indexed-store operations, the provider derives
route and artifact ABI facts from that body, target validation fails closed on
stale facts, generated bundle evidence records those facts, and real `ssh rvv`
correctness proves `dst[index[i]] = src[i]`.

## Why Now

The previous completed `indexed_gather_unit_store` round validated provider-owned
index-buffer gather semantics for `data,index,out,n`: unit-indexed source loads,
unit output stores, target artifact validation, generated-bundle evidence, and
real RVV correctness. This task handles the symmetric scatter route, where the
index buffer controls destination addresses rather than source addresses and
those indexed-store semantics must be carried by typed `tcrv_rvv`
body/config/runtime facts before the RVV provider builds a route.

## Current Direction Brief

Module owner:

- One bounded `indexed_scatter_unit_load` production compiler path centered on
  provider-owned indexed-store memory facts.

Required execution chain:

- selected `tcrv.exec` RVV variant
- `typed_indexed_scatter_memory_pre_realized_body`
- RVV plugin-local realization into `setvl` / `with_vl` / unit-stride `load` /
  `index_load` / `move` / `indexed_store`
- provider-built `TCRVEmitCLowerableRoute`
- target artifact validation
- generated bundle/header/object
- `ssh rvv` correctness

## Requirements

- The fact authority for this path must come from typed RVV body/config/runtime
  structure, not intrinsic spelling, route ids, artifact names, descriptors, C
  strings, test names, runtime counts, or mirror metadata.
- ABI order must be `src,index,dst,n`.
- Data vector configuration must be `SEW=32`, `LMUL=m1`.
- Index memory must use `index_load` with `index_eew=32`.
- Indexed destination store must use element offsets (`offset_unit=element`).
- Indexes must be marked unique (`index_uniqueness=unique`) so runtime evidence
  can unambiguously prove scatter placement.
- The typed compute operation is `tcrv_rvv.move`.
- Memory form is unit-stride source load plus indexed destination store.
- Indexed memory layout is
  `unit-stride-source-indexed-destination-index-runtime-abi`.
- Index source is `runtime_abi:index`.
- Source memory form is `unit-stride-load`.
- Indexed destination memory form is `indexed-store`.
- Destination memory form is `indexed-store`.
- Route operand binding plan and summary must expose ABI/header participation.
- Route family plan must identify the base indexed scatter memory movement route.
- Header and C type mapping facts must be provider-derived and validated.
- Target artifact validation must check the target leaf profile.
- Mirror fields must be explicitly labeled, e.g. `provider_supported_mirror`,
  and must not become authority.
- Fail closed on stale ABI order, index EEW, offset unit, index uniqueness,
  index source, indexed layout, route-family plan, provider mirror, header/type
  facts, target profile, and accidental gather/unit-store/strided fallback.

## Non-Goals

- No broad indexed gather/scatter matrix.
- No masked indexed paths, segment paths, dtype/LMUL/index-width clone batches,
  or high-level Linalg/Vector/frontend lowering.
- No source-front-door positive route.
- No redo of indexed gather, strided add, scalar broadcast, conversion,
  dot-reduce, macc, compare/select, or reduction paths.
- No common EmitC/export logic choosing RVV indexed semantics.
- No descriptor-driven computation or route authority.
- No new dtype-prefixed `tcrv_rvv.i32_*` helper family.

## Acceptance Criteria

- [x] RVV plugin/provider consumes provider-owned indexed scatter facts for ABI
      order, src/index/dst buffers, runtime AVL/VL, data SEW/LMUL,
      `index_eew=32`, `offset_unit=element`, `index_uniqueness=unique`, index
      source, source/indexed destination memory forms, indexed layout, route
      family, headers, C types, target profile, and provider mirror label.
- [x] Target artifact validation rejects stale/missing provider facts for ABI
      order, index EEW, offset unit, index uniqueness, index source, indexed
      layout, route-family plan, provider mirror, header/type facts, target
      profile, and accidental gather/unit-store/strided fallback.
- [x] REALIZED/PLAN/HEADER fixtures are tightened for both pre-realized and
      explicit selected-body `indexed_scatter_unit_load` forms.
- [x] Generated-bundle dry-run records provider-derived indexed scatter facts,
      base memory boundary evidence, operand/header/type summaries, and mirror
      labels.
- [x] Real `ssh rvv` generated-bundle correctness passes counts `0,1,16,17,257`
      with at least two unique index patterns including non-monotonic and
      output-order-distinguishing lanes, proving `dst[indices[i]]` receives
      `src[i]`, index offsets are element-based, source gaps and tail sentinels
      are preserved, and runtime `n`/AVL is honored.
- [x] Focused lit/FileCheck-equivalent tests and script tests cover stale fact
      rejection and no accidental old-authority/gather/unit-store/strided
      fallback.
- [x] Bounded old-authority scan over touched files finds no positive legacy
      `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, or
      descriptor/source-front-door route authority introduced by this task.
- [x] `git diff --check` passes.
- [x] Task is finished/archived and one coherent commit is created if complete.

## Technical Approach

Reuse the provider-owned base memory movement fact surface created for indexed
gather, but add the scatter-specific fact row and validation expectations. The
realization path must produce the explicit source load, index load, move, and
indexed store sequence; route planning and target validation must consume the
same provider-owned facts rather than reconstructing scatter semantics in common
EmitC or target-local constants. Generated-bundle dry-run and RVV runtime
evidence must distinguish scatter from gather by checking destination-indexed
writes, untouched source gaps, untouched destination tails, and unique index
patterns.

## Relevant Specs And Prior Art

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-indexed-gather-unit-store-artifact-abi-boundary/`
- `.trellis/workspace/codex/journal-21.md` Session 398

## Likely Implementation Areas

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/pre-realized-selected-body-artifact-indexed-scatter-unit-load.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-indexed-scatter-unit-load.mlir`
- Matching indexed scatter dry-run and direct pre-realized fail-closed script
  tests.

## Definition Of Done

- Production code implements the bounded path and fail-closed validation.
- Focused tests and generated-bundle evidence pass.
- Real RVV hardware correctness is either recorded, or an exact external blocker
  is reported with the task left open.
- Trellis task status, notes, and archive state are truthful.
- Worktree is clean after a coherent commit when complete.

## Completion Notes

- Tightened the pre-realized selected-body indexed scatter fixture so PLAN and
  HEADER checks include base-memory route-family plan, target profile, provider
  mirror, required headers, and C type mapping facts.
- Added C++ target artifact fail-closed coverage for stale indexed scatter ABI
  order, route-family plan, index source, indexed layout, index EEW, offset
  unit, index uniqueness, source/destination memory forms, indexed destination
  form, gather residue, target profile, provider mirror, header facts, type
  facts, binding summary, and candidate mirror metadata.
- Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the durable
  indexed scatter/unit load base-memory fact-surface contract.
- Verified generated bundle dry-runs and real `ssh rvv` correctness for both
  explicit and pre-realized selected-body indexed scatter/unit load with counts
  `0,1,16,17,257` and two unique index patterns.
