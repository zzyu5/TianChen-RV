# Stage2 RVV indexed gather unit-store artifact ABI boundary

## Goal

Make one bounded Stage 2 RVV production compiler path real for
`indexed_gather_unit_store`: a selected `tcrv.exec` RVV variant carries a typed
pre-realized indexed gather body, the RVV plugin realizes it into typed
`tcrv_rvv` memory/control/move/store operations, the provider derives route and
artifact ABI facts from that body, target validation fails closed on stale
facts, generated bundle evidence records those facts, and real `ssh rvv`
correctness proves `out[i] = data[index[i]]`.

## Why Now

The previous completed strided-add round validated independent runtime
`lhs_stride`, `rhs_stride`, and `out_stride` ABI facts through RVV provider-owned
route planning, target artifacts, generated bundle evidence, and real RVV
hardware correctness. This task moves from affine runtime stride memory to
indexed gather memory movement, where the index buffer and element-offset
semantics must be carried by typed `tcrv_rvv` body/config/runtime facts before
route construction.

## Current Direction Brief

Module owner:

- One bounded `indexed_gather_unit_store` production compiler path centered on
  provider-owned indexed memory facts.

Required execution chain:

- selected `tcrv.exec` RVV variant
- `typed_indexed_gather_memory_pre_realized_body`
- RVV plugin-local realization into `setvl` / `with_vl` / `index_load` /
  `indexed_load` / `move` / unit-stride `store`
- provider-built `TCRVEmitCLowerableRoute`
- target artifact validation
- generated bundle/header/object
- `ssh rvv` correctness

## Requirements

- The fact authority for this path must come from typed RVV body/config/runtime
  structure, not intrinsic spelling, route ids, artifact names, descriptors, C
  strings, test names, runtime counts, or mirror metadata.
- ABI order must be `data,index,out,n`.
- Data vector configuration must be `SEW=32`, `LMUL=m1`.
- Index memory must use `index_load` with `index_eew=32`.
- Indexed data load must use element offsets (`offset_unit=element`).
- The typed compute operation is `tcrv_rvv.move`.
- Memory form is indexed-load plus unit-stride output store.
- Indexed memory layout is
  `element-indexed-data-index-unit-stride-output-runtime-abi`.
- Index source is `runtime_abi:index`.
- Route operand binding plan and summary must expose ABI/header participation.
- Route family plan must identify the base indexed memory movement route.
- Header and C type mapping facts must be provider-derived and validated.
- Target artifact validation must check the target leaf profile.
- Mirror fields must be explicitly labeled, e.g. `provider_supported_mirror`,
  and must not become authority.
- Fail closed on stale ABI order, index EEW, offset unit, index source, indexed
  layout, route-family plan, provider mirror, header/type facts, target profile,
  and accidental unit-load or strided residue.

## Non-Goals

- No broad indexed gather/scatter matrix.
- No masked indexed path, segment path, dtype/LMUL/index-width clone batch, or
  high-level Linalg/Vector/frontend lowering.
- No source-front-door positive route.
- No redo of strided add, scalar broadcast, conversion, dot-reduce, macc,
  compare/select, or reduction paths.
- No common EmitC/export logic choosing RVV indexed semantics.
- No descriptor-driven computation or route authority.
- No new dtype-prefixed `tcrv_rvv.i32_*` helper family.

## Acceptance Criteria

- [ ] RVV plugin/provider consumes provider-owned indexed gather facts for
      ABI order, data/index/out buffers, runtime AVL/VL, data SEW/LMUL,
      `index_eew=32`, `offset_unit=element`, index source, memory forms, layout,
      route family, headers, C types, target profile, and provider mirror label.
- [ ] Target artifact validation rejects stale/missing provider facts for ABI
      order, index EEW, offset unit, index source, indexed layout, route-family
      plan, provider mirror, header/type facts, target profile, and accidental
      strided/unit-load fallback.
- [ ] REALIZED/PLAN/HEADER fixtures are tightened for both pre-realized and
      explicit selected-body `indexed_gather_unit_store` forms.
- [ ] Generated-bundle dry-run records provider-derived indexed memory facts.
- [ ] Real `ssh rvv` generated-bundle correctness passes counts `0,1,16,17,257`
      with at least two index patterns including non-monotonic and
      output-order-distinguishing lanes, proving element-based offsets,
      unit-store output, preserved tail sentinels, and honored runtime `n`/AVL.
- [ ] Focused lit/FileCheck-equivalent tests and script tests cover stale fact
      rejection and no accidental old-authority/strided/unit-load fallback.
- [ ] Bounded old-authority scan over touched files finds no positive legacy
      `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, or
      descriptor/source-front-door route authority introduced by this task.
- [ ] `git diff --check` passes.
- [ ] Task is finished/archived and one coherent commit is created if complete.

## Relevant Specs And Prior Art

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-strided-add-artifact-abi-boundary/`

## Likely Implementation Areas

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- RVV provider/planning route-family implementation
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/pre-realized-selected-body-artifact-indexed-gather-unit-store.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-indexed-gather-unit-store.mlir`
- Matching indexed gather dry-run and direct pre-realized fail-closed script
  tests.

## Definition Of Done

- Production code implements the bounded path and fail-closed validation.
- Focused tests and generated-bundle evidence pass.
- Real RVV hardware correctness is either recorded, or an exact external blocker
  is reported with the task left open.
- Trellis task status, notes, and archive state are truthful.
- Worktree is clean after a coherent commit when complete.
