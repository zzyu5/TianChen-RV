# Journal - codex (Part 13)

> Continuation from `journal-12.md` (archived at ~2000 lines)
> Started: 2026-05-22

---



## Session 155: Stage2 RVV closure-gated strided masked load movement

**Date**: 2026-05-22
**Task**: Stage2 RVV closure-gated strided masked load movement
**Branch**: `main`

### Summary

Added closure-gated computed-mask byte-strided masked-load to unit-store RVV route support with typed tcrv_rvv.masked_strided_load body facts, RouteOperandBindingPlan materialization, explicit/pre-realized fixtures, generated-bundle dry-runs, and real ssh rvv evidence.

### Main Changes

- Added `tcrv_rvv.masked_strided_load` and the bounded
  `tcrv_rvv.typed_computed_mask_strided_load_pre_realized_body` surface for
  computed-mask byte-strided loads to unit-stride destination stores.
- Rewired RVV selected-body realization, route planning, route provider, route
  operand binding closure, construction metadata, target artifact fixtures, and
  generated-bundle ABI/runtime harness support for
  `computed_masked_strided_load_unit_store`.
- Added explicit and pre-realized route fixtures plus fail-closed negative
  coverage for missing/wrong typed body facts, stale or wrong binding plans,
  mirror-only authority, materialized-use mismatch, old masked_move fallback,
  route-id/helper fallback, source/front-door/descriptor/direct-C authority,
  and common/export semantic inference.

### Git Commits

Final round commit is created after task archive in the same Codex turn.

### Testing

- [OK] Focused RVV build targets for `tcrv-opt`, `tcrv-translate`, RVV plugin,
  construction protocol, and target artifact export tests.
- [OK] RVV dialect, RVV plugin, construction protocol, and target artifact
  export binaries.
- [OK] Positive explicit and pre-realized computed masked strided-load target
  lit fixtures plus bounded negative fixture.
- [OK] `rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Generated-bundle dry-runs for explicit and pre-realized
  `computed_masked_strided_load_unit_store`, counts `7,16,23`, source byte
  strides `4,8,12`.
- [OK] Real `ssh rvv` generated-bundle runs for explicit and pre-realized
  `computed_masked_strided_load_unit_store`, counts `7,16,23`, source byte
  strides `4,8,12`, proving active strided loads, inactive passthrough,
  untouched source gaps, runtime n/AVL behavior, and tail/sentinel
  preservation.
- [OK] Focused conversion lit self-repair after diagnostic surface growth.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 300/300 passed.
- [OK] `git diff --check`.
- [OK] Diff-level authority scan found no new positive legacy/source/descriptor
  route authority; the exact intrinsic diff hit is the provider-owned target
  leaf selected after typed closure, not route authority.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 156: Stage2 RVV closure-gated indexed masked gather-load movement

**Date**: 2026-05-22
**Task**: Stage2 RVV closure-gated indexed masked gather-load movement
**Branch**: `main`

### Summary

Added typed closure-gated computed-mask indexed gather-load RVV route support with explicit/pre-realized fixtures, generated-bundle dry-runs, and real ssh rvv evidence.

### Main Changes

- Added the typed `tcrv_rvv.masked_indexed_load` and bounded pre-realized computed-mask indexed gather body surface for signed i32 / SEW32 / LMUL m1 gather-load to unit-store behavior.
- Rewired selected-body realization, config ABI ordering, construction protocol metadata, RVV route planning, route provider materialization, binding-plan closure, target artifact mirrors, and generated-bundle ABI/runtime harness support for `computed_masked_indexed_gather_load_unit_store`.
- Added explicit and pre-realized route fixtures, dialect dataflow coverage, generated-bundle dry-run tests, and regression updates for generic RVV allowlist and fail-closed diagnostic surfaces.
- Checks passed: focused dialect/target lit fixtures, script py_compile, generated-bundle self-test, generated-bundle dry-runs for counts `7,16,23`, real `ssh rvv` explicit and pre-realized runs for counts `7,16,23`, focused self-repair reruns, `git diff --check`, and `cmake --build build --target check-tianchenrv -j2` with `305/305` tests.
- Spec update review: no `.trellis/spec/**` change was needed because the existing RVV plugin, EmitC route, and MLIR testing contracts already define the long-term boundary; this round added bounded route coverage inside that boundary.
- Authority scan: no new positive legacy i32/source-front-door/descriptor/direct-C/common-export route authority was introduced. The exact intrinsic occurrence is provider-owned target leaf spelling after typed closure.


### Git Commits

Final round commit is created after task archive in the same Codex turn.

### Testing

- [OK] Focused RVV dialect/target lit fixtures for explicit and pre-realized
  computed-mask indexed gather-load routes.
- [OK] Script validation: `python3 -m py_compile
  scripts/rvv_generated_bundle_abi_e2e.py` and
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Generated-bundle dry-runs for explicit and pre-realized
  `computed_masked_indexed_gather_load_unit_store`, counts `7,16,23`.
- [OK] Real `ssh rvv` generated-bundle runs for explicit and pre-realized
  `computed_masked_indexed_gather_load_unit_store`, counts `7,16,23`,
  proving active indexed loads, inactive passthrough, noncontiguous/permuted
  index behavior, runtime n/AVL handling, source preservation, and
  tail/sentinel preservation.
- [OK] Focused self-repair reruns for FileCheck metadata ordering and route
  allowlist diagnostic growth.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 305/305 passed.
- [OK] `git diff --check`.
- [OK] Diff-level authority scan found no new positive legacy/source/descriptor
  or common-export route authority; the exact intrinsic diff hit is the
  provider-owned target leaf selected after typed closure.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 157: Stage2 RVV closure-gated indexed masked scatter-store movement

**Date**: 2026-05-22
**Task**: Stage2 RVV closure-gated indexed masked scatter-store movement
**Branch**: `main`

### Summary

Added closure-gated computed-mask indexed scatter-store RVV route support with explicit/pre-realized fixtures, generated-bundle dry-runs, and real ssh rvv evidence.

### Main Changes

- Added the bounded `computed_masked_indexed_scatter_store_unit_load` route family on typed RVV body facts, including `tcrv_rvv.masked_indexed_store` and `tcrv_rvv.typed_computed_mask_indexed_scatter_pre_realized_body`.
- Rewired RVV config ABI ordering, selected-body realization, construction protocol metadata, route planning, route provider materialization, closure-gated operand binding, target artifact mirrors, and generated-bundle ABI/runtime harness support for indexed masked scatter-store movement.
- The route carries compare lhs/rhs, source payload, index vector, destination memory, runtime n/AVL, SEW/LMUL/policy, unique index policy, inactive-lane no-write policy, and tail preservation through the RVV dialect/plugin-owned path.
- Added explicit and pre-realized target fixtures plus dialect dataflow/negative coverage; updated focused conversion negative allowlists for the new generic RVV op surface.
- Self-repair performed: excluded indexed scatter from the computed-mask masked_load provider branch, added compare predicate metadata for target artifact validation, and repaired conversion FileCheck allowlists after the new op surface changed diagnostics.
- Checks passed: manual FileCheck for explicit/pre-realized PLAN/REALIZED/HEADER fixtures and dialect negative verifier coverage; `cmake --build build -j2`; construction/RVV extension/target artifact C++ tests; `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`; generated-bundle self-test; explicit and pre-realized dry-runs for counts `7,16,23`; real `ssh rvv` explicit and pre-realized PASS for counts `7,16,23`; `cmake --build build --target check-tianchenrv -j2` with `308/308` tests; `git diff --check`.
- Runtime evidence proved active indexed stores, mixed true/false mask behavior, inactive-lane no-write preservation, unrelated/tail sentinel preservation, runtime n/AVL variation, and noncontiguous/permuted unique index behavior.
- Authority scan found no new positive legacy/source/descriptor/common-export route authority; staged hits are negative-boundary wording, one fail-closed stale route-id input, and the provider-owned target leaf selected after typed closure.
- Spec update review: no `.trellis/spec/**` change was needed because existing RVV plugin, EmitC route, and MLIR testing contracts already define the long-term boundary; this round added bounded route coverage inside that boundary.
- Segmented movement, broad gather/scatter matrices, additional dtype/LMUL clones, source-front-door routes, and Stage2 class expansion were intentionally not converted in this bounded owner.

Final round commit is created after task archive in the same Codex turn.


### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
