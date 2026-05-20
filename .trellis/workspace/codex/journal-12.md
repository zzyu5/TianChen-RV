# Journal - codex (Part 12)

> Continuation from `journal-11.md` (archived at ~2000 lines)
> Started: 2026-05-20

---



## Session 141: Stage2 RVV masked memory movement executable slice

**Date**: 2026-05-20
**Task**: Stage2 RVV masked memory movement executable slice
**Branch**: `main`

### Summary

Implemented bounded i32 SEW32 LMUL m1 masked unit-stride memory movement with typed source/mask/destination roles, inactive-lane preservation, provider-derived route planning, generated artifacts, and ssh rvv correctness evidence.

### Main Changes

- Added typed masked memory RVV surface: `typed_masked_memory_pre_realized_body`, `mask_load`, and `masked_move`, with verifier diagnostics for mask role, mask memory form, inactive-lane policy, VL consistency, and old-destination preservation.
- Added `MaskInputBuffer` runtime ABI role and selected-body runtime ABI order `src,mask,dst,n`.
- Extended RVV selected-body realization to materialize pre-realized masked memory into `setvl`, source load, mask load, old destination load, masked move, and store.
- Extended RVV EmitC route planning/provider to derive mask load, compare-mask, masked merge, unit-load/store leaves, ABI metadata, inactive-lane contract, and target artifact facts from typed structure.
- Extended construction protocol, target artifact export tests, generated-bundle dry-run tests, and script harness checks for explicit and pre-realized `masked_unit_load_store`.
- Runtime evidence: explicit and pre-realized generated bundles passed `ssh rvv` for counts `7,16,23`, proving active masked lanes update while inactive lanes and tail sentinels are preserved.
- Checks: focused lit 11/11, C++ unit tests, script py_compile/self-test, generated-bundle dry-runs, `cmake --build build --target check-tianchenrv` 211/211, `git diff --check`, and active-authority scan.
- Self-repair: synchronized C++ construction-protocol tests and stale legacy fail-closed FileCheck strings after adding `mask_load` and `masked_move` to the generic Stage2 op list.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 142: Stage2 RVV computed-mask memory dataflow executable slice

**Date**: 2026-05-20
**Task**: Stage2 RVV computed-mask memory dataflow executable slice
**Branch**: `main`

### Summary

Implemented one bounded i32 SEW32 LMUL m1 computed-mask memory dataflow slice:
`dst[i] = cmp_lhs[i] < cmp_rhs[i] ? src[i] : old_dst[i]`. The selected RVV
body now carries compare lhs/rhs, active source, old/result destination,
runtime n/AVL, predicate, mask role/source/form, inactive-lane policy, and
typed RVV config through plugin-local realization, route planning/provider,
generated artifact export, and ssh rvv correctness evidence.

### Main Changes

- Added `source-input-buffer` runtime ABI role and a five-parameter computed-mask memory ABI order `cmp_lhs,cmp_rhs,src,dst,n`.
- Added `typed_computed_mask_memory_pre_realized_body` and verifier coverage for predicate `slt`, mask source `compare-produced-mask-same-vl-scope`, old-destination preservation, typed config, ABI roles, and stale authority metadata rejection.
- Extended RVV selected-body realization to materialize the pre-realized body into `setvl`, compare lhs/rhs loads, active source load, old destination load, `tcrv_rvv.compare`, `tcrv_rvv.masked_move`, and store.
- Extended route planning/provider, construction protocol, metadata, header ABI, and generated-bundle harness support for `computed_masked_unit_load_store`.
- Added positive and fail-closed MLIR tests for computed-mask materialization/route planning and updated masked memory diagnostics to allow compare-produced masks.
- Runtime evidence: pre-realized computed-mask generated bundle passed `ssh rvv` for counts `7,16,23`, proving compare-active lanes update while compare-false lanes and tail sentinels preserve old destination.
- Checks: script py_compile/self-test, focused dialect/target FileCheck, C++ unit tests, generated-bundle dry-run, real ssh rvv generated-bundle run, `git diff --check`, `check-tianchenrv` 214/214, and active-authority scan.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] focused MLIR positive/fail-closed checks for computed-mask memory
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_masked_unit_load_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --dry-run ...`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_masked_unit_load_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --ssh-target rvv ...`
- [OK] `cmake --build build --target check-tianchenrv -j2`
- [OK] active-authority scan: no new `RVVI32M1`, `rvv-i32m1`, positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed, descriptor, direct-C, or source-export route authority.

### Status

[OK] **Completed**

### Next Steps

- None - task complete
