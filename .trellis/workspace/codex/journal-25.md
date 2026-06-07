# Journal - codex (Part 25)

> Continuation from `journal-24.md` (archived at ~2000 lines)
> Started: 2026-06-07

---



## Session 518: Stage2 RVV runtime-scalar indexed gather ABI

**Date**: 2026-06-07
**Task**: Stage2 RVV runtime-scalar indexed gather ABI
**Branch**: `main`

### Summary

Hardened runtime-scalar-cmp masked indexed gather-load target artifact ABI validation with runtime-scalar gather provider/candidate coverage, synced indexed-memory spec surface, and verified explicit/pre-realized generated bundles on ssh rvv.

### Main Changes

- Added runtime-scalar-cmp masked indexed scatter-store provider fail-closed coverage for stale inactive-lane contract, passthrough/store policy, index uniqueness, and indexed destination memory form.
- Added matching candidate mirror fail-closed coverage for stale `tcrv_rvv.inactive_lane_contract`, `tcrv_rvv.masked_passthrough_layout`, `tcrv_rvv.index_uniqueness`, and `tcrv_rvv.indexed_destination_memory_form`.
- Confirmed existing production validator already rejects the stale facts; no validator semantics or route production code changed.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Testing

- [OK] `git diff --check`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v . --filter runtime-scalar-cmp-masked-indexed-scatter-store` from `build/test`
- [OK] Explicit selected-body generated bundle on `ssh rvv` for counts `0,1,16,17,257`, `rhs_scalar=-37,91`, patterns `0,1`
- [OK] Pre-realized selected-body generated bundle on `ssh rvv` for the same counts, scalar values, and patterns

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 519: Stage2 RVV runtime-scalar indexed scatter ABI

**Date**: 2026-06-07
**Task**: Stage2 RVV runtime-scalar indexed scatter ABI
**Branch**: `main`

### Summary

Closed the runtime-scalar-cmp masked indexed scatter-store target artifact ABI boundary gap by adding manual provider/candidate validation coverage, stale runtime-scalar binding/producer/ABI fail-closed checks, focused dry-run tests, and ssh rvv evidence.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 520: Stage2 RVV runtime-scalar-cmp masked indexed gather-load ABI boundary

**Date**: 2026-06-07
**Task**: Stage2 RVV runtime-scalar-cmp masked indexed gather-load ABI boundary
**Branch**: `main`

### Summary

Added focused target artifact fail-closed coverage for runtime-scalar-cmp masked indexed gather-load provider and candidate stale operand binding, mask producer, and runtime ABI facts; verified target/plugin tests, lit dry-runs, and explicit/pre-realized generated bundles on ssh rvv.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 521: Stage2 RVV runtime-scalar-cmp masked indexed scatter-store ABI boundary

**Date**: 2026-06-07
**Task**: Stage2 RVV runtime-scalar-cmp masked indexed scatter-store ABI boundary
**Branch**: `main`

### Summary

Added runtime-scalar-cmp masked indexed scatter-store target artifact fail-closed coverage for stale inactive-lane, passthrough, index uniqueness, and indexed destination mirrors; verified target/plugin tests, lit dry-runs, and explicit/pre-realized ssh rvv generated bundles.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 522: Stage2 RVV composite masked indexed gather-MAcc-scatter fail-closed boundary

**Date**: 2026-06-07
**Task**: Stage2 RVV composite masked indexed gather-MAcc-scatter fail-closed boundary
**Branch**: `main`

### Summary

Added RVV plugin-owned fail-closed diagnostics and focused C++ coverage for explicit and pre-realized runtime-scalar computed-mask indexed gather-MAcc-scatter composite bodies; no executable artifact or ssh rvv claim is made until a composite owner/provider route exists.

### Main Changes

- Added an explicit realized-body route-analysis fail-closed seam for a runtime-scalar computed-mask indexed gather -> masked MAcc -> indexed scatter body in one `tcrv_rvv.with_vl` scope.
- Added a selected-body realization registry fail-closed seam for the corresponding pre-realized multi-family composite assembled from separate indexed gather, computed-mask MAcc, and indexed scatter body ops.
- Added focused C++ smoke coverage that proves both diagnostics name the missing composite selected-body realization / migrated statement-plan / provider contract rather than falling into generic single-route or multiple-body errors.
- Did not add a Common EmitC semantic branch, target artifact mirror, generated bundle, source-front-door route, or runtime correctness claim for the composite.

### Git Commits

pending-in-this-commit

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Bounded added-diff old-authority scan found no descriptor/direct-C/source-front-door/source-export/legacy `tcrv_rvv.i32_*`/`RVVI32M1`/`rvv-i32m1` matches.

### Status

[OK] **Completed**

### Next Steps

- Replace the fail-closed seam with a plugin-local composite owner that imports typed gather, MAcc, scatter, mask, index, accumulator, ABI, and AVL/VL facts into one realized body and one provider-built `TCRVEmitCLowerableRoute`, then add target artifact/generated-bundle/`ssh rvv` evidence.
