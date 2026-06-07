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
