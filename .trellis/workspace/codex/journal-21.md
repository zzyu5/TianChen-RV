# Journal - codex (Part 21)

> Continuation from `journal-20.md` (archived at ~2000 lines)
> Started: 2026-06-02

---



## Session 386: Stage2 RVV computed-masked segment2 store artifact ABI boundary

**Date**: 2026-06-02
**Task**: Stage2 RVV computed-masked segment2 store artifact ABI boundary
**Branch**: `main`

### Summary

Tightened computed-mask segment2 store target artifact provider/candidate validation, expanded store dry-run facts, proved generated bundle correctness on ssh rvv, archived Trellis task.

### Main Changes

- Generalized computed-mask segment2 target binding-summary validation so
  load, store, and update routes all structurally require provider-owned
  logical ABI operands with `abi|hdr` participation markers.
- Tightened computed-mask segment2 store/update provider validation to reject
  stale segment-load facts before artifact export.
- Added store-specific target artifact fail-closed regressions for provider
  facts, candidate mirrors, route statement operands, tuple/store facts, mask
  facts, header/type summaries, field roles/forms, and inactive-lane no-write
  contract.
- Expanded pre-realized computed-mask segment2 store dry-run FileCheck coverage
  over the provider-derived fact surface and recorded the matching RVV plugin
  spec contract.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- [OK] `rtk ./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter computed-masked-segment2-store` from `build/test` (5 passed, 470 excluded)
- [OK] Generated-bundle dry-run for pre-realized `computed_masked_segment2_store_unit_load`
- [OK] Real `ssh rvv` generated-bundle correctness for counts `0,1,7,16,23,257` and patterns `0,1`
- [OK] `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `rtk git diff --check`
- [OK] Bounded changed-line authority scan found no new legacy i32m1, descriptor, source-front-door, direct-C/source-export, or source-export route-authority markers

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 388: Stage2 RVV runtime-scalar cmp masked standalone reduce-min artifact ABI boundary

**Date**: 2026-06-02
**Task**: Stage2 RVV runtime-scalar cmp masked standalone reduce-min artifact ABI boundary
**Branch**: `main`

### Summary

Tightened runtime-scalar computed-mask standalone reduce-min provider/target
artifact facts, expanded reduce-min fail-closed coverage, tightened generated
bundle dry-run facts, and proved the generated artifact on real `ssh rvv`.

### Main Changes

- Extended runtime-scalar computed-mask standalone reduction route facts with
  canonical compare predicate, mask role/source/form, computed-mask
  accumulation plan/suffix/contracts, result layout/store VL, and scalar-result
  runtime boundary.
- Made target artifact validation reject stale reduce-min provider facts for
  compare predicate, mask facts, scalar seed/result layout, accumulation
  contracts, and scalar lane0 result boundary.
- Added reduce-min provider and candidate mirror regressions for stale
  predicate/mask, header/type, reduction intrinsic, scalar-result boundary, and
  accumulation facts.
- Tightened the pre-realized generated-bundle dry-run checks to record
  provider-derived target leaf/profile, headers/types, channel types,
  intrinsics, accumulation facts, neutral inactive lanes, and scalar lane0
  result boundary.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | `rvv: validate runtime scalar reduce min artifact facts` |

### Testing

- [OK] `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- [OK] `rtk ./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-runtime-scalar-cmp-masked-standalone-reduce-min` from `build/test` (2 passed, 473 excluded)
- [OK] Direct dry-run generated-bundle evidence at `artifacts/tmp/stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-min/dry-run`
- [OK] Real `ssh rvv` generated-bundle correctness at `artifacts/tmp/stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-min/ssh-rvv` for counts `0,1,16,23,257`, rhs scalars `-37,91`, seeds `-11,17`, patterns `0,1`, source preservation, and tail preservation
- [OK] `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `rtk git diff --check`
- [OK] Bounded changed-diff authority scan found no new positive legacy i32m1, descriptor, source-front-door, direct-C/source-export, or source-export route authority

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 387: Stage2 RVV computed-masked segment2 update artifact ABI boundary

**Date**: 2026-06-02
**Task**: Stage2 RVV computed-masked segment2 update artifact ABI boundary
**Branch**: `main`

### Summary

Tightened computed-mask segment2 update target provider fact validation, dry-run evidence checks, ssh rvv generated-bundle correctness, and archived the Trellis task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
