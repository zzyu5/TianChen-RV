# Journal - codex (Part 4)

> Continuation from `journal-3.md` (archived at ~2000 lines)
> Started: 2026-05-12

---



## Session 35: RVV selected emission descriptor exit

**Date**: 2026-05-12
**Task**: RVV selected emission descriptor exit
**Branch**: `main`

### Summary

Removed descriptor authority from RVV typed-source selected emission planning and kept typed body plus exec ABI as production authority.

### Main Changes

### Main Changes

- Rewired RVV selected emission/readiness so typed-source i32/i64 add/sub/mul paths no longer call descriptor-selected plan reconstruction.
- Added typed i64 microkernel control/dataflow validation in selected emission planning, matching the existing typed i32 selected body authority.
- Added post-typed-plan mirror checks for stale tcrv_rvv.lowering_descriptor and tcrv_rvv.element_count metadata.
- Added C++ coverage for descriptor-only i32/i64 selected emission/readiness fail-closed behavior and stale descriptor mirror rejection.
- Updated one lit diagnostic to the new typed-plan mirror mismatch wording.

### Testing

- artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test
- artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test
- artifacts/tmp/tianchenrv-build/bin/tianchenrv-runtime-abi-callable-plan-test
- artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test
- artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test
- focused lit filter: rvv-microkernel auto-materialization/family-mul/i32 descriptor mismatch/i64 vadd-vsub-vmul, 9/9 passed
- focused lit filter: rvv-microkernel-descriptor-element-mismatch-fails, 1/1 passed
- git diff --check
- cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2, 207/207 passed

### Status

[OK] Completed; no ssh rvv runtime, correctness, or performance claim was made.


### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
