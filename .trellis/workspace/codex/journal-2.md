# Journal - codex (Part 2)

> Continuation from `journal-1.md` (archived at ~2000 lines)
> Started: 2026-05-10

---



## Session 16: Target-owned artifact route translation registration

**Date**: 2026-05-10
**Task**: Target-owned artifact route translation registration
**Branch**: `main`

### Summary

Moved tcrv-translate RVV direct and RVV+scalar route-family helper command registration behind a target-owned translate route registry, added built-in route contribution tests, updated the durable route-registration spec, archived the Trellis task, and passed check-tianchenrv 193/193.

### Main Changes

- Added `TargetTranslateRoute` / `TargetTranslateRouteRegistry` plus `registerBuiltinTargetTranslateRoutes`.
- Moved RVV direct binary microkernel source/header/object translate route-family contribution into RVV target support.
- Moved RVV+scalar dispatch source/header/object/self-check route-family contribution into target support.
- Simplified `tcrv-translate` to one generic built-in target translate route registration loop while leaving RVV smoke probe and standalone RVV self-check C as direct legacy helpers.
- Added C++ registry coverage for malformed routes, duplicate route ids, representative direct RVV routes, representative RVV+scalar dispatch routes, and legacy helper exclusion.
- Updated `.trellis/spec/lowering-runtime/emission-runtime-contract.md` with the built-in target translate route registration boundary.

### Testing

- [OK] `git diff --check`
- [OK] focused build: `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- [OK] focused lit filter: `rvv-microkernel-i64-vadd|rvv-microkernel-header|rvv-scalar-i32-vadd-dispatch-generic-route|rvv-scalar-i64-vsub-dispatch-generic-route` passed 6/6
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`: 193/193
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-10-target-owned-artifact-route-translation-registration`

### Status

[OK] **Completed and archived**
