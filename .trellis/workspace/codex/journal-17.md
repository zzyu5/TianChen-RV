# Journal - codex (Part 17)

> Continuation from `journal-16.md` (archived at ~1976 lines)
> Started: 2026-05-27

---

## Session 270: Stage2 RVV standalone reduce selected realization migration

**Date**: 2026-05-27
**Task**: Stage2 RVV standalone reduce selected realization migration
**Branch**: `main`

### Summary

Finished the dirty standalone_reduce_add selected-body realization migration by
demoting direct pre-realized route-entry authority, preserving the RVV
plugin-local selected-boundary producer path, verifying generated-bundle and
real `ssh rvv` evidence, and preparing the Trellis task for archive and commit.

### Main Changes

- `RVVSelectedBodyRealization.cpp`: removed standalone reduction from direct
  pre-realized route-entry owner eligibility while preserving the standalone
  reduction selected-body realization owner.
- `rvv_generated_bundle_abi_e2e.py`: removed `standalone_reduce_add` from the
  direct pre-realized route-entry allowlist and updated the CLI diagnostic/help
  text to keep direct route-entry bounded to segment2 families.
- `RVVExtensionPluginTest.cpp`: tightened owner-registry and production
  route-path coverage so standalone reduction is selected-boundary producer
  eligible but direct route-entry fail-closed.
- Script/lit coverage: changed standalone_reduce_add dry-run coverage to
  require `route_entry_realization: false`, added direct fail-closed coverage,
  and retained a segment2 direct route-entry non-regression.
- Specs: recorded that standalone reduction belongs to the selected-body
  realization owner but remains selected-boundary-only until a future explicit
  direct route-entry owner task adds matching provider facts and evidence.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | `rvv: demote standalone reduce route entry` |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] manual direct pre-realized route-entry fail-closed probe for
  `standalone_reduce_add`
- [OK] manual pre-realized selected-boundary dry-run for
  `standalone_reduce_add`, with `route_entry_realization: false`
- [OK] focused lit from `build/test` for standalone reduction selected-boundary
  dry-run, standalone reduction direct fail-closed, selected-body target
  artifact, and a direct segment2 non-regression: 4/4 passed
- [OK] real `ssh rvv` generated-bundle execution for
  `standalone_reduce_add` counts `0,1,7,16,23,257` and seeds `-11,17`
- [OK] bounded touched-file authority scan, `git diff --check`, and Trellis
  context validation
- [OK] `cmake --build build --target check-tianchenrv -j2`: 401/401 passed

### Status

[OK] **Completed**

### Next Steps

- None - task complete
