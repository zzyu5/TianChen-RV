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


## Session 271: Stage2 RVV computed-masked segment2 selected realization migration

**Date**: 2026-05-27
**Task**: Stage2 RVV computed-masked segment2 selected realization migration
**Branch**: `main`

### Summary

Demoted computed-masked segment2 load/store/update direct route-entry shortcuts, preserved selected-boundary realization, verified focused C++/lit, ssh rvv evidence, authority scan, and check-tianchenrv 403/403.

### Main Changes

- `RVVSelectedBodyRealization.cpp`: removed computed-masked segment2
  load/store/update from the direct segment2 route-entry family registry while
  preserving the segment2 memory selected-body realization owner.
- `RVVEmitCRoutePlanning.cpp`: removed provider planning dependence on
  selected-body route-entry family owners so computed-mask segment2 selected
  routes are backed by typed route-family/materialization/runtime facts.
- `rvv_generated_bundle_abi_e2e.py`: bounded direct pre-realized route-entry
  mode to plain segment2 deinterleave/interleave fixtures and made migrated
  computed-mask segment2 direct requests fail closed.
- `RVVExtensionPluginTest.cpp` and script tests: updated registry expectations,
  selected-boundary producer coverage, direct fail-closed diagnostics, and
  `route_entry_realization: false` generated-bundle evidence.
- Specs/task notes: recorded computed-masked segment2 load/store/update as
  selected-boundary-only for direct route-entry purposes.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | `rvv: demote computed mask segment2 route entries` |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused lit from `build/test` for three computed-mask segment2
  direct-fail tests and three selected-boundary dry-run tests: 6/6 passed
- [OK] real `ssh rvv` generated-bundle execution for
  `computed_masked_segment2_load_unit_store`,
  `computed_masked_segment2_store_unit_load`, and
  `computed_masked_segment2_update_unit_load` counts `0,1,16,17,23,257`
- [OK] bounded touched-file authority scan and `git diff --check`
- [OK] Trellis context validation
- [OK] `cmake --build build --target check-tianchenrv -j2`: 403/403 passed

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 272: Stage2 RVV plain segment2 selected realization closure

**Date**: 2026-05-28
**Task**: Stage2 RVV plain segment2 selected realization closure
**Branch**: `main`

### Summary

Demoted plain segment2 direct pre-realized route-entry support, kept selected-boundary provider/artifact path, updated tests/spec, and verified ssh rvv counts 0,1,16,17,257 plus check-tianchenrv 405/405.

### Main Changes

- `RVVSelectedBodyRealization.cpp`: removed active segment2 direct
  route-entry family entries and made the `segment2 memory` realization owner
  selected-boundary-only.
- `rvv_generated_bundle_abi_e2e.py`: made
  `--direct-pre-realized-route-entry` fail closed for current pre-realized
  selected-body fixtures, including the plain segment2 pair.
- `RVVExtensionPluginTest.cpp` and script tests: updated direct-fail,
  selected-boundary, registry, and `route_entry_realization: false` coverage
  for plain segment2 and adjacent selected-boundary-only families.
- Specs/task notes: recorded that plain segment2 deinterleave/interleave are
  selected-boundary-only unless a future explicit owner task reintroduces
  direct route-entry support with matching facts and evidence.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | `rvv: demote plain segment2 route entries` |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] generated-bundle dry-runs for
  `segment2_deinterleave_unit_store` and `segment2_interleave_unit_load` with
  `route_entry_realization: false`
- [OK] direct pre-realized route-entry fail-closed probes for both plain
  segment2 ops
- [OK] computed-mask segment2 selected-boundary dry-run non-regression
- [OK] real `ssh rvv` generated-bundle execution for both plain segment2 ops,
  counts `0,1,16,17,257`
- [OK] bounded authority scan and `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 405/405 passed

### Status

[OK] **Completed**

### Next Steps

- None - task complete
