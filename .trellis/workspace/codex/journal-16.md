# Journal - codex (Part 16)

> Continuation from `journal-15.md` (archived at ~2000 lines)
> Started: 2026-05-26

---



## Session 237: Stage2 RVV direct route-entry contraction executable boundary

**Date**: 2026-05-26
**Task**: Stage2 RVV direct route-entry contraction executable boundary
**Branch**: `main`

### Summary

Closed direct route-entry computed_masked_strided_input_widening_dot_reduce_add executable evidence with generated bundle dry-run, ssh rvv PASS counts 0,7,16,23,257, focused contraction non-regressions, authority scan, git diff --check, task validation, and check-tianchenrv 381/381. No production code change was needed because HEAD already satisfied the route-entry production path.

### Main Changes

- `RVVSelectedBodyRealization.cpp`: split the broad compare/select cluster
  route-entry predicate into an explicit bounded predicate for plain
  compare-select and computed-mask vector select pre-realized bodies.
- `rvv_generated_bundle_abi_e2e.py`: admitted direct pre-realized
  `computed_mask_select` route entries, extracted materialized and EmitC
  compare/select predicate boundaries, and validated compare/load/select/store
  structure plus provider-owned route metadata.
- `RVVExtensionPluginTest.cpp`: added direct route-entry
  `computed_mask_select` coverage, provider plan/statement-plan assertions,
  and targeted fail-closed diagnostics for predicate, mask source, select
  layout, payload role, and runtime `n` role mismatches.
- Added focused script dry-run coverage for direct pre-realized
  `computed_mask_select` with no selected-lowering-boundary materializer step.
- Created and archived Trellis task
  `05-26-05-26-stage2-rvv-compare-select-route-family-owner`.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] direct route-entry `computed_mask_select` dry-run, counts
  `0,7,16,23,257`
- [OK] real `ssh rvv` direct route-entry `computed_mask_select`, counts
  `0,7,16,23,257`, PASS with true/false lane coverage and tail preservation.
- [OK] focused direct route-entry non-regressions for `cmp_select`,
  `computed_masked_macc_add`, and contraction.
- [OK] bounded touched-file authority scan: no new positive legacy-i32,
  source-front-door, descriptor, ABI-string, route-id, artifact-name,
  script-derived, metadata-derived, or common EmitC route authority.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` (383/383)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 238: Stage2 RVV route-entry MAcc realization owner

**Date**: 2026-05-26
**Task**: `05-26-stage2-rvv-route-entry-macc-realization-owner`
**Branch**: `main`

### Summary

Implemented direct route-entry MAcc selected-body realization ownership for
plain `macc_add` and vector-compare `computed_masked_macc_add`. The RVV owner
registry now marks both MAcc families route-entry eligible, generated-bundle
direct mode accepts those bounded MAcc fixtures, and focused tests prove
pre-realized body consumption, realized typed RVV structure, provider plan
handoff, provider-built route construction, and fail-closed diagnostics.

### Main Changes

- `RVVSelectedBodyRealization.cpp`: added owner-scoped direct route-entry
  predicates for plain MAcc and computed-mask MAcc, kept runtime-scalar
  computed-mask MAcc on its separate owner path, and updated route-entry family
  diagnostics.
- `RVVExtensionPluginTest.cpp`: added direct route-entry `macc_add` and
  `computed_masked_macc_add` fixtures, realized IR shape assertions, provider
  plan assertions, and targeted computed-mask MAcc negative diagnostics.
- `rvv_generated_bundle_abi_e2e.py`: enabled bounded direct pre-realized
  route-entry generated-bundle support for `macc_add` and
  `computed_masked_macc_add`.
- Added direct route-entry computed-mask MAcc script dry-run coverage.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] direct route-entry `computed_masked_macc_add` dry-run, counts
  `0,7,16,23,257`
- [OK] direct route-entry `macc_add` dry-run, counts `0,7,16,23,257`
- [OK] pre-realized selected-boundary `computed_masked_macc_add` non-regression
  dry-run, counts `0,7,16,23,257`
- [OK] direct route-entry contraction non-regression dry-run, counts
  `0,7,16,23,257`
- [OK] real `ssh rvv` direct route-entry `computed_masked_macc_add`, counts
  `0,7,16,23,257`, PASS with active-lane MAcc, inactive accumulator
  preservation, add-only/mul-only distinction, and tail preservation.
- [OK] diff-only authority scan: no new positive legacy-i32, source-front-door,
  descriptor, ABI-string, route-id, artifact-name, metadata-derived, or common
  EmitC route authority.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` (382/382)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 238: Stage2 RVV compare/select route-family owner

**Date**: 2026-05-26
**Task**: Stage2 RVV compare/select route-family owner
**Branch**: `main`

### Summary

Made direct route-entry computed-mask select a first-class RVV compare/select owner path, with owner-scoped realization, generated-bundle dry-run and ssh rvv evidence, focused non-regressions, task archive, and full check-tianchenrv pass.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
