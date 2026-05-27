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


## Session 273: Stage2 RVV plain segment2 executable artifact closure

**Date**: 2026-05-28
**Task**: Stage2 RVV plain segment2 executable artifact closure
**Branch**: `main`

### Summary

Revalidated selected-boundary plain segment2 generated-bundle executable artifacts on ssh rvv; no production repair needed.

### Main Changes

Created and completed the Trellis validation-closure task for Stage2 RVV plain segment2 selected-boundary executable artifacts. Current HEAD already satisfied the production path; no source code repair was needed.

### Main Evidence

- Dry-run selected-boundary evidence for `segment2_deinterleave_unit_store` and `segment2_interleave_unit_load` under `artifacts/tmp/stage2_plain_segment2_executable_closure/selected-boundary-dry-run`.
- Both dry-runs reported `route_entry_realization: false`, `pre_realized_body_consumed: true`, and selected-boundary materialization through `tcrv-materialize-selected-lowering-boundaries`.
- Direct pre-realized route-entry probes for both plain segment2 ops failed closed with the selected-boundary-only diagnostic.
- Non-dry-run generated-bundle evidence under `artifacts/tmp/stage2_plain_segment2_executable_closure/ssh-rvv-executable` compiled and ran on `ssh rvv` for both ops with counts `0,1,16,17,257`.
- Harness output passed field-order-distinguishing and tail-preservation checks for both deinterleave and interleave.

### Checks

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Focused lit filter for two plain segment2 dry-run tests, two direct fail-closed tests, and two pre-realized selected-body artifact tests: 6/6 passed.
- Computed-mask segment2 selected-boundary dry-run non-regression passed.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`: 405/405 passed after cleaning duplicate-run output races.

### Notes

A duplicated concurrent full-check attempt created generated-output races and false failures. After both duplicate jobs exited, generated lit `Output` directories were cleaned and a single full check passed 405/405. No `.trellis/spec/` update was needed because the relevant segment2 selected-boundary, artifact ABI, and mirror-authority contracts were already present.


### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] selected-boundary dry-run for both plain segment2 ops with counts `0,1,16,17,257`
- [OK] direct pre-realized route-entry fail-closed probes for both plain segment2 ops
- [OK] non-dry-run generated-bundle execution on `ssh rvv` for both plain segment2 ops with counts `0,1,16,17,257`
- [OK] computed-mask segment2 selected-boundary dry-run non-regression
- [OK] focused lit filter for the two dry-run tests, two direct fail-closed tests, and two pre-realized selected-body artifact tests: 6/6 passed
- [OK] bounded authority scan and `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 405/405 passed after cleaning duplicate-run output races

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 274: Stage2 RVV widening conversion executable artifacts

**Date**: 2026-05-28
**Task**: Stage2 RVV widening conversion executable artifacts
**Branch**: `main`

### Summary

Closed selected-boundary generated-bundle evidence for both widening conversion fixtures on ssh rvv and kept direct route-entry fail-closed diagnostics op-specific.

### Main Changes

- Reconciled dirty shared direct-pre-realized diagnostic diff: retained op-specific selected-boundary-only diagnostics because widening conversion fail-closed probes depend on the shared helper.
- Closed selected-boundary dry-run evidence for widen_i16_to_i32 and widen_i32_to_i64 with route_entry_realization=false and pre_realized_body_consumed=true.
- Ran ssh rvv generated-bundle compile/run/correctness for both widening conversions with counts 0,1,16,23,257; both reported PASS.
- Verified direct route-entry fail-closed probes, focused lit 25/405, RVV C++ plugin smoke, segment2/contraction non-regression dry-run, authority scan, git diff --check, and check-tianchenrv 405/405.
- Spec update review concluded no .trellis/spec change was needed; existing conversion/SEW and selected-boundary contracts already cover this round.


### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] selected-boundary dry-run for `widen_i16_to_i32` and `widen_i32_to_i64`
- [OK] direct pre-realized route-entry fail-closed probes for both widening conversion fixtures
- [OK] `ssh rvv` generated-bundle compile/run/correctness for both widening conversion fixtures with counts `0,1,16,23,257`
- [OK] focused lit filter: 25/405 selected tests passed
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] segment2 and contraction-family selected-body dry-run non-regression
- [OK] bounded authority scan and `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 405/405 passed

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 275: Stage2 RVV compare/select executable artifacts

**Date**: 2026-05-28
**Task**: Stage2 RVV compare/select executable artifacts
**Branch**: `main`

### Summary

Closed selected-boundary generated-bundle evidence for RVV compare/select and computed-mask select fixtures on ssh rvv; extended dry-run/direct-fail lit coverage for computed-mask signed-less-equal cases.

### Main Changes

- Created and archived the Trellis task for Stage2 RVV compare/select executable artifact closure.
- Extended computed-mask select generated-bundle dry-run coverage to include `computed_mask_select_sle`, `runtime-count 1`, `compare_select_predicate_boundary`, provider-derived predicate facts, selected-boundary materialization, ABI/header/object facts, and harness evidence.
- Extended direct pre-realized route-entry fail-closed coverage for `computed_mask_select_sle` while keeping compare/select selected-boundary-only.
- No production C++ or script change was required; the existing selected-boundary compiler path produced the executable artifacts honestly.

### Evidence

- Selected-boundary dry-run: `cmp_select`, `cmp_select_sle`, `computed_mask_select`, and `computed_mask_select_sle` with `route_entry_realization=false` and `pre_realized_body_consumed=true`.
- Direct route-entry negative probe failed closed with the selected-boundary-only diagnostic.
- `ssh rvv` generated-bundle compile/run/correctness passed for all four compare/select fixtures with counts `0,1,16,23,257`, covering true/false lanes, signed compare behavior, and computed-mask tail preservation.
- Widening conversion executable artifact non-regression passed for `widen_i16_to_i32` and `widen_i32_to_i64` on `ssh rvv` with counts `0,1,16,23,257`.

### Checks

- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Focused lit filter for compare/select generated-bundle and direct-route probes: 4/4 passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`: 405/405 passed.

### Notes

A spec update was not needed. Existing RVV plugin, EmitC route, and MLIR testing contracts already cover selected-boundary compare/select predicate boundaries, direct route-entry fail-closed behavior, artifact ABI facts, and mirror-only evidence roles.


### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] focused lit filter for compare/select generated-bundle and direct-route probes: 4/4 passed
- [OK] RVV extension plugin build and smoke test
- [OK] selected-boundary dry-run for compare/select and computed-mask select fixtures
- [OK] direct pre-realized route-entry fail-closed probes for compare/select fixtures
- [OK] `ssh rvv` generated-bundle execution for `cmp_select`, `cmp_select_sle`, `computed_mask_select`, and `computed_mask_select_sle` with counts `0,1,16,23,257`
- [OK] widening conversion executable artifact non-regression on `ssh rvv`
- [OK] bounded authority scan and `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 405/405 passed

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 276: Stage2 RVV runtime scalar cmp/select boundary

**Date**: 2026-05-28
**Task**: Stage2 RVV runtime scalar cmp/select boundary
**Branch**: `main`

### Summary

Closed runtime_scalar_cmp_select selected-boundary evidence with typed runtime scalar splat/compare/select extraction, two-RHS generated-bundle harness coverage, ssh rvv proof, and archived Trellis task.

### Main Changes

- Tightened `scripts/rvv_generated_bundle_abi_e2e.py` so
  `runtime_scalar_cmp_select` evidence now requires two RHS scalar values and
  explicitly extracts materialized `tcrv_rvv.splat`/`compare`/`select` facts.
- Added emitted RVV C++ boundary extraction for the runtime scalar splat,
  compare predicate, select layout, runtime AVL/VL control, and ABI order.
- Added selected-boundary dry-run and direct pre-realized route-entry
  fail-closed lit tests for `runtime_scalar_cmp_select`.
- Archived the Trellis task with production-owner proof, artifact paths, and
  check evidence.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] focused runtime-scalar lit filter: 5/5 passed
- [OK] selected-boundary dry-run for `runtime_scalar_cmp_select`
- [OK] direct pre-realized route-entry fail-closed dry-run
- [OK] `ssh rvv` generated-bundle compile/run/correctness for counts
  `0,1,16,23,257` and RHS scalars `-37,91`
- [OK] compare/select selected-boundary non-regression dry-run
- [OK] RVV extension plugin and selected-lowering-boundary C++ tests
- [OK] bounded authority scan and `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 407/407 passed

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 277: Stage2 RVV runtime-scalar dual cmp mask select boundary

**Date**: 2026-05-28
**Task**: Stage2 RVV runtime-scalar dual cmp mask select boundary
**Branch**: `main`

### Summary

Closed runtime_scalar_dual_cmp_mask_and_select generated-bundle/runtime ABI evidence with dual runtime scalar splats, two compares, mask-and/select facts, direct-route fail-closed lit, ssh rvv correctness, and check-tianchenrv 409/409.

### Main Changes

- Tightened `scripts/rvv_generated_bundle_abi_e2e.py` evidence extraction for
  `runtime_scalar_dual_cmp_mask_and_select`: two runtime scalar threshold ABI
  bindings, two scalar splats, two compares, mask-and composition, true/false
  select roles, store, runtime AVL/VL, provider metadata, artifact ABI mirrors,
  and generated harness coverage.
- Added dry-run lit coverage for the selected-boundary generated bundle and a
  direct pre-realized route-entry fail-closed test for the same op.
- Archived the Trellis task after recording PRD completion evidence.

### Git Commits

`this commit` - final round commit

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] focused lit filter: 7/7 passed
- [OK] selected-boundary dry-run for `runtime_scalar_dual_cmp_mask_and_select`
- [OK] direct pre-realized route-entry fail-closed dry-run
- [OK] non-regression dry-runs for `runtime_scalar_cmp_select` and
  compare/select selected-boundary paths
- [OK] `ssh rvv` generated-bundle correctness for counts `0,1,16,23,257` and
  threshold pairs `(-37,-37)`, `(-37,91)`, `(91,-37)`, `(91,91)`
- [OK] RVV extension plugin and selected-lowering-boundary C++ tests
- [OK] bounded authority scan and `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 409/409 passed

### Status

[OK] **Completed**

### Next Steps

- None - task complete
