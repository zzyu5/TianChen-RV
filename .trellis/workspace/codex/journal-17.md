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


## Session 286: Stage2 RVV typed runtime-scalar compare-masked memory route-family derivation

**Date**: 2026-05-28
**Task**: Stage2 RVV typed runtime-scalar compare-masked memory route-family derivation
**Branch**: `main`

### Summary

Completed typed runtime_scalar_cmp_masked_store and runtime_scalar_cmp_masked_load_store derivation for baseline SEW32 LMUL m1, SEW64 LMUL m1, and SEW32 LMUL m2 selected-boundary witnesses while preserving direct pre-realized route-entry fail-closed behavior.

### Main Changes

- Extended RVV config/runtime ABI contracts, dialect verifiers, construction metadata checks, selected-body realization validation, and EmitC route planning so runtime-scalar computed-mask memory derives scalar C type, vector/mask C types, SEW, LMUL, policy, ABI order, provider mirrors, and target leaves from typed body/config/runtime facts.
- Added typed i64 and LMUL m2 pre-realized target fixtures for masked store and masked load-store; updated baseline fixture mirrors from e32m1-specific leaf/profile text to typed route-family mirrors.
- Updated generated-bundle ABI tooling for typed runtime-scalar compare-masked memory witnesses, including typed prototypes, runtime parameter facts, intrinsic expectations, harness value types, and direct route-entry fail-closed coverage across baseline/i64/m2 op kinds.
- Preserved selected-boundary-only behavior: direct pre-realized route-entry fails closed for all baseline/i64/m2 masked-memory witnesses; selected-boundary dry-run reports `route_entry_realization=false` and `pre_realized_body_consumed=true`.
- Verified `ssh rvv` compile/run/correctness for i64 and LMUL m2 typed masked store/load-store witnesses over counts 0,1,8,16,23,32,257 and rhs scalar values -37/91, covering active/inactive masks, inactive-lane preservation, old-destination passthrough, source preservation, payload distinction, and tail preservation. Baseline masked store/load-store remote non-regression passed over counts 0,1,16,23,257.
- Spec-update judgment: no `.trellis/spec/**` edit needed; existing RVV plugin, `tcrv.exec`, EmitC route, and testing specs already encode typed body authority, runtime-scalar computed-mask memory statement-plan ownership, mirror-only metadata, selected-boundary realization, and direct route-entry fail-closed contracts.
- Checks: py_compile; script self-test; focused build for `tcrv-opt`, `tcrv-translate`, and `tianchenrv-rvv-extension-plugin-test`; plugin smoke test; typed REALIZED/PLAN/HEADER FileCheck for four new fixtures; generated-bundle typed dry-run and evidence FileChecks; baseline dry-run; runtime_scalar_cmp_select and runtime_scalar_dual_cmp_mask_and_select typed dry-run non-regression; direct pre-realized route-entry fail-closed probe; `ssh rvv` typed and baseline correctness; `git diff --check`; bounded added-line authority scan; final `cmake --build build --target check-tianchenrv -j2` 434/434.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] Focused local compiler/script checks passed.
- [OK] Generated-bundle selected-boundary dry-runs passed for baseline and typed masked-memory witnesses.
- [OK] Direct pre-realized route-entry remains fail-closed for baseline/i64/m2 masked-memory witnesses.
- [OK] `ssh rvv` compile/run/correctness passed for typed and baseline masked-memory witnesses.
- [OK] `check-tianchenrv` passed 434/434.

### Status

[OK] **Completed**

### Next Steps

- Archive the Trellis task and create the final commit.

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


## Session 278: Stage2 RVV runtime-scalar cmp masked memory side-effect boundary

**Date**: 2026-05-28
**Task**: Stage2 RVV runtime scalar cmp masked memory side-effect boundary
**Branch**: `main`

### Summary

Closed `runtime_scalar_cmp_masked_store` and
`runtime_scalar_cmp_masked_load_store` selected-boundary/generated-bundle/runtime
ABI evidence with structured typed-body, provider route, emitted RVV C, harness,
direct fail-closed, `ssh rvv` correctness, and `check-tianchenrv` 411/411.

### Main Changes

- Added `runtime_scalar_computed_mask_memory_boundary` evidence extraction in
  `scripts/rvv_generated_bundle_abi_e2e.py` for materialized typed `tcrv_rvv`
  bodies, provider metadata, emitted RVV C masked-store and masked-load-plus-
  store structure, runtime scalar splat/compare facts, ABI order,
  inactive-lane behavior, and runtime AVL/VL.
- Added focused dry-run and direct fail-closed Script tests for
  `runtime_scalar_cmp_masked_store` and
  `runtime_scalar_cmp_masked_load_store`.
- Kept production C++ stable; existing selected-body/provider/target path was
  sufficient, and the round closed the missing executable evidence boundary.

### Git Commits

`this commit` - final round commit

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] selected-boundary dry-run for both masked memory op kinds
- [OK] direct pre-realized route-entry fail-closed probe for both op kinds
- [OK] new Script lit filter: 2/2 passed
- [OK] `ssh rvv` generated-bundle compile/run/correctness for counts
  `0,1,16,23,257` and RHS scalars `-37,91`
- [OK] non-regression lit for `runtime_scalar_dual_cmp_mask_and_select` and
  `runtime_scalar_cmp_select`: 4/4 passed
- [OK] Target/RVV selected-body fixture lit for both op kinds: 2/2 passed
- [OK] bounded authority scan and `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 411/411 passed

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 279: Stage2 RVV runtime-scalar masked standalone reduction boundary

**Date**: 2026-05-28
**Task**: Stage2 RVV runtime-scalar masked standalone reduction boundary
**Branch**: `main`

### Summary

Closed runtime_scalar_cmp_masked_standalone_reduce_add generated-bundle executable boundary with script evidence extraction, focused lit coverage, direct route-entry fail-closed behavior, ssh rvv evidence, and 413/413 check-tianchenrv.

### Main Changes

- `scripts/rvv_generated_bundle_abi_e2e.py`: added runtime-scalar computed-mask standalone reduction emitted-C extraction and non-empty `mask_tail_policy_boundary` / `reduction_accumulation_boundary` summaries, including rhs scalar splat, compare mask, inactive zeroing, scalar seed/result layout, runtime AVL/VL, and artifact ABI order.
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-dry-run.test`: added selected-boundary dry-run coverage for provider route facts, operand binding, mask/reduction evidence, generated ABI, and harness coverage.
- `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-fail-closed.test`: added fail-closed coverage for the deprecated direct pre-realized route-entry shortcut.
- Trellis PRD recorded completion evidence and production-owner conclusion: C++ production path unchanged; the missing boundary was evidence/test coverage.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | `rvv: close runtime scalar masked standalone reduction boundary` |

### Evidence

- [OK] selected-boundary dry-run for `runtime_scalar_cmp_masked_standalone_reduce_add`: `route_entry_realization=false`, `pre_realized_body_consumed=true`, ABI order `cmp_lhs,rhs_scalar,src,acc,out,n`, provider mirror `provider_supported_mirror:rvv-runtime-scalar-cmp-masked-standalone-reduction-plan-validated`, and non-empty mask/reduction boundaries.
- [OK] direct pre-realized route-entry fail-closed probe for `runtime_scalar_cmp_masked_standalone_reduce_add` with selected-boundary-only diagnostic.
- [OK] real `ssh rvv` generated-bundle compile/run for counts `0,1,16,23,257`, thresholds `-37,91`, seeds `-11,17`, mixed signed payloads, inactive nonzero exclusion, scalar seed/result behavior, and tail preservation.
- [OK] non-regression dry-runs for `runtime_scalar_cmp_masked_store`, `runtime_scalar_cmp_masked_load_store`, and `runtime_scalar_cmp_masked_macc_add`.
- [OK] manual FileCheck replay for both new lit tests.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] `git diff --check`.
- [OK] bounded added-line authority scan: exact intrinsic spellings appear only in emitted-C verification/test evidence; no descriptor/source-front-door/direct-C/source-export/legacy `tcrv_rvv.i32_` executable authority was added.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 413/413 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 280: Stage2 RVV typed dtype LMUL elementwise route-family derivation

**Date**: 2026-05-28
**Task**: Stage2 RVV typed dtype LMUL elementwise route-family derivation
**Branch**: `main`

### Summary

Completed RVV elementwise route-family derivation from typed dtype/SEW/LMUL facts; i64_add and lmul_m2_add dry-run and ssh rvv evidence passed; check-tianchenrv 413/413 passed.

### Main Changes

Implemented provider-owned typed elementwise route-family derivation for non-i32m1 witnesses. Route planning/provider now carry and verify element type, signed C type, SEW, LMUL, policy, config contract, runtime n/AVL, ABI roles, header/type mapping, provider-supported mirrors, and artifact mirror metadata from typed tcrv_rvv body/config/capability/runtime facts.

Validated with check-tianchenrv 413/413, generated-bundle dry-run for i64_add and lmul_m2_add counts 0,1,16,23,257, ssh rvv compile/run/correctness for both witnesses, git diff --check, and bounded authority scan. Self-repaired FileCheck ordering, strided ABI validator over-narrowing, and duplicate check artifact collisions by rerunning a single clean check.


### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | `rvv: derive typed elementwise route facts` |

### Testing

- [OK] `ninja -C build check-tianchenrv`: 413/413 passed.
- [OK] generated-bundle dry-run for `i64_add` and `lmul_m2_add`, counts `0,1,16,23,257`.
- [OK] `ssh rvv` compile/run/correctness for `i64_add` and `lmul_m2_add`, counts `0,1,16,23,257`.
- [OK] `git diff --check` and bounded authority scan.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 281: Stage2 RVV typed masked elementwise route-family derivation

**Date**: 2026-05-28
**Task**: Stage2 RVV typed masked elementwise route-family derivation
**Branch**: `main`

### Summary

Completed typed masked elementwise route-family derivation for masked_i64_add and masked_lmul_m2_sub: RVV verifier/realization/planning/provider now derive vector and mask facts from typed body/config/runtime facts; generated-bundle dry-run, ssh rvv counts 0,1,16,23,257, non-regression dry-runs, git diff --check, authority scan, and check-tianchenrv 416/416 passed.

### Main Changes

- Extended typed masked selected-body validation from the bounded i32m1 case to
  typed SEW64/LMUL m1 and SEW32/LMUL m2 masked elementwise witnesses.
- Realization now creates compare mask types from the realized vector element
  type and LMUL, so `masked_i64_add` produces `!tcrv_rvv.mask<i64, "m1">`
  / `vbool64_t` and `masked_lmul_m2_sub` produces
  `!tcrv_rvv.mask<i32, "m2">` / `vbool16_t`.
- Route planning/provider now validate masked elementwise mask type/C type,
  compare leaf, masked merge leaf, mask role/source, inactive-lane contract,
  runtime AVL/VL facts, and provider mirrors before building
  `TCRVEmitCLowerableRoute`.
- Generated-bundle evidence covers `masked_i64_add` and
  `masked_lmul_m2_sub`, including typed mask suffix derivation and type-correct
  i64 harness oracle formatting.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | `rvv: derive typed masked elementwise route facts` |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] generated-bundle dry-run for `masked_i64_add` and
  `masked_lmul_m2_sub`, counts `0,1,16,23,257`.
- [OK] generated-bundle dry-run non-regression for `i64_add` and
  `lmul_m2_add`, counts `0,1,16,23,257`.
- [OK] `ssh rvv` generated-bundle compile/run/correctness for
  `masked_i64_add` and `masked_lmul_m2_sub`, counts `0,1,16,23,257`,
  with inactive-lane passthrough preservation.
- [OK] `git diff --check`.
- [OK] bounded added-line authority scan: production additions did not add
  legacy `RVVI32M1` / `rvv-i32m1` / `tcrv_rvv.i32_`, descriptor,
  source-front-door, direct-C/source-export, or exact i32m1 intrinsic
  authority. New script mirror labels remain explicit
  `provider_supported_mirror` evidence consumers.
- [OK] `ninja -C build check-tianchenrv`: 416/416 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 282: Stage2 RVV typed compare-select route derivation

**Date**: 2026-05-28
**Task**: Stage2 RVV typed compare-select route derivation
**Branch**: `main`

### Summary

Derived bounded i64 m1 and i32 m2 compare/select route facts from typed tcrv_rvv body/config/runtime facts; added generated-bundle and ssh rvv evidence.

### Main Changes

- Extended runtime-scalar compare-select verifier/config and pre-realized body checks so SEW64 LMUL m1 and SEW32 LMUL m2 typed witnesses are accepted only with matching scalar SSA type and ABI C type facts.
- Realized typed runtime-scalar compare-select bodies through setvl/with_vl, vector load, runtime scalar splat, compare mask, true/false loads, select, and store while preserving predicate, policy, AVL/VL, and ABI order.
- Updated RVV provider route planning and construction evidence so runtime-scalar compare/select vector, mask, scalar C types, route-family diagnostics, intrinsic mirrors, and generated bundle ABI are derived from structural typed body/config/runtime facts.
- Added i64 and LMUL m2 target fixtures, generated-bundle dry-run evidence, direct pre-realized route-entry fail-closed coverage, stale scalar type/C type negative tests, computed-mask select non-regression, and ssh rvv artifact evidence.
- Archived `.trellis/tasks/05-28-05-28-stage2-rvv-typed-runtime-scalar-cmp-select-route-family-derivation` under `.trellis/tasks/archive/2026-05/`.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] Focused verifier/FileCheck checks for `pre-realized-runtime-scalar-compare-select-negative.mlir`, `generic-stage2-dataflow.mlir`, and i64/m2 REALIZED/PLAN/HEADER target fixtures.
- [OK] Generated-bundle dry-run for `runtime_scalar_cmp_select_i64` and `runtime_scalar_cmp_select_lmul_m2`, plus artifact JSON/C/harness FileChecks.
- [OK] Direct pre-realized route-entry fail-closed for baseline/i64/m2 runtime-scalar compare-select.
- [OK] Computed-mask select i64/m2 generated-bundle dry-run non-regression.
- [OK] `ssh rvv` correctness for both typed witnesses with counts `0,1,16,23,257` and RHS scalars `-37,91`.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`; `ninja -C build tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test`; `build/bin/tianchenrv-rvv-extension-plugin-test`; `ninja -C build check-tianchenrv` (426/426); `git diff --check`; Trellis validate; bounded authority scan.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 283: Stage2 RVV typed computed-mask select route-family derivation

**Date**: 2026-05-28
**Task**: Stage2 RVV typed computed-mask select route-family derivation
**Branch**: `main`

### Summary

Completed bounded typed computed-mask select route-family derivation for i64 m1 and i32 m2, with selected-boundary realization, provider-derived typed route facts, generated-bundle evidence, ssh rvv runtime correctness, task archive, and final quality gates.

### Main Changes

- Extended computed-mask select verifier/realization/provider planning so typed SEW64 LMUL m1 and SEW32 LMUL m2 variants are ordinary provider-derived route-family instances, while unsupported i64 m2 fails closed.
- Added typed pre-realized target fixtures, generated-bundle dry-run coverage, direct pre-realized route-entry negative coverage, and typed harness/runtime ABI evidence for computed_mask_select_i64 and computed_mask_select_lmul_m2.
- Verified direct route-entry remains selected-boundary-only for typed computed-mask select; artifact fields stay mirror-only after provider route construction.
- Spec-update judgment: no `.trellis/spec/**` edit needed because existing RVV plugin, EmitC route, and testing specs already encode the durable contract; this task instantiates bounded witnesses rather than adding a new long-term rule.
- Checks: py_compile; script self-test; typed generated-bundle dry-run; expected direct-route-entry fail-closed probes; non-regression dry-run after self-repair from a concurrent artifact collision; ssh rvv counts 0,1,16,23,257 for both typed witnesses; git diff --check; task validate; check-tianchenrv 422/422; bounded authority scan.


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


## Session 284: Stage2 RVV typed runtime-scalar compare-select route-family derivation

**Date**: 2026-05-28
**Task**: Stage2 RVV typed runtime-scalar compare-select route-family derivation
**Branch**: `main`

### Summary

Closed runtime_scalar_cmp_select_i64 and runtime_scalar_cmp_select_lmul_m2 through selected-boundary realization, provider-derived route facts, generated-bundle ABI dry-run, direct-entry fail-closed evidence, ssh rvv correctness, computed-mask non-regression, check-tianchenrv, archive, and commit preparation.

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


## Session 285: Stage2 RVV typed runtime-scalar dual compare-mask/select route-family derivation

**Date**: 2026-05-28
**Task**: Stage2 RVV typed runtime-scalar dual compare-mask/select route-family derivation
**Branch**: `main`

### Summary

Completed typed runtime_scalar_dual_cmp_mask_and_select i64 and LMUL m2 derivation through selected-boundary realization, provider-derived route facts, generated-bundle ABI, direct-entry fail-closed evidence, ssh rvv correctness, and final check-tianchenrv.

### Main Changes

- Extended RVV config/runtime ABI construction, verifier checks, selected-body realization validation, construction metadata verification, and EmitC route planning so dual runtime-scalar compare-mask/select derives element type, SEW, LMUL, scalar/vector/mask C types, mask-and intrinsic, ABI order, and provider mirrors from typed body/config/runtime facts.
- Added typed i64 and LMUL m2 pre-realized target fixtures and generated-bundle dry-run coverage; updated baseline dual fixture mirrors from e32m1-specific profile text to typed route-family mirrors.
- Preserved selected-boundary-only behavior: direct pre-realized route-entry fails closed for baseline/i64/m2 dual witnesses, while selected-boundary dry-run reports route_entry_realization=false and consumes the pre-realized body before provider route construction.
- Verified ssh rvv compile/run/correctness for runtime_scalar_dual_cmp_mask_and_select_i64 and runtime_scalar_dual_cmp_mask_and_select_lmul_m2 over counts 0,1,16,23,257 and threshold pairs -37/-37, -37/91, 91/-37, 91/91, covering both masks, composed lanes, single-mask-only lanes, selected payloads, and tail preservation.
- Spec-update judgment: no .trellis/spec/** edit needed; existing RVV plugin, tcrv.exec, EmitC route, and testing specs already encode the durable selected-body realization and typed authority contract. The transient failures observed during closeout came from accidental concurrent generated-bundle/check invocations sharing artifact roots; clean single reruns passed.
- Checks: py_compile; script self-test; tcrv-opt/tcrv-translate/plugin-test build; build/bin/tianchenrv-rvv-extension-plugin-test; typed dual dry-run; direct fail-closed probe; runtime_scalar_cmp_select and computed_mask_select non-regression dry-runs; ssh rvv dual correctness; git diff --check; task validate; final single ninja -C build check-tianchenrv 429/429; bounded production added-line authority scan.


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


## Session 286: Stage2 RVV typed runtime-scalar compare-masked standalone reduction route-family derivation

**Date**: 2026-05-28
**Task**: Stage2 RVV typed runtime-scalar compare-masked standalone reduction route-family derivation
**Branch**: `main`

### Summary

Completed typed runtime_scalar_cmp_masked_standalone_reduce_add derivation for baseline SEW32 LMUL m1 and SEW64 LMUL m1; SEW32 LMUL m2 remains targeted fail-closed on missing separate LMUL m1 scalar reduction accumulator/result channel; direct pre-realized route-entry remains fail-closed; py_compile, self-test, focused FileCheck, ssh rvv correctness, non-regression dry-runs, git diff --check, authority scan, and check-tianchenrv 436/436 passed.

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
