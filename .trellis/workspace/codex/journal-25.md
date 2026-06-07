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


## Session 523: Stage2 RVV composite gather-MAcc-scatter route contract

**Date**: 2026-06-07
**Task**: Stage2 RVV composite gather-MAcc-scatter selected-body realization and route contract
**Branch**: `main`

### Summary

Replaced the explicit selected-body composite fail-closed seam with a positive
RVV provider-owned route-supported contract for runtime-scalar computed-mask
indexed gather -> masked MAcc -> masked indexed scatter. Pre-realized
multi-family composites remain fail-closed at the named composite
realization-owner boundary.

### Main Changes

- Added composite operation/memory-form route facts, ABI order, runtime ABI
  parameter contract, operand-binding plan, target leaf profile, provider
  mirror facts, construction protocol role steps, and computed-mask memory
  family integration.
- Added explicit composite collector validation for gather/index/mask/MAcc/
  scatter/accumulator/destination/VL facts and stale scatter-value rejection.
- Updated the RVV plugin spec with the executable composite route contract.
- Updated focused C++ smoke coverage from old explicit fail-closed expectation
  to positive route-contract assertions while retaining the pre-realized
  fail-closed owner-boundary diagnostic.

### Git Commits

pending-in-this-commit

### Testing

- [OK] `rtk ninja -C build tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk ninja -C build tianchenrv-target-artifact-export-test`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk git diff --check`
- [OK] `rtk git diff --cached --check`
- [OK] Added-diff old-authority scan found no descriptor/source-front-door/
  legacy `tcrv_rvv.i32_*`/`RVVI32M1`/`rvv-i32m1` route-authority additions.
  Added `__riscv_*_i32m1` strings are provider-derived leaf mirror assertions
  only.

### Status

[OK] **Completed explicit route contract**

### Next Steps

- Implement the plugin-local pre-realized composite realization owner that
  rewrites separate gather, MAcc, and scatter family bodies into the explicit
  composite realized body shape.
- Add target artifact/generated-bundle/header mirror evidence, then `ssh rvv`
  evidence only after artifact/export support is structurally present.


## Session 523: Stage2 RVV composite gather-MAcc-scatter pre-realized realization

**Date**: 2026-06-07
**Task**: Stage2 RVV composite gather-MAcc-scatter pre-realized realization
**Branch**: `main`

### Summary

Implemented the RVV plugin-local composite selected-body realization owner that fuses pre-realized runtime-scalar computed-mask indexed gather, MAcc, and scatter family bodies into the explicit route-supported tcrv_rvv body; added positive provider-contract evidence and stale-index fail-closed coverage.

### Main Changes

- Added `RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner` and wired
  it into the RVV selected-body realization dispatch.
- The owner validates compatible pre-realized gather, MAcc, and scatter family
  facts, materializes one explicit composite `setvl`/`with_vl` body, and
  removes family-only placeholder ABI values before route construction.
- Updated RVV plugin C++ coverage so explicit and pre-realized composite bodies
  reach the same provider-owned route contract, with stale scatter-index
  fail-closed coverage.

### Git Commits

| Hash | Message |
|------|---------|
| pending-in-this-commit | rvv: realize composite gather macc scatter bodies |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 524: Stage2 RVV composite artifact ABI boundary

**Date**: 2026-06-07
**Task**: Stage2 RVV composite artifact ABI boundary
**Branch**: `main`

### Summary

Carried runtime-scalar composite gather-MAcc-scatter selected bodies through provider route facts, target artifact mirrors, generated-bundle dry-run evidence, focused fail-closed checks, and script/lit coverage; no ssh rvv runtime claim was made.

### Main Changes

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
