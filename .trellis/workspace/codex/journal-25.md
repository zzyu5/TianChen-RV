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


## Session 528: Stage2 RVV selected-dispatch composite executable bundle boundary

**Date**: 2026-06-07
**Task**: Stage2 RVV selected-dispatch composite executable bundle boundary
**Branch**: `main`

### Summary

Extended the composite gather-MAcc-scatter generated-bundle evidence boundary
from selected-envelope runtime execution to selected dispatch/fallback
execution. The production compiler seam already failed closed through target
artifact validation; this round hardened the generated-bundle evidence tooling
so object/header bundle metadata and per-op evidence explicitly carry the
selected dispatch case/fallback boundary as mirror-only facts after RVV
provider route and target artifact validation.

### Main Changes

- Created the Trellis task
  `06-07-stage2-rvv-selected-dispatch-composite-executable-bundle-boundary`
  with a bounded PRD for selected-dispatch/fallback composite generated-bundle
  execution.
- Added optional selected dispatch case/fallback mirror expectations to
  `scripts/rvv_generated_bundle_abi_e2e.py` and populated them for the
  explicit and pre-realized runtime-scalar-cmp masked indexed
  gather-MAcc-scatter paths.
- Added generated-bundle object/header metadata verification for selected
  dispatch case, fallback, selected-envelope ABI bindings, runtime ABI order,
  route operand binding plan/summary, and provider support mirror.
- Added `selected_dispatch_bundle_boundary` to per-op and root evidence JSON,
  labeled as mirror-only after provider route and selected-dispatch validation.
- Updated the focused generated-bundle dry-run lit test to require the new
  explicit and pre-realized selected-dispatch bundle boundary evidence fields.
- Updated `.trellis/spec/testing/mlir-testing-contract.md` with the reusable
  selected-dispatch generated-bundle evidence JSON contract.
- Verified explicit and pre-realized non-dry-run generated bundles on `ssh rvv`
  under `artifacts/tmp/stage2-selected-dispatch-composite-gms-ssh-rvv/`.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | rvv: prove selected dispatch composite bundle |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Explicit selected-dispatch composite dry-run generated-bundle evidence.
- [OK] Pre-realized selected-dispatch composite dry-run generated-bundle
  evidence.
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run`
  from `build/test`.
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter|pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter|selected-dispatch-fallback-envelope-scalar-broadcast-macc-negative|materialize-dispatch-runtime-guards|variant-dispatch-synthesis|rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run'`
  from `build/test`.
- [OK] Explicit selected-dispatch composite non-dry-run generated-bundle
  evidence on `ssh rvv`, including
  `PASS op=runtime_scalar_cmp_masked_indexed_gather_macc_scatter counts=0,1,16,17,257 rhs_scalars=-37,91 patterns=0,1`.
- [OK] Pre-realized selected-dispatch composite non-dry-run generated-bundle
  evidence on `ssh rvv`, including the same PASS marker.
- [OK] Bounded old-authority scan over touched files and added diff lines found
  no new positive legacy route authority.
- [OK] `git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 527: Stage2 RVV selected dispatch/fallback composite artifact boundary

**Date**: 2026-06-07
**Task**: Stage2 RVV selected dispatch/fallback composite artifact boundary
**Branch**: `main`

### Summary

Hardened the composite gather-MAcc-scatter selected dispatch/fallback artifact
boundary with focused fail-closed tests. No production source change was needed:
the existing RVV provider and target artifact bridge already rebuild the route
from actual `tcrv.exec` dispatch/fallback, runtime guard, selected variant, ABI,
and provider facts before accepting candidate metadata.

### Main Changes

- Created and completed the Trellis task with a bounded PRD for the selected
  dispatch/fallback composite artifact boundary.
- Added composite-specific target artifact fail-closed coverage for missing
  actual `tcrv.exec.case`, missing actual `tcrv.exec.fallback`, missing
  runtime guard linkage, stale selected dispatch case/fallback mirrors, and
  missing selected dispatch case/fallback mirrors.
- Confirmed no `.trellis/spec/` update was needed because existing specs already
  define the mirror-non-authority and fail-closed selected dispatch/fallback
  contract.
- Made no new runtime correctness claim and did not run new `ssh rvv` evidence.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Testing

- [OK] `ninja -C build tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter|pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter|selected-dispatch-fallback-envelope-scalar-broadcast-macc-negative|materialize-dispatch-runtime-guards|variant-dispatch-synthesis|rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run'`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-07-stage2-rvv-selected-dispatch-fallback-composite-artifact-boundary`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Bounded old-authority scan over added test diff lines found no legacy
  positive route authority; task-note match was negative out-of-scope wording.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 527: Stage2 RVV composite selected-envelope generated-bundle runtime evidence

**Date**: 2026-06-07
**Task**: Stage2 RVV composite gather-MAcc-scatter generated-bundle runtime evidence
**Branch**: `main`

### Summary

Closed the selected-envelope composite gather-MAcc-scatter generated-bundle
runtime evidence gap with explicit and pre-realized non-dry-run `ssh rvv`
execution. No compiler source change was needed because the selected-envelope
artifact/export seam from `3ae4859e` already executed correctly.

### Main Changes

- Created the Trellis task
  `06-07-stage2-rvv-composite-gather-macc-scatter-generated-runtime` with a
  bounded PRD for selected-envelope generated-bundle runtime evidence.
- Verified explicit and pre-realized selected-envelope composite generated
  bundles on `ssh rvv` under
  `artifacts/tmp/stage2-selected-envelope-composite-gms-ssh-rvv/`.
- Confirmed both generated bundle indexes carry selected-envelope
  `tcrv_rvv.exec_abi_bindings`, runtime ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`, and provider support
  mirrors.
- Recorded that no `.trellis/spec/` update was needed because the existing RVV
  plugin, EmitC route, emission runtime, and testing specs already cover this
  evidence contract.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | rvv: record selected envelope composite ssh evidence |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-cmp-masked-indexed-gather-macc-scatter'`
  from `build/test` passed 3 focused tests.
- [OK] Explicit selected-envelope non-dry-run `ssh rvv` generated bundle:
  `artifacts/tmp/stage2-selected-envelope-composite-gms-ssh-rvv/explicit-selected-envelope-composite-gms-ssh-rvv/runtime_scalar_cmp_masked_indexed_gather_macc_scatter/evidence.json`.
- [OK] Pre-realized selected-envelope non-dry-run `ssh rvv` generated bundle:
  `artifacts/tmp/stage2-selected-envelope-composite-gms-ssh-rvv/pre-selected-envelope-composite-gms-ssh-rvv/runtime_scalar_cmp_masked_indexed_gather_macc_scatter/evidence.json`.
- [OK] Both remote runs reported
  `PASS op=runtime_scalar_cmp_masked_indexed_gather_macc_scatter counts=0,1,16,17,257 rhs_scalars=-37,91 patterns=0,1`.
- [OK] Bounded old-authority scan over touched files and added diff lines found
  no new positive legacy route authority.
- [OK] `git diff --check`
- [OK] `git diff --cached --check`

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


## Session 525: Stage2 RVV composite gather-MAcc-scatter ssh rvv runtime boundary

**Date**: 2026-06-07
**Task**: Stage2 RVV composite gather-MAcc-scatter ssh rvv runtime boundary
**Branch**: `main`

### Summary

Closed the composite gather-MAcc-scatter runtime boundary with explicit and pre-realized generated-bundle ssh rvv evidence; no production source change was needed because the existing artifact/runtime seam executed correctly.

### Main Changes

- Created and archived the Trellis runtime-boundary task with a bounded PRD for
  the composite gather-MAcc-scatter `ssh rvv` evidence owner.
- Verified the existing production artifact/runtime seam without changing
  compiler source or the generated-bundle script.
- Collected explicit and pre-realized selected-body generated-bundle evidence
  under `artifacts/tmp/stage2-composite-gms-ssh-rvv/`, both with remote compile
  and remote run success on `ssh rvv`.
- Recorded that no `.trellis/spec/` update was needed because the current RVV
  plugin, EmitC route, emission runtime, and testing specs already cover the
  composite runtime evidence contract.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Testing

- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind runtime_scalar_cmp_masked_indexed_gather_macc_scatter ...`
  non-dry-run explicit selected-body evidence on `ssh rvv`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind runtime_scalar_cmp_masked_indexed_gather_macc_scatter ...`
  non-dry-run pre-realized selected-body evidence on `ssh rvv`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-cmp-masked-indexed-gather-macc-scatter'`
  from `build/test`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Bounded old-authority scan over task/journal additions found no new
  positive legacy route authority; only historical journal text and negative
  out-of-scope wording matched.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 526: Stage2 RVV composite selected envelope ABI

**Date**: 2026-06-07
**Task**: Stage2 RVV composite selected envelope ABI
**Branch**: `main`

### Summary

Bound the composite gather-MAcc-scatter route through the selected tcrv.exec envelope ABI mirror and target artifact fail-closed validation.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `11603636` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
