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


## Session 388: Stage2 RVV runtime-scalar cmp masked standalone reduce-add artifact ABI boundary

**Date**: 2026-06-02
**Task**: Stage2 RVV runtime-scalar cmp masked standalone reduce-add artifact ABI boundary
**Branch**: `main`

### Summary

Moved runtime-scalar computed-mask standalone reduce-add inactive neutral literal authority into the RVV provider API, tightened add provider/candidate fail-closed coverage and generated-bundle facts, proved dry-run and real ssh rvv correctness, and archived the Trellis task.

### Main Changes

- Added provider-owned canonical facts for runtime-scalar dual compare/mask-and/select routes, including ABI order, two scalar roles, predicate/mask/select layout, header/type summaries, and a 499-byte complete route operand binding summary.
- Rewired target artifact validation and C++ fail-closed checks to consume provider facts instead of target-local duplicate route truth.
- Updated generated-bundle evidence and MLIR/FileCheck fixtures for the full eight-parameter ABI binding summary.
- Recorded the SEW/LMUL-aware route fact accessor contract in `.trellis/spec/lowering-runtime/emitc-route.md`.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] targeted lit from `build/test` with filter `runtime-scalar-dual-cmp-mask-and-select`: 8 tests passed.
- [OK] real `ssh rvv` generated-bundle run for counts `0,1,16,23,257` and rhs scalar values `-37,91`; all 20 pair/count cases passed with tail preservation.
- [OK] `git diff --check`
- [OK] bounded added-line old-authority scan over touched diff.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 389: Stage2 RVV runtime-scalar reduce-max artifact facts

**Date**: 2026-06-02
**Task**: Stage2 RVV runtime-scalar reduce-max artifact facts
**Branch**: `main`

### Summary

Validated runtime-scalar computed-mask standalone reduce-max provider facts, target artifact fail-closed checks, generated-bundle dry-run, and ssh rvv correctness evidence.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `999c875e` | (see git log) |
| `297e075e` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 390: Stage2 RVV runtime scalar masked macc add

**Date**: 2026-06-02
**Task**: Stage2 RVV runtime scalar masked macc add
**Branch**: `main`

### Summary

Validated runtime-scalar computed-mask MAcc add provider facts, target artifact fail-closed checks, generated bundle evidence, and ssh rvv correctness.

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


## Session 391: Stage2 RVV computed-masked macc-add artifact ABI boundary

**Date**: 2026-06-02
**Task**: Stage2 RVV computed-masked macc-add artifact ABI boundary
**Branch**: `main`

### Summary

Added provider-owned computed-mask MAcc facts, rewired provider/target validation, tightened generated-bundle checks, and verified ssh rvv correctness for counts 0,1,16,17,257 patterns 0,1.

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


## Session 392: Stage2 RVV signed widening macc-add artifact ABI boundary

**Date**: 2026-06-02
**Task**: Stage2 RVV signed widening macc-add artifact ABI boundary
**Branch**: `main`

### Summary

Validated signed widening MAcc provider facts, target artifact fail-closed checks, generated bundle dry-run evidence, and real ssh rvv correctness for counts 0,1,7,16,23,257 patterns 0,1.

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


## Session 393: Stage2 RVV runtime-scalar dual-cmp mask-and-select artifact ABI boundary

**Date**: 2026-06-03
**Task**: Stage2 RVV runtime-scalar dual-cmp mask-and-select artifact ABI boundary
**Branch**: `main`

### Summary

Finished runtime_scalar_dual_cmp_mask_and_select provider facts, target artifact validation, generated-bundle dry-run checks, ssh rvv correctness, and Trellis archive.

### Main Changes

- Added provider-owned canonical facts for runtime-scalar dual compare/mask-and/select routes, including ABI order, two scalar roles, predicate/mask/select layout, header/type summaries, and a 499-byte complete route operand binding summary.
- Rewired target artifact validation and C++ fail-closed checks to consume provider facts instead of target-local duplicate route truth.
- Updated generated-bundle evidence and MLIR/FileCheck fixtures for the full eight-parameter ABI binding summary.
- Recorded the SEW/LMUL-aware route fact accessor contract in `.trellis/spec/lowering-runtime/emitc-route.md`.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] targeted lit from `build/test` with filter `runtime-scalar-dual-cmp-mask-and-select`: 8 tests passed.
- [OK] real `ssh rvv` generated-bundle run for counts `0,1,16,23,257` and rhs scalar values `-37,91`; all 20 pair/count cases passed with tail preservation.
- [OK] `git diff --check`
- [OK] bounded added-line old-authority scan over touched diff.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 394: Stage2 RVV computed-masked strided-input widening dot-reduce artifact ABI boundary

**Date**: 2026-06-03
**Task**: Stage2 RVV computed-masked strided-input widening dot-reduce artifact ABI boundary
**Branch**: `main`

### Summary

Finished computed-mask strided-input widening dot-reduce provider binding facts, target artifact mirror validation evidence, generated-bundle dry-run checks, ssh rvv correctness, and Trellis archive.

### Main Changes

- Strengthened RVV math operand-binding facts so computed-mask strided widening dot compare inputs, dot source inputs, and stride inputs all require `hdr` when they participate in the generated header/prototype.
- Updated explicit selected-body and generated-bundle dry-run expectations for the complete nine-parameter binding summary.
- Added focused C++ fail-closed coverage for missing `hdr` on the exported lhs stride parameter.
- Verified target artifact validation consumes provider-owned widening-dot route facts and exact candidate mirrors; no spec update was needed because the existing EmitC route and MLIR testing contracts already cover this rule.

Checks:
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter computed-masked-strided-input-widening-dot-reduce` from `build/test` (5 passed, 472 excluded)
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] real `ssh rvv` generated-bundle run for `computed_masked_strided_input_widening_dot_reduce_add` with counts `0,1,16,17,257`, stride pairs `2:3,3:2`, mask patterns `0,1`, and input patterns `0,1`
- [OK] `git diff --check`
- [OK] bounded touched-diff old-authority scan


### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] Build: `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] C++ tests: RVV extension plugin and target artifact export.
- [OK] Focused lit filter `computed-masked-strided-input-widening-dot-reduce`:
  5 passed, 472 excluded.
- [OK] Script self-test and real `ssh rvv` generated-bundle evidence for
  counts `0,1,16,17,257`, stride pairs `2:3,3:2`, mask patterns `0,1`, and
  input patterns `0,1`.
- [OK] `git diff --check` and bounded touched-diff old-authority scan.

### Status

[OK] **Completed**

### Next Steps

- None - task complete
