# Journal - codex (Part 24)

> Continuation from `journal-23.md` (archived at ~2000 lines)
> Started: 2026-06-06

---



## Session 486: Stage2 RVV product-reduction dequant executable artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV product-reduction dequant executable artifact ABI boundary
**Branch**: `main`

### Summary

Expanded product-reduction-dequant-clamp generated artifact ABI binding evidence from compressed route-level summary to per-operand provider-derived abi/hdr facts; kept metadata bounded; verified focused C++ tests, lit dry-runs, and ssh rvv correctness evidence.

### Main Changes

- Archived `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-computed-masked-macc-artifact-abi/`.
- Recorded the no-source-change audit conclusion: computed-mask MAcc selected-body realization, MAcc provider facts, computed-mask accumulation statement plan, target artifact validation, and generated-bundle script support were already production-valid.
- Closed the missing executable blocker with non-dry-run pre-realized computed-masked MAcc generated-bundle execution on `ssh rvv`.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-commit` | rvv: tighten product dequant artifact abi evidence |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 516: Stage2 RVV computed-mask indexed scatter-store ABI

**Date**: 2026-06-07
**Task**: Stage2 RVV computed-masked indexed scatter-store executable artifact ABI boundary
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes brief, confirmed the existing
computed-mask indexed scatter production seam already reaches provider-owned
indexed validation plus compare/select mask statement-plan validation, and
added focused target artifact fail-closed evidence for stale scatter residue.

### Main Changes

- Added target artifact tests rejecting stale scatter masked-indexed-load
  residue, stale ordinary unit-store residue, unit-stride destination pointer
  residue, and unscaled index-vector payload residue.
- Verified explicit and pre-realized scatter generated bundles on `ssh rvv`
  for counts `0,1,16,17,257` and patterns `0,1`.
- Local readobj header/symbol checks were not covered because the host lacks
  `llvm-readobj`; dry-run materialization/export passed with readobj skipped.

### Status

[OK] Completed


## Session 516: Stage2 RVV runtime-scalar indexed gather-load ABI

**Date**: 2026-06-07
**Task**: Stage2 RVV runtime-scalar-cmp masked indexed gather-load ABI boundary
**Branch**: `main`

### Summary

Completed the runtime-scalar-cmp masked indexed gather-load executable artifact/ABI boundary. The route now has an explicit and pre-realized selected-body surface, RVV plugin-local realization, provider/target validation, generated bundle ABI harness coverage, direct-pre-realized fail-closed evidence, and explicit/pre-realized `ssh rvv` correctness.

### Main Changes

- Added `tcrv_rvv.typed_runtime_scalar_computed_mask_indexed_gather_pre_realized_body` and runtime ABI contract for `lhs,rhs_scalar,src,index,dst,n`.
- Realized the pre-realized body into setvl/lhs load/rhs splat/old-dst load/index load/sle compare/masked indexed load/store before route construction.
- Extended computed-mask indexed gather planning/provider/statement/control-policy/target validation to derive runtime-scalar producer, indexed source, passthrough, ABI, header, and operand-binding facts from typed body structure.
- Extended generated-bundle ABI tooling and tests for explicit/pre-realized dry-run, direct-pre-realized fail-closed, runtime scalar RHS value coverage, indexed gather source preservation, inactive-lane passthrough, and noncontiguous index coverage.

### Testing

- [OK] `scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] explicit and pre-realized generated-bundle dry-run commands with FileCheck prefix validation.
- [OK] direct pre-realized route-entry fail-closed command and FileCheck validation.
- [OK] explicit/pre-realized target fixture PLAN, HEADER, REALIZED, and stale producer checks with `FileCheck-20`.
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -- -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] explicit/pre-realized `ssh rvv` correctness for counts `0,1,16,17,257`, rhs scalars `-37,91`, patterns `0,1`.
- [OK] `git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 516: Stage2 RVV runtime-scalar segment2 load ABI

**Date**: 2026-06-07
**Task**: Stage2 RVV runtime-scalar-cmp masked segment2 load executable artifact ABI boundary
**Branch**: `main`

### Summary

Implemented the read-side runtime-scalar-cmp masked segment2 load selected-body route family through typed/pre-realized RVV body realization, provider-owned route facts, common EmitC materialization, target artifact validation, generated bundle ABI, and real `ssh rvv` correctness evidence.

### Main Changes

- Added `tcrv_rvv.typed_runtime_scalar_computed_mask_segment2_load_pre_realized_body` and verifier/config-contract support for `lhs,rhs_scalar,src,out0,out1,n`.
- Wired RVV owner-local selected-body realization, route-family planning, route-control facts, memory/segment2 statement planning, route provider ABI binding, construction protocol recognition, and target artifact validation for runtime-scalar masked segment2 load.
- Extended generated-bundle support and dry-run lit tests for explicit and pre-realized selected bodies, including runtime scalar producer source, mask provenance, old field passthrough preservation, field0/field1 order, header prototype, and provider-supported mirrors.
- Added explicit and pre-realized artifact fixtures with positive materialization/export checks and stale mask-producer fail-closed coverage.
- Updated RVV plugin and EmitC route specs with the runtime-scalar segment2 load owner/fact-surface boundary.

### Git Commits

| Hash | Message |
|------|---------|
| `created-after-journal-entry` | (see git log) |

### Testing

- [OK] Built `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`, and `tianchenrv-target-artifact-export-test`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test` and `build/bin/tianchenrv-target-artifact-export-test`.
- [OK] `scripts/rvv_generated_bundle_abi_e2e.py` `py_compile` and `--self-test`.
- [OK] Focused lit 4/4 for explicit/pre-realized runtime-scalar segment2 load MLIR fixtures and generated-bundle dry-run tests.
- [OK] Non-dry-run `ssh rvv` generated-bundle evidence for counts `0,1,16,17,257` and RHS scalars `-37,91`, proving active/inactive lanes, old-field passthrough preservation, field ordering, source preservation, and tail preservation.
- [OK] `git diff --check` and bounded old-authority scan.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 514: Stage2 RVV runtime-scalar-cmp segment2 store ABI

**Date**: 2026-06-07
**Task**: Stage2 RVV runtime-scalar-cmp masked segment2 store executable artifact ABI boundary
**Branch**: `main`

### Summary

Completed the explicit selected-body runtime-scalar compare masked segment2 store executable artifact/ABI boundary. The production path now carries `lhs,rhs_scalar,src0,src1,dst,n` through typed `tcrv_rvv` body facts, runtime scalar splat mask construction, provider-owned segment2 store route planning, common EmitC materialization, target artifact validation, generated-bundle ABI evidence, and ssh rvv correctness.

### Main Changes

- Added the `runtime_scalar_cmp_masked_segment2_store_unit_load` selected-body operation and runtime ABI contract.
- Extended RVV construction protocol, computed-mask memory owner classification, segment2 route-family provider plans, statement plans, route provider ABI/type mappings, and target artifact route-family validation.
- Added generated-bundle script support, dry-run lit coverage, explicit target artifact fixture coverage, and a stale producer-source mirror fail-closed test.
- Proved ssh rvv correctness for runtime counts `0,1,16,17,257` with two runtime scalar patterns, active/inactive lane coverage, inactive-lane preservation, source preservation, field distinction, and tail preservation.

### Git Commits

| Hash | Message |
|------|---------|
| `created-after-journal-entry` | (see git log) |

### Testing

- [OK] Focused build, C++ plugin/target tests, focused lit, script py_compile, ssh rvv generated-bundle run, whitespace checks, and old-authority added-line scan passed.

### Status

[OK] **Completed**

### Next Steps

- Optional continuation: pre-realized runtime-scalar-cmp masked segment2 store realization/evidence.


## Session 514: Stage2 RVV computed-mask segment2 store artifact ABI

**Date**: 2026-06-07
**Task**: Stage2 RVV computed-masked segment2 store executable artifact ABI boundary
**Branch**: `main`

### Summary

Completed the computed-masked segment2 store executable artifact/ABI boundary for this round. The production provider/target seam already carried typed store route facts; this round closed the paired store evidence gap by requiring generated-bundle and target-artifact evidence to expose provider-owned mask-tail route-family plan/owner mirrors, and by adding stale provider/candidate fail-closed coverage for those store boundary facts.

### Main Changes

- Created Trellis task `06-07-rvv-computed-masked-segment2-store-artifact-abi` from the Hermes brief and bounded it to `computed_masked_segment2_store_unit_load`.
- Hardened `scripts/rvv_generated_bundle_abi_e2e.py` so computed-masked segment2 store metadata and boundary summaries include `tcrv_rvv.mask_tail_policy_route_family_plan` and `tcrv_rvv.mask_tail_policy_owner`.
- Added script self-test coverage so fake computed-masked segment2 store bundles fail if the store mask-tail plan/owner facts disappear or diverge.
- Updated explicit and pre-realized generated-bundle dry-run FileCheck tests and target artifact MLIR fixtures to expose the store mask-tail plan/owner mirrors.
- Extended `test/Target/TargetArtifactExportTest.cpp` so stale store provider and candidate mask-tail route-family plan/owner facts fail closed before artifact acceptance.
- Produced explicit and pre-realized ssh rvv evidence under `artifacts/tmp/06-07-rvv-computed-masked-segment2-store-artifact-abi`, counts `0,1,7,16,23,257`, patterns `0,1`, with active writes, inactive old destination preservation, field0/field1 distinction, source preservation, and tail preservation.

### Git Commits

| Hash | Message |
|------|---------|
| `created-after-journal-entry` | (see git log) |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] lit filter `computed-masked-segment2-store`: 5 passed, 495 excluded
- [OK] explicit and pre-realized non-dry-run `ssh rvv` generated-bundle runs
- [OK] bounded added-line old-authority scan
- [OK] `git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 497: Stage2 RVV runtime-scalar-cmp masked MAcc LMUL m2 artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV runtime-scalar-cmp masked MAcc LMUL m2 artifact ABI boundary
**Branch**: `main`

### Summary

Revalidated the pre-realized runtime-scalar-cmp masked MAcc add LMUL m2 executable artifact ABI boundary on current HEAD. No compiler source change was required; the production provider and target artifact validator already derive or check LMUL m2 facts from selected typed body/config/runtime facts.

### Main Changes

* Created the Trellis task and PRD for the LMUL m2 executable artifact ABI seam.
* Audited the RVV MAcc owner, runtime-scalar computed-mask MAcc facts accessors, selected-body validation, route planning, target artifact validation, script expectation, and LMUL m2 fixture.
* Recorded the no-source-change conclusion: LMUL m2 uses parameterized `sew/lmul` facts in provider and target contracts, carrying `vint32m2_t`, `vbool16_t`, m2 setvl/load/splat/compare/MAcc/merge/store intrinsics, runtime ABI order, operand binding, inactive-lane preservation, and AVL/VL facts.
* Ran generated bundle compile/run on `ssh rvv` for counts `0,1,7,16,17,23,257`, RHS scalar values `-37,91`, and patterns `0,1`.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-commit` | (see git log) |

### Testing

* [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
* [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
* [OK] `build/bin/tianchenrv-target-artifact-export-test`
* [OK] pre-realized generated bundle `ssh rvv` run for `runtime_scalar_cmp_masked_macc_add_lmul_m2`, counts `0,1,7,16,17,23,257`, RHS scalars `-37,91`, patterns `0,1`
* [OK] focused lit filter for LMUL m2 dry-run, direct pre-realized fail-closed, and target artifact fixture: 3/3 passed
* [OK] `git diff --check`
* [OK] `git diff --cached --check`

### Status

[OK] **Completed**

### Next Steps

* None - task complete


## Session 497: Stage2 RVV runtime-scalar-cmp masked MAcc add executable artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV runtime-scalar-cmp masked MAcc add executable artifact ABI boundary
**Branch**: `main`

### Summary

Revalidated runtime-scalar-cmp masked MAcc add executable artifact ABI on
current HEAD with explicit and pre-realized generated bundles on ssh rvv; no
compiler source change required.

### Main Changes

- Created and archived the Trellis task for runtime-scalar-cmp masked MAcc add
  artifact ABI evidence.
- Audited the production runtime-scalar computed-mask MAcc selected-body
  realization, provider facts, computed-mask accumulation statement plan,
  target validation, script, and fixtures.
- Recorded a no-source-change conclusion: current production code already
  derives `rhs_scalar` compare/splat facts, active MAcc, inactive accumulator
  preservation, accumulator/result roles, ABI/header bindings, runtime AVL/VL,
  and target metadata from provider-owned typed body/config/runtime facts.
- Re-ran current-HEAD explicit and pre-realized generated bundle evidence on
  `ssh rvv`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | rvv: prove runtime scalar masked macc artifact abi |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] explicit generated bundle `ssh rvv` run for
  `runtime_scalar_cmp_masked_macc_add`, counts `0,1,7,16,23,257`, RHS scalars
  `-37,91`, patterns `0,1`
- [OK] pre-realized generated bundle `ssh rvv` run for
  `runtime_scalar_cmp_masked_macc_add`, counts `0,1,7,16,23,257`, RHS scalars
  `-37,91`, patterns `0,1`
- [OK] focused lit filter for runtime-scalar-cmp masked MAcc dry-run,
  fail-closed, and target artifact fixtures: 6/6 passed
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-stage2-rvv-runtime-scalar-cmp-masked-macc-add-artifact-abi`
- [OK] bounded old-authority scan over task files: only PRD negative guardrails
  matched
- [OK] `git diff --check`
- [OK] `git diff --cached --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete

## Session 488: Stage2 RVV runtime-scalar compare-masked memory executable artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV runtime scalar compare masked memory artifact ABI boundary
**Branch**: `main`

### Summary

Audited the runtime-scalar compare-masked store/load-store memory executable artifact ABI seam and confirmed the production path is already provider-owned and fail-closed. No compiler source change was justified; this round records the precise no-source-change production justification and adds real ssh rvv generated-bundle correctness evidence for both memory operations.

### Main Changes

- Created and completed the Trellis task/PRD for the runtime-scalar compare-masked memory artifact ABI boundary.
- Confirmed selected-body realization fixes ABI order `lhs,rhs_scalar,src,dst,n`, runtime setvl/VL, compare-mask provenance, source/destination memory roles, and inactive-lane preservation.
- Confirmed RVV provider validation owns runtime-scalar producer facts, typed config, memory operand bindings, statement plans, route-control facts, mask/tail policy, and stale-fact rejection before route construction.
- Confirmed target artifact validation checks runtime-scalar computed-mask store/load-store header/prototype, ABI parameters, scalar splat, compare, masked memory statements, runtime AVL/VL contract, and provider mirrors.
- Produced non-dry-run generated-bundle `ssh rvv` evidence for `runtime_scalar_cmp_masked_load_store` and `runtime_scalar_cmp_masked_store`; counts `0,1,16,23,257`, rhs scalars `-37,91`, source preservation, tail preservation, mixed masks, and inactive-lane preservation were covered.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-commit` | rvv: prove runtime scalar masked memory artifact abi |

### Testing

- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] generated-bundle dry-run scripts for runtime-scalar compare-masked load-store/store
- [OK] generated-bundle fail-closed scripts for retired direct pre-realized route-entry shortcut
- [OK] 4/4 generated-bundle Script lit tests selected by the runtime-scalar compare-masked memory filter
- [OK] 6/6 Target/RVV lit tests selected by the runtime-scalar compare-masked memory artifact filter
- [OK] non-dry-run generated-bundle `ssh rvv` correctness evidence for load-store and store

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 487: Stage2 RVV runtime-scalar masked standalone reduction executable artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV runtime-scalar masked standalone reduction executable artifact ABI boundary
**Branch**: `main`

### Summary

Confirmed the existing runtime-scalar compare-masked standalone reduce-add selected-body route already had aligned provider-owned ABI, mask, reduction, scalar-result, header/type, and statement evidence; produced real ssh rvv generated-bundle correctness evidence for m1 and m2 without compiler source changes.

### Main Changes

- No compiler source change was required; the production seam already exposed provider-owned runtime ABI order `cmp_lhs,rhs_scalar,src,acc,out,n`, per-operand `abi|hdr` route binding, compare predicate `sle`, mask provenance, inactive zeroing, scalar-result runtime boundary, and header/type mirrors.
- Non-dry-run generated-bundle command over `runtime_scalar_cmp_masked_standalone_reduce_add` and `_lmul_m2` returned `rvv_generated_bundle_abi_e2e: success` with `ssh_evidence: true`.
- Runtime PASS covered counts `0,1,16,23,257`, rhs scalars `-37,91`, seeds `-11,17`, patterns `0,1`, and source/tail preservation.
- Direct pre-realized route-entry mode remained fail-closed with the retired shortcut diagnostic.
- Focused checks passed: `build/bin/tianchenrv-rvv-extension-plugin-test`, `build/bin/tianchenrv-target-artifact-export-test`, 2/2 generated-bundle Script lit tests, and 4/4 Target/RVV fixture lit tests.


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
## Session 489: Stage2 RVV runtime-scalar masked memory production contract hardening

**Date**: 2026-06-06
**Task**: Stage2 RVV runtime-scalar masked memory production fail-closed contract hardening
**Branch**: `main`

### Summary

Hardened the target artifact route payload validation contract for unit-stride
runtime-scalar compare-masked memory. The previous task proved the generated
bundles execute on `ssh rvv`; this round made the target consumer validate
rebuilt route headers, type mappings, and ABI mappings directly against the
provider-owned `RVVUnitStrideMaskedMemoryRouteValidationContract`.

### Main Changes

- Added unit-stride masked memory target helpers for route headers, type
  mappings, and ABI mappings in `RVVTargetArtifactRouteFamilyValidation.cpp`.
- Rewired the unit-stride masked memory branch in
  `validateRVVCompareSelectMaskRoutePayloadFacts` to use those contract-bound
  helpers before statement-plan validation.
- Added target C++ regression checks proving runtime-scalar masked memory
  rejects a missing `riscv_vector.h` route header, stale `!tcrv_rvv.vl` type
  mapping, and stale `rhs_scalar` ABI value mapping before executable artifact
  acceptance.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-commit` | rvv: harden runtime scalar masked memory artifact contract |

### Testing

- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-cmp-masked-(load-store|store)(\\.mlir|-dry-run\\.test|.*fail-closed\\.test)'` from `build/test`: 8/8 passed
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded added-line old-authority scan over touched production/test files
- [OK] spec update review: no `.trellis/spec/` edit needed because the change
  implements existing contract-first target artifact rules

### Status

[OK] **Ready to archive and commit**

### Next Steps

- Archive task and create the final commit.

## Session 492: Stage2 RVV computed-masked segment2 executable artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV computed-masked segment2 artifact ABI boundary
**Branch**: `main`

### Summary

Proved the existing pre-realized computed-masked segment2 load/store artifact
ABI seam with non-dry-run `ssh rvv` generated-bundle evidence. The production
audit found the code path already goes through RVV plugin-local selected-body
realization, computed-mask/segment2 provider plans, provider-built
`TCRVEmitCLowerableRoute`, and segment2 target validation contracts, so this
round did not require production source changes.

### Main Changes

- Created and archived the Trellis PRD for the computed-masked segment2
  load/store executable artifact ABI boundary.
- Recorded that existing target artifact C++ coverage already validates
  computed-mask segment2 load/store/update provider facts, route statement
  shape, ABI/header/type/binding facts, mask facts, field facts, and metadata
  mirror rejection.
- Ran non-dry-run generated-bundle evidence on `ssh rvv` for
  `computed_masked_segment2_load_unit_store` and
  `computed_masked_segment2_store_unit_load`, covering runtime counts
  `0,1,7,16,23,257`, two compare-mask patterns, active/inactive lanes,
  inactive-lane preservation, field-order distinguishing lanes, source
  preservation, and tail preservation.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-commit` | rvv: prove computed masked segment2 artifact abi |

### Testing

- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'computed-masked-segment2-(load|store)'` from `build/test`: 10/10 passed
- [OK] Non-dry-run `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_masked_segment2_load_unit_store --op-kind computed_masked_segment2_store_unit_load` passed with `ssh_evidence: true`
- [OK] `ssh rvv` PASS covered counts `0,1,7,16,23,257`, patterns `0,1`, active/inactive lanes, inactive preservation, field-distinguishing lanes, source preservation, and tail preservation for both load and store
- [OK] `git diff --check`
- [OK] bounded old-authority scan over the touched Trellis task files

### Status

[OK] **Ready to commit**

### Next Steps

- Create the final commit.

## Session 490: Stage2 RVV computed-masked segment2 update executable artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV computed-masked segment2 update executable artifact ABI boundary
**Branch**: `main`

### Summary

Audited the computed-mask segment2 update selected-body route and found the
production seam already goes through the RVV segment2 planning owner,
provider-built `TCRVEmitCLowerableRoute`, and
`RVVSegment2MemoryRouteValidationContract` target validation. This round did
not invent a production source change; it added focused target regression
coverage for rebuilt route payload facts and collected non-dry-run `ssh rvv`
generated-bundle correctness evidence for the pre-realized segment2 update
path.

### Main Changes

- Added target C++ route clone helpers that mutate rebuilt
  `TCRVEmitCLowerableRoute` payloads while preserving statements/provenance.
- Added computed-mask segment2 update negative checks proving the target
  validator rejects a missing `riscv_vector.h` route header, stale mask type
  mapping, and stale `src0` ABI value mapping before artifact acceptance.
- Recorded PRD audit/evidence showing the selected executable boundary is
  provider-owned, with metadata remaining mirror-only after route validation.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-commit` | rvv: prove segment2 update artifact abi boundary |

### Testing

- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'computed-masked-segment2-update'` from `build/test`: 5/5 passed
- [OK] Non-dry-run `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_masked_segment2_update_unit_load` passed with `ssh_evidence: true`
- [OK] `ssh rvv` PASS covered counts `0,1,7,16,23,257`, patterns `0,1`, active/inactive lanes, inactive preservation, field-distinguishing lanes, source preservation, and tail preservation

### Status

[OK] **Completed and archived**

### Next Steps

- None - task complete

## Session 491: Stage2 RVV segment2 unit-memory executable artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV segment2 unit-memory artifact ABI boundary
**Branch**: `main`

### Summary

Hardened focused target artifact evidence for the plain segment2 unit-memory
interleave/deinterleave executable ABI seam. The production validator already
consumes `RVVSegment2MemoryRouteValidationContract`; this round added rebuilt
route payload negative coverage for plain segment2 and collected non-dry-run
`ssh rvv` generated-bundle correctness evidence for both routes.

### Main Changes

- Created the Trellis PRD for the segment2 unit-memory artifact ABI boundary.
- Added target C++ checks proving plain segment2 deinterleave rejects a missing
  rebuilt `riscv_vector.h` header, stale vector type mapping, and stale `src`
  ABI value mapping.
- Added target C++ checks proving plain segment2 interleave rejects a missing
  rebuilt `riscv_vector.h` header, stale vector type mapping, and stale `src0`
  ABI value mapping.
- Ran non-dry-run generated-bundle evidence on `ssh rvv` for
  `segment2_interleave_unit_load` and `segment2_deinterleave_unit_store`,
  covering runtime counts `0,1,7,16,23,257` for interleave and
  `0,1,7,16,17,23,257` for deinterleave.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-commit` | rvv: prove segment2 unit memory artifact abi |

### Testing

- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'segment2-(interleave-unit-load|deinterleave-unit-store)'` from `build/test`: 8/8 passed
- [OK] Non-dry-run `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind segment2_interleave_unit_load` passed with `ssh_evidence: true`
- [OK] Non-dry-run `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind segment2_deinterleave_unit_store` passed with `ssh_evidence: true`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded staged added-line old-authority scan over the touched C++ test

### Status

[OK] **Ready to archive and commit**

### Next Steps

- Archive task and create the final commit.


## Session 493: Stage2 RVV scalar-broadcast elementwise artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV scalar-broadcast elementwise artifact ABI boundary
**Branch**: `main`

### Summary

Hardened scalar-broadcast sub/mul operand binding from mirror-only tokens to executable abi/hdr provider facts; added fail-closed and pre-realized mul generated-bundle coverage; verified C++ tests, scalar-broadcast lit, and non-dry-run ssh rvv add/sub/mul evidence.

### Main Changes

- Created and archived Trellis task `06-06-stage2-rvv-widening-conversion-artifact-abi`.
- Audited widening conversion selected-body realization, route-provider
  preflight, conversion dtype-policy route-family owner, target artifact
  validation, generated-bundle harness, and existing fail-closed fixtures.
- Found no production source gap: provider and target code already validate
  source/result dtype, SEW/LMUL, conversion relation, runtime AVL/VL,
  ABI/header/type mappings, statement leaves, and stale non-conversion mirrors
  before executable artifact acceptance.
- Produced non-dry-run `ssh rvv` generated-bundle evidence for pre-realized
  `widen_i16_to_i32` and `widen_i32_to_i64` over counts `0,1,16,23,257` and
  two input patterns.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-commit` | (see git log) |

### Testing

- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] pre-realized widening conversion generated-bundle dry-run for
  `widen_i16_to_i32` and `widen_i32_to_i64`
- [OK] explicit selected-body `widen_i32_to_i64` generated-bundle dry-run
- [OK] direct pre-realized route-entry shortcut fail-closed for
  `widen_i16_to_i32` and `widen_i32_to_i64`
- [OK] non-dry-run `ssh rvv` generated-bundle run for pre-realized
  `widen_i16_to_i32` and `widen_i32_to_i64`
- [WARN] local `lit` runner is not installed; checked the relevant lit
  command surfaces directly.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 494: Stage2 RVV scalar-broadcast MAcc executable artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV scalar-broadcast MAcc executable artifact ABI boundary
**Branch**: `main`

### Summary

Audited the pre-realized scalar-broadcast MAcc executable artifact ABI seam and
confirmed the production provider/target/script path is already provider-owned,
fail-closed, and executable. Added the missing pre-realized generated-bundle
dry-run positive lit test and collected non-dry-run `ssh rvv` correctness
evidence.

### Main Changes

- Created the Trellis PRD for the scalar-broadcast MAcc executable artifact ABI
  boundary.
- Confirmed no production source change was justified: the route already
  realizes `typed_macc_pre_realized_body` into load/splat/load/MAcc/store,
  derives runtime ABI order `lhs,rhs_scalar,acc,out,n`, emits `abi|hdr`
  operand facts, and is validated by the MAcc target artifact family validator.
- Added `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-scalar-broadcast-macc-add-dry-run.test`
  to lock positive pre-realized generated-bundle evidence.
- Ran non-dry-run generated-bundle evidence on `ssh rvv` for
  `scalar_broadcast_macc_add`, covering runtime counts `0,1,16,17,257`, RHS
  scalars `-37,91`, explicit accumulator use, signed products, scalar-broadcast
  RHS, and tail sentinel preservation.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-commit` | rvv: prove scalar broadcast macc artifact abi |

### Testing

- [OK] `ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter "pre-realized-scalar-broadcast-macc-add"` from `build/test`: 2/2 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter "scalar-broadcast-macc-add"` from `build/test`: 5/5 passed
- [OK] non-dry-run `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind scalar_broadcast_macc_add` passed with `ssh_evidence: true`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded staged added-line old-authority scan over touched files

### Status

[OK] **Ready to archive and commit**

### Next Steps

- Archive task and create the final commit.


## Session 489: Stage2 RVV computed-masked MAcc artifact ABI evidence

**Date**: 2026-06-06
**Task**: Stage2 RVV computed-masked MAcc artifact ABI evidence
**Branch**: `main`

### Summary

Archived the computed-masked MAcc artifact ABI task, audited the production seam as already provider/target valid, and proved pre-realized computed-masked MAcc generated-bundle execution on ssh rvv without source changes.

### Main Changes

- Created and archived the Trellis task
  `stage2-rvv-widening-product-reduce-dequant-clamp-f32-artifact-abi`.
- Audited the selected-body realization owner, contraction family plan owner,
  route provider, target artifact validator, generated-bundle script, and
  explicit/pre-realized fixtures for
  `widening_product_reduce_dequant_clamp_f32`.
- Made no production source changes: the audited seam was already
  provider-owned and fail-closed.
- Produced explicit and pre-realized generated-bundle dry-run and non-dry-run
  `ssh rvv` evidence covering counts `0,1,16,17,257`, patterns `0,1`, scales
  `-0.125,0.375`, and bound pairs `-1.5:2.25,-8:-0.75`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | rvv: prove computed masked macc artifact abi |

### Testing

- [OK] `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_masked_macc_add --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id pre-realized-computed-masked-macc-add --overwrite`
- [OK] `ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter computed-masked-macc-add` from `build/test`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 490: Stage2 RVV compare/select artifact ABI

**Date**: 2026-06-06
**Task**: Stage2 RVV compare/select artifact ABI
**Branch**: `main`

### Summary

Hardened and proved the pre-realized compare/select generated-bundle artifact ABI boundary with tail-preservation evidence.

### Main Changes

- Created Trellis task stage2-rvv-compare-select-artifact-abi from the Hermes direction brief.
- Audited the production compare/select provider, route planning, statement-plan owner, and target artifact validator seam. The C++ seam already consumes structural compare predicate, mask provenance, select role, runtime AVL/VL, header/type, and ABI facts before accepting artifact routes.
- Fixed the plain cmp_select generated-bundle harness so it allocates a guard region after runtime n, initializes tail sentinels, fails with a targeted touched-tail diagnostic if the artifact writes past n, and prints tail_preserved after verification.
- Updated explicit and pre-realized cmp-select dry-run lit tests to assert the tail-sentinel failure diagnostic and tail_preserved success marker.
- Verified focused dry-run, focused lit, build targets, C++ smoke tests, and non-dry-run ssh rvv evidence for plain and selected compare/select family routes.


### Git Commits

(No commits - planning session)

### Testing

- [OK] `ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] explicit `widening_product_reduce_dequant_clamp_f32` dry-run generated-bundle command
- [OK] pre-realized `widening_product_reduce_dequant_clamp_f32` dry-run generated-bundle command
- [OK] explicit `widening_product_reduce_dequant_clamp_f32` non-dry-run `ssh rvv` generated-bundle command
- [OK] pre-realized `widening_product_reduce_dequant_clamp_f32` non-dry-run `ssh rvv` generated-bundle command
- [OK] manual stale `lower_bound` operand-binding mirror fail-closed check
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter widening-product-reduce-dequant-clamp-f32`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 491: Stage2 RVV widening conversion artifact ABI

**Date**: 2026-06-06
**Task**: Stage2 RVV widening conversion artifact ABI
**Branch**: `main`

### Summary

Audited and proved pre-realized RVV widening conversion generated artifact ABI boundary with dry-run, fail-closed, C++ checks, and ssh rvv evidence; no production source change required.

### Main Changes

- Created and archived the Trellis task/PRD for the plain MAcc add executable
  artifact ABI boundary.
- Confirmed no compiler source change was required: the existing production path
  already uses selected `tcrv_rvv.macc` body facts, provider-owned MAcc route
  facts, `TCRVEmitCLowerableRoute`, Common EmitC materialization, and target
  artifact validation for ABI/header/type/runtime/statement facts.
- Produced non-dry-run `ssh rvv` evidence for explicit and pre-realized
  generated bundles, covering counts `0,1,16,17,257`, patterns `0,1`,
  accumulator contribution, signed products, source preservation, and tail
  preservation.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] 5/5 lit tests selected by the plain MAcc generated-bundle/fixture/fail-closed filter
- [OK] non-dry-run generated-bundle `ssh rvv` correctness evidence for explicit and pre-realized `macc_add`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded old-authority scan over staged added lines

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 492: Stage2 RVV dequantize i32-to-f32 Gearbox artifact ABI

**Date**: 2026-06-06
**Task**: Stage2 RVV dequantize i32-to-f32 Gearbox artifact ABI
**Branch**: `main`

### Summary

Audited and proved explicit selected-body RVV dequantize i32-to-f32 Gearbox generated artifact ABI boundary with dry-run, fail-closed, C++ checks, and ssh rvv evidence; no production source change required.

### Main Changes

- Created and archived the Trellis task/PRD for the computed-masked widening
  dot-reduce-add executable artifact ABI boundary.
- Audited the RVV contraction selected-body realization validator, direct
  contraction provider facts, direct statement-plan owner, route provider,
  target artifact validation, generated-bundle runner, and explicit/pre-realized
  computed-mask fixtures.
- Recorded a no-source-change conclusion: current production code already
  derives compare-produced mask facts, inactive-lane zeroing, i16/mf2 dot
  source roles, i32/m1 scalar seed/result boundary, runtime AVL/VL,
  `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n` ABI order, provider operand bindings,
  and target validation facts from RVV-owned typed body/config/runtime facts.
- Re-ran explicit and pre-realized computed-mask generated bundle evidence on
  `ssh rvv`.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] explicit generated bundle `ssh rvv` run for
  `computed_masked_widening_dot_reduce_add`, counts `0,1,7,16,17,23,257`
- [OK] pre-realized generated bundle `ssh rvv` run for
  `computed_masked_widening_dot_reduce_add`, counts `0,1,7,16,17,23,257`
- [OK] focused lit filter for computed-mask widening dot-reduce target
  artifact, dry-run, and direct pre-realized fail-closed tests: 5/5 passed
- [OK] `git diff --check` and `git diff --cached --check`
- [OK] bounded old-authority scan found no production source changes and no
  positive legacy route authority in added task notes

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 493: Stage2 RVV widening product-reduce dequant-clamp f32 artifact ABI

**Date**: 2026-06-06
**Task**: Stage2 RVV widening product-reduce dequant-clamp f32 artifact ABI
**Branch**: `main`

### Summary

Audited the product-reduce dequant-clamp f32 generated artifact ABI seam, found production code already provider-owned and fail-closed, and proved explicit plus pre-realized generated bundles on ssh rvv with dry-run, fail-closed, C++ and lit evidence.

### Main Changes

- Created and archived the Trellis task/PRD for the computed-masked
  strided-input widening dot-reduce-add executable artifact ABI boundary.
- Audited the production direct contraction provider plan, statement owner,
  selected-body realization fixture path, target artifact validation contract,
  generated bundle runner, and explicit/pre-realized fixtures.
- Recorded a no-source-change conclusion: current production code already
  derives compare mask facts, element-strided i16/mf2 dot source loads,
  lhs/rhs stride ABI bindings, inactive-lane zeroing, i32 scalar seed/result,
  runtime AVL/VL, route operand bindings, header/type mappings, and target
  validation facts from RVV-owned typed body/config/runtime facts.
- Re-ran current-HEAD explicit and pre-realized generated bundle evidence on
  `ssh rvv`.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] explicit generated bundle `ssh rvv` run for
  `computed_masked_strided_input_widening_dot_reduce_add`, counts
  `0,1,7,16,17,23,257`, stride pairs `2:3,3:2`, mask patterns `0,1`,
  input patterns `0,1`
- [OK] pre-realized generated bundle `ssh rvv` run for
  `computed_masked_strided_input_widening_dot_reduce_add`, counts
  `0,1,7,16,17,23,257`, stride pairs `2:3,3:2`, mask patterns `0,1`,
  input patterns `0,1`
- [OK] focused lit filter for computed-masked strided-input widening
  dot-reduce target artifact, dry-run, and direct pre-realized fail-closed
  tests: 5/5 passed
- [OK] `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-06-stage2-rvv-computed-masked-strided-widening-dot-reduce-add-artifact-abi`
- [OK] `git diff --check` and `git diff --cached --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 494: Stage2 RVV MAcc add executable artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV MAcc add executable artifact ABI boundary
**Branch**: `main`

### Summary

Archived the MAcc add artifact ABI task after proving explicit and pre-realized generated bundles execute on ssh rvv with accumulator/product/source/tail preservation; no compiler source change required.

### Main Changes

- Extended `--tcrv-rvv-materialize-gearbox-schedules` to materialize bounded `tcrv_rvv.low_precision_resource.*` facts for `widening_product_reduce_dequantize_f32`.
- Wired provider route-family planning to consume pass-produced resource facts and reject missing/stale resource candidate fields before route acceptance.
- Copied pre-realized resource facts into realized `with_vl` bodies and allowed the new attrs on the owning RVV ops.
- Updated generated-bundle materialization and lit/C++ fixtures for the new pass/provider contract.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] targeted lit filter for Gearbox pass and product-dequant explicit/pre-realized target fixtures
- [OK] added-line legacy authority scan
- [OK] `git diff --check` and `git diff --cached --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 495: Stage2 RVV scalar-broadcast MAcc add executable artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV scalar-broadcast MAcc add executable artifact ABI boundary
**Branch**: `main`

### Summary

Revalidated scalar-broadcast MAcc add executable artifact ABI on current HEAD with explicit and pre-realized generated bundles on ssh rvv; no compiler source change required.

### Main Changes

- Created and archived the Trellis task for scalar-broadcast MAcc add artifact
  ABI evidence.
- Audited the production scalar-broadcast MAcc selected-body realization,
  provider facts, statement plan, target validation, script, and fixtures.
- Recorded a no-source-change conclusion: current production code already
  derives scalar RHS splat, accumulator/result roles, ABI/header bindings,
  runtime AVL/VL, and target metadata from provider-owned typed body/config/
  runtime facts.
- Re-ran current-HEAD explicit and pre-realized generated bundle evidence on
  `ssh rvv`.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] explicit generated bundle `ssh rvv` run for
  `scalar_broadcast_macc_add`, counts `0,1,7,16,23,257`, RHS scalars
  `-37,91`
- [OK] pre-realized generated bundle `ssh rvv` run for
  `scalar_broadcast_macc_add`, counts `0,1,7,16,23,257`, RHS scalars
  `-37,91`
- [OK] focused lit filter for scalar-broadcast MAcc dry-run, fail-closed,
  target artifact, and dispatch-negative tests: 6/6 passed
- [OK] `git diff --check`
- [OK] `git diff --cached --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 496: Stage2 RVV computed-masked MAcc add executable artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV computed-masked MAcc add executable artifact ABI boundary
**Branch**: `main`

### Summary

Revalidated computed-masked MAcc add executable artifact ABI on current HEAD
with explicit and pre-realized generated bundles on ssh rvv; no compiler source
change required.

### Main Changes

- Created the Trellis task for computed-masked MAcc add artifact ABI evidence.
- Audited the production computed-mask MAcc selected-body realization, provider
  facts, computed-mask accumulation statement plan, target validation, script,
  and fixtures.
- Recorded a no-source-change conclusion: current production code already
  derives compare mask, active MAcc, inactive accumulator preservation,
  accumulator/result roles, ABI/header bindings, runtime AVL/VL, and target
  metadata from provider-owned typed body/config/runtime facts.
- Re-ran current-HEAD explicit and pre-realized generated bundle evidence on
  `ssh rvv`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | rvv: prove computed masked macc add artifact abi |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] explicit generated bundle `ssh rvv` run for
  `computed_masked_macc_add`, counts `0,1,7,16,23,257`, patterns `0,1`
- [OK] pre-realized generated bundle `ssh rvv` run for
  `computed_masked_macc_add`, counts `0,1,7,16,23,257`, patterns `0,1`
- [OK] focused lit filter for computed-masked MAcc dry-run, fail-closed, and
  target artifact fixtures: 6/6 passed
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-stage2-rvv-computed-masked-macc-add-artifact-abi`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 498: Stage2 RVV widening dot-reduce-add executable artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV widening dot-reduce-add executable artifact ABI boundary
**Branch**: `main`

### Summary

Revalidated the base RVV widening dot-reduce-add executable artifact ABI seam
on current HEAD with explicit and pre-realized generated bundles on `ssh rvv`;
no compiler source change was required.

### Main Changes

- Created the Trellis task for base widening dot-reduce-add artifact ABI
  evidence.
- Audited the production contraction family provider facts, direct contraction
  statement owner, target artifact validation contract, script, and fixtures.
- Recorded a no-source-change conclusion: current production code already
  derives the i16/mf2 lhs/rhs source roles, i32/m1 accumulator/result boundary,
  scalar seed/result semantics, ABI/header bindings, runtime AVL/VL, and target
  validation facts from RVV-owned typed body/config/runtime facts.
- Re-ran current-HEAD explicit and pre-realized generated bundle evidence on
  `ssh rvv`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | rvv: prove widening dot reduce artifact abi |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] explicit generated bundle `ssh rvv` run for
  `widening_dot_reduce_add`, counts `0,1,7,16,17,23,257`, patterns `0,1`
- [OK] pre-realized generated bundle `ssh rvv` run for
  `widening_dot_reduce_add`, counts `0,1,7,16,17,23,257`, patterns `0,1`
- [OK] focused lit filter for widening dot-reduce target artifact, dry-run,
  and direct pre-realized fail-closed tests: 5/5 passed

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 499: Stage2 RVV strided-input widening dot-reduce-add executable artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV strided-input widening dot-reduce-add executable artifact ABI boundary
**Branch**: `main`

### Summary

Revalidated the RVV strided-input widening dot-reduce-add executable artifact
ABI seam on current HEAD with explicit and pre-realized generated bundles on
`ssh rvv`; no compiler source change was required.

### Main Changes

- Created and archived the Trellis task/PRD for the strided-input widening
  dot-reduce-add executable artifact ABI boundary.
- Audited the production contraction family provider facts, selected-body
  realization validator, statement owner, target artifact validation contract,
  generated bundle script, and strided explicit/pre-realized fixtures.
- Recorded a no-source-change conclusion: current production code already
  derives the i16/mf2 strided lhs/rhs source roles, i32/m1 accumulator/result
  boundary, lhs/rhs stride ABI bindings, scalar seed/result semantics,
  runtime AVL/VL, `abi|str|addr|hdr` operand bindings, and target validation
  facts from RVV-owned typed body/config/runtime facts.
- Re-ran current-HEAD explicit and pre-realized strided generated bundle
  evidence on `ssh rvv`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | rvv: prove strided widening dot reduce artifact abi |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] explicit generated bundle `ssh rvv` run for
  `strided_input_widening_dot_reduce_add`, counts `0,1,7,16,17,23,257`,
  stride pairs `2:3,3:2`, data patterns `0,1`
- [OK] pre-realized generated bundle `ssh rvv` run for
  `strided_input_widening_dot_reduce_add`, counts `0,1,7,16,17,23,257`,
  stride pairs `2:3,3:2`, data patterns `0,1`
- [OK] focused lit filter for strided-input widening dot-reduce target
  artifact, dry-run, and direct pre-realized fail-closed tests: 5/5 passed
- [OK] `git diff --check` and `git diff --cached --check`
- [OK] bounded old-authority scan over added task diff lines found no new
  positive legacy route authority

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 500: Stage2 RVV computed-mask widening dot reduce artifact ABI

**Date**: 2026-06-06
**Task**: Stage2 RVV computed-mask widening dot reduce artifact ABI
**Branch**: `main`

### Summary

Closed computed-masked widening dot-reduce-add executable artifact ABI boundary with explicit and pre-realized generated bundle ssh rvv evidence; no production source change required.

### Main Changes

- Extended RVV contraction route-family planning so computed-mask strided
  widening dot-reduce-add consumes a provider-selected low-precision resource
  before `TCRVEmitCLowerableRoute` construction.
- Added computed-mask strided resource candidate, legality, memory form,
  dtype/SEW/LMUL/EMUL, ABI order, runtime AVL, policy, and target mirror checks.
- Updated target artifact and generated-bundle tests for explicit and
  pre-realized selected bodies, including stale selected-candidate fail-closed
  coverage.
- Archived Trellis task
  `.trellis/tasks/archive/2026-06/06-06-06-06-rvv-computed-masked-strided-dot-resource-abi/`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] Built `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] `tianchenrv-rvv-extension-plugin-test` and
  `tianchenrv-target-artifact-export-test`.
- [OK] `scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Focused lit filter
  `computed-masked-strided-input-widening-dot-reduce-add` passed 5/5.
- [OK] Explicit and pre-realized generated-bundle dry-runs passed.
- [OK] Explicit and pre-realized `ssh rvv` runs passed for counts
  `0,1,16,17,257`, stride pairs `2:3` and `3:2`, and two mask/input patterns.
- [OK] `git diff --check` and bounded old-authority scan over touched files and
  added diff lines.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 501: Stage2 RVV computed-masked strided-input widening dot-reduce-add executable artifact ABI boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV computed-masked strided-input widening dot-reduce-add executable artifact ABI boundary
**Branch**: `main`

### Summary

Closed computed-masked strided-input widening dot-reduce-add executable artifact ABI boundary with explicit and pre-realized generated bundle ssh rvv evidence; no production source change required.

### Main Changes

- Created and archived the Trellis task/PRD for the computed-masked
  strided-input widening dot-reduce-add executable artifact ABI boundary.
- Audited the production direct contraction provider plan, statement owner,
  selected-body realization fixture path, target artifact validation contract,
  generated bundle runner, and explicit/pre-realized fixtures.
- Recorded a no-source-change conclusion: current production code already
  derives compare mask facts, element-strided i16/mf2 dot source loads,
  lhs/rhs stride ABI bindings, inactive-lane zeroing, i32 scalar seed/result,
  runtime AVL/VL, route operand bindings, header/type mappings, and target
  validation facts from RVV-owned typed body/config/runtime facts.
- Re-ran current-HEAD explicit and pre-realized generated bundle evidence on
  `ssh rvv`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] explicit generated bundle `ssh rvv` run for
  `computed_masked_strided_input_widening_dot_reduce_add`, counts
  `0,1,7,16,17,23,257`, stride pairs `2:3,3:2`, mask patterns `0,1`,
  input patterns `0,1`
- [OK] pre-realized generated bundle `ssh rvv` run for
  `computed_masked_strided_input_widening_dot_reduce_add`, counts
  `0,1,7,16,17,23,257`, stride pairs `2:3,3:2`, mask patterns `0,1`,
  input patterns `0,1`
- [OK] focused lit filter for computed-masked strided-input widening
  dot-reduce target artifact, dry-run, and direct pre-realized fail-closed
  tests: 5/5 passed
- [OK] `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-06-stage2-rvv-computed-masked-strided-widening-dot-reduce-add-artifact-abi`
- [OK] `git diff --check` and `git diff --cached --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 502: Stage2 RVV direct contraction artifact contract core

**Date**: 2026-06-06
**Task**: Stage2 RVV direct contraction artifact contract core
**Branch**: `main`

### Summary

Consolidated the provider-owned artifact validation core shared by MAcc and widening dot-reduce/direct-contraction paths, updated target validators to consume the core, documented the executable contract, archived the Trellis task, and passed focused C++ and lit checks.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `52bb0204` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 503: Stage2 RVV low-precision widening-product contract core

**Date**: 2026-06-06
**Task**: Stage2 RVV low-precision widening-product contract core
**Branch**: `main`

### Summary

Integrated standalone low-precision widening-product artifact validation with the shared RVV contraction artifact contract core and preserved owner-local widening-product facts.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `b344d149` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 504: Stage2 RVV product dequant family-plan mirror repair

**Date**: 2026-06-06
**Task**: Stage2 RVV product dequant family-plan mirror repair
**Branch**: `main`

### Summary

Resolved prior steering/spec dirty residue, repaired product-reduction dequantize f32 direct contraction family-plan type/config mirror, restored focused target artifact lit and generated-bundle dry-run evidence.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `2f483410` | (see git log) |
| `b9bb2569` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 505: Stage2 RVV low-precision resource candidate seed

**Date**: 2026-06-06
**Task**: Stage2 RVV low-precision resource candidate seed
**Branch**: `main`

### Summary

Added RVV low-precision direct-contraction resource selection contract, provider/target consumption, focused tests, and residue scanner fix.

### Main Changes

- Added RVV-owned low-precision direct-contraction resource selection facts to
  the contraction family/direct provider plans, selected route description, and
  target validation contract.
- Threaded the selected resource candidate into provider route acceptance,
  artifact metadata mirrors, header evidence, and stale metadata rejection.
- Added focused C++ and lit coverage for legal candidate construction, stale
  candidate mirrors, EMUL mismatch, vector pressure rejection, and artifact
  mirror consumption.
- Repaired direct-C residue scanners so `direct-contraction` is not rejected as
  source-export authority while true `direct-C` residue remains forbidden.

Final implementation commit is created after this journal entry so it can stay
in the same coherent commit as code, task archive, and test updates.

### Git Commits

(No commits - planning session)

### Testing

- [OK] `ninja -C build tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `lit -sv . --filter explicit-selected-body-artifact-widening-product-reduce-dequantize-f32`
- [OK] `lit -sv . --filter widening-product-reduce-dequant-clamp-f32`
- [OK] `lit -sv . --filter widening-product-reduce-dequantize-f32`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 506: Stage2 RVV resource-selected product dequant executable ABI

**Date**: 2026-06-06
**Task**: Stage2 RVV resource-selected product dequant executable ABI
**Branch**: `main`

### Summary

Closed widening_product_reduce_dequantize_f32 executable artifact ABI boundary with explicit and pre-realized generated-bundle ssh rvv correctness evidence; no production source change required.

### Main Changes

### Summary

Closed the Stage2 RVV resource-selected
`widening_product_reduce_dequantize_f32` executable artifact ABI boundary with
focused production seam audit, generated-bundle dry-run evidence, and explicit
plus pre-realized `ssh rvv` correctness evidence. No production source change
was required because the current provider/target path already consumes and
validates provider-selected low-precision resource facts before route
construction and artifact acceptance.

### Main Changes

- Created and completed the Trellis task
  `06-06-stage2-rvv-resource-selected-dequantize-f32-executable-abi`.
- Wrote the bounded PRD for this module owner, including non-goals, executable
  evidence requirements, and evidence artifact paths.
- Audited the RVV provider, direct-contraction resource selection, target
  artifact validation, support bundle export, and generated-bundle script seam.
- Recorded that no `.trellis/spec/` update is needed: the existing RVV
  low-precision resource-aware closure, EmitC route, emission runtime, and
  testing contracts already cover this round's evidence path.

### Evidence

- Explicit selected-body dry-run evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-explicit-dry/evidence.json`
- Pre-realized selected-body dry-run evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-prerealized-dry/evidence.json`
- Explicit selected-body `ssh rvv` correctness evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-explicit-ssh/evidence.json`
- Pre-realized selected-body `ssh rvv` correctness evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-prerealized-ssh/evidence.json`
- Both `ssh rvv` runs reported `success`, `ssh_evidence=true`, remote compile
  and run success, runtime counts `1,7,16,17,257`, two data patterns, scale
  values `-0.125,0.375`, and `source_preserved accumulator_preserved
  tail_preserved`.

### Testing

- [OK] `ninja -C build tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `lit -sv . --filter explicit-selected-body-artifact-widening-product-reduce-dequantize-f32`
- [OK] `lit -sv . --filter pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Explicit and pre-realized generated-bundle dry-runs for
  `widening_product_reduce_dequantize_f32`
- [OK] Explicit and pre-realized generated-bundle `ssh rvv` correctness runs
  for `widening_product_reduce_dequantize_f32`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Bounded added-diff old-authority scan

### Status

[OK] **Completed**

### Next Steps

- None - task complete


### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 507: Stage2 RVV resource-selected dequant-clamp executable ABI

**Date**: 2026-06-06
**Task**: Stage2 RVV resource-selected dequant-clamp executable ABI
**Branch**: `main`

### Summary

Closed the resource-selected widening_product_reduce_dequant_clamp_f32 executable artifact ABI boundary with focused seam audit, dry-run evidence, and explicit plus pre-realized ssh rvv correctness; no production source change required.

### Main Changes

### Summary

Closed the Stage2 RVV resource-selected `widening_product_reduce_dequant_clamp_f32` executable artifact ABI boundary. Production code did not require changes: the current RVV provider/target path already validates typed product/reduction/dequant/clamp facts, low-precision resource selection, runtime scale/lower/upper/out/n ABI bindings, route operand bindings, header/type mappings, provider mirror facts, and target artifact mirrors before bundle acceptance.

### Main Changes

- Created and archived Trellis task `06-06-stage2-rvv-resource-selected-dequant-clamp-f32-executable-abi`.
- Wrote the bounded PRD and implement/check context JSONL for this module owner.
- Audited the RVV contraction body validator, route planning/binding path, target artifact/construction bridge, generated-bundle harness, explicit fixture, and pre-realized fixture.
- Recorded that no `.trellis/spec/` update is needed because existing RVV plugin, EmitC route, testing, and variant-pipeline specs already cover this contract.

### Evidence

- Explicit dry-run: `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-clamp-explicit-dry/evidence.json`
- Pre-realized dry-run: `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-clamp-prerealized-dry/evidence.json`
- Explicit `ssh rvv`: `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-clamp-explicit-ssh/evidence.json`
- Pre-realized `ssh rvv`: `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-clamp-prerealized-ssh/evidence.json`
- Both `ssh rvv` runs reported `success`, `ssh_evidence=true`, counts `0,1,16,17,257`, patterns `0,1`, scale values `-0.125,0.375`, bound pairs `-1.5:2.25,-8:-0.75`, and `source_preserved accumulator_preserved tail_preserved`.

### Testing

- [OK] `ninja -C build tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `lit -sv . --filter widening-product-reduce-dequant-clamp-f32` from `build/test`: 4/4 filtered tests passed.
- [OK] Explicit and pre-realized generated-bundle dry-runs for `widening_product_reduce_dequant_clamp_f32`
- [OK] Explicit and pre-realized generated-bundle `ssh rvv` correctness for `widening_product_reduce_dequant_clamp_f32`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-stage2-rvv-resource-selected-dequant-clamp-f32-executable-abi` before archive
- [OK] `git diff --check`


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | rvv: prove product dequant clamp executable abi |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 508: Redirect RVV Gearbox wording worktree coherence

**Date**: 2026-06-06
**Task**: Redirect RVV Gearbox wording worktree coherence
**Branch**: `main`

### Summary

Classified and retained the stray RVV Gearbox wording/spec/artifact archive residue as a bounded coherence cleanup; repaired stale clamp evidence commit references to ccd029b2 and preserved route-authority boundaries.

### Main Changes

### Main Changes

- Created and archived Trellis task `06-06-rvv-gearbox-worktree-coherence`.
- Classified the dirty RVV Gearbox wording/spec/artifact/archive paths as valid to retain after fact repair.
- Corrected stale product-dequant-clamp evidence references from non-current `dec212f4` to current branch commit `ccd029b2` in the Gearbox artifact and archived wording PRD.
- Kept the retained wording bounded to current implementation calibration: Gearbox MVP pass, future resource-aware pass contract, provider/target validation consumption, and no completed-autotuner/performance/llama.cpp parity claim.
- Did not modify compiler source, tests, pass registration, route providers, artifact generation, runtime scripts, or ssh rvv evidence.

### Testing

- [OK] `python3 ./.trellis/scripts/get_context.py --mode packages`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-rvv-gearbox-worktree-coherence`
- [OK] retained wording stale-hash scan for `dec212f4`
- [OK] bounded added-line old-authority scan; matches were negative authority wording only
- [OK] `git diff --check`
- [FIXED] Post-archive `task.py validate` initially found redirect task JSONL
  entries pointing at the pre-archive PRD path; corrected them to the archive
  path and reran validation successfully before amending the commit.

### Status

[OK] **Archived; ready for one coherent commit**


### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 509: Stage2 RVV Gearbox low-precision resource facts

**Date**: 2026-06-06
**Task**: Stage2 RVV Gearbox low-precision resource facts
**Branch**: `main`

### Summary

Implemented RVV Gearbox low-precision resource candidate facts for product-reduce-dequantize bodies, wired provider consumption/fail-closed validation, added lit/C++ coverage, and archived the Trellis task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| this commit | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 510: Stage2 RVV Gearbox resource-fact artifact boundary

**Date**: 2026-06-06
**Task**: Stage2 RVV Gearbox resource-fact artifact boundary
**Branch**: `main`

### Summary

Repaired generated-bundle materialization order so pre-realized widening_product_reduce_dequantize_f32 runs the Gearbox resource-fact pass before selected-body realization, then proved explicit and pre-realized generated bundles through dry-run and non-dry-run ssh rvv correctness.

### Main Changes

- Created Trellis task `06-06-rvv-gearbox-resource-fact-artifact-boundary` from the Hermes brief and bounded it to the existing Gearbox low-precision resource-fact artifact/runtime boundary.
- Found a real production script seam bug: pre-realized `widening_product_reduce_dequantize_f32` generated-bundle evidence ran `--tcrv-materialize-selected-lowering-boundaries` before `--tcrv-rvv-materialize-gearbox-schedules`, so provider planning failed closed on missing pass-produced `tcrv_rvv.low_precision_resource.*` facts.
- Reordered `scripts/rvv_generated_bundle_abi_e2e.py` so product-dequant Gearbox resource facts are materialized before selected-body realization; kept plain `dequantize_i32_to_f32` ordering unchanged because that MVP path consumes realized `with_vl` structure.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the generated-bundle pass-order contract.
- Produced fresh evidence:
  - `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-gearbox-resource-fact-product-dequant-explicit-dry/evidence.json`
  - `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-gearbox-resource-fact-product-dequant-prerealized-dry/evidence.json`
  - `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-gearbox-resource-fact-product-dequant-explicit-ssh/evidence.json`
  - `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-gearbox-resource-fact-product-dequant-prerealized-ssh/evidence.json`
- Both non-dry-run ssh runs reported `status=success`, `ssh_evidence=true`, `ssh_target=rvv`, remote compile/run success, counts `1,7,16,17,257`, scales `-0.125,0.375`, two data patterns, and `source_preserved accumulator_preserved tail_preserved`.
- Checks: build targets, RVV plugin C++ smoke, target artifact C++ smoke, three focused lit filters via `/usr/lib/llvm-20/build/utils/lit/lit.py`, script self-test, explicit/pre-realized dry-runs, explicit/pre-realized ssh rvv runs, task validate, `git diff --check`, `git diff --cached --check`, and bounded added-line old-authority scan.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 511: Stage2 RVV strided dot resource path

**Date**: 2026-06-06
**Task**: Stage2 RVV strided dot resource path
**Branch**: `main`

### Summary

Implemented provider-consumed low-precision resource selection for base strided-input widening dot-reduce-add, added target/generated-bundle fail-closed and evidence checks, and proved explicit/pre-realized ssh rvv correctness.

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


## Session 512: Stage2 computed-mask strided dot resource ABI

**Date**: 2026-06-06
**Task**: Stage2 computed-mask strided dot resource ABI
**Branch**: `main`

### Summary

Completed and archived the Stage2 RVV computed-mask strided-input widening dot-reduce-add resource/artifact ABI boundary. Production route planning now requires provider-owned low-precision resource selection for the computed-mask strided direct dot route; target artifact and generated-bundle checks validate resource mirrors and stale selected candidates; explicit and pre-realized dry-runs and ssh rvv runs passed for counts 0,1,16,17,257 with stride pairs 2:3 and 3:2. Final source/task/journal commit is created after this entry.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `created-after-journal-entry` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 513: Stage2 RVV computed-mask segment2 load artifact ABI

**Date**: 2026-06-06
**Task**: Stage2 RVV computed-masked segment2 load executable artifact ABI boundary
**Branch**: `main`

### Summary

Completed the computed-masked segment2 load executable artifact/ABI boundary. The production provider/target seam already carried segment2 computed-mask route facts; this round closed the evidence gap by requiring mask-tail route-family plan/owner mirrors in generated-bundle evidence, adding target artifact stale mirror rejection for those fields, and proving explicit/pre-realized ssh rvv correctness.

### Main Changes

- Created Trellis task `06-06-rvv-computed-masked-segment2-load-artifact-abi` from the Hermes brief and bounded it to the computed-masked segment2 load seam only.
- Hardened `scripts/rvv_generated_bundle_abi_e2e.py` so computed-masked segment2 load expects and summarizes `tcrv_rvv.mask_tail_policy_route_family_plan` and `tcrv_rvv.mask_tail_policy_owner`.
- Updated explicit and pre-realized generated-bundle dry-run FileCheck tests and target artifact MLIR fixtures to expose the mask-tail plan/owner mirrors.
- Extended `test/Target/TargetArtifactExportTest.cpp` so positive computed-mask segment2 load contracts compare mask-tail plan/owner fields, and stale provider/candidate mask-tail mirrors fail closed before artifact acceptance.
- Produced fresh ssh rvv evidence under `/tmp/tianchenrv-rvv-cmseg-load-ssh`: explicit counts `0,1,16,17,257` and pre-realized counts `0,1,7,16,23,257`, both with patterns `0,1`, active/inactive lane coverage, field0/field1 distinction, old-field inactive-lane preservation, source preservation, and tail preservation.
- Checks: focused build targets, RVV plugin C++ smoke, target artifact C++ test, three generated-bundle dry-run lit tests, two target artifact fixture lit tests, script py_compile/self-test, explicit/pre-realized ssh rvv runs, `git diff --check`, and bounded added-line old-authority scan.

### Git Commits

| Hash | Message |
|------|---------|
| `created-after-journal-entry` | (see git log) |

### Testing

- [OK] Focused checks and ssh rvv evidence passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 515: Stage2 RVV pre-realized runtime-scalar segment2 store ABI

**Date**: 2026-06-07
**Task**: Stage2 RVV pre-realized runtime-scalar segment2 store ABI
**Branch**: `main`

### Summary

Implemented pre-realized runtime-scalar-cmp masked segment2 store selected-body realization and executable artifact ABI evidence; added typed pre-realized op, segment2 owner realization, target/generated-bundle tests, spec contract, focused C++/lit checks, and ssh rvv correctness.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `created-after-journal-entry` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 517: Stage2 RVV runtime-scalar indexed scatter ABI

**Date**: 2026-06-07
**Task**: Stage2 RVV runtime-scalar indexed scatter ABI
**Branch**: `main`

### Summary

Implemented runtime-scalar-cmp masked indexed scatter-store artifact ABI path with focused lit and ssh rvv evidence.

### Main Changes

Implemented the runtime-scalar-cmp masked indexed scatter-store seam from typed/pre-realized tcrv_rvv bodies through RVV plugin realization, provider route planning, EmitC metadata, target artifact export, generated bundle ABI, and ssh rvv correctness evidence.

Evidence:
- RVV extension plugin smoke test passed.
- Target artifact export smoke test passed.
- Focused lit filter runtime-scalar-cmp-masked-indexed-scatter-store passed 4/4.
- Explicit and pre-realized generated bundles compiled and ran on ssh rvv with counts 0,1,16,17,257; rhs scalars -37,91; patterns 0,1.
- Script self-test initially caught runtime-scalar scatter harness/metadata issues; fixed and reran to pass.


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
