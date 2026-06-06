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

(Add details)

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

(Add details)

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
