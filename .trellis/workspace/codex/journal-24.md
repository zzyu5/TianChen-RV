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
