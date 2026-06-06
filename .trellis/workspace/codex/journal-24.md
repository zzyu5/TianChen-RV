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
