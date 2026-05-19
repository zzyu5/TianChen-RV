# Journal - codex (Part 11)

> Continuation from `journal-10.md` (archived at ~2000 lines)
> Started: 2026-05-19

---



## Session 133: Stage2 generic RVV reduction accumulation route

**Date**: 2026-05-19
**Task**: Stage2 generic RVV reduction accumulation route
**Branch**: `main`

### Summary

Added a bounded generic typed RVV reduce(add) selected-body route skeleton with verifier, construction protocol, provider materialization, artifact/header dry-run, fail-closed negatives, and no ssh-rvv runtime claim.

### Main Changes

- Added generic `tcrv_rvv.reduce` for a typed vector input plus typed accumulator/result under matching `tcrv_rvv.with_vl`; the bounded supported reduction kind is `add`.
- Extended RVV construction protocol route recognition with `reduce_add`, generic `tcrv_rvv.reduce`, and provider-owned `rvv-generic-reduce-add-*` route/runtime ABI identity.
- Extended the RVV EmitC route provider to derive a reduction leaf from typed body/config/runtime facts and to reject RHS-broadcast reduction in this slice.
- Added positive EmitC materialization and target header/artifact dry-run fixtures plus negative verifier/provider coverage.

### Testing

- [OK] Trellis task context validation.
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-construction-protocol-common-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused reduction lit set: 5/5 passed.
- [OK] focused existing first-slice EmitC regression lit set: 5/5 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 154/154.
- [OK] `git diff --check`
- [OK] active-authority scan: diff-only legacy-shaped addition is provider-derived `__riscv_vredsum_vs_i32m1_i32m1`; no new `tcrv_rvv.i32_reduction_*`, `tcrv_rvv.i32_accumulator_*`, or `tcrv_rvv.i32_macc` helper surface.

### Status

[OK] **Completed** as route-supported and artifact/header dry-run evidence. No `ssh rvv` correctness or performance claim was made in this round.

### Next Steps

- Future continuation, if requested: turn the reduction skeleton into executable runtime correctness evidence on `ssh rvv` and refine accumulator/result layout beyond this bounded vector accumulator form.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |


## Session 134: Stage1 Gate A RVV route identity cleanup

**Date**: 2026-05-19
**Task**: `stage1-gate-a-rvv-route-identity-cleanup`
**Branch**: `main`

### Summary

Closed the bounded Stage1 Gate A route-identity residues for the active RVV selected-body route path. Route ids, runtime ABI names, artifact route/header/bundle names, config-contract APIs, source-front-door pass naming, and pre-realized selected-body entry points now use generic selected-body / typed-body authority rather than `rvv-i32m1`, `RVVI32M1`, or `i32_binary_pre_realized_body` as positive route identity.

### Main Changes

- Renamed active RVV construction identities to `rvv-generic-typed-body-*` and per-operation provider-derived `rvv-generic-binary-*`, `rvv-generic-cmp-select-*`, and `rvv-generic-reduce-add-*` route/runtime ABI labels.
- Renamed active config-contract APIs and metadata to selected-body generic names while preserving SEW32/LMUL/policy facts as typed config facts.
- Replaced positive `tcrv_rvv.i32_binary_pre_realized_body` fixtures with `tcrv_rvv.typed_binary_pre_realized_body`.
- Kept retained i32 add/sub/mul as ordinary generic typed `tcrv_rvv.binary {kind = ...}` instances, and kept legacy `tcrv_rvv.i32_*` selected-body snippets only as negative fail-closed fixtures.
- Renamed the RVV source-front-door pass to a fail-closed legacy vector-source front-door identity.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] focused lit filter for `Target/RVV`, RVV lowering/source-front-door transforms, and RVV script dry-runs: 40/40 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 154/154 passed.
- [OK] `git diff --check`
- [OK] exact Stage1 Gate A brief scan over `include/TianChenRV lib/Plugin/RVV lib/Dialect/RVV test/Target/RVV`: no `rvv-i32m1`, `RVVI32M1`, or `i32_binary_pre_realized_body` matches.
- [OK] remaining extended legacy matches are classified as negative fail-closed Target/RVV fixtures, dialect parse/verifier debt, and provider-derived intrinsic leaf spellings.

### Status

[OK] **Completed and ready to archive**. No `ssh rvv` runtime, correctness, or performance claim was made.

### Next Steps

- Future continuation, if requested: delete or further fail-close the remaining parseable legacy dialect debt under a separate Stage1 deletion task; do not treat it as Stage2 coverage work.
