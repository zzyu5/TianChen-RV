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
