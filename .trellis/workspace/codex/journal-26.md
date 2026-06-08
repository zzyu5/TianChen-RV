# Journal - codex (Part 26)

> Continuation from `journal-25.md` (archived at ~2000 lines)
> Started: 2026-06-08

---



## Session 550: Stage2 RVV strided-input widening dot-reduce executable artifact ABI boundary

**Date**: 2026-06-08
**Task**: Stage2 RVV strided-input widening dot-reduce executable artifact ABI boundary
**Branch**: `main`

### Summary

Closed the executable evidence gap for the base strided-input widening dot-reduce route. Current production code already derives stride ABI, strided memory, widened dot/reduction, accumulator/seed, scalar result, runtime AVL/VL, and header/prototype facts through RVV-owned plan/validation; this round proved the generated bundle on ssh rvv without compiler source changes.

### Main Changes

- Created and completed Trellis task `06-08-stage2-rvv-strided-input-widening-dot-reduce-executable-artifact-abi-boundary`.
- Recorded no-source-change justification in the PRD: pre-realized strided-input validator requires `lhs_stride`/`rhs_stride` ABI roles and strided-load body shape; contraction route-family plan and operand binding plan carry `lhs_stride`/`rhs_stride`; statement owner verifies strided-input facts and strided source-load leaf before `TCRVEmitCLowerableRoute`; target artifact validation requires strided mirrors to match provider facts exactly.
- Proved generated bundle execution on `ssh rvv` for pre-realized `strided_input_widening_dot_reduce_add`: counts `0,1,16,17,257`, stride pairs `2:3` and `3:2`, two data patterns, signed widened i16*i16 products, seed contribution, skipped source elements ignored, scalar-output-only behavior, and tail preservation.
- Kept computed-masked and computed-masked-strided dot-reduce as bounded references only; no expansion in this round.
- No `.trellis/spec/` update: existing RVV plugin, EmitC route, and MLIR testing specs already define this artifact/ABI evidence boundary.

Focused checks:
- [OK] dry-run generated bundle: `strided_input_widening_dot_reduce_add`
- [OK] fail-closed direct pre-realized route-entry shortcut: `--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): strided_input_widening_dot_reduce_add`
- [OK] non-dry-run `ssh rvv` generated bundle: `PASS op=strided_input_widening_dot_reduce_add counts=0,1,16,17,257 stride_pairs=2:3,3:2 data_patterns=2`
- [OK] remote evidence: `remote_arch=riscv64`, `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`, `ssh_evidence=true`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] filtered lit for explicit/pre-realized strided-input dry-run and direct pre-realized fail-closed tests: `Passed 3, Excluded 528`
- [OK] provider residue scan: no direct-contraction provider-plan getter or verifier call remains in `RVVEmitCRouteProvider.cpp`
- [OK] `git diff --check`


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


## Session 551: Stage2 RVV computed-masked strided widening dot artifact ABI

**Date**: 2026-06-08
**Task**: Stage2 RVV computed-masked strided widening dot artifact ABI
**Branch**: `main`

### Summary

Closed the executable evidence gap for the computed-masked strided-input
widening dot-reduce route. Production C++ already consumes the combined
computed-mask, strided-input, widening dot/reduction, accumulator/result,
runtime AVL/VL, ABI order, header/type, and target mirror facts through RVV
owners/providers/target validation; this round hardened the generated bundle
harness so runtime evidence also proves compare/source/accumulator preservation
instead of only scalar output correctness.

### Main Changes

- Created and completed Trellis task
  `06-08-stage2-rvv-computed-masked-strided-widening-dot-artifact-abi`.
- Updated `scripts/rvv_generated_bundle_abi_e2e.py` for
  `computed_masked_strided_input_widening_dot_reduce_add`: the generated harness
  snapshots `cmp_lhs`, `cmp_rhs`, `lhs`, `rhs`, and `acc`; computes expected
  results from snapshots; checks compare/source/accumulator buffers after the
  generated call; and prints `source_preserved accumulator_preserved
  tail_preserved` only after those checks pass.
- Updated explicit and pre-realized generated-bundle dry-run tests to check
  snapshot buffers, mutation diagnostics, and the strengthened preservation
  success markers.
- Proved the pre-realized generated bundle on `ssh rvv` for counts
  `0,1,16,17,257`, stride pairs `2:3` and `3:2`, mask/input patterns `0` and
  `1`, signed widened dot, seed contribution, inactive-lane skipping,
  skipped-source ignoring, scalar-output-only behavior, and
  source/accumulator/tail preservation.
- No `.trellis/spec/` update: existing RVV plugin, EmitC route, and MLIR
  testing specs already require provider-owned route authority and runtime
  preservation evidence.

Focused checks:
- [OK] script dry-run:
  `computed_masked_strided_input_widening_dot_reduce_add`
- [OK] filtered lit for explicit/pre-realized computed-masked strided-input
  dry-run and direct pre-realized fail-closed tests: `Total Discovered Tests: 3`
- [OK] non-dry-run `ssh rvv` generated bundle:
  `PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,1,16,17,257 stride_pairs=2:3,3:2 mask_patterns=2 input_patterns=2 source_preserved accumulator_preserved tail_preserved`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] bounded added-line residue scan for legacy i32/source-front-door/common
  EmitC authority
- [OK] `git diff --check`
- [OK] `git diff --cached --check`

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Status

[OK] **Completed**

### Next Steps

- None - task complete
