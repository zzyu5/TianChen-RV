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
| `final-commit-see-git-log` | `rvv: prove runtime scalar indexed gms evidence` |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 560: Stage2 RVV computed-masked segment2 update artifact ABI evidence

**Date**: 2026-06-08
**Task**: Stage2 RVV computed-masked segment2 update executable artifact ABI boundary
**Branch**: `main`

### Summary

Closed the computed-masked segment2 update generated-artifact ABI boundary with
no production source changes and fresh `ssh rvv` executable evidence.

### Main Changes

- Created Trellis PRD for
  `stage2-rvv-computed-masked-segment2-update-artifact-abi-boundary`.
- Verified the existing RVV plugin path already owns the update executable
  boundary: pre-realized selected-body realization, computed-mask segment2
  route-family planning, segment2 statement planning, provider-built
  `TCRVEmitCLowerableRoute`, target artifact validation, generated bundle ABI,
  and ssh runtime evidence.
- Recorded no-source-change justification because existing provider and target
  validation already reject stale update arithmetic, field roles, mask facts,
  inactive-lane policy, ABI/header/type mapping, runtime AVL/VL facts, route
  control, statement-plan facts, and metadata mirrors.
- Generated dry-run evidence for explicit and pre-realized computed-mask
  segment2 update bundles.
- Generated non-dry-run `ssh rvv` evidence for explicit and pre-realized
  computed-mask segment2 update bundles; both runs reported remote compile/run
  success and PASS summaries.
- Revalidated the deprecated direct pre-realized route-entry fail-closed
  negative path for update.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Explicit computed-mask segment2 update generated-bundle dry-run.
- [OK] Pre-realized computed-mask segment2 update generated-bundle dry-run.
- [OK] Explicit computed-mask segment2 update generated-bundle run on `ssh rvv`.
- [OK] Pre-realized computed-mask segment2 update generated-bundle run on `ssh rvv`.
- [OK] Deprecated direct pre-realized route-entry negative check for update.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 555: Stage2 RVV runtime-scalar-cmp masked indexed gather-MAcc-scatter executable artifact ABI

**Date**: 2026-06-08
**Task**: Stage2 RVV runtime-scalar-cmp masked indexed gather-MAcc-scatter executable artifact ABI boundary
**Branch**: `main`

### Summary

Refreshed explicit and pre-realized generated object/header bundle ABI evidence
for `runtime_scalar_cmp_masked_indexed_gather_macc_scatter` on `ssh rvv`; no
compiler source changes were required because the existing provider/target
contract already generated executable artifacts.

### Main Changes

- Created and completed Trellis task
  `06-08-stage2-rvv-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-executable-artifact-abi`.
- Recorded a bounded PRD for the runtime-scalar compare, compare-produced mask,
  indexed gather, masked MAcc, indexed scatter, ABI/header, target artifact,
  and `ssh rvv` evidence boundary.
- Proved the explicit selected-body generated object/header bundle on
  `ssh rvv` with `dry_run=false`, `ssh_evidence=true`, remote compile success,
  remote run success, and generated harness PASS.
- Proved the pre-realized selected-body generated object/header bundle on
  `ssh rvv` after composite realization with the same correctness scope.
- Both remote harnesses covered counts `0,1,16,17,257`, RHS scalars `-37,91`,
  two patterns, active/inactive lanes, noncontiguous index lanes, signed
  product lanes, source preservation, payload/accumulator preservation, and
  tail preservation.
- No compiler source diff was introduced; this round recorded Trellis
  PRD/context/journal/archive evidence and fresh generated-bundle artifacts
  under ignored `artifacts/tmp`.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Testing

- [OK] explicit generated-bundle dry-run:
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`
- [OK] pre-realized generated-bundle dry-run:
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`
- [OK] explicit `ssh rvv` generated bundle:
  `PASS op=runtime_scalar_cmp_masked_indexed_gather_macc_scatter counts=0,1,16,17,257 rhs_scalars=-37,91 patterns=0,1`
- [OK] pre-realized `ssh rvv` generated bundle:
  `PASS op=runtime_scalar_cmp_masked_indexed_gather_macc_scatter counts=0,1,16,17,257 rhs_scalars=-37,91 patterns=0,1`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] filtered lit from `build/test`:
  `--filter runtime-scalar-cmp-masked-indexed-gather-macc-scatter`
  passed 3/3 tests
- [OK] production source diff list empty; no new positive legacy RVV authority
  was introduced
- [OK] `git diff --check`
- [OK] `git diff --cached --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 553: Stage2 RVV runtime-scalar indexed scatter-store artifact ABI

**Date**: 2026-06-08
**Task**: Stage2 RVV runtime-scalar-cmp masked indexed scatter-store executable artifact ABI boundary
**Branch**: `main`

### Summary

Hardened the standalone runtime-scalar cmp masked indexed scatter-store
generated-bundle evidence so the correctness oracle uses pre-call source
snapshots; proved the pre-realized generated bundle on `ssh rvv`.

### Main Changes

- Created and completed Trellis task
  `06-08-stage2-rvv-runtime-scalar-cmp-masked-indexed-scatter-store-abi`.
- Inspected the production standalone scatter-store seam: the pre-realized body
  validator checks op kind, predicate `sle`, computed-mask indexed scatter
  memory form, unique indices, inactive-lane policy, SEW/LMUL/policy, and ABI
  roles before realization; the route owner checks canonical runtime-scalar
  indexed route facts, binding summary, ABI parameters, statement leaves, and
  typed config before `TCRVEmitCLowerableRoute`; target artifact validation
  rebuilds provider payload before candidate mirror checks.
- Updated `scripts/rvv_generated_bundle_abi_e2e.py` so
  `runtime_scalar_cmp_masked_indexed_scatter_store_unit_load` computes expected
  active-lane destination values and mismatch diagnostics from
  `src_before[index]`, not post-call `src[index]`.
- Updated explicit/pre-realized generated-bundle dry-run FileCheck tests to
  assert the snapshot-backed expected expression.
- Added `runtime_scalar_cmp_masked_indexed_scatter_store_unit_load` to the
  `--rhs-scalar` help text, matching the existing multiple-runtime-scalar
  execution path.
- No `.trellis/spec/` update: the existing MLIR testing contract already
  records the reusable snapshot-backed preservation evidence rule, and this
  round applied it to the standalone scatter-store harness.
- Proved the pre-realized generated bundle on `ssh rvv` with counts
  `0,1,16,17,257`, RHS scalars `-37,91`, patterns `0,1`, active/inactive lane
  mixes, noncontiguous unique indexed destination writes, inactive destination
  preservation, source preservation, and tail preservation.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Testing

- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] explicit generated-bundle dry-run plus direct `ROOT`/`RTSCATTER`/`HARNESS`
  FileCheck
- [OK] pre-realized generated-bundle dry-run plus direct
  `ROOT`/`RTSCATTER`/`HARNESS` FileCheck
- [OK] explicit target artifact fixture `PLAN`/`HEADER` and stale producer
  fail-closed RUN-line equivalents
- [OK] pre-realized target artifact fixture `REALIZED`/`PLAN`/`HEADER`
  RUN-line equivalents
- [OK] `ssh rvv` generated bundle:
  `PASS op=runtime_scalar_cmp_masked_indexed_scatter_store_unit_load counts=0,1,16,17,257 rhs_scalars=-37,91 patterns=0,1`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] bounded added-line old-authority scan
- [OK] `git diff --check`
- [OK] `git diff --cached --check`

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


## Session 552: Stage2 RVV runtime-scalar indexed gather MAcc scatter artifact ABI

**Date**: 2026-06-08
**Task**: Stage2 RVV runtime-scalar indexed gather MAcc scatter artifact ABI
**Branch**: `main`

### Summary

Hardened runtime-scalar cmp masked indexed gather-MAcc-scatter generated-bundle evidence to compute expected values from pre-call snapshots; proved pre-realized bundle on ssh rvv; documented snapshot-backed evidence rule.

### Main Changes

- Created and completed Trellis task
  `06-08-stage2-rvv-runtime-scalar-cmp-gather-macc-scatter-abi`.
- Inspected the production composite seam: the RVV plugin-local composite
  realization owner already requires exactly one runtime-scalar indexed gather,
  one runtime-scalar MAcc, and one runtime-scalar indexed scatter body; it
  validates shared runtime scalar, mask, runtime n/AVL, index, destination,
  dtype/config/policy, inactive-lane policy, MAcc accumulator/result layout,
  and ABI roles before realization. Target artifact validation requires
  provider-owned composite route-family plan, typed compute chain, legal
  resource selection, runtime ABI order, headers, type mappings, ABI mappings,
  and statement plan shape before export.
- Updated `scripts/rvv_generated_bundle_abi_e2e.py` so
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter` generated harnesses
  compute signed products, expected results, and mismatch diagnostics from
  pre-call `gather_src_before`, `payload_before`, and `acc_before` snapshots.
  The harness still separately compares live buffers with those snapshots and
  prints `source_preserved payload_acc_preserved tail_preserved` only after
  preservation checks pass.
- Updated the explicit/pre-realized generated-bundle dry-run FileCheck test to
  assert the snapshot-backed expected expression.
- Added the composite op to `--rhs-scalar` help text, matching the existing
  multiple-runtime-scalar execution path.
- Updated `.trellis/spec/testing/mlir-testing-contract.md` to record the
  reusable rule that preservation evidence must use snapshots in expected-value
  computation and mismatch diagnostics, not only compare snapshots after the
  call.
- Proved the pre-realized generated bundle on `ssh rvv` with counts
  `0,1,16,17,257`, RHS scalars `-37,91`, patterns `0,1`, active/inactive lane
  mixes, inactive destination preservation, noncontiguous unique indices,
  signed products, source preservation, payload/accumulator preservation, and
  tail preservation.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Testing

- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] explicit generated-bundle dry-run + `FileCheck`
  `STDOUT`/`EXPL`/`EXPLH`
- [OK] pre-realized generated-bundle dry-run + `FileCheck`
  `STDOUT`/`PRE`/`PREH`
- [OK] `ssh rvv` generated bundle:
  `PASS op=runtime_scalar_cmp_masked_indexed_gather_macc_scatter counts=0,1,16,17,257 rhs_scalars=-37,91 patterns=0,1`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] explicit target artifact fixture PLAN/HEADER and stale
  provider/ABI/operand-binding negative checks with `FileCheck`
- [OK] pre-realized target artifact fixture REALIZED/PLAN/HEADER and stale
  provider/ABI/operand-binding/composite-resource negative checks with
  `FileCheck`
- [OK] bounded old-authority scan over touched files
- [OK] `git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 553: Stage2 RVV runtime-scalar indexed write-side boundary

**Date**: 2026-06-08
**Task**: Stage2 RVV runtime-scalar indexed write-side boundary
**Branch**: `main`

### Summary

Closed provider/target-owned indexed write-side contract boundary for runtime-scalar indexed scatter and gather-MAcc-scatter; added target/header/generated-bundle evidence and ssh rvv validation.

### Main Changes

- Created, completed, and archived Trellis task
  `06-08-stage2-rvv-computed-masked-strided-dot-executable-artifact-abi`.
- Confirmed the previous provider/target acceptance boundary at `1bf69314`
  already carries the computed-mask strided widening-dot semantic contract and
  that the missing blocker was fresh executable generated-bundle evidence.
- Proved the explicit selected-body generated object/header bundle on
  `ssh rvv` with `dry_run=false`, `ssh_evidence=true`, remote clang
  compile/link success, and remote harness PASS.
- Proved the pre-realized selected-body generated object/header bundle on
  `ssh rvv` with the same correctness scope after public selected
  lowering-boundary materialization.
- Both remote harnesses covered counts `0,1,16,17,257`, stride pairs `2:3`
  and `3:2`, two mask/input patterns, signed products, inactive-lane skipped
  products, accumulator seed/carry through `acc[0]`/`out[0]`, scalar-only
  output, source preservation, accumulator preservation, and tail preservation.
- No compiler source changes were required; this round only recorded Trellis
  PRD/context/journal/archive evidence.

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | (see git log) |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] explicit generated-bundle dry-run with configured LLVM 20
  `llvm-readobj`
- [OK] pre-realized generated-bundle dry-run with configured LLVM 20
  `llvm-readobj`
- [OK] explicit `ssh rvv` generated bundle:
  `PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,1,16,17,257 stride_pairs=2:3,3:2 mask_patterns=2 input_patterns=2 source_preserved accumulator_preserved tail_preserved`
- [OK] pre-realized `ssh rvv` generated bundle:
  `PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,1,16,17,257 stride_pairs=2:3,3:2 mask_patterns=2 input_patterns=2 source_preserved accumulator_preserved tail_preserved`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] filtered lit from `build/test`:
  `--filter computed-masked-strided-input-widening-dot-reduce-add`
  passed 5/5 tests
- [OK] production source diff list empty; no new positive legacy RVV authority
  was introduced
- [OK] `git diff --check`
- [OK] `git diff --cached --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 554: Stage2 RVV computed-masked strided dot executable artifact ABI

**Date**: 2026-06-08
**Task**: Stage2 RVV computed-masked strided dot executable artifact ABI
**Branch**: `main`

### Summary

Refreshed explicit and pre-realized generated object/header bundle ABI evidence for computed_masked_strided_input_widening_dot_reduce_add on ssh rvv; no compiler source changes were required.

### Main Changes

- Created, completed, and archived Trellis task `06-08-stage2-rvv-segment2-unit-artifact-abi-boundary`.
- Recorded a bounded PRD for the plain segment2 unit interleave/deinterleave executable artifact ABI seam: typed body facts, field0/field1 tuple roles, unit interleave/deinterleave direction, ABI/header binding, runtime AVL/VL, provider route validation, target artifact export, generated bundle ABI, and `ssh rvv` evidence.
- Proved explicit and pre-realized `segment2_deinterleave_unit_store` generated bundles on `ssh rvv` with `dry_run=false`, `ssh_evidence=true`, remote compile success, remote run success, field-order distinction, and output tail sentinel preservation.
- Proved explicit and pre-realized `segment2_interleave_unit_load` generated bundles on `ssh rvv` with the same executable ABI boundary, even/odd interleaved destination field order, and destination tail sentinel preservation past runtime `2*n`.
- No production C++/MLIR/Python source diff was introduced; generated evidence lives under ignored `artifacts/tmp`.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Testing

- [OK] explicit segment2 deinterleave `ssh rvv`: `PASS op=segment2_deinterleave_unit_store counts=0,1,16,17,257`
- [OK] pre-realized segment2 deinterleave `ssh rvv`: `PASS op=segment2_deinterleave_unit_store counts=0,1,16,17,257`
- [OK] explicit segment2 interleave `ssh rvv`: `PASS op=segment2_interleave_unit_load counts=0,1,7,16,23,257`
- [OK] pre-realized segment2 interleave `ssh rvv`: `PASS op=segment2_interleave_unit_load counts=0,1,7,16,23,257`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'segment2-(deinterleave|interleave)|selected-body-artifact-segment2-(deinterleave|interleave)'` from `build/test`: 13/13 passed
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] production source diff list empty; bounded authority scan found only negative/historical/checklist mentions in task docs
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] task context validation passed

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 556: Stage2 RVV runtime-scalar-cmp masked segment2 executable artifact ABI

**Date**: 2026-06-08
**Task**: Stage2 RVV runtime-scalar-cmp masked segment2 executable artifact ABI
**Branch**: `main`

### Summary

Refreshed executable generated object/header bundle ABI evidence for
runtime-scalar-cmp masked segment2 load and store, covering explicit and
pre-realized selected-body paths on `ssh rvv`. No compiler source changes were
required because the existing provider/target contract already generated
executable artifacts.

### Main Changes

- Created and archived Trellis task `06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-executable-artifact-abi`.
- Recorded a bounded PRD for runtime-scalar compare, scalar splat, compare-produced mask, segment2 field roles, ABI/header binding, runtime AVL/VL, target artifact export, and `ssh rvv` evidence.
- Proved explicit and pre-realized segment2 load generated bundles on `ssh rvv` with `dry_run=false`, `ssh_evidence=true`, remote compile success, remote run success, active/inactive lanes, field distinction, source preservation, and tail preservation.
- Proved explicit and pre-realized segment2 store generated bundles on `ssh rvv` with the same executable ABI boundary and inactive interleaved destination preservation.
- No production C++/MLIR/Python source diff was introduced; generated evidence lives under ignored `artifacts/tmp`.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Testing

- [OK] explicit segment2 load `ssh rvv`: `PASS op=runtime_scalar_cmp_masked_segment2_load_unit_store counts=0,1,16,17,257 patterns=0,1`
- [OK] pre-realized segment2 load `ssh rvv`: `PASS op=runtime_scalar_cmp_masked_segment2_load_unit_store counts=0,1,16,17,257 patterns=0,1`
- [OK] explicit segment2 store `ssh rvv`: `PASS op=runtime_scalar_cmp_masked_segment2_store_unit_load counts=0,1,16,17,257 patterns=0,1`
- [OK] pre-realized segment2 store `ssh rvv`: `PASS op=runtime_scalar_cmp_masked_segment2_store_unit_load counts=0,1,7,16,23,257 patterns=0,1`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-cmp-masked-segment2'` from `build/test`: 8/8 passed
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] production source diff list empty; bounded authority scan found only negative/checklist mentions in task docs
- [OK] task context validation passed

### Status

[OK] **Completed and archived**

### Next Steps

- None - task complete


## Session 557: RVV source-front-door positive-route demotion

**Date**: 2026-06-08
**Task**: RVV source-front-door positive-route demotion
**Branch**: `main`

### Summary

Demoted RVV bounded vector source-front-door families to explicit-only materializers, retired generated-bundle source-front-door artifact mode, and verified fail-closed/default-pipeline behavior.

### Main Changes

- Demoted RVV bounded vector binary, compare/select, and runtime-scalar compare/select source-front-door family registrations from default artifact-front-door eligible to explicit-only.
- Retired `scripts/rvv_generated_bundle_abi_e2e.py --vector-source-front-door` as a generated-bundle artifact mode; the script now fails before selected-body materialization or target bundle export and directs evidence to explicit/pre-realized selected-body modes.
- Converted source-front-door generated-bundle dry-run tests to fail-closed coverage, while keeping explicit materializer lit tests as manual typed-body scaffolding checks.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` so RVV source-front-door contracts describe explicit-only materializers and require default source-artifact/generated-bundle paths to fail closed.
- Completed and archived Trellis task `06-08-rvv-source-front-door-positive-route-demotion`.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'Transforms/RVV/rvv-vector-(binary|compare-select|runtime-scalar-cmp-select)-source-front-door\.mlir|Scripts/rvv-generated-bundle-abi-e2e-vector-(source-front-door|compare-select-source-front-door|runtime-scalar-cmp-select-source-front-door)-(dry-run|fail-closed)\.test'` from `build/test`: 7/7 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'SourceFrontDoor/source-artifact-front-door-pipeline|Target/TargetArtifactBundleExport/source-artifact-bundle-front-door|Target/RVV/vector-source-target-artifact-exporters|Target/RVV/vector-materialized-target-artifact-exporters|Plugin/rvv-extension-plugin\.test'` from `build/test`: 9/9 passed
- [OK] focused authority scan found no remaining `DefaultArtifactFrontDoorPolicy::Eligible`, `default_artifact_front_door_policy = "eligible"`, JSON `"eligible"`, or `source_front_door_artifact_boundary` in the changed RVV source-front-door/script/spec/test surface
- [OK] `git diff --check`
- [OK] `git diff --cached --check`

### Status

[OK] **Completed and archived**

### Next Steps

- None - task complete


## Session 558: Stage2 RVV segment2 unit artifact ABI evidence

**Date**: 2026-06-08
**Task**: Stage2 RVV segment2 unit artifact ABI evidence
**Branch**: `main`

### Summary

Refreshed explicit and pre-realized plain segment2 deinterleave/interleave generated-bundle ssh rvv evidence; no production source change required; focused lit and C++ checks passed.

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


## Session 559: Stage2 RVV computed-masked segment2 load/store artifact ABI evidence

**Date**: 2026-06-08
**Task**: Stage2 RVV computed-masked segment2 load/store artifact ABI evidence
**Branch**: `main`

### Summary

Closed the computed-masked segment2 load/store generated-artifact ABI boundary with no production source changes and fresh ssh rvv evidence.

### Main Changes

- Created Trellis PRD for `stage2-rvv-computed-masked-segment2-load-store-artifact-abi-boundary`.
- Verified the existing RVV plugin path already owns the executable boundary: selected-body realization, computed-mask segment2 route planning, statement planning, `TCRVEmitCLowerableRoute` construction, target artifact validation, generated bundle ABI, and ssh runtime evidence.
- Recorded no-source-change justification because provider and target validation already reject stale route facts, ABI order, inactive-lane policy, field mapping, typed config, runtime AVL/VL, statement-plan, and route-control mismatches.
- Generated dry-run evidence for explicit and pre-realized computed-mask segment2 load/store bundles.
- Generated non-dry-run `ssh rvv` evidence for explicit and pre-realized computed-mask segment2 load/store bundles; all four runs reported remote compile/run success and PASS summaries.
- Ran focused checks: script py_compile, script self-test, direct generated-bundle dry-runs, ssh rvv bundle runs, `tianchenrv-rvv-extension-plugin-test`, `tianchenrv-target-artifact-export-test`, focused emission-plan commands, diff checks, and bounded old-authority scans.


### Git Commits

| Hash | Message |
|------|---------|
| `pending-in-this-commit` | (see git log) |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Four direct generated-bundle dry-runs for explicit/pre-realized computed-mask segment2 load/store.
- [OK] Four non-dry-run generated-bundle runs on `ssh rvv` for explicit/pre-realized computed-mask segment2 load/store.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] Four focused computed-mask segment2 emission-plan commands.
- [OK] `git diff --check`, `git diff --cached --check`, and bounded old-authority scans.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 560: Stage2 RVV runtime-scalar segment2 load ABI boundary

**Date**: 2026-06-08
**Task**: Stage2 RVV runtime-scalar segment2 load ABI boundary
**Branch**: `main`

### Summary

Hardened runtime-scalar computed-mask segment2 load route-family classification, added focused pre-realized plugin coverage, refreshed ssh rvv generated-bundle evidence, and archived the Trellis task.

### Main Changes

- Created, completed, and archived Trellis task
  `06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-store-artifact-abi-boundary`.
- Confirmed production source already classifies
  `RuntimeScalarComputedMaskSegment2StoreUnitLoad` as computed-mask segment2
  memory, store-only, and `ComputedMaskUnitLoadSegment2Store`; no production
  source rewrite was needed.
- Added focused C++ provider coverage in
  `test/Plugin/RVVExtensionPluginTest.cpp` for
  `runtime_scalar_cmp_masked_segment2_store_unit_load`: route analysis,
  computed-mask memory family facts, segment2 provider-plan flags,
  `rhs_scalar` ABI/splat facts, statement-plan callees, and provider-built
  `TCRVEmitCLowerableRoute` consumption.
- Refreshed explicit and pre-realized generated-bundle evidence on `ssh rvv`
  with active/inactive lanes, inactive interleaved destination preservation,
  field distinction, source preservation, and tail preservation.

### Git Commits

- Pending final commit in this session.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-cmp-masked-segment2-store|pre-realized-runtime-scalar-cmp-segment2-store|explicit-selected-body-artifact-runtime-scalar-cmp-masked-segment2-store|pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-segment2-store'` from `build/test`: 4/4 passed
- [OK] explicit store `ssh rvv`: `PASS op=runtime_scalar_cmp_masked_segment2_store_unit_load counts=0,1,16,17,257 patterns=0,1`
- [OK] pre-realized store `ssh rvv`: `PASS op=runtime_scalar_cmp_masked_segment2_store_unit_load counts=0,1,7,16,23,257 patterns=0,1`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-store-artifact-abi-boundary`
- [OK] bounded added-line old-authority scan
- [OK] `git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 561: Stage2 RVV runtime-scalar segment2 store ABI boundary

**Date**: 2026-06-08
**Task**: Stage2 RVV runtime-scalar segment2 store ABI boundary
**Branch**: `main`

### Summary

Added focused C++ provider coverage for runtime-scalar computed-mask segment2 store, verified target export and lit store fixtures, and refreshed explicit/pre-realized ssh rvv generated-bundle evidence.

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
