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


## Session 573: RVV Gate 5 measurement-scaffolding cleanup

**Date**: 2026-06-08
**Task**: RVV low-precision production-kernel performance optimization campaign
**Branch**: `main`

### Summary

Repaired the active macro PRD to make Gate 5 the current cleanup milestone and
completed one Gate 5 source slice. This supersedes the earlier Gate 4
ready-to-archive note: the macro task remains active because the repaired Gate 3
and Gate 4 production-loop gates are still partial.

### Main Changes

- Removed product-reduction dequant/dequant-clamp dependence on standalone
  vector `dequantize_convert_intrinsic` / `dequantize_scale_intrinsic` metadata.
- Kept scalar dequantization owned by the RVV provider as
  `dot_acc_scalar * scale`, then mirrored the RVV f32 scalar splat intrinsic via
  `tcrv_rvv.rhs_broadcast_intrinsic` at the emission-plan/validation boundary.
- Updated target artifact validation to reject stale standalone vector dequant
  keys on product-reduction dequant candidates while preserving standalone
  dequant route contracts.
- Updated generated-bundle ABI dry-runs, focused Target/RVV FileCheck tests,
  plugin smoke expectations, and the RVV plugin spec contract.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tcrv-opt --version`
- [OK] `./build/bin/tcrv-translate --version`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py scripts/rvv_generated_bundle_same_target_measure.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32|explicit-selected-body-artifact-widening-product-reduce-dequantize-f32|explicit-selected-body-realization-widening-product-reduce-dequant-clamp-f32'` from `build/test`: 4/4 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequantize-f32-dry-run|rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequant-clamp-f32-dry-run|rvv-generated-bundle-abi-e2e-explicit-widening-product-reduce-dequant-clamp-f32-dry-run'` from `build/test`: 3/3 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'widening-product|product-reduction|generic-widening-product'` from `build/test`: 14/14 passed
- [OK] Added-line old-authority scan found no positive legacy i32m1,
  descriptor-driven, q8, q4, or llama route authority.
- [OK] Old dequant convert/scale scan found remaining uses only in standalone
  dequant contracts or negative stale-metadata checks.

### Runtime Evidence

No new `ssh rvv` measurement was run in this Gate 5 cleanup slice. The executable
generated product-reduction dequant/dequant-clamp code was not changed from the
previous scalar-dequant-splat measured path; this slice changed route metadata,
target validation, script consumers, tests, and spec text.

### Status

[OPEN] Gate 5 cleanup is complete. The macro task remains active and is not
archived.

### Next Steps

Continue with main low-precision product/reduction loop resource-aware
selected-body realization and statement planning for repeated VL/control,
accumulator/reduction structure, and live vector group tradeoffs.


## Session 564: Stage2 RVV runtime-scalar-cmp masked standalone minmax executable artifact ABI boundary

**Date**: 2026-06-08
**Task**: Stage2 RVV runtime-scalar-cmp masked standalone minmax executable artifact ABI boundary
**Branch**: `main`

### Summary

Closed the runtime-scalar-cmp masked standalone min/max generated-artifact ABI
boundary with no production source changes. Current HEAD already derives the
pre-realized min/max route from typed `tcrv_rvv` runtime-scalar standalone
reduction body facts through RVV-owned selected-body realization, route
planning, target validation, generated bundle ABI, and `ssh rvv` runtime
evidence.

### Main Changes

- Created Trellis task
  `06-08-stage2-rvv-runtime-scalar-cmp-masked-standalone-minmax-executable-artifact-abi-boundary`.
- Recorded a bounded PRD for the runtime scalar comparison, compare-produced
  mask, standalone min/max reduction kind, inactive neutral policy, scalar
  seed/result ABI, runtime AVL/VL, header/prototype binding, target artifact
  validation, generated bundle ABI, and `ssh rvv` evidence boundary.
- Verified production code already validates runtime-scalar min/max op kinds,
  `rhs_scalar` ABI role/type, SEW/LMUL config, compare-produced mask facts,
  accumulator/result layout, runtime AVL/VL, and pre-realized/realized body
  exclusivity in the RVV standalone reduction realization owner.
- Verified target validation already requires provider-built runtime-scalar
  standalone reduction loop statements: loop setvl, compare lhs load,
  RHS scalar splat, source load, compare, inactive neutral splat, merge,
  scalar seed, min/max reduction, and scalar result store.
- Proved pre-realized runtime-scalar-cmp masked standalone min/max generated
  bundles on `ssh rvv` for LMUL m1 and LMUL m2, counts `0,1,16,23,257`,
  runtime scalar thresholds `-37,91`, seeds `-11,17`, and patterns `0,1`.

### Git Commits

- Recorded in the final commit for this session.

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-08-stage2-rvv-runtime-scalar-cmp-masked-standalone-minmax-executable-artifact-abi-boundary`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-cmp-masked-standalone-minmax'` from `build/test`: 2/2 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-(min|max)'` from `build/test`: 4/4 passed
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Direct generated-bundle dry-run for pre-realized runtime-scalar-cmp masked standalone min/max m1 and m2
- [OK] Non-dry-run `ssh rvv` generated bundles for
  `runtime_scalar_cmp_masked_standalone_reduce_min`,
  `runtime_scalar_cmp_masked_standalone_reduce_min_lmul_m2`,
  `runtime_scalar_cmp_masked_standalone_reduce_max`, and
  `runtime_scalar_cmp_masked_standalone_reduce_max_lmul_m2`: all reported
  `PASS`, `remote_compile_succeeded: true`, `remote_run_succeeded: true`,
  `dry_run: false`, and `ssh_evidence: true`
- [OK] Evidence JSON spot checks for `runtime-scalar-splat-compare-rhs`,
  `cmp_lhs,rhs_scalar,src,acc,out,n`, reduction kind min/max, inactive neutral
  contract, all-inactive seed preservation, source preservation, and tail
  preservation
- [OK] `git diff --check`
- [OK] bounded old-authority scan over the task directory; only negative PRD
  language matched and there was no production source diff

### Status

[OK] **Completed, archived, and committed**

### Next Steps

- None - task complete


## Session 563: Stage2 RVV runtime-scalar-cmp masked segment2 memory executable artifact ABI boundary

**Date**: 2026-06-08
**Task**: Stage2 RVV runtime-scalar-cmp masked segment2 memory executable artifact ABI boundary
**Branch**: `main`

### Summary

Closed the current-HEAD executable artifact ABI evidence boundary for
runtime-scalar-cmp masked segment2 load/store after the shared provider splat
contract landed. No production source changes were required; the current
provider, target validator, generated bundle ABI, and `ssh rvv` runtime
evidence already line up.

### Main Changes

- Created Trellis task
  `06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-memory-executable-artifact-abi-boundary`.
- Recorded a bounded PRD for selected explicit/pre-realized
  runtime-scalar-cmp masked segment2 load/store bodies, runtime scalar
  `rhs_scalar` binding and splat, computed-mask facts, active/inactive lane
  policy, segment2 field layout, ABI/header binding, runtime AVL/VL, target
  artifact export, generated bundle ABI, and `ssh rvv` evidence.
- Confirmed the current production path already validates the seam through
  RVV-owned provider plans, the shared runtime-scalar splat provider contract,
  migrated segment2 statement-plan owner selection, target rebuilt-route
  validation, and generated bundle ABI checks.
- Refreshed explicit and pre-realized segment2 load generated-bundle evidence
  on `ssh rvv` with active/inactive lanes, runtime scalar compare, field
  distinction, source preservation, old-field passthrough preservation, and
  tail preservation.
- Refreshed explicit and pre-realized segment2 store generated-bundle evidence
  on `ssh rvv` with active/inactive lanes, inactive interleaved destination
  preservation, field distinction, source preservation, and tail preservation.

### Git Commits

- This commit: `rvv: add low precision primitive route facts`

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-memory-executable-artifact-abi-boundary`
- [OK] Four generated-bundle dry-runs for explicit/pre-realized runtime-scalar-cmp masked segment2 load/store, rerun with explicit `build/bin/tcrv-opt`, `build/bin/tcrv-translate`, and `/usr/bin/llvm-readobj-20` after PATH-only tool lookup failed.
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-cmp-masked-segment2'` from `build/test`: 8/8 passed.
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] explicit load `ssh rvv`: `PASS op=runtime_scalar_cmp_masked_segment2_load_unit_store counts=0,1,16,17,257 patterns=0,1`
- [OK] pre-realized load `ssh rvv`: `PASS op=runtime_scalar_cmp_masked_segment2_load_unit_store counts=0,1,16,17,257 patterns=0,1`
- [OK] explicit store `ssh rvv`: `PASS op=runtime_scalar_cmp_masked_segment2_store_unit_load counts=0,1,16,17,257 patterns=0,1`
- [OK] pre-realized store `ssh rvv`: `PASS op=runtime_scalar_cmp_masked_segment2_store_unit_load counts=0,1,7,16,23,257 patterns=0,1`
- [OK] Evidence JSON spot checks found `dry_run=false`, `ssh_evidence=true`, `remote_compile_succeeded=true`, `remote_run_succeeded=true`, provider-supported mirrors, route operand binding summaries, and expected runtime ABI orders.

### Status

[OK] **Completed and archived; pending commit**

### Next Steps

- Create the final evidence commit.


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

- Final commit created in this session after focused checks.

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


## Session 562: Stage2 RVV runtime-scalar computed-mask memory provider contract owner

**Date**: 2026-06-08
**Task**: Stage2 RVV runtime-scalar computed-mask memory provider contract owner
**Branch**: `main`

### Summary

Added a shared RVV provider-owned runtime-scalar computed-mask splat contract
and used it to fail-close stale segment2 load/store provider preflight facts
before `TCRVEmitCLowerableRoute` construction.

### Main Changes

- Created Trellis task
  `06-08-stage2-rvv-runtime-scalar-computed-mask-memory-provider-contract-owner`.
- Added `RVVRuntimeScalarComputedMaskMemorySplatProviderContract` and
  `verifyRVVRuntimeScalarComputedMaskMemorySplatProviderContract(...)`.
- Rewired non-segment runtime-scalar computed-mask memory provider preflight to
  use the shared splat contract for `rhs_scalar` ABI, splat materialization,
  provider mirror, runtime ABI order, and statement-plan leaf checks.
- Rewired segment2 runtime-scalar computed-mask load/store provider preflight
  to use the same contract. This repairs the stale load-side consumer that did
  not previously require `rhsScalarBroadcastLeaf` and the splat statement leaf
  before route construction.
- Added focused C++ regression coverage that producer-realizes
  runtime-scalar segment2 load, runs positive provider preflight, mutates the
  splat materialization leaf, and checks fail-closed provider diagnostics.
- Confirmed target validation already checks runtime-scalar segment2
  statement shape from rebuilt provider contracts, so no target source rewrite
  was required.
- Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the executable
  `RVVRuntimeScalarComputedMaskMemorySplatProviderContract` signature,
  contract, error matrix, and tests required.

### Git Commits

- One coherent Gate 2 second-slice commit created at session close; see final
  report for the hash.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-cmp-masked-segment2'` from `build/test`: 8/8 passed
- [OK] `git diff --check`
- [OK] bounded added-line old-authority scan

### Status

[OK] **Completed, archived, and committed**

### Next Steps

- None - task complete


## Session 562: Stage2 RVV runtime-scalar reduce-add executable ABI evidence

**Date**: 2026-06-08
**Task**: Stage2 RVV runtime-scalar reduce-add executable ABI evidence
**Branch**: `main`

### Summary

Archived the runtime-scalar-cmp masked standalone reduce-add executable artifact/ABI boundary task. No production source change was required after focused provider/target/script inspection. Verified dry-run and ssh rvv generated-bundle evidence for add m1, add LMUL m2, and add i64; ran RVV plugin, target artifact export, lit reduce-add, script self-test, task validation, diff checks, and old-authority scan.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 565: RVV production-kernel low-precision primitive surface

**Date**: 2026-06-08
**Task**: RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Started the macro production-kernel campaign Gate 2 by adding provider-owned
low-precision primitive facts for signed i8 widening product and
product-reduction routes, with target mirror validation and a fail-closed u8
typed-surface diagnostic.

### Main Changes

- Created and kept active the macro Trellis task
  `06-08-rvv-production-kernel-capability-campaign`.
- Added `lowPrecisionPrimitive*` fields to RVV contraction route-family plans,
  selected-body route descriptions, product route facts, and product route
  validation contracts.
- Derived signed i8 widening product facts from typed body/config facts:
  source `i8`, product/result `i16`, no accumulator.
- Derived product-reduction primitive facts for the existing i8/i16/i32/f32
  contraction routes, including accumulator and result dtype mirrors.
- Emitted `tcrv_rvv.low_precision_primitive.*` metadata and registered the
  corresponding target support bundle keys for header artifact export.
- Extended product-only target artifact validation to reject stale primitive
  contract/kind/source/product/accumulator/result mirrors.
- Added a dialect fail-closed diagnostic for unsigned u8 widening-product typed
  vectors until provider/target unsigned intrinsic facts exist.
- Added product-only target lit coverage and widened product-reduction lit
  checks; repaired two dequant-clamp negative RUN commands to target the actual
  `tcrv_rvv.runtime_abi_order` candidate mirror.

### Git Commits

- `rvv: consume low precision facts in realization` (this session commit).

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'generic-widening-product-dataflow|explicit-selected-body-artifact-widening-product(\.mlir|$)|explicit-selected-body-artifact-widening-product-reduce-add'` from `build/test`: 3/3 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'widening-product|product-reduction|generic-widening-product'` from `build/test`: 11/11 passed
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] bounded production added-line old-authority scan
- [OK] `git diff --check`
- [OK] `git diff --cached --check`

### Status

[OPEN] Macro task remains active. Gate 2 first slice is complete; Gates 1, 3,
and 4 remain open.

### Next Steps

Continue Gate 2 by adding accepted u8 widening-product provider/target facts or
extending the unsigned fail-closed matrix, then deepen `vwredsum`/accumulator
route-family validation for Gearbox consumption.


## Session 566: RVV production-kernel u8 primitive boundary and vwredsum validation

**Date**: 2026-06-08
**Task**: RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Continued the active macro campaign Gate 2 second slice. Promoted unsigned u8
widening-product from dialect-only rejection to verifier-legal typed RVV
surface, kept it fail-closed at the RVV provider boundary with a diagnostic that
names the missing unsigned intrinsic/target mirror facts, and added
product-reduction stale-mirror coverage for `vwredsum` and primitive
accumulator dtype facts.

### Main Changes

- Repaired the active macro PRD for the Gate 2 second slice rather than creating
  a neighboring task.
- Extended `tcrv_rvv.widening_product` verifier support for
  `unsigned_widening_product` with
  `!tcrv_rvv.vector<ui8, "mf4"> -> !tcrv_rvv.vector<ui16, "mf2">` typed facts.
- Allowed the minimal u8 ABI/load surface needed for typed u8 bodies to reach
  provider route analysis: `const uint8_t *` lhs/rhs, `uint16_t *` output, and
  standalone u8 widening-product source-load bypass under SEW16/LMUL mf2.
- Added provider fail-closed rejection before `TCRVEmitCLowerableRoute`
  construction for unsigned u8, naming missing
  `__riscv_vwmulu_vv_u16mf2` and matching target type/header mirror validation.
- Added conversion lit coverage for the provider-owned u8 fail-closed boundary.
- Added product-reduction target artifact stale-mirror checks for
  `tcrv_rvv.widening_reduction_intrinsic` and
  `tcrv_rvv.low_precision_primitive.accumulator_dtype`.

### Git Commits

- Pending final commit in this session.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'generic-widening-product-dataflow|rvv-generic-stage2-widening-product-unsigned-u8-negative|explicit-selected-body-artifact-widening-product(\.mlir|$)|explicit-selected-body-artifact-widening-product-reduce-add'` from `build/test`: 4/4 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'widening-product|product-reduction|generic-widening-product'` from `build/test`: 12/12 passed
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] bounded added-line old-authority scan

### Status

[OPEN] Macro task remains active. Gate 2 second slice is complete as a
fail-closed unsigned u8 provider-boundary slice plus `vwredsum` validation
coverage; Gates 1, 3, and 4 remain open.

### Next Steps

Continue Gate 2 by implementing accepted unsigned u8 provider/target facts if
the materializer is ready: unsigned vector C type mapping, u8/u16 load-store
leaves, `vwmulu.vv` intrinsic mapping, and target mirror support. Then continue
Gearbox-consumable primitive/resource facts for product, accumulator, and
result validation.


## Session 567: RVV accepted unsigned u8 widening-product provider/target facts

**Date**: 2026-06-08
**Task**: RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Continued the active macro campaign Gate 2 accepted unsigned u8 slice. Moved
the verifier-legal unsigned u8 widening-product body from provider
fail-closed to route-supported through typed RVV body/config facts,
provider-owned unsigned intrinsic/type facts, common EmitC materialization,
and target artifact mirror validation.

### Main Changes

- Removed the provider fail-closed boundary for bounded
  `unsigned_widening_product` on
  `!tcrv_rvv.vector<ui8, "mf4"> -> !tcrv_rvv.vector<ui16, "mf2">`.
- Added relation-aware unsigned route facts for `u8/u16`, `vuint8mf4_t`,
  `vuint16mf2_t`, `__riscv_vle8_v_u8mf4`,
  `__riscv_vwmulu_vv_u16mf2`, `__riscv_vse16_v_u16mf2`, unsigned C type
  mapping, unsigned target leaf profile, and
  `rvv-route-operand-binding:widening_product_u8_u16.v1`.
- Threaded unsigned facts through route planning, contraction route-family
  validation, runtime ABI/config construction contracts,
  `TCRVEmitCLowerableRoute`, and target artifact mirror validation.
- Replaced the previous unsigned u8 fail-closed conversion fixture with an
  accepted conversion fixture that checks provider facts and emitted unsigned
  RVV intrinsics.
- Added target artifact coverage for accepted unsigned header export and stale
  unsigned intrinsic/source-dtype/C-type mirror rejection.

### Git Commits

- Pending final commit in this session.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
  - Existing `TargetArtifactExportTest.cpp` switch-coverage warnings were
    observed during compilation; the build linked successfully.
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'generic-widening-product-dataflow|rvv-generic-stage2-widening-product-unsigned-u8|explicit-selected-body-artifact-widening-product(\.mlir|$)|explicit-selected-body-artifact-widening-product-unsigned-u8|explicit-selected-body-artifact-widening-product-reduce-add'` from `build/test`: 5/5 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'widening-product|product-reduction|generic-widening-product'` from `build/test`: 13/13 passed
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] bounded added-line old-authority scan

### Status

[OPEN] Macro task remains active. Gate 2 accepted unsigned u8
widening-product provider/target facts slice is complete; Gates 1, 3, and 4
remain open.

### Next Steps

Continue the production-kernel capability campaign with Gate 1
Gearbox/resource-aware selected-body realization and the remaining
Gearbox-consumable product/accumulator/result primitive-resource facts. Gate 3
runtime evidence should only be collected when a newly executable unsigned
path is explicitly claimed.


## Session 568: RVV low-precision widening-reduction primitive facts

**Date**: 2026-06-08
**Task**: RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Continued the active macro campaign Gate 2 widening-reduction slice. Added
provider-owned low-precision product-reduction primitive facts for the bounded
signed i8 -> i16 -> i32 `vwredsum` chain, threaded them through provider route
validation and target mirror validation, and kept the macro campaign open.

### Main Changes

- Added `RVVLowPrecisionWideningReductionPrimitiveFacts` to make the
  product-reduction primitive contract explicit: source/product/accumulator/
  result dtype, SEW/LMUL, product and chain relations, `vwmul`, `vwredsum`,
  scalar seed splat, layouts, and reduction store VL.
- Threaded the primitive facts into widening dot-reduce route facts and
  validation contracts so stale provider facts fail before
  `TCRVEmitCLowerableRoute` construction.
- Hardened target artifact validation so `low_precision_primitive.*`,
  product/reduction relation, intrinsic, layout, seed, and store-VL metadata are
  mirrors of provider facts only.
- Added focused C++ negative coverage for stale accumulator dtype and kept the
  lit stale `vwredsum` / primitive accumulator mirror diagnostics aligned.
- Updated the RVV plugin and EmitC route specs with the executable
  product-reduction primitive-fact contract.

### Git Commits

- Pending final commit in this session.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
  - Existing `TargetArtifactExportTest.cpp` switch-coverage warnings and target
    validation helper warnings were observed during compilation; the build
    linked successfully.
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'generic-widening-product-dataflow|generic-widening-product-reduction-chain-dataflow|rvv-generic-stage2-widening-product-unsigned-u8|explicit-selected-body-artifact-widening-product(\.mlir|$)|explicit-selected-body-artifact-widening-product-unsigned-u8|explicit-selected-body-artifact-widening-product-reduce-add|explicit-selected-body-artifact-widening-product-reduce-dequantize-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|explicit-selected-body-realization-widening-product-reduce-dequant-clamp-f32'` from `build/test`: 9/9 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'widening-product|product-reduction|generic-widening-product'` from `build/test`: 13/13 passed
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] bounded production/test added-line old-authority scan

### Status

[OPEN] Macro task remains active. Gate 2 widening-reduction primitive facts
slice is complete as provider/target compiler-surface work; Gates 1, 3, and 4
remain open.

### Next Steps

Continue the macro campaign by making low-precision product/reduction primitive
and resource facts consumable by Gearbox/resource-aware selected-body
realization. Collect Gate 3 runtime evidence only when a newly executable path
is claimed, then continue to Gate 4 same-target comparison.


## Session 569: RVV low-precision realization fact integration

**Date**: 2026-06-08
**Task**: RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Continued the active macro campaign with a bounded Gate 1/Gate 2 integration
slice. The RVV contraction selected-body realization owner now consumes
provider-owned low-precision widening-reduction primitive facts and
pass-produced Gearbox/resource facts before it materializes realized
product-reduction dequantize/dequant-clamp selected-body structure.

### Main Changes

- Added selected-body realization validation for provider-owned primitive facts:
  source/product/result SEW/LMUL, signed product kind, product relation,
  product-reduction chain relation, accumulator/result layout,
  reduction-store VL, and required `vwmul` / `vwredsum` / scalar seed facts.
- Hardened low-precision resource-fact consumption before route construction:
  candidate set/selection, legality scope, dtype and SEW/LMUL facts, memory
  form, policy, unroll, accumulator count, reduction layout, `vsetvl` region
  count, runtime AVL source, runtime ABI order, live-vector pressure, target
  profile, and legal-result mirrors.
- Added focused lit coverage for accepted realized fact consumption and
  fail-closed missing/stale resource facts on the product-reduction
  dequant-clamp path.
- Updated the RVV plugin resource-aware closure spec to make primitive/resource
  fact consumption a selected-body realization contract, with Common EmitC and
  artifact metadata remaining mirror consumers only.

### Git Commits

- Pending final commit in this session.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32'` from `build/test`: 1/1 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'widening-product|product-reduction|generic-widening-product'` from `build/test`: 13/13 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'generic-widening-product-dataflow|generic-widening-product-reduction-chain-dataflow|rvv-generic-stage2-widening-product-unsigned-u8|explicit-selected-body-artifact-widening-product(\\.mlir|$)|explicit-selected-body-artifact-widening-product-unsigned-u8|explicit-selected-body-artifact-widening-product-reduce-add|explicit-selected-body-artifact-widening-product-reduce-dequantize-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|explicit-selected-body-realization-widening-product-reduce-dequant-clamp-f32'` from `build/test`: 9/9 passed

### Status

[OPEN] Macro task remains active. This slice completes the current
fact-to-selected-body-realization integration milestone, but the broader Gate 1
resource-aware build/prune/select closure, Gate 3 runtime evidence for any newly
claimed executable behavior, and Gate 4 same-target measurement remain open.

### Next Steps

Continue the macro campaign by replacing more of the fixed Gearbox MVP with
real resource-aware candidate build/prune/select behavior, then validate the
selected candidate against realized `tcrv_rvv` marker count/order/resource
decision before any Gate 3 runtime or Gate 4 measurement claim.


## Session 570: RVV Gearbox candidate build/prune/select closure

**Date**: 2026-06-08
**Task**: RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Continued the active macro campaign with the Gate 1 resource-aware Gearbox
candidate build/prune/select closure slice. The bounded low-precision
product-reduction dequantize/dequant-clamp path now shares one compiler-owned
candidate decision chain across the Gearbox pass, selected-body realization, and
provider pre-route validation.

### Main Changes

- Added canonical low-precision product-reduction Gearbox resource candidate
  helpers carrying memory form, typed source/product/accumulator/result facts,
  policy, unroll, accumulator count, reduction layout, `vsetvl` region count,
  live-vector pressure, runtime AVL source, producer/consumer scopes, ABI order,
  legality, and rejection reason.
- Reworked the Gearbox schedule pass to build, prune, and select the bounded
  product-reduction-dequantize/dequant-clamp resource candidate before writing
  `tcrv_rvv.low_precision_resource.*` facts.
- Reworked selected-body realization so the selected candidate drives realized
  attrs, producer/consumer `tcrv_rvv.vsetvl_region_marker` count/order/resource
  decisions, and the Gearbox cross-region handoff.
- Reworked provider pre-route validation so stale selected candidates, realized
  attrs, marker count/order/resource decisions, or handoff scope/resource facts
  fail before `TCRVEmitCLowerableRoute` construction.
- Added focused budget-pruning coverage for the low-precision product-reduction
  fixture and updated the RVV plugin spec with the shared candidate
  build/prune/select contract.

### Self-Repair

- The first RVV plugin C++ test run exposed a semantic check bug: the selected
  resource candidate's final result dtype is `f32`, while the intermediate
  reduction result dtype is `i32`. The realization consistency check now compares
  the candidate result dtype against provider primitive `finalResultElementTypeName`.

### Git Commits

- Pending final commit in this session.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test` after the dtype self-repair
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|explicit-selected-body-realization-widening-product-reduce-dequant-clamp-f32'` from `build/test`: 3/3 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'widening-product|product-reduction|generic-widening-product'` from `build/test`: 13/13 passed
- [OK] `git diff --check`
- [OK] Bounded added-line old-authority scan over touched production/test files: no hits

### Status

[OPEN] Macro task remains active. This slice completes the Gate 1 bounded
resource-aware candidate build/prune/select closure milestone, but the macro
campaign is not complete.

### Next Steps

Continue from the remaining macro gates: finish any explicitly named Gate 2
low-precision primitive integration gap, collect Gate 3 runtime correctness
evidence only when a newly executable production behavior is claimed, and defer
Gate 4 same-target measurement until correctness evidence exists.


## Session 571: RVV Gate 3 resource-aware runtime correctness

**Date**: 2026-06-08
**Task**: RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Continued the active macro campaign with the Gate 3 runtime correctness slice
for the resource-aware Gearbox selected-body path changed in `ad665e20`.
The production C++ path already executed with the required pass order, so this
slice did not change compiler source. It added one narrow script dry-run hook
for the pre-realized product-reduction dequantize path and collected real
`ssh rvv` correctness evidence for both dequantize and dequant-clamp.

### Main Changes

- Repaired `.trellis/tasks/06-08-rvv-production-kernel-capability-campaign/prd.md`
  so the current round is explicitly Gate 3 runtime correctness for the
  resource-aware Gearbox path, while Gate 4 same-target measurement remains open.
- Added `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequantize-f32-dry-run.test`.
  The new script lit fixes coverage symmetry with the existing dequant-clamp
  hook and checks pre-realized input mode, selected-body realization, Gearbox
  selected candidate/resource facts, cross-region handoff, provider route facts,
  target artifact validation, generated ABI agreement, and harness assertions.
- Kept C++ production source unchanged because the live production path already
  ran through `--tcrv-rvv-materialize-gearbox-schedules` before
  `--tcrv-materialize-selected-lowering-boundaries`, then emission-plan
  materialization, RVV EmitC C++ export, target artifact bundle export, and
  external harness execution.

### Runtime Evidence

- [OK] Non-dry-run `ssh rvv` run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/gate3-resource-aware-pre-realized-runtime --run-id gate3-resource-aware-wpr-dequant-and-clamp-rvv --overwrite --op-kind widening_product_reduce_dequantize_f32 --op-kind widening_product_reduce_dequant_clamp_f32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180`
- Evidence root:
  `artifacts/tmp/gate3-resource-aware-pre-realized-runtime/gate3-resource-aware-wpr-dequant-and-clamp-rvv`
- `widening_product_reduce_dequantize_f32/evidence.json`: `status=success`,
  `ssh_evidence=true`, `remote_compile_succeeded=true`,
  `remote_run_succeeded=true`.
- `widening_product_reduce_dequant_clamp_f32/evidence.json`: `status=success`,
  `ssh_evidence=true`, `remote_compile_succeeded=true`,
  `remote_run_succeeded=true`.
- Harness coverage: `n = 0, 1, 16, 17, 257`, two signed i8 source patterns,
  runtime scales `-0.125` and `0.375`; dequant-clamp additionally checks ordered
  bound pairs `[-1.5, 2.25]` and `[-8, -0.75]`. Both harnesses check expected
  accumulation, output value, source/accumulator preservation, and tail sentinel
  preservation.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequantize-f32-dry-run'` from `build/test`: 1/1 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequant-clamp-f32-dry-run'` from `build/test`: 1/1 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|explicit-selected-body-realization-widening-product-reduce-dequant-clamp-f32'` from `build/test`: 3/3 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'widening-product|product-reduction|generic-widening-product'` from `build/test`: 14/14 passed

### Status

[OPEN] Macro task remains active. Gate 3 runtime correctness for the bounded
resource-aware product-reduction dequantize/dequant-clamp path is complete.
Gate 4 same-target measurement remains open.

### Next Steps

Continue with Gate 4: same-target measurement for the bounded low-precision
product-reduction dequantize/dequant-clamp generated TianChen-RV RVV artifacts
against an explicitly named baseline on the same `ssh rvv` environment, after
correctness passes and with timing method, input sizes, compile flags, target
profile, and raw evidence recorded.


## Session 572: RVV Gate 4 same-target measurement path

**Date**: 2026-06-08
**Task**: RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Completed Gate 4 for the active macro campaign. The resource-aware
pre-realized low-precision product-reduction dequantize/dequant-clamp generated
RVV artifacts are now measured on the same `ssh rvv` target against explicitly
named scalar C baselines after correctness guards pass.

### Main Changes

- Repaired the macro PRD from the completed Gate 3 slice to the Gate 4
  same-target measurement milestone and marked all campaign gates complete.
- Added `scripts/rvv_generated_bundle_same_target_measure.py`. The script
  reuses the existing generated-bundle ABI e2e production path to generate and
  verify pre-realized RVV object/header bundles, then builds a same-target C
  measurement harness.
- The measurement harness records baseline identity, generated artifact
  identity, target profile, compile flags, input sizes, warmup/repetition
  policy, `clock_gettime(CLOCK_MONOTONIC_RAW)` timing, correctness-before-timing
  guards, raw stdout, parsed per-repeat timing records, and parsed summary
  records.
- Added
  `test/Scripts/rvv-generated-bundle-same-target-measure-gate4-dry-run.test`
  to lock the dry-run evidence schema and generated harness structure.
- Updated `.trellis/spec/testing/mlir-testing-contract.md` with the concrete
  RVV same-target measurement evidence contract.

### Runtime Evidence

- [OK] Non-dry-run `ssh rvv` run:
  `python3 scripts/rvv_generated_bundle_same_target_measure.py --artifact-root artifacts/tmp/gate4-same-target-measurement --run-id gate4-wpr-dequant-and-clamp-rvv --overwrite --op-kind widening_product_reduce_dequantize_f32 --op-kind widening_product_reduce_dequant_clamp_f32 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 240`
- Evidence root:
  `artifacts/tmp/gate4-same-target-measurement/gate4-wpr-dequant-and-clamp-rvv`
- Target profile captured: `ssh_target=rvv`, `remote_arch=riscv64`, `CPU(s): 64`,
  Ubuntu clang `18.1.3`.
- Compile flags: `-O2 -march=rv64gcv -mabi=lp64d -I.`
- Input sizes: `257`, `4096`, `65536`; scales `-0.125`, `0.375`; clamp bound
  pairs `[-1.5, 2.25]`, `[-8, -0.75]`; warmups `2`; repeats `5`;
  measurement iterations `8`.
- `widening_product_reduce_dequantize_f32`: `12` summary records and `60`
  repeat timing records.
- `widening_product_reduce_dequant_clamp_f32`: `24` summary records and `120`
  repeat timing records.
- Largest-size summaries show the generated path is slower than the scalar C
  baseline in this first measurement harness: for `n=65536`, dequantize is about
  `34.36-34.37 us` baseline vs `152.18-152.21 us` generated; dequant-clamp is
  about `54.35-55.09 us` baseline vs `242.54-242.63 us` generated. This is raw
  same-target measurement evidence, not a parity claim.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-same-target-measure-gate4-dry-run|rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequantize-f32-dry-run|rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequant-clamp-f32-dry-run'` from `build/test`: 3/3 passed
- [OK] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'widening-product|product-reduction|generic-widening-product'` from `build/test`: 14/14 passed

### Status

[DONE] All macro campaign gates are complete. The task is ready to archive and
commit as one coherent Gate 4 closeout.


## Session 566: RVV low-precision production-kernel performance optimization slice

**Date**: 2026-06-08
**Task**: RVV low-precision production-kernel performance optimization slice
**Branch**: `main`

### Summary

Created the RVV low-precision production-kernel performance optimization macro task, completed Gate 1 bottleneck attribution, landed a Gate 2 production-source scalar dequant epilogue optimization for widening product reduction dequant/dequant-clamp, validated focused build/lit/ssh rvv correctness and same-target measurement, and left the macro task active for main-loop/reduction resource optimization.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `c3c4dc4e` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
