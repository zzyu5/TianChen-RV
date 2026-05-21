# Journal - codex (Part 12)

> Continuation from `journal-11.md` (archived at ~2000 lines)
> Started: 2026-05-20

---


## Session 146: Stage2 RVV masked strided-input widening dot-reduction executable slice

**Date**: 2026-05-20
**Task**: `05-20-stage2-rvv-masked-strided-widening-dot-reduction`
**Branch**: `main`

### Summary

Implemented one bounded signed i16mf2 computed-mask plus runtime-strided-input
widening dot-product reduction slice:

```text
out_i32[0] = acc_i32[0] + sum_i(
  (cmp_lhs_i32[i] < cmp_rhs_i32[i])
    ? sign_extend(lhs_i16[i * lhs_stride]) *
      sign_extend(rhs_i16[i * rhs_stride])
    : 0)
```

The production path now carries compare mask provenance, dot lhs/rhs source
mem_windows, explicit lhs/rhs element stride ABI roles, scalar seed/result
boundary, source/result config, signed masked widening dot-reduction kind,
mask/tail policy, runtime n/AVL, route metadata, generated bundle emission, and
real `ssh rvv` correctness evidence.

### Main Changes

- Added `typed_computed_mask_strided_input_widening_dot_reduce_pre_realized_body`
  and verifier diagnostics for mask source, stride unit, dot input roles,
  source/accumulator/result config, scalar layouts, and stale route-id residue.
- Extended RVV selected-body realization to produce
  `setvl/with_vl/load/load/strided_load/strided_load/compare/
  masked_widening_dot_reduce/store` from typed facts.
- Extended RVV route planning/provider and construction protocol to derive
  compare loads, strided source loads, byte-stride derivation, masked widening
  product, horizontal reduction, scalar store, ABI order, metadata, and
  fail-closed diagnostics from the typed body.
- Added generated-bundle script support and a harness that checks active and
  inactive mask lanes, skipped strided source elements, positive/negative
  signed products, nonzero seed, scalar-only output, and tail sentinels.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate`
- [OK] Focused lit:
  `pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`
- [OK] Focused negative lit:
  `computed-mask-strided-input-widening-dot-reduction-negative.mlir`
- [OK] Adjacent regression lit for computed-mask dot and strided-input dot.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Generated-bundle dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/masked_strided_dot_dry`
- [OK] Real `ssh rvv` generated-bundle evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/masked_strided_dot_ssh`,
  counts `7,16,23`, `lhs_stride=2`, `rhs_stride=3`, each reporting
  `compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=2,3 skipped_source_elements_ignored scalar_output tail_preserved`
- [OK] Script dry-run lit:
  `rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test`
- [OK] `cmake --build build --target check-tianchenrv` - 245/245 passed
- [OK] `git diff --check`
- [OK] Active-authority scan found no newly introduced positive RVV legacy
  route authority. New hits are a negative `route_id = "rvv-i32m1"` verifier
  test and `implicit-check-not` residue guards in the generated-bundle dry-run
  test.

### Self-Repair

- Fixed a route-description verifier placement mistake caught by focused build.
- Added the combined route profile memory facts after lit exposed a missing
  profile mirror.
- Updated construction-protocol common tests after `check-tianchenrv` exposed
  stale executable role-step expectations.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 151: Stage2 RVV computed-mask byte-strided store executable slice

**Date**: 2026-05-21
**Task**: Stage2 RVV computed-mask byte-strided store executable slice
**Branch**: `main`

### Summary

Implemented the bounded Stage2 RVV `computed_masked_strided_store` slice where
a compare-produced mask selects payload lanes and stores active lanes to
`dst + index * dst_stride_bytes`, while false lanes, skipped byte-stride slots,
and tail sentinels are preserved.

### Main Changes

- Switched the computed-mask strided-store pre-realized body contract from
  element stride to runtime destination byte stride: `dst_stride_bytes` with
  runtime ABI role `destination-byte-stride`.
- Updated RVV selected-body realization to require and materialize the byte
  stride operand through `setvl`, `with_vl`, compare, payload load, old
  byte-strided destination load, `masked_move`, and byte-strided store.
- Updated route planning/provider metadata and codegen so computed-mask store
  uses provider-derived byte-addressed destination pointer mechanics, not
  element-stride multiplication.
- Updated construction protocol, target artifact checks, negative verifier
  coverage, generated-bundle dry-run checks, and script harness evidence for
  counts `7,16,23` crossed with `dst_stride_bytes` `4,8,12`.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Focused lit for computed-mask byte-strided-store verifier, generic
  dataflow regression, selected-body artifact, and generated-bundle dry-run:
  4/4 passed.
- [OK] Generated-bundle dry-run for `computed_masked_strided_store`,
  pre-realized selected-body input, counts `7,16,23`, stride bytes `4,8,12`.
- [OK] Real `ssh rvv` PASS for all nine count/stride cases. Each case reported
  computed mask use, byte-strided store, selected destination writes,
  false-lane preservation, sentinel preservation, and tail preservation.
- [OK] Active-authority scan on the generated computed-mask byte-strided-store
  artifact: no positive `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor, direct-C, source-export, or source-front-door
  residue.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 259/259 passed.

### Self-Repair

- Fixed the generated-bundle FileCheck order after the first focused lit run
  showed the harness function call appears in `run_case` before the `main`
  stride array.
- Kept the retained older element-stride computed-mask path separate from this
  byte-stride contract; this round does not promote it as byte-stride authority.

### Status

[OK] Completed and ready to archive.

### Next Steps

- None for this bounded computed-mask byte-strided-store slice.

## 2026-05-21 - Stage2 RVV Runtime-Strided Load Executable Slice

### Task

- `.trellis/tasks/05-21-stage2-rvv-runtime-strided-load`
- Goal: make bounded `strided_load_unit_store` executable with runtime byte
  stride authority, not old `src_stride` / `lhs-input-stride` element-stride
  authority.

### Implementation

- Added `source-byte-stride` as a runtime ABI role and threaded it through RVV
  dialect verification, config contracts, selected-body realization, route
  planning/provider, construction protocol metadata, and generated-bundle
  evidence.
- Tightened `typed_strided_memory_pre_realized_body` for
  `strided_load_unit_store` to require `source-input-buffer`,
  `source-byte-stride`, and `stride_unit = "byte"`.
- Updated provider emission so `stride_bytes` is passed directly to the RVV
  strided-load leaf; the vector base advances by runtime byte offset through a
  provider-supplied casted byte-scaled pointer expression.
- Extended common EmitC materialization only as neutral mechanics for C-style
  casted scaled pointer expressions, lowering them via EmitC cast/mul/add/cast
  without choosing RVV semantics.
- Updated generated-bundle script/harness to cover `stride_bytes = 4,8,12`
  with sentinel-padded raw source bytes and contiguous output tail checks.

### Evidence

- [OK] Focused lit 6/6:
  dialect negative, EmitC negative, explicit/pre-realized target artifact, and
  explicit/pre-realized generated-bundle dry-run tests.
- [OK] Script syntax and self-test:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Touched C++ tests:
  `tianchenrv-rvv-dialect-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`,
  `tianchenrv-target-artifact-export-test`.
- [OK] Dry-run generated bundles for explicit and pre-realized
  `strided_load_unit_store`, counts `7,16,23`, byte strides `4,8,12`.
- [OK] Real `ssh rvv` PASS for explicit and pre-realized
  `strided_load_unit_store`, all nine count/stride cases, with
  `byte_strided_load contiguous_output tail_preserved`.
- [OK] Active-authority scan: no added positive `RVVI32M1`, `rvv-i32m1`,
  finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `__riscv_*_i32m1`,
  source-front-door/source-export, descriptor/direct-C, or old `src_stride`
  authority.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 259/259 passed.

### Status

[OK] Completed and ready to archive.


## Session 149: Stage2 RVV runtime AVL/VL control boundary

**Date**: 2026-05-21
**Task**: Stage2 RVV runtime AVL/VL control boundary
**Branch**: `main`

### Summary

Implemented a bounded RVV plugin-owned runtime AVL/VL control plan consumed by
the existing `scalar_broadcast_add` and `standalone_reduce_add` executable
selected-body routes. The plan validates runtime `n`/AVL, selected setvl/with_vl
structure, typed config, agnostic policy, runtime ABI order, and mirror metadata
before route planning and generated artifact evidence.

### Main Changes

- Added `RVVRuntimeAVLVLControlPlan` and derivation/verification APIs.
- Routed scalar-broadcast and standalone-reduction pre-realized selected-body
  realization through the shared control plan before materializing `setvl`,
  `with_vl`, and typed RVV operations.
- Made both route-family plans consume the control plan and copy validated
  config/runtime/VL facts into provider route descriptions and artifact mirrors.
- Extended target header metadata and generated-bundle evidence checks for
  `tcrv_rvv.runtime_control_plan`.
- Added focused positive and negative coverage for runtime-control validation.

### Testing

- [OK] Focused FileCheck for scalar-broadcast and standalone selected-body
  realization, route plan mirrors, header export, pre-realized negatives,
  scalar-broadcast EmitC materialization, and runtime-control negative cases.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Generated-bundle dry-runs for `scalar_broadcast_add`, counts `7,16,23`,
  RHS scalars `-37,91`.
- [OK] Generated-bundle dry-runs for `standalone_reduce_add`, counts `7,16,23`,
  seeds `-11,17`.
- [OK] Real `ssh rvv` PASS for `scalar_broadcast_add`, counts `7,16,23`, RHS
  scalars `-37,91`.
- [OK] Real `ssh rvv` PASS for `standalone_reduce_add`, counts `7,16,23`, seeds
  `-11,17`, scalar output correctness, seed preservation, and output tail
  sentinel preservation.
- [OK] Active-authority scan on production diff: no positive `RVVI32M1`,
  `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-export/direct-C/descriptor authority, or public exact
  i32m1 intrinsic route authority introduced.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 252/252 passed.

### Self-Repair

- Split the new realized-route runtime-control negative test out of an existing
  `not --split-input-file` file because the old negative stopped before the new
  split could be observed.
- Adjusted the standalone generated-bundle dry-run FileCheck harness order after
  full lit exposed an assertion-order mismatch.

### Status

[OK] Completed and ready to archive.

### Next Steps

- Later RVV Stage2 consumers can adopt `RVVRuntimeAVLVLControlPlan`; this round
  intentionally bounded production consumption to `scalar_broadcast_add` and
  `standalone_reduce_add`.


## Session 147: Stage2 RVV runtime scalar broadcast add production evidence

**Date**: 2026-05-21
**Task**: Stage2 RVV runtime scalar broadcast add executable slice
**Branch**: `main`

### Summary

Created a fresh production task from the Hermes brief, confirmed current HEAD already had the typed RVV scalar-broadcast route, then closed the active evidence gap by extending the generated-bundle ABI harness to run `scalar_broadcast_add` with multiple runtime RHS scalar values. Verified counts 7/16/23 with rhs_scalar -37 and 91 on real `ssh rvv`.

### Main Changes

- Added repeated `--rhs-scalar` support to `scripts/rvv_generated_bundle_abi_e2e.py` for `scalar_broadcast_add`.
- Recorded RHS scalar values in root/op evidence JSON and generated harness metadata.
- Updated the scalar-broadcast dry-run lit test to require two scalar addends in evidence and harness output.
- Added and completed the Trellis implementation PRD for the bounded task.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Generated-bundle dry-run for `scalar_broadcast_add`, counts `7,16,23`, rhs scalars `-37,91`.
- [OK] Manual FileCheck equivalents for scalar script ROOT/SCALAR/HARNESS.
- [OK] Selected-body REALIZED/PLAN/HEADER FileCheck for scalar broadcast add.
- [OK] EmitC scalar-broadcast materialization and negative fail-closed checks.
- [OK] Real `ssh rvv` PASS for counts `7,16,23` and rhs scalars `-37,91`.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` - 248/248 passed.

### Status

[OK] Completed and ready to archive.

### Next Steps

- None for this bounded slice.


## Session 147: Stage2 RVV runtime scalar broadcast add current-head validation

**Date**: 2026-05-21
**Task**: Stage2 RVV runtime scalar broadcast add executable slice
**Branch**: `main`

### Summary

Closed a duplicate stale scalar-broadcast implementation brief by validating
current HEAD rather than adding a second production path. Current HEAD already
contains the bounded `scalar_broadcast_add` path from `a4cac384` and the later
current-head validation record from `aafa8210`; this round refreshed focused
compiler checks, generated-bundle dry-run, real `ssh rvv` evidence, and
`check-tianchenrv`.

### Main Changes

- Repaired the new Trellis task PRD/status to state the stale-brief finding and
  current-head validation result.
- Did not change compiler production code.
- Refreshed real RVV evidence for `scalar_broadcast_add`, counts `7,16,23`,
  with runtime `rhs_scalar=-37`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | Trellis closeout only |

### Testing

- [OK] Task context validation for `.trellis/tasks/05-21-stage2-rvv-runtime-scalar-broadcast-add-executable-slice`.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV dialect/plugin,
  construction protocol, and target artifact tests.
- [OK] Focused C++ tests:
  `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] Focused scalar-broadcast selected-body, emission-plan, header,
  conversion/materialization, negative, generic dataflow, and selected-body
  negative MLIR checks.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Generated-bundle dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/current-head-scalar-broadcast-20260521-dryrun`.
- [OK] Real `ssh rvv`:
  `PASS op=scalar_broadcast_add counts=7,16,23`.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` - 248/248 passed.
- [OK] Spec update judgment: no `.trellis/spec/**` change needed because this
  round changed no executable contract or convention.

### Status

[OK] **Completed**

### Next Steps

- None - task archived and ready to commit.


## Session 146: Stage2 RVV contraction selected-body realization family boundary

**Date**: 2026-05-21
**Task**: Stage2 RVV contraction selected-body realization family boundary
**Branch**: `main`

### Summary

Consolidated the five existing Stage2 RVV contraction pre-realized
selected-body realization paths behind one plugin-local family realization
plan/helper while preserving route planning/provider authority and executable
behavior.

### Main Changes

- Created `RVVSelectedBodyContractionRealizationPlan` in
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`.
- Added one shared contraction realization helper for:
  `widening_macc_add`, `widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`.
- Rewired the five existing pre-realized selected-body branches to validate
  typed facts, build the shared family plan, and materialize through the shared
  helper.
- Kept route planning/provider as the authority for contraction family route
  emission; common EmitC/export and target metadata were not given RVV
  semantic authority.

### Self-Repair

- Fixed the initial plan design after C++ compile showed `mlir::Location` is
  not default-constructible.
- Restored the previous computed-mask construction order after focused
  FileCheck detected a construction-order mismatch.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Manual FileCheck for five positive pre-realized contraction artifact tests, with `REALIZED`, `PLAN`, and `HEADER` prefixes.
- [OK] Manual fail-closed checks for widening macc, strided dot, computed-mask dot, and computed-mask strided dot negative tests.
- [OK] `cmake --build build --target tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Generated-bundle dry-run for all five named contraction routes at counts `7,16,23`.
- [OK] Real `ssh rvv` PASS for `widening_macc_add`, `widening_dot_reduce_add`, and `computed_masked_strided_input_widening_dot_reduce_add` at counts `7,16,23`.
- [OK] Active-authority diff scan: no new legacy/source/descriptor/common-export authority introduced.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` - 245/245 passed.

### Status

[OK] Completed and ready to archive.

### Next Steps

- None for this bounded selected-body realization consolidation.


## Session 145: Stage2 RVV widening multiply-accumulate executable slice

**Date**: 2026-05-20
**Task**: `stage2-rvv-widening-macc-executable-slice`
**Branch**: `main`

### Summary

Implemented one bounded signed mixed-width RVV widening multiply-accumulate
slice:

```text
out_i32[i] = acc_i32[i] + sign_extend(lhs_i16[i]) * sign_extend(rhs_i16[i])
```

The production path now carries i16mf2 lhs/rhs source facts, i32m1
accumulator/result facts, runtime `n`/AVL, ABI order `lhs,rhs,acc,out,n`,
operation kind, relation, accumulator/result layout, policy, route metadata,
generated bundle emission, and real `ssh rvv` correctness evidence.

### Main Changes

- Added `accumulator-input-buffer` runtime ABI role and the widening macc
  selected-body ABI contract: `const int16_t *lhs`, `const int16_t *rhs`,
  `const int32_t *acc`, `int32_t *out`, `size_t n`.
- Added `typed_widening_macc_pre_realized_body` and generic
  `tcrv_rvv.widening_macc` with fail-closed verifier diagnostics for kind,
  source/accumulator/result configs, relation, layout, roles, memory form, and
  policy.
- Extended RVV selected-body realization to produce
  `setvl/with_vl/load/load/load/widening_macc/store` from typed body/runtime
  facts.
- Extended RVV route planning/provider, construction protocol, and target
  artifact header metadata to derive the mixed-width load/accumulate/store
  route from typed structure, including source/accumulator/result metadata and
  the provider-derived widening macc leaf.
- Added generated-bundle script and harness support for `widening_macc_add`
  with positive/negative i16 operands, nonzero i32 accumulator values, and tail
  sentinel checks.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] Focused verifier/materialization/header checks for
  `generic-widening-macc-dataflow.mlir` and
  `pre-realized-selected-body-artifact-widening-macc-add.mlir`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] Focused lit `-j1`: 3/3 passed
- [OK] Generated-bundle dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-widening-macc-add`
- [OK] Real `ssh rvv` generated-bundle evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-widening-macc-add-rvv`,
  counts `7,16,23`, each reporting
  `signed_products accumulation tail_preserved`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `cmake --build build --target check-tianchenrv -j2` - 235/235 passed
- [OK] `git diff --check`
- [OK] Diff authority scan found no newly introduced positive `RVVI32M1`,
  `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor/direct-C/source-export, or
  common/export RVV semantic authority. The only forbidden-word diff hit is a
  negative description in the new op docs; the exact RVV intrinsic spelling is
  a provider-derived leaf, not route authority.

### Self-Repair

- Fixed construction-protocol conformance route count after adding
  `widening_macc_add`.
- Changed route ABI validation so widening macc validates `lhs,rhs,acc,out,n`
  instead of the older finite-binary `const int32_t *` lhs/rhs contract.
- Extended RVV header metadata evidence so accumulator/result config and
  widening macc relation/layout reach exported artifact comments.
- Updated construction-protocol and target-artifact C++ tests for the new route
  enum and step count after `check-tianchenrv` exposed stale expectations.

### Status

[OK] **Completed**

### Next Steps

- None - task complete



## Session 141: Stage2 RVV masked memory movement executable slice

**Date**: 2026-05-20
**Task**: Stage2 RVV masked memory movement executable slice
**Branch**: `main`

### Summary

Implemented bounded i32 SEW32 LMUL m1 masked unit-stride memory movement with typed source/mask/destination roles, inactive-lane preservation, provider-derived route planning, generated artifacts, and ssh rvv correctness evidence.

### Main Changes

- Added typed masked memory RVV surface: `typed_masked_memory_pre_realized_body`, `mask_load`, and `masked_move`, with verifier diagnostics for mask role, mask memory form, inactive-lane policy, VL consistency, and old-destination preservation.
- Added `MaskInputBuffer` runtime ABI role and selected-body runtime ABI order `src,mask,dst,n`.
- Extended RVV selected-body realization to materialize pre-realized masked memory into `setvl`, source load, mask load, old destination load, masked move, and store.
- Extended RVV EmitC route planning/provider to derive mask load, compare-mask, masked merge, unit-load/store leaves, ABI metadata, inactive-lane contract, and target artifact facts from typed structure.
- Extended construction protocol, target artifact export tests, generated-bundle dry-run tests, and script harness checks for explicit and pre-realized `masked_unit_load_store`.
- Runtime evidence: explicit and pre-realized generated bundles passed `ssh rvv` for counts `7,16,23`, proving active masked lanes update while inactive lanes and tail sentinels are preserved.
- Checks: focused lit 11/11, C++ unit tests, script py_compile/self-test, generated-bundle dry-runs, `cmake --build build --target check-tianchenrv` 211/211, `git diff --check`, and active-authority scan.
- Self-repair: synchronized C++ construction-protocol tests and stale legacy fail-closed FileCheck strings after adding `mask_load` and `masked_move` to the generic Stage2 op list.


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


## Session 144: Stage2 RVV widening integer conversion executable slice

**Date**: 2026-05-20
**Task**: Stage2 RVV widening integer conversion executable slice
**Branch**: `main`

### Summary

Implemented one bounded signed `i16mf2 -> i32m1` RVV widening sign-extension
slice through typed pre-realized body facts, selected-body realization,
provider-owned route planning, generated bundle emission, and real `ssh rvv`
correctness evidence.

### Main Changes

- Added SEW16/LMUL mf2 config helpers and the `lhs: const int16_t *`, `out:
  int32_t *`, `n: size_t` runtime ABI triplet for the new widening slice.
- Extended RVV dialect verification and selected-body realization for
  `sign_extend_widen_vf2` / `signed-i16mf2-to-i32m1`, including explicit C type
  checks for source/result ABI dtype mismatch.
- Extended RVV construction metadata, route planning/provider classification,
  conversion intrinsic derivation, source vector type/load intrinsic metadata,
  and generated-bundle script expectations for `widen_i16_to_i32`.
- Added positive MLIR/script fixtures and fail-closed tests for unsupported
  kind, invalid SEW/LMUL relation, source/result C type mismatch, missing
  runtime role, stale route-id authority, and route/materialization metadata.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [OK] `./build/bin/tianchenrv-construction-protocol-common-test && ./build/bin/tianchenrv-target-artifact-export-test && ./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused `tcrv-opt` verifier/materialization checks for the new
  generic/pre-realized widening fixtures.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `cmake --build build --target check-tianchenrv -j2` - 232/232 passed
- [OK] Generated-bundle dry-run for `widen_i16_to_i32`, counts `7,16,23`
- [OK] `ssh rvv` generated-bundle harness PASS for counts `7,16,23`, with
  `sign_extension_checked tail_preserved`
- [OK] New SSH evidence active-authority scan returned no forbidden
  legacy/source/descriptor/direct-C/source-export tokens

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 144: Stage2 RVV runtime scalar broadcast current-head validation

**Date**: 2026-05-20
**Task**: `stage2-rvv-runtime-scalar-broadcast`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and checked the
bounded runtime scalar broadcast executable slice against current HEAD
`77f9f098`. The brief was stale as an implementation selector: commit
`a4cac384 rvv: add scalar broadcast executable path` already added the
production path for explicit runtime `rhs_scalar` -> generic `tcrv_rvv.splat`
-> binary add -> provider-built route -> generated artifact -> `ssh rvv`
evidence. Fresh current-head validation showed the path still works after the
later Stage2 memory movement commits, so no production code repair was needed.

### Main Findings

- Current HEAD still has `rhs-scalar-value`, generic `tcrv_rvv.splat`,
  `memory_form = "rhs-scalar-broadcast"`, selected-body realization into
  `setvl/with_vl/load/splat/binary/store`, route planning/provider metadata,
  and generated-bundle script support for `scalar_broadcast_add`.
- The generated header still exposes the provider-derived ABI:
  `const int32_t *lhs, int32_t rhs_scalar, int32_t *out, size_t n`.
- Generated-bundle harness uses non-default `rhs_scalar = -37` and checks
  `expected = lhs[index] + rhs_scalar`, proving scalar-driven active-lane
  updates for counts `7,16,23`.
- No production files were changed in this round.

### Testing

- [OK] Trellis task context validation and start.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] Focused C++ tests:
  `tianchenrv-rvv-dialect-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`,
  `tianchenrv-target-artifact-export-test`.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Focused FileCheck pipelines for scalar-broadcast selected-body
  realization, emission plan, header export, EmitC materialization, negative
  fail-closed route cases, dialect dataflow, and pre-realized selected-body
  diagnostics.
- [OK] Generated-bundle dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/current-head-scalar-broadcast-dryrun`.
- [OK] Fresh real `ssh rvv` evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/current-head-scalar-broadcast-ssh`
  printed `PASS op=scalar_broadcast_add counts=7,16,23` with
  `rhs_scalar=-37` for every case.
- [OK] `git diff --check`
- [OK] Diff authority scan excluding the task directory found no newly
  introduced positive `RVVI32M1`, `rvv-i32m1`, finite positive
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed,
  descriptor/direct-C/source-export, or common/export RVV semantic authority.

### Self-Repair

- Re-ran manual FileCheck commands with `/usr/lib/llvm-20/bin/FileCheck` after
  first trying nonexistent `build/bin/FileCheck`.
- Re-ran generated-bundle dry-run with `/usr/lib/llvm-20/bin/llvm-readobj`
  after `llvm-readobj` was not found on PATH.

### Status

[OK] **Completed as current-head validation / stale-brief closure**. The
module behavior is complete in existing production code; this round refreshed
evidence and recorded the task truthfully.

### Next Steps

- None for runtime scalar broadcast. Future work should move to the next
  genuinely missing Stage2 RVV coverage class instead of duplicating this
  already-complete slice.


## Session 143: Stage2 RVV segment2 interleave memory executable slice

**Date**: 2026-05-20
**Task**: Stage2 RVV segment2 interleave memory executable slice
**Branch**: `main`

### Summary

Implemented one bounded i32 SEW32 LMUL m1 RVV segment2 interleave route:
`dst[2*i] = src0[i]` and `dst[2*i+1] = src1[i]`. The selected RVV path now
carries typed source0/source1 field roles, interleaved destination role,
segment count/order, memory forms, runtime n/AVL, config/policy, route
planning metadata, generated artifact export, and real RVV correctness
evidence.

### Main Changes

- Added runtime ABI roles for segment field input buffers and interleaved
  segment output buffers, plus the segment2 interleave runtime ABI contract.
- Added `typed_segment2_interleave_memory_pre_realized_body` and
  `segment2_store` RVV dialect surfaces with verifier coverage for roles,
  segment count, memory forms, config/policy, AVL, stale route metadata, and
  field type/config matching.
- Extended selected-body realization to lower the pre-realized interleave body
  into `setvl`, two generic unit-stride `tcrv_rvv.load` ops, and one
  `tcrv_rvv.segment2_store`.
- Extended RVV route planning/provider/construction metadata to derive ABI
  order `src0,src1,dst,n`, tuple type `vint32m1x2_t`, segment store leaf
  `__riscv_vsseg2e32_v_i32m1x2`, tuple create leaf
  `__riscv_vcreate_v_i32m1x2`, and segment metadata from typed body facts.
- Added positive target/header/generated-bundle fixtures and fail-closed
  tests for unsupported segment count, missing or duplicated source roles,
  swapped field order, mismatched field config, missing destination/AVL role,
  invalid memory form, stale route id, and incomplete typed store body.
- Runtime evidence: generated bundle passed `ssh rvv` for counts `7,16,23`,
  proving field0/source0 lanes land at even destination positions, field1/src1
  lanes land at odd positions, and tail sentinels after `2*n` are preserved.

### Git Commits

- Final commit created after journal update in this session.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] generated-bundle dry-run for `segment2_interleave_unit_load`, counts `7,16,23`
- [OK] `ssh rvv` generated-bundle harness PASS for counts `7,16,23`, with `field_order_distinguishing_lanes` and `tail_preserved`
- [OK] `ninja -C build check-tianchenrv` - 221/221 passed
- [OK] active-authority scan found no new positive `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed,
  descriptor/direct-C/source-export, or common/export semantic route authority;
  the only new `rvv-i32m1` occurrence is the fail-closed negative `route_id`
  diagnostic test.
- [OK] `git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 142: Stage2 RVV computed-mask memory dataflow executable slice

**Date**: 2026-05-20
**Task**: Stage2 RVV computed-mask memory dataflow executable slice
**Branch**: `main`

### Summary

Implemented one bounded i32 SEW32 LMUL m1 computed-mask memory dataflow slice:
`dst[i] = cmp_lhs[i] < cmp_rhs[i] ? src[i] : old_dst[i]`. The selected RVV
body now carries compare lhs/rhs, active source, old/result destination,
runtime n/AVL, predicate, mask role/source/form, inactive-lane policy, and
typed RVV config through plugin-local realization, route planning/provider,
generated artifact export, and ssh rvv correctness evidence.

### Main Changes

- Added `source-input-buffer` runtime ABI role and a five-parameter computed-mask memory ABI order `cmp_lhs,cmp_rhs,src,dst,n`.
- Added `typed_computed_mask_memory_pre_realized_body` and verifier coverage for predicate `slt`, mask source `compare-produced-mask-same-vl-scope`, old-destination preservation, typed config, ABI roles, and stale authority metadata rejection.
- Extended RVV selected-body realization to materialize the pre-realized body into `setvl`, compare lhs/rhs loads, active source load, old destination load, `tcrv_rvv.compare`, `tcrv_rvv.masked_move`, and store.
- Extended route planning/provider, construction protocol, metadata, header ABI, and generated-bundle harness support for `computed_masked_unit_load_store`.
- Added positive and fail-closed MLIR tests for computed-mask materialization/route planning and updated masked memory diagnostics to allow compare-produced masks.
- Runtime evidence: pre-realized computed-mask generated bundle passed `ssh rvv` for counts `7,16,23`, proving compare-active lanes update while compare-false lanes and tail sentinels preserve old destination.
- Checks: script py_compile/self-test, focused dialect/target FileCheck, C++ unit tests, generated-bundle dry-run, real ssh rvv generated-bundle run, `git diff --check`, `check-tianchenrv` 214/214, and active-authority scan.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] focused MLIR positive/fail-closed checks for computed-mask memory
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_masked_unit_load_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --dry-run ...`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_masked_unit_load_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --ssh-target rvv ...`
- [OK] `cmake --build build --target check-tianchenrv -j2`
- [OK] active-authority scan: no new `RVVI32M1`, `rvv-i32m1`, positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed, descriptor, direct-C, or source-export route authority.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 142: Stage2 RVV segment2 deinterleave executable slice

**Date**: 2026-05-20
**Task**: Stage2 RVV segment2 deinterleave executable slice
**Branch**: `main`

### Summary

Implemented bounded i32 segment2 deinterleave RVV route with typed body, route planning/provider, generated-bundle dry-run, ssh rvv evidence, and archived Trellis task.

### Main Changes

- Added typed segment2 pre-realized body and segment2_load RVV dataflow surface for one bounded SEW32 LMUL m1 route.
- Realized selected bodies into setvl/with_vl/segment2_load/field0 move/field1 move/dual unit-stride stores.
- Extended RVV route planning/provider/runtime ABI metadata and common EmitC materialization support for constant scaled pointer expressions used by segmented source loads.
- Added positive target/lit/script tests and negative verifier diagnostics for segment count, field roles, destination/source roles, memory form, and stale route_id authority.
- Validation: check-tianchenrv 217/217; generated-bundle dry-run for counts 7,16,23; ssh rvv PASS for counts 7,16,23 with field_order_distinguishing_lanes and tail_preserved output; git diff --check passed.


### Git Commits

- Final commit created after journal update in this session.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `ninja -C build check-tianchenrv` - 217/217 passed
- [OK] generated-bundle dry-run for `segment2_deinterleave_unit_store`, counts `7,16,23`
- [OK] `ssh rvv` generated-bundle harness PASS for counts `7,16,23`, with `field_order_distinguishing_lanes` and `tail_preserved`
- [OK] active-authority scan found only existing fail-closed/negative legacy references plus the new negative `route_id` diagnostic
- [OK] `git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 143: Stage2 RVV strided-store memory executable slice

**Date**: 2026-05-20
**Task**: Stage2 RVV strided-store memory executable slice
**Branch**: `main`

### Summary

Implemented bounded RVV unit-load to runtime destination-strided-store route, added positive/negative lit and generated-bundle evidence, and validated explicit plus pre-realized ssh rvv correctness at counts 7,16,23.

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


## Session 144: Stage2 RVV widening dot-product reduction executable slice

**Date**: 2026-05-20
**Task**: Stage2 RVV widening dot-product reduction executable slice
**Branch**: `main`

### Summary

Implemented bounded signed i16mf2 x i16mf2 widening dot-product reduction with i32 scalar seed/result, generated-bundle evidence, and ssh rvv correctness.

### Main Changes

### Summary

Implemented one bounded Stage2 RVV signed widening dot-product reduction executable slice:
`out_i32[0] = acc_seed_i32 + sum_i sign_extend(lhs_i16[i]) * sign_extend(rhs_i16[i])`.

### Main Changes

- Added typed pre-realized `tcrv_rvv.typed_widening_dot_reduce_pre_realized_body` and realized `tcrv_rvv.widening_dot_reduce` dataflow surface for i16mf2 lhs/rhs plus i32 scalar seed/result boundary.
- Extended RVV selected-body realization to lower the pre-realized body into setvl/with_vl, two i16mf2 loads, widening dot-reduction compute, and scalar-result store boundary.
- Extended RVV route planning/provider to derive ABI order `lhs,rhs,acc,out,n`, source/result vector facts, scalar seed/result layout, widening product/reduction leaves, metadata mirrors, and fail-closed diagnostics from typed body/config/runtime facts.
- Extended construction protocol, target export metadata, generated-bundle script, and C++/MLIR tests for the new route without adding source-front-door, descriptor, direct-C, or dtype-prefixed helper authority.

### Testing

- [OK] Focused build: `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] Manual FileCheck RUNs for `test/Target/RVV/pre-realized-selected-body-artifact-widening-dot-reduce-add.mlir` realization, emission plan, and header export.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Generated-bundle dry-run for `widening_dot_reduce_add`, counts `7,16,23`.
- [OK] Real `ssh rvv` generated-bundle harness PASS for counts `7,16,23`, proving signed horizontal dot product, nonzero seed addition, scalar `out[0]`, and tail sentinel preservation.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 237/237 passed.
- [OK] Active-authority scan found no new positive `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_reduction_*`, `tcrv_rvv.i32_accumulator_*`, `tcrv_rvv.i32_macc`, source-seed, descriptor, direct-C, or source-export authority; remaining hits are existing fail-closed/negative tests or unsupported-mode diagnostics.
- [OK] `git diff --check`

### Status

[OK] Completed and ready to archive.

### Next Steps

- None for this bounded slice.


### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 145: Stage2 RVV computed-mask widening dot reduction

**Date**: 2026-05-20
**Task**: Stage2 RVV computed-mask widening dot reduction
**Branch**: `main`

### Summary

Implemented bounded computed-mask signed i16 widening dot reduction through RVV typed body, selected-body realization, route planning/provider, generated bundle harness, fail-closed tests, ssh rvv counts 7/16/23, and check-tianchenrv 239/239.

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


## Session 146: Stage2 RVV computed-mask vector select executable slice

**Date**: 2026-05-21
**Task**: Stage2 RVV computed-mask vector select executable slice
**Branch**: `main`

### Summary

Implemented bounded computed-mask vector select through RVV selected-body realization, route planning/provider emission, generated bundle dry-run, real ssh rvv counts 7/16/23, fail-closed tests, authority scan, and check-tianchenrv 248/248.

### Main Changes

- Added bounded pre-realized `tcrv_rvv.typed_computed_mask_select_pre_realized_body` and runtime ABI roles for `cmp_lhs`, `cmp_rhs`, `true_value`, `false_value`, `out`, and `n`.
- Extended RVV selected-body realization to materialize setvl/with_vl, compare lhs/rhs loads, `slt` compare-produced mask, true/false vector loads, generic select/merge, and output store.
- Extended RVV route planning/provider, construction protocol, target artifact export tests, and generated-bundle script for `computed_mask_select` without moving semantics into common EmitC/export.
- Added positive artifact/FileCheck coverage and fail-closed verifier coverage for stale route-id authority, bad predicate, bad true/false roles, bad mask provenance, and incomplete structures.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] Focused build: `cmake --build build --target tcrv-opt tcrv-translate`
- [OK] Focused lit: computed-mask select tests, 3/3 passed.
- [OK] Adjacent compare/select and computed-mask memory lit tests, 5/5 passed.
- [OK] Real `ssh rvv` generated-bundle PASS for `computed_mask_select`, counts `7,16,23`, mixed true/false lanes, runtime n, and tail sentinel preservation.
- [OK] `cmake --build build --target check-tianchenrv` - 248/248 passed.
- [OK] Active-authority scan: no positive legacy route authority in the new computed-mask select path; remaining hits are negative/guardrail checks.
- [OK] `git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 147: Stage2 RVV scalar-broadcast elementwise production family consolidation

**Date**: 2026-05-21
**Task**: Stage2 RVV scalar-broadcast elementwise production family consolidation
**Branch**: `main`

### Summary

Consolidated the existing `scalar_broadcast_add` path into a plugin-local
scalar-broadcast elementwise route-family plan. The route now derives its
runtime ABI order, memory form, target leaf profile, provider support mirror,
header declarations, C type mapping, VL/vector type facts, scalar splat leaf,
elementwise add leaf, and store leaf from typed body/config/runtime facts
instead of scattered route-id or artifact-string authority.

### Main Changes

- Added `RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan` to the RVV
  route analysis surface.
- Extended RVV route planning to build, validate, and apply the scalar-broadcast
  family plan for `ScalarBroadcastAdd` / `RHSScalarBroadcast`.
- Updated RVV provider emission to consume plan-derived setvl/load/splat/add/store
  leaves and required headers.
- Added scalar-broadcast target artifact mirrors for target leaf profile,
  provider-supported mirror, required header declarations, and C type mapping.
- Added focused plugin and MLIR coverage for plan facts, artifact mirrors, stale
  mirror wording, stale RHS load-as-broadcast authority, and stale route-id
  authority.

### Testing

- [OK] Focused build: `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused FileCheck for scalar-broadcast selected-body realization, emission
  plan mirrors, target header export, EmitC materialization, and negative
  selected-body/EmitC/RVV verifier cases.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Generated-bundle dry-run for `scalar_broadcast_add`, counts `7,16,23`,
  RHS scalars `-37,91`.
- [OK] Real `ssh rvv` generated-bundle PASS for `scalar_broadcast_add`, counts
  `7,16,23`, RHS scalars `-37,91`, preserving runtime scalar addend, runtime
  `n`, and tail sentinels.
- [OK] Focused build: `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] Active-authority scan: no new positive `RVVI32M1`, `rvv-i32m1`,
  finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed,
  descriptor/direct-C/source-export, public exact-intrinsic route authority, or
  common/export RVV semantic authority. New exact intrinsic hits are RVV
  plugin-owned leaf outputs/tests; the only new `rvv-i32m1` hit is a negative
  stale route-id test.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 248/248 passed.

### Status

[OK] Completed and ready to archive.

### Next Steps

- None for this bounded slice.


## Session 148: Stage2 RVV standalone reduction executable slice

**Date**: 2026-05-21
**Task**: Stage2 RVV standalone reduction executable slice
**Branch**: `main`

### Summary

Implemented bounded standalone i32 add-reduction executable slice with generated-bundle and ssh rvv evidence.

### Main Changes

### Summary

Implemented the bounded Stage2 RVV standalone horizontal add-reduction executable slice:
`out[0] = acc[0] + sum(lhs[0:n])` for signed i32 / SEW32 / LMUL m1. The route now carries input mem_window, scalar seed input, scalar output binding, runtime n/AVL, typed config, reduction kind, accumulator/result layout, tail policy, provider route plan, generated header/object metadata, and real `ssh rvv` correctness evidence through the RVV-owned path.

### Main Changes

- Added `tcrv_rvv.typed_standalone_reduce_pre_realized_body` and generic `tcrv_rvv.standalone_reduce` verifier coverage.
- Added the standalone reduction ABI contract `lhs,acc,out,n`.
- Extended selected-body realization to materialize setvl/with_vl/load/standalone_reduce/scalar store structure.
- Extended RVV route planning/provider with `RVVSelectedBodyStandaloneReductionRouteFamilyPlan`, scalar seed splat, standalone reduction leaf validation, scalar-output store VL=1, target leaf profile, provider mirror, header declarations, and C type mapping.
- Extended construction protocol, target artifact metadata, generated-bundle ABI harness, and stale diagnostic tests for the standalone reduction path.

### Testing

- [OK] Focused build: `ninja -C build tcrv-opt tcrv-translate`.
- [OK] Focused FileCheck for standalone selected-body realization, route plan mirrors, target header export, and verifier negative cases.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] Generated-bundle dry-run for `standalone_reduce_add`, counts `7,16,23`.
- [OK] Real `ssh rvv` generated-bundle PASS for `standalone_reduce_add`, counts `7,16,23`, seeds `-11,17`, runtime `n`, multi-VL coverage, scalar output correctness, seed preservation, and output tail sentinel preservation.
- [OK] `ninja -C build check-tianchenrv` - 250/250 passed after self-repairing stale test expectations.
- [OK] `git diff --check`.
- [OK] Active-authority scan: no positive standalone reduction route authority from `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-export/direct-C/descriptor, or public exact intrinsic strings. The only standalone `rvv-i32m1` hit is the negative stale route-id verifier test.

### Self-Repair

- Added a standalone config branch and null-value guard in route planning after the first positive test exposed RHS-vector assumptions in config validation.
- Updated construction-protocol and legacy fail-closed tests after the new generic op changed the accepted Stage2 generic-op diagnostic list.

### Status

[OK] Completed and ready to archive.

### Next Steps

- None for this bounded standalone reduction slice.


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


## Session 149: Stage2 RVV masked-store policy executable slice

**Date**: 2026-05-21
**Task**: Stage2 RVV masked-store policy executable slice
**Branch**: `main`

### Summary

Implemented bounded masked unit-store route with explicit mask/tail policy, generated-bundle dry-run, and ssh rvv evidence.

### Main Changes

### Summary

Implemented the bounded Stage2 RVV masked unit-store executable slice for signed i32 / SEW32 / LMUL m1. The selected/pre-realized body now carries explicit mask import, payload source, output ABI buffer, runtime n/AVL, masked-store memory form, tail/mask undisturbed policy, false-lane preservation semantics, and ABI order `src,mask,dst,n` through RVV selected-body realization, route planning/provider, generated target artifact, and real `ssh rvv` execution.

### Main Changes

- Added `tcrv_rvv.masked_store` plus verifier coverage for output-buffer role, mask producer, mask/vector/VL shape, masked-store memory form, and false-lane preservation policy.
- Added the `masked_unit_store` pre-realized selected-body hook with tail/mask undisturbed config contract.
- Extended RVV selected-body realization to materialize `setvl`, `with_vl`, payload `load`, explicit `mask_load`, and direct `masked_store` without old-destination load or `masked_move`.
- Extended RVV route planning/provider/construction protocol with `rvv-generic-masked-unit-store-emitc-route`, ABI `src,mask,dst,n`, masked-store metadata mirrors, masked store target leaf emission, and fail-closed diagnostics.
- Extended generated-bundle ABI evidence tooling with `masked_unit_store` dry-run and ssh harness coverage for mixed true/false masks, false-lane preservation, and tail sentinels.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`.
- [OK] Focused verifier/negative checks for `masked_store`, pre-realized masked-store policy, and route planner compare-mask rejection.
- [OK] Selected-body realization, route plan, and generated header checks for `pre-realized-selected-body-artifact-masked-unit-store.mlir`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Generated-bundle dry-run for `masked_unit_store`, counts `7,16,23`.
- [OK] Real `ssh rvv` PASS for `masked_unit_store`, counts `7,16,23`, active/inactive lanes `3/4`, `7/9`, `10/13`, inactive preservation, and `tail_preserved`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 257/257 passed.
- [OK] `git diff --check`.
- [OK] Active-authority scan: no positive masked-store associated `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export, source-front-door, public exact intrinsic residue, or common/export masked-store semantic authority in production/generated public evidence.

### Self-Repair

- Added masked-store-specific metadata mirrors after the first route-plan check showed only generic memory form/config metadata.
- Updated construction protocol and stale fail-closed diagnostics/tests to include the new generic `tcrv_rvv.masked_store` surface.
- Fixed generated-bundle self-test duplication by making `masked_unit_store` pre-realized-only in expectation routing.

### Status

[OK] Completed and ready to archive.

### Next Steps

- None for this bounded masked-store slice.


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


## Session 150: Stage2 RVV runtime byte-strided store executable slice

**Date**: 2026-05-21
**Task**: Stage2 RVV runtime byte-strided store executable slice
**Branch**: `main`

### Summary

Implemented bounded RVV runtime byte-strided destination store through typed body, provider route, generated bundle, and ssh rvv evidence.

### Main Changes

### Summary

Implemented the bounded Stage2 RVV `unit_load_strided_store` executable slice for signed i32 / SEW32 / LMUL m1 where contiguous source lanes store to `dst + i * dst_stride_bytes`.

### Main Changes

- Added `destination-byte-stride` runtime ABI role and enforced `dst_stride_bytes : size_t` for this route.
- Updated pre-realized and explicit selected-body paths to carry `stride_unit = "byte"`, `src,dst,n,dst_stride_bytes`, unit source load, move, and byte-strided destination store.
- Extended RVV route planning/provider to derive byte-strided destination mirrors and provider-owned byte pointer address mechanics while preserving computed-mask strided-store element-stride mirrors.
- Updated generated-bundle ABI dry-run and ssh harnesses to cover counts `7,16,23` and byte strides `4,8,12`, including selected destination writes, skipped sentinel preservation, and tail sentinel preservation.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] focused lit for byte-strided store verifier/route/artifact/script coverage, 6/6 passed.
- [OK] focused regression lit for computed-mask strided-store and generic dataflow, 3/3 passed.
- [OK] RVV dialect/extension/construction/target C++ smoke tests.
- [OK] generated-bundle dry-run for explicit and pre-realized `unit_load_strided_store`, counts `7,16,23`, stride bytes `4,8,12`.
- [OK] real `ssh rvv` PASS for explicit and pre-realized `unit_load_strided_store`, all nine count/stride cases each, with `selected_destination_writes sentinel_preserved tail_preserved`.
- [OK] active-authority scan: no added positive legacy i32, descriptor/direct-C/source-front-door/source-export, public exact intrinsic, or common/export RVV semantic authority; only negative fixture and FileCheck forbidden strings matched.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 259/259 passed.

### Self-Repair

- Fixed generated harness sentinel logic so `dst_stride_bytes=4` is treated as selected contiguous byte-stride writes, not skipped sentinel slots.
- Split unit-load-strided-store byte-stride mirror from computed-mask strided-store element-stride mirror after full lit exposed accidental metadata reuse.
- Removed `destination-byte-stride` from `strided_load` binding candidates and synchronized the verifier expectation.

### Status

[OK] Completed and ready to archive.

### Next Steps

- None for this bounded byte-strided destination-store slice.


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


## Session 151: Stage2 RVV multiply-add accumulator executable slice

**Date**: 2026-05-21
**Task**: Stage2 RVV multiply-add accumulator executable slice
**Branch**: `main`

### Summary

Implemented bounded RVV macc_add with explicit accumulator input, provider-derived route facts, generated bundle evidence, and ssh rvv PASS.

### Main Changes

- Added explicit `lhs,rhs,acc,out,n` runtime ABI order for bounded `macc_add` and carried `accumulator-input-buffer` through RVV config/runtime ABI contracts.
- Reworked `typed_macc_pre_realized_body` and selected-body realization so pre-realized macc materializes `setvl`, `with_vl`, lhs/rhs/acc loads, `tcrv_rvv.macc`, and output store.
- Extended RVV route planning/provider/construction and target/export tests so macc route facts, header prototype, role sequence, and artifact metadata derive from typed body/config/runtime ABI facts.
- Fixed a real ssh-rvv failure where provider materialization loaded the accumulator from `out` instead of the new `acc` argument; added an EmitC structural check that the third load uses the `acc` function argument and store uses `out`.
- Updated generated-bundle evidence harness for explicit/pre-realized `macc_add` with signed positive/negative lhs/rhs/acc values, signed positive/negative products, active lane correctness, and tail sentinel preservation.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Focused lit for macc dialect/dataflow, pre-realized negatives, EmitC materialization, target header/plan, and generated-bundle dry-run: 8/8 passed.
- [OK] C++ tests: `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`, `tianchenrv-construction-protocol-common-test`, `tianchenrv-target-artifact-export-test`.
- [OK] Generated-bundle dry-run for explicit and pre-realized `macc_add`, counts `7,16,23`.
- [OK] Real `ssh rvv` PASS for pre-realized `macc_add`, counts `7,16,23`, explicit accumulator, signed products, tail preserved.
- [OK] Real `ssh rvv` PASS for explicit selected-body `macc_add`, counts `7,16,23`, explicit accumulator, signed products, tail preserved.
- [OK] Diff-level active-authority scan: no newly introduced positive `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door, or public exact intrinsic authority.
- [OK] `git diff --check` / `git diff --cached --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 259/259 passed.

### Status

[OK] Completed and archived.

### Next Steps

- None for this bounded `macc_add` accumulator slice.


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


## Session 152: Stage2 RVV route operand ABI binding contract

**Date**: 2026-05-21
**Task**: Stage2 RVV route operand ABI binding contract
**Branch**: `main`

### Summary

Implemented a bounded RVV plugin-owned route operand binding contract consumed by existing executable routes, covering `macc_add`, `strided_load_unit_store`, and `unit_load_strided_store`.

### Main Changes

- Added `RVVRouteOperandBindingPlan` to record logical operands, runtime ABI roles/C parameters, and materialized uses for route/provider/header mirrors.
- Rewired provider operand materialization so the bounded consumers derive setvl/control, loads, stores, call operands, and route/header mirrors from the binding contract.
- Added fail-closed validation for plan id, logical operand uniqueness, runtime role uniqueness, logical-operand-to-role mapping, runtime ABI mirror order, missing materialized uses, and runtime ABI mirror mismatch.
- Extended generated-bundle metadata checks and target artifact/header FileCheck coverage for macc and representative byte-strided load/store routes.
- Added negative tests for macc accumulator loaded from `out` and unit-load-strided-store using source as destination.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Generated-bundle dry-run for explicit and pre-realized `macc_add`, `strided_load_unit_store`, and `unit_load_strided_store`, counts `7,16,23`, stride bytes `4,8,12`.
- [OK] Real `ssh rvv` PASS for explicit and pre-realized `macc_add`, `strided_load_unit_store`, and `unit_load_strided_store`, counts `7,16,23`, stride bytes `4,8,12`.
- [OK] Active-authority scan: no added positive legacy i32/RVVI32M1, descriptor/direct-C/source-front-door/source-export, public exact intrinsic, or common/export RVV semantic authority in production/test diff.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 261/261 passed.

### Self-Repair

- Added logical-operand-to-runtime-role validation after the initial contract only checked plan id, uniqueness, and ABI mirror order; plugin tests now catch `n`/stride role swaps directly in the binding contract.
- Updated the unit-load-strided-store negative test to match the earlier verifier-level fail-closed diagnostic.

### Status

[OK] Completed and archived.

### Next Steps

- None for this bounded operand-binding contract.


## Session 153: Stage2 RVV arithmetic and select operand binding adoption

**Date**: 2026-05-21
**Task**: Stage2 RVV arithmetic and select operand binding adoption
**Branch**: `main`

### Summary

Adopted `RVVRouteOperandBindingPlan` for the active ordinary arithmetic,
compare/select, masked arithmetic, reduction, and strided arithmetic route
cluster left outside the previous operand-binding tasks.

### Main Changes

- Added binding-plan IDs, logical-operand role validation, and binding summaries
  for `add`, `sub`, `mul`, `cmp_select`, `computed_mask_select`, `masked_add`,
  `reduce_add`, and `strided_add`.
- Rewired RVV provider materialization so the converted routes consume bound
  ABI operands and materialized-use checks for loads, compares, select values,
  masked arithmetic, reductions, strided operands, stores, setvl/control, and
  header mirrors.
- Extended generated-bundle metadata expectations and target artifact/header
  FileCheck fixtures so plan mirrors and header mirrors match materialized
  operands.
- Added C++ binding-plan validation coverage for binary lhs/rhs swaps,
  computed-mask true/false swaps, and strided rhs/out stride swaps.

### Testing

- [OK] Focused `PLAN`/`HEADER` FileCheck for 15 converted explicit and
  pre-realized target artifact fixtures.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Generated-bundle dry-run for pre-realized `add`, `sub`, `mul`,
  `cmp_select`, `computed_mask_select`, `masked_add`, `reduce_add`, and
  `strided_add`, counts `7,16,23`.
- [OK] Real `ssh rvv` PASS for the same eight routes, counts `7,16,23`;
  predicate true/false lanes, computed-mask select tail preservation,
  masked-add passthrough lanes, reduction output, and strided-add runtime
  operands were exercised.
- [OK] Diff-level active-authority scan: no newly introduced positive
  `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  descriptor/direct-C/source-export/source-front-door, public exact intrinsic,
  route-id, artifact-name, or common/export RVV semantic authority.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 261/261 passed.

### Self-Repair

- Fixed target artifact `PLAN` check ordering after FileCheck showed route
  operand binding metadata is emitted before runtime ABI name and before some
  reduction/strided layout metadata.
- Re-ran the generated-bundle evidence without `--stride-bytes` after checking
  the script contract: `strided_add` uses its own runtime element-stride ABI
  operands, while `--stride-bytes` is reserved for byte-strided load/store
  slices.

### Status

[OK] Completed and archived.

### Next Steps

- Continue with a separate bounded PRD only if adopting operand binding for a
  later Stage2 cluster such as indexed, segmented, widening, contraction, or
  conversion routes.


## Session 154: Stage2 RVV segmented movement operand binding adoption

**Date**: 2026-05-21
**Task**: Stage2 RVV segmented movement operand binding adoption
**Branch**: `main`

### Summary

Adopted `RVVRouteOperandBindingPlan` for the current active segment2 movement
cluster: `segment2_deinterleave_unit_store` and
`segment2_interleave_unit_load`.

### Main Changes

- Added segmented binding-plan IDs, logical operand role validation, and route
  binding summaries for `segment2_deinterleave_unit_store` and
  `segment2_interleave_unit_load`.
- Rewired provider materialization so segment2 source load, field0/field1
  load/store bases, interleaved destination store, setvl/control, route
  metadata, and generated header mirrors consume the binding plan.
- Updated generated-bundle metadata expectations and target artifact/header
  FileCheck fixtures so segmented plan mirrors match materialized operands.
- Added negative coverage for source/output ABI swaps and segment field-source
  swaps.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused PLAN/HEADER FileCheck for both pre-realized segment2 target
  artifact fixtures.
- [OK] Focused negative FileCheck for segment2 deinterleave ABI name swap and
  segment2 interleave field-source swap.
- [OK] Generated-bundle dry-run for pre-realized segment2 deinterleave and
  interleave, counts `7,16,23`.
- [OK] Real `ssh rvv` PASS for both routes, counts `7,16,23`, with
  field-order distinguishing lanes and tail preservation.
- [OK] Diff-level active-authority scan: no newly introduced positive
  `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  descriptor/direct-C/source-export/source-front-door, public exact intrinsic,
  route-id, artifact-name, or common/export RVV semantic authority.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 267/267 passed.

### Self-Repair

- The first interleave negative test exposed that
  `assignRVVGenericSegment2StoreBinding` overwrote field values previously
  bound from field0/field1 source loads. That made the field-order dataflow
  check tautological. The planner now preserves load-owned field values and
  rejects swapped segment2-store field consumption.

### Status

[OK] Completed and archived.

### Next Steps

- Continue with a separate bounded PRD only for another route cluster such as
  widening, contraction, or conversion operand binding adoption.


## Session 155: Stage2 RVV widening macc operand binding adoption

**Date**: 2026-05-21
**Task**: Stage2 RVV widening operand binding adoption
**Branch**: `main`

### Summary

Adopted `RVVRouteOperandBindingPlan` for the active `widening_macc_add`
subcluster. This round intentionally did not convert dot-reduction contraction
routes or widening conversion routes.

### Main Changes

- Added `rvv-route-operand-binding:widening_macc_add.v1` and role validation
  for `lhs`, `rhs`, `acc`, `out`, and runtime `n`.
- Rewired `widening_macc_add` provider materialization so source loads,
  accumulator load, widening-macc compute roles, result store, source/result
  width mirrors, and generated header mirrors are required through the binding
  contract.
- Updated target artifact/header FileCheck fixtures and generated-bundle
  metadata expectations to carry the compact widening-macc binding mirror.
- Added negative C++ coverage for accumulator/output role swaps and
  source/result-width materialized-use mismatch.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] `build/bin/tianchenrv-target-artifact-export-test`.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Focused lit for
  `pre-realized-selected-body-artifact-widening-macc-add.mlir` and
  `rvv-generated-bundle-abi-e2e-pre-realized-widening-macc-add-dry-run.test`.
- [OK] Generated-bundle dry-run for pre-realized `widening_macc_add`, counts
  `7,16,23`.
- [OK] Real `ssh rvv` PASS for `widening_macc_add`, counts `7,16,23`, with
  signed product accumulation and tail preservation.
- [OK] Diff-level active-authority scan: no newly introduced positive
  `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  descriptor/direct-C/source-front-door, public exact intrinsic, route-id,
  artifact-name, or common/export RVV semantic authority.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 267/267 passed.

### Self-Repair

- The first focused lit pass exposed that the full-length
  `route_operand_binding_operands` mirror exceeded the execution-plan
  coherence limit for bounded single-line metadata. The widening-macc mirror
  now uses compact provider-checked materialized-use tokens while preserving
  ABI, load/store, compute-role, width, and header facts.

### Status

[OK] Completed and archived.

### Next Steps

- Continue with a separate bounded PRD for contraction dot-reduction operand
  binding adoption or widening conversion operand binding adoption. Do not
  treat either deferred cluster as completed by this widening-macc round.


## Session 156: Stage2 RVV contraction dot-reduction operand binding adoption

**Date**: 2026-05-21
**Task**: Stage2 RVV contraction dot-reduction operand binding adoption
**Branch**: `main`

### Summary

Adopted `RVVRouteOperandBindingPlan` for the active contraction
dot-reduction cluster. The converted routes are `widening_dot_reduce_add`,
`strided_input_widening_dot_reduce_add`,
`computed_masked_widening_dot_reduce_add`, and
`computed_masked_strided_input_widening_dot_reduce_add`. No inactive scoped
dot-reduction route was found, and `widening_macc_add` remained an already
converted predecessor from commit `4db6013d`.

### Main Changes

- Added compact route operand binding plan IDs and role validation for the four
  scoped dot-reduction routes.
- Rewired RVV route provider materialization so dot lhs/rhs values, optional
  compare/mask values, scalar accumulator seed, scalar output, runtime `n`, and
  optional lhs/rhs stride roles are consumed through the binding plan.
- Updated target artifact/header FileCheck fixtures and generated-bundle
  metadata expectations so materialized operands and mirrors use the same
  contract.
- Added structural positive/negative plugin coverage for role swaps,
  accumulator/output swaps, dot-source swaps, mask/materialized-use mismatch,
  stride-role swaps, and source/result-width materialized-use mismatch.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused target artifact lit for the four scoped dot-reduction MLIR
  tests: 4/4 passed.
- [OK] Focused generated-bundle dry-run lit for existing dot-reduction script
  tests: 3/3 passed.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Generated-bundle dry-run for all four scoped routes at counts
  `7,16,23`: `rvv_generated_bundle_abi_e2e: dry_run_success`, artifact root
  `artifacts/tmp/stage2_contraction_operand_binding/dry-run-all`.
- [OK] Real `ssh rvv` generated-bundle run for all four scoped routes at
  counts `7,16,23`: PASS for ordinary dot, strided dot, computed-mask dot, and
  computed-mask strided dot, including signed dot correctness, accumulator seed
  addition, mask/stride checks where applicable, and tail preservation.
- [OK] Diff-level active-authority scan: no newly added positive `RVVI32M1`,
  `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  descriptor/direct-C/source-export, source-front-door, public exact intrinsic,
  stale route-id, artifact-name, or common/export RVV semantic authority.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 267/267 passed.

### Self-Repair

- Reordered new HEADER FileCheck assertions after the existing contraction
  relation/profile/provider mirrors to match the production header writer.
- Compactified the computed-mask strided dot binding mirror after the first
  target artifact run exposed the bounded single-line metadata limit, while
  preserving provider-checked ABI, mask, stride, width, accumulator, output,
  runtime, and header facts.

### Status

[OK] Completed and ready to archive.

### Next Steps

- None for the bounded active dot-reduction operand binding cluster. A future
  task may handle widening conversion operand binding separately.


## Session 157: Stage2 RVV conversion operand binding adoption

**Date**: 2026-05-21
**Task**: Stage2 RVV conversion operand binding adoption
**Branch**: `main`

### Summary

Adopted `RVVRouteOperandBindingPlan` for the active widening conversion route
cluster. The converted routes are `widen_i32_to_i64` and `widen_i16_to_i32`.
No additional active conversion route was found or converted; parser-only,
narrowing, unsigned, floating, frontend, and clone variants were left out of
scope.

### Main Changes

- Added compact conversion route operand binding plan IDs and role validation
  for `lhs`, `out`, and `n`.
- Rewired RVV conversion provider checks so source load, conversion source,
  destination store, runtime loop/control, source/result config mirrors,
  conversion relation mirrors, and generated header mirrors are consumed
  through the binding plan before emission.
- Updated conversion target artifact/header FileCheck fixtures and
  generated-bundle metadata expectations so mirrors and materialized operands
  share the same binding contract.
- Added plugin positive and negative coverage for conversion binding plans,
  including source/destination role swap and conversion-direction/materialized
  use mismatch.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] `build/bin/tianchenrv-target-artifact-export-test`.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Focused FileCheck for `widen_i32_to_i64` and `widen_i16_to_i32`
  REALIZED, PLAN, and HEADER outputs.
- [OK] Focused `tcrv-opt`/FileCheck validation for
  `test/Dialect/RVV/generic-widening-conversion-dataflow.mlir`.
- [OK] Generated-bundle dry-run for both converted routes at counts `7,16,23`:
  `rvv_generated_bundle_abi_e2e: dry_run_success`.
- [OK] Real `ssh rvv` generated-bundle run for both converted routes at counts
  `7,16,23`: PASS for `widen_i32_to_i64` and `widen_i16_to_i32`, including
  sign/value-distinguishing correctness and tail preservation.
- [OK] Diff-level and generated-artifact active-authority scans: no newly added
  positive `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export, source-front-door,
  public exact intrinsic, stale route-id, artifact-name, or common/export RVV
  semantic authority.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 267/267 passed.

### Self-Repair

- Raw source-tree `llvm-lit` was not available/configured for direct use
  (`llvm-lit` absent from `/usr/lib/llvm-20/bin` and source lit config lacked
  `tianchenrv_obj_root`), so focused direct FileCheck commands were used and
  then superseded by the build-tree `check-tianchenrv` target.

### Status

[OK] Completed and ready to archive.

### Next Steps

- None for the bounded active conversion operand binding cluster.


## Session 156: Stage2 RVV closure-gated compare/select `sle` expansion

**Date**: 2026-05-21
**Task**: `05-21-stage2-rvv-closure-gated-compare-select`
**Branch**: `main`

### Summary

Implemented a bounded Stage2 RVV compare/select coverage expansion for signed
less-or-equal (`sle`) on the corrected generic typed `tcrv_rvv` surface. The
route support remains RVV plugin-owned, selected-body-derived, and
`RVVRouteOperandBindingPlan` closure-gated.

### Main Changes

- Extended generic typed compare predicate verification to accept `eq`, `slt`,
  and bounded signed `sle` where compare/select routes support them.
- Extended pre-realized `cmp_select` and `computed_mask_select` realization so
  `sle` materializes into explicit `setvl/with_vl/load/compare/select/store`
  structure.
- Added `comparePredicateKind` route description authority, predicate-derived
  RVV compare intrinsic selection, emission metadata, and target header mirror
  validation for `tcrv_rvv.compare_predicate_kind`.
- Added explicit and pre-realized target fixtures for `cmp_select_sle` and
  `computed_mask_select_sle`.
- Extended generated-bundle ABI evidence and harnesses for explicit and
  pre-realized `cmp_select_sle` and `computed_mask_select_sle` with
  equality-distinguishing signed data, runtime counts `7,16,23`, and
  computed-mask tail sentinel preservation.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Focused FileCheck for explicit/pre-realized `cmp_select_sle` and
  `computed_mask_select_sle` PLAN/HEADER/REALIZED paths.
- [OK] Focused verifier checks:
  `test/Dialect/RVV/computed-mask-select-dataflow.mlir` and
  `test/Transforms/LoweringBoundary/rvv-pre-realized-compare-select-selected-body-negative.mlir`.
- [OK] Generated-bundle dry-runs for explicit/pre-realized `cmp_select_sle`
  and `computed_mask_select_sle`, counts `7,16,23`.
- [OK] Real `ssh rvv` PASS for explicit/pre-realized `cmp_select_sle` and
  `computed_mask_select_sle`, counts `7,16,23`. `cmp_select_sle` reported both
  predicate true and false lanes; `computed_mask_select_sle` reported both
  select true and false lanes plus `tail_preserved`.
- [OK] Active-authority scan: full scan hits were existing legacy inventory,
  negative tests, and guard text; diff-level scan found no newly added positive
  `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  descriptor/direct-C/source-export/source-front-door authority.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 279/279 passed.

### Self-Repair

- Fixed one stale masked-arithmetic helper call after the compare intrinsic
  helper was split into explicit predicate-derived variants.
- Added target header mirror support after focused FileCheck showed the plan
  carried `compare_predicate_kind` but header export did not.
- Added explicit generated-bundle expectation for `computed_mask_select_sle`
  after the first explicit dry-run showed script support existed only for the
  pre-realized table.

### Status

[OK] Completed and ready to archive.

### Next Steps

- None for this bounded signed `sle` compare/select submodule.
