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
