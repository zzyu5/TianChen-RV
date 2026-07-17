# Stage2 RVV computed-masked indexed scatter-store artifact ABI boundary

## Goal

Make the bounded Stage 2 RVV
`computed_masked_indexed_scatter_store_unit_load` compiler path a tested
artifact ABI boundary:

```text
selected tcrv.exec RVV variant
  -> typed_computed_mask_indexed_scatter_pre_realized_body
  -> RVV plugin-local realization into setvl / with_vl / load / index_load /
     compare / masked_indexed_store
  -> provider-built TCRVEmitCLowerableRoute
  -> target artifact validation
  -> generated bundle/header/object
  -> ssh rvv correctness evidence
```

This round is not a new broad indexed-memory matrix. It is one scatter-store
witness for the memory-movement class paired with the archived gather-load
task. The route authority must remain the typed/realized `tcrv_rvv` body and
RVV provider facts, not route ids, metadata strings, exact intrinsic spelling,
artifact names, descriptors, or script-only evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-masked indexed scatter-store artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` was clean.
* Initial `git log --oneline -8` started at
  `ab188268 chore(task): archive 06-02-stage2-rvv-computed-masked-indexed-gather-load-artifact-abi-boundary`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* The archived gather-load task exists under
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-indexed-gather-load-artifact-abi-boundary/`
  and records a verification-only closeout for an already-existing gather path.
* Live code already contains the scatter ODS surface, selected-body realization,
  provider route facts, target validation, fixture, dry-run script coverage, and
  direct pre-realized shortcut fail-close hook. This task therefore should not
  duplicate the path; it should tighten the missing provider/target fact
  boundaries and evidence checks that make the existing path fail closed.

## Current Repository Evidence

Live inspection found:

* `TypedComputedMaskIndexedScatterPreRealizedBodyOp` in
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`, binding runtime ABI operands
  `cmp_lhs, cmp_rhs, src, index, dst, n`, compare predicate, index facts,
  inactive-lane no-write policy, and RVV config.
* `MaskedIndexedStoreOp` carries `memory_form`, `index_eew`, `offset_unit`,
  `index_uniqueness`, and `inactive_lane_policy` for the bounded scatter slice.
* RVV plugin realization/route planning requires compare lhs/rhs loads,
  unit-stride payload source load, `index_load`, compare-produced mask, and one
  `masked_indexed_store` consuming the same mask/index/VL and source payload.
* Provider facts already include runtime ABI order, binding plan/summary,
  target leaf/profile, required headers, C type mapping, provider-supported
  mirror, source/destination memory forms, index source, index EEW, offset unit,
  index uniqueness, and indexed destination memory form.
* Target validation already consumes provider description fields for indexed
  computed-mask memory rather than accepting artifact metadata alone.
* Existing tests cover the realized structure, plan/header positive checks,
  stale binding summary, stale index uniqueness, generated-bundle dry-run, and
  direct route-entry shortcut rejection.

## Gap This Round Closes

The remaining focused gap is regression proof, not a second production path:

* The generated-bundle dry-run does not explicitly assert all scatter-specific
  provider-derived facts named in the brief, especially `provider_supported_mirror`,
  `target_leaf_profile`, headers/C type mapping, `index_eew`, `offset_unit`, and
  `index_uniqueness`.
* The target artifact C++ regression matrix has scatter binding and uniqueness
  negative cases, but it does not yet prove stale scatter `index_eew`,
  `offset_unit`, `indexed_destination_memory_form`, provider mirror, target leaf,
  required header declaration, or C type mapping mirrors fail closed for this
  exact route.

## Requirements

* Keep route support rooted in the selected typed/pre-realized `tcrv_rvv` body,
  RVV plugin-local realization, and provider-built route facts.
* Preserve runtime ABI order `cmp_lhs,cmp_rhs,src,index,dst,n`.
* Preserve the binding summary plan
  `rvv-route-operand-binding:computed_masked_indexed_scatter_store_unit_load.v1`
  and all exported runtime ABI entries with `abi` and `hdr`.
* Preserve compare predicate `slt`, mask role/source/form, unit-stride source
  load, runtime index vector ABI, `index_eew = 32`, `offset_unit = element`,
  `index_uniqueness = unique`, masked indexed destination store, and
  false-lane no-write destination preservation.
* Target artifact validation must fail closed when provider facts or candidate
  mirrors are stale/missing for the scatter route.
* Generated-bundle dry-run evidence must expose provider-derived route-family,
  target, header/type, mask, index, source/destination memory, and no-write
  preservation facts.
* Real `ssh rvv` evidence must cover counts `0,1,16,17,257` and patterns `0,1`
  when runtime correctness is claimed.

## Acceptance Criteria

* [x] Focused production/test diff strengthens the scatter artifact ABI
      boundary; no metadata-only archive closeout.
* [x] Target artifact C++ tests prove stale scatter provider mirror, target
      leaf/profile, header declarations, C type mapping, index EEW, offset unit,
      and indexed destination memory form fail closed.
* [x] Existing or tightened FileCheck coverage proves the selected body realizes
      into compare-produced mask, unit-stride payload source load, index load,
      and `masked_indexed_store`, not gather/strided/legacy i32 paths.
* [x] Generated-bundle dry-run checks record provider-derived scatter facts:
      runtime ABI order, binding plan/summary, route-family plan,
      `provider_supported_mirror`, target leaf/profile, required headers,
      C type mapping, source memory form, indexed destination memory form,
      index source, `index_eew`, `offset_unit`, and `index_uniqueness`.
* [x] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,16,17,257` and patterns `0,1`, proving active lanes store `src[i]`
      to `dst[index[i]]`, inactive indexed lanes preserve old destination,
      noncontiguous unique indices are used, source/tail sentinels are
      preserved, and runtime `n`/AVL is honored.
* [x] Focused checks, `git diff --check`, and bounded old-authority scan pass.
* [x] Task is finished/archived and one coherent commit is created if the task
      completes.

## Out Of Scope

* Reworking the already-archived computed-masked indexed gather-load path.
* Broad indexed-memory matrices, scatter variants beyond this witness,
  dtype/LMUL clone batches, frontend/source-front-door positive routes,
  high-level Linalg lowering, dashboards, global tuning databases, descriptor
  routes, direct-C/source exporters, or common EmitC RVV semantic inference.
* Treating route ids, artifact names, exact intrinsic spelling, test names,
  metadata mirrors, or harness sentinels as indexed-store, mask, no-write,
  dtype/config, runtime ABI, policy, or route authority.

## Technical Notes

Specs and task context read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/guides/index.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-indexed-gather-load-artifact-abi-boundary/prd.md`

Relevant files inspected:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-indexed-scatter-store.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-indexed-scatter-store-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-indexed-scatter-store-fail-closed.test`
* `test/Target/TargetArtifactExportTest.cpp`

## Implementation Notes

This round added operation-specific target validator checks for compare/select
mask route `required_header_declarations` and `c_type_mapping` summaries. This
prevents a route from passing target artifact validation merely because the
rebuilt route object has usable headers/type mappings while the provider
summary or artifact mirror is stale. The manual indexed gather fixture was
also corrected to keep `index:u32m1` explicit in its C type mapping summary,
matching the existing provider/script/fixture truth.

The scatter-specific C++ target artifact regressions now fail closed when the
provider route description or candidate artifact metadata carries stale
provider mirror, target leaf/profile, header declarations, C type mapping,
index EEW, offset unit, index uniqueness, binding summary, or indexed
destination memory form.

The generated-bundle dry-run FileCheck now asserts scatter-specific target
leaf/profile, provider-supported mirror, headers, C type mapping, index EEW,
offset unit, and index uniqueness facts.

The lowering-runtime spec now records the provider header/type summary contract
so future route families do not repeat this mirror-only validation gap.

## Validation Evidence

Commands run:

```bash
rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py
rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2
rtk build/bin/tianchenrv-target-artifact-export-test
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter computed-masked-indexed-scatter-store
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-indexed-scatter-store-artifact-abi-boundary/dry-run --run-id pre-realized-computed-mask-indexed-scatter-store --overwrite --op-kind computed_masked_indexed_scatter_store_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-indexed-scatter-store-artifact-abi-boundary/ssh-rvv --run-id pre-realized-computed-mask-indexed-scatter-store --overwrite --op-kind computed_masked_indexed_scatter_store_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test
rtk rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|descriptor-driven|source-front-door" lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-indexed-scatter-store-dry-run.test .trellis/tasks/06-02-06-02-stage2-rvv-computed-masked-indexed-scatter-store-artifact-abi-boundary
rtk git diff -U0 -- lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-indexed-scatter-store-dry-run.test | rg -n "^\\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|descriptor-driven|source-front-door)"
rtk git diff --check
```

Focused lit result:

```text
Total Discovered Tests: 475
  Excluded: 470 (98.95%)
  Passed  :   5 (1.05%)
```

Generated-bundle dry-run result:

```text
rvv_generated_bundle_abi_e2e: dry_run_success
artifact_dir: artifacts/tmp/06-02-stage2-rvv-computed-masked-indexed-scatter-store-artifact-abi-boundary/dry-run/pre-realized-computed-mask-indexed-scatter-store
```

Real `ssh rvv` result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=computed_masked_indexed_scatter_store_unit_load counts=0,1,16,17,257 patterns=0,1
```

The remote output includes per-count/per-pattern success lines. Counts `16`,
`17`, and `257` include active lanes, inactive preserved lanes, noncontiguous
unique indexed destination lanes, source preservation, and tail preservation.

The bounded touched-file old-authority scan found only existing legacy-negative
fixture text in `test/Target/TargetArtifactExportTest.cpp` and negative
`implicit-check-not` strings in the dry-run test. The diff-only old-authority
scan produced no matches.

## Current Phase

Finish/archive.
