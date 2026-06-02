# Stage2 RVV computed-masked segment2 update artifact ABI boundary

## Goal

Make the bounded Stage 2 RVV
`computed_masked_segment2_update_unit_load` compiler path a tested artifact ABI
boundary:

```text
selected tcrv.exec RVV variant
  -> typed_computed_mask_segment2_store_pre_realized_body
  -> RVV plugin-local realization into setvl / with_vl / compare loads /
     field0 and field1 unit-stride payload loads / compare / binary add /
     masked_segment2_store
  -> provider-built TCRVEmitCLowerableRoute
  -> target artifact validation
  -> generated bundle/header/object
  -> ssh rvv correctness evidence
```

The route authority must remain the typed/realized `tcrv_rvv` body and RVV
provider facts. Route ids, artifact names, metadata mirrors, exact intrinsic
spellings, descriptors, source-front-door markers, C strings, test names, or
script expectations are mirrors or evidence only.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-masked segment2 update artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` showed no short-status entries through `rtk`
  filtering.
* Initial `git log --oneline -8` started at
  `fdad8778 rvv: validate computed masked segment2 store artifact facts`,
  followed by `ceb90af2 chore: record journal` and
  `de423987 rvv: validate computed masked segment2 load artifact facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* The archived segment2-store task under
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-segment2-store-artifact-abi-boundary/`
  provides the closest adjacent route pattern: tighten provider/target
  fail-closed checks, tighten generated-bundle dry-run facts, run real
  `ssh rvv` evidence, finish/archive, then commit.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` agree that the active path
  must be selected typed RVV body -> RVV plugin-owned realization/provider ->
  common EmitC -> target artifact -> `ssh rvv` evidence for correctness
  claims.
* The relevant guides confirm that this is plugin-local RVV execution work:
  capability/profile facts constrain legality, `tcrv_rvv` owns compute/memory
  body structure, the RVV plugin owns realization/provider facts, and common
  EmitC/export must stay neutral.

## Current Repository Evidence

Live inspection found:

* `TypedComputedMaskSegment2StorePreRealizedBodyOp` in
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` already models both
  `computed_masked_segment2_store_unit_load` and
  `computed_masked_segment2_update_unit_load`; the update spelling requires
  `arithmetic_kind = "add"` and binds runtime ABI operands
  `cmp_lhs, cmp_rhs, src0, src1, dst, n`.
* `MaskedSegment2StoreOp` carries the interleaved destination, compare-produced
  mask, field0/field1 payload vectors, VL, `segment_count = 2`, destination
  memory form, field roles, and inactive-lane no-write policy.
* `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp` already has
  a computed-mask segment2 update planning owner, update-specific route
  operand binding plan
  `rvv-route-operand-binding:cmseg2_update_unit_load.v1`, and plugin-local
  realization that loads compare operands and field payloads, computes
  `binary {kind = add}` over field0/field1, and stores the updated field0 plus
  original field1 through `tcrv_rvv.masked_segment2_store`.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` already has
  computed-mask segment2 update constants and validates provider facts for
  runtime ABI order, binding plan/summary, target leaf/profile, provider
  mirror, headers, C type mapping, computed-mask plan, mask facts, memory
  forms, field roles/names, source forms, inactive-lane contract, update
  arithmetic kind/callee, and stale segment-load rejection.
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-update.mlir`
  already proves positive selected-body realization and core PLAN/HEADER
  facts, with several stale mirror negative cases.
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-update-dry-run.test`
  already checks the generated harness for active update behavior and
  inactive/tail/source preservation, but it only checks a subset of the
  provider-derived metadata fact surface named by the brief.
* `test/Target/TargetArtifactExportTest.cpp` already has a segment2 registry
  section with positive load/store/update fixtures and several update negative
  mutations, but this round must verify whether update-specific store/arithmetic
  mirrors and stale load/extract residue are covered as tightly as the newly
  completed store path.

## Gap This Round Closes

This round closes the update-specific validation/evidence gap around the
existing production path:

* Strengthen computed-mask segment2 update target validation/tests where stale
  provider facts or stale candidate mirrors can still look like acceptable
  update evidence.
* Tighten generated-bundle dry-run FileCheck coverage so evidence records the
  full provider-derived update fact surface, not just route id, binding,
  arithmetic, store intrinsic, and harness behavior.
* Run real generated-bundle `ssh rvv` correctness for counts
  `0,1,7,16,23,257` and patterns `0,1`, proving active lanes store
  `add(src0, src1)` into the first interleaved slot and original `src1` into
  the second slot while inactive lanes preserve both old destination fields,
  sources/tails remain unchanged, and runtime `n`/AVL is honored.

## Requirements

* Keep route support rooted in selected typed/pre-realized `tcrv_rvv` body,
  RVV plugin-local realization, segment2 route-family planning owner, segment2
  memory statement-plan boundary, and provider-built route facts.
* Preserve runtime ABI order `cmp_lhs,cmp_rhs,src0,src1,dst,n`.
* Preserve the binding summary plan
  `rvv-route-operand-binding:cmseg2_update_unit_load.v1` and exported runtime
  ABI entries with `abi` and `hdr`.
* Preserve compare predicate `slt`, compare-produced mask role/source/form,
  `segment_count = 2`, field0/field1 roles as input buffers, source0/source1
  unit-stride loads, `arithmetic_kind = add`, `typed_compute_op =
  tcrv_rvv.binary`, active field0 update as `add(field0, field1)`, field1
  passthrough into the second segment slot, masked segment2 interleaved
  destination store, false-lane no-write preservation of both destination
  fields, SEW `32`, LMUL `m1`, and agnostic tail/mask policy.
* Target artifact validation must fail closed when provider facts or candidate
  mirrors are stale/missing for this exact computed-mask segment2 update route,
  including stale segment-load, field-extract, store-only, or arithmetic-residue
  facts.
* Generated-bundle dry-run evidence must expose provider-derived route-family,
  target, header/type, mask, segment, source/destination memory, field role,
  tuple/store, update arithmetic, no-write, and provider support mirror facts.
* Runtime correctness evidence must use real `ssh rvv` output for counts
  `0,1,7,16,23,257` and patterns `0,1`.

## Acceptance Criteria

* [x] Focused production/test diff strengthens the segment2 update artifact ABI
      boundary; no metadata-only archive closeout.
* [x] Target artifact C++ tests prove stale or missing update provider facts
      fail closed for provider mirror, target leaf/profile, header
      declarations, C type mapping, computed-mask route-family plan, mask
      role/source/form, compare predicate, segment count, segment memory
      layout, source0/source1 unit-stride forms, interleaved destination memory
      form, field roles/names, inactive-lane no-write contract, binding
      summary, update arithmetic kind/callee, tuple create/store mirror facts,
      stale segment-load facts, stale field-extract facts, and route statement
      operands.
* [x] Existing or tightened FileCheck coverage proves the selected body
      realizes into compare-produced mask, field0/field1 unit-stride loads,
      `binary {kind = add}`, tuple creation, and masked interleaved segment2
      store, not descriptor/source-front-door/legacy i32 paths.
* [x] Generated-bundle dry-run checks record provider-derived update facts:
      runtime ABI order, binding plan/summary, computed-mask route-family
      plan, `provider_supported_mirror`, `target_leaf_profile`, required
      headers, C type mapping, mask role/source/form, segment count, segment
      memory layout, source/destination memory forms, field roles/names, field
      source forms, update arithmetic facts, tuple/store mirrors, and
      inactive-lane no-write contract.
* [x] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,7,16,23,257` and patterns `0,1`, proving active lanes store
      `src0[i] + src1[i]` into `dst[2*i]` and `src1[i]` into `dst[2*i+1]`,
      inactive lanes preserve old interleaved destination fields, source and
      tail sentinels are preserved, and runtime `n`/AVL is honored.
* [x] Focused checks, `git diff --check`, and bounded old-authority scan pass.
* [x] Task is finished/archived and one coherent commit is created if the task
      completes.

## Out Of Scope

* Segment2 load/store redo or broad segment matrix expansion.
* Reworking indexed-memory gather/scatter paths, dtype/LMUL clone batches,
  frontend/source-front-door positive routes, high-level Linalg lowering,
  reduction/matmul expansion, dashboards, global tuning databases, descriptor
  routes, direct-C/source exporters, or common EmitC RVV semantic inference.
* Treating route ids, artifact names, exact intrinsic spelling, test names,
  metadata mirrors, harness sentinels, or generated script expectations as
  segment count, mask, field, no-write, dtype/config, runtime ABI, policy, or
  route authority.

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
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-segment2-store-artifact-abi-boundary/prd.md`

Relevant files inspected:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-update.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-update-dry-run.test`
* `test/Target/TargetArtifactExportTest.cpp`

## Implementation Notes

This round tightened the production computed-mask segment2 update artifact ABI
boundary in target validation and tests:

* `validateRVVComputedMaskSegment2MemoryProviderFacts` now validates
  provider-derived computed-mask segment2 callee facts directly:
  * segment2 load routes require the masked segment-load callee, tuple-create
    callee, and field-extract callee.
  * segment2 store/update routes require no segment-load callee, require the
    masked segment-store callee, and require the tuple-create callee in the
    store/update tuple slot.
  * computed-mask segment2 update requires the exact provider-derived
    `__riscv_vadd_vv_i32m1` update arithmetic callee, not merely a non-empty
    arithmetic mirror.
* `test/Target/TargetArtifactExportTest.cpp` now has update-specific
  fail-closed regressions for stale target leaf profile, computed-mask
  route-family plan, mask role/form, segment count, stale segment-load fact,
  wrong masked segment-store callee, stale field-extract/tuple-create fact,
  and stale update arithmetic callee. The adjacent store callee expectation was
  updated to the new earlier provider-fact failure.
* The generated-bundle dry-run FileCheck for
  `computed_masked_segment2_update_unit_load` now asserts the full
  provider-derived update fact surface: route family, target profile,
  headers/type map, mask role/source/form, segment layout/count, tuple C type,
  tuple create, masked segment-store, field roles/names/source forms, update
  arithmetic, and no-write contracts.

Self-repair performed during validation:

* The first provider-callee patch treated the load route's
  `segmentStoreIntrinsic` field as a store callee. Live code inspection showed
  that computed-mask segment2 load uses that field as a tuple-create callee
  while candidate metadata keeps `segment_store_intrinsic` empty. The validator
  was corrected before running tests: load validates load/tuple-create/
  field-extract; store/update validate no segment-load plus masked store and
  tuple-create.
* The store negative test expected the older route-statement failure for a
  stale segment-store callee. Because target validation now rejects that fact
  earlier as a provider-derived callee mismatch, the expected fragments were
  updated to the new fail-closed boundary.

Spec-update judgment:

* No `.trellis/spec/` change was needed. The existing RVV plugin spec already
  requires computed-mask segment2 update target export to validate compare/mask
  construction, field payload loads, update arithmetic, tuple creation, masked
  segment-store statements, and stale segment-load/field-extract rejection.
  This round implements and tests that existing contract more tightly rather
  than introducing a new contract.

## Validation Evidence

Commands run:

```bash
rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2
rtk ./build/bin/tianchenrv-target-artifact-export-test
rtk cmake --build build --target tcrv-opt tcrv-translate -j2
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter computed-masked-segment2-update  # from build/test
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-segment2-update-artifact-abi-boundary/dry-run --run-id pre-realized-computed-mask-segment2-update --overwrite --op-kind computed_masked_segment2_update_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-segment2-update-artifact-abi-boundary/ssh-rvv --run-id pre-realized-computed-mask-segment2-update --overwrite --op-kind computed_masked_segment2_update_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20
rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test
rtk git diff --check
rtk git diff -- lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-update-dry-run.test | rg -n "^\\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|source-front-door|source-export|direct-C|descriptor)"
```

Results:

* `tianchenrv-target-artifact-export-test` passed.
* focused lit discovered 475 tests, excluded 470, and passed the 5
  `computed-masked-segment2-update` tests.
* generated-bundle dry-run returned `rvv_generated_bundle_abi_e2e:
  dry_run_success`.
* real `ssh rvv` generated-bundle run returned:

```text
rvv_generated_bundle_abi_e2e: success
tcrv_rvv_generated_bundle_abi_computed_masked_segment2_update_unit_load_ok counts=0,1,7,16,23,257 patterns=0,1
PASS op=computed_masked_segment2_update_unit_load counts=0,1,7,16,23,257 patterns=0,1
```

* The run printed per-count/per-pattern success lines for `n =
  0,1,7,16,23,257` and patterns `0,1`, including active/inactive lane counts,
  inactive preservation counts, field-distinguishing lane counts,
  `source_preserved`, and `tail_preserved`.
* `py_compile`, script `--self-test`, and `git diff --check` passed.
* Full touched-file old-authority scan found only pre-existing legacy/fail-closed
  inventory in the large C++ test and negative FileCheck strings. The bounded
  added-diff scan found no new positive legacy/source-front-door/direct-C/
  descriptor authority.
