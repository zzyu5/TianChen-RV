# Stage2 RVV computed-masked segment2 store artifact ABI boundary

## Goal

Make the bounded Stage 2 RVV
`computed_masked_segment2_store_unit_load` compiler path a tested artifact ABI
boundary:

```text
selected tcrv.exec RVV variant
  -> typed_computed_mask_segment2_store_pre_realized_body
  -> RVV plugin-local realization into setvl / with_vl / compare loads /
     field0 and field1 unit-stride payload loads / compare /
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

`Stage2 RVV computed-masked segment2 store artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` was clean.
* Initial `git log --oneline -8` started at
  `ceb90af2 chore: record journal`, then
  `de423987 rvv: validate computed masked segment2 load artifact facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits and then started.
* The archived segment2-load task under
  `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-segment2-load-artifact-abi-boundary/`
  provides the closest adjacent route pattern: tighten provider/target
  fail-closed checks, generated-bundle dry-run facts, real `ssh rvv` evidence,
  finish/archive, then commit.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` agree that the active path
  must be selected typed RVV body -> RVV plugin-owned realization/provider ->
  common EmitC -> target artifact -> `ssh rvv` evidence for correctness
  claims.

## Current Repository Evidence

Live inspection found:

* `TypedComputedMaskSegment2StorePreRealizedBodyOp` in
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`, binding runtime ABI operands
  `cmp_lhs, cmp_rhs, src0, src1, dst, n`, compare predicate, segment count,
  source0/source1 unit-stride memory forms, interleaved destination memory
  form, field0/field1 input roles, compare-produced mask role/source/form,
  inactive-lane no-write policy, SEW/LMUL, and policy.
* `MaskedSegment2StoreOp` carries the interleaved destination, compare-produced
  mask, field0/field1 payload vectors, VL, `segment_count = 2`, destination
  memory form, field roles, and inactive-lane no-write policy.
* `RVVSelectedBodyEmitCRouteDescription` already exposes provider-side fields
  for runtime ABI order, binding summary, provider support mirror,
  target leaf/profile, header/type summaries, computed-mask plan, segment
  memory layout, segment count, tuple/store facts, field roles/names,
  source/destination memory forms, and field source/destination forms.
* `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp` already has
  a computed-mask segment2 store planning owner and realization path. The
  pre-realized store body realizes into compare lhs/rhs loads, field0/field1
  loads, compare, optional update add for the update route only, and
  `tcrv_rvv.masked_segment2_store`.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` already validates
  computed-mask segment2 store provider facts and statement sequence, including
  the store-specific runtime ABI order, binding summary, provider mirror, C
  type mapping, memory layout, mask facts, tuple create, masked segment store,
  field forms, and stale mirror rejection.
* `test/Target/TargetArtifactExportTest.cpp` currently has only a small number
  of store-specific negative cases: wrong destination memory form,
  metadata-only provider mirror, stale header markers in the binding summary,
  and stale candidate binding mirrors. Many required store facts are currently
  covered indirectly by the computed-mask segment2 update fixture or generic
  segment2 paths rather than the store fixture itself.
* The pre-realized store MLIR fixture already checks realization into
  `masked_segment2_store`, runtime ABI order, binding summary, computed-mask
  plan, target profile, provider mirror, headers, C type mapping, mask facts,
  segment count, segment store intrinsic, and header prototype.
* The existing generated-bundle dry-run test already uses counts
  `0,1,7,16,23,257` and checks core route/binding, segment layout, mask
  source, segment store intrinsic, provider mirror, inactive-lane contract, and
  harness behavior, but does not yet assert every provider-derived fact named
  in the task brief.

## Gap This Round Closes

This round closes the store-specific validation/evidence gap around the
existing production path:

* Add computed-mask segment2 store-specific target artifact C++ fail-closed
  regressions for stale or missing provider/mirror facts named by the brief,
  rather than relying on update/load/plain segment2 fixtures to prove them.
* Tighten generated-bundle dry-run FileCheck coverage for provider-derived
  computed-mask segment2 store facts.
* Run real generated-bundle `ssh rvv` correctness for counts
  `0,1,7,16,23,257` and patterns `0,1`, proving active-lane interleaved
  segment stores, inactive-lane no-write preservation of old destination
  fields, source/tail sentinel preservation, and runtime `n`/AVL behavior.

## Requirements

* Keep route support rooted in the selected typed/pre-realized `tcrv_rvv` body,
  RVV plugin-local realization, segment2 route-family planning owner, segment2
  memory statement-plan boundary, and provider-built route facts.
* Preserve runtime ABI order `cmp_lhs,cmp_rhs,src0,src1,dst,n`.
* Preserve the binding summary plan
  `rvv-route-operand-binding:computed_masked_segment2_store_unit_load.v1` and
  exported runtime ABI entries with `abi` and `hdr`.
* Preserve compare predicate `slt`, compare-produced mask role/source/form,
  `segment_count = 2`, field0/field1 roles as input buffers, source0/source1
  unit-stride loads, masked segment2 interleaved destination store, false-lane
  no-write preservation of destination, SEW `32`, LMUL `m1`, and agnostic
  tail/mask policy.
* Target artifact validation must fail closed when provider facts or candidate
  mirrors are stale/missing for this exact computed-mask segment2 store route.
* Generated-bundle dry-run evidence must expose provider-derived route-family,
  target, header/type, mask, segment, source/destination memory, field role,
  tuple/store, no-write, and provider support mirror facts.
* Runtime correctness evidence must use real `ssh rvv` output for counts
  `0,1,7,16,23,257` and patterns `0,1`.

## Acceptance Criteria

* [x] Focused production/test diff strengthens the segment2 store artifact ABI
      boundary; no metadata-only archive closeout.
* [x] Target artifact C++ tests prove stale computed-mask segment2 store
      provider mirror, target leaf/profile, header declarations, C type
      mapping, computed-mask route-family plan, mask source/form/role,
      compare predicate, segment count, segment memory layout,
      source0/source1 unit-stride forms, interleaved destination memory form,
      field roles/names, inactive-lane no-write contract, binding summary,
      tuple create/store mirror facts, and route statement operands fail
      closed.
* [x] Existing or tightened FileCheck coverage proves the selected body
      realizes into compare-produced mask, field0/field1 unit-stride loads,
      tuple creation, and masked interleaved segment2 store, not
      descriptor/source-front-door/legacy i32 paths.
* [x] Generated-bundle dry-run checks record provider-derived store facts:
      runtime ABI order, binding plan/summary, computed-mask route-family
      plan, `provider_supported_mirror`, `target_leaf_profile`, required
      headers, C type mapping, mask role/source/form, segment count, segment
      memory layout, source/destination memory forms, field roles/names,
      field source forms, empty field destination forms, tuple/store mirrors,
      and inactive-lane no-write contract.
* [x] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,7,16,23,257` and patterns `0,1`, proving active lanes store
      `src0[i]`/`src1[i]` into `dst[2*i]`/`dst[2*i+1]`, inactive lanes
      preserve old interleaved destination fields, source and tail sentinels
      are preserved, and runtime `n`/AVL is honored.
* [x] Focused checks, `git diff --check`, and bounded old-authority scan pass.
* [ ] Task is finished/archived and one coherent commit is created if the task
      completes.

## Out Of Scope

* Segment2 load redo, segment2 update expansion, or a broad segment matrix.
* Reworking indexed-memory gather/scatter paths, dtype/LMUL clone batches,
  frontend/source-front-door positive routes, high-level Linalg lowering,
  dashboards, global tuning databases, descriptor routes, direct-C/source
  exporters, or common EmitC RVV semantic inference.
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
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-segment2-load-artifact-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-segment2-load-artifact-abi-boundary/implement.jsonl`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-segment2-load-artifact-abi-boundary/check.jsonl`
* `.trellis/workspace/codex/journal-20.md`

Relevant files inspected:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-store.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-store-dry-run.test`
* `test/Target/TargetArtifactExportTest.cpp`

## Implementation Notes

This round tightened the production computed-mask segment2 store artifact ABI
boundary in target validation and tests:

* `validateComputedMaskSegment2LoadHeaderBindingSummary` was generalized to
  `validateComputedMaskSegment2HeaderBindingSummary`, so load, store, and
  update computed-mask segment2 routes all structurally validate the
  provider-owned route operand binding plan, logical ABI operands, and
  `abi|hdr` prototype/header participation markers before target artifact
  export.
* The computed-mask segment2 store/update target provider fact validator now
  rejects a stale provider-derived segment-load callee fact. Store/update
  routes may carry tuple-create and masked segment-store facts, but a
  `segmentLoadIntrinsic` mirror belongs to segment2 load routes and must not
  repair or authorize a store path.
* The target artifact C++ regression matrix now has store-specific provider,
  route, and candidate-mirror failures for target leaf profile, provider
  mirror, header/type summaries, computed-mask route-family plan, stale plain
  segment2 plan, mask role/source/form, compare predicate, segment count,
  segment memory layout, source/destination memory forms, field roles/names,
  field source forms, inactive-lane no-write contract, passthrough layout,
  stale segment-load fact, segment-store callee, tuple field order, masked
  store pointer, source provenance, tuple-create mirror, segment-store mirror,
  and stale segment-load mirror.
* The pre-realized generated-bundle dry-run FileCheck now asserts the full
  provider-derived store fact surface named by the task brief, not only route
  id, binding, segment layout, mask source, segment store intrinsic, and
  provider mirror.
* `.trellis/spec/extension-plugins/rvv-plugin.md` now records the executable
  target-export contract that computed-mask segment2 store/update must reject
  stale segment-load or field-extract facts before artifact export.

Self-repair performed during validation:

* The new store negative test first revealed a production gap: store/update
  provider facts did not reject a stale `segmentLoadIntrinsic`. The validator
  was tightened and the test now passes.
* Generalizing the computed-mask segment2 binding-summary helper changed two
  older update-failure diagnostics to fail earlier at provider-plan/start-with
  checks. The C++ expected error fragments were updated to match the new
  fail-closed order.

## Validation Evidence

Commands run:

```bash
rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2
rtk ./build/bin/tianchenrv-target-artifact-export-test
rtk cmake --build build --target tcrv-opt tcrv-translate -j2
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter computed-masked-segment2-store  # from build/test
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-segment2-store-artifact-abi-boundary/dry-run --run-id pre-realized-computed-mask-segment2-store --overwrite --op-kind computed_masked_segment2_store_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-segment2-store-artifact-abi-boundary/ssh-rvv --run-id pre-realized-computed-mask-segment2-store --overwrite --op-kind computed_masked_segment2_store_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20
rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test
rtk git diff --check
rtk bash -lc 'git diff -U0 -- .trellis/spec/extension-plugins/rvv-plugin.md lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-store-dry-run.test | rg -n "^\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|descriptor-driven|source-front-door|direct-C|source-export)" || true'
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
artifact_dir: artifacts/tmp/06-02-stage2-rvv-computed-masked-segment2-store-artifact-abi-boundary/dry-run/pre-realized-computed-mask-segment2-store
```

Real `ssh rvv` result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=computed_masked_segment2_store_unit_load counts=0,1,7,16,23,257 patterns=0,1
```

The remote output includes per-count/per-pattern success lines. Counts `7`,
`16`, `23`, and `257` include mixed active and inactive lanes, inactive
interleaved destination preservation, distinguishable field0/field1 payload
stores, source preservation, tail preservation, and runtime `n`/AVL coverage.

The diff-only old-authority scan produced no matches.

## Current Phase

Finish.
