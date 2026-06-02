# Stage2 RVV computed-masked segment2 load artifact ABI boundary

## Goal

Make the bounded Stage 2 RVV
`computed_masked_segment2_load_unit_store` compiler path a tested artifact ABI
boundary:

```text
selected tcrv.exec RVV variant
  -> typed_computed_mask_segment2_load_pre_realized_body
  -> RVV plugin-local realization into setvl / with_vl / load / load / old
     field passthrough loads / compare / masked_segment2_load / store / store
  -> provider-built TCRVEmitCLowerableRoute
  -> target artifact validation
  -> generated bundle/header/object
  -> ssh rvv correctness evidence
```

This task is not a broad segment2 matrix or a segment2 store/update expansion.
The route authority must remain the typed/realized `tcrv_rvv` body and RVV
provider facts, not route ids, artifact names, metadata mirrors, exact
intrinsic spellings, descriptors, source-front-door markers, or script-only
evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-masked segment2 load artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` was clean.
* Initial `git log --oneline -8` started at
  `9e14620f rvv: validate computed masked indexed scatter artifact facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* The archived indexed scatter-store task under
  `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-indexed-scatter-store-artifact-abi-boundary/`
  provides the closest pattern: tighten provider/target fail-closed checks,
  generated-bundle dry-run facts, real `ssh rvv` evidence, then finish/archive
  and commit.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` agree that the active path
  must be selected typed RVV body -> RVV plugin-owned realization/provider ->
  common EmitC -> target artifact -> `ssh rvv` evidence for correctness claims.

## Current Repository Evidence

Live inspection found:

* `TypedComputedMaskSegment2LoadPreRealizedBodyOp` in
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`, binding runtime ABI operands
  `cmp_lhs, cmp_rhs, src, out0, out1, n`, compare predicate, segment count,
  source/destination memory forms, field0/field1 output roles, mask
  role/source/form, inactive-lane preservation, SEW/LMUL, and policy.
* `MaskedSegment2LoadOp` carries the interleaved source, compare-produced mask,
  old field0/field1 passthrough values, VL, `segment_count = 2`, source memory
  form, field roles, and inactive-lane preservation policy.
* `RVVSelectedBodyEmitCRouteDescription` already has provider-side fields for
  runtime ABI order, binding summary, provider mirror, target leaf/profile,
  header/type summaries, segment memory layout, segment count, segment tuple
  and field extraction facts, field roles/names, source/destination memory
  forms, and field source/destination forms.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` already validates
  the computed-mask segment2 load provider facts and statement sequence, but
  C++ regressions currently cover only part of the load-specific fail-closed
  surface. Several computed-mask segment2 negative cases are covered through
  the update fixture rather than the load fixture.
* The existing pre-realized segment2-load dry-run test proves the selected body
  is consumed and checks binding/source/segment load facts, but does not yet
  assert all provider-derived facts named in this task brief, including
  provider mirror, target leaf/profile, headers/C type mapping, segment count,
  destination/field destination forms, mask role/form/producer, and computed
  mask route-family facts.
* The dry-run currently uses counts `0,1,16,17,257`; this task requires
  `0,1,7,16,23,257`.

## Gap This Round Closes

This round closes a focused validation/evidence gap around the existing
production path:

* Add load-specific target artifact C++ fail-closed regressions for stale or
  missing provider/mirror facts named by the brief, rather than relying on the
  store/update segment2 fixtures to prove them.
* Tighten generated-bundle dry-run FileCheck coverage for provider-derived
  computed-mask segment2 load facts.
* Run real generated-bundle `ssh rvv` correctness for counts
  `0,1,7,16,23,257` and patterns `0,1`, proving active interleaved loads,
  inactive old-field passthrough preservation, source/tail preservation, dual
  unit-stride output stores, and runtime `n`/AVL behavior.

## Requirements

* Keep route support rooted in the selected typed/pre-realized `tcrv_rvv` body,
  RVV plugin-local realization, and provider-built route facts.
* Preserve runtime ABI order `cmp_lhs,cmp_rhs,src,out0,out1,n`.
* Preserve the binding summary plan
  `rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1` and
  exported runtime ABI entries with `abi` and `hdr`.
* Preserve compare predicate `slt`, compare-produced mask role/source/form,
  `segment_count = 2`, segment2 interleaved unit-stride source load, old
  field0/field1 passthrough, inactive-lane preservation, dual unit-stride
  output stores, SEW `32`, LMUL `m1`, and agnostic tail/mask policy.
* Target artifact validation must fail closed when provider facts or candidate
  mirrors are stale/missing for this exact computed-mask segment2 load route.
* Generated-bundle dry-run evidence must expose provider-derived route-family,
  target, header/type, mask, segment, source/destination memory, field role,
  passthrough, and provider support mirror facts.
* Runtime correctness evidence must use real `ssh rvv` output for counts
  `0,1,7,16,23,257` and patterns `0,1`.

## Acceptance Criteria

* [x] Focused production/test diff strengthens the segment2 load artifact ABI
      boundary; no metadata-only archive closeout.
* [x] Target artifact C++ tests prove stale computed-mask segment2 load
      provider mirror, target leaf/profile, header declarations, C type
      mapping, computed-mask route-family plan, mask source/form/role, segment
      count, segment memory layout, source/destination memory form, field
      roles/names, field destination memory form, passthrough/inactive-lane
      contract, binding summary, and route statements fail closed.
* [x] Existing or tightened FileCheck coverage proves the selected body
      realizes into compare-produced mask, interleaved masked segment2 load,
      old field0/field1 passthrough, field extraction, and dual output stores,
      not descriptor/source-front-door/legacy i32 paths.
* [x] Generated-bundle dry-run checks record provider-derived load facts:
      runtime ABI order, binding plan/summary, computed-mask route-family plan,
      `provider_supported_mirror`, `target_leaf_profile`, required headers,
      C type mapping, mask role/source/form, segment count, segment memory
      layout, source/destination memory forms, field roles/names, field
      destination memory forms, segment tuple/field extraction mirrors, and
      inactive-lane passthrough contract.
* [x] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,7,16,23,257` and patterns `0,1`, proving active lanes load
      `src[2*i]`/`src[2*i+1]` into `out0`/`out1`, inactive lanes preserve old
      field outputs, source and tail sentinels are preserved, dual output
      stores are correct, and runtime `n`/AVL is honored.
* [x] Focused checks, `git diff --check`, and bounded old-authority scan pass.
* [x] Task is finished/archived and one coherent commit is created if the task
      completes.

## Out Of Scope

* Segment2 store/update expansion in this task.
* Reworking the already-archived indexed-memory gather/scatter paths.
* Broad segment matrices, dtype/LMUL clone batches, frontend/source-front-door
  positive routes, high-level Linalg lowering, dashboards, global tuning
  databases, descriptor routes, direct-C/source exporters, or common EmitC RVV
  semantic inference.
* Treating route ids, artifact names, exact intrinsic spelling, test names,
  metadata mirrors, harness sentinels, or generated script expectations as
  segment count, mask, field, passthrough, dtype/config, runtime ABI, policy,
  or route authority.

## Technical Notes

Specs and task context read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-indexed-scatter-store-artifact-abi-boundary/prd.md`

Relevant files inspected:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-load.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-load-dry-run.test`
* `test/Target/TargetArtifactExportTest.cpp`

## Implementation Notes

This round tightened the production segment2 load artifact mirror boundary:
the load route's passthrough tuple materializer is now emitted as
`tcrv_rvv.segment_tuple_create_intrinsic`, while
`tcrv_rvv.segment_store_intrinsic` is not emitted for this load route. The
target artifact validator now requires that split and rejects a stale
segment-store mirror on computed-mask segment2 load candidates. This keeps
tuple materialization and segment-store facts separate at the artifact ABI
boundary.

The target artifact C++ regression matrix now uses the computed-mask segment2
load fixture directly for stale provider mirror, target leaf/profile, required
headers, C type mapping, computed-mask route-family plan, stale plain segment2
route-family facts, mask source/role/form, segment count, segment layout,
destination memory form, inactive-lane/passthrough contract, field destination
form, binding summary, candidate mirrors, tuple C type, tuple create, stale
segment-store mirror, field names, route statement operands, old-field
passthrough tuple, field extraction, field-store VL, and selected typed RVV
source provenance.

The generated-bundle dry-run checks now assert the provider-derived segment2
load facts named in the brief, and the pre-realized dry-run/runtime counts were
changed to `0,1,7,16,23,257`.

Self-repair performed during validation:

* Adjusted C++ error-fragment expectations to match the actual provider loop
  names `offset` and `vl`.
* After inspection, changed the production mirror from the old
  `segment_store_intrinsic = __riscv_vcreate_v_i32m1x2` load metadata to the
  corrected `segment_tuple_create_intrinsic` mirror, then updated fixtures,
  dry-run tests, script expectations, and target validator checks.

## Validation Evidence

Commands run:

```bash
rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2
rtk build/bin/tianchenrv-target-artifact-export-test
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter computed-masked-segment2-load  # from build/test
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-segment2-load-artifact-abi-boundary/dry-run --run-id pre-realized-computed-mask-segment2-load --overwrite --op-kind computed_masked_segment2_load_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-segment2-load-artifact-abi-boundary/ssh-rvv --run-id pre-realized-computed-mask-segment2-load --overwrite --op-kind computed_masked_segment2_load_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20
rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test
rtk git diff --check
rtk bash -lc 'git diff -U0 -- lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp scripts/rvv_generated_bundle_abi_e2e.py test/Target/TargetArtifactExportTest.cpp test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-load.mlir test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-load.mlir test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-segment2-load-dry-run.test test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-load-dry-run.test | rg -n "^\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|descriptor-driven|source-front-door|direct-C|source-export)" || true'
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
artifact_dir: artifacts/tmp/06-02-stage2-rvv-computed-masked-segment2-load-artifact-abi-boundary/dry-run/pre-realized-computed-mask-segment2-load
```

Real `ssh rvv` result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=computed_masked_segment2_load_unit_store counts=0,1,7,16,23,257 patterns=0,1
```

The remote output includes per-count/per-pattern success lines. Counts `7`,
`16`, `23`, and `257` include mixed active and inactive lanes, inactive old
field passthrough preservation, distinguishable even/odd interleaved source
fields, source preservation, tail preservation, and runtime `n`/AVL coverage.

The diff-only old-authority scan produced no matches.

## Current Phase

Finish/archive.
