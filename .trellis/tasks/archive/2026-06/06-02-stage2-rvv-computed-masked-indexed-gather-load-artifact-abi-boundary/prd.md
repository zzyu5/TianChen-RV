# Stage2 RVV computed-masked indexed gather-load artifact ABI boundary

## Goal

Resolve the Hermes direction brief for
`computed_masked_indexed_gather_load_unit_store` against the current repository
state. The intended module behavior is one bounded Stage 2 RVV memory-movement
path:

```text
selected tcrv.exec RVV variant
  -> typed_computed_mask_indexed_gather_pre_realized_body
  -> RVV plugin-local realization into setvl / with_vl / loads / index_load /
     compare / masked_indexed_load / store
  -> computed-mask memory provider facts
  -> TCRVEmitCLowerableRoute
  -> target artifact validation
  -> generated bundle/header/object
  -> ssh rvv correctness evidence
```

Live repository evidence shows this direction brief is stale/duplicate: the
same production owner was already completed on the current `main` history by
`47777bab rvv: prove computed masked indexed gather artifact abi` and archived
under
`.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-indexed-gather-load-unit-store-artifact-abi/`.
This round therefore verifies the current behavior and records a truthful
Trellis state instead of reworking an already-complete production path.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-masked indexed gather-load artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` was clean.
* Initial `git log --oneline -8` started at
  `1b2183b7 rvv: canonicalize computed masked strided dot facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before any file edits.
* Current `main` contains prior computed-masked indexed gather commits:
  `3610b2d2 rvv: add closure-gated masked indexed gather load` and
  `47777bab rvv: prove computed masked indexed gather artifact abi`.
* The archived duplicate task records completed production changes for provider
  binding normalization, target artifact fail-closed validation,
  generated-bundle dry-run checks, direct pre-realized fail-closed coverage,
  and real `ssh rvv` evidence.

## Current Repository Evidence

Live code inspection confirmed the current production path contains:

* `tcrv_rvv.typed_computed_mask_indexed_gather_pre_realized_body` with runtime
  ABI order `cmp_lhs, cmp_rhs, src, index, dst, n`.
* RVV plugin-local realization and validation around
  `setvl`, `with_vl`, compare input loads, old-destination passthrough load,
  `index_load`, `compare`, `masked_indexed_load`, and unit-stride destination
  `store`.
* Structural facts for `predicate_kind = slt`,
  `mask_role = predicate-mask-produced-by-compare`,
  `mask_source = compare-produced-mask-same-vl-scope`,
  `mask_memory_form = compare-produced-mask`,
  `index_eew = 32`, `offset_unit = element`, and inactive-lane passthrough
  policy.
* Provider route operand binding plan
  `rvv-route-operand-binding:computed_masked_indexed_gather_load_unit_store.v1`
  with all exported runtime ABI operands carrying `abi` and `hdr`:
  `cmp_lhs`, `cmp_rhs`, `src`, `index`, `dst`, and `n`.
* Target artifact validation for stale/missing binding summaries and stale
  candidate mirror metadata.
* Generated-bundle evidence fields for provider/target route facts, indexed
  memory layout, masked memory layout, C type mapping, required headers,
  provider-supported mirror, old-destination passthrough, index facts, and
  route-family facts.
* Focused direct pre-realized route-entry fail-closed coverage.

## Requirements

* Do not add a second implementation beside the existing production path.
* Verify that provider facts and target validation currently consume the
  plugin-owned computed-mask indexed gather fact surface.
* Verify that the generated-bundle route binding summary uses provider `abi`
  and exported header/prototype `hdr` markers for runtime ABI operands.
* Verify that real `ssh rvv` generated-bundle correctness covers counts
  `0,1,7,16,23` plus one larger tail-crossing count, with two index/mask
  patterns proving active indexed loads, inactive-lane preservation,
  noncontiguous/permuted index use, source preservation, unit-stride
  destination store, tail preservation, and runtime `n`/AVL behavior.
* Keep common EmitC/export neutral and do not move RVV mask/index/passthrough
  semantics into common code.

## Acceptance Criteria

* [x] Current production code contains the RVV plugin/provider computed-mask
      indexed gather selected-body realization, provider plan, and target
      artifact validation path.
* [x] Focused lit coverage proves the pre-realized selected body realizes into
      compare-produced mask, index load, masked indexed source load,
      old-destination passthrough, and unit-stride destination store facts.
* [x] Target artifact regression tests pass, including stale or missing route
      operand binding validation for the computed-mask indexed gather route.
* [x] Generated-bundle dry-run with counts `0,1,7,16,23,257` succeeds and
      records provider-derived binding, mask, index, passthrough, header/type,
      and route-family facts.
* [x] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,7,16,23,257` and patterns `0,1`, including active/inactive lanes,
      inactive preserved lanes, noncontiguous indexed gather lanes, source
      preservation, tail preservation, and runtime `n`/AVL.
* [x] No production code change is needed in this round because the requested
      behavior already exists on current `main`.

## Out Of Scope

* Computed-masked indexed scatter.
* Segment2 memory.
* Broad memory matrices, scatter clones, strided-load clones, LMUL/dtype clone
  batches, high-level Linalg lowering, source-front-door positive routes,
  dashboards, report/index work, or harness-only code changes.
* Moving RVV semantic authority into common EmitC/export, route ids, artifact
  names, manifests, descriptors, exact intrinsic spellings, or mirror metadata.

## Technical Notes

Specs and prior task context read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-strided-widening-dot-reduction-artifact-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-indexed-gather-unit-store-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-indexed-gather-load-unit-store-artifact-abi/prd.md`

Production/test files inspected:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-indexed-gather-load.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-indexed-gather-load-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-indexed-gather-load-fail-closed.test`
* `test/Target/TargetArtifactExportTest.cpp`

## Validation Evidence

Commands run in this round:

```bash
rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test
rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2
rtk build/bin/tianchenrv-target-artifact-export-test
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter computed-masked-indexed-gather-load
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-indexed-gather-load-artifact-abi-boundary/dry-run --run-id pre-realized-computed-mask-indexed-gather-load --overwrite --op-kind computed_masked_indexed_gather_load_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-indexed-gather-load-artifact-abi-boundary/ssh-rvv --run-id pre-realized-computed-mask-indexed-gather-load --overwrite --op-kind computed_masked_indexed_gather_load_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20
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
artifact_dir: artifacts/tmp/06-02-stage2-rvv-computed-masked-indexed-gather-load-artifact-abi-boundary/dry-run/pre-realized-computed-mask-indexed-gather-load
```

Real `ssh rvv` result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=computed_masked_indexed_gather_load_unit_store counts=0,1,7,16,23,257 patterns=0,1
```

The remote output includes per-count/per-pattern success lines. Counts `7`,
`16`, `23`, and `257` include both active and inactive mask lanes, inactive
preserved lanes, noncontiguous indexed gather lanes, source preservation, and
tail preservation.

## Current Phase

Finish/archive. The implementation phase is intentionally a no-op because the
requested production compiler path already exists and passed focused local plus
real RVV validation in this round.
