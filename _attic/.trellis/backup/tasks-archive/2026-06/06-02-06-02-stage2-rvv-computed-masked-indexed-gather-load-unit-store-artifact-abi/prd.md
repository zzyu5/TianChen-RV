# Stage2 RVV computed-masked indexed gather load unit-store artifact ABI boundary

## Goal

Prove and tighten one bounded Stage 2 executable path for an existing
pre-realized `computed_masked_indexed_gather_load_unit_store` selected RVV body:

```text
selected tcrv.exec RVV variant
  -> RVV plugin-owned realized/validated tcrv_rvv computed-mask indexed gather body
  -> computed-mask memory route-family provider facts
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV target artifact bundle
  -> ssh rvv correctness evidence
```

The module goal is the ABI boundary where compare-produced mask ownership and
indexed gather memory movement meet. Compare lhs/rhs, source data base, index
buffer, output/passthrough destination, runtime `n`/AVL, mask facts, index
vector/token facts, dtype/config, memory form, VL/tail policy, provider route
operand binding, and generated header/prototype facts must come from typed
`tcrv_rvv` structure and RVV plugin/provider validation. They must not be
inferred from route ids, artifact names, manifests, test names, C strings,
descriptors, exact intrinsic spellings, source-front-door residue, or common
EmitC/export logic.

## What I Already Know

* The repository started clean on `main`.
* No `.trellis/.current-task` existed; this task was created from the current
  Hermes Direction Brief.
* The two immediately previous archived tasks closed plain indexed gather and
  plain indexed scatter ABI boundaries with provider/header operand-binding
  markers, target artifact validation, focused lit/script coverage, and real
  `ssh rvv` evidence over counts `0,1,16,17,257`.
* Current code already has:
  * `tcrv_rvv.typed_computed_mask_indexed_gather_pre_realized_body`,
  * realization through `RVVComputedMaskMemorySelectedBodyRealizationOwner`,
  * `tcrv_rvv.index_load` plus `tcrv_rvv.masked_indexed_load`,
  * computed-mask memory statement-plan/provider checks,
  * explicit and pre-realized target artifact fixtures,
  * generated-bundle dry-run script coverage.
* Live inspection shows the route binding summary for this route still exposes
  header/prototype participation as `hdr-mirror` for indexed/output/runtime
  operands and does not fail closed for stale/missing computed-mask indexed
  gather binding summary markers at target artifact validation.

## Requirements

* Validate or realize the pre-realized selected body in the RVV plugin
  computed-mask memory owner before route construction.
* Preserve provider-derived facts for compare mask producer, mask role/source,
  mask memory form, inactive-lane policy, old-destination passthrough, index
  buffer role, index vector/token facts, indexed source data-base role,
  masked indexed-load behavior, unit output store role, runtime `n`/AVL,
  dtype/config, memory form, VL/tail policy, and provider route operand
  binding.
* Tighten the artifact-facing
  `computed_masked_indexed_gather_load_unit_store` route operand binding
  summary so exported runtime ABI parameters carry provider ABI marker `abi`
  and header/prototype marker `hdr`, rather than mirror-only wording.
* Add focused target artifact fail-closed validation for stale or missing
  computed-mask indexed gather binding facts.
* Keep common EmitC/export neutral; no RVV semantic inference may move into
  common materialization or target packaging.
* Strengthen focused generated-bundle dry-run evidence so the route binding
  summary, mask facts, index facts, runtime ABI order, and typed memory forms
  are explicitly visible.
* Record real `ssh rvv` correctness evidence for counts `0`, `1`, a VL-boundary
  count, a tail count, and a larger count, with at least two mask/index/data
  patterns that distinguish masked indexed gather from unmasked gather,
  contiguous load, or strided load and prove masked-off/tail sentinel
  preservation.

## Acceptance Criteria

* [x] Focused evidence shows the pre-realized body is consumed by the RVV
      computed-mask memory selected-body owner before provider route
      construction.
* [x] Focused EmitC/export or target-artifact coverage shows compare/mask
      facts, index buffer ABI, index vector/token facts, indexed data-base
      role, output store role, runtime `n`/AVL, dtype/config, memory form,
      and tail policy survive into the generated bundle/header/prototype.
* [x] Target artifact validation fails closed when the computed-mask indexed
      gather binding summary is stale, missing, or lacks required `abi`/`hdr`
      markers for exported runtime ABI parameters.
* [x] Focused script dry-run coverage checks the exact provider binding
      summary and computed-mask indexed gather boundary fields.
* [x] Real `ssh rvv` compile/run evidence passes counts `0`, `1`, `16`, `17`,
      and `257` with at least two non-vacuous mask/index/data patterns and
      verifies inactive-lane plus tail sentinel preservation.
* [x] Bounded old-authority scan over touched plugin/provider/materializer/
      target/script/test/spec files classifies remaining hits for:
      `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
      `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
      `emission_plan`, `descriptor`, and `selected route`.
* [x] Smallest relevant build/test/script commands pass, with self-repair for
      failures.
* [x] The task is finished/archived and one coherent commit is created if the
      module behavior is complete.

## Out Of Scope

* Computed-masked indexed scatter.
* Segment2 memory.
* More plain indexed gather/scatter rework.
* Strided memory rework beyond direct dependencies.
* Contraction follow-up.
* Dtype/LMUL clone batches.
* High-level frontend authority or per-Linalg lowering.
* Source-front-door positive routes.
* Common EmitC semantic inference.
* Dashboards, broad smoke matrices, or performance claims.

## Technical Notes

Relevant specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`

Closest archived patterns:

* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-indexed-gather-unit-store-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-indexed-scatter-unit-load-artifact-abi/prd.md`

Initial implementation focus:

* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* `test/Target/RVV/*computed-masked-indexed-gather-load.mlir`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Scripts/*computed-masked-indexed-gather-load-dry-run.test`

## Completed Behavior

* `computed_masked_indexed_gather_load_unit_store` route operand binding
  summaries now normalize provider ABI participation to `abi` and
  header/prototype participation to `hdr` for all exported runtime operands:
  `cmp_lhs`, `cmp_rhs`, `src`, `index`, `dst`, and `n`.
* The computed-mask indexed gather route plan now records header/prototype
  participation for compare lhs/rhs and source data-base operands in addition
  to index/output/runtime operands.
* RVV target artifact validation now fails closed unless this route carries the
  expected provider plan id, starts the summary with that provider plan id, has
  runtime ABI order `cmp_lhs,cmp_rhs,src,index,dst,n`, and gives every exported
  operand matching role/C-name plus `abi` and `hdr` markers.
* Target artifact tests cover stale provider plan id, stale missing `hdr` in
  the route description, and stale candidate mirror metadata before export.
* Generated bundle dry-run tests now check the exact route binding summary and
  the generated C harness records two mask/index/data patterns.
* The script harness executes each runtime count for `patterns[] = {0, 1}` and
  varies compare inputs, source data values, output sentinels, and index
  permutation formulas while checking active lanes, inactive-lane preservation,
  noncontiguous indexed gather, source preservation, and tail preservation.
* A focused direct pre-realized route-entry script test proves the retired
  shortcut still fails closed before bundle generation.
* `.trellis/spec/testing/mlir-testing-contract.md` now records the dry-run
  HARNESS FileCheck gotcha discovered during self-repair: generated-bundle
  dry-run checks inspect generated C source, while concrete runtime lines
  belong in remote output or evidence JSON checks.

## Validation Evidence

Commands run:

* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `build/bin/tianchenrv-target-artifact-export-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter computed-masked-indexed-gather-load`
* explicit dry-run with counts `0,1,16,17,257`
* pre-realized dry-run with counts `0,1,16,17,257`
* real `ssh rvv` run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-indexed-gather-load-unit-store-artifact-abi/ssh-rvv-pre-realized --run-id pre-realized-computed-mask-indexed-gather-load --overwrite --op-kind computed_masked_indexed_gather_load_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20`

Remote RVV result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=computed_masked_indexed_gather_load_unit_store counts=0,1,16,17,257 patterns=0,1
```

The remote output includes per-count/per-pattern success lines. For `n=16`,
`n=17`, and `n=257`, both patterns report active and inactive mask lanes,
inactive-preserved lane counts, noncontiguous indexed gather lanes,
`source_preserved`, and `tail_preserved`.

Self-repair performed:

* The first ad-hoc dry-run used `llvm-readobj`, which was not in PATH; reran
  with `llvm-readobj-20`.
* The first focused lit rerun failed because the HARNESS checks expected
  runtime output strings while dry-run inspects generated C source. The checks
  were corrected to source-level facts (`patterns[] = {0, 1}` and
  `pattern=%zu`), then focused lit passed.

Old-authority scan:

* Diff-only scan over the touched code/test/script files found no newly added
  hits for `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, `descriptor`, or `selected route`.
* Full touched-file scan still reports pre-existing legacy/negative-test hits
  in broad target artifact tests and script tables, plus intentional
  `implicit-check-not` and PRD wording. No new computed-mask indexed gather
  route authority was introduced from those strings.
