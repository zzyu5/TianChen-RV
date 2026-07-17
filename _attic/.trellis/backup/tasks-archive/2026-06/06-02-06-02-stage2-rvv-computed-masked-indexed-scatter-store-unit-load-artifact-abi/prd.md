# Stage2 RVV computed-masked indexed scatter store unit-load artifact ABI boundary

## Goal

Prove and tighten one bounded Stage 2 executable path for an existing
pre-realized `computed_masked_indexed_scatter_store_unit_load` selected RVV
body:

```text
selected tcrv.exec RVV variant
  -> RVV plugin-owned realized/validated tcrv_rvv computed-mask indexed scatter body
  -> computed-mask memory route-family provider facts
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV target artifact bundle
  -> ssh rvv correctness evidence
```

The module goal is the ABI boundary where compare-produced mask ownership,
unit-stride source load, index-vector facts, and masked indexed destination
scatter meet. Compare lhs/rhs, source unit-load base, index buffer, index
vector/token facts, destination indexed-store role, runtime `n`/AVL, mask
facts, dtype/config, memory form, VL/tail policy, provider route operand
binding, and generated header/prototype facts must come from typed `tcrv_rvv`
structure and RVV plugin/provider validation. They must not be inferred from
route ids, artifact names, manifests, test names, C strings, descriptors,
exact intrinsic spellings, source-front-door residue, or common EmitC/export
logic.

## What I Already Know

* The repository started clean on `main`.
* No `.trellis/.current-task` existed; this task was created from the current
  Hermes Direction Brief.
* The immediately previous archived tasks closed plain indexed scatter and
  computed-masked indexed gather ABI boundaries with provider/header
  operand-binding markers, target artifact validation, focused lit/script
  coverage, and real `ssh rvv` evidence over counts `0,1,16,17,257`.
* Current code already has:
  * `tcrv_rvv.typed_computed_mask_indexed_scatter_pre_realized_body`,
  * realization through `RVVComputedMaskMemorySelectedBodyRealizationOwner`,
  * explicit and pre-realized target artifact fixtures,
  * computed-mask memory statement-plan/provider support,
  * `tcrv_rvv.index_load` plus `tcrv_rvv.masked_indexed_store`,
  * generated-bundle expectation entries for
    `computed_masked_indexed_scatter_store_unit_load`.
* Live inspection shows the scatter route binding summary still exposes
  header/prototype participation as `hdr-mirror` for index/destination/runtime
  operands and omits header/prototype markers for compare/source operands,
  while the last gather task tightened the analogous summary to provider ABI
  marker `abi` plus header/prototype marker `hdr`.
* Live inspection shows target artifact route-family validation has a focused
  `validateComputedMaskIndexedGatherHeaderBindingSummary`, but no matching
  computed-mask indexed scatter header-binding summary validator yet.
* No focused `test/Scripts/rvv-generated-bundle-abi-e2e-*computed-masked-indexed-scatter*`
  dry-run/fail-closed script tests currently exist.

## Requirements

* Validate or realize the pre-realized selected body in the RVV plugin
  computed-mask memory owner before provider route construction.
* Preserve provider-derived facts for compare mask producer, mask role/source,
  mask memory form, inactive-lane policy, source unit-load role, index buffer
  role, index vector/token facts, indexed destination-store role, runtime
  `n`/AVL, dtype/config, memory form, VL/tail policy, and provider route
  operand binding.
* Tighten the artifact-facing
  `computed_masked_indexed_scatter_store_unit_load` route operand binding
  summary so all exported runtime ABI parameters carry provider ABI marker
  `abi` and header/prototype marker `hdr`.
* Add focused target artifact fail-closed validation for stale or missing
  computed-mask indexed scatter binding facts.
* Keep common EmitC/export neutral; no RVV semantic inference may move into
  common materialization or target packaging.
* Strengthen focused generated-bundle dry-run evidence so the route binding
  summary, mask facts, index facts, runtime ABI order, source unit-load role,
  destination masked-indexed-store role, and typed memory forms are explicitly
  visible.
* Add direct pre-realized route-entry negative script coverage for
  `computed_masked_indexed_scatter_store_unit_load` if it is not already
  covered.
* Record real `ssh rvv` correctness evidence for counts `0`, `1`, a
  VL-boundary count, a tail count, and a larger count, with at least two
  mask/index/source patterns that distinguish masked indexed scatter from
  unmasked scatter, contiguous store, or strided store and prove masked-off
  destination, source, and tail sentinel preservation.

## Acceptance Criteria

* [x] Focused evidence shows the pre-realized body is consumed by the RVV
      computed-mask memory selected-body owner before provider route
      construction.
* [x] Focused EmitC/export or target-artifact coverage shows compare/mask
      facts, source unit-load ABI, index buffer ABI, index vector/token facts,
      indexed destination-store role, runtime `n`/AVL, dtype/config, memory
      form, and tail policy survive into the generated bundle/header/prototype.
* [x] Target artifact validation fails closed when the computed-mask indexed
      scatter binding summary is stale, missing, or lacks required `abi`/`hdr`
      markers for exported runtime ABI parameters.
* [x] Focused script dry-run coverage checks the exact provider binding
      summary and computed-mask indexed scatter boundary fields.
* [x] Real `ssh rvv` compile/run evidence passes counts `0`, `1`, `16`, `17`,
      and `257` with at least two non-vacuous mask/index/source patterns and
      verifies masked-off destination, source, and tail sentinel preservation.
* [x] Bounded old-authority scan over touched plugin/provider/materializer/
      target/script/test/spec files classifies remaining hits for:
      `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
      `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
      `emission_plan`, `descriptor`, and `selected route`.
* [x] Smallest relevant build/test/script commands pass, with self-repair for
      failures.
* [x] The task is finished/archived and one coherent commit is created if the
      module behavior is complete.

## Technical Approach

Use the just-completed computed-masked indexed gather ABI boundary and the
plain indexed scatter ABI boundary as the closest production patterns, but
apply them only to computed-mask indexed scatter:

* tighten `RVVEmitCRoutePlanning.cpp` scatter operand-binding summary tokens
  from mirror-only `hdr-mirror` wording to exported ABI/header markers;
* add a target-owned computed-mask indexed scatter binding-summary validator
  adjacent to the gather validator;
* update explicit and pre-realized computed-mask indexed scatter target
  fixtures and script expectations to the new provider summary;
* add C++ target artifact negative coverage for stale provider plan/summary
  and stale candidate mirror metadata;
* add focused script dry-run and direct pre-realized route-entry fail-closed
  coverage for this op kind;
* strengthen the generated-bundle runtime harness only as needed to prove
  masked indexed scatter behavior, source preservation, and tail preservation
  over the required counts and patterns.

## Out Of Scope

* Segment2 memory.
* More computed-masked indexed gather rework.
* Plain indexed gather/scatter rework beyond direct dependencies.
* Strided memory rework.
* Contraction follow-up.
* Dtype or LMUL clone batches.
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
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`

Closest archived patterns:

* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-indexed-scatter-unit-load-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-indexed-gather-load-unit-store-artifact-abi/prd.md`

Initial implementation focus:

* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* `test/Target/RVV/*computed-masked-indexed-scatter-store.mlir`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Scripts/*computed-masked-indexed-scatter-store*.test`

## Completed Behavior

* Tightened `stringifyRVVRouteOperandBindingPlan` so
  `rvv-route-operand-binding:computed_masked_indexed_scatter_store_unit_load.v1`
  normalizes provider ABI participation to `abi` and header/prototype
  participation to `hdr`, matching the computed-mask indexed gather ABI
  boundary.
* Extended the computed-mask indexed scatter route operand-binding plan to
  record header/prototype participation for `cmp_lhs`, `cmp_rhs`, `src`,
  `index`, `dst`, and `n`.
* Strengthened computed-mask memory operand-binding validation so the scatter
  branch requires compare lhs/rhs and source unit-load header markers before
  route statement construction.
* Added a target-owned
  `validateComputedMaskIndexedScatterHeaderBindingSummary` validator. Target
  artifact export now fails closed unless the rebuilt provider description has
  the expected scatter provider plan id, exact provider summary prefix,
  runtime ABI order `cmp_lhs,cmp_rhs,src,index,dst,n`, and matching `abi` plus
  `hdr` markers for every exported ABI parameter.
* Updated explicit and pre-realized computed-mask indexed scatter artifact
  fixtures and generated-bundle metadata expectations to the provider/header
  summary.
* Added C++ target artifact negative coverage for wrong scatter binding plan,
  stale scatter binding summary without `hdr`, and stale candidate
  `tcrv_rvv.route_operand_binding_operands` mirror metadata.
* Added explicit dry-run, pre-realized dry-run, and direct pre-realized
  route-entry fail-closed script tests for
  `computed_masked_indexed_scatter_store_unit_load`.
* Strengthened the generated C harness for computed-mask indexed scatter to
  run two patterns. Pattern 1 changes index permutation, compare mask shape,
  and source data values while checking unique noncontiguous scatter,
  masked-off destination preservation, source preservation, and tail
  preservation.

## Validation Evidence

Commands run:

```bash
rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test
rtk git diff --check
rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2
rtk build/bin/tianchenrv-target-artifact-export-test
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter computed-masked-indexed-scatter-store
```

Focused lit result:

```text
Total Discovered Tests: 471
  Excluded: 466
  Passed  :   5
```

Direct pre-realized route-entry fail-closed evidence:

```bash
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-indexed-scatter-store-unit-load-artifact-abi/direct-pre-realized-computed-mask-indexed-scatter-store-fail-closed --run-id direct-pre-realized-computed-mask-indexed-scatter-store --overwrite --op-kind computed_masked_indexed_scatter_store_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20
```

Expected diagnostic:

```text
--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): computed_masked_indexed_scatter_store_unit_load; the direct route-entry shortcut is retired and these fixtures must use the public selected lowering-boundary producer before target bundle export
```

Real `ssh rvv` evidence command:

```bash
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-indexed-scatter-store-unit-load-artifact-abi/ssh-rvv-pre-realized --run-id pre-realized-computed-mask-indexed-scatter-store --overwrite --op-kind computed_masked_indexed_scatter_store_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20
```

Remote RVV result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=computed_masked_indexed_scatter_store_unit_load counts=0,1,16,17,257 patterns=0,1
```

The remote output includes per-count/per-pattern success lines. Counts `16`,
`17`, and `257` reported both active and inactive mask lanes for both patterns,
nonzero `noncontiguous_index_lanes`, `source_preserved`, and
`tail_preserved`.

## Self-Repair

* The first C++ build failed because the long target artifact test function
  already had a local `staleIndexedScatterBindingSummary` variable for the
  plain indexed scatter path. Renamed the new computed-mask scatter variable to
  `staleComputedMaskIndexedScatterBindingSummary` and reran the build
  successfully.

## Old-Authority Scan

Cached diff scan command after task archive:

```bash
rtk bash -lc 'git diff --cached -U0 -- lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp scripts/rvv_generated_bundle_abi_e2e.py test/Target/TargetArtifactExportTest.cpp test/Target/RVV/explicit-selected-body-artifact-computed-masked-indexed-scatter-store.mlir test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-indexed-scatter-store.mlir test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-indexed-scatter-store-dry-run.test test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-indexed-scatter-store-dry-run.test test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-indexed-scatter-store-fail-closed.test .trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-indexed-scatter-store-unit-load-artifact-abi/prd.md .trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-indexed-scatter-store-unit-load-artifact-abi/implement.jsonl .trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-indexed-scatter-store-unit-load-artifact-abi/check.jsonl | rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|__riscv_.*_i32m1|source-front-door|source-artifact|emission_plan|descriptor|selected route" || true'
```

Result: no hits in production code, target fixtures, or script logic added by
this task. Cached diff hits are limited to PRD/context documentation and new
FileCheck `implicit-check-not` guardrails, classified below.

Bounded full touched-file scan classifications:

* New script tests contain `descriptor`, `source-export`, and `tcrv_rvv.i32_`
  only in `implicit-check-not` guardrails.
* This PRD and context JSONL contain scan terms only as acceptance,
  constraints, or spec-context documentation.
* `test/Target/TargetArtifactExportTest.cpp` has broad preexisting legacy,
  exact-intrinsic, descriptor, and emission-plan negative tests outside this
  computed-mask indexed scatter change. The new scatter additions use exact
  intrinsic strings only inside manual target-artifact fixtures that validate a
  rebuilt provider description, not as route authority.
* `scripts/rvv_generated_bundle_abi_e2e.py` has broad preexisting exact
  intrinsic constants and descriptor-residue checks for other route families
  and self-tests. This task only changed scatter binding expectations and
  harness evidence.
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` full-file `selected route`
  hits are existing diagnostics/guardrails such as same-analysis checks. The
  new provider path change only adds scatter `hdr` marker normalization and
  header marker validation.

## Spec Update Decision

No `.trellis/spec/` update is needed. The durable rule used here already
exists in `.trellis/spec/lowering-runtime/emitc-route.md` under provider
operand-binding summaries and in `.trellis/spec/extension-plugins/rvv-plugin.md`
under computed-mask memory statement-plan and target artifact route-family
validator boundaries.
