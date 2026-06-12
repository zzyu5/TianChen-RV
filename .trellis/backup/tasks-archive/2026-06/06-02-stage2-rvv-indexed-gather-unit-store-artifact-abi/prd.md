# Stage2 RVV indexed gather unit-store artifact ABI boundary

## Goal

Prove and, where needed, tighten one bounded Stage 2 path for an existing
pre-realized `indexed_gather_unit_store` selected RVV body:

```text
selected tcrv.exec RVV variant
  -> RVV plugin-owned realized/validated tcrv_rvv indexed gather body
  -> base memory route-family provider facts
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV target artifact bundle
  -> ssh rvv correctness evidence
```

The module goal is the ABI boundary for indexed memory movement. Index-buffer
role, index vector/token facts, indexed data-base role, unit output store role,
runtime `n`/AVL, dtype/config, memory form, VL/tail policy, provider route
operand binding, and generated artifact ABI must be structural facts from the
typed selected `tcrv_rvv` body or plugin validation/realization. They must not
be inferred from route ids, artifact names, manifests, test names, C strings,
descriptors, or exact intrinsic spellings.

## What I already know

* The repository started this task from a clean `main` worktree.
* Recent commits completed RVV artifact ABI boundaries for widening conversion,
  compare/select, mask-select, widening dot, strided widening dot, and combined
  computed-masked strided dot-reduction.
* The next requested Stage 2 bottleneck is irregular memory movement, centered
  on one existing `indexed_gather_unit_store` selected body.
* The expected owner files are around RVV ops, RVV base-memory selected-body
  realization, EmitC route-family planning, route provider, common EmitC
  materialization, RVV target artifact validation/support bundle, generated
  bundle ABI script, remote probe script, and focused tests.
* No `.trellis/.current-task` existed before this task was created.

## Assumptions

* The task should prefer proving and tightening the existing production path
  over inventing a new indexed gather implementation.
* If the path already carries most facts structurally, the production change
  should be the smallest missing fail-closed guard or route-fact validation.
* Runtime/correctness claims require real `ssh rvv` evidence.

## Requirements

* Validate or realize the pre-realized `indexed_gather_unit_store` selected body
  in the RVV plugin/base-memory owner before route construction.
* Preserve, expose, or validate structural provider facts for:
  * indexed data-base ABI role,
  * index-buffer ABI role,
  * index vector/token facts,
  * unit output store role,
  * runtime `n`/AVL binding,
  * dtype/config, SEW, LMUL, and policy,
  * indexed memory form and unit output store form,
  * VL/tail policy,
  * provider route operand binding.
* Ensure generated artifact bundle/header/prototype facts match that structural
  route instead of test-name/header-string inference.
* Add focused fail-closed coverage for direct pre-realized route/artifact export
  or missing index/data/output/runtime binding where production guards are
  missing.
* Keep common EmitC/export neutral; RVV semantics must remain in RVV plugin
  route planning/provider/validation.
* Update task context files (`implement.jsonl`, `check.jsonl`) with relevant
  specs/research only.

## Acceptance Criteria

* [x] Focused evidence shows `indexed_gather_unit_store` is realized or
      validated before route construction by RVV plugin/base-memory ownership.
* [x] Focused EmitC/export or target-artifact test shows index buffer ABI,
      index vector/token facts, indexed data-base role, output store role,
      runtime `n`/AVL, dtype/config, memory form, and tail policy survive into
      the generated bundle/header/prototype.
* [x] `ssh rvv` compile/run correctness is recorded for counts `0`, `1`, a
      VL-boundary count, a tail count, and a larger count, with at least two
      index/data patterns that distinguish indexed gather from contiguous or
      strided loads and prove output store order.
* [x] Focused fail-closed evidence covers direct pre-realized route/artifact
      export or missing index/data/output/runtime binding.
* [x] A bounded old-authority scan over touched plugin/provider/materializer/
      target/script/test/spec files classifies remaining hits for:
      `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
      `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
      `emission_plan`, `descriptor`, and `selected route`.
* [x] Smallest relevant build/test/script commands pass, with self-repair for
      failures.
* [x] The task is finished/archived and one coherent commit is created if the
      module behavior is complete.

## Implementation Completed

* Tightened `stringifyRVVRouteOperandBindingPlan` for
  `rvv-route-operand-binding:indexed_gather_unit_store.v1` so artifact-facing
  route operand binding facts use provider/header markers `abi` and `hdr` while
  preserving the indexed-gather materialized facts:
  `materialized-indexed-data-base`, `indexed-load-base`,
  `materialized-index-load-base`, `index-offset-scale`,
  `index-source-mirror`, `materialized-store-base`, `setvl-avl`, and
  `loop-control`.
* Added RVV target artifact provider validation for
  `IndexedGatherUnitStore` route operand binding summaries. The validator now
  requires `data,index,out,n` entries to match the provider runtime ABI order,
  runtime ABI role, C ABI name, `abi` marker, and `hdr` header/prototype marker
  before artifact export.
* Added target artifact regression coverage that fails closed when the indexed
  gather provider summary falls back to stale `header-mirror` wording, and when
  candidate metadata rewrites the `tcrv_rvv.route_operand_binding_operands`
  mirror away from the provider-built summary.
* Strengthened `scripts/rvv_generated_bundle_abi_e2e.py` indexed gather
  expectations and harness generation. The harness now runs two distinct
  index/data patterns, covers counts `0,1,16,17,257`, checks
  `data[indices[index]]`, rejects vacuous contiguous behavior, records
  output-order-distinguishing lanes, and checks tail sentinel preservation past
  `n`.
* Added direct pre-realized route-entry fail-closed script coverage for
  `indexed_gather_unit_store`.

## Validation Evidence

* `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
* `rtk git diff --check` passed.
* `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
  passed after self-repair. The first build attempt found a real C++ type
  error: `std::string` route operand summaries cannot call `split` directly;
  the validator now wraps the summary as `llvm::StringRef` before splitting.
* `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter indexed-gather-unit-store`
  from `build/test` passed 5 focused tests:
  explicit selected-body artifact, pre-realized selected-body artifact,
  explicit dry-run script, pre-realized dry-run script, and direct
  pre-realized route-entry fail-closed script.
* Focused dry-run evidence:
  `artifacts/tmp/06-02-stage2-rvv-indexed-gather-unit-store-artifact-abi/focused-dry-run/pre-realized-indexed-gather-unit-store`.
* Direct route-entry fail-closed evidence:
  `--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): indexed_gather_unit_store; the direct route-entry shortcut is retired...`.
* Real `ssh rvv` evidence:
  `artifacts/tmp/06-02-stage2-rvv-indexed-gather-unit-store-artifact-abi/final-ssh-rvv/pre-realized-indexed-gather-unit-store`.
  The run passed counts `0,1,16,17,257` with two index/data patterns. Counts
  `16`, `17`, and `257` reported nonzero
  `output_order_distinguishing_lanes`, and all cases reported
  `tail_preserved`.

## Old-Authority Scan

Command used over the changed diff:

```bash
rtk git diff -U0 | rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|__riscv_.*_i32m1|source-front-door|source-artifact|emission_plan|descriptor|selected route"
```

Result: no hits in the changed diff.

Bounded full touched-file scan classifications:

* `test/Scripts/*indexed-gather-unit-store*.test`: remaining `descriptor`,
  `source-export`, and `tcrv_rvv.i32_` hits are `implicit-check-not`
  guardrails.
* `test/Target/RVV/*indexed-gather-unit-store.mlir`: remaining
  `emission_plan` hits are diagnostic reason fields, not route authority.
* This PRD contains the scan terms only as acceptance/scan documentation.
* `test/Target/TargetArtifactExportTest.cpp` and
  `scripts/rvv_generated_bundle_abi_e2e.py` have broad preexisting suite hits
  outside the indexed-gather changes: legacy-negative tests, non-indexed
  feature intrinsic checks, descriptor-residue rejection, and old-authority
  guardrails. The changed diff introduced no new `i32m1`, descriptor,
  source-front-door/source-artifact, or selected-route authority.
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` full-file `selected route`
  hits are existing diagnostic/context strings and same-analysis guardrails.
  The changed indexed-gather stringifier adds only `abi`/`hdr` marker handling
  and does not introduce route-id or artifact-name authority.

## Out of Scope

* Indexed scatter.
* Computed-masked indexed gather.
* Segment2 load/store.
* Strided-store rework.
* Contraction follow-up.
* Dtype or LMUL clone batches.
* High-level frontend authority or per-Linalg lowering.
* Source-front-door positive routes.
* Common EmitC semantic inference.
* Dashboards, broad smoke matrices, or performance claims.

## Technical Notes

Read first:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-strided-input-widening-dot-reduce-artifact-abi/`
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVBaseMemoryMovementSelectedBodyRealizationOwner.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `scripts/rvv_remote_probe.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-indexed-gather-unit-store.mlir`
* focused script tests for `indexed_gather_unit_store`
