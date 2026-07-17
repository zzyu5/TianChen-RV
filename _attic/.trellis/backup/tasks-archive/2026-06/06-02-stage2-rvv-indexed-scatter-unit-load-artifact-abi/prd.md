# Stage2 RVV indexed scatter unit-load artifact ABI boundary

## Goal

Prove and tighten one bounded Stage 2 path for an existing pre-realized
`indexed_scatter_unit_load` selected RVV body:

```text
selected tcrv.exec RVV variant
  -> RVV plugin-owned realized/validated tcrv_rvv indexed scatter body
  -> base memory route-family provider facts
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV target artifact bundle
  -> ssh rvv correctness evidence
```

The module goal is the ABI boundary for reverse indexed memory movement.
Source unit-load role, index-buffer role, index vector/token facts, indexed
destination-store role, runtime `n`/AVL, dtype/config, memory form, VL/tail
policy, provider route operand binding, and generated artifact ABI must be
structural facts from the typed selected `tcrv_rvv` body or plugin
validation/realization. They must not be inferred from route ids, artifact
names, manifests, test names, C strings, descriptors, or exact intrinsic
spellings.

## What I already know

* The repository started this task from a clean `main` worktree.
* The previous completed task closed the matching `indexed_gather_unit_store`
  artifact ABI boundary with provider/header operand-binding markers, target
  artifact fail-closed validation, focused lit/script coverage, and `ssh rvv`
  correctness over counts `0,1,16,17,257`.
* Current `indexed_scatter_unit_load` artifacts already have a selected-body
  explicit fixture, a pre-realized fixture, base-memory route-family support,
  generated-bundle script support, and dry-run tests.
* Live inspection shows the scatter route operand binding summary still uses
  mirror wording such as `runtime-abi-mirror` and `header-mirror`, while the
  last gather task tightened the analogous summary to provider/header markers
  `abi` and `hdr` and added target artifact validation.
* No `.trellis/.current-task` existed before this task was created.

## Requirements

* Validate or realize the pre-realized `indexed_scatter_unit_load` selected
  body in RVV plugin/base-memory ownership before route construction.
* Preserve, expose, or validate structural provider facts for:
  * source unit-load ABI role,
  * index-buffer ABI role,
  * index vector/token facts,
  * indexed destination-store role,
  * runtime `n`/AVL binding,
  * dtype/config, SEW, LMUL, and policy,
  * unit-load/indexed-store memory form,
  * VL/tail policy,
  * provider route operand binding.
* Tighten artifact-facing scatter route operand binding summaries so generated
  bundle/header/prototype facts carry provider ABI and header/prototype markers
  rather than mirror-only wording.
* Add focused target artifact fail-closed coverage for stale scatter binding
  summaries and stale candidate binding mirrors.
* Add direct pre-realized route-entry fail-closed script coverage for
  `indexed_scatter_unit_load`.
* Strengthen generated-bundle evidence for scatter to cover counts `0,1,16,17`
  and a larger count, at least two unique non-monotonic index/source patterns,
  source preservation, destination tail sentinel preservation, and behavior
  that distinguishes indexed scatter from contiguous or strided stores.
* Keep common EmitC/export neutral; RVV semantics must remain in RVV plugin
  route planning/provider/validation.
* Update task context files (`implement.jsonl`, `check.jsonl`) with relevant
  specs/research only.

## Acceptance Criteria

* [x] Focused evidence shows `indexed_scatter_unit_load` is realized or
      validated before route construction by RVV plugin/base-memory ownership.
* [x] Focused EmitC/export or target-artifact test shows source unit-load ABI,
      index buffer ABI, index vector/token facts, indexed destination-store
      role, runtime `n`/AVL, dtype/config, memory form, and tail policy survive
      into the generated bundle/header/prototype.
* [x] `ssh rvv` compile/run correctness is recorded for counts `0`, `1`, a
      VL-boundary count, a tail count, and a larger count, with at least two
      index/source patterns that distinguish indexed scatter from contiguous or
      strided stores and prove unwritten/tail sentinel preservation.
* [x] Focused fail-closed evidence covers direct pre-realized route/artifact
      export or missing source/index/destination/runtime binding.
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
  `rvv-route-operand-binding:indexed_scatter_unit_load.v1` so artifact-facing
  route operand binding facts use provider/header markers `abi` and `hdr`,
  while preserving the scatter materialized facts:
  `materialized-load-base`, `move-source`, `materialized-index-load-base`,
  `index-offset-scale`, `index-source-mirror`,
  `materialized-indexed-store-base`, `setvl-avl`, and `loop-control`.
* Generalized RVV target artifact binding-summary validation from
  `indexed_gather_unit_store` to indexed base-memory routes. The validator now
  requires `indexed_scatter_unit_load` logical operands `src,index,dst,n` to
  match provider runtime ABI order, runtime ABI role, C ABI name, `abi`, and
  `hdr` before artifact export.
* Added target artifact regression coverage that fails closed when the scatter
  provider summary falls back to stale `header-mirror` wording, and when
  candidate metadata rewrites the
  `tcrv_rvv.route_operand_binding_operands` mirror away from the
  provider-built summary.
* Updated explicit and pre-realized indexed scatter target fixtures and script
  dry-runs to check the provider/header `abi`/`hdr` summary.
* Strengthened `scripts/rvv_generated_bundle_abi_e2e.py` scatter expectations
  and harness generation. The harness now runs two unique non-monotonic
  index/source patterns, covers counts `0,1,16,17,257`, rejects vacuous
  same-order scatter behavior, records output-order-distinguishing lanes, and
  checks source and tail sentinel preservation.
* Added direct pre-realized route-entry fail-closed script coverage for
  `indexed_scatter_unit_load`.

## Validation Evidence

* `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
* `rtk git diff --check` passed.
* `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
  passed.
* `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter indexed-scatter-unit-load`
  from `build/test` passed 5 focused tests:
  explicit selected-body artifact, pre-realized selected-body artifact,
  explicit dry-run script, pre-realized dry-run script, and direct
  pre-realized route-entry fail-closed script.
* Direct route-entry fail-closed evidence:
  `artifacts/tmp/06-02-stage2-rvv-indexed-scatter-unit-load-artifact-abi/direct-pre-realized-indexed-scatter-unit-load-fail-closed`.
  The diagnostic was:
  `--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): indexed_scatter_unit_load; the direct route-entry shortcut is retired...`.
* Real `ssh rvv` evidence:
  `artifacts/tmp/06-02-stage2-rvv-indexed-scatter-unit-load-artifact-abi/final-ssh-rvv-pre-realized-indexed-scatter-unit-load`.
  The run passed counts `0,1,16,17,257` with index/source patterns `0` and
  `1`. Counts `16`, `17`, and `257` reported nonzero
  `output_order_distinguishing_lanes`, and all cases reported
  `source_preserved tail_preserved`.

## Self-Repair

* The first focused lit run found a FileCheck ordering issue in the two scatter
  dry-run harness checks: `unique_non_monotonic_indexed_scatter` appeared
  before `output_order_distinguishing_lanes` in the generated `printf`.
  Reordered the checks and reran the focused lit filter successfully.

## Old-Authority Scan

Diff-only scan command:

```bash
rtk git diff -U0 -- lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp scripts/rvv_generated_bundle_abi_e2e.py test/Scripts/rvv-generated-bundle-abi-e2e-indexed-scatter-unit-load-dry-run.test test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-indexed-scatter-unit-load-dry-run.test test/Target/RVV/explicit-selected-body-artifact-indexed-scatter-unit-load.mlir test/Target/RVV/pre-realized-selected-body-artifact-indexed-scatter-unit-load.mlir test/Target/TargetArtifactExportTest.cpp | rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|__riscv_.*_i32m1|source-front-door|source-artifact|emission_plan|descriptor|selected route"
```

Result: no hits in the changed diff.

Bounded full touched-file scan classifications:

* `test/Target/RVV/*indexed-scatter-unit-load.mlir`: remaining
  `emission_plan` hits are diagnostic reason fields, not route authority.
* `test/Scripts/*indexed-scatter-unit-load*.test`: remaining `descriptor`,
  `source-export`, and `tcrv_rvv.i32_` hits are `implicit-check-not`
  guardrails.
* This PRD contains the scan terms only as acceptance/scan documentation.
* `test/Target/TargetArtifactExportTest.cpp` and
  `scripts/rvv_generated_bundle_abi_e2e.py` have broad preexisting suite hits
  outside this indexed-scatter change: legacy-negative tests, descriptor
  residue rejection, non-scatter intrinsic checks, and old-authority
  guardrails. The changed diff introduced no new `i32m1`, descriptor,
  source-front-door/source-artifact, or selected-route authority.
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` full-file `selected route`
  hits are existing diagnostic/context strings and same-analysis guardrails.
  The changed scatter stringifier adds only `abi`/`hdr` marker handling and
  does not introduce route-id or artifact-name authority.

## Spec Update Decision

No `.trellis/spec/` update was needed. The durable contracts used in this task
already exist in `.trellis/spec/extension-plugins/rvv-plugin.md` under the
base memory movement statement-plan boundary and in
`.trellis/spec/lowering-runtime/emitc-route.md` under the provider operand
binding summary contract.

## Technical Approach

Use the already-working indexed gather ABI boundary as the closest production
pattern, but apply it only to scatter:

* extend RVV provider operand-binding summary stringification for
  `rvv-route-operand-binding:indexed_scatter_unit_load.v1` so
  `runtime-abi-mirror` becomes `abi` and `header-mirror` becomes `hdr`;
* generalize target artifact validation from gather-only to indexed
  base-memory binding summaries, with scatter logical operands
  `src,index,dst,n` and the corresponding provider runtime ABI order;
* update focused MLIR/FileCheck and script expectations to the new provider
  summary;
* strengthen the scatter generated-bundle harness and self-test to prove
  unique non-monotonic indexed scatter behavior, source preservation, and tail
  preservation across multiple patterns and runtime counts;
* add the direct pre-realized route-entry negative script test for scatter.

## Decision (ADR-lite)

**Context**: `indexed_scatter_unit_load` already reaches selected-body
realization, base-memory route planning, EmitC materialization, and target
artifact generation, but the artifact-facing binding summary still exposes
mirror-only wording and lacks the same target validation that now protects
indexed gather.

**Decision**: Tighten the existing production path in place rather than adding a
new scatter route or broad harness. The boundary will be enforced at provider
summary generation, target artifact validation, focused lit/script coverage,
and `ssh rvv` evidence.

**Consequences**: This keeps the task bounded and preserves RVV plugin/common
EmitC locality. Other base-memory routes that still use mirror wording remain
out of scope unless their own tasks promote them to provider/header marker
validation.

## Out of Scope

* Indexed gather rework beyond matching the existing pattern where needed.
* Computed-masked indexed scatter.
* Segment2 memory.
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
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-indexed-gather-unit-store-artifact-abi/prd.md`
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
* `test/Target/RVV/pre-realized-selected-body-artifact-indexed-scatter-unit-load.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-indexed-scatter-unit-load.mlir`
* focused script tests for `indexed_scatter_unit_load`
