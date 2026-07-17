# Stage2 RVV computed-mask memory executable artifact closure

## Goal

Close executable artifact evidence for one bounded owner-driven computed-mask
memory route. The chosen representative route is
`computed_masked_indexed_scatter_store_unit_load`, carried from a selected
`tcrv.exec` RVV variant with a pre-realized typed `tcrv_rvv` body through RVV
selected-body realization, the migrated computed-mask memory statement-plan
route-provider owner, `TCRVEmitCLowerableRoute`, common EmitC
materialization, generated RVV C artifact/header/harness, and real `ssh rvv`
correctness execution.

## What I Already Know

* Repository state before task creation was clean on `main`, with HEAD
  `7fe6f7a3 chore(task): archive direct contraction executable closure`.
* `.trellis/.current-task` did not exist, so this task was created from the
  Hermes direction brief before source edits.
* The predecessor task
  `.trellis/tasks/archive/2026-05/05-24-05-24-stage2-rvv-computed-mask-memory-statement-plan-ownership/`
  moved the active non-segment computed-mask memory statement sequence behind
  `RVVSelectedBodyComputedMaskMemoryRouteStatementPlan`.
* The predecessor task
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-route-provider-family-owner-registry/`
  moved migrated statement-plan provider consumption behind an exact-one
  owner registry that includes computed-mask memory.
* The latest completed task
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-direct-contraction-executable-artifact-closure/`
  proved a direct-provider contraction route can traverse the same generated
  bundle and `ssh rvv` correctness path without production repair.
* Current generated-bundle tooling lists non-segment computed-mask memory
  op kinds including `computed_masked_unit_load_store`,
  `computed_masked_strided_store`,
  `computed_masked_strided_load_unit_store`,
  `computed_masked_indexed_gather_load_unit_store`, and
  `computed_masked_indexed_scatter_store_unit_load`.
* `computed_masked_indexed_scatter_store_unit_load` is the chosen bounded
  representative because it exercises compare-produced mask facts, indexed
  addressing, masked scatter store, unit source load, runtime element count,
  output memory mutation, and tail/sentinel preservation.

## Context Read

* `.trellis/spec/index.md` defines the RVV-first authority chain:
  `tcrv.exec` envelope -> selected RVV variant -> typed low-level
  `tcrv_rvv` body -> RVV plugin-owned legality/realization/route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires computed-mask memory
  statement-plan construction to flow through the RVV-owned computed-mask
  memory family plan, materialization facts, memory operand-binding facts,
  route-control provider plan, and migrated statement-plan owner registry.
* `.trellis/spec/lowering-runtime/emitc-route.md` keeps common EmitC neutral:
  RVV headers, vector C types, intrinsic strings, runtime ABI bindings, and
  route payloads are provider-owned and only materialized by common code.
* `.trellis/spec/plugin-protocol/locality-contract.md` requires route
  construction to stay inside the origin plugin and forbids common/core code
  from filling in RVV runtime ABI, artifact kind, intrinsic, dtype, or support
  claims.
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md` requires
  selected-body realization to consume code-affecting hints/config into real
  typed body structure before route construction.
* `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  output for RVV runtime/correctness claims and requires memory-writing routes
  to check active-lane behavior plus guard/tail preservation.

## Requirements

* Use one representative non-segment computed-mask memory route:
  `computed_masked_indexed_scatter_store_unit_load`.
* Prove the selected path starts from a selected `tcrv.exec` RVV fixture and a
  typed pre-realized `tcrv_rvv` body, not a source-front-door artifact.
* Prove selected-body realization consumes the pre-realized body before route
  facts are collected.
* Prove the migrated computed-mask memory statement-plan route-provider owner
  remains the statement construction authority for the chosen route.
* Prove route-control provider-plan facts, materialization facts, memory
  operand-binding facts, typed config/capability facts, runtime AVL/VL facts,
  compare/mask facts, indexed address facts, source/destination memory roles,
  and ABI/mem_window role bindings are carried through provider route
  construction.
* Prove common EmitC only materializes provider output and does not invent RVV
  semantics.
* Produce a generated RVV artifact/header/harness and run it on `ssh rvv` with
  a focused correctness oracle over multiple runtime `n` values.
* Run one representative direct-contraction or migrated statement-plan
  non-regression dry-run.
* If current production code already satisfies the executable boundary, avoid
  unnecessary compiler or script edits and record the exact evidence instead.
* If execution fails, repair only the production runtime/artifact boundary
  needed for the chosen route, or record the exact blocker if hardware or
  toolchain access prevents completion.

## Acceptance Criteria

* [x] Task context files contain real spec/task context and no placeholder
      JSONL entries.
* [x] The chosen computed-mask memory route is named and bounded.
* [x] Current production route generation for the chosen route succeeds
      through pre-realized selected-body materialization, emission-plan
      construction, provider-built route, common EmitC materialization, and
      target bundle export.
* [x] The generated artifact/header/harness compiles and runs on `ssh rvv`.
* [x] The correctness oracle checks compare-produced active/inactive mask lanes,
      indexed scatter destination positions, unit source load values, runtime
      AVL/VL counts, output memory writes, and tail/sentinel preservation.
* [x] A direct-contraction or migrated statement-plan owner non-regression
      dry-run still succeeds.
* [x] Bounded touched-file/current-path scan shows no new legacy
      i32/source-front-door/descriptor/direct-C/source-export/ABI-string/
      artifact-name/script/common-EmitC/metadata/route-id authority leaks.
* [x] `git diff --check` passes.
* [x] `check-tianchenrv` passes, or an exact blocker is recorded.

## Out Of Scope

* New RVV operation families, new computed-mask memory variants, segment2
  computed-mask memory closure, dtype/LMUL clone batches, source-front-door
  positive routes, frontend/Linalg routes, common EmitC RVV semantics,
  descriptor/direct-C/source-export computation, dashboards, broad smoke
  matrices, performance claims, non-RVV plugin work, dispatch/fallback changes,
  and changes that weaken owner-driven route construction or fail-closed
  diagnostics.

## Validation Plan

1. Validate and start this Trellis task.
2. Run a local generated-bundle dry-run for
   `computed_masked_indexed_scatter_store_unit_load` using the pre-realized
   selected-body fixture and runtime counts `1,7,17,33,257`.
3. Run non-dry-run `ssh rvv` generated-bundle execution for the same route and
   counts.
4. Run a representative non-regression dry-run for
   `computed_masked_strided_input_widening_dot_reduce_add` or another already
   completed owner-driven path.
5. Run a bounded current-path authority scan over RVV planning/provider/target
   artifact/script/test surfaces.
6. Run `git diff --check`.
7. Run `cmake --build build --target check-tianchenrv -j2`.

## Definition Of Done

The task either records real executable evidence on the existing production
path or repairs the exact production boundary needed to obtain it. PRD,
`implement.jsonl`, `check.jsonl`, Trellis status, journal, and final report are
truthful. Completed work is archived and committed as one coherent commit, or
the unfinished continuation point is explicit.

## Evidence Captured

No production compiler, target, or script repair was required. Current HEAD
already carries `computed_masked_indexed_scatter_store_unit_load` through the
owner-driven computed-mask memory route path:

```text
selected pre-realized tcrv.exec RVV fixture
  -> tcrv_rvv pre-realized computed-mask indexed scatter body
  -> selected-body realization
  -> computed-mask memory route-family plan
  -> migrated statement-plan route-provider owner
  -> provider-built TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> generated RVV object/header/harness
  -> ssh rvv compile/run correctness evidence
```

Local generated-bundle dry-run for the chosen route passed:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --dry-run \
  --pre-realized-selected-body \
  --op-kind computed_masked_indexed_scatter_store_unit_load \
  --runtime-count 1 --runtime-count 7 --runtime-count 17 \
  --runtime-count 33 --runtime-count 257 \
  --artifact-root artifacts/tmp/stage2_rvv_computed_mask_memory_executable_artifact_closure \
  --run-id pre-realized-computed-mask-indexed-scatter-dry \
  --overwrite \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate
```

The dry-run artifact is:

```text
artifacts/tmp/stage2_rvv_computed_mask_memory_executable_artifact_closure/pre-realized-computed-mask-indexed-scatter-dry
```

Non-dry-run `ssh rvv` generated-bundle execution for the chosen route passed:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --op-kind computed_masked_indexed_scatter_store_unit_load \
  --runtime-count 1 --runtime-count 7 --runtime-count 17 \
  --runtime-count 33 --runtime-count 257 \
  --artifact-root artifacts/tmp/stage2_rvv_computed_mask_memory_executable_artifact_closure \
  --run-id pre-realized-computed-mask-indexed-scatter-ssh \
  --overwrite \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --timeout 180 \
  --connect-timeout 10
```

The `ssh rvv` run compiled on `riscv64` with `/usr/bin/clang` and printed:

```text
PASS op=computed_masked_indexed_scatter_store_unit_load counts=1,7,17,33,257
```

Per-count correctness output covered:

```text
n=1   active_lanes=1   inactive_lanes=0   noncontiguous_index_lanes=0
n=7   active_lanes=3   inactive_lanes=4   inactive_preserved_lanes=4   noncontiguous_index_lanes=6
n=17  active_lanes=9   inactive_lanes=8   inactive_preserved_lanes=8   noncontiguous_index_lanes=16
n=33  active_lanes=17  inactive_lanes=16  inactive_preserved_lanes=16  noncontiguous_index_lanes=32
n=257 active_lanes=129 inactive_lanes=128 inactive_preserved_lanes=128 noncontiguous_index_lanes=256
```

Each nontrivial count also reported `source_preserved tail_preserved`.

The generated evidence records:

* input mode: `pre-realized-selected-body`;
* selected input:
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-indexed-scatter-store.mlir`;
* ABI prototype:
  `void tcrv_emitc_pre_realized_body_cmidx_store_kernel_pre_realized_body_rvv_cmidx_store(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src, const uint32_t *index, int32_t *dst, size_t n);`
* runtime ABI order: `cmp_lhs,cmp_rhs,src,index,dst,n`;
* route operand binding plan:
  `rvv-route-operand-binding:computed_masked_indexed_scatter_store_unit_load.v1`;
* computed-mask memory route family plan:
  `rvv-computed-mask-memory-route-family-plan.v1`;
* provider-supported mirror:
  `provider_supported_mirror:rvv-computed-mask-indexed-scatter-store-plan-validated`;
* mask source:
  `compare-produced-mask-same-vl-scope`;
* inactive lane contract:
  `masked-indexed-store-false-lanes-preserve-output-buffer`;
* indexed scatter contract:
  active lanes store `source[i]` to `destination[index[i]]` while inactive
  indexed lanes, source buffers, and destination tail slots preserve sentinels.

Materialized selected-body evidence shows the pre-realized body was consumed
into typed low-level RVV structure:

```text
tcrv_rvv.runtime_abi_value cmp_lhs/cmp_rhs/src/index/dst/n
tcrv_rvv.setvl %n
tcrv_rvv.with_vl
tcrv_rvv.load cmp_lhs/cmp_rhs/src
tcrv_rvv.index_load index
tcrv_rvv.compare {kind = "slt"}
tcrv_rvv.masked_indexed_store
```

Generated EmitC/C++ evidence uses provider-derived RVV intrinsic calls:

```text
__riscv_vsetvl_e32m1
__riscv_vle32_v_i32m1
__riscv_vle32_v_u32m1
__riscv_vmslt_vv_i32m1_b32
__riscv_vsoxei32_v_i32m1_m
```

Direct-contraction non-regression dry-run passed:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --dry-run \
  --pre-realized-selected-body \
  --op-kind computed_masked_strided_input_widening_dot_reduce_add \
  --runtime-count 1 --runtime-count 17 \
  --artifact-root artifacts/tmp/stage2_rvv_computed_mask_memory_executable_artifact_closure \
  --run-id direct-contraction-nonregression-dry \
  --overwrite \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate
```

The direct-contraction dry-run artifact is:

```text
artifacts/tmp/stage2_rvv_computed_mask_memory_executable_artifact_closure/direct-contraction-nonregression-dry
```

Current-path authority scan:

```bash
rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|source-front-door|source-artifact|descriptor|__riscv_.*_i32m1" \
  include/TianChenRV/Plugin/RVV lib/Plugin/RVV/EmitC lib/Target/RVV \
  include/TianChenRV/Target/RVV scripts/rvv_generated_bundle_abi_e2e.py \
  test/Target/RVV
```

reported existing provider-derived intrinsic mirror checks, descriptor/source
front-door negative checks, and legacy fail-closed fixtures. No production
files were changed in this task, and:

```bash
git diff --name-only -- ':!.trellis/tasks/**'
```

was empty.

Task-file authority scan over this task's diff found no forbidden authority
terms outside explicit non-goal/fail-closed wording.

Final checks:

* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-stage2-rvv-computed-mask-memory-executable-artifact-closure`
  passed.
* `git diff --check` passed.
* `cmake --build build --target check-tianchenrv -j2` passed: 379/379 tests.

## Spec Update Judgment

No `.trellis/spec/**` update is needed in this round. The executable closure
confirmed existing durable contracts for computed-mask memory statement-plan
ownership, migrated route-provider owner selection, common EmitC neutrality,
and RVV generated-bundle evidence. It did not introduce a new API, ownership
boundary, failure mode, or convention beyond the already recorded specs.
