# Stage2 RVV direct route-entry compare/select executable closure

## Goal

Close executable artifact evidence for one bounded direct pre-realized route-entry
compare/select path. The chosen representative route is `cmp_select`, carried
from a selected `tcrv.exec` RVV variant with a typed pre-realized
`tcrv_rvv` compare/select body through the RVV route-entry realization bridge,
RVV-owned route/family facts and statement-plan provider path,
`TCRVEmitCLowerableRoute`, common EmitC materialization, generated RVV
C artifact/header/harness, and real `ssh rvv` correctness execution.

## What I Already Know

* Repository state before task creation was clean on `main`, with HEAD
  `42b06f5b chore(task): archive computed mask memory executable closure`.
* `.trellis/.current-task` did not exist, so this task was created from the
  Hermes direction brief before source edits.
* The last two completed executable closures proved:
  * direct-provider contraction route
    `computed_masked_strided_input_widening_dot_reduce_add` traverses selected
    pre-realized body, direct contraction route-provider owner, common EmitC,
    generated bundle, and `ssh rvv` correctness;
  * migrated statement-plan computed-mask memory route
    `computed_masked_indexed_scatter_store_unit_load` traverses selected
    pre-realized body, computed-mask memory statement-plan owner, common EmitC,
    generated bundle, and `ssh rvv` correctness.
* A previous compare/select executable ABI task proved pre-realized
  `cmp_select` on `ssh rvv` and added predicate true/false lane checks, but
  this round must prove the current direct route-entry path at current HEAD
  after the later route-provider owner/registry work.
* The route-provider family owner registry includes compare/select as a
  migrated statement-plan family and recorded a representative direct
  pre-realized route-entry `cmp_select` dry-run.

## Context Read

* `.trellis/spec/index.md` defines the RVV-first authority chain:
  `tcrv.exec` envelope -> selected RVV variant -> typed low-level
  `tcrv_rvv` body -> RVV plugin-owned legality/realization/route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires
  elementwise/compare-select pre-realized bodies to be realized by the RVV
  plugin before route facts are collected; the direct route-entry bridge may
  realize supported pre-realized selected bodies and must fail closed for
  unsupported or malformed route-entry families.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires compare/select
  statement construction to flow through RVV-owned family/materialization/
  operand-binding/route-control facts into a compare/select statement plan
  before provider-built route construction.
* `.trellis/spec/lowering-runtime/emitc-route.md` keeps common EmitC neutral:
  RVV headers, vector C types, intrinsic strings, runtime ABI bindings, and
  route payloads are provider-owned and only materialized by common code.
* `.trellis/spec/testing/mlir-testing-contract.md` requires generated-bundle
  compare/select evidence to expose predicate/select boundary facts, prove
  realized body structure and emitted C/C++ predicate/select intrinsics, and
  include real `ssh rvv` output for runtime/correctness claims.
* `.trellis/spec/core-dialect/tcrv-exec-contract.md` treats `mem_window` and
  `runtime_param` as ABI/runtime role declarations only; selected typed
  extension bodies must explicitly import and consume them before executable
  lowering.
* Archived tasks read: route-provider family owner registry, direct contraction
  executable artifact closure, computed-mask memory executable artifact
  closure, plain compare-select route-family ownership, and prior compare/select
  executable ABI closure.

## Requirements

* Use one representative direct route-entry compare/select route: `cmp_select`.
* Prove the selected path starts from a selected `tcrv.exec` RVV fixture and a
  typed pre-realized `tcrv_rvv` compare/select body, not a source-front-door
  artifact.
* Prove direct route-entry realization consumes the pre-realized body before
  route facts are collected.
* Prove compare/select route construction consumes typed body/config/capability/
  runtime facts, route materialization facts, elementwise/select operand-binding
  facts, and route-control provider facts before constructing the provider
  route.
* Carry runtime `n`/AVL, predicate/condition facts, true/false selected value
  operands, policy/config facts, result memory role, operand bindings, selected
  capability, and `mem_window`/ABI facts from IR to artifact evidence.
* Prove common EmitC only materializes provider output and does not invent RVV
  semantics.
* Produce a generated RVV artifact/header/harness and run it on `ssh rvv` with a
  focused correctness oracle over multiple runtime `n` values, including both
  predicate-true and predicate-false lanes.
* Run one representative non-regression dry-run for a recent owner-driven path
  outside plain compare/select.
* If current production code already satisfies the executable boundary, avoid
  unnecessary compiler or script edits and record exact evidence instead.
* If execution fails, repair only the production runtime/artifact boundary
  needed for the chosen route, or record the exact blocker if hardware or
  toolchain access prevents completion.

## Acceptance Criteria

* [x] Task context files contain real spec/task context and no placeholder
      JSONL entries.
* [x] The chosen direct route-entry compare/select route is named and bounded.
* [x] Current production route generation for `cmp_select` succeeds through
      pre-realized selected-body route-entry realization, emission-plan
      construction, provider-built route, common EmitC materialization, and
      target bundle export.
* [x] The generated artifact/header/harness compiles and runs on `ssh rvv`.
* [x] The correctness oracle checks runtime counts, predicate-true lanes,
      predicate-false lanes, true/false selected value behavior, runtime AVL/VL
      handling, output memory writes, and tail/sentinel preservation where the
      harness exposes them.
* [x] Generated evidence records compare/select predicate boundary facts with
      mirror-only artifact metadata and runtime counts as execution cases only.
* [x] A migrated statement-plan or direct-provider non-regression dry-run still
      succeeds.
* [x] Bounded current-path scan shows no new legacy i32/source-front-door/
      descriptor/direct-C/source-export/ABI-string/artifact-name/script/
      common-EmitC/metadata/route-id authority leaks.
* [x] `git diff --check` passes.
* [x] `check-tianchenrv` passes, or an exact blocker is recorded.

## Out Of Scope

* New RVV operation families, new compare/select variants beyond `cmp_select`,
  predicate expansion, dtype/LMUL clone batches, source-front-door positive
  routes, frontend/Linalg routes, common EmitC RVV semantics,
  descriptor/direct-C/source-export computation, dashboards, broad smoke
  matrices, performance claims, non-RVV plugin work, dispatch/fallback changes,
  runtime ABI redesign, and changes that weaken owner-driven route construction
  or fail-closed diagnostics.

## Validation Plan

1. Validate and start this Trellis task.
2. Run a local generated-bundle dry-run for `cmp_select` using the pre-realized
   selected-body fixture and runtime counts `1,7,16,23,257`.
3. Run non-dry-run `ssh rvv` generated-bundle execution for the same route and
   counts.
4. Inspect generated evidence/artifacts enough to record direct route-entry
   realization, compare/select boundary facts, runtime ABI order, route metadata
   mirror fields, and emitted RVV predicate/select intrinsics.
5. Run a representative non-regression dry-run for an owner-driven path outside
   plain compare/select.
6. Run a bounded current-path authority scan over RVV planning/provider/target
   artifact/script/test surfaces.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`.

## Definition Of Done

The task either records real executable evidence on the existing production
path or repairs the exact production boundary needed to obtain it. PRD,
`implement.jsonl`, `check.jsonl`, Trellis status, journal, and final report are
truthful. Completed work is archived and committed as one coherent commit, or
the unfinished continuation point is explicit.

## Evidence Captured

No production compiler, target, script, or test repair was required. Current
HEAD already carries `cmp_select` through the direct pre-realized route-entry
compare/select path:

```text
selected pre-realized tcrv.exec RVV fixture
  -> tcrv_rvv typed_compare_select_pre_realized_body
  -> RVV selected-body route-entry realization bridge
  -> realized tcrv_rvv setvl/with_vl/load/compare/select/store body
  -> plain compare-select route-family plan
  -> elementwise/select operand-binding facts
  -> route-control provider plan
  -> migrated compare/select statement-plan owner
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
  --op-kind cmp_select \
  --runtime-count 1 --runtime-count 7 --runtime-count 16 \
  --runtime-count 23 --runtime-count 257 \
  --artifact-root artifacts/tmp/stage2_rvv_direct_cmp_select_executable_closure \
  --run-id pre-realized-cmp-select-dry \
  --overwrite \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate
```

The dry-run artifact is:

```text
artifacts/tmp/stage2_rvv_direct_cmp_select_executable_closure/pre-realized-cmp-select-dry
```

Non-dry-run `ssh rvv` generated-bundle execution for the chosen route passed:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --op-kind cmp_select \
  --runtime-count 1 --runtime-count 7 --runtime-count 16 \
  --runtime-count 23 --runtime-count 257 \
  --artifact-root artifacts/tmp/stage2_rvv_direct_cmp_select_executable_closure \
  --run-id pre-realized-cmp-select-ssh \
  --overwrite \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --timeout 180 \
  --connect-timeout 10
```

The `ssh rvv` run compiled on `riscv64` with `/usr/bin/clang` and printed:

```text
PASS op=cmp_select counts=1,7,16,23,257
```

Per-count correctness output covered:

```text
n=1   predicate_true_lanes=1   predicate_false_lanes=0
n=7   predicate_true_lanes=2   predicate_false_lanes=5
n=16  predicate_true_lanes=4   predicate_false_lanes=12
n=23  predicate_true_lanes=6   predicate_false_lanes=17
n=257 predicate_true_lanes=65  predicate_false_lanes=192
```

The generated evidence records:

* input mode: `pre-realized-selected-body`;
* selected input:
  `test/Target/RVV/pre-realized-selected-body-artifact-cmp-select.mlir`;
* selected variant: `pre_realized_body_rvv_cmp_select`;
* ABI prototype:
  `void tcrv_emitc_pre_realized_body_cmp_select_kernel_pre_realized_body_rvv_cmp_select(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);`
* runtime ABI order: `lhs,rhs,out,n`;
* route operand binding plan:
  `rvv-route-operand-binding:cmp_select.v1`;
* plain compare-select route family plan:
  `rvv-plain-compare-select-route-family-plan.v1`;
* provider-supported mirror:
  `provider_supported_mirror:rvv-plain-compare-select-plan-validated`;
* compare predicate kind: `eq`;
* predicate source: `compare-produced-mask-same-vl-scope`;
* predicate role: `predicate-mask-produced-by-compare`;
* select layout: `select-lhs-when-mask-else-rhs`;
* selected value operands: true value `lhs`, false value `rhs`, output `out`;
* runtime counts are execution cases, not predicate or route authority.

Materialized selected-body evidence shows the pre-realized body was consumed
into typed low-level RVV structure:

```text
tcrv_rvv.runtime_abi_value lhs/rhs/out/n
tcrv_rvv.setvl %n
tcrv_rvv.with_vl
tcrv_rvv.load lhs/rhs
tcrv_rvv.compare {kind = "eq"}
tcrv_rvv.select
tcrv_rvv.store
```

Generated EmitC/C++ evidence uses provider-derived RVV intrinsic calls:

```text
__riscv_vsetvl_e32m1
__riscv_vle32_v_i32m1
__riscv_vmseq_vv_i32m1_b32
__riscv_vmerge_vvm_i32m1
__riscv_vse32_v_i32m1
```

Direct-provider non-regression dry-run passed:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --dry-run \
  --pre-realized-selected-body \
  --op-kind computed_masked_strided_input_widening_dot_reduce_add \
  --runtime-count 1 --runtime-count 17 \
  --artifact-root artifacts/tmp/stage2_rvv_direct_cmp_select_executable_closure \
  --run-id direct-contraction-nonregression-dry \
  --overwrite \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate
```

The non-regression dry-run artifact is:

```text
artifacts/tmp/stage2_rvv_direct_cmp_select_executable_closure/direct-contraction-nonregression-dry
```

Focused compare/select lit coverage passed:

```bash
cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter=cmp-select .
```

Result: 10/10 selected tests passed.

Current-path authority scan:

```bash
rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|source-front-door|source-artifact|descriptor|__riscv_.*_i32m1" \
  include/TianChenRV/Plugin/RVV lib/Plugin/RVV/EmitC lib/Target/RVV \
  include/TianChenRV/Target/RVV scripts/rvv_generated_bundle_abi_e2e.py \
  test/Target/RVV test/Plugin/RVVExtensionPluginTest.cpp
```

reported existing provider-derived intrinsic mirrors, descriptor/source-front
door negative checks, and legacy fail-closed fixtures. No production files were
changed in this task.

Task-file authority scan over this task found only explicit non-goal,
fail-closed, or mirror-only wording.

Final checks:

* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-stage2-rvv-direct-cmp-select-executable-closure`
  passed before execution.
* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
* `git diff --check` passed.
* `cmake --build build --target check-tianchenrv -j2` passed: 379/379 tests.

## Spec Update Judgment

No `.trellis/spec/**` update is needed in this round. The executable closure
confirmed existing durable contracts for direct route-entry selected-body
realization, compare/select statement-plan ownership, common EmitC neutrality,
and generated-bundle compare/select predicate evidence. It did not introduce a
new API, ownership boundary, failure mode, or convention beyond the already
recorded specs.
