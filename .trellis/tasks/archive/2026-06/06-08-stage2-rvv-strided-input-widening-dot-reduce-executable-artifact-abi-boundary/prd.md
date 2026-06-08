# Stage2 RVV strided-input widening dot-reduce executable artifact ABI boundary

## Goal

Complete one bounded production workflow submodule: the existing base
`strided_input_widening_dot_reduce_add` selected-body route must line up from
typed `tcrv_rvv` body facts through RVV plugin-owned stride/dot/reduction route
validation, common EmitC materialization, target artifact export, generated
bundle ABI, and `ssh rvv` correctness evidence. If current production code
already has the executable path, this round closes the non-dry-run evidence gap
and records the exact no-source-change justification. If inspection finds a
stale or under-validated executable boundary, this round repairs only that
strided-input widening dot-reduce artifact/ABI seam.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief and started as the current Trellis task.
* Repository inspection started from clean `main` at commit `53870e34 rvv:
  archive widening dot executable evidence`.
* The previous archived task proved base `widening_dot_reduce_add` generated
  bundle execution on `ssh rvv`, including signed horizontal dot, seed
  contribution, widened product distinction, scalar-output-only behavior,
  source/accumulator preservation, tail preservation, and route-provider
  ownership scan.
* Existing strided-input fixtures and script support already name
  `strided_input_widening_dot_reduce_add`, including explicit selected-body,
  pre-realized selected-body, and direct-pre-realized fail-closed dry-run tests.
* The added responsibility in this round is the strided-input ABI/memory
  boundary: `lhs_stride`, `rhs_stride`, strided load memory form, stride source
  facts, widened dot/reduction facts, accumulator/seed role, scalar result role,
  dtype/config/policy, runtime AVL/VL, header/prototype order, and generated
  bundle correctness must agree without route-id, metadata, helper-name,
  test-name, or common EmitC semantic authority.

## Requirements

* Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python remains support tooling only.
* Stay on the base strided-input widening dot-reduce add selected-body route.
  Do not expand to computed-masked or computed-masked-strided dot-reduce unless
  the base seam is already complete and the same coherent executable boundary
  naturally fits this round.
* Preserve authority placement: typed `tcrv_rvv` body/config/runtime facts and
  RVV plugin-owned owner/provider validation define route support; common
  EmitC/export only materializes and packages provider-built facts.
* Treat route ids, artifact names, emission-plan metadata, status/result fields,
  helper names, test names, and generated evidence files as mirrors only.
* Validate or prove fail-closed handling for stale/missing strided-input
  executable boundary facts, especially stride ABI binding, strided memory
  form, widened reduction facts, accumulator/seed role, scalar result type,
  header/prototype binding, route-family validation contract, C type mapping,
  ABI order, runtime AVL/VL, and statement facts.
* Positive executable evidence must run generated artifacts on real `ssh rvv`
  before claiming runtime correctness.
* The generated harness/evidence must distinguish true strided-input behavior
  from unit-stride dot-reduce behavior, and must check source, accumulator, and
  tail/sentinel preservation.
* Keep the task bounded: no broad dot-reduce matrix, no dtype/LMUL clone batch,
  no MAcc/product-reduce/dequant/clamp rework, no high-level frontend, no
  performance tuning database, and no source-front-door positive route.

## Acceptance Criteria

* [x] Current code inspection proves the strided-input widening dot-reduce
  route is owner/provider-consumed before route construction, or a focused
  source diff repairs only that executable artifact/ABI seam.
* [x] Positive generated-bundle evidence covers materialized selected boundary,
  emission plan, target artifact export, generated bundle compile, and `ssh rvv`
  correctness for `strided_input_widening_dot_reduce_add`.
* [x] Positive evidence records and checks `lhs_stride`/`rhs_stride` ABI order,
  stride-source facts, strided memory form, widened dot/reduction facts,
  accumulator/seed contribution, scalar output behavior, dtype/config/policy,
  runtime AVL/VL, header/prototype binding, and source/accumulator/tail
  preservation.
* [x] A focused fail-closed check rejects at least one stale direct route-entry,
  stale stride binding, stale strided memory form, stale widened reduction fact,
  stale accumulator/seed role, stale scalar result type, stale header/prototype,
  or stale route-family validation contract.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant generated-bundle dry-run lit tests for explicit/pre-realized
  strided-input widening dot-reduce and direct-pre-realized fail-closed behavior
  pass.
* [x] Bounded old-authority scan over touched files and added diff lines shows
  no new positive legacy `i32m1` route authority, source-front-door route
  authority, descriptor-driven compute, direct-C/source export authority, or
  mirror-only acceptance.
* [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean/reported.
* [x] Task status, workspace journal, archive, and one coherent commit are
  completed if the task finishes.

## Out Of Scope

* No broad dot-reduce or contraction matrix.
* No computed-masked or computed-masked-strided expansion as a required success
  condition.
* No dtype/SEW/LMUL clone batch.
* No MAcc/product-reduce/dequant/clamp rework except bounded reference reading.
* No high-level Linalg/Vector/StableHLO frontend work.
* No source-front-door positive route.
* No per-Linalg route authority.
* No artifact/report/dashboard-only completion.
* No common EmitC RVV semantic selection.
* No route-id, metadata, helper-name, status, result, or mirror-field
  acceptance authority.

## Technical Notes

* Specs read or queued for this task:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Previous completed task:
  `.trellis/tasks/archive/2026-06/06-08-06-08-stage2-rvv-owner-consumed-widening-dot-reduce-executable-artifact-abi-boundary/`.
* Primary code paths to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.
* Primary fixtures/tests to inspect:
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-strided-input-widening-dot-reduce-add-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-strided-input-widening-dot-reduce-add-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-strided-input-widening-dot-reduce-add-fail-closed.test`,
  `test/Target/RVV/explicit-selected-body-artifact-strided-input-widening-dot-reduce-add.mlir`, and
  `test/Target/RVV/pre-realized-selected-body-artifact-strided-input-widening-dot-reduce-add.mlir`.

## Completion Evidence

No compiler source change was needed. Inspection showed the existing production
strided-input widening dot-reduce route already derives and verifies the
executable boundary through RVV-owned facts:

* `validatePreRealizedRVVSelectedStridedInputWideningDotReduceBody(...)`
  requires op kind `signed_widening_dot_reduce_add`, memory form
  `strided-input-widening-dot-reduce`, `stride_unit = "element"`, supported
  source/result config/relation facts, runtime ABI roles for `lhs`, `rhs`,
  `acc`, `out`, `n`, `lhs_stride`, and `rhs_stride`, and a body shape limited
  to `StridedLoadOp`, `WideningDotReduceOp`, and `StoreOp`.
* `deriveRVVSelectedBodyContractionRouteFamilyPlan(...)` appends
  `lhs_stride` and `rhs_stride` into the runtime ABI parameter list when the
  operation uses strided inputs, sets the strided memory layout, stride-source
  mirrors, source memory form, destination memory form, and strided-load
  intrinsic from the validated family plan, then validates the plan.
* `deriveRVVSelectedBodyContractionRouteOperandBindingPlan(...)` binds
  `lhs_stride` and `rhs_stride` with provider-owned `abi|str|addr|hdr` tokens
  in the same runtime ABI order used by the generated header/prototype.
* `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` reaches the direct
  contraction statement plan owner before route construction, and the owner
  verifies strided-input math operand-binding facts, exact stride ABI roles,
  and the presence of the strided source-load leaf before creating the
  `TCRVEmitCLowerableRoute`.
* `RVVTargetArtifactRouteFamilyValidation.cpp` requires candidate metadata for
  `tcrv_rvv.strided_memory_layout`, `tcrv_rvv.lhs_stride_source`,
  `tcrv_rvv.rhs_stride_source`, `tcrv_rvv.source_memory_form`,
  `tcrv_rvv.destination_memory_form`, and `tcrv_rvv.strided_load_intrinsic` to
  mirror the provider route validation contract exactly.
* `RVVEmitCRouteProvider.cpp` still has no direct calls to
  `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)` or
  `verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)`.

Positive dry-run generated-bundle evidence:

```text
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --dry-run \
  --pre-realized-selected-body \
  --artifact-root artifacts/stage2-strided-input-widening-dot-reduce-executable \
  --run-id dry-run-pre-realized-strided-input-widening-dot-reduce-add \
  --overwrite \
  --op-kind strided_input_widening_dot_reduce_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 16 \
  --runtime-count 17 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/bin/llvm-readobj-20
```

Result:

```text
rvv_generated_bundle_abi_e2e: dry_run_success
```

Positive non-dry-run executable evidence:

```text
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --artifact-root artifacts/stage2-strided-input-widening-dot-reduce-executable \
  --run-id ssh-rvv-pre-realized-strided-input-widening-dot-reduce-add \
  --overwrite \
  --op-kind strided_input_widening_dot_reduce_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 16 \
  --runtime-count 17 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/bin/llvm-readobj-20 \
  --timeout 180 --connect-timeout 20
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
ssh_evidence: true
remote_arch=riscv64
clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)
PASS op=strided_input_widening_dot_reduce_add counts=0,1,16,17,257 stride_pairs=2:3,3:2 data_patterns=2
```

The generated harness ran both `lhs_stride=2, rhs_stride=3` and
`lhs_stride=3, rhs_stride=2` cases for counts `0,1,16,17,257`. The oracle used
`lhs[i * lhs_stride] * rhs[i * rhs_stride]`, checked signed widened i16*i16
products through i32 horizontal reduction with `acc[0]` seed contribution,
ensured skipped source elements were ignored, and verified scalar-output-only
behavior plus tail sentinel preservation. The evidence recorded prototype

```text
void tcrv_emitc_pre_realized_strided_dot_kernel_rvv_strided_input_dot(const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n, size_t lhs_stride, size_t rhs_stride);
```

and runtime ABI order:

```text
lhs,rhs,acc,out,n,lhs_stride,rhs_stride
```

with route operand binding summary:

```text
rvv-route-operand-binding:strided_widening_dot_reduce.v1;lhs=lhs-input-buffer:lhs:abi|sld|dot-lhs|i16|hdr;rhs=rhs-input-buffer:rhs:abi|sld|dot-rhs|i16|hdr;acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;out=output-buffer:out:abi|store|i32|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr;lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr;rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr
```

Focused fail-closed evidence:

```text
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --dry-run \
  --pre-realized-selected-body \
  --direct-pre-realized-route-entry \
  --artifact-root artifacts/stage2-strided-input-widening-dot-reduce-executable \
  --run-id fail-closed-direct-pre-realized-strided-input-widening-dot-reduce-add \
  --overwrite \
  --op-kind strided_input_widening_dot_reduce_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 16 \
  --runtime-count 23 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/bin/llvm-readobj-20
```

Result:

```text
rvv_generated_bundle_abi_e2e: failed
--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): strided_input_widening_dot_reduce_add; the direct route-entry shortcut is retired and these fixtures must use the public selected lowering-boundary producer before target bundle export
```

Focused checks:

```text
build/bin/tianchenrv-rvv-extension-plugin-test
build/bin/tianchenrv-target-artifact-export-test
cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter "Scripts/rvv-generated-bundle-abi-e2e-(explicit-strided-input-widening-dot-reduce-add-dry-run|pre-realized-strided-input-widening-dot-reduce-add-dry-run|direct-pre-realized-strided-input-widening-dot-reduce-add-fail-closed)"
```

Results:

```text
RVV extension plugin smoke test passed
tianchenrv-target-artifact-export-test exit code 0
lit: Passed 3, Excluded 528
```

No `.trellis/spec/` update was needed. The existing RVV plugin, EmitC route,
and MLIR testing specs already define typed-body authority, provider-owned
route/operand binding, target artifact mirror validation, generated-bundle
evidence, preservation checks, and `ssh rvv` requirements used by this
closeout. The generated artifact directory was validation output only and was
not committed.
