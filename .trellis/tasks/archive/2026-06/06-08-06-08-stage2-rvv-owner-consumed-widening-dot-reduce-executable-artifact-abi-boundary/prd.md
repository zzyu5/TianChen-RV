# Stage2 RVV owner-consumed widening dot-reduce executable artifact ABI boundary

## Goal

Prove or complete one production workflow submodule: the existing
pre-realized `widening_dot_reduce_add` selected-body route must flow through
owner-local direct-contraction aggregate statement-plan facts, route
construction, common EmitC materialization, target artifact export, generated
bundle creation, and `ssh rvv` correctness evidence. If current HEAD already
has the production path, this round closes the missing non-dry-run executable
evidence gap. If the path is broken, this round repairs only the
owner-consumed aggregate route, validation, and executable artifact ABI boundary
needed for this path.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository inspection started from clean `main`.
* Recent history includes `3835dc8b rvv: consume contraction aggregate facts
  through owner`, which moved direct-contraction aggregate provider-plan
  derivation and fact validation out of central `RVVEmitCRouteProvider.cpp`
  into statement-plan owner selection.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-contraction-aggregate-owner-consumed-boundary/`
  established dry-run positive and fail-closed evidence for the owner-consumed
  direct-contraction aggregate boundary.
* The current bottleneck is executable generated-artifact evidence for
  `widening_dot_reduce_add`, not selection of a new contraction family.
* Runtime/correctness claims must use real `ssh rvv` evidence.

## Requirements

* Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python remains support tooling only.
* Stay on the direct pre-realized `widening_dot_reduce_add` selected-body
  workflow. Do not choose a different contraction family.
* Preserve owner-local authority: typed `tcrv_rvv` body facts, aggregate/result
  facts, widening input/product/accumulator roles, runtime AVL/VL, ABI/header
  bindings, statement-plan owner selection, and route construction must line up
  without central-provider semantic authority.
* `RVVEmitCRouteProvider.cpp` must not regain direct calls to
  `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)` or
  `verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)`.
* Common EmitC/export may materialize provider-built routes, but must not infer
  RVV semantics, dtype/config, accumulator/result behavior, or intrinsic
  spelling.
* The generated-bundle evidence must check aggregate result correctness,
  widened product/accumulator behavior, source preservation, and tail
  preservation over representative runtime counts/patterns.
* A focused fail-closed check must prove a stale direct pre-realized route-entry
  shortcut or another stale aggregate/ABI fact is rejected with targeted
  diagnostics.
* If source changes are unnecessary, report exact no-source-change
  justification tied to current code and evidence.

## Acceptance Criteria

* [x] Current code inspection proves the direct-contraction aggregate route is
  owner-consumed before route construction, or a focused source diff repairs
  only that executable ABI seam.
* [x] Positive non-dry-run generated-bundle `ssh rvv` evidence passes for
  pre-realized `widening_dot_reduce_add` with representative counts and
  patterns.
* [x] The positive evidence records aggregate result, widened
  product/accumulator behavior, source preservation, tail preservation, expected
  selected variant/function/prototype, and no descriptor/direct-C/source-export
  authority.
* [x] A fail-closed direct pre-realized route-entry shortcut or stale
  aggregate/ABI fact remains rejected with targeted diagnostics.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant lit/dry-run generated-bundle tests for `widening_dot_reduce_add`
  pass.
* [x] Residue scan confirms `RVVEmitCRouteProvider.cpp` still has no direct
  calls to `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)` or
  `verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)`.
* [x] Bounded old-authority scan over touched files and added diff lines shows
  no new positive legacy `i32m1` route authority, source-front-door route
  authority, descriptor-driven compute, direct-C/source export authority, or
  mirror-only acceptance.
* [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean/reported.
* [x] Task status, workspace journal, archive, and one coherent commit are
  completed if the task finishes.

## Out Of Scope

* No broad contraction or MAcc matrix.
* No dtype/SEW/LMUL clone batch.
* No new broadcast, mask, reduction, conversion, or contraction family
  expansion.
* No high-level Linalg/Vector/StableHLO frontend work.
* No source-front-door positive route.
* No per-Linalg route authority.
* No artifact/report/dashboard-only completion.
* No common EmitC RVV semantic selection.
* No restoration of central-provider direct aggregate fact consumption.
* No route-id, metadata, helper-name, status, or mirror-field acceptance
  authority.

## Technical Notes

* Specs to use:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and the relevant testing contract sections
  for generated-bundle executable evidence.
* Previous task:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-contraction-aggregate-owner-consumed-boundary/`.
* Primary code paths:
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.
* Primary tests/fixtures:
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-dot-reduce-add-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-widening-dot-reduce-add-fail-closed.test`,
  relevant `test/Target/RVV/*widening-dot-reduce-add*.mlir`, and the C++ RVV
  plugin/target artifact tests.

## Completion Evidence

No compiler source change was needed. Inspection showed that
`RVVEmitCRouteProvider.cpp` reaches direct-contraction statements only through
`getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)`; it has no direct
call to `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)` or
`verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)`. Inside the
owner-selection module, the direct-contraction provider plan is derived from
the same analysis/materialization/math operand-binding facts, converted into
the statement plan, and verified before the provider attaches statements to
`TCRVEmitCLowerableRoute`.

Positive executable evidence:

```text
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --artifact-root artifacts/stage2-owner-consumed-widening-dot-reduce-executable \
  --run-id ssh-rvv-pre-realized-widening-dot-reduce-add \
  --overwrite \
  --op-kind widening_dot_reduce_add \
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
PASS op=widening_dot_reduce_add counts=0,1,16,17,257 patterns=0,1
```

The generated harness checked
`acc[0] + sum_i((int32_t)lhs[i] * (int32_t)rhs[i])`, positive and negative
widened products, seed contribution, scalar-result-only output, source
preservation, accumulator preservation, and tail/non-scalar sentinel
preservation. The evidence JSON recorded the selected prototype
`void tcrv_emitc_pre_realized_body_widening_dot_reduce_add_kernel_pre_realized_body_rvv_widening_dot_reduce_add(const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n);`,
runtime ABI order `lhs,rhs,acc,out,n`, `provider_supported_mirror:
rvv-contraction-family-plan-validated`, `widening_dot_reduction_boundary`,
target validator
`RVVTargetArtifactRouteFamilyValidation.cpp:widening-dot-reduction
target-owned consumer`, and mirror-only metadata role.

Focused fail-closed evidence:

```text
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --dry-run \
  --pre-realized-selected-body \
  --direct-pre-realized-route-entry \
  --artifact-root artifacts/stage2-owner-consumed-widening-dot-reduce-executable \
  --run-id fail-closed-direct-pre-realized-widening-dot-reduce-add \
  --overwrite \
  --op-kind widening_dot_reduce_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 16 \
  --runtime-count 23 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: failed
--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): widening_dot_reduce_add; the direct route-entry shortcut is retired and these fixtures must use the public selected lowering-boundary producer before target bundle export
```

Focused checks:

```text
build/bin/tianchenrv-rvv-extension-plugin-test
build/bin/tianchenrv-target-artifact-export-test
cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter "Scripts/rvv-generated-bundle-abi-e2e-(pre-realized-widening-dot-reduce-add-dry-run|direct-pre-realized-widening-dot-reduce-add-fail-closed)"
```

Results:

```text
RVV extension plugin smoke test passed
tianchenrv-target-artifact-export-test exit code 0
lit: Passed 2, Excluded 529
```
