# RVV i64m1 vsub front-door dispatch ssh evidence

## Goal

Prove the bounded `i64-vsub` / `i64m1` linalg front-door route consumes the
shared finite binary frontend contract and reaches a generated RVV plus scalar
dispatch artifact bundle with real `ssh rvv` runtime output validation. This
is the second i64 operator-family proof after `i64-vadd`; it must show the
route is not overfit to add semantics and does not collapse to scalar-only,
direct RVV-only, i32, add, or multiply artifacts.

## What I Already Know

- The repository root is `/home/kingdom/phdworks/TianchenRV`.
- The worktree was clean before task setup; HEAD was
  `25c094a feat(rvv): add finite binary frontend contract`.
- No current Trellis task existed when this round started.
- The previous finite frontend task is archived at
  `.trellis/tasks/archive/2026-05/05-11-finite-binary-frontend-lowering-contract/`
  and must not be reopened.
- The previous i64-vadd front-door dispatch runtime ABI task is archived at
  `.trellis/tasks/archive/2026-05/05-11-rvv-i64m1-frontdoor-dispatch-runtime-abi/`
  and proved real `ssh rvv` dispatch-wrapper evidence for add only.
- `FiniteBinaryFrontendLoweringDescriptor` already includes `i64-vsub` with
  int64 ABI spellings and `-` operator metadata.
- RVV family planning already registers `i64-vsub` with `i64m1`,
  `__riscv_vsub_vv_`, and i64-vsub route/runtime ABI names.
- Scalar and RVV+scalar dispatch descriptors derive finite i64 sub route names,
  scalar fallback route names, dispatch wrapper route names, and subtraction
  operator spelling from the shared family descriptors.
- Existing fixtures and script surfaces already mention
  `plan-linalg-i64-vsub-and-export-target-artifact-bundle.mlir` and
  `scripts/rvv_scalar_dispatch_e2e.py --arithmetic-family=i64-vsub`.

## Requirements

- Use the existing shared `FiniteBinaryFrontendLoweringDescriptor` and RVV
  binary family planning contract; do not add generic core special cases for
  `i64-vsub`.
- Preserve plugin and target ownership:
  RVV intrinsic emission stays in RVV target code, scalar fallback emission
  stays scalar/target-owned, and dispatch wrapper generation stays
  target/export-owned.
- Preserve `tcrv.exec` as compute-free execution organization. It may carry
  kernel, target, capability, variant, mem_window, runtime_param, dispatch,
  fallback, and diagnostics; it must not express subtraction compute semantics.
- Validate operator semantics from generated artifacts:
  RVV code must contain `__riscv_vsub_vv_i64m1`, scalar fallback must compute
  `lhs - rhs`, and the dispatch wrapper must exercise both RVV and scalar
  branches through the external ABI.
- Preserve parameter layering:
  hardware and selected vector-shape facts remain compile-time
  capability/metadata, `n` and `rvv_available` remain runtime ABI/control
  values, and descriptor-local family/operator metadata remains separate from
  runtime evidence.
- Python changes, if any, are limited to runner/evidence orchestration. Python
  must not implement compiler internals, dialects, lowering, emission, or the
  capability model.

## Acceptance Criteria

- A local dry-run succeeds with:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vsub --expect-selected-kernel=frontend_bundle_i64_vsub --run-id <id> --overwrite --timeout 120`.
- The dry-run evidence and generated bundle show selected kernel
  `frontend_bundle_i64_vsub`, selected variant `rvv_first_slice`, vector shape
  `i64m1`, int64 ABI parameters, `__riscv_vsub_vv_i64m1`, scalar fallback
  subtraction, dispatch wrapper source/header/object records, and dispatch
  external ABI metadata.
- Real `ssh rvv` compile/link/run succeeds for the same plan-and-export
  dispatch wrapper path, exercises `rvv_available=0` and `rvv_available=1`,
  validates deterministic `lhs - rhs` output for runtime counts 7 and 16, and
  records evidence JSON with remote architecture, remote clang, selected
  kernel, selected variant, wrapper symbol, RVV intrinsic, scalar fallback
  symbol, branches exercised, and pass/fail result.
- Focused lit/FileCheck coverage proves the i64-vsub front-door dispatch
  bundle cannot silently become i64-vadd, i64-vmul, i32, scalar-only,
  direct-selected fixture-only, or direct microkernel-only output.
- Focused build/test checks for touched code or scripts pass.
- `git diff --check` passes.
- `check-tianchenrv` is run if practical after focused checks pass.
- The Trellis task validates and is archived only after real `ssh rvv`
  evidence exists for the i64-vsub dispatch wrapper path.

## Out Of Scope

- Generic RVV backend, arbitrary vector lowering, arbitrary linalg lowering,
  reductions, tensors, matmul, or new element types.
- Broad add/sub/mul matrix, benchmarks, throughput, latency, or performance
  claims.
- Hand-written runtime artifacts as source of truth.
- RVV/scalar semantic branches in generic core passes.
- Compute semantics in `tcrv.exec`.
- Docs-only, helper-only, dry-run-only, or report-only completion.

## Technical Notes

- Specs read for this task:
  `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/validation/experiment-reference.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Archived PRDs read:
  `.trellis/tasks/archive/2026-05/05-11-finite-binary-frontend-lowering-contract/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-11-rvv-i64m1-frontdoor-dispatch-runtime-abi/prd.md`.
- Primary code surfaces inspected:
  `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h`,
  `lib/Transforms/LowerLinalgRVVBinaryToExec.cpp`,
  `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`,
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`,
  `include/TianChenRV/Target/RVVScalarBinaryFamily.h`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  `scripts/rvv_scalar_dispatch_e2e.py`,
  `scripts/rvv_microkernel_e2e.py`,
  `test/Transforms/LinalgToExec/linalg-i64-vsub-to-rvv-artifact.mlir`,
  `test/Target/TargetArtifactBundleExport/plan-linalg-i64-vsub-and-export-target-artifact-bundle.mlir`,
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`, and
  `test/Scripts/rvv-microkernel-bundle-e2e.test`.

## Definition Of Done

The task is done when the bounded `i64-vsub` / `i64m1` front-door path exports
a coherent RVV+scalar dispatch bundle through the existing compiler/target
ownership boundaries, the focused local checks pass, real `ssh rvv` evidence
validates the dispatch wrapper and subtraction output on both branches, the
Trellis task validates and archives, and one coherent commit records the work.
If unfinished, leave the task open and record the exact blocker category,
failing command, artifact path, and source file/function for continuation.

## Completion Notes

Completed in this round:

- Created this Trellis task and PRD from the Hermes brief, with implement/check
  context limited to relevant `.trellis/spec/` files.
- Confirmed the active C++/MLIR/target stack already routes `i64-vsub` through
  the shared finite frontend descriptor, RVV i64m1 family planning, scalar
  fallback descriptor, and RVV+scalar dispatch bundle exporter without adding
  generic core special cases.
- Strengthened the focused script lit coverage for the i64-vsub
  plan-and-export dispatch bundle. The regression now requires
  `--expect-selected-kernel=frontend_bundle_i64_vsub` and checks generated
  dispatch source for selected kernel, `rvv_first_slice`, i64m1 vector metadata,
  `__riscv_vsub_vv_i64m1`, scalar `lhs - rhs`, int64 ABI parameters, runtime
  guard linkage, scalar fallback linkage, and the dispatch wrapper symbol.
- Produced local dry-run and real `ssh rvv` evidence for the same
  plan-and-export front-door dispatch wrapper path.

Runtime evidence:

- Evidence path:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-i64-vsub-frontdoor-dispatch-ssh/evidence.json`.
- Result: `status=success`, `pass_fail_result=pass`,
  `ssh_evidence_verified=true`.
- Remote target: `ssh_target=rvv`, architecture `riscv64`, clang
  `/usr/bin/clang`, clang version `Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Dispatch wrapper:
  `tcrv_dispatch_i64_vsub_frontend_bundle_i64_vsub(const int64_t *lhs, const int64_t *rhs, int64_t *out, size_t n, int rvv_available)`.
- Selected RVV callable:
  `tcrv_rvv_i64_vsub_microkernel_frontend_bundle_i64_vsub_rvv_first_slice`.
- Scalar fallback callable:
  `tcrv_scalar_i64_vsub_microkernel_frontend_bundle_i64_vsub_scalar_fallback_first_slice`.
- RVV intrinsic: `__riscv_vsub_vv_i64m1`.
- Compile flags: `-O2 -march=rv64gcv -mabi=lp64d`; link flags include
  `-no-pie`.
- The generated external caller checked runtime counts `7` and `16`, exercised
  `rvv_available=0` and `rvv_available=1`, validated `lhs - rhs`, and checked
  output overrun preservation. Both source-built and bundle-object linked
  executables printed
  `tcrv_rvv_scalar_i64_vsub_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.

Validation:

- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`: passed.
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run
  --use-target-artifact-bundle --use-plan-and-export-bundle-front-door
  --arithmetic-family=i64-vsub
  --expect-selected-kernel=frontend_bundle_i64_vsub --run-id
  codex-i64-vsub-frontdoor-dispatch-dry --overwrite --timeout 120`: passed.
- `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle
  --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vsub
  --expect-selected-kernel=frontend_bundle_i64_vsub --ssh-target rvv --run-id
  codex-i64-vsub-frontdoor-dispatch-ssh --overwrite --timeout 120`: passed.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Scripts/rvv-scalar-dispatch-bundle-e2e.test` from
  `artifacts/tmp/tianchenrv-build/test`: 1/1 passed.
- `git diff --check`: passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2`: 200/200 lit tests passed.

No performance or throughput claim is made. The runtime claim is limited to the
exact bounded `i64-vsub` / `i64m1` RVV+scalar dispatch wrapper evidence above.
