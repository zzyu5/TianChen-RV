# RVV i32m2 selected VL dispatch ssh evidence

## Goal

Complete one bounded runtime-evidence module proving that the non-m1 RVV
`i32-vsub` / `i32m2` selected-config path survives the active TianChen-RV front
door, plugin-owned selected-config and VL dataflow materialization, target
artifact export, RVV+scalar dispatch wrapper generation, and real `ssh rvv`
compile/link/run correctness checks.

## Background

The previous archived task
`.trellis/tasks/archive/2026-05/05-11-rvv-selected-config-vl-dataflow-materialization/`
completed the selected-config-to-VL-dataflow bridge in C++ and proved
contract-driven i32m2/i64m1 materialization with local compiler tests. It did
not collect `ssh rvv` evidence. Earlier i64m1 add/sub/mul tasks already proved
front-door dispatch runtime evidence for the m1 family.

The current bottleneck is the non-m1 i32m2 path. This task must prove that
`i32-vsub` / `i32m2` is not metadata-only, dry-run-only, or a unit-test helper:
it must reach a generated RVV+scalar dispatch artifact whose RVV branch uses the
selected i32m2 RVV intrinsic family and whose scalar branch uses int32
subtraction, then run correctly on the RVV machine.

## Requirements

- Start from the existing bounded `i32-vsub` / `i32m2` front-door or generic
  route fixture, not an unrelated hand-written route.
- Preserve the active compiler ownership chain:
  - bounded linalg or existing exec front-door input creates
    `tcrv.exec.kernel`, `mem_window`, and runtime `n` ABI boundaries;
  - RVV plugin proposes and materializes the selected
    `RVVBinarySelectedConfigContract`;
  - `RVVBinaryVLDataflowMaterialization` drives the `tcrv_rvv.setvl`,
    `tcrv_rvv.with_vl`, i32 load/sub/store body;
  - scalar plugin materializes the matching i32-vsub fallback callable;
  - target/export code emits an RVV+scalar dispatch wrapper from selected RVV
    case plus scalar fallback, including the dispatch availability guard.
- Generated RVV source must show the i32m2 selected shape, including SEW 32,
  LMUL m2, tail/mask policy, selected vector type/suffix, setvl suffix,
  `__riscv_vsetvl_e32m2`, i32m2 vector loads/stores, and
  `__riscv_vsub_vv_i32m2` or the repository's exact current spelling.
- Generated scalar fallback must use int32 ABI types and
  `lhs[index] - rhs[index]`, not i64, add, mul, scalar-only, direct-only,
  stale i32m1, or stale vadd/vmul output.
- Dispatch evidence must exercise both branches with host-provided
  availability guard values and runtime counts including at least one
  non-trivial count that is not an obvious multiple-of-VL case.
- Evidence JSON must record bounded, sanitized fields: selected kernel,
  selected variant, selected family, selected vector shape, SEW, LMUL,
  tail/mask policy, descriptor-local operator metadata, runtime element-count
  / AVL boundary, dispatch guard, wrapper symbol, RVV callable, scalar
  callable, remote architecture, remote clang, commands, counts exercised,
  branch coverage, pass/fail result, and artifact paths.
- If the route bypasses selected-config/VL dataflow, fix the C++/MLIR/plugin/
  target/export path. Do not close this task by patching Python checks or
  FileCheck labels around a stale route.
- Keep generic core passes extension-neutral. No RVV/scalar/dtype/family/LMUL
  branch may be added to shared generic orchestration.
- Keep `tcrv.exec` compute-free. Compute semantics and executable bodies stay
  in extension dialects/plugins/target exporters.

## Acceptance Criteria

- A local focused dry-run for `i32-vsub` / `i32m2` dispatch route succeeds and
  verifies selected config, i32m2 intrinsic spelling, scalar subtract fallback,
  dispatch wrapper, component metadata, and absence of stale i32m1/i64/add/mul/
  direct-only regressions.
- Real `ssh rvv` evidence compiles, links, and runs the generated dispatch
  wrapper path, validates deterministic subtract outputs for both scalar and
  RVV branches, and writes a successful evidence JSON under `artifacts/tmp`.
- A focused lit/FileCheck regression prevents silent regression to dry-run-only,
  scalar-only, direct microkernel-only, stale i32m1, stale i64m1, add, or mul
  output.
- Focused C++/lit tests pass for the touched RVV binary planning/materialization,
  selected lowering boundary, target artifact export, dispatch script, and
  script-level evidence behavior.
- `git diff --check` passes.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  is run if practical after focused checks pass.
- Trellis task validation passes before finishing, and the task is archived only
  after real `ssh rvv` runtime correctness evidence exists for the i32m2
  dispatch wrapper path.

## Non-Goals

- No generic RVV backend.
- No broad add/sub/mul or dtype/LMUL evidence matrix.
- No performance, throughput, latency, or benchmark claim.
- No MLIR vector/scalable-vector lowering route.
- No arbitrary dtype, reductions, predication semantics, tensor/tile IR, or new
  high-level compute ops.
- No compute semantics in `tcrv.exec`.
- No compiler internals in Python. Python may only orchestrate runners,
  artifacts, redaction, and evidence checks.
- No docs-only, helper-only, smoke-only, report-only, metadata-only, local-only,
  or dry-run-only closeout.

## Validation Plan

- Inspect and, where needed, repair the active i32-vsub/i32m2 fixture and route
  through the current `scripts/rvv_scalar_dispatch_e2e.py` or the smallest
  matching runner extension.
- Build focused targets before lit/script runs:
  `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-selected-lowering-boundary-test`,
  `tianchenrv-rvv-extension-plugin-test`, and target/export tests as needed.
- Run local dry-run and FileCheck coverage for generated source/evidence.
- Run real `ssh rvv` compile/link/run/output validation for the same dispatch
  wrapper path.
- Run focused lit/C++ tests covering changed compiler/export/script behavior.
- Run `git diff --check`.
- Run full `check-tianchenrv -j2` if practical.
- Validate the Trellis task, finish/archive it only when the evidence module is
  complete, and create one coherent commit.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/validation/experiment-reference.md`.
- Previous PRDs read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-selected-config-boundary-family-contract/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-11-rvv-selected-config-vl-dataflow-materialization/prd.md`.
- Primary code surfaces to inspect:
  selected-config contract, RVV binary microkernel materialization, RVV binary
  planning, selected emission planning, RVV target exporter, RVV+scalar dispatch
  exporter, `rvv_microkernel_e2e.py`, `rvv_scalar_dispatch_e2e.py`, and the
  existing i32m2/i32-vsub lit fixtures.

## Definition Of Done

The task is done when the active front-door i32-vsub/i32m2 dispatch route
produces compiler-generated RVV+scalar dispatch artifacts from selected-config
VL dataflow, local focused tests prevent stale route regressions, real `ssh rvv`
evidence proves both scalar and RVV branch subtract correctness, generic core
boundaries remain extension-neutral, Trellis validates and archives the task,
and one coherent commit records the module.

If unfinished, leave the task open and record the exact continuation point:
front-door fixture selection, selected-config contract production, VL dataflow
materialization, target source export, scalar fallback export, dispatch wrapper
ABI, runner branch exercise, remote compile/link/run, or numeric output
validation.

## Completion Notes

- The active i32-vsub/i32m2 route already preserved the selected-config to VL
  dataflow bridge in C++: the plan-and-export front door emits
  `frontend_bundle_i32m2_vsub`, selected i32m2 metadata, `RVVBinarySelectedConfig`
  facts, and descriptor-local i32-vsub metadata into the dispatch bundle.
- Generated RVV dispatch source contains the selected i32m2 contract facts
  (`shape=i32m2`, `sew=32`, `lmul=m2`, tail/mask agnostic,
  `vint32m2_t`, suffix `i32m2`, setvl suffix `e32m2`) and uses
  `__riscv_vsetvl_e32m2`, `__riscv_vle32_v_i32m2`,
  `__riscv_vsub_vv_i32m2`, and `__riscv_vse32_v_i32m2`.
- Generated scalar fallback remains int32 and uses
  `out[index] = lhs[index] - rhs[index];`.
- Dispatch ABI remains target/export-owned:
  `void tcrv_dispatch_i32_vsub_frontend_bundle_i32m2_vsub(const int32_t *lhs,
  const int32_t *rhs, int32_t *out, size_t n, int rvv_available)`.
- Real `ssh rvv` evidence was collected at
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-20260511-i32m2-vsub-bundle-ssh/evidence.json`.
  It compiled, linked, and ran both the generated-source object and selected
  bundle object on `ssh rvv`.
- Remote facts recorded in evidence: `architecture=riscv64`,
  `clang_path=/usr/bin/clang`, `clang_version=Ubuntu clang version 18.1.3
  (1ubuntu1)`, compile flags `-O2 -march=rv64gcv -mabi=lp64d`, link flags
  `-O2 -march=rv64gcv -mabi=lp64d -no-pie`.
- Branch coverage recorded `rvv_available=0` for scalar fallback and
  `rvv_available=1` for RVV dispatch case, each with runtime element counts
  `7` and `16`. Output validation observed
  `tcrv_rvv_scalar_i32_vsub_bundle_external_abi_ok runtime_counts=7,16
  branches=scalar_and_rvv` for both source-built and bundle-object runs.
- Strengthened the existing i32m2 bundle lit regression so it now checks
  selected binary config, runtime `n`, dispatch guard `rvv_available`,
  descriptor-local i32-vsub operator metadata, index ABI records, scalar
  subtract source, branch evidence, and absence of runtime/performance claims in
  local dry-run evidence.
- No generic RVV backend, performance claim, benchmark claim, Python compiler
  implementation, or compute semantics in `tcrv.exec` were added.

## Checks Run

- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run
  --arithmetic-family=i32-vsub --vector-shape=i32m2
  --lower-linalg-frontend --run-id codex-20260511-i32m2-vsub-dispatch-dry
  --overwrite --expect-selected-kernel frontend_dispatch_i32m2_vsub`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run
  --use-target-artifact-bundle --use-plan-and-export-bundle-front-door
  --arithmetic-family=i32-vsub --vector-shape=i32m2
  --run-id codex-20260511-i32m2-vsub-bundle-dry --overwrite
  --expect-selected-kernel frontend_bundle_i32m2_vsub`
- `python3 scripts/rvv_scalar_dispatch_e2e.py
  --use-target-artifact-bundle --use-plan-and-export-bundle-front-door
  --arithmetic-family=i32-vsub --vector-shape=i32m2
  --run-id codex-20260511-i32m2-vsub-bundle-ssh --overwrite
  --expect-selected-kernel frontend_bundle_i32m2_vsub --timeout 180
  --connect-timeout 20`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-selected-lowering-boundary-test
  tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test
  -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-i32m2-generic-route.mlir
  Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir
  Scripts/rvv-scalar-dispatch-e2e.test
  Scripts/rvv-scalar-dispatch-bundle-e2e.test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Dialect/RVV/i32m2-dataflow.mlir Scripts/rvv-microkernel-e2e.test`
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  200/200 lit tests passed.
