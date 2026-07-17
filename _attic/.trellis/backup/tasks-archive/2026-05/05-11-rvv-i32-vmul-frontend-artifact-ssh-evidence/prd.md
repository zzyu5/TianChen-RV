# RVV i32-vmul frontend-to-artifact ssh evidence slice

## Goal

Complete one bounded RVV `i32-vmul` compiler/evidence slice from the existing
marked finite `linalg.generic` frontend input through the real C++/MLIR
TianChen-RV path, selected RVV variant and selected lowering boundary,
route-metadata-preflighted source/header/object artifact export, and fresh
`ssh rvv` compile/run self-check evidence.

This is one family proof, not a broad matrix. If the current path already
works, the task should add only narrow missing regression/evidence records. If a
step fails because active C++/MLIR behavior is missing or stale, the task should
make the smallest plugin/target/transform-owned fix needed for the same
`i32-vmul` path.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state was clean on `main` at
  `10b0b6d feat(scalar): cover finite source route metadata`.
- No `.trellis/.current-task` existed before this task was created.
- The immediately previous scalar finite source route metadata task completed
  and archived at
  `.trellis/tasks/archive/2026-05/05-11-05-11-scalar-finite-source-route-metadata-coverage/`.
- The prior RVV finite source route metadata task generalized
  descriptor-derived route metadata and generic source-route preflight for all
  finite RVV source routes.
- Older archived ssh-evidence tasks already proved adjacent paths such as
  `i32-vmul` RVV+scalar dispatch bundle evidence, `i32m2` `i32-vsub` linalg
  bundle evidence, and i64 add/sub/mul frontend or dispatch evidence.
- The current task intentionally returns to the direct RVV `i32-vmul`
  frontend-to-artifact microkernel slice and must not reopen archived tasks.

## Requirements

- Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Python may only orchestrate existing compiler tools, artifact parsing,
  evidence capture, and remote compile/run checks. It must not implement
  compiler core, dialects, capability model, variant selection, lowering,
  emission, or route selection.
- Start from the existing finite marked `linalg.generic` `i32-vmul` frontend
  fixture and run the real lowering/planning/export path rather than using
  hand-authored final artifact text as the source of truth.
- Preserve the architecture boundary: `tcrv.exec` remains compute-free; RVV
  multiply dataflow and C intrinsic emission stay plugin/target-owned.
- Selected RVV family, dtype, operator, lowering descriptor, selected vector
  shape, selected SEW/LMUL/tail/mask policy, runtime ABI, and runtime
  element-count metadata must be visible in generated IR or artifact evidence
  and must agree with the finite `i32-vmul` descriptor.
- Route metadata preflight from recent RVV/scalar metadata rounds must be
  exercised before `i32-vmul` source emission. Stale route/runtime ABI or
  selected-plan metadata must fail closed in focused coverage unless existing
  coverage already proves the same route.
- The source artifact must include the RVV C intrinsic path through
  `riscv_vector.h` and `__riscv_vmul_vv_i32m1` or the selected-shape-correct
  `i32-vmul` intrinsic, not generic pseudo-intrinsics or Python-generated
  compiler IR.
- Header/object export for the same selected `i32-vmul` family should be
  verified directly. If the coherent submodule for this round is source plus
  real ssh compile/run, record that boundary explicitly.
- Fresh `ssh rvv` evidence must be recorded under `artifacts/tmp/...` with
  enough deterministic data to audit: input path, route, selected family,
  generated source or artifact hashes, sanitized compile command summary, run
  status, and self-check output marker.

## Acceptance Criteria

- The checked-in `i32-vmul` frontend input lowers through the real MLIR/C++
  pass pipeline to a `tcrv.exec.kernel`, selected RVV variant, selected
  lowering-boundary IR, and supported artifact route metadata.
- Generated IR or artifacts visibly carry descriptor-consistent
  `i32-vmul-microkernel.v1`, family/operator/dtype selected-plan metadata,
  selected RVV vector-shape metadata, runtime ABI metadata, and runtime
  element-count metadata.
- Generic source-route metadata preflight is exercised on the `i32-vmul` source
  route and fails closed for stale or missing selected-plan/runtime ABI metadata
  if that focused route is not already covered.
- Source export emits `riscv_vector.h` and the selected-shape-correct RVV
  multiply intrinsic for `i32-vmul`.
- Header and object export for the same selected family are either verified or
  the task records a source/ssh-only boundary with a concrete reason.
- Focused plugin/planning/export tests affected by `i32-vmul` descriptor
  behavior pass.
- Fresh real `ssh rvv` compile/run evidence passes for the bounded `i32-vmul`
  slice, or the task remains open with an exact hardware/toolchain blocker.
- `git diff --check` passes.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-rvv-i32-vmul-frontend-artifact-ssh-evidence` passes
  before finish/archive.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  is run after focused checks if practical.

## Non-Goals

- No broad finite-family matrix.
- No new RVV family, dtype, LMUL, vector shape, generic RVV backend, or
  MLIR vector/scalable-vector lowering path.
- No performance, latency, throughput, or benchmark claim.
- No IME, AME, Sophgo/offload, Template, Toy, or Scalar behavior changes except
  narrow compatibility fixes caused by the `i32-vmul` path.
- No compute semantics in `tcrv.exec`.
- No Python compiler internals.
- No report-only, helper-only, smoke-only, fixture-only, or
  evidence-packaging-only closeout if active C++/MLIR behavior is broken.

## Validation Plan

- Build focused tools/tests as needed:
  `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- Run focused compiler route commands carrying
  `test/Transforms/LinalgToExec/linalg-i32-vmul-to-rvv-artifact.mlir` through
  the selected RVV artifact path.
- Run focused RVV planning/plugin/export C++ tests touched by this task.
- Run focused lit/FileCheck coverage for deterministic `i32-vmul`
  frontend-to-artifact behavior and the compile/export versus runtime evidence
  boundary.
- Run the focused RVV microkernel e2e script with `ssh rvv` for `i32-vmul` and
  store evidence under `artifacts/tmp/...`.
- Run `git diff --check`.
- Run the Trellis task validator before finishing/archive.
- Run full `check-tianchenrv` if the focused checks and build directory are
  healthy.

## Technical Notes

- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/validation/experiment-reference.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-finite-source-route-metadata-coverage/prd.md`,
  `.trellis/tasks/archive/2026-05/05-11-05-11-scalar-finite-source-route-metadata-coverage/prd.md`,
  `.trellis/tasks/archive/2026-05/05-10-05-10-linalg-i32m2-vsub-rvv-scalar-ssh-external-abi-evidence/prd.md`,
  `.trellis/tasks/archive/2026-05/05-10-rvv-i64-binary-sub-mul-frontend-artifact-ssh-evidence/prd.md`,
  `.trellis/tasks/archive/2026-05/05-11-rvv-i64m1-vmul-frontdoor-dispatch-ssh-evidence/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-09-i32-vmul-rvv-scalar-dispatch-ssh-rvv-evidence/prd.md`.
- Primary code/test surfaces to inspect:
  `lib/Transforms/LowerLinalgRVVBinaryToExec.cpp`,
  `lib/Transforms/ExecutionPlanningPipeline.cpp`,
  `lib/Transforms/VariantSelection.cpp`,
  `lib/Transforms/LoweringBoundary.cpp`,
  `lib/Transforms/EmissionReadiness.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`,
  `include/TianChenRV/Target/RVVScalarBinaryFamily.h`,
  `scripts/rvv_microkernel_e2e.py`,
  `scripts/rvv_scalar_dispatch_e2e.py`,
  `test/Transforms/LinalgToExec/linalg-i32-vmul-to-rvv-artifact.mlir`,
  `test/Scripts/rvv-microkernel-e2e.test`,
  `test/Scripts/rvv-microkernel-bundle-e2e.test`,
  `test/Plugin/RVVBinaryPlanningTest.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.

## Definition Of Done

The task is done when the bounded `i32-vmul` linalg/frontend input reaches the
selected RVV boundary and route-metadata-preflighted source/header/object
artifact path through existing compiler ownership boundaries, focused local
checks pass, real `ssh rvv` evidence validates the bounded runtime self-check,
the Trellis task validates and archives, and one coherent commit records the
work.

If unfinished, leave this task open and record the exact continuation point:
frontend lowering, selected variant selection, selected lowering boundary
materialization, route metadata preflight, source/header/object export, ssh
compile, ssh run, evidence artifact capture, focused lit/C++ regression,
Trellis validation, or commit.

## Completion Notes

Completed in this round:

- Confirmed the active C++/MLIR path already carries
  `test/Transforms/LinalgToExec/linalg-i32-vmul-to-rvv-artifact.mlir` through
  `--tcrv-lower-linalg-rvv-binary-to-exec`,
  `--tcrv-execution-planning-pipeline`, selected `rvv_first_slice`,
  `tcrv_rvv.lowering_boundary`, `tcrv_rvv.i32_vmul_microkernel`, supported
  RVV emission-plan metadata, and target-owned source export.
- Confirmed generic source artifact export emits the selected
  `i32-vmul` RVV C intrinsic path with `riscv_vector.h`,
  `__riscv_vmul_vv_i32m1`, selected vector-shape metadata, runtime ABI
  metadata, and no runtime/performance claim in compiler-only output.
- Added a focused lit regression in
  `test/Scripts/rvv-microkernel-e2e.test` that drives
  `scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i32-vmul
  --lower-linalg-frontend --input
  test/Transforms/LinalgToExec/linalg-i32-vmul-to-rvv-artifact.mlir
  --expect-selected-kernel=frontend_i32_vmul`.
- The new regression checks selected kernel, direct source/header/object route
  metadata, runtime ABI name, i32m1 selected vector-shape metadata, RVV vmul
  intrinsic spelling, and the dry-run versus runtime evidence boundary.
- No C++/MLIR implementation fix was needed; RVV behavior remained
  plugin/target-owned, and `tcrv.exec` remained compute-free.

Fresh ssh evidence:

- Command:
  `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i32-vmul
  --lower-linalg-frontend --input
  test/Transforms/LinalgToExec/linalg-i32-vmul-to-rvv-artifact.mlir
  --expect-selected-kernel=frontend_i32_vmul --run-id
  codex-i32-vmul-frontend-ssh-20260511 --overwrite --timeout 120
  --ssh-target rvv`
- Evidence path:
  `artifacts/tmp/rvv_microkernel_e2e/codex-i32-vmul-frontend-ssh-20260511/evidence.json`.
- Result: `status=success`, `pass_fail_result=pass`, `ssh_evidence.success=true`,
  and `stdout_marker_observed=true`.
- Selected input: `test/Transforms/LinalgToExec/linalg-i32-vmul-to-rvv-artifact.mlir`.
- Selected family/route:
  `i32-vmul`, `tcrv-export-rvv-i32-vmul-microkernel-c`,
  `tcrv-export-rvv-i32-vmul-microkernel-header`, and
  `tcrv-export-rvv-i32-vmul-microkernel-object`.
- Selected function:
  `tcrv_rvv_i32_vmul_microkernel_frontend_i32_vmul_rvv_first_slice`.
- Host artifact hashes recorded:
  source `79e3efe5412caf1ede95dcffd7f4491d7f674302fc71310c645c2c56646c39e9`,
  header `c5b71077f4217b46408fe5dfb359f402a17615026b3d51eb2a6a4153c1866017`,
  object `87ad5826ba02b71b615f8381f1bf07e62d857ea1d482f1e132a1f2779d3ce71c`,
  external caller `28008eb36ea4c95f4540fefbc29a013a63c6272c3e63bac64f2d4ab301fa0a79`.
- Remote compile/link/run succeeded for both source-built and generated-object
  external caller paths. Both observed
  `tcrv_rvv_i32_vmul_microkernel_external_abi_ok`.
- Runtime claim is bounded only to this generated RVV `i32-vmul` direct helper
  artifact handoff plus header/object external caller correctness on `ssh rvv`.

Validation completed:

- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test
  -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tcrv-opt
  test/Transforms/LinalgToExec/linalg-i32-vmul-to-rvv-artifact.mlir
  --tcrv-lower-linalg-rvv-binary-to-exec
  --tcrv-execution-planning-pipeline |
  artifacts/tmp/tianchenrv-build/bin/tcrv-translate
  --tcrv-export-target-source-artifact`
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Transforms/LinalgToExec/linalg-i32-vmul-to-rvv-artifact.mlir
  Scripts/rvv-microkernel-e2e.test
  Scripts/rvv-microkernel-bundle-e2e.test`
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed with 205/205 tests.

Spec update judgment:

- No `.trellis/spec/` update is needed. The existing specs already require
  bounded linalg RVV binary frontend coverage, route-metadata-preflighted
  source/header/object artifacts, and real `ssh rvv` evidence for runtime
  correctness claims. This task added a missing focused regression/evidence
  instance for an already specified path.
