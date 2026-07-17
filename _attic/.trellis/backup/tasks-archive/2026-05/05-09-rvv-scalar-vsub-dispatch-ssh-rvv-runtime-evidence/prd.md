# RVV/scalar vsub dispatch ssh rvv runtime evidence

## Goal

Produce bounded remote runtime evidence for the generated RVV-primary plus scalar-fallback `i32` vsub dispatch artifact path. From a real TianChen-RV MLIR/linalg `i32` vsub input, generate the RVV microkernel source/header/object, scalar fallback source, and RVV/scalar dispatch artifact or bundle; compile and link them on the `ssh rvv` host; run a correctness harness that checks subtract semantics through both dispatch guard cases.

## Why Now

The previous module completed local compiler/export behavior in commit `cee4396`: RVV/scalar dispatch artifacts are add/sub-family-aware, vsub has source/header/object routes, stale vadd/vsub family mismatches fail before artifact emission, and `check-tianchenrv` passed. That round did not run `ssh rvv` and therefore made no runtime correctness claim. This task closes the next bounded evidence gap for vsub dispatch runtime behavior.

## Requirements

- Extend the existing RVV/scalar dispatch e2e workflow where appropriate so `i32` vsub is a first-class arithmetic family alongside the existing add path.
- Generate artifacts from TianChen-RV MLIR instead of hand-forging compiler semantics in Python.
- Build the remote harness around existing generated function names, route IDs, runtime ABI metadata, and runtime guard parameter conventions.
- Compile and link generated vsub dispatch artifacts on `ssh rvv`.
- Run two bounded correctness cases:
  - RVV-primary guard path selected and subtract results verified.
  - Scalar-fallback guard path selected and subtract results verified.
- Capture concise evidence under `artifacts/tmp/...`, including remote command lines, generated artifact names, both guard run outputs, and pass/fail summary.
- Keep only source, test, and task changes in git; generated evidence remains under ignored artifact paths.

## Non-goals

- No performance benchmarking or ratio claims.
- No new arithmetic families beyond bounded `i32` add/sub.
- No `i64` or `e64`, mask, new RVV policy families, dynamic-shape frontend expansion, StableHLO/TOSA lowering, or generic RVV lowering.
- No new `tcrv.exec` compute ops.
- No extension-specific semantic branches in generic core passes.
- No broad smoke matrix or dashboard/report-only closeout.
- No runtime correctness claim unless the fresh `ssh rvv` run compiles, links, runs, and verifies outputs.

## Acceptance Criteria

- [x] Local artifact generation for the bounded `i32` vsub dispatch path produces RVV source/header/object, scalar fallback source, and dispatch artifact or bundle.
- [x] Focused local tests cover the changed script/export/harness path for vsub.
- [x] `ssh rvv` workflow compiles and links the generated vsub dispatch artifacts.
- [x] `ssh rvv` workflow verifies subtract semantics for the RVV-primary guard case.
- [x] `ssh rvv` workflow verifies subtract semantics for the scalar-fallback guard case.
- [x] Evidence summary is written under `artifacts/tmp/...` and records remote toolchain command lines, generated artifact names, guard outputs, and pass/fail status.
- [x] `git diff --check` passes.
- [x] CMake configure with the repo LLVM/MLIR paths passes.
- [x] Focused local tests pass.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passes unless a narrower failure blocks first and is recorded truthfully.
- [x] Trellis task validates, finishes, archives, and the workspace journal records the work if the module is complete.
- [x] One coherent git commit is created if the module is complete.

## Evidence

- `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/vsub-dispatch-ssh-rvv-20260509/evidence.json`
- `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/vsub-dispatch-ssh-rvv-20260509/command_summary.json`
- Remote source-built run output: `tcrv_rvv_scalar_i32_vsub_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`
- Remote bundle-object run output: `tcrv_rvv_scalar_i32_vsub_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`

## Technical Notes

- Primary scripts likely involved:
  - `scripts/rvv_scalar_dispatch_e2e.py`
  - `scripts/rvv_microkernel_e2e.py`
- Primary compiler/export code likely involved:
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
- Existing focused tests likely involved:
  - `test/Scripts/rvv-scalar-dispatch-e2e.test`
  - `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`
  - `test/Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-generic-route.mlir`
  - `test/Target/TargetArtifactBundleExport/plan-linalg-i32-vsub-and-export-target-artifact-bundle.mlir`

## Continuation Rule If Unfinished

If the remote runtime workflow cannot be completed in this round, keep the task open and name the exact blocker stage: local artifact generation, remote compile, remote link, RVV guard run, scalar fallback guard run, or evidence capture. Do not archive and do not claim runtime correctness without fresh successful `ssh rvv` evidence.
