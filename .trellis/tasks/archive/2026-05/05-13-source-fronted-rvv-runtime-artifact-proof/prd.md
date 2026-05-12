# Source-fronted RVV runtime artifact proof

## Goal

Prove that the existing bounded i32-vadd route can start at the
non-final linalg/source boundary, materialize the selected `rvv-plugin`
variant into typed `tcrv_rvv` ops, export generated RVV C/header/object
artifacts through the existing target route, and execute the generated callable
correctly on the real RVV host via `ssh rvv`.

## What I Already Know

* Current HEAD at task start is
  `ae7c40d fix(rvv): validate bounded vector materialization metadata`; the
  worktree was clean before task creation.
* No `.trellis/.current-task` existed at session start. This task was created
  from the Hermes Direction Brief before source edits.
* The previous archived task
  `.trellis/tasks/archive/2026-05/05-13-05-13-bounded-mlir-vector-to-rvv-materialization/prd.md`
  proved the source/linalg path materializes typed RVV body ops and exports
  source/header/object artifacts, but intentionally made no runtime correctness
  claim.
* Existing source-fronted input:
  `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`.
* Existing production route pieces named by the brief include
  `RVVBinaryMicrokernelMaterialization.cpp`,
  `RVVBinarySelectedEmissionPlanning.cpp`, `RVVMicrokernel.cpp`,
  `TargetArtifactExport.cpp`, `tcrv-translate.cpp`, and the
  RVVMicrokernel/EmissionManifest/TargetArtifactBundleExport test surfaces.
* The specs require runtime/correctness claims to use real `ssh rvv` evidence.
  Local compile-only, FileCheck, Python dry-run, or older artifacts are not RVV
  runtime evidence.

## Requirements

* The evidence path must start from
  `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`, not from a
  hand-authored final RVV microkernel fixture.
* The production planning/materialization pipeline must create the selected
  `tcrv.exec` RVV variant, selected lowering boundary, supported emission plan,
  and plugin-owned `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` / i32 load-add-store
  body.
* Artifact export must use the existing compiler routes for generated RVV
  source, generated header, and generated RISC-V relocatable object. One-off
  manual artifact surgery is not acceptable; if a route boundary is missing,
  repair that boundary first.
* The generated artifacts must still expose selected binary config, runtime AVL
  provenance, callable ABI order, and EmitC route provenance.
* Runtime evidence must build a small external correctness harness around the
  generated ABI. The harness may check i32 addition results, but it must consume
  the generated header/object/source ABI instead of hardcoding compiler route
  metadata or replacing the generated callable.
* The external harness must run representative runtime `n` values below, equal
  to, and above the descriptor-local first-slice capacity.
* The remote compile/run must use the real RVV host through `ssh rvv` and
  clang/LLVM RVV flags derived from the generated/provenanced route.
* Python may orchestrate bounded evidence, parse artifacts, copy files, and
  write sanitized command summaries. It must not implement compiler internals,
  route selection, lowering, emission, or descriptor-driven computation.

## Acceptance Criteria

* [x] Trellis task, PRD, and implement/check context are created before source
      edits.
* [x] Exact commands start from the linalg/source input and run the production
      lowering/planning pipeline into materialized MLIR.
* [x] Exact `tcrv-translate` commands export generated RVV source, header, and
      object artifacts, preferably through the existing generic artifact front
      doors or target bundle route when that is the production boundary.
* [x] Focused assertions prove the materialized/generated artifacts contain the
      selected RVV body, selected binary config, runtime AVL provenance,
      callable ABI order, and EmitC route provenance.
* [x] The generated object is tool-readable RISC-V ELF relocatable output with
      the generated callable symbol and no hidden `main`, proven by `file` and
      `llvm-readobj` or the local LLVM equivalent.
* [x] A bounded external correctness harness is compiled and run on `ssh rvv`
      against the generated callable for representative `n` values below,
      equal to, and above the descriptor-local first-slice capacity.
* [x] If the current compiler route cannot support this without manual artifact
      surgery, the route/export contract is repaired or the task remains open
      with the precise blocker recorded.
* [x] Focused build includes touched targets such as `TianChenRVTarget`,
      `TianChenRVTransforms`, `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test` when target/export code changes.
* [x] Focused lit covers affected LinalgToExec, RVVMicrokernel,
      EmissionManifest, TargetArtifactExport, and TargetArtifactBundleExport
      routes.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] Task status, completion notes, exact artifact paths, and runtime evidence
      are recorded truthfully before one coherent commit is created.

## Definition Of Done

* The compiler behavior remains implemented in C++/MLIR/LLVM/TableGen/CMake/lit
  where compiler behavior changes are needed.
* No computation semantics are added to `tcrv.exec`.
* No descriptor-driven compute fallback or descriptor-to-C replacement path is
  added.
* No RVV-specific semantic branch is added to shared core orchestration outside
  established plugin/target interfaces.
* Runtime evidence is bounded to the one source-fronted i32-vadd callable ABI
  proof and is not reported as generic RVV lowering, full runtime integration,
  broad correctness, or performance evidence.

## Out Of Scope

* New arithmetic families, dtype matrices, performance claims, broad smoke
  matrices, generic RVV backend claims, generic tensor/tile IR, or descriptor
  compute authority.
* Replacing the existing compiler route with a Python implementation.
* Treating prompt edits, helper-only changes, or metadata-only coverage as the
  main achievement.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior PRD read:
  `.trellis/tasks/archive/2026-05/05-13-05-13-bounded-mlir-vector-to-rvv-materialization/prd.md`.
* Initial code/test surfaces to inspect before implementation:
  `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Target/RVVMicrokernel/`,
  `test/Target/EmissionManifest/`,
  `test/Target/TargetArtifactBundleExport/`, and
  `test/Target/TargetArtifactBundleExport/plan-linalg-i32-vadd-and-export-target-artifact-bundle.mlir`.

## Completion Notes

* Production-facing change was kept in the bounded evidence runner and lit
  contract: `scripts/rvv_microkernel_e2e.py` can now use the source/linalg
  i32-vadd fixture as the default frontend input, pass repeated
  `--runtime-count` values through the generated external caller, size caller
  inputs from the maximum runtime `n`, and fail closed when direct external ABI
  evidence is requested for a non-direct selected variant.
* The i64 direct family helper specs no longer route through the stale generic
  i32 translation aliases; focused script lit caught that stale boundary while
  adding the source-fronted i32-vadd runtime proof.
* Runtime artifact proof directory:
  `artifacts/tmp/rvv_microkernel_e2e/20260513T-source-fronted-i32-vadd-runtime-artifact-proof`.
* Generated artifacts in that directory:
  `post_planning.mlir`, `emission_manifest.txt`, `rvv_microkernel.c`,
  `rvv_microkernel.h`, `rvv_microkernel.o`,
  `rvv_microkernel_external_caller.c`, `evidence.json`,
  `command_summary.json`, and `hashes.json`.
* Exact compiler artifact commands recorded in
  `command_summary.json`:

```bash
/home/kingdom/phdworks/TianchenRV/build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline
/home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --tcrv-export-emission-manifest artifacts/tmp/rvv_microkernel_e2e/20260513T-source-fronted-i32-vadd-runtime-artifact-proof/post_planning.mlir
/home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --tcrv-export-rvv-microkernel-c artifacts/tmp/rvv_microkernel_e2e/20260513T-source-fronted-i32-vadd-runtime-artifact-proof/post_planning.mlir
/home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --tcrv-export-rvv-microkernel-header artifacts/tmp/rvv_microkernel_e2e/20260513T-source-fronted-i32-vadd-runtime-artifact-proof/post_planning.mlir
/home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --tcrv-export-rvv-microkernel-object artifacts/tmp/rvv_microkernel_e2e/20260513T-source-fronted-i32-vadd-runtime-artifact-proof/post_planning.mlir
```

* Runtime proof command:

```bash
python3 scripts/rvv_microkernel_e2e.py --arithmetic-family i32-vadd --lower-linalg-frontend --expect-selected-kernel frontend_i32_vadd --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --run-id 20260513T-source-fronted-i32-vadd-runtime-artifact-proof --overwrite --timeout 120
```

* Runtime proof result:
  `status=success`, `ssh_evidence=true`, selected kernel
  `frontend_i32_vadd`, runtime counts `7,16,23`. Remote source-built and
  exported-object runs both printed
  `tcrv_rvv_microkernel_external_abi_ok counts=7,16,23`.
* Object evidence:
  `file rvv_microkernel.o` reports
  `ELF 64-bit LSB relocatable, UCB RISC-V, RVC, double-float ABI, version 1 (SYSV), not stripped`.
  `llvm-readobj-20 --file-headers --symbols` reports
  `Format: elf64-littleriscv`, `Arch: riscv64`,
  `Machine: EM_RISCV`, and function symbol
  `tcrv_rvv_i32_vadd_microkernel_frontend_i32_vadd_rvv_first_slice`.
* Generated artifact provenance assertions:
  `rvv_microkernel.c` exposes `selected_binary_config`,
  `runtime_element_count_c_name=n`, `runtime_abi_parameter[3]` with role
  `runtime-element-count`, the public callable ABI, and EmitC route comments
  from `tcrv_rvv.family_ops` through `emitc.call_opaque` to RVV intrinsic C.
* Validation run:
  `python3 scripts/rvv_microkernel_e2e.py --self-test`;
  `python3 -m py_compile scripts/rvv_microkernel_e2e.py`;
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-microkernel-e2e'`
  from `build/test`;
  `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`;
  `./build/bin/tianchenrv-target-artifact-export-test`;
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'LinalgToExec|RVVMicrokernel|EmissionManifest|TargetArtifactBundleExport|rvv-microkernel-e2e'`
  from `build/test`;
  `git diff --check`.
