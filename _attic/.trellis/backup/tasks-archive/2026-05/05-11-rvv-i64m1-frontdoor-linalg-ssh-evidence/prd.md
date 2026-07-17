# RVV i64m1 front-door linalg-to-ssh evidence

## Goal

Prove the repaired i64-vadd / i64m1 RVV front-door route end-to-end from the checked-in linalg input through normal proposal planning, selected RVV microkernel export, generated target artifact bundle, and real `ssh rvv` compile/link/run/output validation.

This task exists because commit `60f3578 fix(rvv): route i64m1 plan-and-export proposals` repaired plan-and-export proposal routing enough for local dry-run bundle export, but it intentionally did not produce fresh real RVV host evidence for the linalg front-door path.

## Module Boundary

Owned behavior:

- Input starts from `test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir` when possible.
- Route goes through `scripts/rvv_microkernel_e2e.py` with `--use-target-artifact-bundle` and `--use-plan-and-export-bundle-front-door`.
- Proposal planning must select `frontend_i64_vadd` and the i64m1 RVV route.
- Exported artifact evidence must include the generated i64 ABI / i64m1 selected metadata and `__riscv_vadd_vv_i64m1`.
- Real `ssh rvv` execution must compile/link/run the generated bundle and validate deterministic numeric output.

Explicit non-goals:

- No generic RVV backend.
- No broad family matrix, benchmark sweep, or performance claim.
- No unrelated i32/i32m2/i64 expansion beyond the bounded i64-vadd route.
- No hand-written runtime source as the source of truth if it bypasses compiler/export output.
- No compute semantics in `tcrv.exec`.
- No RVV semantic branches in generic core passes or generic artifact routing.
- No compiler internals in Python.
- No docs-only, metadata-only, helper-only, dry-run-only, or report-only closeout.

## Requirements

- Create durable evidence that the linalg front-door route selects `frontend_i64_vadd`, not an already-selected direct-only fixture or scalar fallback.
- Validate that the emitted selected microkernel body contains `__riscv_vadd_vv_i64m1`.
- Validate deterministic output on the real `rvv` host over SSH.
- If the route fails, repair the active C++/MLIR/RVV plugin, lowering, target export, or runtime ABI route that is actually blocking it.
- Python changes, if any, are limited to runner/probe/artifact orchestration.
- Add or update a focused lit/FileCheck regression hook for the linalg i64-vadd front-door route.
- Keep RVV logic plugin/target-owned and preserve the compute-free `tcrv.exec` core boundary.

## Acceptance Criteria

- [x] Local dry-run front-door command succeeds:
  `scripts/rvv_microkernel_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vadd --input test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir --expect-selected-kernel=frontend_i64_vadd`
- [x] Equivalent linalg front-door route succeeds on `ssh rvv`.
- [x] Evidence records remote architecture, clang path/version, compile/link/run status, selected kernel, observed RVV intrinsic, runtime counts, and output comparison result.
- [x] Focused lit/FileCheck test checks i64 route ids, i64 ABI, i64m1 selected metadata, and `__riscv_vadd_vv_i64m1`.
- [x] Focused changed-code checks pass, including `git diff --check` and relevant RVV plugin/export/script tests.
- [x] Trellis task status and notes are truthful before archive and after archive if complete.
- [x] One coherent commit records the completed module, or the task remains open with an exact blocker and continuation point.

## Initial Files To Read

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/capability-model/capability-contract.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/validation/experiment-reference.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-11-rvv-i64m1-plan-and-export-frontdoor-route/prd.md`
- `.trellis/tasks/archive/2026-05/05-11-rvv-verified-i64m1-microkernel-ssh-evidence/prd.md`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- `scripts/rvv_microkernel_e2e.py`
- `test/Scripts/rvv-microkernel-bundle-e2e.test`
- `test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir`
- `lib/Transforms/LowerLinalgI32BinaryToExec.cpp`
- adjacent linalg-to-exec/RVV lowering files if the linalg i64 path fails
- `lib/Target/RVV/RVVMicrokernel.cpp`
- `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp`
- relevant bundle/export tests under `test/Target/TargetArtifactBundleExport/` and `test/Target/RVVMicrokernel/`

## Evidence Notes

- Local dry-run command:
  `python3 scripts/rvv_microkernel_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vadd --input test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir --expect-selected-kernel=frontend_i64_vadd --run-id codex-i64-vadd-frontdoor-linalg-dry --overwrite --timeout 120`
- Local dry-run result:
  `status=success`, `mode=dry-run`,
  `bundle_export_mode=plan-and-export-target-artifact-bundle`,
  `selected_kernel=frontend_i64_vadd`, `vector_shape=i64m1`,
  `ssh_evidence=false`.
- SSH RVV command:
  `python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vadd --input test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir --expect-selected-kernel=frontend_i64_vadd --ssh-target rvv --run-id codex-i64-vadd-frontdoor-linalg-ssh --overwrite --timeout 120`
- SSH RVV result:
  `status=success`, `mode=ssh`, `ssh_evidence.success=true`,
  `remote_compile_succeeded=true`, `remote_link_succeeded=true`,
  `remote_run_succeeded=true`, and `output_validation_succeeded=true`.
- Remote architecture:
  `riscv64`; uname
  `Linux ubuntu 6.12.23 #1 SMP Thu Apr 17 11:46:50 EDT 2025 riscv64 riscv64 riscv64 GNU/Linux`.
- Remote clang:
  `/usr/bin/clang`, `Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Selected kernel:
  `frontend_i64_vadd`; selected variant `rvv_first_slice`; active route
  `tcrv-export-rvv-i64-vadd-microkernel-c`.
- Observed intrinsic:
  `__riscv_vadd_vv_i64m1`, with selected vector config
  `dtype=i64, shape=i64m1, sew=64, lmul=m1, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint64m1_t, vector_suffix=i64m1, setvl_suffix=e64m1`.
- Runtime counts:
  `7` and `16` through the generated external ABI caller.
- Output comparison:
  Generated caller validated `lhs + rhs` and observed
  `tcrv_rvv_i64_vadd_microkernel_external_abi_ok` for both source-built object
  and compiler-generated bundle object paths.
- Artifact path:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-i64-vadd-frontdoor-linalg-ssh/evidence.json`.

## Completion Notes

- No C++/MLIR compiler fix was needed. Current HEAD already carries the bounded
  linalg i64-vadd front-door path through normal proposal planning, selected RVV
  microkernel materialization, target artifact bundle export, and real RVV host
  compile/link/run validation.
- Added focused script lit coverage in
  `test/Scripts/rvv-microkernel-bundle-e2e.test` for the exact linalg
  `i64-vadd` plan-and-export bundle front-door. It checks the linalg input path,
  `frontend_i64_vadd`, `rvv_first_slice`, i64 ABI parameter types, i64m1
  selected capability metadata, `__riscv_vadd_vv_i64m1`, and absence of the
  direct selected fixture symbols `export_i64_vadd` / `rvv_i64_slice`.
- No generic RVV backend, generic linalg lowering claim, performance claim, or
  broad family matrix was added. The runtime correctness claim is limited to
  the exact i64-vadd/i64m1 front-door evidence recorded above.

## Validation Results

- `python3 scripts/rvv_microkernel_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vadd --input test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir --expect-selected-kernel=frontend_i64_vadd --run-id codex-i64-vadd-frontdoor-linalg-dry --overwrite --timeout 120`: passed.
- `python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vadd --input test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir --expect-selected-kernel=frontend_i64_vadd --ssh-target rvv --run-id codex-i64-vadd-frontdoor-linalg-ssh --overwrite --timeout 120`: passed.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-microkernel-bundle-e2e'` from `artifacts/tmp/tianchenrv-build/test`: 1/1 selected test passed.
- `git diff --check`: passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`: passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`: passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`: passed.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'plan-linalg-i64-vadd|rvv-extension-plugin|rvv-microkernel-bundle-e2e'` from `artifacts/tmp/tianchenrv-build/test`: 3/3 selected tests passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`: 199/199 lit tests passed.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-11-rvv-i64m1-frontdoor-linalg-ssh-evidence`: passed.

## Definition Of Done

The task is done only when the bounded linalg front-door i64-vadd/i64m1 path has fresh generated-bundle evidence and real `ssh rvv` deterministic runtime validation, plus a focused regression hook and a coherent commit. If runtime correctness lacks real `ssh rvv` evidence, leave the task open.
