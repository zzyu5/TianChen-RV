# Linalg i32m2 vsub SSH RVV evidence handoff

## Goal

Collect a bounded real `ssh rvv` external-ABI evidence handoff for the
frontend-generated `linalg.generic` i32-vsub -> RVV i32m2 path that was just
proved locally through compiler/export FileCheck. The evidence runner must
validate that the exported source/header/object artifacts came from the
expected linalg-origin selected kernel, not from the older explicit typed RVV
fixture, before recording any runtime/correctness claim.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state for this round was a clean worktree at HEAD
  `5b71ba1`.
- `.trellis/.current-task` was absent, so this task was created explicitly as
  `.trellis/tasks/05-10-linalg-i32m2-vsub-ssh-rvv-evidence-handoff/`.
- Latest supervisor audit/review input for
  `20260509T091344Z-r0019-20260509T170108Z` shows the previous task finished
  and archived at
  `.trellis/tasks/archive/2026-05/05-10-linalg-i32-vsub-rvv-i32m2-artifact-handoff/`.
- That previous task proved local compiler/export behavior for
  `test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir`, including
  m2 selected metadata, `!tcrv_rvv.i32m2` body, source/header/object export,
  and a fail-closed missing-config fixture.
- It intentionally did not collect linalg-origin `ssh rvv` evidence, so no
  runtime/correctness claim exists for the frontend-generated source/header/
  object bundle.
- Existing `scripts/rvv_microkernel_e2e.py` can already collect RVV bundle
  evidence and can use `--use-plan-and-export-bundle-front-door` on a linalg
  input, but it does not yet fail closed on an expected selected kernel name.

## Requirements

- Keep compiler truth in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python changes are limited to evidence runner validation,
  artifact orchestration, and sanitized evidence JSON fields.
- Do not change `tcrv.exec` semantics or add RVV-specific branches to core
  compiler passes.
- Extend the evidence runner so a caller can require the compiler-emitted
  selected kernel symbol from generated source comments before evidence success
  is accepted.
- Record bounded compiler path context in evidence JSON, including selected
  kernel, selected variant, selected role, active route, callable ABI source,
  and optional expected selected kernel.
- Add focused local lit coverage proving the linalg-origin i32-vsub i32m2
  bundle dry-run uses:
  - `--use-target-artifact-bundle`;
  - `--use-plan-and-export-bundle-front-door`;
  - the linalg-origin fixture
    `test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir`;
  - `--vector-shape=i32m2`;
  - `--expect-selected-kernel=frontend_i32_vsub`;
  - compiler-emitted `selected_kernel = frontend_i32_vsub` evidence.
- Preserve existing i32m1 add/sub/mul and explicit i32m2 vsub runner behavior.
- Collect one real `ssh rvv` run for the linalg-origin i32-vsub i32m2 bundle
  path when the remote is reachable, and persist sanitized evidence under
  `artifacts/tmp/...`.

## Acceptance Criteria

- `python3 scripts/rvv_microkernel_e2e.py --self-test` passes and covers the
  selected-kernel validation helper.
- A local dry-run for the linalg-origin i32-vsub i32m2 bundle path writes
  evidence JSON with `compiler_path_context.selected_kernel =
  "frontend_i32_vsub"` and `rvv_config.lmul = "m2"`.
- A negative local dry-run with a mismatched expected selected kernel fails
  before success evidence is persisted.
- Focused lit coverage for `rvv-microkernel-bundle-e2e` passes.
- A real `ssh rvv` run succeeds for the same linalg-origin bundle path and
  records `ssh_evidence = true`, `source_stdout_marker_observed = true`, and
  `bundle_object_stdout_marker_observed = true`.
- `git diff --check` and a focused project check pass; run full
  `check-tianchenrv` if local time/build state permits.

## Out of Scope

- New compiler lowering behavior.
- New arithmetic families or vector shapes.
- Performance measurement.
- Broad test matrices.
- Treating local dry-run, lit, or Python self-test as RVV runtime evidence.
- Moving route selection, runtime ABI shape, arithmetic semantics, capability
  modeling, lowering, or emission into Python.

## Technical Approach

- Parse existing generated C comment fields already emitted by the target-owned
  RVV exporter.
- Validate `--expect-selected-kernel` against the parsed `selected_kernel`
  comment after generated source validation and before evidence JSON success.
- Store the parsed compiler path context in both direct and bundle evidence
  modes.
- Add one bundle lit run for the linalg-origin i32m2 path and one mismatched
  expected-kernel negative check.
- Run the same command without `--dry-run` on `ssh rvv` to produce the bounded
  external ABI evidence.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/implementation-stack/compiler-stack-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/validation/experiment-reference.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- Current task evidence source:
  - `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0019-20260509T170108Z/repo_audit.md`
  - `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0019-20260509T170108Z/review_input.md`
- Primary implementation surfaces:
  - `scripts/rvv_microkernel_e2e.py`
  - `test/Scripts/rvv-microkernel-bundle-e2e.test`
  - `test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir`

## Validation Plan

- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- Focused local dry-run for linalg-origin i32-vsub i32m2 bundle evidence.
- Focused mismatched expected-kernel negative dry-run.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-microkernel-bundle-e2e'`
  from `artifacts/tmp/tianchenrv-build/test`.
- Real `ssh rvv` linalg-origin i32-vsub i32m2 bundle run.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if feasible.

## Current Status

Completed.

## Completion Notes

- Extended `scripts/rvv_microkernel_e2e.py` with
  `--expect-selected-kernel`, which validates the target-exporter-emitted
  `selected_kernel` source comment before accepting evidence. This keeps the
  linalg-origin evidence path from accidentally accepting the explicit typed
  RVV fixture.
- The runner now records bounded `compiler_path_context` in evidence JSON:
  microkernel function, selected kernel, selected variant, selected role,
  lowering boundary, active route, and callable ABI source.
- Added local lit coverage for the linalg-origin i32-vsub i32m2 bundle dry-run
  through `--tcrv-plan-and-export-target-artifact-bundle`, plus a negative
  mismatched selected-kernel expectation.
- Collected real `ssh rvv` external-ABI evidence for the frontend-generated
  path:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/20260510-linalg-i32m2-vsub-ssh/`.
- The evidence JSON records:
  - `input = test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir`
  - `compiler_path_context.selected_kernel = frontend_i32_vsub`
  - `expected_selected_kernel = frontend_i32_vsub`
  - `rvv_config.shape = i32m2`
  - `rvv_config.lmul = m2`
  - `ssh_evidence = true`
  - `source_stdout_marker_observed = true`
  - `bundle_object_stdout_marker_observed = true`
  - `expected_stdout_marker =
    tcrv_rvv_i32_vsub_microkernel_external_abi_ok`
- Host artifact hashes from the successful run:
  - source:
    `44fadd6651ccbd187ac8fcd5386e0a9679799acffd6f8790bcf2cdfdd44992e3`
  - header:
    `9e6b0cd9bf14501505b406871f53e3df62cf7895578bf32e78dd700d526091b4`
  - object:
    `486ddd4b64f38ac9e0ea74e3fb04e345d581a28f2333cc2270e6a2a5fe6772f9`
  - external caller:
    `5b33acd0f4e9ab11c7e8dd39fc6efae2318def774ea8cef49bb7eb3fe115de70`
- Remote executable hashes from the successful run:
  - source-built executable:
    `7122f566f61f20d31639d4a6da399b7a7fa3aaadfd8ecbc5498ff492d6742170`
  - bundle-object executable:
    `01ae948bf4bc3196fde72e4bc519542ccb3968c53bf3cd65eed4b511c1365b46`
- This supports only the bounded RVV i32-vsub target-artifact bundle external
  caller correctness claim for this frontend-generated i32m2 path. It is not a
  performance claim, generic linalg lowering claim, or arbitrary RVV emission
  claim.

## Checks Run

- `python3 ./.trellis/scripts/task.py validate 05-10-linalg-i32m2-vsub-ssh-rvv-evidence-handoff`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- `python3 scripts/rvv_microkernel_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --vector-shape=i32m2 --expect-selected-kernel=frontend_i32_vsub --run-id probe-linalg-i32m2-vsub-bundle-dry --overwrite --input test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir`
- Negative selected-kernel mismatch dry-run:
  `python3 scripts/rvv_microkernel_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --vector-shape=i32m2 --expect-selected-kernel=rvv_sub_kernel --run-id probe-linalg-i32m2-vsub-bundle-mismatch --overwrite --input test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir`
  failed as expected.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-microkernel-bundle-e2e'`
  from `artifacts/tmp/tianchenrv-build/test`, passed 1/1.
- Real `ssh rvv` evidence run:
  `python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --vector-shape=i32m2 --expect-selected-kernel=frontend_i32_vsub --run-id 20260510-linalg-i32m2-vsub-ssh --overwrite --timeout 120 --ssh-target rvv --input test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir`
- `git diff --check`
- `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  passed 169/169 tests.
