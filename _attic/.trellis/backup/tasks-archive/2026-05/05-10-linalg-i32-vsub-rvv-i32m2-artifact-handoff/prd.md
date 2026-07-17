# Linalg i32-vsub to RVV i32m2 execution-planning artifact handoff

## Goal

Complete one coherent high-level MLIR frontend slice where a marked
`linalg.generic` i32-vsub wrapper lowers through the normal
`LowerLinalgI32BinaryToExec` and execution-planning pipeline into a selected
RVV i32m2 path, with consistent selected variant metadata, lowering-boundary
metadata, emission plan, `tcrv_rvv` body, and target-owned source/header/object
artifact export.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state for this round was a clean worktree at HEAD
  `65105b3`.
- `.trellis/.current-task` was absent, so this task was created explicitly as
  `.trellis/tasks/05-10-linalg-i32-vsub-rvv-i32m2-artifact-handoff/`.
- The previous task is finished and archived at
  `.trellis/tasks/archive/2026-05/05-10-rvv-i32m2-typed-microkernel-ssh-evidence-handoff/`
  and must not be reopened.
- The previous m2 compiler slice added finite `!tcrv_rvv.i32m2` typing,
  m2 RVV plugin materialization, and target source/header/object export for
  explicit typed RVV fixtures.
- The previous SSH evidence task proved the explicit typed i32-vsub m2
  microkernel/exporter path with real `ssh rvv` evidence and commit `65105b3`.
- The remaining bottleneck is the high-level linalg-origin path: current
  frontend fixtures may still select or export m1-oriented RVV metadata unless
  frontend target/profile capability import and plugin selection carry m2
  through the normal planning pipeline.

## Requirements

- Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only remain tooling, runner, evidence orchestration, or
  artifact parsing.
- Keep `tcrv.exec` compute-free. The linalg frontend may create an exec kernel
  and ABI boundary only; RVV-specific behavior must remain dialect/plugin/target
  owned.
- Preserve existing m1 linalg i32 vadd/vsub/vmul behavior.
- Add or update a real MLIR fixture where a marked `linalg.generic` with
  `tcrv_frontend_lowering = "i32-vsub"` targets an RVV-capable profile/provider
  that makes the finite i32m2 config shape available.
- Running `tcrv-opt --tcrv-lower-linalg-i32-binary-to-exec
  --tcrv-execution-planning-pipeline` on that fixture must produce a selected
  RVV path whose metadata, selected lowering-boundary, emission plan, and
  materialized `tcrv_rvv` body consistently identify i32m2.
- Target-owned export for the linalg-origin path must emit `vint32m2_t`,
  `__riscv_vsetvl_e32m2`, `__riscv_vle32_v_i32m2`,
  `__riscv_vsub_vv_i32m2`, and `__riscv_vse32_v_i32m2`.
- If the current compiler path does not carry m2 from frontend target
  capabilities to RVV plugin selection/materialization, fix the smallest real
  compiler surface responsible: frontend target/provider import, RVV capability
  profile/config recognition, RVV proposal/selection/materialization metadata,
  or emission readiness.
- Add fail-closed coverage for at least one stale layering case: selected or
  advertised m2 with m1 body/export metadata, or requested m2 config absent
  from the capability provider. The diagnostic must fire before artifact or
  runtime correctness claims.

## Acceptance Criteria

- A focused linalg-origin i32-vsub m2 fixture lowers through
  `--tcrv-lower-linalg-i32-binary-to-exec
  --tcrv-execution-planning-pipeline` and FileCheck proves:
  - selected RVV variant metadata requires or otherwise points at the finite
    m2 config/policy capability shape;
  - selected lowering-boundary and emission-plan metadata remain coherent with
    the selected RVV path;
  - materialized `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` use LMUL `m2`;
  - dataflow body values are `!tcrv_rvv.i32m2`;
  - descriptor/family metadata selects `i32-vsub-microkernel.v1`.
- Target source/header/object export checks for that linalg-origin fixture
  prove the target exporter consumes the m2 body and emits i32m2 C intrinsic
  spelling rather than stale m1 spelling.
- At least one negative test fails closed before artifact success when m2 is
  requested but the capability/provider or materialized body/config metadata is
  stale or absent.
- Existing m1 linalg i32 add/sub/mul and explicit typed RVV m2 tests remain
  compatible.
- No runtime, correctness, or performance claim is made unless a real `ssh rvv`
  run is collected for this linalg-origin generated artifact and saved with
  explicit evidence fields.

## Out of Scope

- New dtypes.
- LMUL m4/m8/fractional support.
- A generic vector lowering route.
- Adding i32-vmul m2 merely for breadth.
- Moving compiler truth into Python.
- Adding compute semantics to `tcrv.exec`.
- RVV-specific branches in core orchestration passes.
- Treating docs, reports, broad smoke matrices, prompt edits, or helper wrappers
  as the main deliverable.
- Performance or broad correctness claims.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Archived task PRDs read:
  - `.trellis/tasks/archive/2026-05/05-09-rvv-i32m2-typed-intrinsic-emission-slice/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i32m2-typed-microkernel-ssh-evidence-handoff/prd.md`
- Primary source surfaces to inspect:
  - `lib/Transforms/LowerLinalgI32BinaryToExec.cpp`
  - `lib/Transforms/ExecutionPlanningPipeline.cpp`
  - `lib/Transforms/VariantSelection.cpp`
  - `lib/Transforms/EmissionReadiness.cpp`
  - `lib/Plugin/RVV/RVVCapabilityProfile.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir`
  - `test/Target/RVVMicrokernel/rvv-microkernel-i32m2-family-sub.mlir`

## Validation Plan

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- Focused lit for the new or changed linalg i32-vsub m2 fixture, including
  target source/header/object export checks.
- Focused lit for affected `test/Transforms/ExecutionPlanning`,
  `test/Transforms/EmissionReadiness`, `test/Plugin/rvv-extension-plugin.test`,
  and `test/Target/RVVMicrokernel` m2 tests if those surfaces change.
- Run `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
  or other relevant C++ tests if plugin code changes.
- If `scripts/rvv_microkernel_e2e.py` changes, run its py_compile, self-test,
  and focused script lit coverage.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` before finish/archive if local build state supports it.

## Current Status

Completed.

## Completion Notes

- Confirmed by real pipeline probe that the existing C++ compiler path already
  carries finite i32m2 target capability/provider facts from the linalg-origin
  target profile into RVV plugin proposal, selection, lowering-boundary
  materialization, `tcrv_rvv` body materialization, and target artifact export.
- Updated the focused linalg-origin i32-vsub RVV artifact fixture so its target
  profile provides the finite i32m2 config shape and FileCheck proves:
  selected target/profile metadata advertises m2, the generated RVV body uses
  `setvl` / `with_vl` LMUL `m2`, dataflow values are
  `!tcrv_rvv.i32m2`, and the selected descriptor remains
  `i32-vsub-microkernel.v1`.
- Extended the same linalg-origin fixture beyond source export to header and
  relocatable object artifact export, proving the target-owned exporter consumes
  the generated m2 body and emits the expected external ABI symbol.
- Added a fail-closed linalg-origin negative fixture where the target profile
  advertises an m2-oriented profile but omits the finite
  `rvv.i32_m2.lmul_m2` config provider. Execution planning now has explicit
  coverage that this produces no viable RVV proposal before microkernel or
  emission-plan materialization.
- No C++ compiler changes were required in this round; the module-level
  behavior was completed by converting the high-level linalg-origin coverage to
  consume the already implemented m2 compiler/export path.
- `tcrv.exec` stayed compute-free, RVV behavior stayed dialect/plugin/target
  owned, Python remained tooling-only, and parameter layering was preserved.
- No linalg-origin `ssh rvv` runtime/correctness evidence was collected in this
  round, so no new runtime, correctness, or performance claim is made.

## Checks Run

- `python3 ./.trellis/scripts/task.py validate 05-10-linalg-i32-vsub-rvv-i32m2-artifact-handoff`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='linalg-i32-vsub-(to-rvv-artifact|rvv-i32m2-missing-config-fails)'`
  from `artifacts/tmp/tianchenrv-build/test`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- `git diff --check`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='LinalgToExec'`
  from `artifacts/tmp/tianchenrv-build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-microkernel-i32m2-(family-sub|object|body-mismatch-fails)'`
  from `artifacts/tmp/tianchenrv-build/test`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed 169/169 tests.
