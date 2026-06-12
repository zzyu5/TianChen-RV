# RVV i32m2 typed microkernel SSH evidence handoff

## Goal

Carry the existing typed `!tcrv_rvv.i32m2` RVV i32-vsub microkernel fixture
through the normal TianChen-RV compiler/export path and collect bounded real
`ssh rvv` compile/link/run evidence. This task switches from compiler shape
implementation to end-to-end hardware evidence for that typed m2 shape; it must
not become another helper-only or report-only refinement.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree state was clean and current HEAD was `ee2011e`.
- `.trellis/.current-task` was absent, so this task was created as
  `.trellis/tasks/05-10-rvv-i32m2-typed-microkernel-ssh-evidence-handoff/`.
- The previous task is already finished and archived at
  `.trellis/tasks/archive/2026-05/05-09-rvv-i32m2-typed-intrinsic-emission-slice/`;
  it must not be reopened.
- The archived prior task added finite `!tcrv_rvv.i32m2` type/verifier/plugin
  materialization and source/header/object tests for i32-vsub, but explicitly
  made no runtime, correctness, or performance claim because no real `ssh rvv`
  run was collected.
- `scripts/rvv_microkernel_e2e.py` already orchestrates compiler tools,
  sanitized artifact generation, local dry-runs, and optional `ssh rvv`
  compile/link/run evidence for bounded RVV i32 microkernels.
- The current runner family table and generated-source validation are still
  m1-default: `i32-vsub` expects the m1 fixture and m1 intrinsic spellings unless
  the user manually overrides `--input`, so it needs an intentional typed m2
  evidence mode.
- The existing typed m2 i32-vsub fixtures are:
  - `test/Target/RVVMicrokernel/rvv-microkernel-i32m2-family-sub.mlir`
  - `test/Target/RVVMicrokernel/rvv-microkernel-i32m2-object.mlir`
  - `test/Target/RVVMicrokernel/rvv-microkernel-i32m2-body-mismatch-fails.mlir`

## Requirements

- Keep compiler truth in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only orchestrate tools, parse artifacts, generate an
  external caller, and write sanitized evidence.
- Keep `tcrv.exec` compute-free. RVV behavior must remain dialect/plugin/target
  owned.
- Extend the RVV microkernel evidence workflow with an explicit typed m2 mode,
  rather than inferring m2 from a manual input override alone.
- For `i32-vsub` m2, start from a real MLIR fixture containing typed
  `!tcrv_rvv.i32m2` body/materialization evidence and selected m2
  capability/config facts, not a Python-only source template.
- Run the normal compiler path, preferably `tcrv-opt
  --tcrv-execution-planning-pipeline` or the existing materialize-plan path
  followed by `tcrv-translate`, to produce source/header/object artifacts.
- Verify generated source/header/object correspond to the same m2 run before
  any remote runtime/correctness claim:
  - source must contain SEW=32, LMUL=m2, agnostic tail/mask policy metadata;
  - source must contain `vint32m2_t`, `__riscv_vsetvl_e32m2`,
    `__riscv_vle32_v_i32m2`, `__riscv_vsub_vv_i32m2`, and
    `__riscv_vse32_v_i32m2`;
  - source/header/object hashes recorded in evidence must be the artifacts used
    by the remote compile/link/run step.
- Compile/link/run on `ssh rvv` using generated artifact(s) and an external
  caller or self-check suitable for the m2 artifact.
- Record sanitized evidence JSON with arithmetic family, SEW/LMUL/policy,
  route ids, artifact paths, hashes, remote command summaries, expected marker,
  observed marker, `ssh_evidence=true`, and bounded claim scope.
- Fail closed before any runtime/correctness claim if the artifact is m1, if m2
  metadata is absent or stale, if source/header/object hashes do not correspond
  to the run, or if `ssh rvv` evidence is missing.
- Do not add vmul m2 compiler support just for this task. If vmul m2 is not
  already a narrow fixture-level extension, leave it as future work.

## Acceptance Criteria

- `scripts/rvv_microkernel_e2e.py` exposes an intentional m2 evidence mode for
  `i32-vsub` that selects the typed m2 fixture by default.
- Dry-run for the m2 mode produces evidence with `ssh_evidence=false`, no
  runtime/correctness claim, and explicit SEW=32/LMUL=m2/policy metadata.
- The m2 mode rejects m1-generated source before an SSH claim.
- Focused script/lit coverage proves the m2 dry-run emits and records m2
  metadata, routes, artifacts, hashes, and no forbidden runtime/performance
  wording.
- A real `ssh rvv` i32-vsub m2 run succeeds and saves evidence under
  `artifacts/tmp/rvv_microkernel_e2e/.../evidence.json` with
  `ssh_evidence=true` and marker observation true.
- Existing m1 add/sub/mul runner behavior remains compatible.
- No performance or broad correctness claim is made; the only runtime claim is
  the bounded generated i32-vsub m2 microkernel artifact/caller behavior proven
  by this SSH run.

## Out of Scope

- New RVV dtypes or LMULs beyond the existing i32m2 slice.
- New compiler route selection, lowering, emission, ABI truth, or capability
  facts in Python.
- Changes to `rvv_remote_probe` or `rvv_probe_to_mlir` unless a truly required
  probed fact is backed by remote evidence.
- RVV-specific branches in core orchestration passes.
- Docs/status/report-only work as the main deliverable.
- Performance claims, broad correctness claims, or benchmark matrices.

## Technical Notes

- Specs read for this task:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/validation/experiment-reference.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- Prior task archive read:
  - `.trellis/tasks/archive/2026-05/05-09-rvv-i32m2-typed-intrinsic-emission-slice/prd.md`
  - `.trellis/tasks/archive/2026-05/05-09-rvv-i32m2-typed-intrinsic-emission-slice/implement.jsonl`
  - `.trellis/tasks/archive/2026-05/05-09-rvv-i32m2-typed-intrinsic-emission-slice/check.jsonl`
- Workspace journal entries around the older RVV family evidence bridge were
  read to preserve direct-helper SSH evidence conventions and avoid mistaking
  prior m1/vsub SSH evidence for typed m2 evidence.
- Primary implementation surface is expected to be
  `scripts/rvv_microkernel_e2e.py` plus focused script tests. C++ exporter files
  should remain read-only unless the m2 evidence run exposes a real exporter
  bug.

## Validation Plan

- `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- Dry-run for the new m2 mode, proving `ssh_evidence=false` and no
  runtime/correctness claim.
- Focused lit for `test/Scripts/rvv-microkernel-e2e.test` and any touched RVV
  microkernel script tests.
- Focused lit for
  `test/Target/RVVMicrokernel/rvv-microkernel-i32m2-family-sub.mlir` and
  `test/Target/RVVMicrokernel/rvv-microkernel-i32m2-object.mlir`.
- One real `ssh rvv` i32-vsub m2 run with saved evidence and observed stdout
  marker.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate -j2`
  if the local build state supports it.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  before finishing if the local build state supports it.

## Completion Notes

- Added an explicit `--vector-shape` evidence mode to
  `scripts/rvv_microkernel_e2e.py`; `--vector-shape=i32m2` with
  `--arithmetic-family=i32-vsub` selects the existing typed m2 MLIR fixture by
  default.
- The m2 mode runs `tcrv-opt --tcrv-execution-planning-pipeline` into
  `tcrv-translate`, validates source/header/object artifacts against the
  compiler-emitted m2 source metadata, and records SEW=32, LMUL=m2, agnostic
  tail/mask policy, route ids, paths, hashes, marker expectations, and bounded
  claim scope in evidence JSON.
- The runner fails closed when m2 is requested but m1 generated source metadata
  is observed, before any SSH/runtime claim.
- Dry-run evidence now records `ssh_evidence=false`; real SSH evidence records
  `ssh_evidence=true` plus `ssh_evidence_details`.
- Real i32-vsub m2 evidence was collected at
  `artifacts/tmp/rvv_microkernel_e2e/20260510-rvv-i32m2-vsub-ssh/evidence.json`.
  It records `ssh_evidence=true`, `stdout_marker_observed=true`, and expected
  marker `tcrv_rvv_i32_vsub_microkernel_external_abi_ok`.
- `tcrv.exec` stayed compute-free, RVV behavior stayed dialect/plugin/target
  owned, and Python remained runner/evidence tooling only.
- No performance or broad correctness claim is made; the runtime claim is
  limited to the generated i32-vsub m2 direct-helper artifact plus external
  caller behavior proven by this `ssh rvv` run.

## Checks Run

- `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- `python3 scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i32-vsub --vector-shape=i32m2 --run-id codex-i32m2-vsub-dry --overwrite`
- `python3 scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i32-vsub --vector-shape=i32m2 --run-id codex-i32m2-reject-m1 --overwrite --input test/Target/RVVMicrokernel/rvv-microkernel-family-sub.mlir` expectedly failed closed on stale i32m1 metadata.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-microkernel-e2e'` from `artifacts/tmp/tianchenrv-build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-microkernel-i32m2-(family-sub|object)'` from `artifacts/tmp/tianchenrv-build/test`
- `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i32-vsub --vector-shape=i32m2 --run-id 20260510-rvv-i32m2-vsub-ssh --overwrite --timeout 120 --ssh-target rvv`
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate -j2`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 168/168 tests.
