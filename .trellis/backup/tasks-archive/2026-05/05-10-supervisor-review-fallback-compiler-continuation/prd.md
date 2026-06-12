# Prove profile-replayed i64 sub/mul RVV paths on ssh rvv

## Goal

Continue from the malformed Hermes review fallback by selecting the next
module-sized compiler path from current repository evidence: extend the already
working profile-replay i64 path beyond the prior `i64-vadd` evidence point and
prove the finite i64m1 `i64-vsub` and `i64-vmul` RVV routes through the existing
C++/MLIR planning/export path with bounded real `ssh rvv` evidence.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Session start worktree was clean at HEAD `062e61d`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes fallback brief before source changes.
- Latest supervisor artifacts are:
  - `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0040-20260510T055211Z/repo_audit.md`
  - `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0040-20260510T055211Z/review_input.md`
- Latest audit says the previous round added only commit `062e61d`, which
  completed and archived `05-10-05-10-rvv-i64-profile-replay-ssh-evidence`.
- The previous completed module proved profile-replayed `i64-vadd` on real
  `ssh rvv`, but the checked-in script/lit coverage only names one
  profile-replay family case.
- Existing dry-run preflight already shows the profile-replay path can drive
  `i64-vsub` and `i64-vmul` through `rvv_probe_to_mlir.py --emit-target-profile`,
  `tcrv-opt --tcrv-execution-planning-pipeline`, emission manifest export, and
  direct source/header helper export.

## Boundaries

- Compiler implementation stays in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
- Python remains evidence orchestration, artifact parsing, command
  construction, and sanitization only.
- Python must not decide compiler legality, selection, lowering, emission,
  arithmetic semantics, runtime ABI shape, or correctness.
- The generated RVV kernel bodies must remain compiler/exporter-owned. The
  evidence runner may generate only a small external C caller that calls the
  generated ABI.
- RVV correctness claims require fresh real `ssh rvv` evidence for the exact
  generated artifacts.
- No performance, throughput, broad backend maturity, or generic matrix claim is
  in scope.

## Requirements

1. Keep the route starting from sanitized RVV probe/profile replay facts via
   `scripts/rvv_probe_to_mlir.py --emit-target-profile`.
2. Drive each selected family through the real compiler path:
   replayed MLIR -> `tcrv-opt --tcrv-execution-planning-pipeline` ->
   target-owned RVV source/header/object helpers.
3. Add focused checked-in coverage proving profile-replay dry-run support for
   `i64-vsub` and `i64-vmul`, including generated replay input MLIR,
   family-specific route metadata, generated source, and claim-free evidence
   fields.
4. Preserve the existing `i64-vadd` profile-replay behavior and evidence schema.
5. Run real `ssh rvv` evidence for `i64-vsub` and `i64-vmul`; the generated
   source-built executable and generated object executable must both exit 0 and
   emit the bounded external ABI success marker.
6. If real evidence exposes missing compiler metadata, route selection,
   lowering, emission, or runtime ABI bugs, fix the C++/MLIR/target-export
   boundary rather than masking the issue in Python.
7. Keep evidence artifacts under `artifacts/tmp/rvv_i64_profile_replay_e2e/`
   and keep generated binaries/logs untracked.
8. Update durable specs only if this round changes a long-term contract.

## Acceptance Criteria

- [x] `scripts/rvv_microkernel_e2e.py --self-test` passes.
- [x] Focused lit/FileCheck coverage includes profile-replay `i64-vsub` and
      `i64-vmul` dry-runs.
- [x] Profile-replay evidence JSON for `i64-vsub` records the replay input,
      `frontend_lowering = "i64-vsub"`, route
      `tcrv-export-rvv-i64-vsub-microkernel-c`, `ssh_evidence = false` in
      dry-run mode, and no runtime/performance claims.
- [x] Profile-replay evidence JSON for `i64-vmul` records the replay input,
      `frontend_lowering = "i64-vmul"`, route
      `tcrv-export-rvv-i64-vmul-microkernel-c`, `ssh_evidence = false` in
      dry-run mode, and no runtime/performance claims.
- [x] Real `ssh rvv` profile-replay evidence succeeds for `i64-vsub`.
- [x] Real `ssh rvv` profile-replay evidence succeeds for `i64-vmul`.
- [x] `git diff --check` passes.
- [x] Focused build/lit checks for the changed behavior pass.
- [x] Full `check-tianchenrv` passes if feasible before archive.
- [x] Task context validates, task is finished/archived, and one coherent commit
      records the work.

## Minimal Validation Plan

- `git diff --check`
- `python3 -m py_compile scripts/rvv_microkernel_e2e.py scripts/rvv_probe_to_mlir.py`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- Focused profile-replay dry-runs for `i64-vsub` and `i64-vmul`.
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- Focused lit on `test/Scripts/rvv-microkernel-e2e.test`.
- Real `ssh rvv` profile-replay evidence for `i64-vsub` and `i64-vmul`.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if feasible.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-supervisor-review-fallback-compiler-continuation`

## Out Of Scope

- Broad i32/i64/add/sub/mul matrices beyond the two missing profile-replay i64
  family members.
- Performance, throughput, latency, tuning, or benchmark claims.
- Python replacement for compiler internals.
- New `tcrv.exec` compute semantics.
- Moving RVV-specific behavior into core dialect/passes.
- Committed generated artifacts under `artifacts/tmp`.
- Supervisor prompt/policy work unrelated to the compiler path.

## Completion Evidence

- Fresh sanitized RVV probe input:
  `artifacts/tmp/rvv_probe/20260510T-rvv-i64-sub-mul-profile-replay-probe/rvv_probe_evidence.json`
- Successful profile-replayed `i64-vsub` ssh evidence:
  `artifacts/tmp/rvv_i64_profile_replay_e2e/20260510T-rvv-i64-vsub-profile-replay-ssh/evidence.json`
- Successful profile-replayed `i64-vmul` ssh evidence:
  `artifacts/tmp/rvv_i64_profile_replay_e2e/20260510T-rvv-i64-vmul-profile-replay-ssh/evidence.json`
- Both real evidence runs generated profile replay MLIR through
  `scripts/rvv_probe_to_mlir.py --emit-target-profile`, then ran
  `tcrv-opt --tcrv-execution-planning-pipeline` and direct RVV
  source/header/object helper exports.
- Remote result for both family paths:
  source-built executable exit code `0`, generated-object executable exit
  code `0`, source marker observed, and object marker observed.
- The default profile-replay kernel name now remains
  `rvv_probe_i64_replay` for `i64-vadd` compatibility and becomes
  `rvv_probe_i64_vsub_replay` / `rvv_probe_i64_vmul_replay` for the other two
  i64 binary families.
- Checks run:
  `git diff --check`;
  `python3 -m py_compile scripts/rvv_microkernel_e2e.py scripts/rvv_probe_to_mlir.py`;
  `python3 scripts/rvv_microkernel_e2e.py --self-test`;
  focused `i64-vsub` and `i64-vmul` profile-replay dry-runs;
  `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`;
  focused lit filter `rvv-microkernel-e2e` passed `1/1`;
  real `ssh rvv` probe plus both profile-replay evidence runs;
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed `192/192`.
- No `.trellis/spec/` update was needed: this round preserves the existing
  profile/replay and RVV evidence contracts and adds family-specific evidence
  runner behavior plus coverage.

## Technical Notes

- Required specs read for this round:
  - `.trellis/spec/index.md`
  - `.trellis/spec/implementation-stack/compiler-stack-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/capability-model/profiles.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Latest review artifacts read:
  - `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0040-20260510T055211Z/repo_audit.md`
  - `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0040-20260510T055211Z/review_input.md`
- Immediate predecessor task read:
  - `.trellis/tasks/archive/2026-05/05-10-05-10-rvv-i64-profile-replay-ssh-evidence/prd.md`
- Current implementation surfaces inspected:
  - `scripts/rvv_microkernel_e2e.py`
  - `scripts/rvv_probe_to_mlir.py`
  - `test/Scripts/rvv-microkernel-e2e.test`
  - `test/Transforms/LinalgToExec/linalg-i64-vsub-to-rvv-artifact.mlir`
  - `test/Transforms/LinalgToExec/linalg-i64-vmul-to-rvv-artifact.mlir`
  - `test/Target/RVVMicrokernel/rvv-microkernel-i64-vsub.mlir`
  - `test/Target/RVVMicrokernel/rvv-microkernel-i64-vmul.mlir`
