# post-contract RVV runtime ABI evidence

## Goal

Revalidate the current C++ runtime ABI contract consumers against real `ssh rvv`
execution evidence for the bounded i32-vadd RVV paths. This round is evidence
oriented: prove that the RVV microkernel callable artifact path and the
RVV+scalar selected target-artifact bundle external ABI path still execute
correctly after the runtime ABI contract refactor.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Expected HEAD is `2dd71dd feat: centralize i32 vadd runtime abi contract`.
* Current scope is bounded i32-vadd correctness evidence only.
* Python may be used for scripts, runner orchestration, artifact parsing, and
  evidence checks only.
* Core compiler/runtime ABI ownership remains in C++/MLIR/LLVM/TableGen/CMake.
* Existing dirty worktree entry is not owned by this round:
  `include/TianChenRV/Dialect/Offload/IR/OffloadOps.td`.

## Requirements

* Refresh/build the local evidence tools needed for `tcrv-opt` and
  `tcrv-translate`.
* Run script self-tests when supported by inspected scripts.
* Run real `ssh rvv` evidence for the RVV+scalar selected target-artifact
  bundle external ABI path using bundle mode and a fresh run id.
* Run real `ssh rvv` evidence for the bounded RVV i32-vadd microkernel path
  using the existing script-supported mode from the lit test and a fresh run id.
* Assert retained `evidence.json` / `command_summary.json` show successful
  remote setup, compile, link when applicable, and run commands.
* For dispatch bundle evidence, assert scalar and RVV branches were covered and
  the external ABI success marker is present.
* Scan retained artifacts for secrets and accidental performance claim fields.
* If source changes are needed, keep them focused in the relevant C++ owner,
  add/update one focused test, rerun local checks and failed real evidence, and
  create one coherent commit.
* If only remote/toolchain/credential access blocks the evidence, do not make
  source changes.

## Acceptance Criteria

* [x] `cmake --build build --target tcrv-opt tcrv-translate -j2` passes, or
      build directory is reconfigured and then passes.
* [x] `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test` passes when
      supported.
* [x] `python3 scripts/rvv_microkernel_e2e.py --self-test` passes when
      supported.
* [x] RVV+scalar bundle real `ssh rvv` evidence passes with a fresh run id.
* [x] RVV microkernel real `ssh rvv` evidence passes with a fresh run id.
* [x] Evidence artifacts are inspected for required success markers, branch
      coverage, command success, secrets, and performance-claim leakage.
* [x] No artifacts/tmp outputs or unrelated changes are committed.
* [x] Trellis task is archived before final report.

## Out of Scope

* Generic RVV lowering, dynamic runtime integration, new kernel families, new
  high-level tensor/tile IR, and performance claims.
* Python implementations of compiler internals or runtime ABI decisions.
* Source changes that only rename evidence, loosen assertions, or mask failures.

## Technical Notes

* Required specs and source files are listed in the user request and must be
  read before evidence execution.
* The final report must state whether the repository is clean, whether a commit
  was created, and which invariants were preserved.
* Dispatch bundle ssh evidence:
  `artifacts/tmp/rvv_bundle_e2e/codex-runtime-abi-contract-rvv-bundle-20260508T090715Z`.
* RVV microkernel ssh evidence:
  `artifacts/tmp/rvv_microkernel_e2e/codex-runtime-abi-contract-rvv-microkernel-20260508T090746Z`.
* Evidence assertion scan covered both `evidence.json`, both
  `command_summary.json`, logs, and the bundle index.
