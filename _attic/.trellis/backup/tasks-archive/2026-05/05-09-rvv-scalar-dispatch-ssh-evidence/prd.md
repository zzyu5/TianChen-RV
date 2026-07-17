# RVV scalar dispatch ssh evidence

## Goal

Turn the existing bounded RVV+scalar i32-vadd dispatch bundle path from local dry-run evidence into real `ssh rvv` runtime/correctness evidence. The normal compiler/export pipeline must produce the source/header/object/caller artifacts, copy them to the RVV host, compile/link/run there, and verify both `rvv_available=0` and `rvv_available=1` dispatch branches.

## What I already know

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Expected starting HEAD: `d2a4db4 fix: tighten dispatch runtime guard semantics`.
* Work starts from a clean worktree and no active Trellis task.
* Previous work established planned RVV+scalar dispatch, target artifact bundle export, runtime ABI metadata, dry-run bundle evidence, and remote RVV availability.
* This task must be one serial full-access worker. No subagents, spawned agents, parallel agents, background agent queues, or multi-agent workflows.

## Requirements

* First exercise the supported CLI in `scripts/rvv_scalar_dispatch_e2e.py`; inspect `--help` instead of guessing flags.
* Run the existing local dry-run bundle path before the live remote path.
* If live evidence already works, do not fabricate compiler changes.
* If generated compiler artifacts are malformed, fix the emitting C++/MLIR target/export/runtime-ABI owner, not generated C by hand and not Python compiler semantics.
* If remote orchestration is missing, make the smallest runner/probe/evidence tooling change needed in `scripts/rvv_scalar_dispatch_e2e.py`.
* Keep Python limited to runner/probe/evidence tooling. Python must not decide IR, capability legality, variant selection, lowering, runtime ABI shape, plugin registry behavior, or target artifact semantics.
* Keep RVV-specific behavior plugin-local or target-export-local, not hard-coded in core orchestration passes.
* Preserve parameter layering: hardware facts and compile flags are target capability/plugin metadata; compile-time variant config is selected-plan/lowering metadata; runtime dispatch guard is an explicit ABI/control input; descriptor-local boundaries stay in target artifact/export layers.

## Evidence Requirements

* Produce sanitized evidence JSON under `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/<run-id>/evidence.json`.
* Evidence distinguishes local compiler planning/export steps, selected artifact paths from compiler-emitted bundle/index, remote host identity and bounded toolchain facts, remote compile/link/run commands or sanitized summaries, both dispatch branches, correctness pass/fail, `ssh_evidence`, and explicit bounded `claim_scope`.
* Evidence/logs must not contain secrets, raw credentials, tokens, private keys, proxy URLs, or unbounded environment dumps.

## Acceptance Criteria

* `git diff --check` passes.
* Configure passes:
  `cmake -S . -B artifacts/tmp/tianchenrv-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
* Build/check passes:
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
* Relevant local dry-run RVV+scalar dispatch bundle export path passes.
* Real `ssh rvv` compile/link/run path runs unless hardware/credentials genuinely block it.
* If script behavior changes, local lit/self-test coverage validates dry-run behavior and evidence schema without requiring ssh in default local checks.
* If tracked source/spec/test/tooling files change, create one coherent commit and leave the worktree clean.
* Archive and validate this Trellis task before final report.

## Out of Scope

* No RVV performance, latency, throughput, or speedup claim.
* No general RVV backend claim beyond bounded RVV+scalar i32-vadd dispatch runtime/correctness.
* No broad rewrite of variant selection, dispatch semantics, capability model, or target artifact registry unless a concrete live-evidence blocker demands it.
* No fake evidence via x86, emulation-only output, canned logs, or success strings without remote execution.
* No Sophgo/offload custom RISC-V ISA treatment and no IME/AME current hardware target claim.

## Technical Notes

* Relevant specs: validation experiment reference, MLIR testing contract, lowering/runtime emission contract, variant generation/selection/tuning, RVV plugin contract, capability model contract.
* Relevant tooling/tests/code owners are listed in the user request and will be inspected before edits.
