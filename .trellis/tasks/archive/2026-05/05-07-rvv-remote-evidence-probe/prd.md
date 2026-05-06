# RVV Remote Evidence Probe

## Goal

Add a repeatable `ssh rvv` evidence-producing probe for TianChen-RV that records bounded RVV hardware/toolchain facts and a tiny RVV intrinsic compile/run result into non-committed artifacts under `artifacts/tmp/rvv_probe/<run-id>/`.

## What I Already Know

- The repository root is `/home/kingdom/phdworks/TianchenRV`.
- `HEAD` is `72dc04a feat: wire built-in RVV plugin registry into tcrv-opt`.
- The previous round kept RVV plugin emission readiness unsupported and only added built-in registry routing.
- The current live gap is not compiler lowering; it is repeatable remote evidence for RVV hardware/toolchain availability.
- Python is allowed here only because the probe is runner/evidence tooling, not compiler internals.

## Requirements

- Add or strengthen a small reusable probe under `scripts/`.
- Connect to `ssh rvv` and write a JSON evidence artifact plus raw sanitized command logs under `artifacts/tmp/rvv_probe/<run-id>/`.
- Collect bounded facts: uname/kernel, architecture, hart count, clang path/version, cmake path/version, RISC-V CPU/vector hints, and sudo availability as a boolean.
- Compile and run a tiny deterministic RVV intrinsic program when clang and `riscv_vector.h` are available.
- Record command, exit code, stdout/stderr, compiler version, source digest, and binary digest or source digest in JSON.
- Record exact failing command and diagnostic when headers, flags, clang, or runtime execution are unavailable.
- Sanitize logs to avoid secrets, credentials, private keys, environment tokens, and unrelated raw environment dumps.
- Add local self-test mode that validates schema/parsing and log sanitization without `ssh rvv`.
- Add lit or CMake-backed coverage so `check-tianchenrv` validates the local self-test path.
- Update docs/specs with the RVV evidence artifact contract.

## Acceptance Criteria

- [ ] Probe utility exists and can run local self-tests without remote hardware.
- [ ] `check-tianchenrv` covers the local self-test path.
- [ ] Real `ssh rvv` probe is attempted and produces a JSON artifact with success or structured failure.
- [ ] RVV plugin emission readiness remains unsupported.
- [ ] No generated artifacts, remote binaries, secrets, or raw logs are committed.
- [ ] Worktree is clean after one coherent commit.

## Out of Scope

- No compiler lowering to LLVM/RISC-V/RVV IR.
- No runtime ABI glue, executable emission path, or toolchain invocation from compiler core.
- No RVV compiler correctness, executable support, or performance claim.
- No core target-family branches or computation semantics in `tcrv.exec`.

## Technical Notes

- Relevant specs: implementation stack, RVV plugin, emission/runtime contract, plugin registry, MLIR testing, validation reference.
- Existing RVV behavior is plugin-local and metadata-only; this task must preserve that boundary.
- Generated evidence belongs below `artifacts/tmp/rvv_probe`, already ignored through `artifacts/tmp/`.
