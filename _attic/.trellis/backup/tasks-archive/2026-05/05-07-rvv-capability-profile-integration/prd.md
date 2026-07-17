# RVV capability profile integration

## Goal

Connect sanitized RVV hardware/toolchain probe facts to a plugin-local C++ RVV capability profile that can populate TargetCapabilitySet for existing registry, proposal, and legality paths, without enabling RVV lowering or executable emission support.

## What I already know

* The user requires a single serial Codex worker with no subagents or parallel agents.
* HEAD is expected to be 010a7f3 feat: add RVV remote evidence probe.
* Previous work added scripts/rvv_remote_probe.py, self-test/lit coverage, and real ssh rvv evidence.
* Current live gap is probe evidence not yet connected to the C++ capability decision model or RVV plugin capability surface.
* Python may only remain probe/artifact/helper tooling; compiler-facing capability mapping must be plugin-local C++.

## Requirements

* Add a structured C++ sanitized RVV probe fact object and validation producing llvm::Expected or llvm::Error.
* Populate TargetCapabilitySet with deterministic, stable RVV capability entries from validated facts.
* Integrate through the RVV plugin as a capability/proposal/legality input only.
* Update the RVV probe artifact schema with a sanitized capability_facts section if absent.
* Preserve unsupported RVV emission readiness/plan behavior.
* Update specs for capability profiles, RVV plugin boundary, testing, and validation.
* Add C++ and lit/script tests for positive and negative capability-profile behavior.

## Acceptance Criteria

* [x] Positive sanitized ssh-rvv-like facts map to expected RVV TargetCapabilitySet entries.
* [x] Missing vector hint, non-riscv64 architecture, compile/run failure, and missing clang/cmake are rejected.
* [x] Capability names/symbols are deterministic and sanitized.
* [x] RVV emission readiness/plan remains unsupported even when capabilities are present.
* [x] RVV probe self-test validates capability_facts and sanitizer behavior without ssh rvv.
* [x] Required local checks pass.
* [x] Task is archived before the final commit if Trellis state is used.

## Evidence

* `git diff --check` passed.
* `python3 scripts/rvv_remote_probe.py --self-test` passed.
* `cmake -S . -B artifacts/tmp/tianchenrv-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir` passed.
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed.
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-capability-model-test` passed.
* Focused lit filter `rvv|capability|emission-readiness|materialize-emission` passed.
* Fresh `ssh rvv` probe artifact: `artifacts/tmp/rvv_probe/20260507T231545Z-capability-profile/rvv_probe_evidence.json`.

## Out of Scope

* RVV lowering, LLVM/RISC-V/RVV codegen, runtime calls, executable emission, or performance claims.
* Core registry parsing RVV-specific JSON or branching on RVV family semantics.
* Raw logs, secrets, environment dumps, or benchmark/performance data in compiler capability identity.

## Technical Notes

* Required inspection files and existing patterns are read in the main session before editing.
