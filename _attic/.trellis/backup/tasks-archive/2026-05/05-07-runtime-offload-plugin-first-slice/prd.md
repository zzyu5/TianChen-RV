# Generic Runtime Offload Extension Plugin First Slice

## Goal

Add the first bounded `tcrv.offload` runtime-offload extension slice through the existing TianChen-RV plugin, dialect, lowering-boundary, runtime ABI metadata, and emission manifest architecture. The slice proves that a runtime-offload extension can be added plugin-locally without treating offload or Sophgo as a custom RISC-V ISA and without changing target-neutral core orchestration.

## Requirements

* Add or extend a conventional MLIR offload dialect surface under `TianChenRV/Dialect/Offload/IR`.
* Keep the offload dialect minimal and compute-free; it may expose a plugin-owned runtime handoff/lowering boundary op only.
* Add a built-in offload extension plugin under `TianChenRV/Plugin/Offload`.
* Make offload availability depend only on explicit runtime-offload capability metadata, using the repo's capability model and stable generic capability naming discovered from current code/specs.
* Let the plugin propose one generic runtime-offload variant, verify legality plugin-locally, expose conservative preference metadata, materialize offload-owned lowering boundary metadata, and provide plugin-owned runtime ABI/emission metadata.
* Register the offload dialect and plugin through the existing RVV/scalar initialization and built-in registry paths.
* Preserve target-neutral core transforms and manifest export; do not add Sophgo/vendor/RVV/IME/scalar/offload branches to generic orchestration.
* Keep scalar fallback available and selectable when offload is absent, declined, invalid, or lower preference and scalar fallback capability is present.
* Update specs only for durable generic runtime-offload behavior, not as a progress log.

## Acceptance Criteria

* [ ] Offload dialect parse/verify lit coverage exists for the minimal lowering-boundary or metadata op.
* [ ] Offload plugin C++ tests cover registration, proposal, legality, and preference using explicit offload runtime capability metadata.
* [ ] Execution planning lit coverage proves offload selection only when explicit runtime-offload capability is available.
* [ ] Lowering/emission tests prove selected offload paths materialize offload-owned boundary, runtime ABI, runtime glue role, emission status, and capability refs.
* [ ] Manifest export lit coverage serializes selected offload handoff with generic origin, variant, dispatch/fallback, lowering boundary, runtime ABI kind/name, runtime glue role, emission status, and required capabilities.
* [ ] Negative tests cover missing explicit offload capability, malformed offload capability metadata, missing offload boundary or runtime ABI metadata, no RVV probe dependency, vendor strings not triggering offload, and unknown/unregistered offload origin failing through generic registry diagnostics.
* [ ] Existing RVV/scalar/plugin/capability/transform/target/script tests continue to pass.
* [ ] Required checks run: `git diff --check`, RVV probe self-tests, CMake configure, and `check-tianchenrv`.

## Definition of Done

* Active C++/MLIR/TableGen/CMake/lit integration is implemented and tested.
* No Python compiler-internal implementation is added.
* No accelerator kernels, object generation, runtime library calls, hardware execution, benchmarks, correctness claims, or performance claims are added.
* No `predoc/`, build output, remote logs, credentials, raw hardware logs, or generated artifacts are committed.
* The Trellis task is archived before final report if task state is used.
* A single coherent git commit is created and the worktree is clean.

## Out of Scope

* Sophgo-specific code, vendor-specific runtime contracts, AME/IME implementation, DMA/runtime library integration, accelerator kernels, executable lowering, LLVM lowering, object generation, link steps, benchmarks, correctness execution, and performance measurement.
* Modeling offload as a custom RISC-V ISA extension.
* Weakening RVV/scalar validation, registry diagnostics, capability-requires checks, variant legality/preference selection, runtime ABI metadata validation, emission manifest validation, or replay helper validation.

## Technical Notes

* Required repo root: `/home/kingdom/phdworks/TianchenRV`.
* Starting HEAD observed before edits: `24f2375 feat: add emission manifest export target`.
* Starting worktree observed clean before task creation.
* Implementation must follow current `.trellis/spec/` authority and repo-local `AGENTS.md`.
* The user explicitly requires a single serial worker and forbids subagents, spawned agents, parallel agents, background agent queues, and multi-agent workflows.
