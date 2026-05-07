# Plugin-Owned Runtime ABI Emission Metadata

## Goal

Add a target-neutral C++ plugin interface and emission/readiness integration so the compiler can ask the selected variant's origin plugin for bounded runtime ABI and emission ownership metadata after legality, preference selection, dispatch, and lowering-boundary discovery have succeeded.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* The current compiler spine is high-level MLIR op to target capabilities, plugin variants, legality, selection/dispatch, plugin-owned lowering/emission/runtime glue, and RVV/IME/offload/fallback executable paths.
* The previous hygiene round is expected to have HEAD `0868d2d chore: clean stale predoc and transform test layout`.
* `predoc/` is obsolete and must not be resurrected or used as live evidence.
* Transform tests are organized under subdirectories such as `EmissionReadiness`, `ExecutionPlanning`, `VariantSelection`, `LoweringBoundary`, and related pass/stage directories.
* This round must include active C++/MLIR plugin interface or emission/readiness integration and tests, not only docs or task state.

## Requirements

* Extend the existing plugin interface if a suitable hook exists; otherwise add one target-neutral C++ hook for runtime ABI / emission ownership metadata.
* Core emission/readiness code may orchestrate selected variant lookup, target capability construction, lowering-boundary lookup, generic diagnostics, and metadata materialization.
* Core emission/readiness code must not inspect RVV, scalar, IME, offload, Sophgo, AME, vendor, dtype, shape, microarchitecture, runtime, or probe-evidence semantics.
* Plugin-specific runtime ABI interpretation must stay inside the owning plugin.
* RVV may return unsupported or deferred executable emission with plugin-owned reason and bounded ABI boundary metadata.
* Scalar fallback may return a conservative host/scalar fallback runtime ABI metadata plan only when legally selected.
* Metadata should be generic and bounded: origin plugin, selected variant, lowering boundary symbol, runtime ABI kind/name, emission status, required runtime glue role, reason/explanation, and capability refs.
* Runtime ABI metadata must not resurrect illegal variants, bypass selection, or convert malformed/unavailable paths into executable paths.
* Specs may be updated only for durable interface behavior and must state that metadata is not executable code, correctness evidence, or performance evidence.

## Acceptance Criteria

* [ ] Selected scalar fallback path with scalar lowering boundary materializes plugin-owned runtime ABI/emission ownership metadata.
* [ ] Selected RVV path with valid capability metadata reaches plugin-owned emission metadata and reports unsupported/deferred executable emission.
* [ ] `--tcrv-execution-planning-pipeline` preserves legality to selection to dispatch to lowering boundary to plugin-owned metadata ordering.
* [ ] Metadata contains deterministic generic origin/variant/boundary/runtime-ABI/status fields.
* [ ] Public `tcrv-opt` built-in registry behavior covers the new hook.
* [ ] Unknown origin, missing boundary, stale selected variant, malformed plugin metadata, no legal selected variant, and invalid explicit RVV path are covered by deterministic negative tests.
* [ ] Existing relevant lit and C++ tests continue to pass.

## Definition Of Done

* C++/MLIR/TableGen/CMake/lit remain the primary compiler implementation stack.
* Python is used only for existing probe/helper self-tests in this round.
* Normal checks pass: `git diff --check`, RVV helper self-tests, CMake configure, and `check-tianchenrv`.
* Trellis task state is completed and archived before final commit if task workflow supports it.
* One coherent commit is created and the worktree is clean.

## Out Of Scope

* LLVM lowering, RVV intrinsic emission, object generation, runtime library linking, benchmarks, correctness execution, and performance measurement.
* Any new compute ops, tensor/tile IR, executable runtime bodies, or RVV intrinsic ops in `tcrv.exec`.
* Any new RVV hardware/runtime/correctness/performance claim without `ssh rvv` evidence.
* Resurrecting `predoc/` or flattening organized `test/Transforms` layout.

## Technical Notes

* Required initial inspection and file reads are taken from the Hermes Supervisor Delta.
* Relevant spec layers include architecture, capability model, core dialect, plugin protocol, variant pipeline, lowering/runtime, extension plugins, and testing.
