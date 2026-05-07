# tcrv-opt execution planning pipeline

## Goal

Add a bounded public tcrv-opt planning pipeline that composes the existing C++/MLIR transform passes from tcrv.exec kernel and capability anchors through deterministic plugin variant materialization, capability-require checking, variant selection, dispatch synthesis, and the deepest already-composable selected lowering or emission metadata stage.

## What I Already Know

* The repository root is `/home/kingdom/phdworks/TianchenRV`.
* HEAD is expected to be `5d40102 feat: preserve fallback proposals after plugin declines`.
* The round is local MLIR pipeline composition only. It must not make RVV runtime, correctness, hardware, or performance claims.
* Existing individual public passes include plugin variant materialization, capability-require checking, variant selection, dispatch synthesis, selected lowering-boundary materialization, and emission/readiness diagnostics.
* The implementation must remain in C++ / MLIR / LLVM / TableGen / CMake / lit or C++ tests. Python is only allowed for support scripts and must not implement compiler decisions.
* The pipeline must be generic and must not contain RVV, scalar, IME, Sophgo, AME, offload, vendor, dtype, shape, or microarchitecture branches. Tool-boundary built-in registry injection is acceptable if it follows existing tcrv-opt style.

## Requirements

* Register a named public MLIR pass pipeline surface invokable from `tcrv-opt`.
* Prefer MLIR `PassPipelineRegistration` or the existing local registration style after inspection.
* Compose existing pass factories and shared interfaces instead of duplicating pass logic in a monolithic pass.
* Preserve fallback-preserving proposal behavior: recoverable plugin-local declines allow later viable proposals; invalid proposals, malformed core IR, duplicate/rerun mismatch, and invalid selected variant metadata remain fatal.
* Keep tcrv.exec compute-free and keep extension-specific semantics plugin-local.
* Add lit/FileCheck coverage that invokes the pipeline by public name.
* Update stable specs only for the durable pipeline contract and test expectations.
* Finish/archive the Trellis task before the final commit if workflow state is used.

## Acceptance Criteria

* [ ] A valid RVV-capability plus scalar-fallback kernel runs through the public pipeline and produces deterministic materialized variants, selected variant metadata, dispatch/fallback organization, and selected lowering or emission metadata if included.
* [ ] Pipeline output is parseable by `tcrv-opt` and can be re-run or piped into the next included stage without weakening verifiers.
* [ ] Deterministic ordering is checked when RVV and scalar/fallback proposals are both viable.
* [ ] Plugin-local RVV metadata such as `tcrv_rvv.required_march` is preserved when RVV is viable.
* [ ] Missing or malformed RVV property evidence with valid scalar fallback still completes through scalar fallback selection/dispatch without materializing RVV.
* [ ] Only recoverably declined plugins and no viable fallback fail with a deterministic no-viable-proposals diagnostic.
* [ ] A rerun or pre-existing mismatched variant case remains deterministic and does not duplicate symbols.
* [ ] Explicit invalid RVV selected metadata remains fatal in the relevant legality/selection/lowering-boundary path.
* [ ] `git diff --check`, configure, and `check-tianchenrv` pass.
* [ ] A single coherent commit is created and the worktree is clean.

## Out of Scope

* New high-level frontend, tensor/tile IR, compute op, or generic compute semantics in `tcrv.exec`.
* LLVM/RISC-V lowering, RVV intrinsic emission, inline assembly, runtime ABI, object generation, hardware execution, benchmark, correctness result, or performance result.
* Any Python implementation of IR, passes, plugin registry behavior, materialization, legality, selection, dispatch, lowering, emission, runtime glue, or capability decisions.
* Any target-specific branch in generic transforms or support.

## Technical Notes

* Required source/spec inspection is listed in the user prompt and must be completed before code edits.
* Expected implementation locations include transform pass headers/registration, CMake wiring, and `tools/tcrv-opt/tcrv-opt.cpp`.
* Existing tests under `test/Transforms` and C++ transform/plugin tests must remain green.
