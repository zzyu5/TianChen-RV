# Host RVV Scalar Dispatch C Artifact

## Goal

Add one bounded host-side runtime dispatch C source artifact for the current finite i32 vector-add slice. The artifact must consume the selected RVV dispatch case plus scalar fallback metadata, include or materialize both existing runtime-callable C functions, and emit a deterministic dispatcher function that calls RVV when an explicit host-provided availability parameter is non-zero and calls scalar fallback otherwise.

## Requirements

* Add a target-owned C++ export path for exactly one selected `rvv-plugin` dispatch case and exactly one selected `scalar-plugin` dispatch fallback in the same `tcrv.exec.kernel`.
* Keep `tcrv.exec` compute-free and keep RVV/scalar computation semantics in target/plugin-owned code.
* Validate selected dispatch surface, lowering-boundary ops, executable microkernel attachments, supported emission-plan diagnostics, runtime ABI kind/name, runtime glue role, artifact kind, required capability refs, and export route ids for both paths before source output.
* Fail closed on missing or stale RVV callable metadata, missing or stale scalar fallback callable metadata, unsupported artifact kinds, wrong roles, duplicate paths, missing lowering boundaries, missing microkernels, ambiguous kernels, or unsupported origins.
* Emit deterministic C source with `#include <stddef.h>`, `#include <stdint.h>`, `#include <riscv_vector.h>`, bounded metadata comments, the RVV i32-vadd callable body, the scalar i32-vadd callable body, and one dispatcher function.
* The dispatcher ABI is explicitly host-guarded, not an automatic hardware probe: `void tcrv_dispatch_i32_vadd_<kernel>(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available)`.
* Preserve existing individual RVV and scalar callable exports except for narrowly necessary shared helper refactors.

## Acceptance Criteria

* [ ] A lit/FileCheck positive test proves dispatcher signature, RVV branch call, scalar fallback branch call, deterministic runtime-callable ABI symbols, both metadata comment sets, RVV intrinsics, and scalar addition.
* [ ] A focused negative test proves fail-closed behavior for one malformed slice case, preferably missing scalar fallback callable metadata.
* [ ] Existing focused tests for RVV callable export, scalar callable export, emission readiness, execution plan coherence, emission manifest, and target artifact routes still pass.
* [ ] `git diff --check` passes.
* [ ] CMake configure succeeds with LLVM/MLIR 20 paths.
* [ ] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passes.

## Definition Of Done

* Code is implemented in C++/MLIR/TableGen/CMake/lit/FileCheck, with no Python compiler-core implementation.
* Trellis task is archived at the end of the work.
* One coherent compiler commit is created.
* Final git status is clean.

## Technical Approach

Add a target-layer export unit for the combined RVV+scalar i32-vadd dispatch artifact. The exporter should reuse existing target-owned record validation/source rendering logic where that can be done narrowly, or introduce a small shared target-owned helper if reuse avoids duplicating selected-path/microkernel validation. Expose one bounded translate action such as `--tcrv-export-rvv-scalar-i32-vadd-dispatch-c`, and register a route only if metadata-routed generic export can support a combined artifact without target-name branching in generic transforms. Generic preflight and transforms remain target-neutral.

The generated dispatcher uses an explicit `int rvv_available` guard parameter. This is honest host ABI glue for the current slice and does not claim automatic RVV hardware probing.

## Out Of Scope

* Arbitrary kernels, dtypes, dispatch policies, multiple RVV variants, or multiple scalar fallbacks.
* Dynamic loader/linker integration, object generation, executable packaging, benchmarking, or performance measurement.
* `ssh rvv` hardware compile/run/correctness/performance claims.
* Sophgo/offload changes.
* Python implementation of compiler IR, dialects, passes, plugin registry, capability model, lowering, emission, or runtime glue.
* Standalone smoke/probe/dashboard/evidence-packaging/helper-only work.

## Technical Notes

* Required initial inspection completed: repo root, clean git status, last 12 commits, HEAD diff summary, AGENTS/README/specs, ODS files, RVV/scalar target exporters, target artifact export, emission manifest, RVV/scalar plugins, emission readiness, execution plan coherence, dispatch synthesis, translate tool, and focused tests.
* Relevant specs: `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`, `.trellis/spec/testing/mlir-testing-contract.md`.
* Current gap: individual RVV and scalar runtime-callable source artifacts exist, but there is no host-side artifact that consumes both selected-path metadata records and emits a bounded dispatcher.
