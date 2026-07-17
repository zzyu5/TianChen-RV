# capability-aware dispatch fallback synthesis

## Goal

Implement a target-neutral C++/MLIR dispatch/fallback synthesis slice for the TianChen-RV variant pipeline. A kernel with RVV and scalar fallback capabilities should materialize both plugin proposals, select the preferred lower-cost route, and preserve a scalar fallback route in `tcrv.exec.dispatch` without claiming executable lowering/runtime/correctness/performance evidence.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* HEAD is expected to include `3d0c3b1 feat: add scalar fallback plugin first slice`.
* Previous work added plugin-local scalar fallback metadata/emission-plan support but did not implement executable scalar lowering.
* This round must leave active C++ transform/plugin/test changes and one coherent commit.
* Work must stay within the capability-driven MLIR execution-layer spine and keep `tcrv.exec` compute-free.

## Requirements

* Strengthen generic variant selection/dispatch synthesis to consume proposals from multiple built-in plugins.
* Produce coherent `tcrv.exec.dispatch` with selected candidate case(s) and explicit `tcrv.exec.fallback` when a plugin-provided fallback candidate exists.
* Define fallback eligibility through generic plugin-provided metadata or cost/policy fields, not target-family branching.
* Keep scalar fallback explicit-capability-gated and plugin-local.
* Make scalar fallback metadata-route diagnostics unambiguous as non-executable evidence.
* Do not implement scalar/RVV lowering, runtime ABI, object generation, correctness testing, benchmarking, or fresh RVV hardware probing.

## Acceptance Criteria

* [ ] Built-in RVV plus scalar fallback registry can produce both proposals when required capabilities are declared.
* [ ] Materialization creates both `tcrv.exec.variant` ops with correct generic origin/requires metadata.
* [ ] Selection prefers RVV over scalar fallback when RVV is profile-available and scalar fallback is present.
* [ ] Dispatch synthesis emits selected/preferred case(s) plus `tcrv.exec.fallback` pointing to the fallback candidate.
* [ ] Scalar-only target emits/selects a scalar fallback metadata route without executable claims.
* [ ] RVV-only target does not invent fallback and records or emits a useful generic diagnostic where required.
* [ ] Malformed/stale variants still fail through existing legality paths.
* [ ] C++ tests and lit/FileCheck coverage are updated.
* [ ] Required local checks pass.

## Out of Scope

* Actual scalar lowering, LLVM lowering, RISC-V lowering, RVV intrinsics, object generation, runtime execution, correctness proof, benchmarking, or fresh RVV hardware evidence.
* Python implementation of compiler internals.
* Core target-family branches for scalar, RVV, IME, offload, vendor, dtype, shape, or runtime identity.
* Treating scalar fallback metadata-route support as executable evidence.

## Technical Notes

* Core transforms may inspect generic variant metadata, selected/fallback markers, generic cost estimates, plugin-provided cost/policy fields, and emission-plan status.
* Plugin origin strings may be used as opaque diagnostic metadata only.
* If existing APIs already have dispatch and fallback concepts, extend them rather than adding a parallel pipeline.
* Required checks include `git diff --check`, CMake configure/build `check-tianchenrv`, focused C++ binaries, and focused lit filter.
