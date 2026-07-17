# emission manifest export handoff

## Goal

Add a first-class C++ target/export handoff surface that consumes post-planning TianChen-RV MLIR and emits a deterministic emission manifest for downstream lowering/runtime glue. The manifest serializes selected `tcrv.exec` paths, dispatch/fallback metadata, lowering-boundary metadata, and plugin-owned runtime ABI / emission ownership metadata without implementing executable lowering, runtime linking, correctness execution, or performance measurement.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* HEAD is expected to be `bc8670a feat: add plugin-owned runtime ABI emission metadata`.
* The previous round added plugin-owned runtime ABI / emission ownership metadata but did not add executable RVV/scalar lowering or runtime evidence.
* Primary compiler structure must remain C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck.
* Python is allowed only for scripts, probes, artifact parsing, fixtures, and support utilities, not compiler internals.
* The exporter must be target-neutral and must not interpret RVV, scalar, IME, offload, Sophgo, AME, vendor, dtype, shape, microarchitecture, probe evidence, performance, or runtime semantics.
* Runtime ABI / emission ownership meaning remains plugin-local behind the registry/interface; the exporter only serializes the already-materialized generic contract.

## Requirements

* Add or extend a C++ target/export utility, preferably `tools/tcrv-translate/` with `--tcrv-export-emission-manifest` if no equivalent tool exists.
* Add a reusable C++ library under `include/TianChenRV/Target/` and `lib/Target/` if the project layout supports it.
* Wire the library/tool through CMake conventionally.
* Emit a deterministic, FileCheck-friendly manifest format from post-planning MLIR.
* Include when present: module identifier, kernel symbol, selected variant symbol, variant origin plugin, selected/fallback role, dispatch cases and fallback target, lowering boundary symbol or diagnostic reference, runtime ABI kind/name, runtime glue role, emission status, required capability refs, preference metadata, and bounded plugin explanation/reason.
* Validate before exporting with deterministic diagnostics for malformed or incomplete selected emission ownership state.
* Add positive and negative lit/FileCheck coverage under an organized test directory.
* Update specs only to describe the durable target/export handoff contract, not progress.

## Acceptance Criteria

* [ ] Scalar fallback selected path manifest includes kernel, selected variant, origin, dispatch/fallback, scalar lowering boundary, runtime ABI kind/name, runtime glue role, emission status, and required capabilities.
* [ ] RVV selected path manifest includes plugin-owned unsupported/deferred runtime ABI emission metadata and does not claim executable codegen.
* [ ] Pipeline test runs `tcrv-opt` to post-planning MLIR and pipes to the exporter.
* [ ] Multiple kernels export in symbol/order-stable form.
* [ ] Exporter is built as a public tool target and included in `check-tianchenrv`.
* [ ] Negative tests cover missing selection, stale selected variant, missing runtime ABI/emission metadata, duplicate metadata, malformed fields, unsafe plugin explanation text, and no manifest on invalid input.
* [ ] Existing project tests continue to pass.

## Out Of Scope

* LLVM lowering, RVV intrinsic emission, scalar code generation, inline asm emission, runtime library implementation, object generation, runtime library linking, benchmark, correctness execution, and performance claims.
* Any Python implementation of compiler decisions, plugin registry behavior, capability legality, variant selection, lowering, emission planning, or manifest export.
* Any target/vendor-specific branches in core export code.
* Generated manifest artifacts outside deterministic tests.

## Technical Notes

* Required inspection commands and file reads are listed in the user request and must be completed before code edits.
* If Trellis state is created or updated, it must be finished/archive-coherent before final commit.
