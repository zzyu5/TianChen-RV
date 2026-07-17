# Runtime Offload Handoff Metadata Descriptor Path

## Goal

Move the bounded runtime-offload compiler handoff path forward by preserving deterministic runtime-offload handoff metadata from the offload plugin emission plan through generic target artifact collection and the offload runtime descriptor export.

## Requirements

* Keep runtime offload modeled as a runtime handoff capability, not a custom RISC-V ISA extension.
* Keep offload-specific behavior in the offload plugin, offload dialect, or offload target descriptor/export owner.
* Preserve explicit handoff metadata for the selected offload path, including runtime-offload handoff kind, descriptor component role, required capability identity, and non-claim status.
* Make the deterministic descriptor/bundle output expose the preserved selected-plan metadata without adding vendor runtime calls, DMA, object generation, hardware execution, correctness, or performance claims.
* Keep core `tcrv.exec` compute-free and avoid RVV/IME/Sophgo target-family branches in generic passes.

## Acceptance Criteria

* [x] A selected `offload-plugin` path materializes emission-plan metadata with deterministic selected-plan handoff metadata.
* [x] The target artifact bundle index preserves that metadata for the offload descriptor route.
* [x] The offload runtime handoff descriptor artifact emits the same sanitized selected-plan metadata.
* [x] The plan-and-export bundle front door can run the offload planning path from capability-only input and emit the descriptor bundle.
* [x] Existing RVV/scalar dispatch, capability model, hart-parallel, and target artifact tests remain passing.

## Definition of Done

* C++/MLIR/TableGen/lit/CMake remain the primary implementation stack.
* Focused lit/C++ tests cover the positive descriptor/bundle path.
* Relevant `.trellis/spec/` behavior documentation is updated if durable behavior changes.
* `git diff --check`, configure, and `check-tianchenrv` pass.
* Task is archived before the final commit.

## Technical Approach

Use the existing offload plugin, emission-plan selected metadata mechanism, generic target artifact bundle selected-plan metadata plumbing, and offload target descriptor exporter. Add only narrow fields and validation where the current path already owns the data.

## Out of Scope

* No Sophgo hardware integration.
* No runtime execution, DMA, queue, RPC, thread pool, object generation, correctness, or performance claim.
* No new generic compute operation in `tcrv.exec`.
* No Python compiler semantics.
* No broad RVV/scalar dispatch or hart-parallel refactor.

## Technical Notes

* Existing offload plugin already proposes `offload.runtime`, materializes `tcrv_offload.lowering_boundary`, and registers `tcrv-export-offload-runtime-descriptor`.
* Existing bundle export already prints selected-plan metadata when a candidate carries it.
* The bounded missing link for this task is preserving offload handoff metadata as selected-plan metadata into both bundle index and descriptor artifact output.
