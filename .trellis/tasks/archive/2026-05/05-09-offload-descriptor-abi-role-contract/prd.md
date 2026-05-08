# Offload Descriptor ABI Role Contract

## Owner

Strengthen the compiler-owned offload runtime descriptor handoff so it exports a deterministic ABI role contract for host buffers and runtime scalar/control parameters.

## Scope

- Reuse the existing C++/MLIR runtime ABI, `mem_window`, and `runtime_param` structures.
- Keep offload as compiler handoff metadata only.
- Preserve selected-plan handoff metadata, lowering-boundary metadata, descriptor route identity, and non-claim evidence wording.

## Non-Scope

- No vendor runtime adapter, DMA, queueing, accelerator execution, runtime correctness, or performance claim.
- No Python implementation of compiler decisions or descriptor semantics.
- No new generic compute operations in `tcrv.exec`.
