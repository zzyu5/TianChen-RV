# RVV plugin capability property consumer

## Goal

Add one bounded C++ RVV plugin-local consumer slice that reads preserved `CapabilityDescriptor::properties` from MLIR-origin `tcrv.exec.capability` descriptors and uses those properties in a real compiler decision path without moving RVV semantics into core support or transforms.

## What I already know

* The repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial HEAD is `b4f302d feat: preserve capability properties from MLIR`.
* The initial worktree was clean before this task was created.
* The previous round preserved non-core structured MLIR attributes from `tcrv.exec.capability` into `CapabilityDescriptor::properties`.
* This round must remain a single serial Codex worker run and must not use subagents, background workers, or multi-agent workflows.

## Requirements

* Implement the compiler decision in C++/MLIR/CMake/lit or C++ tests only.
* Keep `Support/CapabilityModel` generic; do not add RVV-specific interpretation there.
* Keep RVV property parsing and decisions behind the RVV plugin interface.
* Use actual RVV property keys produced by `RVVCapabilityProfile.cpp` and preserved from MLIR.
* Affect at least one real RVV plugin decision path, preferably proposal generation or legality.
* Reject missing or malformed RVV property evidence deterministically without crashing.
* Preserve existing generic capability requirements and plugin-local origin metadata.
* Do not implement LLVM/RISC-V lowering, RVV intrinsic emission, inline assembly, runtime ABI glue, object generation, hardware execution, benchmark timing, or correctness validation.
* Do not add target-specific branches in core transforms or support.
* Do not treat Sophgo/offload as a RISC-V ISA extension.

## Acceptance Criteria

* [ ] C++ tests construct or parse real MLIR modules with `tcrv.exec.capability` properties, build a `TargetCapabilitySet`, and exercise the RVV plugin decision API.
* [ ] Positive tests show valid preserved RVV properties enable the chosen decision path.
* [ ] Positive tests show plugin-local origin and existing generic requirements are preserved.
* [ ] Negative tests cover missing required RVV property evidence.
* [ ] Negative tests cover malformed numeric/text property evidence.
* [ ] Negative tests cover unsupported RVV-namespaced variant metadata.
* [ ] Tests show unknown non-core properties remain preserved generically.
* [ ] Tests show runtime-offload/Sophgo properties remain generic descriptor properties, not ISA semantics.
* [ ] Existing checks pass: `git diff --check`, CMake configure, and `cmake --build build --target check-tianchenrv -j2`.

## Out of Scope

* Runtime RVV execution, ssh rvv validation, correctness claims, and performance claims.
* LLVM/RISC-V lowering, RVV intrinsic emission, inline assembly, object generation, and runtime ABI work.
* IME, AME, Sophgo/offload ISA modeling, or target-specific logic outside plugin-local RVV code.
* Python compiler decision logic.

## Technical Notes

* Required inspection commands and file reads are provided by the user and must be completed before implementation edits.
* If a decision is not visible through an existing pass without a broader pass change, this task may stay at the C++ plugin API test surface and explicitly report the remaining pass-surface gap.
