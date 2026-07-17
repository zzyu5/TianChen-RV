# target artifact runtime ABI preflight

## Goal

Make the generic target-artifact/export front door validate required runtime ABI role contracts before route-specific descriptor text or bundle files are emitted. This round is scoped to the generic target artifact boundary, exercised by the offload runtime handoff descriptor route.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting HEAD is `7602ce8 feat: export offload descriptor ABI roles`.
- The worktree was clean before this task was created.
- The previous round made offload emission plans/exporter carry deterministic `runtime_abi_parameters` derived from `tcrv.exec.mem_window` and `tcrv.exec.runtime_param`.
- The current gap is earlier validation: missing or inconsistent offload runtime ABI role contracts should fail at the selected target artifact front door, not only inside the descriptor exporter.
- `lib/Target/BuiltinTargetArtifactExporters.cpp` does not exist in this tree; the actual implementation is `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`.

## Requirements

- Keep implementation in C++/MLIR/LLVM/CMake/lit/FileCheck.
- Do not implement compiler decisions, runtime ABI semantics, plugin registry, variant selection, target artifact export, or descriptor semantics in Python.
- Extend the generic target artifact exporter metadata with the smallest route-local validation hook needed for runtime ABI preflight.
- Register the offload runtime descriptor exporter with typed required runtime ABI roles and a route-local callback that validates the selected candidate against the IR-backed `mem_window` and `runtime_param` ABI plan.
- Generic validation must reject missing roles and role/name/type/ownership mismatches before calling the offload descriptor exporter.
- Offload-specific descriptor/handoff policy remains local to the offload target/export owner.
- Preserve descriptor exporter runtime ABI validation as the final safety net.
- Do not alter RVV/scalar route semantics except for compatibility with the generic exporter API.

## Acceptance Criteria

- A valid offload descriptor route passes the generic target-artifact preflight and still emits deterministic ABI role metadata.
- A missing or inconsistent offload ABI role contract is rejected through the front-door validation path with a precise diagnostic and no descriptor output.
- Existing offload descriptor exporter negative coverage remains relevant.
- `git diff --check` passes.
- The artifact build configure command succeeds unless the local toolchain is unavailable.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passes.

## Out of Scope

- RVV hardware probing or `ssh rvv` execution.
- Runtime/offload vendor integration, DMA, queues, accelerator objects, runtime success, correctness proof, or performance proof.
- Broad refactors of RVV/scalar dispatch, hart_parallel capacity, or unrelated artifact routes.
- Any claim that runtime-offload is a custom RISC-V ISA extension.

## Technical Notes

- Relevant generic code: `include/TianChenRV/Target/TargetArtifactExport.h`, `lib/Target/TargetArtifactExport.cpp`, `lib/Transforms/ExecutionPlanCoherence.cpp`.
- Relevant support code: `include/TianChenRV/Support/RuntimeABI.h`, `lib/Support/RuntimeABICallablePlan.cpp`, `lib/Support/RuntimeABIMemWindow.cpp`, `lib/Support/RuntimeABIParam.cpp`.
- Relevant offload code: `lib/Target/Offload/OffloadRuntimeDescriptor.cpp`, `lib/Plugin/Offload/OffloadExtensionPlugin.cpp`.
- Relevant tests: `test/Target/ArtifactExport/offload-runtime-descriptor-artifact-route.test`, `test/Target/ArtifactExport/artifact-export-coherence-preflight.test`, `test/Target/TargetArtifactBundleExport/offload-runtime-descriptor-bundle.mlir`.
