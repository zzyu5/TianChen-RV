# RVV binary mem-window runtime ABI IR coherence

## Goal

Make the descriptor-owned RVV binary callable ABI visible and checked at the
existing IR boundary. Selected RVV binary add/sub/mul emission paths for i32
and i64 must carry descriptor-derived `tcrv.exec.mem_window` and
`tcrv.exec.runtime_param` metadata, materialize supported emission-plan
`runtime_abi_parameters` as a mirror of that IR-backed ABI plan, and fail
closed when readiness/coherence/export sees stale or mismatched metadata.

## Requirements

- Preserve the existing IR boundary: use direct kernel-child
  `tcrv.exec.mem_window`, `tcrv.exec.runtime_param`, and emission-plan
  diagnostic `runtime_abi_parameters`; do not introduce a parallel metadata
  system.
- Keep RVV dtype/family ownership in RVV descriptor, plugin, and target/export
  code. Core readiness/coherence may validate generic agreement only through
  registry route callbacks and candidate surfaces.
- Selected direct RVV binary i32/i64 add/sub/mul paths must expose exactly the
  descriptor-owned callable ABI boundary:
  - `lhs-input-buffer`: read-only host kernel argument, target-export ABI-owned,
    dtype-specific const pointer C type.
  - `rhs-input-buffer`: read-only host kernel argument, target-export ABI-owned,
    dtype-specific const pointer C type.
  - `output-buffer`: writable host kernel argument, target-export ABI-owned,
    dtype-specific output pointer C type.
  - `runtime-element-count`: direct `tcrv.exec.runtime_param`, C type `size_t`,
    target-export ABI-owned, C name supplied by the IR boundary.
- Supported RVV binary emission-plan `runtime_abi_parameters` must be an exact
  mirror of the IR-backed callable ABI plan for role, C name, C type, and
  ownership. Stale diagnostic metadata must fail before target-owned source,
  header, object, or bundle materialization.
- Emission manifests and target artifact export must continue to consume the
  same verified metadata. They must not reconstruct incompatible defaults.
- Add focused negative coverage for stale or mismatched metadata, especially
  i64 routes carrying i32 pointer types, missing output windows, duplicate
  roles, and missing runtime element-count linkage.

## Acceptance Criteria

- [x] Current representation is inspected and implementation uses existing
      `mem_window`, `runtime_param`, and emission-plan diagnostic surfaces.
- [x] RVV binary i32 and i64 source/header/object route preflight validates
      candidate `runtime_abi_parameters` against the same IR-backed callable
      ABI plan that target source export consumes.
- [x] At least one i32 route and one i64 route have positive focused coverage
      proving descriptor-derived pointer types and runtime element-count
      linkage are accepted.
- [x] Negative coverage proves stale/mismatched metadata fails at
      coherence/export preflight, including i64-with-i32-pointer metadata and
      missing ABI boundary roles.
- [x] Core passes remain free of RVV family/dtype branches; RVV-specific
      descriptor validation remains target/plugin-local.
- [x] Focused build/tests and `check-tianchenrv` pass, or any failure is
      recorded with the exact blocker.

## Definition of Done

- Code and tests are implemented in C++/MLIR/LLVM/TableGen/CMake/lit as
  appropriate.
- `git diff --check` passes.
- Focused changed C++/lit tests pass.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passes, unless blocked by a reported environment issue.
- The Trellis task is truthful, finished/archived only after checks pass, and
  validated with `python3 ./.trellis/scripts/task.py validate <archived-task-path>`.
- One coherent git commit records the module behavior.

## Technical Approach

Use the existing descriptor-backed helpers as the source of truth. The RVV
plugin already ensures/collects descriptor-specific buffer windows and runtime
element-count params when materializing callable binary paths, and target
source export already has `buildRVVBinaryCallableABIPlanFromIR` plus
`validateRVVBinaryCallableABIParameterMirror`. Extend the registered RVV
target-artifact candidate validation path so generic execution-plan coherence
and generic export preflight call the same IR-backed mirror check for direct
RVV binary candidates. This keeps the core checks target-neutral while making
stale emission diagnostics fail before target-owned artifact generation.

## Out of Scope

- New RVV families, masks, dynamic VL policy, new vector shapes, performance
  tuning, or broad runtime evidence matrices.
- Python compiler internals or Python-owned ABI models.
- Adding compute semantics to `tcrv.exec`.
- IME, AME, Sophgo/offload, scalar fallback expansion, prompt/report-only
  work, or helper-only cleanup.
- New `ssh rvv` evidence unless generated runtime-callable behavior changes
  enough to require a runtime/correctness claim. This round is IR/coherence/
  export metadata validation.

## Technical Notes

- The task brief maps directly to the existing `tcrv.exec.mem_window` and
  `tcrv.exec.runtime_param` contract in `.trellis/spec/core-dialect`.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` already states
  that callable ABI parameters must mirror real IR-backed `mem_window` and
  runtime-element-count `runtime_param` boundaries.
- `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h` already provides
  descriptor-specific callable parameters, role requirements, mem-window
  specs, and runtime-element-count param specs for i32/i64 add/sub/mul.
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp` already ensures descriptor-backed
  `mem_window` and `runtime_param` IR boundaries before lowering-boundary and
  microkernel materialization.
- `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp` already builds
  supported emission-plan `runtime_abi_parameters` from the IR boundary.
- `lib/Target/RVV/RVVMicrokernel.cpp` already has IR-backed callable ABI plan
  reconstruction and mirror validation used by target export; this round should
  make the route preflight consume it before generic coherence/export success.
