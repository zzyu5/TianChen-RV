# RVV scalar dispatch mem_window ABI boundary

## Goal

Make the bounded RVV+scalar i32-vadd dispatch path use real `tcrv.exec.mem_window` IR for the lhs, rhs, and out buffer ABI meanings before target source/object export claims those roles.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Mandatory precondition found a clean worktree, clean supervisor-policy files, and `HEAD` at `80d0dba feat: add RVV scalar dispatch executable evidence`.
- `tcrv.exec.mem_window` already exists, but only carries `purpose`, `binding`, and `memory_space`.
- Current RVV+scalar dispatch export validates callable `runtime_abi_parameters` but does not require an IR-backed buffer-window boundary.
- Python may only be used for evidence orchestration. Core IR, dialects, passes, plugins, target export decisions, and ABI modeling remain C++/MLIR/TableGen/CMake/lit/FileCheck.

## Requirements

- Extend or repair the existing `tcrv.exec.mem_window` surface minimally so it can express runtime ABI buffer-window roles for lhs, rhs, and out.
- Keep `tcrv.exec.mem_window` compute-free: no tensor shapes, no vector math, no RVV semantics, no generic compute modeling.
- Materialize or preserve lhs/rhs/out mem_window IR in the bounded RVV+scalar i32-vadd planning path before emission/export.
- The source of buffer roles must remain the existing plugin-owned i32-vadd dataflow/runtime ABI metadata.
- RVV+scalar target export must fail closed when required mem_window IR is missing, duplicated, or inconsistent with the selected dispatch path.
- Runtime `n`, AVL, VL, and dispatch availability remain runtime SSA/control/ABI values, not high-level shapes.
- Keep generic core verification target-neutral and keep RVV/scalar-specific export behavior target-owned or plugin-local.

## Acceptance Criteria

- `tcrv.exec.mem_window` parses/prints/verifies lhs/rhs/out runtime ABI buffer roles.
- The public execution-planning pipeline emits or preserves the three buffer windows for the bounded RVV+scalar i32-vadd path.
- RVV+scalar source export emits the expected dispatcher signature and comments proving the buffer roles are backed by `tcrv.exec.mem_window` IR.
- Missing, duplicate, or inconsistent buffer window metadata fails before dispatch source/object export succeeds.
- Focused lit/FileCheck coverage is updated.
- Local build and `check-tianchenrv` pass, or failures are reported clearly.
- Real `ssh rvv` evidence is run only if making a new runtime/correctness claim.

## Out Of Scope

- No broad ABI subsystem.
- No new generic executable route.
- No Python implementation of compiler IR or decisions.
- No generic tensor shape model.
- No automatic hardware probing.
- No hand-authored C fixture as compiler proof.
- No broad negative matrix beyond the changed behavior.

## Technical Notes

- Relevant spec files are listed in `implement.jsonl` and `check.jsonl`.
- Likely implementation files:
  - `include/TianChenRV/Dialect/Exec/IR/ExecOps.td`
  - `lib/Dialect/Exec/IR/ExecOps.cpp`
  - `include/TianChenRV/Support/RuntimeABIMemWindow.h`
  - `lib/Support/RuntimeABIMemWindow.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - focused tests under `test/Dialect/Exec`, `test/Transforms/ExecutionPlanning`, and `test/Target/RVVScalarDispatch`.
