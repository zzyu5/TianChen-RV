# IR-backed callable ABI parameter binding

## Goal

Make the bounded RVV and scalar i32-vadd callable source exporters derive their callable ABI parameter roles, C types, deterministic order, and runtime `n` name from real `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` IR boundaries, with emission-plan/candidate metadata treated only as route summary or a validated mirror.

## What I Already Know

* The repository root is `/home/kingdom/phdworks/TianchenRV`.
* Mandatory precondition found a clean worktree, clean supervisor-policy files, and expected HEAD `b870b84 feat: model RVV scalar dispatch runtime params`.
* The previous slice materialized `tcrv.exec.mem_window` for lhs/rhs/out and `tcrv.exec.runtime_param` for runtime `n` plus dispatch availability.
* Current callable exporters still depend on detached/default metadata in places:
  * `RVVMicrokernel.cpp` resolves callable parameters from emission-plan `runtime_abi_parameters` when present and otherwise falls back to defaults.
  * `ScalarMicrokernel.cpp` uses default `lhs/rhs/out/n` ABI parameters and hard-coded scalar loop names.
  * RVV/scalar plugins add default runtime ABI parameter metadata instead of deriving it from execution IR.
  * RVV+scalar dispatch validates direct `runtime_param`/`mem_window` for the wrapper but still starts callable ABI resolution from candidate metadata.

## Requirements

* Add the smallest reusable C++ helper for the bounded i32-vadd callable ABI bridge.
* The helper must consume direct kernel-child `tcrv.exec.mem_window` roles for `lhs-input-buffer`, `rhs-input-buffer`, and `output-buffer`.
* The helper must consume direct kernel-child `tcrv.exec.runtime_param` for `runtime-element-count`.
* The callable parameter order must be exactly lhs, rhs, out, runtime-element-count.
* Buffer parameter C types must come from validated `mem_window` IR.
* Runtime `n` C name/type/ownership must come from validated `runtime_param` IR, not `element_count`.
* RVV and scalar callable exporters must validate and consume this shared plan before source output.
* RVV+scalar dispatch must reuse the same callable plan for both callable candidates and keep `dispatch-availability-guard` as dispatch-only runtime control.
* Emission-plan / target-artifact candidate parameter metadata must be absent-independent or validated as an exact mirror of the IR-backed callable plan.
* Missing, duplicate, stale, or inconsistent mem_window/runtime_param IR must fail closed before successful source output.
* `element_count` remains descriptor-local bounded fixture metadata only.

## Acceptance Criteria

* Positive lit/FileCheck coverage proves RVV callable source export uses IR-backed lhs/rhs/out/n ABI boundaries.
* Positive lit/FileCheck coverage proves scalar callable source export uses the same IR-backed plan.
* Positive dispatch coverage proves the RVV+scalar wrapper remains consistent and keeps `rvv_available` dispatch-only.
* Pipeline coverage proves execution planning materializes enough IR for export without detached parameter truth.
* Narrow negative coverage proves missing or stale mem_window/runtime_param IR fails before source output.
* Narrow negative coverage proves candidate/emission-plan parameter metadata disagreement is rejected.
* Narrow negative coverage proves `element_count` cannot replace runtime `n`.
* Local build and `check-tianchenrv` pass.
* No new RVV runtime/correctness/performance claim is made unless explicit `ssh rvv` evidence is collected.

## Out Of Scope

* Broad ABI subsystem.
* Generic tensor shape model.
* New runtime integration or automatic hardware probing.
* Python implementation of compiler IR, ABI modeling, lowering, emission, dispatch, target selection, or capability decisions.
* New evidence helpers or hand-authored C fixtures as proof.
* Broad negative matrix beyond focused behavior changed here.

## Technical Notes

* Relevant spec layers: core dialect, capability model, variant pipeline, RVV plugin, scalar fallback plugin, lowering/runtime, and testing.
* Implementation must remain C++/MLIR/TableGen/CMake/lit/FileCheck-first.
* Any Trellis task created in this round must be validated and archived before committing.
